"""Provider Proxy (Layer 3) for LLM Broker.

Dispatches chat completion requests to LLM providers via LiteLLM,
with automatic retry through the fallback chain and structured
cost/latency logging.

The proxy:
* Takes an ordered model list (from the router) and a ChatCompletionRequest
* Tries the primary model first, falling back down the chain on failure
* Streams SSE chunks when ``stream=True``
* Logs per-request: model, provider, latency_ms, estimated cost, success/failure
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import litellm

from llm_broker.compliance import EligibleModel
from llm_broker.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    Usage,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Error types that should trigger fallback to the next model
# ---------------------------------------------------------------------------

_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    litellm.exceptions.APIError,
    litellm.exceptions.APIConnectionError,
    litellm.exceptions.RateLimitError,
    litellm.exceptions.Timeout,
    litellm.exceptions.ServiceUnavailableError,
    litellm.exceptions.InternalServerError,
    litellm.exceptions.BadGatewayError,
)


class AllModelsFailedError(Exception):
    """Raised when every model in the fallback chain has failed."""

    def __init__(self, errors: list[tuple[str, Exception]]) -> None:
        self.errors = errors
        model_names = [name for name, _ in errors]
        super().__init__(f"All models failed: {model_names}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_litellm_kwargs(
    eligible: EligibleModel,
    request: ChatCompletionRequest,
) -> dict[str, Any]:
    """Build the keyword arguments dict for ``litellm.acompletion``."""
    messages = []
    for m in request.messages:
        msg_dict: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.name:
            msg_dict["name"] = m.name
        if m.tool_calls is not None:
            msg_dict["tool_calls"] = m.tool_calls
        if m.tool_call_id is not None:
            msg_dict["tool_call_id"] = m.tool_call_id
        messages.append(msg_dict)

    kwargs: dict[str, Any] = {
        "model": eligible.model.litellm_model,
        "messages": messages,
        "stream": request.stream,
    }

    # Optional parameters — only forward when explicitly set
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p
    if request.max_tokens is not None:
        kwargs["max_tokens"] = request.max_tokens
    if request.stop is not None:
        kwargs["stop"] = request.stop
    if request.tools is not None:
        kwargs["tools"] = request.tools
    if request.tool_choice is not None:
        kwargs["tool_choice"] = request.tool_choice

    return kwargs


def _estimate_cost(
    model: EligibleModel,
    usage: Usage,
) -> float:
    """Estimate cost from token count and model config."""
    total_tokens = usage.total_tokens
    return (total_tokens / 1000.0) * model.model.cost_per_1k_tokens


def _log_request(
    *,
    model_id: str,
    provider: str,
    litellm_model: str,
    latency_ms: float,
    estimated_cost: float,
    success: bool,
    error: str | None = None,
    attempt: int = 1,
) -> None:
    """Emit a structured JSON log line for the request."""
    record = {
        "event": "llm_request",
        "model_id": model_id,
        "provider": provider,
        "litellm_model": litellm_model,
        "latency_ms": round(latency_ms, 2),
        "estimated_cost": estimated_cost,
        "success": success,
        "attempt": attempt,
    }
    if error:
        record["error"] = error
    logger.info(json.dumps(record))


def _convert_response(
    raw: litellm.ModelResponse,
    eligible: EligibleModel,
) -> ChatCompletionResponse:
    """Convert a LiteLLM ``ModelResponse`` to our ``ChatCompletionResponse``."""
    choices = []
    for raw_choice in raw.choices:
        msg = raw_choice.message  # type: ignore[union-attr]
        tool_calls = None
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls = [tc.model_dump() if hasattr(tc, "model_dump") else tc for tc in msg.tool_calls]

        choices.append(
            Choice(
                index=raw_choice.index,
                message=ChatMessage(
                    role=msg.role or "assistant",
                    content=msg.content,
                    tool_calls=tool_calls,
                ),
                finish_reason=raw_choice.finish_reason,
            )
        )

    usage = Usage()
    if raw.usage:
        usage = Usage(
            prompt_tokens=raw.usage.prompt_tokens or 0,
            completion_tokens=raw.usage.completion_tokens or 0,
            total_tokens=raw.usage.total_tokens or 0,
        )

    return ChatCompletionResponse(
        id=raw.id or "",
        object=raw.object or "chat.completion",
        created=raw.created or int(time.time()),
        model=eligible.model.id,
        choices=choices,
        usage=usage,
    )


# ---------------------------------------------------------------------------
# Public API — non-streaming dispatch
# ---------------------------------------------------------------------------


async def dispatch(
    models: list[EligibleModel],
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """Dispatch a non-streaming chat completion through the fallback chain.

    Tries each model in *models* (ordered by the router) until one succeeds.
    Raises :class:`AllModelsFailedError` if every model fails.
    """
    if not models:
        raise AllModelsFailedError(errors=[])

    errors: list[tuple[str, Exception]] = []

    for attempt, eligible in enumerate(models, start=1):
        kwargs = _build_litellm_kwargs(eligible, request)
        # Force non-streaming for this path
        kwargs["stream"] = False
        start = time.monotonic()

        try:
            raw = await litellm.acompletion(**kwargs)
            latency_ms = (time.monotonic() - start) * 1000
            response = _convert_response(raw, eligible)

            _log_request(
                model_id=eligible.model.id,
                provider=eligible.provider_name,
                litellm_model=eligible.model.litellm_model,
                latency_ms=latency_ms,
                estimated_cost=_estimate_cost(eligible, response.usage),
                success=True,
                attempt=attempt,
            )
            return response

        except _RETRYABLE_EXCEPTIONS as exc:
            latency_ms = (time.monotonic() - start) * 1000
            _log_request(
                model_id=eligible.model.id,
                provider=eligible.provider_name,
                litellm_model=eligible.model.litellm_model,
                latency_ms=latency_ms,
                estimated_cost=0.0,
                success=False,
                error=str(exc),
                attempt=attempt,
            )
            errors.append((eligible.model.id, exc))
            logger.warning(
                "Model %s (provider=%s) failed, trying next fallback",
                eligible.model.id,
                eligible.provider_name,
            )

    raise AllModelsFailedError(errors=errors)


# ---------------------------------------------------------------------------
# Public API — streaming dispatch
# ---------------------------------------------------------------------------


async def dispatch_stream(
    models: list[EligibleModel],
    request: ChatCompletionRequest,
) -> AsyncIterator[str]:
    """Dispatch a streaming chat completion through the fallback chain.

    Yields SSE-formatted chunks (``data: {...}\\n\\n``).  On provider
    failure the next model in the chain is attempted — but only if no
    chunks have been yielded yet.

    Raises :class:`AllModelsFailedError` if every model fails before
    producing any output.
    """
    if not models:
        raise AllModelsFailedError(errors=[])

    errors: list[tuple[str, Exception]] = []

    for attempt, eligible in enumerate(models, start=1):
        kwargs = _build_litellm_kwargs(eligible, request)
        kwargs["stream"] = True
        start = time.monotonic()

        try:
            stream = await litellm.acompletion(**kwargs)
            async for chunk in stream:
                # LiteLLM chunks are ModelResponse-like objects
                chunk_data = chunk.model_dump_json() if hasattr(chunk, "model_dump_json") else json.dumps(chunk)
                yield f"data: {chunk_data}\n\n"

            # Stream completed successfully
            latency_ms = (time.monotonic() - start) * 1000
            _log_request(
                model_id=eligible.model.id,
                provider=eligible.provider_name,
                litellm_model=eligible.model.litellm_model,
                latency_ms=latency_ms,
                estimated_cost=0.0,  # token count not available for streaming
                success=True,
                attempt=attempt,
            )
            yield "data: [DONE]\n\n"
            return

        except _RETRYABLE_EXCEPTIONS as exc:
            latency_ms = (time.monotonic() - start) * 1000
            _log_request(
                model_id=eligible.model.id,
                provider=eligible.provider_name,
                litellm_model=eligible.model.litellm_model,
                latency_ms=latency_ms,
                estimated_cost=0.0,
                success=False,
                error=str(exc),
                attempt=attempt,
            )
            errors.append((eligible.model.id, exc))
            logger.warning(
                "Model %s (provider=%s) failed during stream, trying next fallback",
                eligible.model.id,
                eligible.provider_name,
            )

    raise AllModelsFailedError(errors=errors)
