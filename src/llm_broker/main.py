"""FastAPI application for LLM Broker."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from llm_broker.compliance import apply_pii_policy, get_eligible_models
from llm_broker.config import get_config, load_config
from llm_broker.models import ChatCompletionRequest, ChatMessage
from llm_broker.proxy import AllModelsFailedError, dispatch, dispatch_stream
from llm_broker.router import route_request


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load configuration on startup."""
    import llm_broker.config as cfg_module

    cfg_module._config = load_config()
    yield


app = FastAPI(title="LLM Broker", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    config = get_config()
    return {"status": "ok", "repos": config.repo_names}


def _extract_api_key(authorization: str | None) -> str | None:
    """Extract API key from Authorization header (Bearer token)."""
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    # Also accept raw key (no Bearer prefix)
    return authorization.strip()


# ---------------------------------------------------------------------------
# Anthropic <-> OpenAI format conversion
# ---------------------------------------------------------------------------


def _anthropic_messages_to_openai(
    anthropic_messages: list[dict[str, Any]],
) -> list[ChatMessage]:
    """Convert Anthropic-format messages to OpenAI ChatMessage objects.

    Anthropic format uses content blocks:
        {"role": "user", "content": [{"type": "text", "text": "..."}]}
    or plain strings:
        {"role": "user", "content": "..."}

    Converts to flat string content for the internal pipeline.
    """
    result: list[ChatMessage] = []
    for msg in anthropic_messages:
        role = msg.get("role", "user")
        content = msg.get("content")

        if isinstance(content, list):
            # Extract text from content blocks
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            text = "\n".join(text_parts) if text_parts else ""
        elif isinstance(content, str):
            text = content
        else:
            text = ""

        result.append(ChatMessage(role=role, content=text))
    return result


def _openai_response_to_anthropic(
    response_data: dict[str, Any],
    model_requested: str,
) -> dict[str, Any]:
    """Convert an OpenAI-format response dict to Anthropic format.

    Anthropic response format:
        {
            "id": "msg_...",
            "type": "message",
            "role": "assistant",
            "model": "...",
            "content": [{"type": "text", "text": "..."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": ..., "output_tokens": ...}
        }
    """
    choices = response_data.get("choices", [])
    content_blocks: list[dict[str, Any]] = []
    stop_reason = "end_turn"

    if choices:
        choice = choices[0]
        message = choice.get("message", {})
        text = message.get("content") or ""
        if text:
            content_blocks.append({"type": "text", "text": text})

        # Map finish_reason
        finish = choice.get("finish_reason", "stop")
        if finish == "stop":
            stop_reason = "end_turn"
        elif finish == "tool_use":
            stop_reason = "tool_use"
        elif finish == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"

    usage = response_data.get("usage", {})

    return {
        "id": response_data.get("id", ""),
        "type": "message",
        "role": "assistant",
        "model": model_requested,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# ---------------------------------------------------------------------------
# OpenAI-compatible chat completions endpoint
# ---------------------------------------------------------------------------


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    authorization: str | None = Header(None),
    x_size: str | None = Header(None, alias="x-size"),
) -> Response:
    """OpenAI-compatible chat completions endpoint.

    Pipeline: Auth -> Layer 1 (Compliance) -> Layer 2 (Router) -> Layer 3 (Proxy)
    """
    config = get_config()

    # --- Auth ---
    api_key = _extract_api_key(authorization)
    if not api_key:
        return JSONResponse(
            status_code=401,
            content={"error": {"message": "Missing API key", "type": "auth_error"}},
        )

    key_mapping = config.resolve_key(api_key)
    if key_mapping is None:
        return JSONResponse(
            status_code=401,
            content={"error": {"message": "Invalid API key", "type": "auth_error"}},
        )

    repo_config = config.get_repo_config(key_mapping.repo)
    if repo_config is None:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": f"Unknown repo: {key_mapping.repo}",
                    "type": "invalid_request_error",
                }
            },
        )

    # --- Layer 1: Compliance Gateway ---
    eligible = get_eligible_models(repo_config, config.providers)
    if not eligible:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "No eligible models for this repo configuration",
                    "type": "invalid_request_error",
                }
            },
        )

    messages = apply_pii_policy(body.messages, repo_config)
    # Update request with (possibly redacted) messages
    body = body.model_copy(update={"messages": messages})

    # --- Layer 2: Intelligent Router ---
    size_hint = x_size if x_size in ("large", "small", "medium") else None
    ordered_models = route_request(
        eligible, messages, config.router, size_hint=size_hint
    )

    # --- Layer 3: Provider Proxy ---
    try:
        if body.stream:
            return StreamingResponse(
                dispatch_stream(ordered_models, body),
                media_type="text/event-stream",
            )
        else:
            response = await dispatch(ordered_models, body)
            return JSONResponse(content=response.model_dump())
    except AllModelsFailedError:
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": "All providers failed",
                    "type": "provider_error",
                }
            },
        )


# ---------------------------------------------------------------------------
# Anthropic-format messages endpoint (Claude Code compatible)
# ---------------------------------------------------------------------------


@app.post("/v1/messages", response_model=None)
async def messages(
    request: Request,
    authorization: str | None = Header(None),
    x_size: str | None = Header(None, alias="x-size"),
) -> Response:
    """Anthropic-format messages endpoint (used by Claude Code).

    Accepts Anthropic message format, converts to OpenAI format internally,
    routes through the same pipeline, and converts the response back.
    """
    config = get_config()

    # --- Auth ---
    api_key = _extract_api_key(authorization)
    if not api_key:
        return JSONResponse(
            status_code=401,
            content={"error": {"message": "Missing API key", "type": "auth_error"}},
        )

    key_mapping = config.resolve_key(api_key)
    if key_mapping is None:
        return JSONResponse(
            status_code=401,
            content={"error": {"message": "Invalid API key", "type": "auth_error"}},
        )

    repo_config = config.get_repo_config(key_mapping.repo)
    if repo_config is None:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": f"Unknown repo: {key_mapping.repo}",
                    "type": "invalid_request_error",
                }
            },
        )

    # Parse raw JSON body (Anthropic format, not Pydantic-validated)
    raw_body = await request.json()
    anthropic_messages = raw_body.get("messages", [])
    model_requested = raw_body.get("model", "")
    stream = raw_body.get("stream", False)
    max_tokens = raw_body.get("max_tokens")
    temperature = raw_body.get("temperature")
    system_text = raw_body.get("system")

    # Convert Anthropic messages to OpenAI format
    openai_messages = _anthropic_messages_to_openai(anthropic_messages)

    # Prepend system message if provided at top level (Anthropic convention)
    if system_text:
        if isinstance(system_text, list):
            # system can be a list of content blocks
            parts = []
            for block in system_text:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            system_content = "\n".join(parts)
        else:
            system_content = str(system_text)
        openai_messages.insert(0, ChatMessage(role="system", content=system_content))

    # --- Layer 1: Compliance Gateway ---
    eligible = get_eligible_models(repo_config, config.providers)
    if not eligible:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "No eligible models for this repo configuration",
                    "type": "invalid_request_error",
                }
            },
        )

    openai_messages = apply_pii_policy(openai_messages, repo_config)

    # Build an internal ChatCompletionRequest
    internal_request = ChatCompletionRequest(
        model=model_requested,
        messages=openai_messages,
        stream=stream,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # --- Layer 2: Intelligent Router ---
    size_hint = x_size if x_size in ("large", "small", "medium") else None
    ordered_models = route_request(
        eligible, openai_messages, config.router, size_hint=size_hint
    )

    # --- Layer 3: Provider Proxy ---
    try:
        if stream:
            # For streaming, pass through SSE chunks as-is (OpenAI SSE format)
            return StreamingResponse(
                dispatch_stream(ordered_models, internal_request),
                media_type="text/event-stream",
            )
        else:
            response = await dispatch(ordered_models, internal_request)
            response_data = response.model_dump()
            anthropic_response = _openai_response_to_anthropic(
                response_data, model_requested
            )
            return JSONResponse(content=anthropic_response)
    except AllModelsFailedError:
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": "All providers failed",
                    "type": "provider_error",
                }
            },
        )
