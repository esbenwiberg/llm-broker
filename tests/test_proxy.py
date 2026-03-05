"""Tests for the Provider Proxy (Layer 3).

All tests mock ``litellm.acompletion`` to avoid real API calls.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_broker.compliance import EligibleModel
from llm_broker.models import (
    ChatCompletionRequest,
    ChatMessage,
    ModelConfig,
    ProviderConfig,
)
from llm_broker.proxy import (
    AllModelsFailedError,
    _build_litellm_kwargs,
    _estimate_cost,
    dispatch,
    dispatch_stream,
)

# ---------------------------------------------------------------------------
# Test helpers / fixtures
# ---------------------------------------------------------------------------

_PROVIDER_US = ProviderConfig(
    litellm_prefix="openai/",
    region="us",
    deployment="saas",
    models=[],
)


def _model(
    id: str = "test-model",
    litellm_model: str = "openai/test-model",
    tier: str = "standard",
    cost: float = 0.001,
) -> ModelConfig:
    return ModelConfig(
        id=id,
        litellm_model=litellm_model,
        tier=tier,
        quality=0.5,
        cost_per_1k_tokens=cost,
    )


def _eligible(
    model: ModelConfig | None = None,
    provider: ProviderConfig | None = None,
    provider_name: str = "test-provider",
) -> EligibleModel:
    return EligibleModel(
        provider_name=provider_name,
        provider=provider or _PROVIDER_US,
        model=model or _model(),
    )


def _request(
    content: str = "Hello",
    stream: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        messages=[ChatMessage(role="user", content=content)],
        stream=stream,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        tool_choice=tool_choice,
    )


def _mock_litellm_response(
    content: str = "Hi there",
    model: str = "openai/test-model",
    tool_calls: list[dict[str, Any]] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> MagicMock:
    """Create a mock that looks like a LiteLLM ModelResponse."""
    msg = MagicMock()
    msg.role = "assistant"
    msg.content = content
    msg.tool_calls = tool_calls

    choice = MagicMock()
    choice.index = 0
    choice.message = msg
    choice.finish_reason = "stop"

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens

    response = MagicMock()
    response.id = "chatcmpl-test-123"
    response.object = "chat.completion"
    response.created = 1700000000
    response.model = model
    response.choices = [choice]
    response.usage = usage

    return response


def _mock_stream_chunks(
    contents: list[str] | None = None,
) -> AsyncMock:
    """Create a mock async iterator that yields stream chunks."""
    if contents is None:
        contents = ["Hello", " world"]

    chunks = []
    for text in contents:
        chunk = MagicMock()
        chunk.model_dump_json = MagicMock(return_value=json.dumps({
            "choices": [{"delta": {"content": text}}],
        }))
        chunks.append(chunk)

    async def _aiter():
        for c in chunks:
            yield c

    return _aiter()


# ---------------------------------------------------------------------------
# _build_litellm_kwargs
# ---------------------------------------------------------------------------


class TestBuildLitellmKwargs:
    """Tests for the kwargs builder helper."""

    def test_basic_kwargs(self) -> None:
        eligible = _eligible()
        req = _request(content="Hi")
        kwargs = _build_litellm_kwargs(eligible, req)

        assert kwargs["model"] == "openai/test-model"
        assert kwargs["messages"] == [{"role": "user", "content": "Hi"}]
        assert kwargs["stream"] is False

    def test_optional_params_forwarded(self) -> None:
        eligible = _eligible()
        req = _request(temperature=0.7, max_tokens=100)
        kwargs = _build_litellm_kwargs(eligible, req)

        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 100

    def test_optional_params_omitted_when_none(self) -> None:
        eligible = _eligible()
        req = _request()
        kwargs = _build_litellm_kwargs(eligible, req)

        assert "temperature" not in kwargs
        assert "max_tokens" not in kwargs
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs

    def test_tools_forwarded(self) -> None:
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        eligible = _eligible()
        req = _request(tools=tools, tool_choice="auto")
        kwargs = _build_litellm_kwargs(eligible, req)

        assert kwargs["tools"] == tools
        assert kwargs["tool_choice"] == "auto"


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


class TestEstimateCost:
    """Tests for cost estimation helper."""

    def test_basic_cost(self) -> None:
        from llm_broker.models import Usage

        eligible = _eligible(model=_model(cost=0.01))
        usage = Usage(prompt_tokens=500, completion_tokens=500, total_tokens=1000)
        cost = _estimate_cost(eligible, usage)
        assert cost == pytest.approx(0.01)  # 1000/1000 * 0.01

    def test_zero_cost_for_free_model(self) -> None:
        from llm_broker.models import Usage

        eligible = _eligible(model=_model(cost=0.0))
        usage = Usage(prompt_tokens=500, completion_tokens=500, total_tokens=1000)
        cost = _estimate_cost(eligible, usage)
        assert cost == 0.0


# ---------------------------------------------------------------------------
# dispatch — non-streaming
# ---------------------------------------------------------------------------


class TestDispatch:
    """Tests for the non-streaming dispatch function."""

    @patch("llm_broker.proxy.litellm")
    async def test_successful_dispatch(self, mock_litellm: MagicMock) -> None:
        """A successful dispatch should return a properly formatted response."""
        mock_litellm.acompletion = AsyncMock(return_value=_mock_litellm_response())
        mock_litellm.exceptions = __import__("litellm").exceptions

        models = [_eligible()]
        req = _request()
        response = await dispatch(models, req)

        assert response.id == "chatcmpl-test-123"
        assert response.object == "chat.completion"
        assert response.model == "test-model"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Hi there"
        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15

    @patch("llm_broker.proxy.litellm")
    async def test_parameters_forwarded(self, mock_litellm: MagicMock) -> None:
        """Temperature, max_tokens, and other params should be passed to LiteLLM."""
        mock_litellm.acompletion = AsyncMock(return_value=_mock_litellm_response())
        mock_litellm.exceptions = __import__("litellm").exceptions

        models = [_eligible()]
        req = _request(temperature=0.5, max_tokens=200)
        await dispatch(models, req)

        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 200

    @patch("llm_broker.proxy.litellm")
    async def test_tools_passed_through(self, mock_litellm: MagicMock) -> None:
        """Tool definitions should be forwarded to LiteLLM."""
        tool_call_response = MagicMock()
        tool_call_response.id = "call_123"
        tool_call_response.type = "function"
        tool_call_response.function = MagicMock()
        tool_call_response.function.name = "get_weather"
        tool_call_response.function.arguments = '{"location": "NYC"}'
        tool_call_response.model_dump = MagicMock(return_value={
            "id": "call_123",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
        })

        resp = _mock_litellm_response(content=None, tool_calls=[tool_call_response])
        mock_litellm.acompletion = AsyncMock(return_value=resp)
        mock_litellm.exceptions = __import__("litellm").exceptions

        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        models = [_eligible()]
        req = _request(tools=tools, tool_choice="auto")
        response = await dispatch(models, req)

        # Tools should be forwarded in the call
        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        assert call_kwargs["tools"] == tools
        assert call_kwargs["tool_choice"] == "auto"

        # Tool calls should appear in the response
        assert response.choices[0].message.tool_calls is not None
        assert len(response.choices[0].message.tool_calls) == 1

    @patch("llm_broker.proxy.litellm")
    async def test_fallback_on_primary_failure(self, mock_litellm: MagicMock) -> None:
        """If the primary model fails, should fallback to the next model."""
        import litellm as real_litellm

        mock_litellm.exceptions = real_litellm.exceptions

        primary_model = _model(id="primary", litellm_model="openai/primary", cost=0.01)
        fallback_model = _model(id="fallback", litellm_model="openai/fallback", cost=0.005)

        primary = _eligible(model=primary_model, provider_name="primary-provider")
        fallback = _eligible(model=fallback_model, provider_name="fallback-provider")

        # First call raises, second succeeds
        mock_litellm.acompletion = AsyncMock(
            side_effect=[
                real_litellm.exceptions.APIError(
                    message="Service error",
                    model="openai/primary",
                    llm_provider="openai",
                    status_code=500,
                ),
                _mock_litellm_response(content="Fallback response", model="openai/fallback"),
            ]
        )

        response = await dispatch([primary, fallback], _request())

        assert response.model == "fallback"
        assert response.choices[0].message.content == "Fallback response"
        assert mock_litellm.acompletion.call_count == 2

    @patch("llm_broker.proxy.litellm")
    async def test_all_models_fail_raises(self, mock_litellm: MagicMock) -> None:
        """If every model fails, AllModelsFailedError should be raised."""
        import litellm as real_litellm

        mock_litellm.exceptions = real_litellm.exceptions

        mock_litellm.acompletion = AsyncMock(
            side_effect=real_litellm.exceptions.APIConnectionError(
                message="Connection failed",
                model="openai/test",
                llm_provider="openai",
            )
        )

        model_a = _eligible(model=_model(id="model-a"), provider_name="provider-a")
        model_b = _eligible(model=_model(id="model-b"), provider_name="provider-b")

        with pytest.raises(AllModelsFailedError) as exc_info:
            await dispatch([model_a, model_b], _request())

        assert len(exc_info.value.errors) == 2
        assert exc_info.value.errors[0][0] == "model-a"
        assert exc_info.value.errors[1][0] == "model-b"

    async def test_empty_models_raises(self) -> None:
        """Dispatch with no models should raise AllModelsFailedError."""
        with pytest.raises(AllModelsFailedError):
            await dispatch([], _request())

    @patch("llm_broker.proxy.litellm")
    async def test_cost_and_latency_logged(
        self, mock_litellm: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Successful dispatch should log model, provider, latency, cost."""
        mock_litellm.acompletion = AsyncMock(return_value=_mock_litellm_response())
        mock_litellm.exceptions = __import__("litellm").exceptions

        models = [_eligible(model=_model(cost=0.01))]
        req = _request()

        with caplog.at_level(logging.INFO, logger="llm_broker.proxy"):
            await dispatch(models, req)

        # Find the structured log record
        log_records = [r for r in caplog.records if "llm_request" in r.message]
        assert len(log_records) == 1

        log_data = json.loads(log_records[0].message)
        assert log_data["event"] == "llm_request"
        assert log_data["model_id"] == "test-model"
        assert log_data["provider"] == "test-provider"
        assert log_data["success"] is True
        assert log_data["latency_ms"] >= 0
        assert log_data["estimated_cost"] >= 0
        assert log_data["attempt"] == 1

    @patch("llm_broker.proxy.litellm")
    async def test_failure_logged_before_fallback(
        self, mock_litellm: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Each failed attempt should be logged before trying the next model."""
        import litellm as real_litellm

        mock_litellm.exceptions = real_litellm.exceptions

        mock_litellm.acompletion = AsyncMock(
            side_effect=[
                real_litellm.exceptions.RateLimitError(
                    message="Rate limited",
                    model="openai/model-a",
                    llm_provider="openai",
                    response=MagicMock(status_code=429),
                ),
                _mock_litellm_response(content="OK"),
            ]
        )

        model_a = _eligible(model=_model(id="model-a"), provider_name="provider-a")
        model_b = _eligible(model=_model(id="model-b"), provider_name="provider-b")

        with caplog.at_level(logging.INFO, logger="llm_broker.proxy"):
            await dispatch([model_a, model_b], _request())

        log_records = [r for r in caplog.records if "llm_request" in r.message]
        assert len(log_records) == 2

        # First log: failure
        first = json.loads(log_records[0].message)
        assert first["success"] is False
        assert first["model_id"] == "model-a"
        assert first["attempt"] == 1
        assert "error" in first

        # Second log: success
        second = json.loads(log_records[1].message)
        assert second["success"] is True
        assert second["model_id"] == "model-b"
        assert second["attempt"] == 2


# ---------------------------------------------------------------------------
# dispatch_stream — streaming
# ---------------------------------------------------------------------------


class TestDispatchStream:
    """Tests for the streaming dispatch function."""

    @patch("llm_broker.proxy.litellm")
    async def test_streaming_yields_chunks(self, mock_litellm: MagicMock) -> None:
        """Streaming should yield SSE-formatted chunks and a DONE sentinel."""
        mock_litellm.acompletion = AsyncMock(return_value=_mock_stream_chunks(["Hi", " there"]))
        mock_litellm.exceptions = __import__("litellm").exceptions

        models = [_eligible()]
        req = _request(stream=True)

        chunks = []
        async for chunk in dispatch_stream(models, req):
            chunks.append(chunk)

        # Should have content chunks + [DONE]
        assert len(chunks) == 3
        assert chunks[0].startswith("data: ")
        assert chunks[1].startswith("data: ")
        assert chunks[2] == "data: [DONE]\n\n"

        # Verify chunk content
        data_0 = json.loads(chunks[0].removeprefix("data: ").strip())
        assert data_0["choices"][0]["delta"]["content"] == "Hi"

    @patch("llm_broker.proxy.litellm")
    async def test_stream_fallback_on_failure(self, mock_litellm: MagicMock) -> None:
        """If the primary model fails before streaming, fallback should work."""
        import litellm as real_litellm

        mock_litellm.exceptions = real_litellm.exceptions

        mock_litellm.acompletion = AsyncMock(
            side_effect=[
                real_litellm.exceptions.Timeout(
                    message="Timed out",
                    model="openai/primary",
                    llm_provider="openai",
                ),
                _mock_stream_chunks(["Fallback"]),
            ]
        )

        primary = _eligible(model=_model(id="primary"), provider_name="p1")
        fallback = _eligible(model=_model(id="fallback"), provider_name="p2")

        chunks = []
        async for chunk in dispatch_stream([primary, fallback], _request(stream=True)):
            chunks.append(chunk)

        assert len(chunks) == 2  # 1 content chunk + [DONE]
        data = json.loads(chunks[0].removeprefix("data: ").strip())
        assert data["choices"][0]["delta"]["content"] == "Fallback"

    @patch("llm_broker.proxy.litellm")
    async def test_stream_all_fail_raises(self, mock_litellm: MagicMock) -> None:
        """If all models fail during streaming, AllModelsFailedError is raised."""
        import litellm as real_litellm

        mock_litellm.exceptions = real_litellm.exceptions

        mock_litellm.acompletion = AsyncMock(
            side_effect=real_litellm.exceptions.ServiceUnavailableError(
                message="Unavailable",
                model="openai/test",
                llm_provider="openai",
                response=MagicMock(status_code=503),
            )
        )

        model_a = _eligible(model=_model(id="a"), provider_name="p1")
        model_b = _eligible(model=_model(id="b"), provider_name="p2")

        with pytest.raises(AllModelsFailedError) as exc_info:
            async for _ in dispatch_stream([model_a, model_b], _request(stream=True)):
                pass  # pragma: no cover

        assert len(exc_info.value.errors) == 2

    async def test_stream_empty_models_raises(self) -> None:
        """Streaming with no models should raise AllModelsFailedError."""
        with pytest.raises(AllModelsFailedError):
            async for _ in dispatch_stream([], _request(stream=True)):
                pass  # pragma: no cover

    @patch("llm_broker.proxy.litellm")
    async def test_stream_logs_success(
        self, mock_litellm: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Streaming should log a success record after completion."""
        mock_litellm.acompletion = AsyncMock(return_value=_mock_stream_chunks(["OK"]))
        mock_litellm.exceptions = __import__("litellm").exceptions

        models = [_eligible()]
        req = _request(stream=True)

        with caplog.at_level(logging.INFO, logger="llm_broker.proxy"):
            async for _ in dispatch_stream(models, req):
                pass

        log_records = [r for r in caplog.records if "llm_request" in r.message]
        assert len(log_records) == 1
        log_data = json.loads(log_records[0].message)
        assert log_data["success"] is True
