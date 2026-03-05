"""End-to-end integration tests for the LLM Broker pipeline.

Tests the full request flow: Auth -> Compliance -> Router -> Proxy.
LiteLLM is mocked to avoid real API calls, but everything else runs
through the actual pipeline.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from llm_broker.config import BrokerConfig, load_config
from llm_broker.main import app
from tests.conftest import CONFIGS_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_litellm_response(
    content: str = "Hello!",
    model: str = "test/model",
    tool_calls: list[Any] | None = None,
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
    response.id = "chatcmpl-e2e-test"
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
        chunk.model_dump_json = MagicMock(
            return_value=json.dumps(
                {"choices": [{"delta": {"content": text}}]}
            )
        )
        chunks.append(chunk)

    async def _aiter():
        for c in chunks:
            yield c

    return _aiter()


@pytest.fixture()
def e2e_client() -> AsyncClient:
    """Return an async HTTPX test client with real config pre-loaded."""
    import llm_broker.config as cfg_module

    cfg_module._config = load_config(CONFIGS_DIR)
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://testserver")


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------


class TestAuthE2E:
    """Auth edge cases in the full pipeline."""

    @pytest.mark.asyncio
    async def test_missing_auth_returns_401(self, e2e_client: AsyncClient) -> None:
        """Request without Authorization header should return 401."""
        response = await e2e_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        assert response.status_code == 401
        assert response.json()["error"]["message"] == "Missing API key"

    @pytest.mark.asyncio
    async def test_invalid_key_returns_401(self, e2e_client: AsyncClient) -> None:
        """Request with unknown API key should return 401."""
        response = await e2e_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
            headers={"Authorization": "Bearer sk-nonexistent-key"},
        )
        assert response.status_code == 401
        assert response.json()["error"]["message"] == "Invalid API key"

    @pytest.mark.asyncio
    async def test_messages_invalid_key_returns_401(self, e2e_client: AsyncClient) -> None:
        """Anthropic endpoint with unknown key returns 401."""
        response = await e2e_client.post(
            "/v1/messages",
            json={"messages": [{"role": "user", "content": "hello"}]},
            headers={"Authorization": "Bearer sk-bad-key"},
        )
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# Fintech-app (sk-fintech-abc123) — strict compliance
# ---------------------------------------------------------------------------


class TestFintechE2E:
    """E2E tests for fintech-app repo (azure-foundry only, PII redacted)."""

    @patch("llm_broker.proxy.litellm")
    @pytest.mark.asyncio
    async def test_fintech_routes_to_azure_foundry(
        self, mock_litellm: MagicMock, e2e_client: AsyncClient
    ) -> None:
        """fintech-app should only route to azure-foundry models."""
        mock_litellm.acompletion = AsyncMock(
            return_value=_mock_litellm_response(model="azure/claude-sonnet")
        )
        mock_litellm.exceptions = __import__("litellm").exceptions

        response = await e2e_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
            headers={"Authorization": "Bearer sk-fintech-abc123"},
        )

        assert response.status_code == 200
        # Verify litellm was called with an azure model
        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        assert call_kwargs["model"].startswith("azure/")

    @patch("llm_broker.proxy.litellm")
    @pytest.mark.asyncio
    async def test_fintech_pii_redacted(
        self, mock_litellm: MagicMock, e2e_client: AsyncClient
    ) -> None:
        """fintech-app has pii_handling=redact, so PII should be scrubbed."""
        mock_litellm.acompletion = AsyncMock(
            return_value=_mock_litellm_response()
        )
        mock_litellm.exceptions = __import__("litellm").exceptions

        response = await e2e_client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "My email is alice@example.com and SSN 123-45-6789"}
                ]
            },
            headers={"Authorization": "Bearer sk-fintech-abc123"},
        )

        assert response.status_code == 200

        # Check what was actually sent to litellm
        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        sent_content = call_kwargs["messages"][0]["content"]
        assert "alice@example.com" not in sent_content
        assert "123-45-6789" not in sent_content
        assert "[REDACTED_EMAIL]" in sent_content
        assert "[REDACTED_SSN]" in sent_content


# ---------------------------------------------------------------------------
# Internal-tooling (sk-internal-xyz789) — relaxed compliance
# ---------------------------------------------------------------------------


class TestInternalToolingE2E:
    """E2E tests for internal-tooling repo (multiple providers, PII allowed)."""

    @patch("llm_broker.proxy.litellm")
    @pytest.mark.asyncio
    async def test_internal_routes_to_allowed_providers(
        self, mock_litellm: MagicMock, e2e_client: AsyncClient
    ) -> None:
        """internal-tooling can route to anthropic-saas, openai-saas, or local."""
        mock_litellm.acompletion = AsyncMock(
            return_value=_mock_litellm_response()
        )
        mock_litellm.exceptions = __import__("litellm").exceptions

        response = await e2e_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
            headers={"Authorization": "Bearer sk-internal-xyz789"},
        )

        assert response.status_code == 200
        # The model called should be from one of the allowed providers
        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        model = call_kwargs["model"]
        allowed_prefixes = ("anthropic/", "openai/", "ollama/")
        assert any(model.startswith(p) for p in allowed_prefixes), (
            f"Model {model} not from an allowed provider"
        )

    @patch("llm_broker.proxy.litellm")
    @pytest.mark.asyncio
    async def test_internal_pii_not_redacted(
        self, mock_litellm: MagicMock, e2e_client: AsyncClient
    ) -> None:
        """internal-tooling has pii_handling=allow, PII passes through."""
        mock_litellm.acompletion = AsyncMock(
            return_value=_mock_litellm_response()
        )
        mock_litellm.exceptions = __import__("litellm").exceptions

        response = await e2e_client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "My email is alice@example.com"}
                ]
            },
            headers={"Authorization": "Bearer sk-internal-xyz789"},
        )

        assert response.status_code == 200
        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        sent_content = call_kwargs["messages"][0]["content"]
        assert "alice@example.com" in sent_content


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestStreamingE2E:
    """E2E tests for streaming requests."""

    @patch("llm_broker.proxy.litellm")
    @pytest.mark.asyncio
    async def test_streaming_returns_sse(
        self, mock_litellm: MagicMock, e2e_client: AsyncClient
    ) -> None:
        """stream=True should return a text/event-stream response."""
        mock_litellm.acompletion = AsyncMock(
            return_value=_mock_stream_chunks(["Hi", " there"])
        )
        mock_litellm.exceptions = __import__("litellm").exceptions

        response = await e2e_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
            headers={"Authorization": "Bearer sk-internal-xyz789"},
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        # Parse SSE chunks
        body = response.text
        lines = [l for l in body.split("\n") if l.startswith("data: ")]
        # Should have 2 content chunks + [DONE]
        assert len(lines) == 3
        assert lines[-1] == "data: [DONE]"

        # Verify first chunk content
        chunk_0 = json.loads(lines[0].removeprefix("data: "))
        assert chunk_0["choices"][0]["delta"]["content"] == "Hi"

    @patch("llm_broker.proxy.litellm")
    @pytest.mark.asyncio
    async def test_streaming_with_pii_redaction(
        self, mock_litellm: MagicMock, e2e_client: AsyncClient
    ) -> None:
        """Streaming + PII redaction should still redact before dispatch."""
        mock_litellm.acompletion = AsyncMock(
            return_value=_mock_stream_chunks(["Redacted"])
        )
        mock_litellm.exceptions = __import__("litellm").exceptions

        response = await e2e_client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "My SSN is 111-22-3333"}
                ],
                "stream": True,
            },
            headers={"Authorization": "Bearer sk-fintech-abc123"},
        )

        assert response.status_code == 200
        # Verify PII was redacted in what was sent to litellm
        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        sent_content = call_kwargs["messages"][0]["content"]
        assert "111-22-3333" not in sent_content
        assert "[REDACTED_SSN]" in sent_content


# ---------------------------------------------------------------------------
# X-Size header routing
# ---------------------------------------------------------------------------


class TestSizeHintE2E:
    """E2E tests for X-Size header influence on routing."""

    @patch("llm_broker.proxy.litellm")
    @pytest.mark.asyncio
    async def test_x_size_large_selects_premium(
        self, mock_litellm: MagicMock, e2e_client: AsyncClient
    ) -> None:
        """X-Size: large should route to a premium-tier model."""
        mock_litellm.acompletion = AsyncMock(
            return_value=_mock_litellm_response()
        )
        mock_litellm.exceptions = __import__("litellm").exceptions

        response = await e2e_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={
                "Authorization": "Bearer sk-internal-xyz789",
                "X-Size": "large",
            },
        )

        assert response.status_code == 200
        # With X-Size: large, the first model tried should be premium
        # internal-tooling premium models: claude-sonnet, gpt-4o
        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        model = call_kwargs["model"]
        # Premium models from internal-tooling providers
        premium_models = {
            "anthropic/claude-sonnet-4-20250514",
            "openai/gpt-4o",
        }
        assert model in premium_models, f"Expected premium model, got {model}"

    @patch("llm_broker.proxy.litellm")
    @pytest.mark.asyncio
    async def test_x_size_small_selects_weak(
        self, mock_litellm: MagicMock, e2e_client: AsyncClient
    ) -> None:
        """X-Size: small should route to a standard/free-tier model."""
        mock_litellm.acompletion = AsyncMock(
            return_value=_mock_litellm_response()
        )
        mock_litellm.exceptions = __import__("litellm").exceptions

        response = await e2e_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={
                "Authorization": "Bearer sk-internal-xyz789",
                "X-Size": "small",
            },
        )

        assert response.status_code == 200
        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        model = call_kwargs["model"]
        # Weak models from internal-tooling: claude-haiku, gpt-4o-mini, qwen3, llama
        weak_models = {
            "anthropic/claude-3-5-haiku-20241022",
            "openai/gpt-4o-mini",
            "ollama/qwen3",
            "ollama/llama3.1",
        }
        assert model in weak_models, f"Expected weak model, got {model}"


# ---------------------------------------------------------------------------
# Tools / tool_choice passthrough
# ---------------------------------------------------------------------------


class TestToolsE2E:
    """E2E tests for tool definitions passing through the pipeline."""

    @patch("llm_broker.proxy.litellm")
    @pytest.mark.asyncio
    async def test_tools_passed_through(
        self, mock_litellm: MagicMock, e2e_client: AsyncClient
    ) -> None:
        """tools and tool_choice should be forwarded to the LLM provider."""
        tool_call = MagicMock()
        tool_call.model_dump = MagicMock(return_value={
            "id": "call_123",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
        })

        resp = _mock_litellm_response(content=None, tool_calls=[tool_call])
        mock_litellm.acompletion = AsyncMock(return_value=resp)
        mock_litellm.exceptions = __import__("litellm").exceptions

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ]

        response = await e2e_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "What is the weather in NYC?"}],
                "tools": tools,
                "tool_choice": "auto",
            },
            headers={"Authorization": "Bearer sk-internal-xyz789"},
        )

        assert response.status_code == 200

        # Verify tools were forwarded
        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        assert call_kwargs["tools"] == tools
        assert call_kwargs["tool_choice"] == "auto"

        # Verify tool calls in response
        data = response.json()
        assert data["choices"][0]["message"]["tool_calls"] is not None


# ---------------------------------------------------------------------------
# Provider failure -> 502
# ---------------------------------------------------------------------------


class TestProviderFailureE2E:
    """E2E tests for all-providers-fail scenario."""

    @patch("llm_broker.proxy.litellm")
    @pytest.mark.asyncio
    async def test_all_providers_fail_returns_502(
        self, mock_litellm: MagicMock, e2e_client: AsyncClient
    ) -> None:
        """When all providers fail, the endpoint should return 502."""
        import litellm as real_litellm

        mock_litellm.exceptions = real_litellm.exceptions
        mock_litellm.acompletion = AsyncMock(
            side_effect=real_litellm.exceptions.APIConnectionError(
                message="Connection failed",
                model="test/model",
                llm_provider="test",
            )
        )

        response = await e2e_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
            headers={"Authorization": "Bearer sk-internal-xyz789"},
        )

        assert response.status_code == 502
        data = response.json()
        assert data["error"]["message"] == "All providers failed"
        assert data["error"]["type"] == "provider_error"


# ---------------------------------------------------------------------------
# Anthropic /v1/messages endpoint
# ---------------------------------------------------------------------------


class TestAnthropicMessagesE2E:
    """E2E tests for the Anthropic-format /v1/messages endpoint."""

    @patch("llm_broker.proxy.litellm")
    @pytest.mark.asyncio
    async def test_anthropic_format_request(
        self, mock_litellm: MagicMock, e2e_client: AsyncClient
    ) -> None:
        """Anthropic-format messages should be converted and dispatched."""
        mock_litellm.acompletion = AsyncMock(
            return_value=_mock_litellm_response(content="Hi from the broker!")
        )
        mock_litellm.exceptions = __import__("litellm").exceptions

        response = await e2e_client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Hello, how are you?"}],
                    }
                ],
                "max_tokens": 100,
            },
            headers={"Authorization": "Bearer sk-internal-xyz789"},
        )

        assert response.status_code == 200
        data = response.json()

        # Should return Anthropic format
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert len(data["content"]) == 1
        assert data["content"][0]["type"] == "text"
        assert data["content"][0]["text"] == "Hi from the broker!"
        assert data["stop_reason"] == "end_turn"
        assert "input_tokens" in data["usage"]
        assert "output_tokens" in data["usage"]

    @patch("llm_broker.proxy.litellm")
    @pytest.mark.asyncio
    async def test_anthropic_plain_string_content(
        self, mock_litellm: MagicMock, e2e_client: AsyncClient
    ) -> None:
        """Anthropic messages with plain string content should work too."""
        mock_litellm.acompletion = AsyncMock(
            return_value=_mock_litellm_response(content="OK")
        )
        mock_litellm.exceptions = __import__("litellm").exceptions

        response = await e2e_client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 100,
            },
            headers={"Authorization": "Bearer sk-internal-xyz789"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "message"

        # Verify the message was passed correctly to litellm
        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        sent_content = call_kwargs["messages"][0]["content"]
        assert sent_content == "Hello"

    @patch("llm_broker.proxy.litellm")
    @pytest.mark.asyncio
    async def test_anthropic_system_message(
        self, mock_litellm: MagicMock, e2e_client: AsyncClient
    ) -> None:
        """Anthropic top-level system field should be prepended as system message."""
        mock_litellm.acompletion = AsyncMock(
            return_value=_mock_litellm_response(content="I am a helper")
        )
        mock_litellm.exceptions = __import__("litellm").exceptions

        response = await e2e_client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet",
                "system": "You are a helpful assistant.",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 100,
            },
            headers={"Authorization": "Bearer sk-internal-xyz789"},
        )

        assert response.status_code == 200

        # System message should be first in the messages sent to litellm
        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"

    @patch("llm_broker.proxy.litellm")
    @pytest.mark.asyncio
    async def test_anthropic_pii_redaction(
        self, mock_litellm: MagicMock, e2e_client: AsyncClient
    ) -> None:
        """PII should be redacted for fintech-app even on the /v1/messages endpoint."""
        mock_litellm.acompletion = AsyncMock(
            return_value=_mock_litellm_response()
        )
        mock_litellm.exceptions = __import__("litellm").exceptions

        response = await e2e_client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "My email is secret@corp.com"}
                        ],
                    }
                ],
                "max_tokens": 100,
            },
            headers={"Authorization": "Bearer sk-fintech-abc123"},
        )

        assert response.status_code == 200
        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        sent_content = call_kwargs["messages"][0]["content"]
        assert "secret@corp.com" not in sent_content
        assert "[REDACTED_EMAIL]" in sent_content

    @patch("llm_broker.proxy.litellm")
    @pytest.mark.asyncio
    async def test_anthropic_streaming(
        self, mock_litellm: MagicMock, e2e_client: AsyncClient
    ) -> None:
        """Streaming via /v1/messages should return SSE chunks."""
        mock_litellm.acompletion = AsyncMock(
            return_value=_mock_stream_chunks(["Stream", "ed"])
        )
        mock_litellm.exceptions = __import__("litellm").exceptions

        response = await e2e_client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "max_tokens": 100,
            },
            headers={"Authorization": "Bearer sk-internal-xyz789"},
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

    @patch("llm_broker.proxy.litellm")
    @pytest.mark.asyncio
    async def test_anthropic_all_fail_returns_502(
        self, mock_litellm: MagicMock, e2e_client: AsyncClient
    ) -> None:
        """When all providers fail on /v1/messages, return 502."""
        import litellm as real_litellm

        mock_litellm.exceptions = real_litellm.exceptions
        mock_litellm.acompletion = AsyncMock(
            side_effect=real_litellm.exceptions.APIConnectionError(
                message="Connection failed",
                model="test/model",
                llm_provider="test",
            )
        )

        response = await e2e_client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 100,
            },
            headers={"Authorization": "Bearer sk-internal-xyz789"},
        )

        assert response.status_code == 502
        data = response.json()
        assert data["error"]["message"] == "All providers failed"
