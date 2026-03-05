"""Tests for configuration loading and API key resolution."""

from __future__ import annotations

import pytest

from llm_broker.config import BrokerConfig


class TestConfigLoading:
    """Tests that config files load correctly."""

    def test_providers_load(self, config: BrokerConfig) -> None:
        """All four providers should be loaded."""
        assert len(config.providers) == 4
        assert "anthropic-saas" in config.providers
        assert "azure-foundry" in config.providers
        assert "openai-saas" in config.providers
        assert "local" in config.providers

    def test_anthropic_saas_has_two_models(self, config: BrokerConfig) -> None:
        """anthropic-saas should have 2 models."""
        provider = config.providers["anthropic-saas"]
        assert len(provider.models) == 2

    def test_azure_foundry_has_two_models(self, config: BrokerConfig) -> None:
        """azure-foundry should have 2 models."""
        provider = config.providers["azure-foundry"]
        assert len(provider.models) == 2

    def test_openai_saas_has_two_models(self, config: BrokerConfig) -> None:
        """openai-saas should have 2 models."""
        provider = config.providers["openai-saas"]
        assert len(provider.models) == 2

    def test_local_has_two_models(self, config: BrokerConfig) -> None:
        """local should have 2 models."""
        provider = config.providers["local"]
        assert len(provider.models) == 2

    def test_provider_model_fields(self, config: BrokerConfig) -> None:
        """Models should have all required fields populated."""
        model = config.providers["anthropic-saas"].models[0]
        assert model.id == "claude-haiku"
        assert model.litellm_model == "anthropic/claude-3-5-haiku-20241022"
        assert model.tier == "standard"
        assert model.quality == 0.6
        assert model.cost_per_1k_tokens == 0.001

    def test_all_repos_load(self, config: BrokerConfig) -> None:
        """Both repo configs should be loaded."""
        assert len(config.repos) == 2
        assert "fintech-app" in config.repos
        assert "internal-tooling" in config.repos

    def test_fintech_repo_config(self, config: BrokerConfig) -> None:
        """fintech-app repo config should match YAML."""
        repo = config.repos["fintech-app"]
        assert repo.allowed_providers == ["azure-foundry"]
        assert repo.data_residency == "eu"
        assert repo.pii_handling == "redact"

    def test_internal_tooling_repo_config(self, config: BrokerConfig) -> None:
        """internal-tooling repo config should match YAML."""
        repo = config.repos["internal-tooling"]
        assert repo.allowed_providers == ["anthropic-saas", "openai-saas", "local"]
        assert repo.data_residency == "any"
        assert repo.pii_handling == "allow"

    def test_router_config(self, config: BrokerConfig) -> None:
        """Router config should match YAML."""
        assert config.router.strategy == "mf"
        assert config.router.cost_threshold == 0.5
        assert config.router.strong_model == "claude-sonnet"
        assert config.router.weak_model == "gpt-4o-mini"

    def test_keys_load(self, config: BrokerConfig) -> None:
        """Both API keys should be loaded."""
        assert len(config.keys) == 2
        assert "sk-fintech-abc123" in config.keys
        assert "sk-internal-xyz789" in config.keys

    def test_repo_names_sorted(self, config: BrokerConfig) -> None:
        """repo_names property should return sorted list."""
        assert config.repo_names == ["fintech-app", "internal-tooling"]


class TestKeyResolution:
    """Tests for API key → repo config resolution."""

    def test_valid_fintech_key(self, config: BrokerConfig) -> None:
        """Valid fintech key should resolve to fintech-app repo."""
        mapping = config.resolve_key("sk-fintech-abc123")
        assert mapping is not None
        assert mapping.repo == "fintech-app"
        assert mapping.team == "fintech-squad"

    def test_valid_internal_key(self, config: BrokerConfig) -> None:
        """Valid internal key should resolve to internal-tooling repo."""
        mapping = config.resolve_key("sk-internal-xyz789")
        assert mapping is not None
        assert mapping.repo == "internal-tooling"
        assert mapping.team == "platform-team"

    def test_invalid_key_returns_none(self, config: BrokerConfig) -> None:
        """Unknown API key should return None."""
        mapping = config.resolve_key("sk-unknown-key")
        assert mapping is None

    def test_empty_key_returns_none(self, config: BrokerConfig) -> None:
        """Empty API key should return None."""
        mapping = config.resolve_key("")
        assert mapping is None

    def test_key_to_repo_config(self, config: BrokerConfig) -> None:
        """Should be able to chain key resolution to repo config."""
        mapping = config.resolve_key("sk-fintech-abc123")
        assert mapping is not None
        repo_config = config.get_repo_config(mapping.repo)
        assert repo_config is not None
        assert repo_config.repo == "fintech-app"
        assert repo_config.pii_handling == "redact"

    def test_unknown_repo_returns_none(self, config: BrokerConfig) -> None:
        """get_repo_config with unknown name should return None."""
        assert config.get_repo_config("nonexistent-repo") is None


class TestEndpoints:
    """Tests for the FastAPI endpoints."""

    @pytest.mark.asyncio
    async def test_health(self, client) -> None:
        """GET /health should return status ok with repo names."""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["repos"] == ["fintech-app", "internal-tooling"]

    @pytest.mark.asyncio
    async def test_chat_completions_no_auth(self, client) -> None:
        """POST /v1/chat/completions without auth should return 401."""
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_chat_completions_invalid_key(self, client) -> None:
        """POST /v1/chat/completions with invalid key should return 401."""
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
            headers={"Authorization": "Bearer sk-invalid-key"},
        )
        assert response.status_code == 401
        data = response.json()
        assert data["error"]["message"] == "Invalid API key"

    @pytest.mark.asyncio
    async def test_chat_completions_valid_key_dispatches(self, client) -> None:
        """POST /v1/chat/completions with valid key should reach proxy layer."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_response = MagicMock()
        mock_response.id = "chatcmpl-test"
        mock_response.object = "chat.completion"
        mock_response.created = 1700000000
        mock_response.model = "azure/claude-sonnet"
        msg = MagicMock()
        msg.role = "assistant"
        msg.content = "Hi"
        msg.tool_calls = None
        choice = MagicMock()
        choice.index = 0
        choice.message = msg
        choice.finish_reason = "stop"
        mock_response.choices = [choice]
        usage = MagicMock()
        usage.prompt_tokens = 5
        usage.completion_tokens = 3
        usage.total_tokens = 8
        mock_response.usage = usage

        with patch("llm_broker.proxy.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            mock_litellm.exceptions = __import__("litellm").exceptions
            response = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}]},
                headers={"Authorization": "Bearer sk-fintech-abc123"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"

    @pytest.mark.asyncio
    async def test_messages_endpoint_valid_key(self, client) -> None:
        """POST /v1/messages with valid key should dispatch and return Anthropic format."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_response = MagicMock()
        mock_response.id = "chatcmpl-test"
        mock_response.object = "chat.completion"
        mock_response.created = 1700000000
        mock_response.model = "openai/gpt-4o"
        msg = MagicMock()
        msg.role = "assistant"
        msg.content = "Hello"
        msg.tool_calls = None
        choice = MagicMock()
        choice.index = 0
        choice.message = msg
        choice.finish_reason = "stop"
        mock_response.choices = [choice]
        usage = MagicMock()
        usage.prompt_tokens = 5
        usage.completion_tokens = 3
        usage.total_tokens = 8
        mock_response.usage = usage

        with patch("llm_broker.proxy.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            mock_litellm.exceptions = __import__("litellm").exceptions
            response = await client.post(
                "/v1/messages",
                json={"model": "claude-sonnet", "messages": [{"role": "user", "content": "hello"}]},
                headers={"Authorization": "Bearer sk-internal-xyz789"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_messages_endpoint_no_auth(self, client) -> None:
        """POST /v1/messages without auth should return 401."""
        response = await client.post(
            "/v1/messages",
            json={},
        )
        assert response.status_code == 401
