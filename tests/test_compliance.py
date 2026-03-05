"""Tests for the Compliance Gateway (Layer 1)."""

from __future__ import annotations

from llm_broker.compliance import apply_pii_policy, get_eligible_models
from llm_broker.config import BrokerConfig
from llm_broker.models import ChatMessage, RepoConfig


class TestEligibleModels:
    """Tests for provider/model filtering via get_eligible_models."""

    def test_fintech_only_azure_foundry(self, config: BrokerConfig) -> None:
        """fintech-app (EU residency) should only see azure-foundry models."""
        repo = config.repos["fintech-app"]
        eligible = get_eligible_models(repo, config.providers)
        provider_names = {e.provider_name for e in eligible}
        assert provider_names == {"azure-foundry"}

    def test_fintech_model_count(self, config: BrokerConfig) -> None:
        """fintech-app should see both azure-foundry models (premium max_tier)."""
        repo = config.repos["fintech-app"]
        eligible = get_eligible_models(repo, config.providers)
        assert len(eligible) == 2
        model_ids = {e.model.id for e in eligible}
        assert model_ids == {"azure-claude", "azure-gpt4o"}

    def test_internal_tooling_providers(self, config: BrokerConfig) -> None:
        """internal-tooling (any residency) should see anthropic, openai, local."""
        repo = config.repos["internal-tooling"]
        eligible = get_eligible_models(repo, config.providers)
        provider_names = {e.provider_name for e in eligible}
        assert provider_names == {"anthropic-saas", "openai-saas", "local"}

    def test_internal_tooling_model_count(self, config: BrokerConfig) -> None:
        """internal-tooling should see all 6 models from its 3 providers."""
        repo = config.repos["internal-tooling"]
        eligible = get_eligible_models(repo, config.providers)
        assert len(eligible) == 6

    def test_max_tier_standard_excludes_premium(self, config: BrokerConfig) -> None:
        """Setting max_tier to 'standard' should exclude premium models."""
        repo = RepoConfig(
            repo="test-repo",
            allowed_providers=["anthropic-saas", "openai-saas", "local"],
            data_residency="any",
            max_tier="standard",
        )
        eligible = get_eligible_models(repo, config.providers)
        tiers = {e.model.tier for e in eligible}
        assert "premium" not in tiers
        # Should have: claude-haiku (standard), gpt-4o-mini (standard),
        # qwen3 (free), llama (free) = 4 models
        assert len(eligible) == 4

    def test_max_tier_free_only(self, config: BrokerConfig) -> None:
        """Setting max_tier to 'free' should only include free models."""
        repo = RepoConfig(
            repo="test-repo",
            allowed_providers=["anthropic-saas", "openai-saas", "local"],
            data_residency="any",
            max_tier="free",
        )
        eligible = get_eligible_models(repo, config.providers)
        tiers = {e.model.tier for e in eligible}
        assert tiers == {"free"}
        assert len(eligible) == 2  # qwen3, llama

    def test_data_residency_eu_filters_non_eu(self, config: BrokerConfig) -> None:
        """EU residency should exclude non-EU providers even if allowed."""
        repo = RepoConfig(
            repo="test-repo",
            allowed_providers=["anthropic-saas", "azure-foundry"],
            data_residency="eu",
            max_tier="premium",
        )
        eligible = get_eligible_models(repo, config.providers)
        provider_names = {e.provider_name for e in eligible}
        # anthropic-saas is region=us, should be filtered out
        assert provider_names == {"azure-foundry"}

    def test_no_matching_providers_returns_empty(self) -> None:
        """If no providers match, the eligible list should be empty."""
        repo = RepoConfig(
            repo="ghost-repo",
            allowed_providers=["nonexistent-provider"],
            data_residency="any",
            max_tier="premium",
        )
        eligible = get_eligible_models(repo, {})
        assert eligible == []

    def test_residency_mismatch_returns_empty(self, config: BrokerConfig) -> None:
        """If residency doesn't match any allowed provider, result is empty."""
        repo = RepoConfig(
            repo="test-repo",
            allowed_providers=["anthropic-saas"],  # region=us
            data_residency="eu",
            max_tier="premium",
        )
        eligible = get_eligible_models(repo, config.providers)
        assert eligible == []

    def test_eligible_model_carries_provider_info(self, config: BrokerConfig) -> None:
        """Each EligibleModel should carry the full provider config."""
        repo = config.repos["fintech-app"]
        eligible = get_eligible_models(repo, config.providers)
        for em in eligible:
            assert em.provider is config.providers[em.provider_name]
            assert em.model in em.provider.models


class TestPiiPolicy:
    """Tests for the apply_pii_policy helper."""

    def test_redact_policy_redacts(self) -> None:
        """pii_handling='redact' should redact user message PII."""
        repo = RepoConfig(
            repo="strict-repo",
            allowed_providers=[],
            data_residency="any",
            pii_handling="redact",
        )
        messages = [ChatMessage(role="user", content="Email: x@y.com")]
        result = apply_pii_policy(messages, repo)
        assert "x@y.com" not in (result[0].content or "")
        assert "[REDACTED_EMAIL]" in (result[0].content or "")

    def test_allow_policy_passes_through(self) -> None:
        """pii_handling='allow' should leave messages unchanged."""
        repo = RepoConfig(
            repo="lax-repo",
            allowed_providers=[],
            data_residency="any",
            pii_handling="allow",
        )
        messages = [ChatMessage(role="user", content="Email: x@y.com")]
        result = apply_pii_policy(messages, repo)
        assert result[0].content == "Email: x@y.com"

    def test_redact_preserves_system_messages(self) -> None:
        """Even with pii_handling='redact', system messages are untouched."""
        repo = RepoConfig(
            repo="strict-repo",
            allowed_providers=[],
            data_residency="any",
            pii_handling="redact",
        )
        messages = [ChatMessage(role="system", content="SSN: 123-45-6789")]
        result = apply_pii_policy(messages, repo)
        assert result[0].content == "SSN: 123-45-6789"
