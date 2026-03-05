"""Tests for the Intelligent Router (Layer 2).

All tests mock the RouteLLM classifier so they work without RouteLLM
model files or an OpenAI API key.
"""

from __future__ import annotations

from unittest.mock import patch

from llm_broker.compliance import EligibleModel
from llm_broker.models import ChatMessage, ModelConfig, ProviderConfig, RouterConfig
from llm_broker.router import (
    _HEURISTIC_LENGTH_THRESHOLD,
    _classify_heuristic,
    _split_by_tier,
    classify_prompt,
    route_request,
)

# ---------------------------------------------------------------------------
# Test helpers / fixtures
# ---------------------------------------------------------------------------

_ROUTER_CONFIG = RouterConfig()

_PROVIDER_US = ProviderConfig(
    litellm_prefix="openai/",
    region="us",
    deployment="saas",
    models=[],  # populated per-model below
)

_PROVIDER_LOCAL = ProviderConfig(
    litellm_prefix="ollama/",
    region="local",
    deployment="on-prem",
    models=[],
)


def _model(
    id: str,
    tier: str = "standard",
    cost: float = 0.001,
) -> ModelConfig:
    return ModelConfig(
        id=id,
        litellm_model=f"test/{id}",
        tier=tier,
        quality=0.5,
        cost_per_1k_tokens=cost,
    )


def _eligible(
    model: ModelConfig,
    provider: ProviderConfig | None = None,
    provider_name: str = "test-provider",
) -> EligibleModel:
    return EligibleModel(
        provider_name=provider_name,
        provider=provider or _PROVIDER_US,
        model=model,
    )


# A realistic set of eligible models spanning tiers
PREMIUM_EXPENSIVE = _model("claude-sonnet", tier="premium", cost=0.015)
PREMIUM_CHEAP = _model("gpt-4o", tier="premium", cost=0.010)
STANDARD_MODEL = _model("claude-haiku", tier="standard", cost=0.001)
STANDARD_CHEAP = _model("gpt-4o-mini", tier="standard", cost=0.0006)
FREE_MODEL_A = _model("qwen3", tier="free", cost=0.0)
FREE_MODEL_B = _model("llama", tier="free", cost=0.0)

ALL_ELIGIBLE = [
    _eligible(PREMIUM_EXPENSIVE, provider_name="anthropic"),
    _eligible(PREMIUM_CHEAP, provider_name="openai"),
    _eligible(STANDARD_MODEL, provider_name="anthropic"),
    _eligible(STANDARD_CHEAP, provider_name="openai"),
    _eligible(FREE_MODEL_A, provider_name="local", provider=_PROVIDER_LOCAL),
    _eligible(FREE_MODEL_B, provider_name="local", provider=_PROVIDER_LOCAL),
]


def _user_msg(content: str) -> ChatMessage:
    return ChatMessage(role="user", content=content)


def _msgs(content: str) -> list[ChatMessage]:
    return [_user_msg(content)]


# ---------------------------------------------------------------------------
# Tier splitting
# ---------------------------------------------------------------------------


class TestSplitByTier:
    """Tests for _split_by_tier helper."""

    def test_splits_premium_as_strong(self) -> None:
        strong, weak = _split_by_tier(ALL_ELIGIBLE)
        assert all(m.model.tier == "premium" for m in strong)
        assert len(strong) == 2

    def test_splits_standard_and_free_as_weak(self) -> None:
        strong, weak = _split_by_tier(ALL_ELIGIBLE)
        assert all(m.model.tier in {"standard", "free"} for m in weak)
        assert len(weak) == 4

    def test_strong_sorted_by_cost(self) -> None:
        strong, _ = _split_by_tier(ALL_ELIGIBLE)
        costs = [m.model.cost_per_1k_tokens for m in strong]
        assert costs == sorted(costs)

    def test_weak_sorted_by_cost(self) -> None:
        _, weak = _split_by_tier(ALL_ELIGIBLE)
        costs = [m.model.cost_per_1k_tokens for m in weak]
        assert costs == sorted(costs)

    def test_empty_input(self) -> None:
        strong, weak = _split_by_tier([])
        assert strong == []
        assert weak == []


# ---------------------------------------------------------------------------
# Heuristic classifier
# ---------------------------------------------------------------------------


class TestHeuristicClassifier:
    """Tests for the length-based fallback classifier."""

    def test_short_prompt_weak(self) -> None:
        msgs = _msgs("Hi")
        assert _classify_heuristic(msgs) == "weak"

    def test_long_prompt_strong(self) -> None:
        msgs = _msgs("x" * (_HEURISTIC_LENGTH_THRESHOLD + 1))
        assert _classify_heuristic(msgs) == "strong"

    def test_threshold_boundary_weak(self) -> None:
        msgs = _msgs("x" * _HEURISTIC_LENGTH_THRESHOLD)
        assert _classify_heuristic(msgs) == "weak"

    def test_just_above_threshold_strong(self) -> None:
        msgs = _msgs("x" * (_HEURISTIC_LENGTH_THRESHOLD + 1))
        assert _classify_heuristic(msgs) == "strong"

    def test_multiple_user_messages_summed(self) -> None:
        # Each message is 120 chars, two of them exceed 200
        msgs = [_user_msg("a" * 120), _user_msg("b" * 120)]
        assert _classify_heuristic(msgs) == "strong"

    def test_system_message_not_counted(self) -> None:
        msgs = [
            ChatMessage(role="system", content="x" * 500),
            _user_msg("Hi"),
        ]
        assert _classify_heuristic(msgs) == "weak"


# ---------------------------------------------------------------------------
# classify_prompt with size hints
# ---------------------------------------------------------------------------


class TestClassifyPrompt:
    """Tests for classify_prompt including size hint overrides."""

    def test_size_hint_large_forces_strong(self) -> None:
        assert classify_prompt(_msgs("Hi"), _ROUTER_CONFIG, size_hint="large") == "strong"

    def test_size_hint_small_forces_weak(self) -> None:
        long_msg = "x" * 1000
        assert classify_prompt(_msgs(long_msg), _ROUTER_CONFIG, size_hint="small") == "weak"

    def test_size_hint_medium_uses_classifier(self) -> None:
        # medium should not force a tier; falls through to classifier
        with patch("llm_broker.router._classify_heuristic", return_value="strong"):
            result = classify_prompt(_msgs("Hi"), _ROUTER_CONFIG, size_hint="medium")
            assert result == "strong"

    def test_no_size_hint_uses_classifier(self) -> None:
        with patch("llm_broker.router._classify_heuristic", return_value="weak"):
            result = classify_prompt(_msgs("Hi"), _ROUTER_CONFIG, size_hint=None)
            assert result == "weak"


# ---------------------------------------------------------------------------
# route_request — main entry point
# ---------------------------------------------------------------------------


class TestRouteRequest:
    """Tests for the full route_request pipeline."""

    def test_empty_eligible_returns_empty(self) -> None:
        result = route_request([], _msgs("Hi"), _ROUTER_CONFIG)
        assert result == []

    def test_complex_prompt_selects_strong(self) -> None:
        """A complex (long) prompt should select a strong-tier model first."""
        long_prompt = "Explain the full theory of " + "x" * 300
        with patch("llm_broker.router.classify_prompt", return_value="strong"):
            result = route_request(ALL_ELIGIBLE, _msgs(long_prompt), _ROUTER_CONFIG)
        # Primary model should be from strong tier
        assert result[0].model.tier == "premium"

    def test_simple_prompt_selects_weak(self) -> None:
        """A simple (short) prompt should select a weak-tier model first."""
        with patch("llm_broker.router.classify_prompt", return_value="weak"):
            result = route_request(ALL_ELIGIBLE, _msgs("Hi"), _ROUTER_CONFIG)
        # Primary model should be from weak tier
        assert result[0].model.tier in {"standard", "free"}

    def test_size_large_selects_strong(self) -> None:
        result = route_request(
            ALL_ELIGIBLE, _msgs("Hi"), _ROUTER_CONFIG, size_hint="large"
        )
        assert result[0].model.tier == "premium"

    def test_size_small_selects_weak(self) -> None:
        result = route_request(
            ALL_ELIGIBLE, _msgs("complex " * 100), _ROUTER_CONFIG, size_hint="small"
        )
        assert result[0].model.tier in {"standard", "free"}

    def test_only_weak_eligible_returns_weak(self) -> None:
        """If only weak models are eligible, they're returned regardless."""
        weak_only = [
            _eligible(STANDARD_MODEL),
            _eligible(FREE_MODEL_A),
        ]
        # Even with a size_hint that would normally pick strong
        result = route_request(
            weak_only, _msgs("complex " * 100), _ROUTER_CONFIG, size_hint="large"
        )
        assert all(m.model.tier in {"standard", "free"} for m in result)
        assert len(result) == 2

    def test_only_strong_eligible_returns_strong(self) -> None:
        """If only strong models are eligible, they're returned regardless."""
        strong_only = [
            _eligible(PREMIUM_EXPENSIVE),
            _eligible(PREMIUM_CHEAP),
        ]
        result = route_request(
            strong_only, _msgs("Hi"), _ROUTER_CONFIG, size_hint="small"
        )
        assert all(m.model.tier == "premium" for m in result)
        assert len(result) == 2

    def test_primary_is_cheapest_in_tier(self) -> None:
        """The primary model should be the cheapest in its selected tier."""
        with patch("llm_broker.router.classify_prompt", return_value="strong"):
            result = route_request(ALL_ELIGIBLE, _msgs("test"), _ROUTER_CONFIG)
        # Cheapest premium is gpt-4o at 0.010
        assert result[0].model.id == "gpt-4o"
        assert result[0].model.cost_per_1k_tokens == 0.010

    def test_fallback_chain_ordering_strong(self) -> None:
        """When strong is selected: strong models by cost, then weak by cost."""
        with patch("llm_broker.router.classify_prompt", return_value="strong"):
            result = route_request(ALL_ELIGIBLE, _msgs("test"), _ROUTER_CONFIG)

        # First two should be strong (premium), sorted by cost
        assert result[0].model.tier == "premium"
        assert result[1].model.tier == "premium"
        assert result[0].model.cost_per_1k_tokens <= result[1].model.cost_per_1k_tokens

        # Remaining should be weak, sorted by cost
        weak_part = result[2:]
        assert all(m.model.tier in {"standard", "free"} for m in weak_part)
        costs = [m.model.cost_per_1k_tokens for m in weak_part]
        assert costs == sorted(costs)

    def test_fallback_chain_ordering_weak(self) -> None:
        """When weak is selected: weak models by cost, then strong by cost."""
        with patch("llm_broker.router.classify_prompt", return_value="weak"):
            result = route_request(ALL_ELIGIBLE, _msgs("test"), _ROUTER_CONFIG)

        # First four should be weak, sorted by cost
        weak_part = result[:4]
        assert all(m.model.tier in {"standard", "free"} for m in weak_part)
        costs_weak = [m.model.cost_per_1k_tokens for m in weak_part]
        assert costs_weak == sorted(costs_weak)

        # Last two should be strong (premium), sorted by cost
        strong_part = result[4:]
        assert all(m.model.tier == "premium" for m in strong_part)
        costs_strong = [m.model.cost_per_1k_tokens for m in strong_part]
        assert costs_strong == sorted(costs_strong)

    def test_all_eligible_models_in_result(self) -> None:
        """Every eligible model should appear exactly once in the result."""
        with patch("llm_broker.router.classify_prompt", return_value="strong"):
            result = route_request(ALL_ELIGIBLE, _msgs("test"), _ROUTER_CONFIG)
        assert len(result) == len(ALL_ELIGIBLE)
        result_ids = {m.model.id for m in result}
        eligible_ids = {m.model.id for m in ALL_ELIGIBLE}
        assert result_ids == eligible_ids

    def test_single_model_returns_single(self) -> None:
        """A single eligible model should be returned as-is."""
        single = [_eligible(STANDARD_MODEL)]
        result = route_request(single, _msgs("Hi"), _ROUTER_CONFIG)
        assert len(result) == 1
        assert result[0].model.id == STANDARD_MODEL.id

    def test_with_real_config(self, config) -> None:
        """Integration test with real provider configs."""
        from llm_broker.compliance import get_eligible_models

        repo = config.repos["internal-tooling"]
        eligible = get_eligible_models(repo, config.providers)
        result = route_request(eligible, _msgs("Hi"), config.router)
        # Should have all 6 eligible models
        assert len(result) == 6
        # All original eligible models should be present
        assert {m.model.id for m in result} == {m.model.id for m in eligible}
