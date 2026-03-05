"""Intelligent Router (Layer 2) for LLM Broker.

Classifies prompts as strong/weak using RouteLLM (when available) or a
simple heuristic fallback, then selects the cheapest model in the chosen
tier and builds an ordered fallback chain.

Tier grouping:
* **Strong:** premium tier models
* **Weak:** standard and free tier models

Size-hint overrides (from ``X-Size`` header):
* ``large``  -- always strong tier (skip classification)
* ``small``  -- always weak tier (skip classification)
* ``medium`` or absent -- let the classifier decide
"""

from __future__ import annotations

import logging
from typing import Literal

from llm_broker.compliance import EligibleModel
from llm_broker.models import ChatMessage, RouterConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier classification helpers
# ---------------------------------------------------------------------------

_STRONG_TIERS: frozenset[str] = frozenset({"premium"})
_WEAK_TIERS: frozenset[str] = frozenset({"standard", "free"})

SizeHint = Literal["large", "small", "medium"] | None
TierChoice = Literal["strong", "weak"]


def _split_by_tier(
    eligible: list[EligibleModel],
) -> tuple[list[EligibleModel], list[EligibleModel]]:
    """Partition *eligible* models into (strong, weak) lists sorted by cost."""
    strong = sorted(
        (m for m in eligible if m.model.tier in _STRONG_TIERS),
        key=lambda m: m.model.cost_per_1k_tokens,
    )
    weak = sorted(
        (m for m in eligible if m.model.tier in _WEAK_TIERS),
        key=lambda m: m.model.cost_per_1k_tokens,
    )
    return strong, weak


# ---------------------------------------------------------------------------
# RouteLLM integration (optional)
# ---------------------------------------------------------------------------

_routellm_available: bool = False
_routellm_controller = None


def _init_routellm(config: RouterConfig) -> bool:
    """Attempt to initialise the RouteLLM controller.

    Returns ``True`` if RouteLLM is usable, ``False`` otherwise.
    The controller is cached at module level after the first successful init.
    """
    global _routellm_available, _routellm_controller  # noqa: PLW0603

    if _routellm_controller is not None:
        return True

    try:
        from routellm.controller import Controller

        _routellm_controller = Controller(
            routers=[config.strategy],
            strong_model=config.strong_model,
            weak_model=config.weak_model,
        )
        _routellm_available = True
        logger.info("RouteLLM controller initialised (strategy=%s)", config.strategy)
        return True
    except Exception:  # noqa: BLE001
        logger.warning(
            "RouteLLM unavailable; falling back to heuristic classifier",
            exc_info=True,
        )
        _routellm_available = False
        return False


def _classify_routellm(
    messages: list[ChatMessage],
    config: RouterConfig,
) -> TierChoice:
    """Use RouteLLM to classify messages as strong or weak."""
    if _routellm_controller is None:
        return _classify_heuristic(messages)

    try:
        # RouteLLM expects OpenAI-format dicts
        oai_messages = [
            {"role": m.role, "content": m.content or ""}
            for m in messages
        ]
        routed = _routellm_controller.route(
            prompt=oai_messages,
            router=config.strategy,
            threshold=config.cost_threshold,
        )
        # route() returns the model name; compare against strong_model
        if routed == config.strong_model:
            return "strong"
        return "weak"
    except Exception:  # noqa: BLE001
        logger.warning("RouteLLM classification failed; using heuristic", exc_info=True)
        return _classify_heuristic(messages)


# ---------------------------------------------------------------------------
# Heuristic fallback classifier
# ---------------------------------------------------------------------------

_HEURISTIC_LENGTH_THRESHOLD = 200


def _classify_heuristic(messages: list[ChatMessage]) -> TierChoice:
    """Simple heuristic: long / complex prompts go to strong tier.

    Uses total character count of user messages as a rough proxy for
    complexity.  Prompts exceeding ``_HEURISTIC_LENGTH_THRESHOLD`` chars
    are routed to the strong tier.
    """
    total_chars = sum(
        len(m.content or "")
        for m in messages
        if m.role == "user"
    )
    if total_chars > _HEURISTIC_LENGTH_THRESHOLD:
        return "strong"
    return "weak"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_prompt(
    messages: list[ChatMessage],
    config: RouterConfig,
    *,
    size_hint: SizeHint = None,
) -> TierChoice:
    """Determine whether *messages* should be routed to strong or weak tier.

    When *size_hint* is ``"large"`` or ``"small"`` the classification is
    forced without consulting any classifier.  Otherwise RouteLLM is used
    if available, falling back to a length heuristic.
    """
    if size_hint == "large":
        return "strong"
    if size_hint == "small":
        return "weak"

    # Try RouteLLM first
    if _routellm_available or _init_routellm(config):
        return _classify_routellm(messages, config)

    return _classify_heuristic(messages)


def reset_router() -> None:
    """Reset module-level RouteLLM state (useful for testing)."""
    global _routellm_available, _routellm_controller  # noqa: PLW0603
    _routellm_available = False
    _routellm_controller = None


def route_request(
    eligible: list[EligibleModel],
    messages: list[ChatMessage],
    config: RouterConfig,
    *,
    size_hint: SizeHint = None,
) -> list[EligibleModel]:
    """Select and rank models for *messages* from the *eligible* set.

    Returns an ordered list where the first element is the primary model
    and the rest form the fallback chain:

    1. Cheapest model in the selected tier
    2. Remaining models in the same tier (by cost)
    3. Models in the other tier (by cost)

    Returns an empty list when *eligible* is empty.
    """
    if not eligible:
        return []

    strong, weak = _split_by_tier(eligible)

    # If only one tier is available, use it regardless of classification
    if strong and not weak:
        return strong
    if weak and not strong:
        return weak

    tier = classify_prompt(messages, config, size_hint=size_hint)

    if tier == "strong":
        return strong + weak
    return weak + strong
