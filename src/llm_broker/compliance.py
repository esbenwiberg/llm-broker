"""Compliance Gateway (Layer 1) for LLM Broker.

Filters the global provider registry down to the models a given repo is
allowed to use, based on:

* ``allowed_providers`` – explicit provider allow-list
* ``data_residency`` – region constraint (``"any"`` permits all regions)
* ``max_tier`` – ceiling on model tier (free < standard < premium)

Also applies PII redaction to chat messages when the repo's
``pii_handling`` is set to ``"redact"``.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_broker.models import ChatMessage, ModelConfig, ProviderConfig, RepoConfig
from llm_broker.pii import redact_messages

# ---------------------------------------------------------------------------
# Tier ordering
# ---------------------------------------------------------------------------

_TIER_RANK: dict[str, int] = {
    "free": 0,
    "standard": 1,
    "premium": 2,
}


# ---------------------------------------------------------------------------
# Eligible model container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EligibleModel:
    """A model that has passed all compliance filters.

    Carries enough context for downstream layers (router, proxy) to
    dispatch a request without re-consulting the provider registry.
    """

    provider_name: str
    provider: ProviderConfig
    model: ModelConfig


# ---------------------------------------------------------------------------
# Filtering logic
# ---------------------------------------------------------------------------


def get_eligible_models(
    repo_config: RepoConfig,
    providers: dict[str, ProviderConfig],
) -> list[EligibleModel]:
    """Return the models that *repo_config* is allowed to use.

    Applies, in order:
    1. ``allowed_providers`` filter
    2. ``data_residency`` filter (skipped when residency is ``"any"``)
    3. ``max_tier`` ceiling
    """
    max_tier_rank = _TIER_RANK[repo_config.max_tier]
    eligible: list[EligibleModel] = []

    for provider_name in repo_config.allowed_providers:
        provider = providers.get(provider_name)
        if provider is None:
            continue

        # Data-residency check
        if repo_config.data_residency != "any" and provider.region != repo_config.data_residency:
            continue

        for model in provider.models:
            if _TIER_RANK[model.tier] <= max_tier_rank:
                eligible.append(
                    EligibleModel(
                        provider_name=provider_name,
                        provider=provider,
                        model=model,
                    )
                )

    return eligible


# ---------------------------------------------------------------------------
# PII handling
# ---------------------------------------------------------------------------


def apply_pii_policy(
    messages: list[ChatMessage],
    repo_config: RepoConfig,
) -> list[ChatMessage]:
    """Redact PII from *messages* if the repo policy requires it.

    When ``repo_config.pii_handling`` is ``"redact"``, user message
    content is scrubbed.  Otherwise messages are returned unchanged.
    """
    if repo_config.pii_handling == "redact":
        return redact_messages(messages)
    return messages
