"""Pydantic models for LLM Broker configuration and API types."""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Provider / model configuration (providers.yml)
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """A single model entry within a provider."""

    id: str
    litellm_model: str
    tier: Literal["free", "standard", "premium"]
    quality: float
    cost_per_1k_tokens: float


class ProviderConfig(BaseModel):
    """A provider entry from providers.yml."""

    litellm_prefix: str
    region: str
    deployment: str
    models: list[ModelConfig]


# ---------------------------------------------------------------------------
# Repo configuration (configs/repos/*.yml)
# ---------------------------------------------------------------------------


class RepoConfig(BaseModel):
    """Per-repo compliance and routing rules."""

    repo: str
    allowed_providers: list[str]
    data_residency: str
    max_tier: Literal["free", "standard", "premium"] = "premium"
    pii_handling: Literal["redact", "allow"] = "allow"


# ---------------------------------------------------------------------------
# Router configuration (router.yml)
# ---------------------------------------------------------------------------


class RouterConfig(BaseModel):
    """Global router settings."""

    strategy: str = "mf"
    cost_threshold: float = 0.5
    strong_model: str = "claude-sonnet"
    weak_model: str = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Key mapping (keys.yml)
# ---------------------------------------------------------------------------


class KeyMapping(BaseModel):
    """Maps an API key to a repo and team."""

    repo: str
    team: str


# ---------------------------------------------------------------------------
# OpenAI-compatible chat completion request / response
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    stop: str | list[str] | None = None
    n: int = 1


class Choice(BaseModel):
    """A single completion choice."""

    index: int = 0
    message: ChatMessage
    finish_reason: str | None = "stop"


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str = ""
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[Choice] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
