"""PII detection and redaction for LLM Broker.

Provides regex-based redaction of common PII patterns (email addresses,
phone numbers, SSNs) from message content.
"""

from __future__ import annotations

import re

from llm_broker.models import ChatMessage

# ---------------------------------------------------------------------------
# PII patterns: (compiled regex, replacement placeholder)
# ---------------------------------------------------------------------------

_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"\+?[\d\s\-().]{7,15}"), "[REDACTED_PHONE]"),
]


def redact_text(text: str) -> str:
    """Apply all PII redaction patterns to *text* and return the result.

    Patterns are applied in order (SSN first, then email, then phone) so
    that more specific patterns match before the general phone pattern
    can consume digits that belong to an SSN.
    """
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Return a new list of messages with user message content redacted.

    Only messages with ``role == "user"`` and non-None content are
    redacted.  All other messages are passed through unchanged.
    """
    result: list[ChatMessage] = []
    for msg in messages:
        if msg.role == "user" and msg.content is not None:
            result.append(msg.model_copy(update={"content": redact_text(msg.content)}))
        else:
            result.append(msg)
    return result
