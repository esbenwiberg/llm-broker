"""Tests for PII detection and redaction."""

from __future__ import annotations

from llm_broker.models import ChatMessage
from llm_broker.pii import redact_messages, redact_text


class TestRedactText:
    """Tests for the redact_text function."""

    def test_email_redacted(self) -> None:
        """Email addresses should be replaced with [REDACTED_EMAIL]."""
        assert redact_text("Contact alice@example.com for help") == (
            "Contact [REDACTED_EMAIL] for help"
        )

    def test_phone_redacted(self) -> None:
        """Phone numbers should be replaced with [REDACTED_PHONE]."""
        assert "[REDACTED_PHONE]" in redact_text("Call me at +1 555-123-4567")

    def test_ssn_redacted(self) -> None:
        """SSNs should be replaced with [REDACTED_SSN]."""
        assert redact_text("My SSN is 123-45-6789") == "My SSN is [REDACTED_SSN]"

    def test_multiple_pii_types(self) -> None:
        """Multiple PII types in one string should all be redacted."""
        text = "Email: bob@corp.io, SSN: 999-88-7777, Phone: +44 20 7946 0958"
        result = redact_text(text)
        assert "[REDACTED_EMAIL]" in result
        assert "[REDACTED_SSN]" in result
        assert "[REDACTED_PHONE]" in result
        # Originals must not survive
        assert "bob@corp.io" not in result
        assert "999-88-7777" not in result

    def test_no_pii_unchanged(self) -> None:
        """Text without PII should pass through unchanged."""
        text = "This is a normal sentence with no sensitive data."
        assert redact_text(text) == text

    def test_multiple_emails(self) -> None:
        """All email addresses in a string should be redacted."""
        text = "Send to a@b.com and c@d.org"
        result = redact_text(text)
        assert "a@b.com" not in result
        assert "c@d.org" not in result
        assert result.count("[REDACTED_EMAIL]") == 2

    def test_empty_string(self) -> None:
        """Empty string should remain empty."""
        assert redact_text("") == ""


class TestRedactMessages:
    """Tests for the redact_messages function."""

    def test_user_message_redacted(self) -> None:
        """User messages should have PII redacted."""
        messages = [ChatMessage(role="user", content="My email is test@test.com")]
        result = redact_messages(messages)
        assert result[0].content is not None
        assert "test@test.com" not in result[0].content
        assert "[REDACTED_EMAIL]" in result[0].content

    def test_system_message_not_redacted(self) -> None:
        """System messages should pass through without redaction."""
        messages = [ChatMessage(role="system", content="Contact admin@example.com")]
        result = redact_messages(messages)
        assert result[0].content == "Contact admin@example.com"

    def test_assistant_message_not_redacted(self) -> None:
        """Assistant messages should pass through without redaction."""
        messages = [ChatMessage(role="assistant", content="Your SSN is 111-22-3333")]
        result = redact_messages(messages)
        assert result[0].content == "Your SSN is 111-22-3333"

    def test_none_content_unchanged(self) -> None:
        """Messages with None content should not cause errors."""
        messages = [ChatMessage(role="user", content=None)]
        result = redact_messages(messages)
        assert result[0].content is None

    def test_mixed_roles(self) -> None:
        """Only user messages are redacted in a mixed conversation."""
        messages = [
            ChatMessage(role="system", content="Email: sys@test.com"),
            ChatMessage(role="user", content="My email is user@test.com"),
            ChatMessage(role="assistant", content="Got it, user@test.com"),
        ]
        result = redact_messages(messages)
        assert result[0].content == "Email: sys@test.com"
        assert "user@test.com" not in (result[1].content or "")
        assert result[2].content == "Got it, user@test.com"

    def test_original_messages_unmodified(self) -> None:
        """redact_messages should not mutate the original message objects."""
        original = ChatMessage(role="user", content="Email: a@b.com")
        redact_messages([original])
        assert original.content == "Email: a@b.com"

    def test_empty_list(self) -> None:
        """Empty message list should return empty list."""
        assert redact_messages([]) == []
