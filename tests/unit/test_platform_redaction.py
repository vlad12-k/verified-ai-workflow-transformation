"""Tests for M6-B defensive telemetry redaction."""

import pytest

from vait.platform.security import (
    REDACTION_MARKER,
    redact_sensitive_text,
)


def test_safe_telemetry_text_is_unchanged() -> None:
    """Ordinary operational messages must remain readable."""
    value = "job_claimed experiment_id=experiment-001"

    assert redact_sensitive_text(value) == value


@pytest.mark.parametrize(
    (
        "value",
        "secret",
    ),
    [
        (
            "api_key=provider-secret-123",
            "provider-secret-123",
        ),
        (
            "API-KEY: provider-secret-456",
            "provider-secret-456",
        ),
        (
            "password=hunter-two",
            "hunter-two",
        ),
        (
            "client_secret=oauth-secret-value",
            "oauth-secret-value",
        ),
        (
            "access_token=access-secret-value",
            "access-secret-value",
        ),
        (
            "refresh-token: refresh-secret-value",
            "refresh-secret-value",
        ),
        (
            "auth_token=service-secret-value",
            "service-secret-value",
        ),
        (
            "database_url=postgresql://user:pass@db/vait",
            "postgresql://user:pass@db/vait",
        ),
        (
            'payload={"api_key":"nested-secret-value"}',
            "nested-secret-value",
        ),
        (
            "url=/callback?token=query-secret&state=safe",
            "query-secret",
        ),
    ],
)
def test_sensitive_assignments_are_redacted(
    value: str,
    secret: str,
) -> None:
    """Known credential-bearing field names must hide their values."""
    redacted = redact_sensitive_text(
        value
    )

    assert REDACTION_MARKER in redacted
    assert secret not in redacted


@pytest.mark.parametrize(
    (
        "value",
        "secret",
    ),
    [
        (
            "outbound Bearer abc.DEF-123_secret",
            "abc.DEF-123_secret",
        ),
        (
            "legacy Basic dXNlcjpwYXNzd29yZA==",
            "dXNlcjpwYXNzd29yZA==",
        ),
        (
            "Authorization: Bearer token-value-123",
            "token-value-123",
        ),
    ],
)
def test_authentication_scheme_credentials_are_redacted(
    value: str,
    secret: str,
) -> None:
    """Bearer and Basic credentials must not survive telemetry formatting."""
    redacted = redact_sensitive_text(
        value
    )

    assert REDACTION_MARKER in redacted
    assert secret not in redacted


@pytest.mark.parametrize(
    (
        "value",
        "secret",
    ),
    [
        (
            "postgresql://vait:database-password@db.internal/vait",
            "vait:database-password",
        ),
        (
            "redis://service:redis-password@cache.internal/0",
            "service:redis-password",
        ),
        (
            "https://client:client-password@example.test/resource",
            "client:client-password",
        ),
    ],
)
def test_uri_userinfo_credentials_are_redacted(
    value: str,
    secret: str,
) -> None:
    """Credential-bearing URI userinfo must not leave the process."""
    redacted = redact_sensitive_text(
        value
    )

    assert REDACTION_MARKER in redacted
    assert secret not in redacted
    assert "://" in redacted


def test_redaction_does_not_expose_secret_length_or_prefix() -> None:
    """The replacement marker must not retain recoverable secret fragments."""
    secret = "super-long-provider-secret-abcdef1234567890"

    redacted = redact_sensitive_text(
        f"api_key={secret}"
    )

    assert redacted == (
        f"api_key={REDACTION_MARKER}"
    )

    assert secret[:8] not in redacted


@pytest.mark.parametrize(
    "value",
    [
        "token_count=1200",
        "token_budget=500",
        "tokenizer=sentence-transformer",
        "secret_rotation=enabled",
        "password_policy=strict",
        "api_key_rotation=scheduled",
        "access_token_budget=100",
    ],
)
def test_non_secret_operational_fields_are_not_over_redacted(
    value: str,
) -> None:
    """Related operational field names must not be mistaken for secrets."""
    assert redact_sensitive_text(value) == value


def test_query_token_redaction_preserves_unrelated_query_parameters() -> None:
    """Redacting one credential must not destroy neighbouring metadata."""
    value = (
        "/callback?"
        "token=query-secret-value"
        "&state=safe-state"
    )

    redacted = redact_sensitive_text(
        value
    )

    assert "query-secret-value" not in redacted
    assert (
        f"token={REDACTION_MARKER}"
        in redacted
    )
    assert "state=safe-state" in redacted
