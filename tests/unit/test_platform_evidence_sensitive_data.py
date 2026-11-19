"""Tests for M6-B durable evidence sensitive-data protection."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import JsonValue, ValidationError

from vait.platform.registry import EvidenceRecord
from vait.platform.registry.sensitive_data import (
    contains_prohibited_evidence_secret,
    validate_evidence_payload,
)


def _evidence(
    payload: dict[str, object],
) -> EvidenceRecord:
    """Create one verification evidence record for security tests."""
    return EvidenceRecord(
        evidence_id=uuid4(),
        experiment_id=uuid4(),
        kind="verification",
        payload=payload,  # type: ignore[arg-type]
        created_at=datetime.now(UTC),
    )


def test_benign_evidence_is_preserved_unchanged() -> None:
    """Valid scientific evidence must not be silently rewritten."""
    payload = {
        "decision": "EXACT",
        "latency_ms": 42,
        "token_count": 154,
        "token_budget": 1000,
        "password_policy": "provider-managed",
        "secret_rotation": "required",
        "api_key_rotation": "scheduled",
        "nested": {
            "scores": [
                0.91,
                0.93,
            ],
        },
    }

    evidence = _evidence(
        payload
    )

    assert evidence.payload == payload


@pytest.mark.parametrize(
    "field_name",
    [
        "authorization",
        "Authorization",
        "proxy-authorization",
        "api_key",
        "api-key",
        "apikey",
        "token",
        "access_token",
        "refresh-token",
        "auth.token",
        "password",
        "passwd",
        "secret",
        "client_secret",
        "database_url",
        "connection-string",
    ],
)
def test_prohibited_secret_field_names_are_rejected(
    field_name: str,
) -> None:
    """Credential-bearing keys must fail closed regardless of separators."""
    with pytest.raises(
        ValidationError,
        match="prohibited secret material",
    ):
        _evidence(
            {
                field_name: "credential-value",
            }
        )


def test_nested_secret_field_is_rejected() -> None:
    """Nested mappings must not bypass the evidence guard."""
    with pytest.raises(
        ValidationError,
        match="prohibited secret material",
    ):
        _evidence(
            {
                "provider": {
                    "metadata": {
                        "api_key": "nested-secret",
                    },
                },
            }
        )


def test_secret_field_inside_list_is_rejected() -> None:
    """Lists containing nested credential mappings must fail closed."""
    with pytest.raises(
        ValidationError,
        match="prohibited secret material",
    ):
        _evidence(
            {
                "requests": [
                    {
                        "headers": {
                            "Authorization": (
                                "Bearer nested-secret"
                            ),
                        },
                    },
                ],
            }
        )


@pytest.mark.parametrize(
    "value",
    [
        "Authorization: Bearer provider-secret",
        "api_key=provider-secret",
        "token=query-secret",
        "Basic dXNlcjpwYXNzd29yZA==",
        "postgresql://user:password@db.internal/vait",
    ],
)
def test_credential_material_hidden_in_free_text_is_rejected(
    value: str,
) -> None:
    """Unambiguous credential text must fail even under a benign field."""
    with pytest.raises(
        ValidationError,
        match="prohibited secret material",
    ):
        _evidence(
            {
                "diagnostic": value,
            }
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "token_count",
        "token_budget",
        "tokenizer",
        "password_policy",
        "secret_rotation",
        "api_key_rotation",
        "access_token_budget",
        "connection_string_policy",
    ],
)
def test_related_non_secret_field_names_are_not_rejected(
    field_name: str,
) -> None:
    """Operational metadata must not be blocked by substring matching."""
    evidence = _evidence(
        {
            field_name: "safe",
        }
    )

    assert evidence.payload[
        field_name
    ] == "safe"


def test_validation_returns_original_payload_object() -> None:
    """The durable evidence guard must validate, not mutate, evidence."""
    payload: dict[str, JsonValue] = {
        "decision": "BOUNDED",
        "metrics": {
            "accuracy": 0.98,
        },
    }

    validated = validate_evidence_payload(
        payload
    )

    assert validated is payload


def test_detector_handles_json_scalar_values() -> None:
    """Ordinary JSON scalar values must remain admissible."""
    values = [
        None,
        True,
        False,
        1,
        1.5,
    ]

    for value in values:
        assert (
            contains_prohibited_evidence_secret(
                value
            )
            is False
        )


def test_rejection_message_does_not_echo_secret_value() -> None:
    """Validation failure text must not reproduce credential material."""
    secret = "do-not-echo-this-secret"

    with pytest.raises(
        ValidationError,
    ) as exc_info:
        _evidence(
            {
                "api_key": secret,
            }
        )

    message = str(
        exc_info.value
    )

    assert (
        "prohibited secret material"
        in message
    )
    assert secret not in message
