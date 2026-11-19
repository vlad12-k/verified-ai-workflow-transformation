"""Tests for M5-H structured platform logging."""

import json
import logging
from io import StringIO
from typing import cast

from vait.platform.observability import (
    bind_log_context,
    configure_platform_logging,
    reset_log_context,
)


def _payload(
    stream: StringIO,
) -> dict[str, object]:
    """Parse exactly one JSON log line."""
    lines = [
        line
        for line in stream.getvalue().splitlines()
        if line
    ]

    assert len(lines) == 1

    value = json.loads(
        lines[0]
    )

    assert isinstance(
        value,
        dict,
    )

    return cast(
        dict[str, object],
        value,
    )


def test_structured_log_contains_fixed_base_fields() -> None:
    """Platform logs must expose a stable machine-readable envelope."""
    stream = StringIO()
    logger = configure_platform_logging(
        stream=stream,
    )

    logger.info(
        "platform_started"
    )

    payload = _payload(
        stream
    )

    assert payload["level"] == "INFO"
    assert payload["logger"] == "vait"
    assert payload["event"] == "platform_started"

    timestamp = payload["timestamp"]

    assert isinstance(
        timestamp,
        str,
    )
    assert timestamp.endswith(
        "Z"
    )

    assert set(payload) == {
        "timestamp",
        "level",
        "logger",
        "event",
    }


def test_structured_log_includes_active_correlation_context() -> None:
    """Bound request and job identifiers must be injected automatically."""
    stream = StringIO()
    logger = configure_platform_logging(
        stream=stream,
    )

    token = bind_log_context(
        request_id="request-001",
        correlation_id="correlation-001",
        experiment_id="experiment-001",
        job_id="job-001",
        worker_id="worker-001",
    )

    try:
        logger.info(
            "job_claimed"
        )
    finally:
        reset_log_context(
            token
        )

    payload = _payload(
        stream
    )

    assert payload["event"] == "job_claimed"
    assert payload["request_id"] == "request-001"
    assert payload["correlation_id"] == "correlation-001"
    assert payload["experiment_id"] == "experiment-001"
    assert payload["job_id"] == "job-001"
    assert payload["worker_id"] == "worker-001"


def test_arbitrary_logging_extra_fields_are_not_serialized() -> None:
    """Untrusted or sensitive extra fields must not enter JSON logs."""
    stream = StringIO()
    logger = configure_platform_logging(
        stream=stream,
    )

    logger.info(
        "safe_event",
        extra={
            "customer_email": "person@example.test",
            "password": "super-secret",
            "api_key": "secret-key",
        },
    )

    serialized = stream.getvalue()
    payload = _payload(
        stream
    )

    assert payload["event"] == "safe_event"

    assert "customer_email" not in payload
    assert "password" not in payload
    assert "api_key" not in payload

    assert "person@example.test" not in serialized
    assert "super-secret" not in serialized
    assert "secret-key" not in serialized


def test_exception_logging_exposes_type_without_exception_message() -> None:
    """Exception logs must not serialize exception messages or tracebacks."""
    stream = StringIO()
    logger = configure_platform_logging(
        stream=stream,
    )

    try:
        raise RuntimeError(
            "sensitive-provider-response"
        )
    except RuntimeError:
        logger.exception(
            "worker_execution_failed"
        )

    serialized = stream.getvalue()
    payload = _payload(
        stream
    )

    assert (
        payload["event"]
        == "worker_execution_failed"
    )
    assert (
        payload["exception_type"]
        == "RuntimeError"
    )

    assert (
        "sensitive-provider-response"
        not in serialized
    )
    assert "Traceback" not in serialized


def test_reconfiguration_replaces_handler_instead_of_duplicating_logs() -> None:
    """Repeated configuration must not multiply platform log lines."""
    first_stream = StringIO()
    second_stream = StringIO()

    logger = configure_platform_logging(
        stream=first_stream,
    )

    logger = configure_platform_logging(
        stream=second_stream,
    )

    logger.info(
        "configured_once"
    )

    assert first_stream.getvalue() == ""
    assert len(
        logger.handlers
    ) == 1

    payload = _payload(
        second_stream
    )

    assert payload["event"] == "configured_once"
    assert logger.level == logging.INFO
    assert logger.propagate is False


def test_formatted_secret_is_redacted_from_log_event() -> None:
    """Logging interpolation must not bypass telemetry redaction."""
    stream = StringIO()
    logger = configure_platform_logging(
        stream=stream,
    )

    secret = "provider-api-secret-123456"

    logger.info(
        "provider api_key=%s",
        secret,
    )

    serialized = stream.getvalue()
    payload = _payload(
        stream
    )

    assert secret not in serialized
    assert (
        payload["event"]
        == "provider api_key=[REDACTED]"
    )


def test_bearer_credential_is_redacted_from_log_event() -> None:
    """Bearer credentials embedded in event text must be removed."""
    stream = StringIO()
    logger = configure_platform_logging(
        stream=stream,
    )

    secret = "bearer-secret-token-123456"

    logger.info(
        "outbound Authorization: Bearer %s",
        secret,
    )

    serialized = stream.getvalue()
    payload = _payload(
        stream
    )

    assert secret not in serialized
    assert "[REDACTED]" in str(
        payload["event"]
    )


def test_credential_uri_is_redacted_from_log_event() -> None:
    """Connection credentials in URI userinfo must not enter logs."""
    stream = StringIO()
    logger = configure_platform_logging(
        stream=stream,
    )

    password = "database-password-123"

    logger.info(
        "database target postgresql://vait:%s@db.internal/vait",
        password,
    )

    serialized = stream.getvalue()
    payload = _payload(
        stream
    )

    assert password not in serialized
    assert "[REDACTED]" in str(
        payload["event"]
    )
    assert "db.internal" in str(
        payload["event"]
    )


def test_exception_log_event_arguments_are_redacted() -> None:
    """Safe exception handling must also redact formatted event arguments."""
    stream = StringIO()
    logger = configure_platform_logging(
        stream=stream,
    )

    secret = "provider-secret-in-log-123"

    try:
        raise RuntimeError(
            "internal-sensitive-exception-message"
        )
    except RuntimeError:
        logger.exception(
            "provider_failed api_key=%s",
            secret,
        )

    serialized = stream.getvalue()
    payload = _payload(
        stream
    )

    assert secret not in serialized
    assert (
        "internal-sensitive-exception-message"
        not in serialized
    )
    assert (
        payload["event"]
        == "provider_failed api_key=[REDACTED]"
    )
    assert (
        payload["exception_type"]
        == "RuntimeError"
    )
