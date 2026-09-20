"""Tests for FastAPI OpenTelemetry tracing integration."""

import json
from io import StringIO
from typing import cast

from fastapi.testclient import TestClient

from vait.platform import PlatformSettings
from vait.platform.api import create_app
from vait.platform.observability import (
    configure_platform_logging,
    current_log_context,
)


def test_http_request_log_contains_active_trace_identifiers() -> None:
    """FastAPI requests must correlate structured logs with OTel spans."""
    stream = StringIO()

    configure_platform_logging(
        stream=stream,
    )

    app = create_app(
        PlatformSettings(
            environment="test",
            service_name="vait-test-platform",
        )
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/health/live",
            headers={
                "X-Correlation-ID": "trace-correlation-001",
            },
        )

    assert response.status_code == 200
    assert current_log_context() == {}

    payloads: list[dict[str, object]] = []

    for line in stream.getvalue().splitlines():
        if not line:
            continue

        value = json.loads(
            line
        )

        if isinstance(value, dict):
            payloads.append(
                cast(
                    dict[str, object],
                    value,
                )
            )

    completed = [
        payload
        for payload in payloads
        if payload.get("event")
        == "http_request_completed"
    ]

    assert len(completed) == 1

    payload = completed[0]

    assert (
        payload["correlation_id"]
        == "trace-correlation-001"
    )
    assert (
        payload["request_id"]
        == response.headers["X-Request-ID"]
    )

    trace_id = payload["trace_id"]
    span_id = payload["span_id"]

    assert isinstance(trace_id, str)
    assert isinstance(span_id, str)

    assert len(trace_id) == 32
    assert len(span_id) == 16

    assert int(trace_id, 16) > 0
    assert int(span_id, 16) > 0
