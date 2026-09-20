"""Tests for the M5-H OpenTelemetry tracing foundation."""

import json
from io import StringIO

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from vait.platform.observability import (
    TracingRuntime,
    bind_log_context,
    configure_platform_logging,
    create_tracing_runtime,
    current_log_context,
    reset_log_context,
    traced_span,
)
from vait.platform.settings import PlatformSettings


def _runtime(
    exporter: InMemorySpanExporter,
) -> TracingRuntime:
    """Create one isolated test tracing runtime."""
    return create_tracing_runtime(
        PlatformSettings(
            environment="test",
            service_name="vait-test-platform",
        ),
        span_exporter=exporter,
    )


def test_tracing_runtime_uses_platform_resource_identity() -> None:
    """Tracing resources must identify the configured service."""
    exporter = InMemorySpanExporter()
    runtime = _runtime(
        exporter
    )

    try:
        attributes = runtime.provider.resource.attributes

        assert (
            attributes["service.name"]
            == "vait-test-platform"
        )
        assert (
            attributes["deployment.environment.name"]
            == "test"
        )
    finally:
        runtime.shutdown()


def test_span_binds_trace_and_span_ids_to_log_context() -> None:
    """Active OTel identifiers must enter the existing log context."""
    exporter = InMemorySpanExporter()
    runtime = _runtime(
        exporter
    )

    try:
        assert current_log_context() == {}

        with traced_span(
            runtime,
            "test.operation",
        ):
            context = current_log_context()

            trace_id = context["trace_id"]
            span_id = context["span_id"]

            assert len(trace_id) == 32
            assert len(span_id) == 16
            assert int(trace_id, 16) > 0
            assert int(span_id, 16) > 0

        assert current_log_context() == {}

        runtime.provider.force_flush()

        spans = exporter.get_finished_spans()

        assert len(spans) == 1
        assert spans[0].name == "test.operation"
    finally:
        runtime.shutdown()


def test_nested_spans_share_trace_and_restore_parent_span() -> None:
    """Nested spans must preserve one trace and restore parent context."""
    exporter = InMemorySpanExporter()
    runtime = _runtime(
        exporter
    )

    try:
        with traced_span(
            runtime,
            "outer",
        ):
            outer_context = current_log_context()

            with traced_span(
                runtime,
                "inner",
            ):
                inner_context = current_log_context()

                assert (
                    inner_context["trace_id"]
                    == outer_context["trace_id"]
                )
                assert (
                    inner_context["span_id"]
                    != outer_context["span_id"]
                )

            assert (
                current_log_context()
                == outer_context
            )

        assert current_log_context() == {}
    finally:
        runtime.shutdown()


def test_span_preserves_existing_request_correlation_context() -> None:
    """Tracing must augment rather than replace request correlation."""
    exporter = InMemorySpanExporter()
    runtime = _runtime(
        exporter
    )

    outer_token = bind_log_context(
        request_id="request-001",
        correlation_id="correlation-001",
    )

    try:
        with traced_span(
            runtime,
            "request.operation",
        ):
            context = current_log_context()

            assert (
                context["request_id"]
                == "request-001"
            )
            assert (
                context["correlation_id"]
                == "correlation-001"
            )
            assert "trace_id" in context
            assert "span_id" in context

        assert current_log_context() == {
            "request_id": "request-001",
            "correlation_id": "correlation-001",
        }
    finally:
        reset_log_context(
            outer_token
        )
        runtime.shutdown()

    assert current_log_context() == {}


def test_structured_logs_include_active_trace_identifiers() -> None:
    """Existing JSON logs must inherit OTel identifiers automatically."""
    exporter = InMemorySpanExporter()
    runtime = _runtime(
        exporter
    )

    stream = StringIO()
    logger = configure_platform_logging(
        stream=stream,
    )

    try:
        with traced_span(
            runtime,
            "logged.operation",
        ):
            context = current_log_context()

            logger.info(
                "operation_inside_span"
            )

        lines = [
            line
            for line in stream.getvalue().splitlines()
            if line
        ]

        assert len(lines) == 1

        payload = json.loads(
            lines[0]
        )

        assert (
            payload["event"]
            == "operation_inside_span"
        )
        assert (
            payload["trace_id"]
            == context["trace_id"]
        )
        assert (
            payload["span_id"]
            == context["span_id"]
        )
    finally:
        runtime.shutdown()


def test_span_context_is_reset_when_operation_raises() -> None:
    """Exceptions must not leak trace identifiers into later work."""
    exporter = InMemorySpanExporter()
    runtime = _runtime(
        exporter
    )

    try:
        with (
            pytest.raises(
                RuntimeError,
                match="test failure",
            ),
            traced_span(
                runtime,
                "failing.operation",
            ),
        ):
            raise RuntimeError(
                "test failure"
            )

        assert current_log_context() == {}
    finally:
        runtime.shutdown()


def test_empty_span_name_is_rejected() -> None:
    """Every emitted platform span must have an explicit operation name."""
    exporter = InMemorySpanExporter()
    runtime = _runtime(
        exporter
    )

    try:
        with (
            pytest.raises(
                ValueError,
                match="span name must not be empty",
            ),
            traced_span(
                runtime,
                "",
            ),
        ):
            pass
    finally:
        runtime.shutdown()
