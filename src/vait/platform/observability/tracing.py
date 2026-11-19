"""OpenTelemetry tracing primitives for the VAIT platform."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.trace import (
    Span,
    Status,
    StatusCode,
    Tracer,
)

from vait.platform.observability.context import (
    bind_log_context,
    reset_log_context,
)
from vait.platform.settings import PlatformSettings

_TRACER_NAME = "vait.platform"
_TRACER_VERSION = "0.1.0"


@dataclass(frozen=True)
class TracingRuntime:
    """Own the platform tracer provider and tracer lifecycle."""

    provider: TracerProvider
    tracer: Tracer

    def shutdown(self) -> None:
        """Flush and release tracing resources."""
        self.provider.shutdown()


def create_tracing_runtime(
    settings: PlatformSettings,
    *,
    span_exporter: SpanExporter | None = None,
) -> TracingRuntime:
    """Create an isolated OpenTelemetry runtime for the platform."""
    resource = Resource.create(
        {
            "service.name": settings.service_name,
            "deployment.environment.name": settings.environment,
        }
    )

    provider = TracerProvider(
        resource=resource
    )

    if span_exporter is not None:
        provider.add_span_processor(
            SimpleSpanProcessor(
                span_exporter
            )
        )

    tracer = provider.get_tracer(
        _TRACER_NAME,
        _TRACER_VERSION,
    )

    return TracingRuntime(
        provider=provider,
        tracer=tracer,
    )


@contextmanager
def traced_span(
    runtime: TracingRuntime,
    name: str,
) -> Iterator[Span]:
    """Create a span and bind its identifiers to structured log context."""
    if not name:
        raise ValueError(
            "span name must not be empty"
        )

    with runtime.tracer.start_as_current_span(
        name,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        span_context = span.get_span_context()

        token = bind_log_context(
            trace_id=f"{span_context.trace_id:032x}",
            span_id=f"{span_context.span_id:016x}",
        )

        try:
            yield span
        except Exception:
            span.set_status(
                Status(
                    StatusCode.ERROR,
                    "operation failed",
                )
            )
            raise
        finally:
            reset_log_context(
                token
            )
