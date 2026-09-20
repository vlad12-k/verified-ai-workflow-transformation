"""HTTP request context middleware for the VAIT platform."""

import logging
import re
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from vait.platform.observability import (
    TracingRuntime,
    bind_log_context,
    reset_log_context,
    traced_span,
)

REQUEST_ID_HEADER = "X-Request-ID"
CORRELATION_ID_HEADER = "X-Correlation-ID"

_MAX_CORRELATION_ID_LENGTH = 128
_SAFE_CORRELATION_ID = re.compile(r"^[A-Za-z0-9._:-]+$")

_LOGGER = logging.getLogger("vait")


def _validated_correlation_id(value: str | None) -> str | None:
    """Accept only bounded, log-safe correlation identifiers."""
    if value is None:
        return None

    if not 1 <= len(value) <= _MAX_CORRELATION_ID_LENGTH:
        return None

    if _SAFE_CORRELATION_ID.fullmatch(value) is None:
        return None

    return value


class RequestContextMiddleware:
    """Attach safe request and correlation identifiers to HTTP traffic."""

    def __init__(
        self,
        app: ASGIApp,
        tracing_runtime: TracingRuntime | None = None,
    ) -> None:
        self.app = app
        self._tracing_runtime = tracing_runtime

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid4())

        request_headers = Headers(scope=scope)
        correlation_id = (
            _validated_correlation_id(
                request_headers.get(CORRELATION_ID_HEADER)
            )
            or request_id
        )

        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        state["correlation_id"] = correlation_id

        async def send_with_context(message: Message) -> None:
            if message["type"] == "http.response.start":
                response_headers = MutableHeaders(scope=message)
                response_headers[REQUEST_ID_HEADER] = request_id
                response_headers[CORRELATION_ID_HEADER] = correlation_id

            await send(message)

        token = bind_log_context(
            request_id=request_id,
            correlation_id=correlation_id,
        )

        async def call_application() -> None:
            try:
                await self.app(
                    scope,
                    receive,
                    send_with_context,
                )
            except Exception:
                _LOGGER.exception(
                    "http_request_failed"
                )
                raise
            else:
                _LOGGER.info(
                    "http_request_completed"
                )

        try:
            if self._tracing_runtime is None:
                await call_application()
            else:
                with traced_span(
                    self._tracing_runtime,
                    "http.request",
                ):
                    await call_application()
        finally:
            reset_log_context(
                token
            )
