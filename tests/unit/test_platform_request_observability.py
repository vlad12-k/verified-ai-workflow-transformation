"""Tests for HTTP observability context propagation."""

import asyncio
from io import StringIO
from uuid import UUID

import pytest
from starlette.types import (
    Message,
    Receive,
    Scope,
    Send,
)

from vait.platform.api.middleware import (
    RequestContextMiddleware,
)
from vait.platform.observability import (
    configure_platform_logging,
    current_log_context,
)


def _http_scope(
    *,
    correlation_id: str,
) -> Scope:
    """Build a minimal HTTP ASGI scope for middleware tests."""
    return {
        "type": "http",
        "asgi": {
            "version": "3.0",
            "spec_version": "2.3",
        },
        "http_version": "1.1",
        "server": (
            "testserver",
            80,
        ),
        "client": (
            "127.0.0.1",
            12345,
        ),
        "scheme": "http",
        "method": "GET",
        "root_path": "",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": [
            (
                b"x-correlation-id",
                correlation_id.encode("ascii"),
            )
        ],
        "state": {},
    }


async def _receive() -> Message:
    """Provide one empty HTTP request message."""
    return {
        "type": "http.request",
        "body": b"",
        "more_body": False,
    }


def test_http_context_is_visible_downstream_and_reset_after_request() -> None:
    """Request metadata must be scoped to the active HTTP request."""
    observed: dict[str, str] = {}
    sent: list[Message] = []
    log_stream = StringIO()

    configure_platform_logging(
        stream=log_stream,
    )

    async def downstream(
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        del scope
        del receive

        observed.update(
            current_log_context()
        )

        await send(
            {
                "type": "http.response.start",
                "status": 204,
                "headers": [],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b"",
            }
        )

    async def capture_send(
        message: Message,
    ) -> None:
        sent.append(message)

    middleware = RequestContextMiddleware(
        downstream
    )

    asyncio.run(
        middleware(
            _http_scope(
                correlation_id="correlation-001",
            ),
            _receive,
            capture_send,
        )
    )

    request_id = observed["request_id"]

    assert str(UUID(request_id)) == request_id
    assert (
        observed["correlation_id"]
        == "correlation-001"
    )

    assert current_log_context() == {}
    assert len(sent) == 2

    serialized = log_stream.getvalue()

    assert (
        '"event":"http_request_completed"'
        in serialized
    )
    assert (
        '"correlation_id":"correlation-001"'
        in serialized
    )
    assert (
        f'"request_id":"{request_id}"'
        in serialized
    )


def test_http_context_is_reset_when_downstream_raises() -> None:
    """Failed requests must not leak correlation state."""
    observed: dict[str, str] = {}
    log_stream = StringIO()

    configure_platform_logging(
        stream=log_stream,
    )

    async def failing_downstream(
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        del scope
        del receive
        del send

        observed.update(
            current_log_context()
        )

        raise RuntimeError(
            "downstream failure"
        )

    async def discard_send(
        message: Message,
    ) -> None:
        del message

    middleware = RequestContextMiddleware(
        failing_downstream
    )

    with pytest.raises(
        RuntimeError,
        match="downstream failure",
    ):
        asyncio.run(
            middleware(
                _http_scope(
                    correlation_id="correlation-error",
                ),
                _receive,
                discard_send,
            )
        )

    assert (
        observed["correlation_id"]
        == "correlation-error"
    )
    assert current_log_context() == {}

    serialized = log_stream.getvalue()

    assert (
        '"event":"http_request_failed"'
        in serialized
    )
    assert (
        '"correlation_id":"correlation-error"'
        in serialized
    )
    assert (
        '"exception_type":"RuntimeError"'
        in serialized
    )
    assert "downstream failure" not in serialized
    assert "Traceback" not in serialized
