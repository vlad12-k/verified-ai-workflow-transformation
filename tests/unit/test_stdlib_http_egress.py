"""Tests for egress enforcement in the stdlib provider transport."""

import json
from dataclasses import dataclass, field
from http.server import (
    BaseHTTPRequestHandler,
    ThreadingHTTPServer,
)
from threading import Thread

import pytest
from pydantic import JsonValue

from vait.inference.providers.stdlib_http import (
    ProviderRedirectDenied,
    StdlibJsonTransport,
)
from vait.platform.security.egress import (
    ProviderEgressDenied,
    ProviderEgressPolicy,
)


@dataclass
class RedirectRecord:
    """Capture request evidence across one redirect sequence."""

    post_path: str | None = None
    get_path: str | None = None
    redirected_authorization: str | None = None


@dataclass
class RecordingValidator:
    """Record validation calls while delegating to the real egress policy."""

    delegate: ProviderEgressPolicy
    calls: list[str] = field(
        default_factory=list
    )

    def validate_url(
        self,
        url: str,
    ) -> None:
        self.calls.append(
            url
        )
        self.delegate.validate_url(
            url
        )


@dataclass
class RejectingValidator:
    """Reject every outbound URL before transport network access."""

    calls: list[str] = field(
        default_factory=list
    )

    def validate_url(
        self,
        url: str,
    ) -> None:
        self.calls.append(
            url
        )

        raise ProviderEgressDenied(
            "origin_not_allowed"
        )


def _build_redirect_handler(
    *,
    record: RedirectRecord,
    location: str,
) -> type[BaseHTTPRequestHandler]:
    """Build a server that redirects POST and serves redirected GET."""

    class Handler(BaseHTTPRequestHandler):
        """Serve one deterministic redirect flow."""

        def do_POST(
            self,
        ) -> None:
            record.post_path = self.path

            self.send_response(
                302
            )
            self.send_header(
                "Location",
                location,
            )
            self.send_header(
                "Content-Length",
                "0",
            )
            self.end_headers()

        def do_GET(
            self,
        ) -> None:
            record.get_path = self.path
            record.redirected_authorization = (
                self.headers.get(
                    "Authorization"
                )
            )

            payload = json.dumps(
                {
                    "redirected": True,
                }
            ).encode(
                "utf-8"
            )

            self.send_response(
                200
            )
            self.send_header(
                "Content-Type",
                "application/json",
            )
            self.send_header(
                "Content-Length",
                str(
                    len(
                        payload
                    )
                ),
            )
            self.end_headers()

            self.wfile.write(
                payload
            )

        def log_message(
            self,
            format: str,
            *args: object,
        ) -> None:
            """Suppress localhost HTTP server logs."""

    return Handler


def _start_redirect_server(
    *,
    record: RedirectRecord,
    location: str,
) -> tuple[
    ThreadingHTTPServer,
    Thread,
]:
    """Start one ephemeral localhost redirect server."""
    server = ThreadingHTTPServer(
        (
            "127.0.0.1",
            0,
        ),
        _build_redirect_handler(
            record=record,
            location=location,
        ),
    )

    thread = Thread(
        target=server.serve_forever,
        daemon=True,
    )
    thread.start()

    return server, thread


def _server_origin(
    server: ThreadingHTTPServer,
) -> str:
    host = server.server_address[0]
    port = server.server_address[1]

    if not isinstance(
        host,
        str,
    ):
        raise TypeError(
            "Expected IPv4 localhost host string."
        )

    if not isinstance(
        port,
        int,
    ):
        raise TypeError(
            "Expected localhost TCP port."
        )

    return (
        f"http://{host}:{port}"
    )


def _payload() -> dict[
    str,
    JsonValue,
]:
    return {
        "model": "portable-model",
    }


def test_transport_validates_initial_url_before_network_access() -> None:
    """A rejected initial destination must never reach urllib networking."""
    validator = RejectingValidator()

    transport = StdlibJsonTransport(
        url_validator=validator,
    )

    blocked_url = (
        "http://127.0.0.1:1/v1"
    )

    with pytest.raises(
        ProviderEgressDenied,
    ) as exc_info:
        transport.post_json(
            url=blocked_url,
            headers={},
            payload=_payload(),
            timeout_seconds=0.1,
        )

    assert (
        exc_info.value.reason
        == "origin_not_allowed"
    )
    assert validator.calls == [
        blocked_url,
    ]


def test_same_origin_redirect_is_validated_before_following() -> None:
    """A same-origin redirect may proceed only after policy validation."""
    record = RedirectRecord()

    server, thread = _start_redirect_server(
        record=record,
        location="/redirected",
    )

    try:
        origin = _server_origin(
            server
        )

        policy = ProviderEgressPolicy(
            allowed_origins=frozenset(
                {
                    origin,
                }
            ),
            allow_non_global_addresses=True,
        )

        validator = RecordingValidator(
            delegate=policy,
        )

        transport = StdlibJsonTransport(
            url_validator=validator,
        )

        initial_url = (
            f"{origin}/v1/chat/completions"
        )
        redirected_url = (
            f"{origin}/redirected"
        )

        response = transport.post_json(
            url=initial_url,
            headers={
                "Authorization": (
                    "Bearer local-redirect-test"
                ),
            },
            payload=_payload(),
            timeout_seconds=5.0,
        )

    finally:
        server.shutdown()
        server.server_close()
        thread.join(
            timeout=5.0
        )

    assert response.status_code == 200
    assert response.body == {
        "redirected": True,
    }

    assert validator.calls == [
        initial_url,
        redirected_url,
    ]

    assert (
        record.post_path
        == "/v1/chat/completions"
    )
    assert (
        record.get_path
        == "/redirected"
    )
    assert (
        record.redirected_authorization
        == "Bearer local-redirect-test"
    )


def test_cross_origin_redirect_is_denied_even_when_target_is_allowlisted(
    ) -> None:
    """Credentials must never follow a redirect to another allowed origin."""
    record = RedirectRecord()

    target_origin = (
        "http://127.0.0.1:1"
    )

    server, thread = _start_redirect_server(
        record=record,
        location=(
            f"{target_origin}/private"
        ),
    )

    try:
        origin = _server_origin(
            server
        )

        policy = ProviderEgressPolicy(
            allowed_origins=frozenset(
                {
                    origin,
                    target_origin,
                }
            ),
            allow_non_global_addresses=True,
        )

        validator = RecordingValidator(
            delegate=policy,
        )

        transport = StdlibJsonTransport(
            url_validator=validator,
        )

        initial_url = (
            f"{origin}/v1/chat/completions"
        )
        target_url = (
            f"{target_origin}/private"
        )

        with pytest.raises(
            ProviderRedirectDenied,
            match=(
                "cross-origin provider "
                "redirect denied"
            ),
        ) as exc_info:
            transport.post_json(
                url=initial_url,
                headers={
                    "Authorization": (
                        "Bearer must-not-leave-origin"
                    ),
                },
                payload=_payload(),
                timeout_seconds=5.0,
            )

    finally:
        server.shutdown()
        server.server_close()
        thread.join(
            timeout=5.0
        )

    assert validator.calls == [
        initial_url,
        target_url,
    ]

    assert target_url not in str(
        exc_info.value
    )
    assert (
        "must-not-leave-origin"
        not in str(
            exc_info.value
        )
    )

    assert (
        record.post_path
        == "/v1/chat/completions"
    )
    assert record.get_path is None
    assert (
        record.redirected_authorization
        is None
    )


def test_redirect_to_disallowed_origin_is_rejected_by_egress_policy(
    ) -> None:
    """Redirect targets must cross the same egress policy as initial URLs."""
    record = RedirectRecord()

    target_url = (
        "http://127.0.0.1:1/private"
    )

    server, thread = _start_redirect_server(
        record=record,
        location=target_url,
    )

    try:
        origin = _server_origin(
            server
        )

        policy = ProviderEgressPolicy(
            allowed_origins=frozenset(
                {
                    origin,
                }
            ),
            allow_non_global_addresses=True,
        )

        validator = RecordingValidator(
            delegate=policy,
        )

        transport = StdlibJsonTransport(
            url_validator=validator,
        )

        initial_url = (
            f"{origin}/v1/chat/completions"
        )

        with pytest.raises(
            ProviderEgressDenied,
        ) as exc_info:
            transport.post_json(
                url=initial_url,
                headers={},
                payload=_payload(),
                timeout_seconds=5.0,
            )

    finally:
        server.shutdown()
        server.server_close()
        thread.join(
            timeout=5.0
        )

    assert (
        exc_info.value.reason
        == "origin_not_allowed"
    )

    assert validator.calls == [
        initial_url,
        target_url,
    ]

    assert record.get_path is None


def test_empty_http_response_body_normalises_to_empty_object() -> None:
    """A successful empty HTTP body must normalise deterministically."""

    class EmptyResponseHandler(
        BaseHTTPRequestHandler
    ):
        """Return a successful response with no body."""

        def do_POST(
            self,
        ) -> None:
            self.send_response(
                204
            )
            self.send_header(
                "Content-Length",
                "0",
            )
            self.end_headers()

        def log_message(
            self,
            format: str,
            *args: object,
        ) -> None:
            """Suppress localhost HTTP server logs."""

    server = ThreadingHTTPServer(
        (
            "127.0.0.1",
            0,
        ),
        EmptyResponseHandler,
    )

    thread = Thread(
        target=server.serve_forever,
        daemon=True,
    )
    thread.start()

    try:
        origin = _server_origin(
            server
        )

        policy = ProviderEgressPolicy(
            allowed_origins=frozenset(
                {
                    origin,
                }
            ),
            allow_non_global_addresses=True,
        )

        transport = StdlibJsonTransport(
            url_validator=policy,
        )

        response = transport.post_json(
            url=f"{origin}/empty",
            headers={},
            payload=_payload(),
            timeout_seconds=5.0,
        )

    finally:
        server.shutdown()
        server.server_close()
        thread.join(
            timeout=5.0
        )

    assert response.status_code == 204
    assert response.body == {}
