"""Tests for egress enforcement in the stdlib provider transport."""

import json
import shutil
import socket
import ssl
import subprocess
from dataclasses import dataclass, field
from http.server import (
    BaseHTTPRequestHandler,
    ThreadingHTTPServer,
)
from pathlib import Path
from threading import Event, Thread
from typing import cast
from urllib.error import URLError

import pytest
from pydantic import JsonValue

from vait.inference.providers import stdlib_http
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

    def resolve_validated_addresses(
        self,
        url: str,
    ) -> tuple[str, ...]:
        self.calls.append(url)
        return self.delegate.resolve_validated_addresses(url)


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

    def resolve_validated_addresses(
        self,
        url: str,
    ) -> tuple[str, ...]:
        self.calls.append(url)
        raise ProviderEgressDenied("origin_not_allowed")


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


def _generate_test_certificate(tmp_path: Path) -> Path:
    """Generate a short-lived local HTTPS identity outside the repository."""
    openssl = shutil.which("openssl")
    if openssl is None:
        raise RuntimeError("OpenSSL is required for local HTTPS tests")

    certificate_path = tmp_path / "server-cert.pem"
    private_key_path = tmp_path / "server-key.pem"
    subprocess.run(
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-sha256",
            "-days",
            "1",
            "-keyout",
            str(private_key_path),
            "-out",
            str(certificate_path),
            "-subj",
            "/CN=provider.test",
            "-addext",
            "subjectAltName=DNS:provider.test",
        ],
        check=True,
        capture_output=True,
        timeout=15,
    )
    return certificate_path


@dataclass
class PinnedRequestRecord:
    """Capture the request details seen by a local test endpoint."""

    host: str | None = None
    authorization: str | None = None
    body: bytes = b""
    reached: bool = False


def _start_https_recording_server(
    certificate_path: Path,
    record: PinnedRequestRecord,
    server_names: list[str | None],
) -> tuple[ThreadingHTTPServer, Thread]:
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(
        certfile=str(certificate_path),
        keyfile=str(certificate_path.with_name("server-key.pem")),
    )

    def record_server_name(
        tls_socket: ssl.SSLSocket | ssl.SSLObject,
        server_name: str | None,
        initial_context: ssl.SSLSocket,
    ) -> int | None:
        server_names.append(server_name)
        return None

    server_context.set_servername_callback(record_server_name)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        _recording_handler(record),
    )
    server.socket = server_context.wrap_socket(
        server.socket,
        server_side=True,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _recording_handler(
    record: PinnedRequestRecord,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            record.reached = True
            record.host = self.headers.get("Host")
            record.authorization = self.headers.get("Authorization")
            record.body = self.rfile.read(
                int(self.headers.get("Content-Length", "0"))
            )
            body = json.dumps({"ok": True}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def _start_recording_server(
    record: PinnedRequestRecord,
) -> tuple[ThreadingHTTPServer, Thread]:
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        _recording_handler(record),
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _trust_test_certificate(
    monkeypatch: pytest.MonkeyPatch,
    certificate_path: Path,
) -> None:
    create_default_context = ssl.create_default_context

    def create_trusting_context() -> ssl.SSLContext:
        return create_default_context(
            cadata=certificate_path.read_text(encoding="ascii")
        )

    monkeypatch.setattr(ssl, "create_default_context", create_trusting_context)


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
    assert stdlib_http._REQUEST_DEADLINE.get() is None
    assert stdlib_http._PINNED_ADDRESSES.get() is None


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

    # The initial URL is preflight-validated and resolved again for its pinned
    # connection; redirect validation and pinning are separate gates too.
    assert validator.calls == [
        initial_url,
        initial_url,
        redirected_url,
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


@dataclass
class FixedResolver:
    """Return a fixed policy-time DNS answer."""

    addresses: tuple[str, ...]

    def resolve(
        self,
        *,
        host: str,
        port: int,
    ) -> tuple[str, ...]:
        return self.addresses


def test_dns_rebinding_cannot_change_validated_destination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connection-time DNS must not escape the policy-approved address set."""
    request_reached_loopback = False

    class RebindingTargetHandler(
        BaseHTTPRequestHandler
    ):
        """Record whether a rebound request reaches the loopback server."""

        def do_POST(
            self,
        ) -> None:
            nonlocal request_reached_loopback

            request_reached_loopback = True

            payload = json.dumps(
                {
                    "reached": True,
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

    server = ThreadingHTTPServer(
        (
            "127.0.0.1",
            0,
        ),
        RebindingTargetHandler,
    )

    thread = Thread(
        target=server.serve_forever,
        daemon=True,
    )
    thread.start()

    try:
        port = server.server_address[1]

        if not isinstance(
            port,
            int,
        ):
            raise TypeError(
                "Expected localhost TCP port."
            )

        policy = ProviderEgressPolicy(
            allowed_origins=frozenset(
                {
                    f"http://rebinding.test:{port}",
                }
            ),
            resolver=FixedResolver(
                addresses=(
                    "8.8.8.8",
                )
            ),
        )

        transport = StdlibJsonTransport(
            url_validator=policy,
        )

        attempted_destinations: list[tuple[str, int]] = []

        original_create_connection = socket.create_connection

        def record_connection(
            address: tuple[str, int],
            timeout: float | None = None,
            source_address: tuple[str, int] | None = None,
        ) -> socket.socket:
            attempted_destinations.append(address)
            if address[0] == "rebinding.test":
                return original_create_connection(
                    ("127.0.0.1", address[1]),
                    timeout=timeout,
                    source_address=source_address,
                )
            raise OSError("controlled test connection failure")

        monkeypatch.setattr(socket, "create_connection", record_connection)

        with pytest.raises(OSError, match="controlled test connection failure"):
            transport.post_json(
                url=(
                    "http://rebinding.test:"
                    f"{port}/v1"
                ),
                headers={
                    "Authorization": (
                        "Bearer must-not-reach-rebound-host"
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

    assert request_reached_loopback is False
    assert attempted_destinations == [("8.8.8.8", port)]


def test_http_posts_to_approved_address_and_preserves_host_header() -> None:
    """HTTP uses the approved IP while sending the original hostname in Host."""
    record = PinnedRequestRecord()
    server, thread = _start_recording_server(record)

    try:
        port = server.server_address[1]
        policy = ProviderEgressPolicy(
            allowed_origins=frozenset({f"http://provider.test:{port}"}),
            allow_non_global_addresses=True,
            resolver=FixedResolver(("127.0.0.1",)),
        )
        response = StdlibJsonTransport(url_validator=policy).post_json(
            url=f"http://provider.test:{port}/v1",
            headers={"Authorization": "Bearer local-test"},
            payload=_payload(),
            timeout_seconds=3.0,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)

    assert response.status_code == 200
    assert response.body == {"ok": True}
    assert record.reached is True
    assert record.host == f"provider.test:{port}"
    assert record.authorization == "Bearer local-test"
    assert record.body == b'{"model":"portable-model"}'


def test_multiple_approved_addresses_fall_back_only_within_approved_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed approved IPv4 destination may fall back to another approved IP."""
    original_create_connection = socket.create_connection
    attempted: list[tuple[str, int]] = []

    def fail_first_approved_address(
        address: tuple[str, int],
        timeout: float | None = None,
        source_address: tuple[str, int] | None = None,
    ) -> socket.socket:
        attempted.append(address)
        if address[0] == "127.0.0.2":
            raise OSError("controlled first-address failure")
        return original_create_connection(
            address,
            timeout=timeout,
            source_address=source_address,
        )

    record = PinnedRequestRecord()
    server, thread = _start_recording_server(record)
    monkeypatch.setattr(socket, "create_connection", fail_first_approved_address)

    try:
        port = server.server_address[1]
        policy = ProviderEgressPolicy(
            allowed_origins=frozenset({f"http://provider.test:{port}"}),
            allow_non_global_addresses=True,
            resolver=FixedResolver(("127.0.0.2", "127.0.0.1")),
        )
        response = StdlibJsonTransport(url_validator=policy).post_json(
            url=f"http://provider.test:{port}/v1",
            headers={},
            payload=_payload(),
            timeout_seconds=3.0,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)

    assert response.status_code == 200
    assert attempted == [("127.0.0.2", port), ("127.0.0.1", port)]
    assert record.reached is True


def test_multiple_address_retries_share_remaining_request_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed approved address consumes the shared monotonic budget."""
    original_create_connection = socket.create_connection
    clock = [100.0]
    timeouts: list[float | None] = []

    monkeypatch.setattr(
        stdlib_http,
        "monotonic",
        lambda: clock[0],
        raising=False,
    )

    def fail_first_approved_address(
        address: tuple[str, int],
        timeout: float | None = None,
        source_address: tuple[str, int] | None = None,
    ) -> socket.socket:
        timeouts.append(timeout)
        if address[0] == "127.0.0.2":
            clock[0] += 0.6
            raise OSError("controlled first-address failure")
        return original_create_connection(
            address,
            timeout=timeout,
            source_address=source_address,
        )

    record = PinnedRequestRecord()
    server, thread = _start_recording_server(record)
    monkeypatch.setattr(socket, "create_connection", fail_first_approved_address)

    try:
        port = server.server_address[1]
        policy = ProviderEgressPolicy(
            allowed_origins=frozenset({f"http://provider.test:{port}"}),
            allow_non_global_addresses=True,
            resolver=FixedResolver(("127.0.0.2", "127.0.0.1")),
        )
        response = StdlibJsonTransport(url_validator=policy).post_json(
            url=f"http://provider.test:{port}/v1",
            headers={},
            payload=_payload(),
            timeout_seconds=1.0,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)

    assert response.status_code == 200
    assert timeouts == pytest.approx([1.0, 0.4])


def test_address_retries_stop_after_shared_budget_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An expired request budget prevents trying later approved addresses."""
    clock = [50.0]
    attempted: list[tuple[str, int]] = []
    timeouts: list[float | None] = []

    monkeypatch.setattr(
        stdlib_http,
        "monotonic",
        lambda: clock[0],
        raising=False,
    )

    def consume_budget_and_fail(
        address: tuple[str, int],
        timeout: float | None = None,
        source_address: tuple[str, int] | None = None,
    ) -> socket.socket:
        attempted.append(address)
        timeouts.append(timeout)
        clock[0] += 1.1
        raise OSError("controlled connection failure")

    monkeypatch.setattr(socket, "create_connection", consume_budget_and_fail)
    policy = ProviderEgressPolicy(
        allowed_origins=frozenset({"http://provider.test:8080"}),
        allow_non_global_addresses=True,
        resolver=FixedResolver(("127.0.0.2", "127.0.0.1")),
    )

    with pytest.raises(URLError):
        StdlibJsonTransport(url_validator=policy).post_json(
            url="http://provider.test:8080/v1",
            headers={},
            payload=_payload(),
            timeout_seconds=1.0,
        )

    assert attempted == [("127.0.0.2", 8080)]
    assert timeouts == pytest.approx([1.0])


def test_resolver_returning_after_budget_expiry_fails_before_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An over-budget synchronous DNS lookup cannot be followed by a request."""
    clock = [10.0]
    resolver_calls: list[str] = []
    attempted: list[tuple[str, int]] = []

    class SlowResolver:
        def resolve(self, *, host: str, port: int) -> tuple[str, ...]:
            resolver_calls.append(host)
            clock[0] += 1.1
            return ("127.0.0.1",)

    def record_connection(
        address: tuple[str, int],
        timeout: float | None = None,
        source_address: tuple[str, int] | None = None,
    ) -> socket.socket:
        attempted.append(address)
        raise OSError("connection must not be attempted")

    monkeypatch.setattr(
        stdlib_http,
        "monotonic",
        lambda: clock[0],
        raising=False,
    )
    monkeypatch.setattr(socket, "create_connection", record_connection)
    policy = ProviderEgressPolicy(
        allowed_origins=frozenset({"http://provider.test:8080"}),
        allow_non_global_addresses=True,
        resolver=SlowResolver(),
    )

    with pytest.raises(TimeoutError):
        StdlibJsonTransport(url_validator=policy).post_json(
            url="http://provider.test:8080/v1",
            headers={"Authorization": "Bearer must-not-be-sent"},
            payload=_payload(),
            timeout_seconds=1.0,
        )

    assert resolver_calls == ["provider.test"]
    assert attempted == []


def test_tls_handshake_timeout_uses_remaining_request_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stalled local TLS peer is stopped by the remaining deadline."""
    accepted = Event()
    release_peer = Event()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)

    def hold_tcp_connection() -> None:
        peer, _ = listener.accept()
        accepted.set()
        release_peer.wait(timeout=5.0)
        peer.close()

    thread = Thread(target=hold_tcp_connection, daemon=True)
    thread.start()
    observed_timeouts: list[float | None] = []

    def no_tls_readiness(
        readers: list[object],
        writers: list[object],
        exceptional: list[object],
        timeout: float | None = None,
    ) -> tuple[list[object], list[object], list[object]]:
        observed_timeouts.append(timeout)
        return ([], [], [])

    monkeypatch.setattr(
        stdlib_http,
        "_select",
        no_tls_readiness,
        raising=False,
    )
    port = listener.getsockname()[1]
    policy = ProviderEgressPolicy(
        allowed_origins=frozenset({f"https://provider.test:{port}"}),
        allow_non_global_addresses=True,
        resolver=FixedResolver(("127.0.0.1",)),
    )

    try:
        with pytest.raises(URLError):
            StdlibJsonTransport(url_validator=policy).post_json(
                url=f"https://provider.test:{port}/v1",
                headers={"Authorization": "Bearer must-not-be-sent"},
                payload=_payload(),
                timeout_seconds=1.0,
            )
    finally:
        release_peer.set()
        listener.close()
        thread.join(timeout=5.0)

    assert accepted.is_set()
    assert len(observed_timeouts) == 1
    assert observed_timeouts[0] is not None
    assert 0 < observed_timeouts[0] <= 1.0


def test_response_stream_read_uses_remaining_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each response socket read receives only the remaining budget."""
    clock = [5.0]
    monkeypatch.setattr(
        stdlib_http,
        "monotonic",
        lambda: clock[0],
        raising=False,
    )

    class StalledSocket:
        def __init__(self) -> None:
            self.timeouts: list[float | None] = []
            self.reads = 0

        def settimeout(self, timeout: float | None) -> None:
            self.timeouts.append(timeout)

        def recv_into(self, buffer: memoryview) -> int:
            self.reads += 1
            if self.reads == 1:
                buffer[0] = ord("a")
                clock[0] += 0.6
                return 1
            raise TimeoutError("controlled stalled response")

        def close(self) -> None:
            return

    stalled_socket = StalledSocket()
    deadline_socket = stdlib_http._DeadlineSocket(
        cast(socket.socket, stalled_socket),
        deadline=6.0,
    )
    response_stream = deadline_socket.makefile("rb", buffering=1)

    try:
        with pytest.raises(TimeoutError):
            response_stream.read(2)
    finally:
        response_stream.close()
        deadline_socket.close()

    assert stalled_socket.timeouts == pytest.approx([1.0, 0.4])


def test_ipv6_approved_destination_is_used_without_hostname_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """IPv6 destinations remain literal and failures never fall back to DNS."""
    attempted: list[tuple[str, int]] = []

    def reject_connection(
        address: tuple[str, int],
        timeout: float | None = None,
        source_address: tuple[str, int] | None = None,
    ) -> socket.socket:
        attempted.append(address)
        raise OSError("controlled IPv6 connection failure")

    monkeypatch.setattr(socket, "create_connection", reject_connection)
    policy = ProviderEgressPolicy(
        allowed_origins=frozenset({"http://ipv6.provider.test:8080"}),
        resolver=FixedResolver(("2001:4860:4860::8888",)),
    )

    with pytest.raises(URLError):
        StdlibJsonTransport(url_validator=policy).post_json(
            url="http://ipv6.provider.test:8080/v1",
            headers={"Authorization": "Bearer must-not-be-sent"},
            payload=_payload(),
            timeout_seconds=3.0,
        )

    assert attempted == [("2001:4860:4860::8888", 8080)]


def test_transport_rejects_non_ip_values_from_address_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A validator hostname must never reach socket connection resolution."""
    attempted: list[tuple[str, int]] = []

    class HostnameAddressValidator:
        def validate_url(self, url: str) -> None:
            return

        def resolve_validated_addresses(self, url: str) -> tuple[str, ...]:
            return ("unvalidated.provider.test",)

    def record_connection(
        address: tuple[str, int],
        timeout: float | None = None,
        source_address: tuple[str, int] | None = None,
    ) -> socket.socket:
        attempted.append(address)
        raise OSError("connection should not be attempted")

    monkeypatch.setattr(socket, "create_connection", record_connection)

    with pytest.raises(URLError):
        StdlibJsonTransport(url_validator=HostnameAddressValidator()).post_json(
            url="http://provider.test:8080/v1",
            headers={"Authorization": "Bearer must-not-be-sent"},
            payload=_payload(),
            timeout_seconds=3.0,
        )

    assert attempted == []


def test_https_preserves_original_hostname_for_sni_and_certificate_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """HTTPS pins TCP to policy IP and verifies the DNS certificate identity."""
    certificate_path = _generate_test_certificate(tmp_path)
    _trust_test_certificate(monkeypatch, certificate_path)
    record = PinnedRequestRecord()
    server_names: list[str | None] = []
    server, thread = _start_https_recording_server(
        certificate_path,
        record,
        server_names,
    )

    try:
        port = server.server_address[1]
        policy = ProviderEgressPolicy(
            allowed_origins=frozenset({f"https://provider.test:{port}"}),
            allow_non_global_addresses=True,
            resolver=FixedResolver(("127.0.0.1",)),
        )
        response = StdlibJsonTransport(url_validator=policy).post_json(
            url=f"https://provider.test:{port}/v1",
            headers={"Authorization": "Bearer local-test"},
            payload=_payload(),
            timeout_seconds=3.0,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)

    assert response.status_code == 200
    assert server_names == ["provider.test"]
    assert record.host == f"provider.test:{port}"
    assert record.authorization == "Bearer local-test"
    assert record.reached is True


def test_https_rejects_certificate_hostname_mismatch_before_http_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A trusted certificate for another hostname cannot receive a request."""
    certificate_path = _generate_test_certificate(tmp_path)
    _trust_test_certificate(monkeypatch, certificate_path)
    record = PinnedRequestRecord()
    server_names: list[str | None] = []
    server, thread = _start_https_recording_server(
        certificate_path,
        record,
        server_names,
    )

    try:
        port = server.server_address[1]
        policy = ProviderEgressPolicy(
            allowed_origins=frozenset({f"https://wrong.provider.test:{port}"}),
            allow_non_global_addresses=True,
            resolver=FixedResolver(("127.0.0.1",)),
        )
        with pytest.raises(URLError):
            StdlibJsonTransport(url_validator=policy).post_json(
                url=f"https://wrong.provider.test:{port}/v1",
                headers={"Authorization": "Bearer must-not-be-sent"},
                payload=_payload(),
                timeout_seconds=3.0,
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)

    assert server_names == ["wrong.provider.test"]
    assert record.reached is False
    assert record.authorization is None
    assert record.body == b""


def test_https_does_not_retry_after_untrusted_certificate_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An untrusted TLS certificate must stop approved-address fallback."""
    certificate_path = _generate_test_certificate(tmp_path)
    record = PinnedRequestRecord()
    server_names: list[str | None] = []
    server, thread = _start_https_recording_server(
        certificate_path,
        record,
        server_names,
    )
    original_create_connection = socket.create_connection
    attempted: list[tuple[str, int]] = []

    def map_second_approved_address_to_local_fixture(
        address: tuple[str, int],
        timeout: float | None = None,
        source_address: tuple[str, int] | None = None,
    ) -> socket.socket:
        attempted.append(address)
        destination = (
            ("127.0.0.1", address[1])
            if address[0] == "127.0.0.2"
            else address
        )
        return original_create_connection(
            destination,
            timeout=timeout,
            source_address=source_address,
        )

    monkeypatch.setattr(
        socket,
        "create_connection",
        map_second_approved_address_to_local_fixture,
    )

    try:
        port = server.server_address[1]
        policy = ProviderEgressPolicy(
            allowed_origins=frozenset({f"https://provider.test:{port}"}),
            allow_non_global_addresses=True,
            resolver=FixedResolver(("127.0.0.1", "127.0.0.2")),
        )

        with pytest.raises(URLError):
            StdlibJsonTransport(url_validator=policy).post_json(
                url=f"https://provider.test:{port}/v1",
                headers={"Authorization": "Bearer must-not-be-sent"},
                payload=_payload(),
                timeout_seconds=3.0,
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)

    assert attempted == [("127.0.0.1", port)]
    assert server_names == ["provider.test"]
    assert record.reached is False
    assert record.authorization is None
    assert record.body == b""
