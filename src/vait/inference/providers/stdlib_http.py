"""Standard-library JSON HTTP transport for inference providers."""

import http.client
import io
import json
import math
import socket
import ssl
from collections.abc import Mapping
from contextvars import ContextVar
from http.client import HTTPMessage
from ipaddress import ip_address
from select import select as _select
from threading import Lock
from time import monotonic
from typing import IO, TYPE_CHECKING, Protocol, cast
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import (
    HTTPHandler,
    HTTPRedirectHandler,
    HTTPSHandler,
    ProxyHandler,
    Request,
    build_opener,
)

from pydantic import JsonValue, TypeAdapter

from vait.inference.providers.openai_compatible import (
    OpenAICompatibleHttpResponse,
)

if TYPE_CHECKING:
    from _typeshed import WriteableBuffer

_JSON_OBJECT_ADAPTER = TypeAdapter(
    dict[str, JsonValue]
)
_PINNED_ADDRESSES: ContextVar[tuple[str, ...] | None] = ContextVar(
    "vait_pinned_provider_addresses",
    default=None,
)
_REQUEST_DEADLINE: ContextVar[float | None] = ContextVar(
    "vait_provider_request_deadline",
    default=None,
)


def _deadline_remaining(deadline: float) -> float:
    """Return the positive time remaining in one monotonic request budget."""
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise TimeoutError("provider request deadline exceeded")
    return remaining


def _request_deadline() -> float:
    deadline = _REQUEST_DEADLINE.get()
    if deadline is None:
        raise OSError("provider request deadline is not active")
    return deadline


def _remaining_request_timeout() -> float:
    return _deadline_remaining(_request_deadline())


class _DeadlineSocketReader(io.RawIOBase):
    """Read response bytes with the current remaining request budget."""

    def __init__(self, owner: "_DeadlineSocket") -> None:
        super().__init__()
        self._owner = owner

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: "WriteableBuffer") -> int:
        if self.closed:
            return 0
        timeout = _deadline_remaining(self._owner.deadline)
        self._owner.socket.settimeout(timeout)
        return self._owner.socket.recv_into(buffer)

    def close(self) -> None:
        if not self.closed:
            self._owner._release_reader()
        super().close()


class _DeadlineSocket:
    """Apply one absolute deadline to urllib's socket reads and writes."""

    def __init__(self, sock: socket.socket, deadline: float) -> None:
        self.socket = sock
        self.deadline = deadline
        self._lock = Lock()
        self._reader_count = 0
        self._close_requested = False
        self._socket_closed = False

    def sendall(self, data: bytes) -> None:
        self.socket.settimeout(_deadline_remaining(self.deadline))
        self.socket.sendall(data)

    def makefile(
        self,
        mode: str,
        buffering: int | None = None,
    ) -> IO[bytes]:
        if mode != "rb":
            raise ValueError("provider response socket supports binary reads only")
        with self._lock:
            if self._close_requested:
                raise OSError("provider response socket is closed")
            self._reader_count += 1
        raw = _DeadlineSocketReader(self)
        buffer_size = (
            io.DEFAULT_BUFFER_SIZE
            if buffering is None or buffering <= 0
            else buffering
        )
        return io.BufferedReader(raw, buffer_size=buffer_size)

    def close(self) -> None:
        with self._lock:
            self._close_requested = True
            self._close_socket_if_ready()

    def _release_reader(self) -> None:
        with self._lock:
            self._reader_count -= 1
            self._close_socket_if_ready()

    def _close_socket_if_ready(self) -> None:
        if (
            self._close_requested
            and self._reader_count == 0
            and not self._socket_closed
        ):
            self._socket_closed = True
            self.socket.close()


class OutboundUrlValidator(Protocol):
    """Validate one outbound provider URL before network access."""

    def validate_url(
        self,
        url: str,
    ) -> None:
        """Reject an outbound URL that violates provider egress policy."""
        ...

    def resolve_validated_addresses(
        self,
        url: str,
    ) -> tuple[str, ...]:
        """Return the exact addresses approved for this outbound URL."""
        ...


class ProviderRedirectDenied(ValueError):
    """Raised when a provider redirect crosses the original origin."""

    def __init__(
        self,
    ) -> None:
        super().__init__(
            "cross-origin provider redirect denied"
        )


class _ValidatedRedirectHandler(
    HTTPRedirectHandler
):
    """Validate redirect targets before urllib follows them."""

    def __init__(
        self,
        *,
        url_validator: OutboundUrlValidator,
    ) -> None:
        self._url_validator = url_validator

    def redirect_request(
        self,
        req: Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> Request | None:
        """Permit only validated same-origin provider redirects."""
        try:
            self._url_validator.validate_url(
                newurl
            )
            _remaining_request_timeout()

            if (
                _origin_identity(
                    req.full_url
                )
                != _origin_identity(
                    newurl
                )
            ):
                raise ProviderRedirectDenied()
        except Exception:
            fp.close()
            raise

        return super().redirect_request(
            req,
            fp,
            code,
            msg,
            headers,
            newurl,
        )


class _PinnedHttpHandler(HTTPHandler):
    """Connect only to the addresses approved immediately before connection."""

    def __init__(self, *, url_validator: OutboundUrlValidator) -> None:
        super().__init__()
        self._url_validator = url_validator

    def http_open(self, req: Request) -> http.client.HTTPResponse:
        _remaining_request_timeout()
        addresses = self._url_validator.resolve_validated_addresses(req.full_url)
        _remaining_request_timeout()
        token = _PINNED_ADDRESSES.set(addresses)
        try:
            return self.do_open(_PinnedHTTPConnection, req)
        finally:
            _PINNED_ADDRESSES.reset(token)


class _PinnedHttpsHandler(HTTPSHandler):
    """Pin TCP address while preserving URL hostname for TLS verification."""

    def __init__(self, *, url_validator: OutboundUrlValidator) -> None:
        self._tls_context = ssl.create_default_context()
        super().__init__(context=self._tls_context)
        self._url_validator = url_validator

    def https_open(self, req: Request) -> http.client.HTTPResponse:
        _remaining_request_timeout()
        addresses = self._url_validator.resolve_validated_addresses(req.full_url)
        _remaining_request_timeout()
        token = _PINNED_ADDRESSES.set(addresses)
        try:
            return self.do_open(
                _PinnedHTTPSConnection,
                req,
                context=self._tls_context,
            )
        finally:
            _PINNED_ADDRESSES.reset(token)


def _approved_addresses() -> tuple[str, ...]:
    addresses = _PINNED_ADDRESSES.get()
    if not addresses:
        raise OSError("No policy-approved destination addresses")
    try:
        return tuple(ip_address(address).compressed for address in addresses)
    except ValueError as exc:
        raise OSError("Invalid policy-approved destination address") from exc


def _connect_to_approved_address(
    *,
    port: int,
    source_address: tuple[str, int] | None,
) -> socket.socket:
    deadline = _request_deadline()
    last_error: OSError | None = None
    for address in _approved_addresses():
        timeout = _deadline_remaining(deadline)
        sock: socket.socket | None = None
        try:
            sock = socket.create_connection(
                (address, port),
                timeout=timeout,
                source_address=source_address,
            )
            sock.settimeout(_deadline_remaining(deadline))
            return cast(socket.socket, _DeadlineSocket(sock, deadline))
        except OSError as exc:
            if sock is not None:
                sock.close()
            last_error = exc
    if last_error is not None:
        raise last_error
    raise OSError("No policy-approved destination addresses")


def _perform_tls_handshake(
    sock: ssl.SSLSocket,
    deadline: float,
) -> None:
    """Complete TLS using readiness waits bounded by the shared deadline."""
    sock.setblocking(False)
    while True:
        timeout = _deadline_remaining(deadline)
        try:
            sock.do_handshake()
            return
        except ssl.SSLWantReadError:
            readable, _, _ = _select([sock], [], [], timeout)
            if not readable:
                raise TimeoutError(
                    "provider TLS handshake deadline exceeded"
                ) from None
        except ssl.SSLWantWriteError:
            _, writable, _ = _select([], [sock], [], timeout)
            if not writable:
                raise TimeoutError(
                    "provider TLS handshake deadline exceeded"
                ) from None


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """HTTP connection that never resolves the request hostname."""

    source_address: tuple[str, int] | None

    def connect(self) -> None:
        self.sock = _connect_to_approved_address(
            port=self.port,
            source_address=self.source_address,
        )


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection pinned to approved IPs with original-host TLS checks."""

    source_address: tuple[str, int] | None
    _context: ssl.SSLContext

    def connect(self) -> None:
        deadline = _request_deadline()
        last_error: OSError | None = None
        for address in _approved_addresses():
            sock: socket.socket | None = None
            tls_sock: ssl.SSLSocket | None = None
            timeout = _deadline_remaining(deadline)
            try:
                sock = socket.create_connection(
                    (address, self.port),
                    timeout=timeout,
                    source_address=self.source_address,
                )
                sock.setblocking(False)
                tls_sock = self._context.wrap_socket(
                    sock,
                    server_hostname=self.host,
                    do_handshake_on_connect=False,
                )
                sock = None
                _perform_tls_handshake(tls_sock, deadline)
                tls_sock.setblocking(True)
                tls_sock.settimeout(_deadline_remaining(deadline))
                self.sock = cast(
                    socket.socket,
                    _DeadlineSocket(tls_sock, deadline),
                )
                return
            except ssl.SSLCertVerificationError:
                if tls_sock is not None:
                    tls_sock.close()
                elif sock is not None:
                    sock.close()
                raise
            except OSError as exc:
                if tls_sock is not None:
                    tls_sock.close()
                elif sock is not None:
                    sock.close()
                last_error = exc
        if last_error is not None:
            raise last_error
        raise OSError("No policy-approved destination addresses")


class StdlibJsonTransport:
    """Execute synchronous JSON POST requests through controlled egress."""

    def __init__(
        self,
        *,
        url_validator: OutboundUrlValidator,
    ) -> None:
        """Configure transport with a mandatory outbound URL validator."""
        self._url_validator = url_validator

        self._opener = build_opener(
            ProxyHandler(
                {}
            ),
            _PinnedHttpHandler(url_validator=url_validator),
            _PinnedHttpsHandler(url_validator=url_validator),
            _ValidatedRedirectHandler(
                url_validator=url_validator,
            ),
        )

    def post_json(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: dict[str, JsonValue],
        timeout_seconds: float,
    ) -> OpenAICompatibleHttpResponse:
        """POST JSON within one monotonic budget after each synchronous DNS call.

        System DNS resolution cannot be cancelled by this stdlib transport.
        DNS time consumes the request budget, but a blocked resolver can delay
        returning beyond it until its synchronous lookup completes.
        """
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")

        deadline = monotonic() + timeout_seconds
        token = _REQUEST_DEADLINE.set(deadline)
        try:
            self._url_validator.validate_url(url)
            _remaining_request_timeout()

            encoded_payload = json.dumps(
                payload,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")

            request = Request(
                url=url,
                data=encoded_payload,
                headers=dict(headers),
                method="POST",
            )

            try:
                with self._opener.open(
                    request,
                    timeout=_remaining_request_timeout(),
                ) as response:
                    _remaining_request_timeout()
                    response_body = response.read()
                    _remaining_request_timeout()
                    body = _decode_json_object(response_body)
                    _remaining_request_timeout()

                    return OpenAICompatibleHttpResponse(
                        status_code=response.status,
                        body=body,
                        headers={
                            name: value
                            for name, value in response.headers.items()
                        },
                    )

            except HTTPError as exc:
                response_body = exc.read()
                _remaining_request_timeout()
                body = _decode_json_object(response_body)
                _remaining_request_timeout()

                return OpenAICompatibleHttpResponse(
                    status_code=exc.code,
                    body=body,
                    headers={
                        name: value
                        for name, value in exc.headers.items()
                    },
                )
        finally:
            _REQUEST_DEADLINE.reset(token)


def _origin_identity(
    url: str,
) -> tuple[str, str]:
    """Return strict scheme-and-authority identity for redirect comparison."""
    parts = urlsplit(
        url
    )

    return (
        parts.scheme.casefold(),
        parts.netloc.casefold(),
    )


def _decode_json_object(
    payload: bytes,
) -> dict[str, JsonValue]:
    """Decode one JSON object returned by a compatible endpoint."""
    if not payload:
        return {}

    decoded: object = json.loads(
        payload.decode("utf-8")
    )

    return _JSON_OBJECT_ADAPTER.validate_python(
        decoded
    )
