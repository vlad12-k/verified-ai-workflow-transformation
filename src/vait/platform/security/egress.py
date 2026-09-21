"""Fail-closed provider egress policy for outbound HTTP endpoints."""

from dataclasses import dataclass, field
from ipaddress import ip_address
from socket import SOCK_STREAM, getaddrinfo
from typing import Literal, Protocol, cast
from urllib.parse import SplitResult, urlsplit

EgressDenialReason = Literal[
    "invalid_url",
    "unsupported_scheme",
    "embedded_credentials",
    "origin_not_allowed",
    "resolution_failed",
    "non_global_address",
]

_DEFAULT_PORTS = {
    "http": 80,
    "https": 443,
}


class ProviderEgressDenied(ValueError):
    """Raised when an outbound provider endpoint violates egress policy."""

    def __init__(
        self,
        reason: EgressDenialReason,
    ) -> None:
        self.reason = reason

        super().__init__(
            "provider egress denied"
        )


class HostResolver(Protocol):
    """Resolve one provider hostname into network addresses."""

    def resolve(
        self,
        *,
        host: str,
        port: int,
    ) -> tuple[str, ...]:
        """Return all resolved IP addresses for one endpoint."""
        ...


class SystemHostResolver:
    """Resolve provider endpoints through the operating system resolver."""

    def resolve(
        self,
        *,
        host: str,
        port: int,
    ) -> tuple[str, ...]:
        """Return unique addresses resolved for one host and TCP port."""
        results = getaddrinfo(
            host,
            port,
            type=SOCK_STREAM,
        )

        addresses = {
            cast(
                str,
                result[4][0],
            )
            for result in results
        }

        return tuple(
            sorted(addresses)
        )


@dataclass(frozen=True)
class ProviderEgressPolicy:
    """Allow only explicitly approved provider origins and addresses."""

    allowed_origins: frozenset[str]
    allow_non_global_addresses: bool = False
    resolver: HostResolver = field(
        default_factory=SystemHostResolver,
        repr=False,
        compare=False,
    )

    def __post_init__(
        self,
    ) -> None:
        if not self.allowed_origins:
            raise ValueError(
                "allowed_origins must not be empty"
            )

        canonical_origins = frozenset(
            _canonical_allowed_origin(
                origin
            )
            for origin in self.allowed_origins
        )

        object.__setattr__(
            self,
            "allowed_origins",
            canonical_origins,
        )

    def validate_url(
        self,
        url: str,
    ) -> None:
        """Reject an outbound URL unless every egress control passes."""
        parts = _split_runtime_url(
            url
        )

        scheme = parts.scheme.casefold()

        if scheme not in {
            "http",
            "https",
        }:
            raise ProviderEgressDenied(
                "unsupported_scheme"
            )

        if (
            parts.username is not None
            or parts.password is not None
        ):
            raise ProviderEgressDenied(
                "embedded_credentials"
            )

        host = parts.hostname

        if host is None:
            raise ProviderEgressDenied(
                "invalid_url"
            )

        try:
            normalised_host = _normalise_host(
                host
            )
            port = _effective_port(
                parts,
                scheme,
            )
        except ValueError as exc:
            raise ProviderEgressDenied(
                "invalid_url"
            ) from exc

        origin = _origin_key(
            scheme=scheme,
            host=normalised_host,
            port=port,
        )

        if origin not in self.allowed_origins:
            raise ProviderEgressDenied(
                "origin_not_allowed"
            )

        addresses = self._resolve_addresses(
            host=normalised_host,
            port=port,
        )

        if not addresses:
            raise ProviderEgressDenied(
                "resolution_failed"
            )

        for raw_address in addresses:
            try:
                address = ip_address(
                    raw_address
                )
            except ValueError as exc:
                raise ProviderEgressDenied(
                    "resolution_failed"
                ) from exc

            if (
                not self.allow_non_global_addresses
                and not address.is_global
            ):
                raise ProviderEgressDenied(
                    "non_global_address"
                )

    def _resolve_addresses(
        self,
        *,
        host: str,
        port: int,
    ) -> tuple[str, ...]:
        try:
            literal_address = ip_address(
                host
            )
        except ValueError:
            try:
                return self.resolver.resolve(
                    host=host,
                    port=port,
                )
            except OSError as exc:
                raise ProviderEgressDenied(
                    "resolution_failed"
                ) from exc

        return (
            literal_address.compressed,
        )


def _split_runtime_url(
    value: str,
) -> SplitResult:
    try:
        parts = urlsplit(
            value
        )
    except ValueError as exc:
        raise ProviderEgressDenied(
            "invalid_url"
        ) from exc

    if not parts.scheme:
        raise ProviderEgressDenied(
            "invalid_url"
        )

    return parts


def _canonical_allowed_origin(
    value: str,
) -> str:
    try:
        parts = urlsplit(
            value
        )
    except ValueError as exc:
        raise ValueError(
            "allowed_origins must contain bare HTTP(S) origins"
        ) from exc

    scheme = parts.scheme.casefold()

    if (
        scheme not in {
            "http",
            "https",
        }
        or not parts.netloc
        or parts.username is not None
        or parts.password is not None
        or parts.path not in {
            "",
            "/",
        }
        or parts.query
        or parts.fragment
    ):
        raise ValueError(
            "allowed_origins must contain bare HTTP(S) origins"
        )

    host = parts.hostname

    if host is None:
        raise ValueError(
            "allowed_origins must contain bare HTTP(S) origins"
        )

    try:
        normalised_host = _normalise_host(
            host
        )
        port = _effective_port(
            parts,
            scheme,
        )
    except ValueError as exc:
        raise ValueError(
            "allowed_origins must contain bare HTTP(S) origins"
        ) from exc

    return _origin_key(
        scheme=scheme,
        host=normalised_host,
        port=port,
    )


def _effective_port(
    parts: SplitResult,
    scheme: str,
) -> int:
    explicit_port = parts.port

    if explicit_port is not None:
        return explicit_port

    return _DEFAULT_PORTS[
        scheme
    ]


def _normalise_host(
    host: str,
) -> str:
    candidate = host.rstrip(".")

    if (
        not candidate
        or "%" in candidate
    ):
        raise ValueError(
            "invalid provider host"
        )

    try:
        address = ip_address(
            candidate
        )
    except ValueError:
        try:
            encoded = candidate.encode(
                "idna"
            ).decode(
                "ascii"
            )
        except UnicodeError as exc:
            raise ValueError(
                "invalid provider host"
            ) from exc

        return encoded.casefold()

    return address.compressed


def _origin_key(
    *,
    scheme: str,
    host: str,
    port: int,
) -> str:
    host_component = (
        f"[{host}]"
        if ":" in host
        else host
    )

    return (
        f"{scheme}://"
        f"{host_component}:"
        f"{port}"
    )
