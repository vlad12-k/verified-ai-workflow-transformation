"""Tests for the M6-B provider egress and SSRF policy."""

from dataclasses import dataclass, field
from ipaddress import ip_address

import pytest

from vait.platform.security.egress import (
    ProviderEgressDenied,
    ProviderEgressPolicy,
    SystemHostResolver,
)


@dataclass
class FakeResolver:
    """Deterministic resolver used without external DNS access."""

    addresses: tuple[str, ...] = (
        "8.8.8.8",
    )
    error: OSError | None = None
    calls: list[
        tuple[str, int]
    ] = field(
        default_factory=list
    )

    def resolve(
        self,
        *,
        host: str,
        port: int,
    ) -> tuple[str, ...]:
        self.calls.append(
            (
                host,
                port,
            )
        )

        if self.error is not None:
            raise self.error

        return self.addresses


def _policy(
    *,
    resolver: FakeResolver | None = None,
    allowed_origins: frozenset[str] | None = None,
    allow_non_global_addresses: bool = False,
) -> ProviderEgressPolicy:
    return ProviderEgressPolicy(
        allowed_origins=(
            allowed_origins
            or frozenset(
                {
                    "https://api.provider.test",
                }
            )
        ),
        allow_non_global_addresses=(
            allow_non_global_addresses
        ),
        resolver=(
            resolver
            or FakeResolver()
        ),
    )


def test_exact_allowlisted_origin_is_accepted() -> None:
    """A configured provider origin may reach a public address."""
    resolver = FakeResolver()

    policy = _policy(
        resolver=resolver,
    )

    policy.validate_url(
        "https://api.provider.test/v1/generate"
    )

    assert resolver.calls == [
        (
            "api.provider.test",
            443,
        )
    ]


def test_origin_matching_normalises_case_default_port_and_trailing_dot(
    ) -> None:
    """Equivalent canonical origins must compare deterministically."""
    resolver = FakeResolver()

    policy = _policy(
        resolver=resolver,
        allowed_origins=frozenset(
            {
                "HTTPS://API.PROVIDER.TEST./",
            }
        ),
    )

    policy.validate_url(
        "https://api.provider.test:443/v1"
    )

    assert resolver.calls == [
        (
            "api.provider.test",
            443,
        )
    ]


def test_unlisted_origin_is_rejected_before_dns_resolution() -> None:
    """An attacker-controlled host must fail before network resolution."""
    resolver = FakeResolver()

    policy = _policy(
        resolver=resolver,
    )

    with pytest.raises(
        ProviderEgressDenied,
    ) as exc_info:
        policy.validate_url(
            "https://attacker.test/v1"
        )

    assert (
        exc_info.value.reason
        == "origin_not_allowed"
    )
    assert resolver.calls == []


def test_non_http_scheme_is_rejected() -> None:
    """Provider egress is limited to HTTP(S) transports."""
    policy = _policy()

    with pytest.raises(
        ProviderEgressDenied,
    ) as exc_info:
        policy.validate_url(
            "file:///etc/passwd"
        )

    assert (
        exc_info.value.reason
        == "unsupported_scheme"
    )


def test_embedded_url_credentials_are_rejected_without_echo() -> None:
    """Authority credentials must not become provider endpoint material."""
    secret = "url-password-secret"

    policy = _policy()

    with pytest.raises(
        ProviderEgressDenied,
    ) as exc_info:
        policy.validate_url(
            "https://user:"
            f"{secret}"
            "@api.provider.test/v1"
        )

    assert (
        exc_info.value.reason
        == "embedded_credentials"
    )
    assert secret not in str(
        exc_info.value
    )


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.10",
        "169.254.169.254",
        "::1",
    ],
)
def test_non_global_resolved_addresses_are_rejected(
    address: str,
) -> None:
    """Loopback, private, and link-local targets must fail closed."""
    resolver = FakeResolver(
        addresses=(
            address,
        )
    )

    policy = _policy(
        resolver=resolver,
    )

    with pytest.raises(
        ProviderEgressDenied,
    ) as exc_info:
        policy.validate_url(
            "https://api.provider.test/v1"
        )

    assert (
        exc_info.value.reason
        == "non_global_address"
    )


def test_mixed_public_and_private_resolution_is_rejected() -> None:
    """One unsafe address invalidates the complete DNS answer set."""
    resolver = FakeResolver(
        addresses=(
            "8.8.8.8",
            "127.0.0.1",
        )
    )

    policy = _policy(
        resolver=resolver,
    )

    with pytest.raises(
        ProviderEgressDenied,
    ) as exc_info:
        policy.validate_url(
            "https://api.provider.test/v1"
        )

    assert (
        exc_info.value.reason
        == "non_global_address"
    )


def test_metadata_ip_literal_is_rejected_without_dns() -> None:
    """Cloud metadata addresses must fail even when explicitly named."""
    resolver = FakeResolver()

    policy = _policy(
        resolver=resolver,
        allowed_origins=frozenset(
            {
                "https://169.254.169.254",
            }
        ),
    )

    with pytest.raises(
        ProviderEgressDenied,
    ) as exc_info:
        policy.validate_url(
            "https://169.254.169.254/latest/meta-data"
        )

    assert (
        exc_info.value.reason
        == "non_global_address"
    )
    assert resolver.calls == []


def test_resolution_failure_is_fail_closed() -> None:
    """DNS failure must never permit the outbound request."""
    resolver = FakeResolver(
        error=OSError(
            "synthetic resolver failure"
        )
    )

    policy = _policy(
        resolver=resolver,
    )

    with pytest.raises(
        ProviderEgressDenied,
    ) as exc_info:
        policy.validate_url(
            "https://api.provider.test/v1"
        )

    assert (
        exc_info.value.reason
        == "resolution_failed"
    )
    assert (
        "synthetic resolver failure"
        not in str(
            exc_info.value
        )
    )


def test_empty_resolution_is_fail_closed() -> None:
    """A hostname without resolved addresses is not a valid target."""
    policy = _policy(
        resolver=FakeResolver(
            addresses=(),
        )
    )

    with pytest.raises(
        ProviderEgressDenied,
    ) as exc_info:
        policy.validate_url(
            "https://api.provider.test/v1"
        )

    assert (
        exc_info.value.reason
        == "resolution_failed"
    )


def test_malformed_resolved_address_is_fail_closed() -> None:
    """Unexpected resolver output must not bypass IP classification."""
    policy = _policy(
        resolver=FakeResolver(
            addresses=(
                "not-an-ip-address",
            )
        )
    )

    with pytest.raises(
        ProviderEgressDenied,
    ) as exc_info:
        policy.validate_url(
            "https://api.provider.test/v1"
        )

    assert (
        exc_info.value.reason
        == "resolution_failed"
    )


def test_explicit_local_test_policy_can_allow_loopback() -> None:
    """Local integration tests require an explicit non-global override."""
    policy = _policy(
        allowed_origins=frozenset(
            {
                "http://127.0.0.1:4567",
            }
        ),
        allow_non_global_addresses=True,
    )

    policy.validate_url(
        "http://127.0.0.1:4567/v1"
    )


@pytest.mark.parametrize(
    "origin",
    [
        "",
        "ftp://api.provider.test",
        "https://user:secret@api.provider.test",
        "https://api.provider.test/v1",
        "https://api.provider.test?query=value",
        "https://api.provider.test#fragment",
        "https://api.provider.test:invalid",
    ],
)
def test_invalid_allowed_origin_configuration_is_rejected(
    origin: str,
) -> None:
    """Configured allow-list entries must be unambiguous bare origins."""
    with pytest.raises(
        ValueError,
        match=(
            "allowed_origins must contain "
            "bare HTTP"
        ),
    ):
        ProviderEgressPolicy(
            allowed_origins=frozenset(
                {
                    origin,
                }
            ),
            resolver=FakeResolver(),
        )


def test_empty_allowlist_is_rejected() -> None:
    """Provider egress cannot start without an explicit destination."""
    with pytest.raises(
        ValueError,
        match="allowed_origins must not be empty",
    ):
        ProviderEgressPolicy(
            allowed_origins=frozenset(),
            resolver=FakeResolver(),
        )


def test_invalid_runtime_port_is_rejected_without_dns() -> None:
    """Malformed endpoint ports must fail before resolution."""
    resolver = FakeResolver()

    policy = _policy(
        resolver=resolver,
    )

    with pytest.raises(
        ProviderEgressDenied,
    ) as exc_info:
        policy.validate_url(
            "https://api.provider.test:invalid/v1"
        )

    assert (
        exc_info.value.reason
        == "invalid_url"
    )
    assert resolver.calls == []


def test_system_resolver_returns_parseable_unique_sorted_addresses() -> None:
    """The production resolver must return canonical deterministic addresses."""
    resolver = SystemHostResolver()

    addresses = resolver.resolve(
        host="localhost",
        port=80,
    )

    assert addresses
    assert addresses == tuple(
        sorted(
            set(addresses)
        )
    )

    for address in addresses:
        ip_address(
            address
        )


def test_runtime_url_without_scheme_is_rejected() -> None:
    """A schemeless endpoint must fail before origin or DNS evaluation."""
    resolver = FakeResolver()

    policy = _policy(
        resolver=resolver,
    )

    with pytest.raises(
        ProviderEgressDenied,
    ) as exc_info:
        policy.validate_url(
            "api.provider.test/v1"
        )

    assert (
        exc_info.value.reason
        == "invalid_url"
    )
    assert resolver.calls == []


def test_runtime_url_without_hostname_is_rejected() -> None:
    """HTTP(S) syntax without a hostname must fail closed."""
    resolver = FakeResolver()

    policy = _policy(
        resolver=resolver,
    )

    with pytest.raises(
        ProviderEgressDenied,
    ) as exc_info:
        policy.validate_url(
            "https:///v1/generate"
        )

    assert (
        exc_info.value.reason
        == "invalid_url"
    )
    assert resolver.calls == []


def test_runtime_url_with_invalid_ipv6_syntax_is_rejected() -> None:
    """Malformed IPv6 authority syntax must fail during URL parsing."""
    policy = _policy()

    with pytest.raises(
        ProviderEgressDenied,
    ) as exc_info:
        policy.validate_url(
            "https://[::1"
        )

    assert (
        exc_info.value.reason
        == "invalid_url"
    )


def test_password_only_userinfo_is_rejected() -> None:
    """Password-bearing authority must fail even without a username."""
    secret = "password-only-secret"

    policy = _policy()

    with pytest.raises(
        ProviderEgressDenied,
    ) as exc_info:
        policy.validate_url(
            f"https://:{secret}"
            "@api.provider.test/v1"
        )

    assert (
        exc_info.value.reason
        == "embedded_credentials"
    )
    assert secret not in str(
        exc_info.value
    )


def test_default_http_port_is_canonicalised() -> None:
    """HTTP without an explicit port must canonicalise to port 80."""
    resolver = FakeResolver()

    policy = _policy(
        resolver=resolver,
        allowed_origins=frozenset(
            {
                "http://api.provider.test",
            }
        ),
    )

    policy.validate_url(
        "http://api.provider.test/v1"
    )

    assert resolver.calls == [
        (
            "api.provider.test",
            80,
        )
    ]


def test_global_ipv6_literal_is_accepted_without_dns() -> None:
    """A globally routable allow-listed IPv6 literal needs no DNS lookup."""
    resolver = FakeResolver()

    policy = _policy(
        resolver=resolver,
        allowed_origins=frozenset(
            {
                "https://[2001:4860:4860::8888]",
            }
        ),
    )

    policy.validate_url(
        "https://[2001:4860:4860::8888]/v1"
    )

    assert resolver.calls == []


@pytest.mark.parametrize(
    "origin",
    [
        "https:///v1",
        "https://:secret@api.provider.test",
        "https://.",
        "https://%25",
        "https://api..provider.test",
    ],
)
def test_additional_malformed_allowed_origins_are_rejected(
    origin: str,
) -> None:
    """Malformed authorities must not survive allow-list canonicalisation."""
    with pytest.raises(
        ValueError,
        match=(
            "allowed_origins must contain "
            "bare HTTP"
        ),
    ):
        ProviderEgressPolicy(
            allowed_origins=frozenset(
                {
                    origin,
                }
            ),
            resolver=FakeResolver(),
        )


def test_invalid_ipv6_allowlist_origin_is_rejected() -> None:
    """Malformed IPv6 allow-list syntax must fail during configuration."""
    with pytest.raises(
        ValueError,
        match=(
            "allowed_origins must contain "
            "bare HTTP"
        ),
    ):
        ProviderEgressPolicy(
            allowed_origins=frozenset(
                {
                    "https://[::1",
                }
            ),
            resolver=FakeResolver(),
        )


def test_invalid_idna_hostname_is_rejected() -> None:
    """A hostname that cannot be IDNA encoded must fail closed."""
    invalid_host = chr(
        0xD800
    )

    with pytest.raises(
        ValueError,
        match=(
            "allowed_origins must contain "
            "bare HTTP"
        ),
    ):
        ProviderEgressPolicy(
            allowed_origins=frozenset(
                {
                    f"https://{invalid_host}",
                }
            ),
            resolver=FakeResolver(),
        )


def test_allowlist_origin_without_hostname_is_rejected() -> None:
    """A syntactically present authority still requires a real hostname."""
    with pytest.raises(
        ValueError,
        match=(
            "allowed_origins must contain "
            "bare HTTP"
        ),
    ):
        ProviderEgressPolicy(
            allowed_origins=frozenset(
                {
                    "https://:443",
                }
            ),
            resolver=FakeResolver(),
        )
