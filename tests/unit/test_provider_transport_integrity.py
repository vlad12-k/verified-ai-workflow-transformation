"""Tests for fail-closed remote transport configuration."""

from typing import cast

import pytest

from vait.inference.providers.cache import DeterministicCachingProvider
from vait.inference.providers.openai_compatible import (
    OpenAICompatibleProvider,
    OpenAICompatibleTransport,
)
from vait.inference.providers.stdlib_http import StdlibJsonTransport
from vait.inference.providers.watsonx import WatsonxProvider
from vait.platform.provider_dispatch_policy import ProviderDispatchPolicyError
from vait.platform.provider_transport_integrity import (
    require_known_remote_transport,
)
from vait.platform.security.egress import (
    HostResolver,
    ProviderEgressPolicy,
    SystemHostResolver,
)


def _known_provider(
    *,
    url: str = "https://api.provider.test",
    allow_non_global: bool = False,
    resolver: HostResolver | None = None,
) -> OpenAICompatibleProvider:
    """Build real transport objects without making network requests."""
    egress = ProviderEgressPolicy(
        allowed_origins=frozenset({"https://api.provider.test"}),
        allow_non_global_addresses=allow_non_global,
        resolver=resolver if resolver is not None else SystemHostResolver(),
    )
    return OpenAICompatibleProvider(
        base_url=url,
        model_id="test-model",
        transport=StdlibJsonTransport(url_validator=egress),
    )


def test_known_https_transport_is_accepted() -> None:
    require_known_remote_transport(_known_provider())


def test_http_without_tls_is_rejected() -> None:
    with pytest.raises(ProviderDispatchPolicyError, match="HTTPS"):
        require_known_remote_transport(
            _known_provider(url="http://api.provider.test")
        )


def test_arbitrary_transport_is_rejected() -> None:
    provider = _known_provider()
    provider._transport = cast(OpenAICompatibleTransport, object())

    with pytest.raises(
        ProviderDispatchPolicyError,
        match="known HTTP transport",
    ):
        require_known_remote_transport(provider)


def test_non_global_egress_configuration_is_rejected() -> None:
    with pytest.raises(
        ProviderDispatchPolicyError,
        match="non-global",
    ):
        require_known_remote_transport(
            _known_provider(allow_non_global=True)
        )


def test_untrusted_dns_resolver_is_rejected() -> None:
    with pytest.raises(
        ProviderDispatchPolicyError,
        match="system host resolver",
    ):
        require_known_remote_transport(
            _known_provider(resolver=cast(HostResolver, object()))
        )


def test_watsonx_transport_is_unverified_and_rejected() -> None:
    provider = object.__new__(WatsonxProvider)

    with pytest.raises(
        ProviderDispatchPolicyError,
        match="has not been audited",
    ):
        require_known_remote_transport(provider)


def test_cache_wrapper_preserves_known_transport() -> None:
    provider = DeterministicCachingProvider(
        provider=_known_provider()
    )

    require_known_remote_transport(provider)


def test_cache_wrapper_cannot_hide_untrusted_transport() -> None:
    provider = _known_provider()
    provider._transport = cast(OpenAICompatibleTransport, object())
    cached = DeterministicCachingProvider(provider=provider)

    with pytest.raises(
        ProviderDispatchPolicyError,
        match="known HTTP transport",
    ):
        require_known_remote_transport(cached)
