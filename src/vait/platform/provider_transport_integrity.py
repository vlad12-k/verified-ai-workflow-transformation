"""Fail-closed allowlist for remote provider transport configuration."""

from typing import cast
from urllib.parse import urlsplit

from vait.inference.providers.base import InferenceProvider
from vait.inference.providers.cache import DeterministicCachingProvider
from vait.inference.providers.openai_compatible import OpenAICompatibleProvider
from vait.inference.providers.stdlib_http import StdlibJsonTransport
from vait.platform.provider_dispatch_policy import ProviderDispatchPolicyError
from vait.platform.security.egress import (
    ProviderEgressPolicy,
    SystemHostResolver,
)


def require_known_remote_transport(provider: InferenceProvider) -> None:
    """Require known HTTPS transport and strict egress configuration.

    This is a configuration gate, not proof of receiver-side fencing,
    zero retries, or an at-most-once remote effect.
    """
    current = provider

    for _ in range(8):
        adapter_type = type(current)

        if adapter_type is DeterministicCachingProvider:
            current = cast(DeterministicCachingProvider, current)._provider
            continue

        if adapter_type is not OpenAICompatibleProvider:
            raise ProviderDispatchPolicyError(
                "Remote provider transport has not been audited."
            )

        adapter = cast(OpenAICompatibleProvider, current)

        if urlsplit(adapter.chat_completions_url).scheme.casefold() != "https":
            raise ProviderDispatchPolicyError(
                "Remote inference requires HTTPS."
            )

        transport = adapter._transport

        if type(transport) is not StdlibJsonTransport:
            raise ProviderDispatchPolicyError(
                "Remote inference requires a known HTTP transport."
            )

        validator = transport._url_validator

        if type(validator) is not ProviderEgressPolicy:
            raise ProviderDispatchPolicyError(
                "Remote inference requires the built-in egress policy."
            )

        egress = validator

        if egress.allow_non_global_addresses:
            raise ProviderDispatchPolicyError(
                "Remote inference cannot allow non-global addresses."
            )

        if type(egress.resolver) is not SystemHostResolver:
            raise ProviderDispatchPolicyError(
                "Remote inference requires the system host resolver."
            )

        return

    raise ProviderDispatchPolicyError(
        "Remote transport wrapper depth exceeded."
    )
