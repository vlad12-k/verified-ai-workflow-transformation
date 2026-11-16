"""Unit tests for trusted built-in provider classification."""

import pytest

from vait.inference.providers.cache import DeterministicCachingProvider
from vait.inference.providers.huggingface import LocalHuggingFaceProvider
from vait.inference.providers.models import (
    ProviderInferenceRequest,
    ProviderInferenceResponse,
)
from vait.inference.providers.openai_compatible import OpenAICompatibleProvider
from vait.inference.providers.watsonx import WatsonxProvider
from vait.platform.provider_dispatch_policy import (
    ProviderDispatchPolicyError,
    ProviderLocation,
)
from vait.platform.provider_identity import classify_builtin_provider


def local_adapter(
    *, local_files_only: bool = True
) -> LocalHuggingFaceProvider:
    """Construct a type-only test instance without loading model weights."""
    provider = object.__new__(LocalHuggingFaceProvider)
    provider._local_files_only = local_files_only
    return provider


def test_openai_compatible_is_remote() -> None:
    provider = object.__new__(OpenAICompatibleProvider)

    assert classify_builtin_provider(provider) is ProviderLocation.REMOTE


def test_watsonx_is_remote() -> None:
    provider = object.__new__(WatsonxProvider)

    assert classify_builtin_provider(provider) is ProviderLocation.REMOTE


def test_local_huggingface_requires_offline_configuration() -> None:
    assert (
        classify_builtin_provider(local_adapter())
        is ProviderLocation.LOCAL
    )


def test_local_huggingface_with_downloads_enabled_is_rejected() -> None:
    with pytest.raises(
        ProviderDispatchPolicyError,
        match="local_files_only",
    ):
        classify_builtin_provider(
            local_adapter(local_files_only=False)
        )


class SpoofedLocalProvider:
    """Unregistered adapter pretending to have trusted identifiers."""

    provider_id = "local-huggingface"
    runtime_id = "transformers-causal-lm"

    def generate(
        self,
        request: ProviderInferenceRequest,
    ) -> ProviderInferenceResponse:
        del request
        raise AssertionError("Provider I/O must not occur")


def test_spoofed_provider_identifiers_do_not_grant_access() -> None:
    with pytest.raises(
        ProviderDispatchPolicyError,
        match="Unknown provider adapter",
    ):
        classify_builtin_provider(SpoofedLocalProvider())


def test_cache_wrapping_remote_provider_remains_remote() -> None:
    provider = DeterministicCachingProvider(
        provider=object.__new__(OpenAICompatibleProvider),
    )

    assert classify_builtin_provider(provider) is ProviderLocation.REMOTE


def test_cache_wrapping_local_provider_remains_local() -> None:
    provider = DeterministicCachingProvider(
        provider=local_adapter(),
    )

    assert classify_builtin_provider(provider) is ProviderLocation.LOCAL


def test_excessively_nested_cache_wrappers_fail_closed() -> None:
    provider: OpenAICompatibleProvider | DeterministicCachingProvider = (
        object.__new__(OpenAICompatibleProvider)
    )

    for _ in range(9):
        provider = DeterministicCachingProvider(provider=provider)

    with pytest.raises(
        ProviderDispatchPolicyError,
        match="nesting limit",
    ):
        classify_builtin_provider(provider)
