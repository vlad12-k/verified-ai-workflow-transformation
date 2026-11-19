"""Fail-closed identification of built-in inference adapters."""

from typing import cast

from vait.inference.providers.base import InferenceProvider
from vait.inference.providers.cache import DeterministicCachingProvider
from vait.inference.providers.openai_compatible import OpenAICompatibleProvider
from vait.inference.providers.watsonx import WatsonxProvider
from vait.platform.provider_dispatch_policy import (
    ProviderDispatchPolicyError,
    ProviderLocation,
)


def classify_builtin_provider(
    provider: InferenceProvider,
) -> ProviderLocation:
    """Classify exact built-in types, including bounded cache wrappers."""
    current = provider

    for _ in range(8):
        adapter_type = type(current)

        if adapter_type is DeterministicCachingProvider:
            cache = cast(DeterministicCachingProvider, current)
            current = cache._provider
            continue

        if adapter_type in (OpenAICompatibleProvider, WatsonxProvider):
            return ProviderLocation.REMOTE

        if (
            adapter_type.__module__
            == "vait.inference.providers.huggingface"
        ):
            # Load the optional heavy local adapter only when needed.
            from vait.inference.providers.huggingface import (
                LocalHuggingFaceProvider,
            )

            if adapter_type is LocalHuggingFaceProvider:
                local = cast(LocalHuggingFaceProvider, current)

                if not local._local_files_only:
                    raise ProviderDispatchPolicyError(
                        "Local adapter requires local_files_only=True."
                    )

                return ProviderLocation.LOCAL

        raise ProviderDispatchPolicyError(
            "Unknown provider adapter: dispatch denied."
        )

    raise ProviderDispatchPolicyError(
        "Provider cache nesting limit exceeded."
    )
