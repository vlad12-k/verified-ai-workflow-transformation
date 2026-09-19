"""Tests for the provider-neutral inference contract."""

import pytest

from vait.inference.providers.base import InferenceProvider
from vait.inference.providers.models import (
    InferenceMessage,
    InferenceRole,
    ProviderError,
    ProviderInferenceRequest,
    ProviderInferenceResponse,
    ProviderTiming,
    ProviderTokenUsage,
)


class FakeProvider:
    """Minimal provider implementation used to validate the protocol."""

    @property
    def provider_id(self) -> str:
        """Return fake provider identity."""
        return "fake-provider"

    @property
    def runtime_id(self) -> str:
        """Return fake runtime identity."""
        return "fake-runtime"

    def generate(
        self,
        request: ProviderInferenceRequest,
    ) -> ProviderInferenceResponse:
        """Return deterministic normalised inference evidence."""
        return ProviderInferenceResponse(
            request_id=request.request_id,
            provider_id=self.provider_id,
            runtime_id=self.runtime_id,
            model_id=request.model_id,
            model_revision="revision-1",
            output_text="verified response",
            usage=ProviderTokenUsage(
                input_tokens=12,
                output_tokens=3,
            ),
            timing=ProviderTiming(
                total_latency_ms=25.0,
                generation_latency_ms=20.0,
            ),
            cost_usd=0.001,
            metadata={
                "source": "test",
            },
        )


def invoke_provider(
    provider: InferenceProvider,
    request: ProviderInferenceRequest,
) -> ProviderInferenceResponse:
    """Exercise the provider through the portable protocol."""
    return provider.generate(request)


def test_provider_contract_normalises_successful_inference() -> None:
    """Concrete providers should expose one stable response shape."""
    request = ProviderInferenceRequest(
        request_id="request-1",
        model_id="portable-model",
        messages=(
            InferenceMessage(
                role=InferenceRole.SYSTEM,
                content="Use supplied evidence only.",
            ),
            InferenceMessage(
                role=InferenceRole.USER,
                content="What is the policy?",
            ),
        ),
        max_output_tokens=64,
    )

    response = invoke_provider(
        FakeProvider(),
        request,
    )

    assert response.succeeded is True
    assert response.provider_id == "fake-provider"
    assert response.runtime_id == "fake-runtime"
    assert response.output_text == "verified response"

    assert response.usage is not None
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 3
    assert response.usage.total_tokens == 15

    assert response.timing.total_latency_ms == 25.0
    assert response.timing.time_to_first_token_ms is None
    assert response.timing.generation_latency_ms == 20.0

    assert response.cost_usd == 0.001


def test_provider_contract_normalises_failure() -> None:
    """Provider failures should be data rather than SDK-specific exceptions."""
    response = ProviderInferenceResponse(
        request_id="request-2",
        provider_id="fake-provider",
        runtime_id="fake-runtime",
        model_id="portable-model",
        timing=ProviderTiming(
            total_latency_ms=10.0,
        ),
        error=ProviderError(
            error_type="rate_limit",
            message="Request rejected by provider.",
            retryable=True,
        ),
    )

    assert response.succeeded is False
    assert response.output_text is None
    assert response.error is not None
    assert response.error.error_type == "rate_limit"
    assert response.error.retryable is True


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens"),
    [
        (-1, 0),
        (0, -1),
    ],
)
def test_provider_usage_rejects_negative_token_counts(
    input_tokens: int,
    output_tokens: int,
) -> None:
    """Invalid usage evidence must not enter optimisation results."""
    with pytest.raises(ValueError):
        ProviderTokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


def test_provider_request_requires_messages() -> None:
    """Generative requests must contain at least one portable message."""
    with pytest.raises(ValueError):
        ProviderInferenceRequest(
            request_id="request-3",
            model_id="portable-model",
            messages=(),
            max_output_tokens=32,
        )
