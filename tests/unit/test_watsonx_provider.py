"""Tests for the provider-neutral watsonx enterprise adapter."""

from collections.abc import Mapping

from pydantic import JsonValue

from vait.inference.providers.models import (
    InferenceMessage,
    InferenceRole,
    ProviderInferenceRequest,
)
from vait.inference.providers.watsonx import (
    WatsonxProvider,
    WatsonxTransportResponse,
)


class FakeWatsonxTransport:
    """Deterministic watsonx transport double."""

    def __init__(
        self,
        *,
        response: WatsonxTransportResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        """Store configured result or transport failure."""
        self.response = response
        self.error = error

        self.calls: list[
            tuple[
                str,
                tuple[dict[str, str], ...],
                int,
                float,
                dict[str, JsonValue],
            ]
        ] = []

    def generate(
        self,
        *,
        model_id: str,
        messages: tuple[dict[str, str], ...],
        max_output_tokens: int,
        temperature: float,
        metadata: Mapping[str, JsonValue],
    ) -> WatsonxTransportResponse:
        """Record request evidence and return configured result."""
        self.calls.append(
            (
                model_id,
                messages,
                max_output_tokens,
                temperature,
                dict(metadata),
            )
        )

        if self.error is not None:
            raise self.error

        if self.response is None:
            raise RuntimeError(
                "Fake watsonx transport has no response."
            )

        return self.response


def build_request(
    *,
    model_id: str = "ibm/test-model",
) -> ProviderInferenceRequest:
    """Build one portable watsonx test request."""
    return ProviderInferenceRequest(
        request_id="watsonx-request-1",
        model_id=model_id,
        messages=(
            InferenceMessage(
                role=InferenceRole.SYSTEM,
                content="Use supplied evidence only.",
            ),
            InferenceMessage(
                role=InferenceRole.USER,
                content="Return the verified result.",
            ),
        ),
        max_output_tokens=48,
        temperature=0.0,
        metadata={
            "experiment": "m4-b4",
        },
    )


def test_watsonx_provider_normalises_success() -> None:
    """watsonx output should map into the shared provider response."""
    transport = FakeWatsonxTransport(
        response=WatsonxTransportResponse(
            model_id="ibm/test-model",
            model_revision="revision-7",
            output_text="verified watsonx response",
            input_tokens=21,
            output_tokens=5,
            provider_request_id="wx-request-123",
            stop_reason="eos",
            metadata={
                "region": "test-region",
            },
        )
    )

    provider = WatsonxProvider(
        model_id="ibm/test-model",
        transport=transport,
        deployment_id="deployment-1",
    )

    response = provider.generate(
        build_request()
    )

    assert response.succeeded is True
    assert response.provider_id == "watsonx"
    assert response.runtime_id == (
        "watsonx-enterprise-adapter"
    )

    assert response.model_id == "ibm/test-model"
    assert response.model_revision == "revision-7"
    assert response.output_text == (
        "verified watsonx response"
    )

    assert response.usage is not None
    assert response.usage.input_tokens == 21
    assert response.usage.output_tokens == 5

    assert response.timing.total_latency_ms >= 0.0
    assert response.timing.time_to_first_token_ms is None
    assert response.timing.generation_latency_ms is None

    assert response.cost_usd is None

    assert response.metadata["deployment_id"] == (
        "deployment-1"
    )
    assert response.metadata["provider_request_id"] == (
        "wx-request-123"
    )
    assert response.metadata["stop_reason"] == "eos"
    assert response.metadata["region"] == "test-region"

    assert len(transport.calls) == 1

    (
        model_id,
        messages,
        max_output_tokens,
        temperature,
        metadata,
    ) = transport.calls[0]

    assert model_id == "ibm/test-model"
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert max_output_tokens == 48
    assert temperature == 0.0
    assert metadata["experiment"] == "m4-b4"
    assert metadata["deployment_id"] == "deployment-1"


def test_watsonx_provider_allows_missing_usage() -> None:
    """Token usage may remain unavailable without invalidating output."""
    transport = FakeWatsonxTransport(
        response=WatsonxTransportResponse(
            model_id="ibm/test-model",
            output_text="response without usage",
        )
    )

    provider = WatsonxProvider(
        model_id="ibm/test-model",
        transport=transport,
    )

    response = provider.generate(
        build_request()
    )

    assert response.succeeded is True
    assert response.output_text == (
        "response without usage"
    )
    assert response.usage is None


def test_watsonx_provider_normalises_transport_failure() -> None:
    """IBM-specific failures must not leak through the portable boundary."""
    transport = FakeWatsonxTransport(
        error=TimeoutError(
            "watsonx request timed out"
        )
    )

    provider = WatsonxProvider(
        model_id="ibm/test-model",
        transport=transport,
    )

    response = provider.generate(
        build_request()
    )

    assert response.succeeded is False
    assert response.error is not None
    assert response.error.error_type == (
        "watsonx_transport_error"
    )
    assert response.error.message == (
        "watsonx request timed out"
    )
    assert response.error.retryable is True


def test_watsonx_provider_rejects_model_mismatch() -> None:
    """Pinned model identity must not silently change."""
    transport = FakeWatsonxTransport(
        response=WatsonxTransportResponse(
            model_id="ibm/test-model",
            output_text="unused",
        )
    )

    provider = WatsonxProvider(
        model_id="ibm/test-model",
        transport=transport,
    )

    response = provider.generate(
        build_request(
            model_id="ibm/different-model",
        )
    )

    assert response.succeeded is False
    assert response.error is not None
    assert response.error.error_type == (
        "model_mismatch"
    )
    assert transport.calls == []


def test_watsonx_provider_keeps_cost_unknown() -> None:
    """Provider adapter must not invent pricing evidence."""
    transport = FakeWatsonxTransport(
        response=WatsonxTransportResponse(
            model_id="ibm/test-model",
            output_text="verified response",
            input_tokens=10,
            output_tokens=2,
        )
    )

    provider = WatsonxProvider(
        model_id="ibm/test-model",
        transport=transport,
    )

    response = provider.generate(
        build_request()
    )

    assert response.succeeded is True
    assert response.cost_usd is None
