"""Tests for the provider-neutral benchmark bridge."""

import pytest

from vait.inference.models import (
    InferenceConfiguration,
)
from vait.inference.providers.benchmark import (
    run_provider_generative_benchmark,
)
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
    """Deterministic provider used to exercise the benchmark bridge."""

    def __init__(
        self,
        *,
        response: ProviderInferenceResponse,
    ) -> None:
        """Store one portable response."""
        self.response = response
        self.calls = 0

    @property
    def provider_id(self) -> str:
        """Return stable test provider identity."""
        return "fake-provider"

    @property
    def runtime_id(self) -> str:
        """Return stable test runtime identity."""
        return "fake-runtime"

    def generate(
        self,
        request: ProviderInferenceRequest,
    ) -> ProviderInferenceResponse:
        """Return deterministic response evidence."""
        self.calls += 1

        return self.response.model_copy(
            update={
                "request_id": request.request_id,
            }
        )


def build_request() -> ProviderInferenceRequest:
    """Build one portable provider benchmark request."""
    return ProviderInferenceRequest(
        request_id="provider-benchmark-request",
        model_id="portable-model",
        messages=(
            InferenceMessage(
                role=InferenceRole.USER,
                content="Return the verified result.",
            ),
        ),
        max_output_tokens=16,
        temperature=0.0,
    )


def build_configuration(
    *,
    provider: str = "fake-provider",
    runtime: str = "fake-runtime",
    model_id: str = "portable-model",
) -> InferenceConfiguration:
    """Build matching inference configuration."""
    return InferenceConfiguration(
        provider=provider,
        runtime=runtime,
        device="remote",
        dtype="provider-managed",
        batch_size=1,
        model_id=model_id,
        model_revision=None,
    )


def test_provider_bridge_uses_shared_generative_harness() -> None:
    """Portable provider evidence should enter the common M4 report shape."""
    provider = FakeProvider(
        response=ProviderInferenceResponse(
            request_id="placeholder",
            provider_id="fake-provider",
            runtime_id="fake-runtime",
            model_id="portable-model",
            output_text="verified response",
            usage=ProviderTokenUsage(
                input_tokens=10,
                output_tokens=4,
            ),
            timing=ProviderTiming(
                total_latency_ms=12.0,
                time_to_first_token_ms=3.0,
                generation_latency_ms=8.0,
            ),
        )
    )

    report = run_provider_generative_benchmark(
        provider=provider,
        request=build_request(),
        configuration=build_configuration(),
        warmup_iterations=2,
        measured_iterations=3,
    )

    assert provider.calls == 5

    assert report.task_family == (
        "provider-generative-inference"
    )
    assert report.warmup_iterations == 2
    assert report.measured_iterations == 3

    assert report.generative is not None
    assert report.generative.sample_count == 3

    assert report.generative.input_tokens == 30
    assert report.generative.output_tokens == 12

    assert report.generative.time_to_first_token is not None
    assert (
        report.generative.time_to_first_token.p50_ms
        == 3.0
    )

    assert report.generative.generation_latency is not None
    assert (
        report.generative.generation_latency.p50_ms
        == 8.0
    )

    assert report.throughput.requests_per_second is not None
    assert report.throughput.requests_per_second > 0.0

    assert report.throughput.tokens_per_second is not None
    assert report.throughput.tokens_per_second > 0.0

    assert (
        report.evidence_metadata["provider_id"]
        == "fake-provider"
    )
    assert (
        report.evidence_metadata["runtime_id"]
        == "fake-runtime"
    )
    assert (
        report.evidence_metadata["provider_model_id"]
        == "portable-model"
    )
    assert (
        report.evidence_metadata["token_usage_required"]
        is True
    )

    assert report.candidate_implementation_id == (
        "provider:fake-provider:"
        "runtime:fake-runtime:"
        "model:portable-model"
    )


def test_provider_bridge_rejects_provider_failure() -> None:
    """Failed provider calls must not become performance evidence."""
    provider = FakeProvider(
        response=ProviderInferenceResponse(
            request_id="placeholder",
            provider_id="fake-provider",
            runtime_id="fake-runtime",
            model_id="portable-model",
            timing=ProviderTiming(
                total_latency_ms=5.0,
            ),
            error=ProviderError(
                error_type="rate_limit",
                message="Request rejected.",
                retryable=True,
            ),
        )
    )

    with pytest.raises(
        RuntimeError,
        match="rate_limit",
    ):
        run_provider_generative_benchmark(
            provider=provider,
            request=build_request(),
            configuration=build_configuration(),
            warmup_iterations=0,
            measured_iterations=1,
        )


def test_provider_bridge_rejects_missing_token_usage() -> None:
    """Missing usage must not be silently represented as zero tokens."""
    provider = FakeProvider(
        response=ProviderInferenceResponse(
            request_id="placeholder",
            provider_id="fake-provider",
            runtime_id="fake-runtime",
            model_id="portable-model",
            output_text="response without usage",
            timing=ProviderTiming(
                total_latency_ms=5.0,
            ),
        )
    )

    with pytest.raises(
        RuntimeError,
        match="did not include token usage",
    ):
        run_provider_generative_benchmark(
            provider=provider,
            request=build_request(),
            configuration=build_configuration(),
            warmup_iterations=0,
            measured_iterations=1,
        )


@pytest.mark.parametrize(
    ("provider", "runtime", "model_id"),
    [
        (
            "wrong-provider",
            "fake-runtime",
            "portable-model",
        ),
        (
            "fake-provider",
            "wrong-runtime",
            "portable-model",
        ),
        (
            "fake-provider",
            "fake-runtime",
            "wrong-model",
        ),
    ],
)
def test_provider_bridge_rejects_contradictory_configuration(
    provider: str,
    runtime: str,
    model_id: str,
) -> None:
    """Configuration identity must agree with the provider boundary."""
    fake_provider = FakeProvider(
        response=ProviderInferenceResponse(
            request_id="placeholder",
            provider_id="fake-provider",
            runtime_id="fake-runtime",
            model_id="portable-model",
            output_text="unused",
            usage=ProviderTokenUsage(
                input_tokens=1,
                output_tokens=1,
            ),
            timing=ProviderTiming(
                total_latency_ms=1.0,
            ),
        )
    )

    with pytest.raises(ValueError):
        run_provider_generative_benchmark(
            provider=fake_provider,
            request=build_request(),
            configuration=build_configuration(
                provider=provider,
                runtime=runtime,
                model_id=model_id,
            ),
            warmup_iterations=0,
            measured_iterations=1,
        )
