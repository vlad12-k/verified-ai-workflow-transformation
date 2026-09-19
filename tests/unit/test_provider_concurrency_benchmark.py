"""Tests for provider-neutral concurrent inference benchmarking."""

from threading import Barrier, Lock

import pytest

from vait.inference.models import (
    InferenceConfiguration,
)
from vait.inference.providers.benchmark import (
    run_provider_concurrent_generative_benchmark,
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


def build_request() -> ProviderInferenceRequest:
    """Build one reusable provider-neutral request template."""
    return ProviderInferenceRequest(
        request_id="concurrency-test",
        model_id="test-model",
        messages=(
            InferenceMessage(
                role=InferenceRole.USER,
                content="Return one deterministic answer.",
            ),
        ),
        max_output_tokens=8,
        temperature=0.0,
    )


def build_configuration(
    *,
    batch_size: int = 1,
) -> InferenceConfiguration:
    """Build matching provider benchmark configuration."""
    return InferenceConfiguration(
        provider="test-provider",
        runtime="test-runtime",
        device="cpu",
        dtype="float32",
        batch_size=batch_size,
        model_id="test-model",
        model_revision="revision-1",
    )


class ConcurrentRecordingProvider:
    """Record requests and require overlapping provider execution."""

    provider_id = "test-provider"
    runtime_id = "test-runtime"

    def __init__(
        self,
        *,
        concurrency: int,
    ) -> None:
        """Configure one barrier per measured wave width."""
        self._barrier = Barrier(
            concurrency
        )
        self._lock = Lock()

        self.requests: list[
            ProviderInferenceRequest
        ] = []

    def generate(
        self,
        request: ProviderInferenceRequest,
    ) -> ProviderInferenceResponse:
        """Record one request after proving concurrent entry."""
        self._barrier.wait(
            timeout=2.0
        )

        with self._lock:
            self.requests.append(
                request
            )

        return ProviderInferenceResponse(
            request_id=request.request_id,
            provider_id=self.provider_id,
            runtime_id=self.runtime_id,
            model_id=request.model_id,
            model_revision="revision-1",
            output_text="answer",
            usage=ProviderTokenUsage(
                input_tokens=10,
                output_tokens=4,
            ),
            timing=ProviderTiming(
                total_latency_ms=5.0,
                generation_latency_ms=4.0,
            ),
        )


def test_provider_concurrent_bridge_records_normalised_evidence() -> None:
    """Concurrent provider bridge must preserve token evidence."""
    provider = ConcurrentRecordingProvider(
        concurrency=2
    )

    report = (
        run_provider_concurrent_generative_benchmark(
            provider=provider,
            request=build_request(),
            configuration=(
                build_configuration()
            ),
            concurrency=2,
            warmup_rounds=0,
            measured_rounds=1,
        )
    )

    assert len(
        provider.requests
    ) == 2

    assert (
        report.measured_iterations
        == 2
    )

    assert (
        report.generative
        is not None
    )

    assert (
        report.generative.input_tokens
        == 20
    )

    assert (
        report.generative.output_tokens
        == 8
    )

    assert (
        report.evidence_metadata[
            "provider_bridge"
        ]
        == "portable-concurrent-generative-v1"
    )

    assert (
        report.evidence_metadata[
            "concurrency"
        ]
        == 2
    )

    assert (
        report.evidence_metadata[
            "batch_size_fixed"
        ]
        == 1
    )


def test_provider_concurrent_bridge_uses_unique_request_ids() -> None:
    """Independent concurrent requests must never share request identity."""
    provider = ConcurrentRecordingProvider(
        concurrency=2
    )

    run_provider_concurrent_generative_benchmark(
        provider=provider,
        request=build_request(),
        configuration=(
            build_configuration()
        ),
        concurrency=2,
        warmup_rounds=0,
        measured_rounds=1,
    )

    request_ids = [
        request.request_id
        for request
        in provider.requests
    ]

    assert len(
        request_ids
    ) == 2

    assert len(
        set(
            request_ids
        )
    ) == 2

    assert all(
        request_id.startswith(
            "concurrency-test-concurrent-"
        )
        for request_id
        in request_ids
    )


def test_provider_concurrent_bridge_records_worker_metadata() -> None:
    """Concurrent request copies must expose worker execution identity."""
    provider = ConcurrentRecordingProvider(
        concurrency=2
    )

    run_provider_concurrent_generative_benchmark(
        provider=provider,
        request=build_request(),
        configuration=(
            build_configuration()
        ),
        concurrency=2,
        warmup_rounds=0,
        measured_rounds=1,
    )

    worker_indices = {
        request.metadata[
            "concurrency_worker_index"
        ]
        for request
        in provider.requests
    }

    invocation_indices = {
        request.metadata[
            "concurrency_invocation_index"
        ]
        for request
        in provider.requests
    }

    assert worker_indices == {
        0,
        1,
    }

    assert invocation_indices == {
        0,
        1,
    }


class WrongRequestIdProvider:
    """Return a response that cannot be correlated to its request."""

    provider_id = "test-provider"
    runtime_id = "test-runtime"

    def generate(
        self,
        request: ProviderInferenceRequest,
    ) -> ProviderInferenceResponse:
        """Return deliberately invalid request identity."""
        return ProviderInferenceResponse(
            request_id="wrong-request-id",
            provider_id=self.provider_id,
            runtime_id=self.runtime_id,
            model_id=request.model_id,
            model_revision="revision-1",
            output_text="answer",
            usage=ProviderTokenUsage(
                input_tokens=1,
                output_tokens=1,
            ),
            timing=ProviderTiming(
                total_latency_ms=1.0,
            ),
        )


def test_provider_concurrent_bridge_rejects_request_id_drift() -> None:
    """Concurrent evidence must remain correlated to exact requests."""
    with pytest.raises(
        RuntimeError,
        match=(
            "response request_id does not match"
        ),
    ):
        run_provider_concurrent_generative_benchmark(
            provider=(
                WrongRequestIdProvider()
            ),
            request=build_request(),
            configuration=(
                build_configuration()
            ),
            concurrency=1,
            warmup_rounds=0,
            measured_rounds=1,
        )


class FailingProvider:
    """Return one normalised provider failure."""

    provider_id = "test-provider"
    runtime_id = "test-runtime"

    def generate(
        self,
        request: ProviderInferenceRequest,
    ) -> ProviderInferenceResponse:
        """Return deterministic failure evidence."""
        return ProviderInferenceResponse(
            request_id=request.request_id,
            provider_id=self.provider_id,
            runtime_id=self.runtime_id,
            model_id=request.model_id,
            model_revision="revision-1",
            timing=ProviderTiming(
                total_latency_ms=1.0,
            ),
            error=ProviderError(
                error_type="SyntheticFailure",
                message="synthetic provider failure",
                retryable=False,
            ),
        )


def test_provider_concurrent_bridge_surfaces_provider_failure() -> None:
    """Provider failures must invalidate concurrent performance evidence."""
    with pytest.raises(
        RuntimeError,
        match="synthetic provider failure",
    ):
        run_provider_concurrent_generative_benchmark(
            provider=FailingProvider(),
            request=build_request(),
            configuration=(
                build_configuration()
            ),
            concurrency=1,
            warmup_rounds=0,
            measured_rounds=1,
        )


def test_provider_concurrent_bridge_rejects_batching_mix() -> None:
    """Concurrency benchmark must keep native batching separate."""
    provider = ConcurrentRecordingProvider(
        concurrency=1
    )

    with pytest.raises(
        ValueError,
        match="batch_size=1",
    ):
        run_provider_concurrent_generative_benchmark(
            provider=provider,
            request=build_request(),
            configuration=(
                build_configuration(
                    batch_size=2
                )
            ),
            concurrency=1,
            warmup_rounds=0,
            measured_rounds=1,
        )
