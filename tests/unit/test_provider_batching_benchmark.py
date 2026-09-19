"""Tests for provider-neutral native-batch benchmarking."""

from collections.abc import Sequence

import pytest

from vait.inference.models import (
    InferenceConfiguration,
)
from vait.inference.providers.benchmark import (
    run_provider_native_batch_generative_benchmark,
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
    """Build one provider request template."""
    return ProviderInferenceRequest(
        request_id="native-batch-test",
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
    batch_size: int = 2,
) -> InferenceConfiguration:
    """Build one native-batch provider configuration."""
    return InferenceConfiguration(
        provider="test-provider",
        runtime="test-runtime",
        device="cpu",
        dtype="float32",
        batch_size=batch_size,
        model_id="test-model",
        model_revision="revision-1",
    )


class NativeBatchRecordingProvider:
    """Record genuine native-batch provider invocations."""

    provider_id = "test-provider"
    runtime_id = "test-runtime"

    def __init__(self) -> None:
        """Track every native batch."""
        self.batches: list[
            tuple[
                ProviderInferenceRequest,
                ...,
            ]
        ] = []

    def generate(
        self,
        request: ProviderInferenceRequest,
    ) -> ProviderInferenceResponse:
        """Fail if scalar generation is accidentally used."""
        del request

        raise AssertionError(
            "Scalar generate() must not be used "
            "by native-batch benchmarking."
        )

    def generate_batch(
        self,
        requests: Sequence[
            ProviderInferenceRequest
        ],
    ) -> tuple[
        ProviderInferenceResponse,
        ...,
    ]:
        """Return one response for every native-batch request."""
        batch = tuple(
            requests
        )

        self.batches.append(
            batch
        )

        return tuple(
            ProviderInferenceResponse(
                request_id=(
                    request.request_id
                ),
                provider_id=(
                    self.provider_id
                ),
                runtime_id=(
                    self.runtime_id
                ),
                model_id=(
                    request.model_id
                ),
                model_revision=(
                    "revision-1"
                ),
                output_text="answer",
                usage=(
                    ProviderTokenUsage(
                        input_tokens=10,
                        output_tokens=4,
                    )
                ),
                timing=(
                    ProviderTiming(
                        total_latency_ms=5.0,
                        generation_latency_ms=4.0,
                    )
                ),
            )
            for request
            in batch
        )


def test_provider_native_batch_bridge_records_request_evidence() -> None:
    """One batch invocation must represent multiple measured requests."""
    provider = (
        NativeBatchRecordingProvider()
    )

    report = (
        run_provider_native_batch_generative_benchmark(
            provider=provider,
            request=build_request(),
            configuration=(
                build_configuration(
                    batch_size=2
                )
            ),
            warmup_rounds=0,
            measured_rounds=2,
        )
    )

    assert len(
        provider.batches
    ) == 2

    assert all(
        len(batch) == 2
        for batch
        in provider.batches
    )

    assert (
        report.measured_iterations
        == 2
    )

    assert (
        report.generative
        is not None
    )

    assert (
        report.generative.sample_count
        == 4
    )

    assert (
        report.generative.input_tokens
        == 40
    )

    assert (
        report.generative.output_tokens
        == 16
    )

    assert (
        report.evidence_metadata[
            "measured_requests"
        ]
        == 4
    )

    assert (
        report.evidence_metadata[
            "provider_bridge"
        ]
        == "portable-native-batch-generative-v1"
    )

    assert (
        report.evidence_metadata[
            "execution_mode"
        ]
        == "native-generative-batch"
    )


def test_provider_native_batch_bridge_uses_unique_request_ids() -> None:
    """Every request across repeated native batches must remain unique."""
    provider = (
        NativeBatchRecordingProvider()
    )

    run_provider_native_batch_generative_benchmark(
        provider=provider,
        request=build_request(),
        configuration=(
            build_configuration(
                batch_size=2
            )
        ),
        warmup_rounds=0,
        measured_rounds=2,
    )

    request_ids = [
        request.request_id
        for batch
        in provider.batches
        for request
        in batch
    ]

    assert len(
        request_ids
    ) == 4

    assert len(
        set(
            request_ids
        )
    ) == 4

    assert all(
        request_id.startswith(
            "native-batch-test-batch-"
        )
        for request_id
        in request_ids
    )


def test_provider_native_batch_bridge_records_batch_metadata() -> None:
    """Request copies must expose batch invocation and item identity."""
    provider = (
        NativeBatchRecordingProvider()
    )

    run_provider_native_batch_generative_benchmark(
        provider=provider,
        request=build_request(),
        configuration=(
            build_configuration(
                batch_size=2
            )
        ),
        warmup_rounds=0,
        measured_rounds=1,
    )

    batch = provider.batches[
        0
    ]

    assert {
        request.metadata[
            "native_batch_item_index"
        ]
        for request
        in batch
    } == {
        0,
        1,
    }

    assert {
        request.metadata[
            "native_batch_invocation_index"
        ]
        for request
        in batch
    } == {
        0,
    }

    assert all(
        request.metadata[
            "native_batch_size"
        ]
        == 2
        for request
        in batch
    )


class WrongCountProvider(
    NativeBatchRecordingProvider
):
    """Return fewer responses than native-batch requests."""

    def generate_batch(
        self,
        requests: Sequence[
            ProviderInferenceRequest
        ],
    ) -> tuple[
        ProviderInferenceResponse,
        ...,
    ]:
        """Return only one response regardless of batch size."""
        responses = super().generate_batch(
            requests
        )

        return responses[
            :1
        ]


def test_provider_native_batch_bridge_rejects_response_count_drift() -> None:
    """Incomplete native-batch results must invalidate the benchmark."""
    with pytest.raises(
        RuntimeError,
        match=(
            "returned 1 responses "
            "for batch_size=2"
        ),
    ):
        run_provider_native_batch_generative_benchmark(
            provider=(
                WrongCountProvider()
            ),
            request=build_request(),
            configuration=(
                build_configuration(
                    batch_size=2
                )
            ),
            warmup_rounds=0,
            measured_rounds=1,
        )


class WrongRequestIdProvider(
    NativeBatchRecordingProvider
):
    """Return an uncorrelated response identity."""

    def generate_batch(
        self,
        requests: Sequence[
            ProviderInferenceRequest
        ],
    ) -> tuple[
        ProviderInferenceResponse,
        ...,
    ]:
        """Change the first response request ID."""
        responses = list(
            super().generate_batch(
                requests
            )
        )

        responses[
            0
        ] = responses[
            0
        ].model_copy(
            update={
                "request_id": (
                    "wrong-request-id"
                )
            }
        )

        return tuple(
            responses
        )


def test_provider_native_batch_bridge_rejects_request_id_drift() -> None:
    """Native-batch responses must correlate exactly to requests."""
    with pytest.raises(
        RuntimeError,
        match=(
            "request_id does not match"
        ),
    ):
        run_provider_native_batch_generative_benchmark(
            provider=(
                WrongRequestIdProvider()
            ),
            request=build_request(),
            configuration=(
                build_configuration(
                    batch_size=2
                )
            ),
            warmup_rounds=0,
            measured_rounds=1,
        )


class FailingNativeBatchProvider(
    NativeBatchRecordingProvider
):
    """Return normalised provider failures."""

    def generate_batch(
        self,
        requests: Sequence[
            ProviderInferenceRequest
        ],
    ) -> tuple[
        ProviderInferenceResponse,
        ...,
    ]:
        """Return one failure response per request."""
        batch = tuple(
            requests
        )

        self.batches.append(
            batch
        )

        return tuple(
            ProviderInferenceResponse(
                request_id=(
                    request.request_id
                ),
                provider_id=(
                    self.provider_id
                ),
                runtime_id=(
                    self.runtime_id
                ),
                model_id=(
                    request.model_id
                ),
                model_revision=(
                    "revision-1"
                ),
                timing=(
                    ProviderTiming(
                        total_latency_ms=1.0,
                    )
                ),
                error=ProviderError(
                    error_type=(
                        "SyntheticFailure"
                    ),
                    message=(
                        "synthetic native batch failure"
                    ),
                    retryable=False,
                ),
            )
            for request
            in batch
        )


def test_provider_native_batch_bridge_surfaces_provider_failure() -> None:
    """Provider failures must invalidate native-batch evidence."""
    with pytest.raises(
        RuntimeError,
        match=(
            "synthetic native batch failure"
        ),
    ):
        run_provider_native_batch_generative_benchmark(
            provider=(
                FailingNativeBatchProvider()
            ),
            request=build_request(),
            configuration=(
                build_configuration(
                    batch_size=2
                )
            ),
            warmup_rounds=0,
            measured_rounds=1,
        )
