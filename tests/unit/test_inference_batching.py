"""Tests for native-batch generative inference benchmarking."""

import pytest

from vait.inference.batching import (
    run_controlled_native_generative_batch_benchmark,
)
from vait.inference.models import (
    GenerativeInferenceSample,
    InferenceConfiguration,
)


def build_configuration(
    *,
    batch_size: int = 2,
) -> InferenceConfiguration:
    """Build one native-batch test configuration."""
    return InferenceConfiguration(
        provider="test-provider",
        runtime="test-runtime",
        device="cpu",
        dtype="float32",
        batch_size=batch_size,
        model_id="test-model",
        model_revision="revision-1",
    )


def test_native_generative_batch_records_batch_and_request_evidence() -> None:
    """Native batching must distinguish batch calls from requests."""
    calls: list[
        int
    ] = []

    def operation() -> tuple[
        GenerativeInferenceSample,
        ...,
    ]:
        calls.append(
            1
        )

        return (
            GenerativeInferenceSample(
                generation_latency_ms=5.0,
                input_tokens=10,
                output_tokens=4,
            ),
            GenerativeInferenceSample(
                generation_latency_ms=5.0,
                input_tokens=11,
                output_tokens=5,
            ),
        )

    report = (
        run_controlled_native_generative_batch_benchmark(
            candidate_implementation_id=(
                "native-batch-v1"
            ),
            operation=operation,
            task_family=(
                "provider-generative-inference"
            ),
            configuration=(
                build_configuration()
            ),
            warmup_rounds=1,
            measured_rounds=3,
        )
    )

    assert len(
        calls
    ) == 4

    assert (
        report.warmup_iterations
        == 1
    )

    assert (
        report.measured_iterations
        == 3
    )

    assert (
        report.latency.count
        == 3
    )

    assert (
        report.generative
        is not None
    )

    assert (
        report.generative.sample_count
        == 6
    )

    assert (
        report.generative.input_tokens
        == 63
    )

    assert (
        report.generative.output_tokens
        == 27
    )

    assert (
        report.throughput.requests_per_second
        is not None
    )

    assert (
        report.throughput.requests_per_second
        > 0.0
    )

    assert (
        report.throughput.tokens_per_second
        is not None
    )

    assert (
        report.throughput.tokens_per_second
        > 0.0
    )

    assert (
        report.evidence_metadata[
            "execution_mode"
        ]
        == "native-generative-batch"
    )

    assert (
        report.evidence_metadata[
            "batch_size"
        ]
        == 2
    )

    assert (
        report.evidence_metadata[
            "measured_batch_invocations"
        ]
        == 3
    )

    assert (
        report.evidence_metadata[
            "measured_requests"
        ]
        == 6
    )

    assert (
        report.evidence_metadata[
            "batch_latency_observations"
        ]
        == 3
    )

    request_equivalent_latency = (
        report.evidence_metadata[
            "request_equivalent_mean_latency_ms"
        ]
    )

    assert isinstance(
        request_equivalent_latency,
        int | float,
    )

    assert (
        float(
            request_equivalent_latency
        )
        >= 0.0
    )

    assert (
        report.evidence_metadata[
            "concurrency_varied"
        ]
        is False
    )


def test_native_generative_batch_supports_batch_size_one_baseline() -> None:
    """Batch-size one remains a valid native-batch scaling baseline."""

    def operation() -> tuple[
        GenerativeInferenceSample,
        ...,
    ]:
        return (
            GenerativeInferenceSample(
                input_tokens=10,
                output_tokens=2,
            ),
        )

    report = (
        run_controlled_native_generative_batch_benchmark(
            candidate_implementation_id=(
                "native-batch-one-v1"
            ),
            operation=operation,
            task_family=(
                "provider-generative-inference"
            ),
            configuration=(
                build_configuration(
                    batch_size=1
                )
            ),
            warmup_rounds=0,
            measured_rounds=2,
        )
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
        == 2
    )

    assert (
        report.evidence_metadata[
            "measured_requests"
        ]
        == 2
    )


def test_native_generative_batch_rejects_wrong_sample_count() -> None:
    """Batch evidence must match the declared native batch size."""

    def operation() -> tuple[
        GenerativeInferenceSample,
        ...,
    ]:
        return (
            GenerativeInferenceSample(
                input_tokens=1,
                output_tokens=1,
            ),
        )

    with pytest.raises(
        RuntimeError,
        match=(
            "returned 1 samples "
            "for configured batch_size=2"
        ),
    ):
        run_controlled_native_generative_batch_benchmark(
            candidate_implementation_id=(
                "wrong-size-v1"
            ),
            operation=operation,
            task_family=(
                "provider-generative-inference"
            ),
            configuration=(
                build_configuration(
                    batch_size=2
                )
            ),
            warmup_rounds=0,
            measured_rounds=1,
        )


@pytest.mark.parametrize(
    (
        "warmup_rounds",
        "measured_rounds",
    ),
    [
        (-1, 1),
        (0, 0),
    ],
)
def test_native_generative_batch_rejects_invalid_rounds(
    warmup_rounds: int,
    measured_rounds: int,
) -> None:
    """Invalid native-batch execution plans must fail immediately."""

    def operation() -> tuple[
        GenerativeInferenceSample,
        ...,
    ]:
        return (
            GenerativeInferenceSample(
                input_tokens=1,
                output_tokens=1,
            ),
        )

    with pytest.raises(
        ValueError
    ):
        run_controlled_native_generative_batch_benchmark(
            candidate_implementation_id=(
                "invalid-batch-plan-v1"
            ),
            operation=operation,
            task_family=(
                "provider-generative-inference"
            ),
            configuration=(
                build_configuration(
                    batch_size=1
                )
            ),
            warmup_rounds=(
                warmup_rounds
            ),
            measured_rounds=(
                measured_rounds
            ),
        )


def test_native_generative_batch_surfaces_operation_failure() -> None:
    """A failed native batch must invalidate performance evidence."""

    def operation() -> tuple[
        GenerativeInferenceSample,
        ...,
    ]:
        raise RuntimeError(
            "synthetic batch failure"
        )

    with pytest.raises(
        RuntimeError,
        match="synthetic batch failure",
    ):
        run_controlled_native_generative_batch_benchmark(
            candidate_implementation_id=(
                "failing-native-batch-v1"
            ),
            operation=operation,
            task_family=(
                "provider-generative-inference"
            ),
            configuration=(
                build_configuration(
                    batch_size=1
                )
            ),
            warmup_rounds=0,
            measured_rounds=1,
        )
