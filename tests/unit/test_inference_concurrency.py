"""Tests for controlled concurrent generative benchmarking."""

from threading import Barrier

import pytest

from vait.inference.concurrency import (
    run_controlled_concurrent_generative_benchmark,
)
from vait.inference.models import (
    GenerativeInferenceSample,
    InferenceConfiguration,
)


def build_configuration(
    *,
    batch_size: int = 1,
) -> InferenceConfiguration:
    """Build one controlled concurrent test configuration."""
    return InferenceConfiguration(
        provider="test-provider",
        runtime="test-runtime",
        device="cpu",
        dtype="float32",
        batch_size=batch_size,
        model_id="test-model",
        model_revision="revision-1",
    )


def test_concurrent_benchmark_records_request_and_token_evidence() -> None:
    """Concurrent execution must preserve usage and throughput evidence."""
    calls: list[
        int
    ] = []

    def operation(
        request_index: int,
    ) -> GenerativeInferenceSample:
        calls.append(
            request_index
        )

        return GenerativeInferenceSample(
            time_to_first_token_ms=None,
            generation_latency_ms=5.0,
            input_tokens=10,
            output_tokens=4,
        )

    report = (
        run_controlled_concurrent_generative_benchmark(
            candidate_implementation_id=(
                "concurrent-generator-v1"
            ),
            operation=operation,
            task_family=(
                "provider-generative-inference"
            ),
            configuration=(
                build_configuration()
            ),
            concurrency=2,
            warmup_rounds=1,
            measured_rounds=3,
        )
    )

    assert len(
        calls
    ) == 8

    assert (
        report.warmup_iterations
        == 2
    )

    assert (
        report.measured_iterations
        == 6
    )

    assert (
        report.latency.count
        == 6
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
        == 60
    )

    assert (
        report.generative.output_tokens
        == 24
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
        == "concurrent-generative"
    )

    assert (
        report.evidence_metadata[
            "concurrency"
        ]
        == 2
    )

    assert (
        report.evidence_metadata[
            "requests_per_wave"
        ]
        == 2
    )

    assert (
        report.evidence_metadata[
            "measured_requests"
        ]
        == 6
    )

    assert (
        report.evidence_metadata[
            "batch_size_fixed"
        ]
        == 1
    )


def test_concurrent_benchmark_executes_real_overlap() -> None:
    """A two-request wave must permit both workers to run concurrently."""
    barrier = Barrier(
        2
    )

    def operation(
        request_index: int,
    ) -> GenerativeInferenceSample:
        del request_index

        barrier.wait(
            timeout=2.0
        )

        return GenerativeInferenceSample(
            input_tokens=1,
            output_tokens=1,
        )

    report = (
        run_controlled_concurrent_generative_benchmark(
            candidate_implementation_id=(
                "overlap-generator-v1"
            ),
            operation=operation,
            task_family=(
                "provider-generative-inference"
            ),
            configuration=(
                build_configuration()
            ),
            concurrency=2,
            warmup_rounds=0,
            measured_rounds=1,
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


@pytest.mark.parametrize(
    (
        "concurrency",
        "warmup_rounds",
        "measured_rounds",
    ),
    [
        (0, 0, 1),
        (-1, 0, 1),
        (1, -1, 1),
        (1, 0, 0),
    ],
)
def test_concurrent_benchmark_rejects_invalid_execution_plan(
    concurrency: int,
    warmup_rounds: int,
    measured_rounds: int,
) -> None:
    """Invalid concurrency plans must fail before measurement."""

    def operation(
        request_index: int,
    ) -> GenerativeInferenceSample:
        del request_index

        return GenerativeInferenceSample(
            input_tokens=1,
            output_tokens=1,
        )

    with pytest.raises(
        ValueError
    ):
        run_controlled_concurrent_generative_benchmark(
            candidate_implementation_id=(
                "invalid-plan-v1"
            ),
            operation=operation,
            task_family=(
                "provider-generative-inference"
            ),
            configuration=(
                build_configuration()
            ),
            concurrency=(
                concurrency
            ),
            warmup_rounds=(
                warmup_rounds
            ),
            measured_rounds=(
                measured_rounds
            ),
        )


def test_concurrent_benchmark_keeps_batching_separate() -> None:
    """Concurrency evidence must not silently mix native batching."""

    def operation(
        request_index: int,
    ) -> GenerativeInferenceSample:
        del request_index

        return GenerativeInferenceSample(
            input_tokens=1,
            output_tokens=1,
        )

    with pytest.raises(
        ValueError,
        match="batch_size=1",
    ):
        run_controlled_concurrent_generative_benchmark(
            candidate_implementation_id=(
                "batched-concurrent-v1"
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
            concurrency=2,
            warmup_rounds=0,
            measured_rounds=1,
        )


def test_concurrent_benchmark_surfaces_worker_failure() -> None:
    """A failed concurrent request must invalidate the benchmark wave."""

    def operation(
        request_index: int,
    ) -> GenerativeInferenceSample:
        if request_index == 1:
            raise RuntimeError(
                "synthetic worker failure"
            )

        return GenerativeInferenceSample(
            input_tokens=1,
            output_tokens=1,
        )

    with pytest.raises(
        RuntimeError,
        match=(
            "failed during measured "
            "for request index 1"
        ),
    ):
        run_controlled_concurrent_generative_benchmark(
            candidate_implementation_id=(
                "failing-concurrent-v1"
            ),
            operation=operation,
            task_family=(
                "provider-generative-inference"
            ),
            configuration=(
                build_configuration()
            ),
            concurrency=2,
            warmup_rounds=0,
            measured_rounds=1,
        )
