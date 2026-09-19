"""Controlled concurrent generative inference benchmarking."""

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter_ns

from pydantic import JsonValue

from vait.inference.benchmark import (
    build_inference_benchmark_report,
    build_throughput_summary,
    summarize_generative_samples,
)
from vait.inference.models import (
    GenerativeInferenceSample,
    InferenceBenchmarkReport,
    InferenceConfiguration,
    InferenceEnvironment,
    InferenceSetupEvidence,
    InferenceWorkload,
)

ConcurrentGenerativeOperation = Callable[
    [int],
    GenerativeInferenceSample,
]

_TimedSample = tuple[
    GenerativeInferenceSample,
    float,
]


def run_controlled_concurrent_generative_benchmark(
    *,
    candidate_implementation_id: str,
    operation: ConcurrentGenerativeOperation,
    task_family: str,
    configuration: InferenceConfiguration,
    concurrency: int,
    warmup_rounds: int,
    measured_rounds: int,
    workload: InferenceWorkload | None = None,
    setup: InferenceSetupEvidence | None = None,
    environment: InferenceEnvironment | None = None,
    evidence_metadata: Mapping[
        str,
        JsonValue,
    ]
    | None = None,
) -> InferenceBenchmarkReport:
    """Benchmark fixed-width waves of concurrent generative requests."""
    if concurrency < 1:
        raise ValueError(
            "Concurrency must be at least one."
        )

    if warmup_rounds < 0:
        raise ValueError(
            "Warm-up rounds must be non-negative."
        )

    if measured_rounds < 1:
        raise ValueError(
            "At least one measured round is required."
        )

    if configuration.batch_size != 1:
        raise ValueError(
            "Concurrent generative benchmarking requires "
            "batch_size=1 so batching and concurrency remain "
            "separate optimisation axes."
        )

    samples: list[
        GenerativeInferenceSample
    ] = []

    latency_samples_ms: list[
        float
    ] = []

    with ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:
        for _ in range(
            warmup_rounds
        ):
            _run_concurrent_wave(
                executor=executor,
                operation=operation,
                concurrency=concurrency,
                phase="warm-up",
            )

        measurement_started = (
            perf_counter_ns()
        )

        for _ in range(
            measured_rounds
        ):
            wave = _run_concurrent_wave(
                executor=executor,
                operation=operation,
                concurrency=concurrency,
                phase="measured",
            )

            for sample, latency_ms in wave:
                samples.append(
                    sample
                )
                latency_samples_ms.append(
                    latency_ms
                )

        elapsed_seconds = (
            perf_counter_ns()
            - measurement_started
        ) / 1_000_000_000

    generative = (
        summarize_generative_samples(
            samples
        )
    )

    measured_requests = (
        measured_rounds
        * concurrency
    )

    throughput = (
        build_throughput_summary(
            elapsed_seconds=(
                elapsed_seconds
            ),
            requests_processed=(
                measured_requests
            ),
            output_tokens=(
                generative.output_tokens
            ),
        )
    )

    metadata: dict[
        str,
        JsonValue,
    ] = dict(
        evidence_metadata or {}
    )

    metadata.update(
        {
            "execution_mode": (
                "concurrent-generative"
            ),
            "concurrency": (
                concurrency
            ),
            "warmup_rounds": (
                warmup_rounds
            ),
            "measured_rounds": (
                measured_rounds
            ),
            "requests_per_wave": (
                concurrency
            ),
            "measured_requests": (
                measured_requests
            ),
            "request_latency_scope": (
                "operation_wall_clock_inside_worker"
            ),
            "throughput_scope": (
                "concurrent_wave_wall_clock"
            ),
            "batch_size_fixed": 1,
        }
    )

    return (
        build_inference_benchmark_report(
            candidate_implementation_id=(
                candidate_implementation_id
            ),
            task_family=(
                task_family
            ),
            configuration=(
                configuration
            ),
            workload=workload,
            setup=setup,
            warmup_iterations=(
                warmup_rounds
                * concurrency
            ),
            latency_samples_ms=(
                latency_samples_ms
            ),
            throughput=(
                throughput
            ),
            generative=(
                generative
            ),
            environment=(
                environment
            ),
            evidence_metadata=(
                metadata
            ),
        )
    )


def _run_concurrent_wave(
    *,
    executor: ThreadPoolExecutor,
    operation: ConcurrentGenerativeOperation,
    concurrency: int,
    phase: str,
) -> list[_TimedSample]:
    """Execute one fixed-width concurrent request wave."""
    futures = [
        executor.submit(
            _execute_timed_operation,
            operation,
            request_index,
        )
        for request_index
        in range(
            concurrency
        )
    ]

    results: list[
        _TimedSample
    ] = []

    for request_index, future in enumerate(
        futures
    ):
        try:
            result = future.result()
        except Exception as exc:
            for pending in futures:
                pending.cancel()

            raise RuntimeError(
                "Concurrent generative inference "
                f"failed during {phase} for "
                f"request index {request_index}: "
                f"{exc}"
            ) from exc

        results.append(
            result
        )

    return results


def _execute_timed_operation(
    operation: ConcurrentGenerativeOperation,
    request_index: int,
) -> _TimedSample:
    """Execute one worker request and record operation latency."""
    started = perf_counter_ns()

    sample = operation(
        request_index
    )

    latency_ms = (
        perf_counter_ns()
        - started
    ) / 1_000_000

    return (
        sample,
        latency_ms,
    )
