"""Controlled native-batch generative inference benchmarking."""

from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)
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

NativeGenerativeBatchOperation = Callable[
    [],
    Sequence[
        GenerativeInferenceSample
    ],
]

_TimedBatch = tuple[
    tuple[
        GenerativeInferenceSample,
        ...,
    ],
    float,
]


def run_controlled_native_generative_batch_benchmark(
    *,
    candidate_implementation_id: str,
    operation: NativeGenerativeBatchOperation,
    task_family: str,
    configuration: InferenceConfiguration,
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
    """Benchmark genuine native-batch generative inference."""
    if warmup_rounds < 0:
        raise ValueError(
            "Warm-up rounds must be non-negative."
        )

    if measured_rounds < 1:
        raise ValueError(
            "At least one measured round is required."
        )

    batch_size = (
        configuration.batch_size
    )

    for _ in range(
        warmup_rounds
    ):
        _execute_native_batch(
            operation=operation,
            batch_size=batch_size,
            phase="warm-up",
        )

    samples: list[
        GenerativeInferenceSample
    ] = []

    batch_latencies_ms: list[
        float
    ] = []

    measurement_started = (
        perf_counter_ns()
    )

    for _ in range(
        measured_rounds
    ):
        (
            batch_samples,
            batch_latency_ms,
        ) = _execute_native_batch(
            operation=operation,
            batch_size=batch_size,
            phase="measured",
        )

        samples.extend(
            batch_samples
        )

        batch_latencies_ms.append(
            batch_latency_ms
        )

    elapsed_seconds = (
        perf_counter_ns()
        - measurement_started
    ) / 1_000_000_000

    measured_batch_invocations = (
        measured_rounds
    )

    measured_requests = (
        measured_batch_invocations
        * batch_size
    )

    generative = (
        summarize_generative_samples(
            samples
        )
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
                "native-generative-batch"
            ),
            "measurement_scope": (
                "native_batch_operation_wall_clock"
            ),
            "batch_size": (
                batch_size
            ),
            "warmup_rounds": (
                warmup_rounds
            ),
            "measured_rounds": (
                measured_rounds
            ),
            "measured_batch_invocations": (
                measured_batch_invocations
            ),
            "measured_requests": (
                measured_requests
            ),
            "batch_latency_observations": (
                len(
                    batch_latencies_ms
                )
            ),
            "request_equivalent_mean_latency_ms": (
                elapsed_seconds
                * 1000.0
                / measured_requests
            ),
            "concurrency_varied": False,
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
            ),
            latency_samples_ms=(
                batch_latencies_ms
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


def _execute_native_batch(
    *,
    operation: NativeGenerativeBatchOperation,
    batch_size: int,
    phase: str,
) -> _TimedBatch:
    """Execute one native batch and validate its request-level evidence."""
    started = (
        perf_counter_ns()
    )

    try:
        batch_samples = tuple(
            operation()
        )
    except Exception as exc:
        raise RuntimeError(
            "Native generative batch inference "
            f"failed during {phase}: {exc}"
        ) from exc

    latency_ms = (
        perf_counter_ns()
        - started
    ) / 1_000_000

    if len(
        batch_samples
    ) != batch_size:
        raise RuntimeError(
            "Native generative batch operation "
            f"returned {len(batch_samples)} samples "
            f"for configured batch_size={batch_size}."
        )

    return (
        batch_samples,
        latency_ms,
    )
