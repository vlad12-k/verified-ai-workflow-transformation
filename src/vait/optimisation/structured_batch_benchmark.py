"""Native batch inference evidence for verified structured candidates."""

from collections.abc import Callable, Mapping, Sequence
from time import perf_counter_ns

from pydantic import JsonValue

from vait.contracts.models import VerificationCase
from vait.inference.benchmark import (
    build_inference_benchmark_report,
    build_throughput_summary,
    capture_inference_environment,
)
from vait.inference.models import (
    InferenceBenchmarkReport,
    InferenceEnvironment,
    InferenceSetupEvidence,
)
from vait.inference.profiling import (
    build_process_resource_evidence,
    capture_process_resource_snapshot,
)
from vait.optimisation.admissibility import (
    SearchPointAdmissibility,
)
from vait.optimisation.structured_benchmark import (
    StructuredBenchmarkRun,
    StructuredBenchmarkSeries,
    StructuredModelFootprint,
    summarize_structured_repetitions,
)

NativeBatchOperation = Callable[
    [Sequence[VerificationCase]],
    Sequence[JsonValue],
]


def run_structured_native_batch_series(
    *,
    admissibility: SearchPointAdmissibility,
    implementation_id: str,
    operation: NativeBatchOperation,
    cases: Sequence[VerificationCase],
    repetitions: int,
    warmup_rounds: int,
    measured_rounds: int,
    setup: InferenceSetupEvidence | None = None,
    environment: InferenceEnvironment | None = None,
    footprint: StructuredModelFootprint | None = None,
    evidence_metadata: Mapping[
        str,
        JsonValue,
    ]
    | None = None,
) -> StructuredBenchmarkSeries:
    """Benchmark one admissible structured native-batch search point."""
    if not admissibility.is_admissible:
        raise ValueError(
            "Native batch benchmarking requires an "
            "admissible search point."
        )

    search_point = admissibility.search_point

    if implementation_id != search_point.implementation_id:
        raise ValueError(
            "Batch implementation_id does not match "
            "the admissible search point."
        )

    if not cases:
        raise ValueError(
            "At least one inference case is required."
        )

    if repetitions < 1:
        raise ValueError(
            "At least one benchmark repetition is required."
        )

    if warmup_rounds < 0:
        raise ValueError(
            "Warm-up rounds must be non-negative."
        )

    if measured_rounds < 1:
        raise ValueError(
            "At least one measured round is required."
        )

    batch_size = (
        search_point.configuration.batch_size
    )

    batches = tuple(
        tuple(
            cases[
                start : start + batch_size
            ]
        )
        for start in range(
            0,
            len(cases),
            batch_size,
        )
    )

    shared_environment = (
        environment
        or capture_inference_environment()
    )

    runs: list[
        StructuredBenchmarkRun
    ] = []

    for repetition_index in range(
        repetitions
    ):
        metadata: dict[
            str,
            JsonValue,
        ] = dict(
            evidence_metadata
            or {}
        )

        report = _run_native_batch_report(
            search_point=admissibility,
            implementation_id=implementation_id,
            operation=operation,
            batches=batches,
            case_count=len(cases),
            warmup_rounds=warmup_rounds,
            measured_rounds=measured_rounds,
            setup=setup,
            environment=shared_environment,
            repetition_index=repetition_index,
            evidence_metadata=metadata,
        )

        runs.append(
            StructuredBenchmarkRun(
                repetition_index=(
                    repetition_index
                ),
                report=report,
            )
        )

    repeatability = (
        summarize_structured_repetitions(
            runs
        )
    )

    return StructuredBenchmarkSeries(
        search_point_id=(
            search_point.search_point_id
        ),
        implementation_id=implementation_id,
        task_family=(
            search_point.task_family
        ),
        runs=tuple(runs),
        repeatability=repeatability,
        footprint=footprint,
    )


def _run_native_batch_report(
    *,
    search_point: SearchPointAdmissibility,
    implementation_id: str,
    operation: NativeBatchOperation,
    batches: Sequence[
        Sequence[VerificationCase]
    ],
    case_count: int,
    warmup_rounds: int,
    measured_rounds: int,
    setup: InferenceSetupEvidence | None,
    environment: InferenceEnvironment,
    repetition_index: int,
    evidence_metadata: Mapping[
        str,
        JsonValue,
    ],
) -> InferenceBenchmarkReport:
    """Run one controlled native-batch repetition."""
    for _ in range(
        warmup_rounds
    ):
        for batch in batches:
            _execute_native_batch(
                operation=operation,
                batch=batch,
                phase="warm-up",
            )

    latency_samples_ms: list[
        float
    ] = []

    resource_started = (
        capture_process_resource_snapshot()
    )

    measurement_started = (
        perf_counter_ns()
    )

    for _ in range(
        measured_rounds
    ):
        for batch in batches:
            latency_samples_ms.append(
                _execute_native_batch(
                    operation=operation,
                    batch=batch,
                    phase="measured",
                )
            )

    measurement_ended = (
        perf_counter_ns()
    )

    resource_ended = (
        capture_process_resource_snapshot()
    )

    elapsed_seconds = (
        measurement_ended
        - measurement_started
    ) / 1_000_000_000

    resources = (
        build_process_resource_evidence(
            started=resource_started,
            ended=resource_ended,
            elapsed_seconds=elapsed_seconds,
            measurement_scope=(
                "controlled-native-batch-measured-region"
            ),
        )
    )

    batch_invocations_per_round = len(
        batches
    )

    measured_batch_invocations = (
        batch_invocations_per_round
        * measured_rounds
    )

    measured_cases = (
        case_count
        * measured_rounds
    )

    throughput = (
        build_throughput_summary(
            elapsed_seconds=elapsed_seconds,
            cases_processed=measured_cases,
            requests_processed=(
                measured_batch_invocations
            ),
        )
    )

    metadata: dict[
        str,
        JsonValue,
    ] = dict(
        evidence_metadata
    )

    batch_size = (
        search_point
        .search_point
        .configuration
        .batch_size
    )

    metadata.update(
        {
            "search_point_id": (
                search_point
                .search_point
                .search_point_id
            ),
            "admissibility_gate": "passed",
            "execution_mode": "native-batch",
            "measurement_scope": (
                "native_batch_operation_wall_clock"
            ),
            "repetition_index": (
                repetition_index
            ),
            "case_count": case_count,
            "batch_size": batch_size,
            "batch_invocations_per_round": (
                batch_invocations_per_round
            ),
            "measured_batch_invocations": (
                measured_batch_invocations
            ),
            "measured_cases": (
                measured_cases
            ),
            "tail_batch_size": (
                case_count % batch_size
                or batch_size
            ),
            "mean_case_latency_ms": (
                elapsed_seconds
                * 1000.0
                / measured_cases
            ),
            "mean_batch_request_latency_ms": (
                elapsed_seconds
                * 1000.0
                / measured_batch_invocations
            ),
        }
    )

    return (
        build_inference_benchmark_report(
            candidate_implementation_id=(
                implementation_id
            ),
            task_family=(
                search_point
                .search_point
                .task_family
            ),
            configuration=(
                search_point
                .search_point
                .configuration
            ),
            setup=setup,
            warmup_iterations=(
                warmup_rounds
                * batch_invocations_per_round
            ),
            latency_samples_ms=(
                latency_samples_ms
            ),
            throughput=throughput,
            cold_start_latency_ms=None,
            resources=resources,
            environment=environment,
            evidence_metadata=metadata,
        )
    )


def _execute_native_batch(
    *,
    operation: NativeBatchOperation,
    batch: Sequence[
        VerificationCase
    ],
    phase: str,
) -> float:
    """Execute one native batch and return outer wall-clock latency."""
    started = perf_counter_ns()

    try:
        outputs = operation(
            batch
        )
    except Exception as exc:
        raise RuntimeError(
            "Native batch inference failed "
            f"during {phase}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    latency_ms = (
        perf_counter_ns()
        - started
    ) / 1_000_000

    if len(outputs) != len(
        batch
    ):
        raise RuntimeError(
            "Native batch operation returned "
            "a different number of outputs "
            "than input cases."
        )

    return latency_ms
