"""Build auditable synthetic evidence for the M4-I multi-objective optimiser."""

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import TypedDict

from pydantic import BaseModel

from vait.decision.models import Decision
from vait.inference.benchmark import (
    capture_inference_environment,
)
from vait.inference.models import (
    InferenceBenchmarkReport,
    InferenceConfiguration,
    InferenceEnvironment,
    InferenceLatencySummary,
    InferenceResourceEvidence,
    InferenceThroughputSummary,
    InferenceWorkload,
)
from vait.optimisation.admissibility import (
    SearchPointAdmissibility,
    SearchPointVerificationEvidence,
    VerificationEvidenceKind,
    evaluate_search_point_admissibility,
)
from vait.optimisation.compatibility import (
    InferenceCompatibilityContext,
    evaluate_search_point_compatibility,
)
from vait.optimisation.multi_objective import (
    ConstraintOperator,
    EvidenceComparabilityKey,
    MultiObjectiveAnalysis,
    MultiObjectiveCandidateEvidence,
    ObjectiveConstraint,
    ObjectiveDirection,
    ObjectiveMetric,
    OptimisationObjective,
    ParetoPreferencePolicy,
    PreferencePriority,
    analyse_multi_objective_candidates,
)
from vait.optimisation.search_space import (
    InferenceSearchPoint,
)
from vait.optimisation.structured_benchmark import (
    StructuredBenchmarkRun,
    StructuredBenchmarkSeries,
    StructuredModelFootprint,
    summarize_structured_repetitions,
)

ARTIFACT_PATH = Path(
    "artifacts/benchmarks/"
    "m4i-multi-objective-analysis-v0.1.json"
)

WORKLOAD_ID = "m4i-synthetic-structured-workload"
WORKLOAD_VERSION = "0.1"

WORKLOAD_FINGERPRINT = sha256(
    b"m4i-synthetic-structured-workload-v0.1"
).hexdigest()

MEASUREMENT_PROTOCOL_ID = (
    "m4i-synthetic-structured-scalar-v1"
)


class SyntheticCandidateMetrics(TypedDict):
    """Typed synthetic measurements used only to exercise M4-I logic."""

    latency_ms: float
    throughput: float
    cpu_time_ms: float
    peak_rss_bytes: int
    model_size_bytes: int


class M4IMultiObjectiveArtifact(BaseModel):
    """Versioned synthetic evidence for M4-I optimiser behaviour."""

    schema_version: str = "0.1"

    artifact_id: str
    evidence_scope: str
    performance_claim: bool

    excluded_before_optimisation: tuple[
        SearchPointAdmissibility,
        ...,
    ]

    optimisation_inputs: tuple[
        MultiObjectiveCandidateEvidence,
        ...,
    ]

    analysis: MultiObjectiveAnalysis


def build_search_point(
    candidate_id: str,
) -> InferenceSearchPoint:
    """Build one synthetic registered search point."""
    implementation_id = (
        f"{candidate_id}-v1"
    )

    return InferenceSearchPoint(
        search_point_id=(
            f"{candidate_id}::cpu-float32-b1"
        ),
        search_space_id="m4i-synthetic-space",
        task_family="structured-decision",
        candidate_id=candidate_id,
        implementation_id=implementation_id,
        configuration_id="cpu-float32-b1",
        configuration=InferenceConfiguration(
            provider="local",
            runtime="synthetic",
            device="cpu",
            dtype="float32",
            batch_size=1,
            model_id=implementation_id,
            model_revision="0.1",
        ),
        requires_verification=True,
        required_capabilities=frozenset(
            {
                "structured-inference",
            }
        ),
    )


def build_context(
    points: tuple[
        InferenceSearchPoint,
        ...,
    ],
) -> InferenceCompatibilityContext:
    """Build one common synthetic execution context."""
    return InferenceCompatibilityContext(
        available_providers=frozenset(
            {"local"}
        ),
        available_runtimes=frozenset(
            {"synthetic"}
        ),
        available_devices=frozenset(
            {"cpu"}
        ),
        available_dtypes=frozenset(
            {"float32"}
        ),
        max_batch_size=1,
        available_model_ids=frozenset(
            point.implementation_id
            for point in points
        ),
        available_capabilities=frozenset(
            {
                "structured-inference",
            }
        ),
    )


def evaluate_admissibility(
    *,
    point: InferenceSearchPoint,
    context: InferenceCompatibilityContext,
    decision: Decision,
) -> SearchPointAdmissibility:
    """Run compatibility and VERIFY-FIRST admissibility."""
    compatibility = (
        evaluate_search_point_compatibility(
            point,
            context,
        )
    )

    evidence = SearchPointVerificationEvidence(
        search_point_id=point.search_point_id,
        implementation_id=point.implementation_id,
        contract_id=(
            f"synthetic-contract:{point.search_point_id}"
        ),
        decision=decision,
        evidence_kind=(
            VerificationEvidenceKind.DETERMINISTIC
        ),
        statistical_evidence_present=False,
    )

    return evaluate_search_point_admissibility(
        compatibility=compatibility,
        verification_evidence=evidence,
    )


def build_series(
    *,
    admissibility: SearchPointAdmissibility,
    environment: InferenceEnvironment,
    latency_ms: float,
    throughput: float,
    cpu_time_ms: float,
    peak_rss_bytes: int,
    model_size_bytes: int,
) -> StructuredBenchmarkSeries:
    """Build one explicitly synthetic benchmark series."""
    if not admissibility.is_admissible:
        raise ValueError(
            "Synthetic benchmark evidence requires "
            "an admissible search point."
        )

    point = admissibility.search_point

    workload = InferenceWorkload(
        workload_id=WORKLOAD_ID,
        version=WORKLOAD_VERSION,
        item_count=100,
        fingerprint_sha256=(
            WORKLOAD_FINGERPRINT
        ),
        metadata={
            "evidence_scope": (
                "synthetic-optimiser-demonstration"
            ),
        },
    )

    resources = InferenceResourceEvidence(
        measurement_scope=(
            "synthetic-controlled-measured-region"
        ),
        wall_time_ms=20.0,
        process_cpu_time_ms=cpu_time_ms,
        cpu_time_to_wall_time_ratio=(
            cpu_time_ms / 20.0
        ),
        process_peak_rss_baseline_bytes=(
            peak_rss_bytes
        ),
        process_peak_rss_bytes=(
            peak_rss_bytes
        ),
        process_peak_rss_growth_bytes=0,
    )

    report = InferenceBenchmarkReport(
        run_id=(
            f"synthetic-run:{point.search_point_id}"
        ),
        created_at=datetime.now(UTC),
        candidate_implementation_id=(
            point.implementation_id
        ),
        task_family=point.task_family,
        workload=workload,
        warmup_iterations=10,
        measured_iterations=100,
        configuration=point.configuration,
        latency=InferenceLatencySummary(
            count=100,
            mean_ms=latency_ms,
            p50_ms=latency_ms,
            p95_ms=latency_ms,
            p99_ms=latency_ms,
            max_ms=latency_ms,
        ),
        throughput=InferenceThroughputSummary(
            cases_per_second=throughput,
            requests_per_second=throughput,
        ),
        resources=resources,
        environment=environment,
        evidence_metadata={
            "milestone": "M4-I",
            "evidence_scope": (
                "synthetic-optimiser-demonstration"
            ),
            "performance_claim": False,
            "measurement_protocol_id": (
                MEASUREMENT_PROTOCOL_ID
            ),
        },
    )

    run = StructuredBenchmarkRun(
        repetition_index=0,
        report=report,
    )

    return StructuredBenchmarkSeries(
        search_point_id=point.search_point_id,
        implementation_id=point.implementation_id,
        task_family=point.task_family,
        runs=(run,),
        repeatability=(
            summarize_structured_repetitions(
                (run,)
            )
        ),
        footprint=StructuredModelFootprint(
            model_size_bytes=model_size_bytes,
            metadata={
                "evidence_scope": (
                    "synthetic-optimiser-demonstration"
                ),
            },
        ),
    )


def main() -> None:
    """Build VERIFY-FIRST synthetic M4-I optimiser evidence."""
    point_by_id = {
        candidate_id: build_search_point(
            candidate_id
        )
        for candidate_id in (
            "rejected-fastest",
            "fast-large",
            "balanced",
            "compact",
            "dominated",
        )
    }

    all_points = tuple(
        point_by_id.values()
    )

    context = build_context(
        all_points
    )

    rejected = evaluate_admissibility(
        point=point_by_id[
            "rejected-fastest"
        ],
        context=context,
        decision=Decision.REJECT,
    )

    if rejected.is_admissible:
        raise RuntimeError(
            "REJECT unexpectedly entered optimisation."
        )

    admitted = {
        candidate_id: evaluate_admissibility(
            point=point_by_id[
                candidate_id
            ],
            context=context,
            decision=Decision.EXACT,
        )
        for candidate_id in (
            "fast-large",
            "balanced",
            "compact",
            "dominated",
        )
    }

    for candidate_id, result in admitted.items():
        if not result.is_admissible:
            raise RuntimeError(
                "Expected synthetic candidate to be "
                f"admissible: {candidate_id}."
            )

    environment = (
        capture_inference_environment()
    )

    comparability = EvidenceComparabilityKey(
        workload_id=WORKLOAD_ID,
        workload_version=WORKLOAD_VERSION,
        workload_fingerprint_sha256=(
            WORKLOAD_FINGERPRINT
        ),
        measurement_protocol_id=(
            MEASUREMENT_PROTOCOL_ID
        ),
    )

    synthetic_metrics: dict[
        str,
        SyntheticCandidateMetrics,
    ] = {
        "fast-large": {
            "latency_ms": 0.80,
            "throughput": 500.0,
            "cpu_time_ms": 8.0,
            "peak_rss_bytes": 220_000_000,
            "model_size_bytes": 12_000,
        },
        "balanced": {
            "latency_ms": 1.20,
            "throughput": 300.0,
            "cpu_time_ms": 12.0,
            "peak_rss_bytes": 120_000_000,
            "model_size_bytes": 4_000,
        },
        "compact": {
            "latency_ms": 1.50,
            "throughput": 240.0,
            "cpu_time_ms": 10.0,
            "peak_rss_bytes": 90_000_000,
            "model_size_bytes": 1_000,
        },
        "dominated": {
            "latency_ms": 2.00,
            "throughput": 180.0,
            "cpu_time_ms": 15.0,
            "peak_rss_bytes": 150_000_000,
            "model_size_bytes": 4_500,
        },
    }

    optimisation_inputs = tuple(
        MultiObjectiveCandidateEvidence(
            admissibility=admitted[
                candidate_id
            ],
            benchmark=build_series(
                admissibility=admitted[
                    candidate_id
                ],
                environment=environment,
                **metrics,
            ),
            comparability=comparability,
        )
        for candidate_id, metrics
        in synthetic_metrics.items()
    )

    analysis = analyse_multi_objective_candidates(
        candidates=optimisation_inputs,
        objectives=(
            OptimisationObjective(
                metric=(
                    ObjectiveMetric.MEAN_LATENCY_MS
                ),
                direction=(
                    ObjectiveDirection.MINIMIZE
                ),
            ),
            OptimisationObjective(
                metric=(
                    ObjectiveMetric
                    .MEAN_CASES_PER_SECOND
                ),
                direction=(
                    ObjectiveDirection.MAXIMIZE
                ),
            ),
            OptimisationObjective(
                metric=(
                    ObjectiveMetric
                    .MEAN_PROCESS_CPU_TIME_MS
                ),
                direction=(
                    ObjectiveDirection.MINIMIZE
                ),
            ),
            OptimisationObjective(
                metric=(
                    ObjectiveMetric.MODEL_SIZE_BYTES
                ),
                direction=(
                    ObjectiveDirection.MINIMIZE
                ),
            ),
        ),
        constraints=(
            ObjectiveConstraint(
                metric=(
                    ObjectiveMetric.MODEL_SIZE_BYTES
                ),
                operator=(
                    ConstraintOperator.AT_MOST
                ),
                threshold=5_000.0,
            ),
        ),
        preference_policy=ParetoPreferencePolicy(
            priorities=(
                PreferencePriority(
                    metric=(
                        ObjectiveMetric.MEAN_LATENCY_MS
                    ),
                    relative_tolerance=0.30,
                ),
                PreferencePriority(
                    metric=(
                        ObjectiveMetric.MODEL_SIZE_BYTES
                    ),
                ),
            ),
        ),
    )

    artifact = M4IMultiObjectiveArtifact(
        artifact_id=(
            "m4i-multi-objective-analysis-v0.1"
        ),
        evidence_scope=(
            "synthetic-optimiser-demonstration"
        ),
        performance_claim=False,
        excluded_before_optimisation=(
            rejected,
        ),
        optimisation_inputs=(
            optimisation_inputs
        ),
        analysis=analysis,
    )

    ARTIFACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT_PATH.write_text(
        artifact.model_dump_json(
            indent=2
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "M4-I multi-objective optimiser evidence"
    )
    print()

    print(
        "Pre-optimisation exclusions:",
        [
            result.search_point.search_point_id
            for result
            in artifact.excluded_before_optimisation
        ],
    )

    print(
        "Constraint rejections:",
        [
            result.search_point_id
            for result
            in analysis.constraint_rejections
        ],
    )

    print(
        "Dominated:",
        list(
            analysis.dominated_search_point_ids
        ),
    )

    print(
        "Pareto frontier:",
        list(
            analysis
            .pareto_frontier_search_point_ids
        ),
    )

    print(
        "Preferred frontier:",
        list(
            analysis
            .preferred_frontier_search_point_ids
        ),
    )

    print()
    print(
        "Evidence scope:",
        artifact.evidence_scope,
    )
    print(
        "Performance claim:",
        artifact.performance_claim,
    )
    print(
        "Source revision:",
        environment.source_revision,
    )
    print(
        "Source dirty:",
        environment.source_dirty,
    )
    print(
        "Evidence artifact:",
        ARTIFACT_PATH,
    )


if __name__ == "__main__":
    main()
