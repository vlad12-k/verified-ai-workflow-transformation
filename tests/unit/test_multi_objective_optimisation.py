"""Tests for verification-gated Pareto multi-objective optimisation."""

from datetime import UTC, datetime

import pytest

from vait.inference.models import (
    InferenceBenchmarkReport,
    InferenceConfiguration,
    InferenceEnvironment,
    InferenceLatencySummary,
    InferenceResourceEvidence,
    InferenceThroughputSummary,
)
from vait.optimisation.admissibility import (
    AdmissibilityIssue,
    AdmissibilityIssueCode,
    AdmissibilityStatus,
    SearchPointAdmissibility,
)
from vait.optimisation.multi_objective import (
    ConstraintOperator,
    EvidenceComparabilityKey,
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


def _point(
    *,
    candidate_id: str,
    task_family: str = "structured-decision",
) -> InferenceSearchPoint:
    """Build one synthetic optimisation search point."""
    return InferenceSearchPoint(
        search_point_id=(
            f"{candidate_id}::cpu"
        ),
        search_space_id="m4i-test",
        task_family=task_family,
        candidate_id=candidate_id,
        implementation_id=(
            f"{candidate_id}-v1"
        ),
        configuration_id="cpu",
        configuration=InferenceConfiguration(
            provider="local",
            runtime="python",
            device="cpu",
            dtype="native",
            batch_size=1,
        ),
        requires_verification=True,
    )


def _admissible(
    point: InferenceSearchPoint,
) -> SearchPointAdmissibility:
    """Mark one synthetic point admissible."""
    return SearchPointAdmissibility(
        search_point=point,
        status=AdmissibilityStatus.ADMISSIBLE,
    )


def _series(
    *,
    point: InferenceSearchPoint,
    latency_ms: float,
    throughput: float,
    cpu_time_ms: float | None = 10.0,
    peak_rss_bytes: int | None = 100_000,
    model_size_bytes: int | None = 1_000,
) -> StructuredBenchmarkSeries:
    """Build deterministic structured benchmark evidence."""
    resources = (
        InferenceResourceEvidence(
            measurement_scope="test",
            wall_time_ms=20.0,
            process_cpu_time_ms=cpu_time_ms,
            cpu_time_to_wall_time_ratio=0.5,
            process_peak_rss_baseline_bytes=(
                peak_rss_bytes
            ),
            process_peak_rss_bytes=(
                peak_rss_bytes
            ),
            process_peak_rss_growth_bytes=0,
        )
        if (
            cpu_time_ms is not None
            and peak_rss_bytes is not None
        )
        else None
    )

    run = StructuredBenchmarkRun(
        repetition_index=0,
        report=InferenceBenchmarkReport(
            run_id=(
                f"run-{point.search_point_id}"
            ),
            created_at=datetime.now(UTC),
            candidate_implementation_id=(
                point.implementation_id
            ),
            task_family=point.task_family,
            warmup_iterations=0,
            measured_iterations=1,
            configuration=point.configuration,
            latency=InferenceLatencySummary(
                count=1,
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
            environment=InferenceEnvironment(
                python_version="3.12.0",
                platform="test-platform",
                system="test-system",
                machine="test-machine",
                processor="test-processor",
            ),
        ),
    )

    footprint = (
        StructuredModelFootprint(
            model_size_bytes=model_size_bytes,
        )
        if model_size_bytes is not None
        else None
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
        footprint=footprint,
    )


def _comparability(
    *,
    fingerprint: str = "a" * 64,
    protocol_id: str = "structured-scalar-v1",
) -> EvidenceComparabilityKey:
    """Build one explicit synthetic comparison scope."""
    return EvidenceComparabilityKey(
        workload_id="synthetic-workload",
        workload_version="0.1",
        workload_fingerprint_sha256=fingerprint,
        measurement_protocol_id=protocol_id,
    )


def _candidate(
    *,
    candidate_id: str,
    latency_ms: float,
    throughput: float,
    cpu_time_ms: float | None = 10.0,
    peak_rss_bytes: int | None = 100_000,
    model_size_bytes: int | None = 1_000,
    task_family: str = "structured-decision",
    fingerprint: str = "a" * 64,
    protocol_id: str = "structured-scalar-v1",
) -> MultiObjectiveCandidateEvidence:
    """Build one admissible candidate evidence bundle."""
    point = _point(
        candidate_id=candidate_id,
        task_family=task_family,
    )

    return MultiObjectiveCandidateEvidence(
        admissibility=_admissible(
            point
        ),
        benchmark=_series(
            point=point,
            latency_ms=latency_ms,
            throughput=throughput,
            cpu_time_ms=cpu_time_ms,
            peak_rss_bytes=peak_rss_bytes,
            model_size_bytes=model_size_bytes,
        ),
        comparability=_comparability(
            fingerprint=fingerprint,
            protocol_id=protocol_id,
        ),
    )


def test_dominated_candidate_is_removed_from_frontier() -> None:
    """A candidate worse on all objectives must be dominated."""
    analysis = analyse_multi_objective_candidates(
        candidates=(
            _candidate(
                candidate_id="fast",
                latency_ms=1.0,
                throughput=200.0,
            ),
            _candidate(
                candidate_id="slow",
                latency_ms=2.0,
                throughput=100.0,
            ),
        ),
        objectives=(
            OptimisationObjective(
                metric=ObjectiveMetric.MEAN_LATENCY_MS,
                direction=ObjectiveDirection.MINIMIZE,
            ),
            OptimisationObjective(
                metric=(
                    ObjectiveMetric
                    .MEAN_CASES_PER_SECOND
                ),
                direction=ObjectiveDirection.MAXIMIZE,
            ),
        ),
    )

    assert (
        analysis.pareto_frontier_search_point_ids
        == ("fast::cpu",)
    )

    assert (
        analysis.dominated_search_point_ids
        == ("slow::cpu",)
    )


def test_tradeoff_candidates_both_remain_on_frontier() -> None:
    """Conflicting objective advantages should preserve both points."""
    analysis = analyse_multi_objective_candidates(
        candidates=(
            _candidate(
                candidate_id="low-latency",
                latency_ms=1.0,
                throughput=100.0,
            ),
            _candidate(
                candidate_id="high-throughput",
                latency_ms=2.0,
                throughput=200.0,
            ),
        ),
        objectives=(
            OptimisationObjective(
                metric=ObjectiveMetric.MEAN_LATENCY_MS,
                direction=ObjectiveDirection.MINIMIZE,
            ),
            OptimisationObjective(
                metric=(
                    ObjectiveMetric
                    .MEAN_CASES_PER_SECOND
                ),
                direction=ObjectiveDirection.MAXIMIZE,
            ),
        ),
    )

    assert (
        analysis.pareto_frontier_search_point_ids
        == (
            "high-throughput::cpu",
            "low-latency::cpu",
        )
    )

    assert (
        analysis.dominated_search_point_ids
        == ()
    )


def test_rejected_candidate_cannot_enter_optimisation() -> None:
    """REJECT-equivalent admissibility cannot be rescued by performance."""
    point = _point(
        candidate_id="rejected"
    )

    rejected = SearchPointAdmissibility(
        search_point=point,
        status=(
            AdmissibilityStatus.NOT_ADMISSIBLE
        ),
        issues=(
            AdmissibilityIssue(
                code=(
                    AdmissibilityIssueCode
                    .VERIFICATION_REJECTED
                ),
                subject=point.search_point_id,
                message="Verification rejected.",
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="admissible search point",
    ):
        MultiObjectiveCandidateEvidence(
            admissibility=rejected,
            benchmark=_series(
                point=point,
                latency_ms=0.001,
                throughput=1_000_000.0,
            ),
            comparability=_comparability(),
        )


def test_missing_requested_resource_evidence_is_rejected() -> None:
    """Resource objectives require evidence for every repetition."""
    candidate = _candidate(
        candidate_id="missing-resources",
        latency_ms=1.0,
        throughput=100.0,
        cpu_time_ms=None,
        peak_rss_bytes=None,
    )

    with pytest.raises(
        ValueError,
        match="complete resource evidence",
    ):
        analyse_multi_objective_candidates(
            candidates=(candidate,),
            objectives=(
                OptimisationObjective(
                    metric=(
                        ObjectiveMetric
                        .MEAN_PROCESS_CPU_TIME_MS
                    ),
                    direction=(
                        ObjectiveDirection.MINIMIZE
                    ),
                ),
            ),
        )


def test_different_task_families_cannot_be_compared() -> None:
    """M4-I must not rank heterogeneous task families together."""
    with pytest.raises(
        ValueError,
        match="different task families",
    ):
        analyse_multi_objective_candidates(
            candidates=(
                _candidate(
                    candidate_id="structured",
                    latency_ms=1.0,
                    throughput=100.0,
                ),
                _candidate(
                    candidate_id="generative",
                    latency_ms=1.0,
                    throughput=100.0,
                    task_family=(
                        "grounded-generation"
                    ),
                ),
            ),
            objectives=(
                OptimisationObjective(
                    metric=(
                        ObjectiveMetric.MEAN_LATENCY_MS
                    ),
                    direction=(
                        ObjectiveDirection.MINIMIZE
                    ),
                ),
            ),
        )


def test_hard_constraint_excludes_otherwise_best_candidate() -> None:
    """Constraint violations must happen before Pareto comparison."""
    analysis = analyse_multi_objective_candidates(
        candidates=(
            _candidate(
                candidate_id="fast-but-large",
                latency_ms=1.0,
                throughput=300.0,
                model_size_bytes=10_000,
            ),
            _candidate(
                candidate_id="slower-small",
                latency_ms=2.0,
                throughput=200.0,
                model_size_bytes=1_000,
            ),
        ),
        objectives=(
            OptimisationObjective(
                metric=ObjectiveMetric.MEAN_LATENCY_MS,
                direction=ObjectiveDirection.MINIMIZE,
            ),
            OptimisationObjective(
                metric=(
                    ObjectiveMetric
                    .MEAN_CASES_PER_SECOND
                ),
                direction=ObjectiveDirection.MAXIMIZE,
            ),
        ),
        constraints=(
            ObjectiveConstraint(
                metric=ObjectiveMetric.MODEL_SIZE_BYTES,
                operator=ConstraintOperator.AT_MOST,
                threshold=5_000.0,
            ),
        ),
    )

    assert (
        analysis.pareto_frontier_search_point_ids
        == ("slower-small::cpu",)
    )

    assert (
        analysis.dominated_search_point_ids
        == ()
    )

    assert len(
        analysis.constraint_rejections
    ) == 1

    rejection = (
        analysis.constraint_rejections[0]
    )

    assert (
        rejection.search_point_id
        == "fast-but-large::cpu"
    )

    assert len(
        rejection.violations
    ) == 1

    violation = rejection.violations[0]

    assert (
        violation.metric
        is ObjectiveMetric.MODEL_SIZE_BYTES
    )
    assert (
        violation.operator
        is ConstraintOperator.AT_MOST
    )
    assert violation.threshold == 5_000.0
    assert violation.observed_value == 10_000.0


def test_constraint_boundary_is_feasible() -> None:
    """Evidence equal to a hard threshold must remain feasible."""
    analysis = analyse_multi_objective_candidates(
        candidates=(
            _candidate(
                candidate_id="boundary",
                latency_ms=1.0,
                throughput=100.0,
            ),
        ),
        objectives=(
            OptimisationObjective(
                metric=ObjectiveMetric.MEAN_LATENCY_MS,
                direction=ObjectiveDirection.MINIMIZE,
            ),
        ),
        constraints=(
            ObjectiveConstraint(
                metric=(
                    ObjectiveMetric
                    .MEAN_CASES_PER_SECOND
                ),
                operator=ConstraintOperator.AT_LEAST,
                threshold=100.0,
            ),
        ),
    )

    assert (
        analysis.pareto_frontier_search_point_ids
        == ("boundary::cpu",)
    )

    assert (
        analysis.constraint_rejections
        == ()
    )


def test_different_workload_fingerprints_cannot_be_compared() -> None:
    """Performance from different workloads must never share a frontier."""
    with pytest.raises(
        ValueError,
        match="same workload fingerprint",
    ):
        analyse_multi_objective_candidates(
            candidates=(
                _candidate(
                    candidate_id="candidate-a",
                    latency_ms=1.0,
                    throughput=100.0,
                    fingerprint="a" * 64,
                ),
                _candidate(
                    candidate_id="candidate-b",
                    latency_ms=2.0,
                    throughput=200.0,
                    fingerprint="b" * 64,
                ),
            ),
            objectives=(
                OptimisationObjective(
                    metric=(
                        ObjectiveMetric.MEAN_LATENCY_MS
                    ),
                    direction=(
                        ObjectiveDirection.MINIMIZE
                    ),
                ),
            ),
        )


def test_different_measurement_protocols_cannot_be_compared() -> None:
    """Equal workloads remain incomparable under different protocols."""
    with pytest.raises(
        ValueError,
        match="measurement protocol",
    ):
        analyse_multi_objective_candidates(
            candidates=(
                _candidate(
                    candidate_id="candidate-a",
                    latency_ms=1.0,
                    throughput=100.0,
                    protocol_id="scalar-v1",
                ),
                _candidate(
                    candidate_id="candidate-b",
                    latency_ms=2.0,
                    throughput=200.0,
                    protocol_id="batch-request-v1",
                ),
            ),
            objectives=(
                OptimisationObjective(
                    metric=(
                        ObjectiveMetric
                        .MEAN_CASES_PER_SECOND
                    ),
                    direction=(
                        ObjectiveDirection.MAXIMIZE
                    ),
                ),
            ),
        )


def test_resource_objective_requires_complete_repetition_coverage() -> None:
    """Partial resource observations must not become optimisation evidence."""
    point = _point(
        candidate_id="partial-resource"
    )

    run_with_resources = (
        _series(
            point=point,
            latency_ms=1.0,
            throughput=100.0,
        ).runs[0]
    )

    report_without_resources = (
        run_with_resources.report.model_copy(
            update={
                "run_id": "run-without-resources",
                "resources": None,
            }
        )
    )

    run_without_resources = StructuredBenchmarkRun(
        repetition_index=1,
        report=report_without_resources,
    )

    runs = (
        run_with_resources,
        run_without_resources,
    )

    series = StructuredBenchmarkSeries(
        search_point_id=point.search_point_id,
        implementation_id=point.implementation_id,
        task_family=point.task_family,
        runs=runs,
        repeatability=(
            summarize_structured_repetitions(
                runs
            )
        ),
        footprint=StructuredModelFootprint(
            model_size_bytes=1_000,
        ),
    )

    candidate = MultiObjectiveCandidateEvidence(
        admissibility=_admissible(
            point
        ),
        benchmark=series,
        comparability=_comparability(),
    )

    assert (
        series.repeatability.resource_observations
        == 1
    )
    assert (
        series.repeatability.repetitions
        == 2
    )

    with pytest.raises(
        ValueError,
        match="complete resource evidence",
    ):
        analyse_multi_objective_candidates(
            candidates=(candidate,),
            objectives=(
                OptimisationObjective(
                    metric=(
                        ObjectiveMetric
                        .MEAN_PROCESS_CPU_TIME_MS
                    ),
                    direction=(
                        ObjectiveDirection.MINIMIZE
                    ),
                ),
            ),
        )


def test_preference_policy_narrows_only_pareto_frontier() -> None:
    """Explicit latency priority may narrow a genuine Pareto trade-off."""
    analysis = analyse_multi_objective_candidates(
        candidates=(
            _candidate(
                candidate_id="low-latency",
                latency_ms=1.0,
                throughput=100.0,
            ),
            _candidate(
                candidate_id="high-throughput",
                latency_ms=2.0,
                throughput=200.0,
            ),
        ),
        objectives=(
            OptimisationObjective(
                metric=ObjectiveMetric.MEAN_LATENCY_MS,
                direction=ObjectiveDirection.MINIMIZE,
            ),
            OptimisationObjective(
                metric=(
                    ObjectiveMetric.MEAN_CASES_PER_SECOND
                ),
                direction=ObjectiveDirection.MAXIMIZE,
            ),
        ),
        preference_policy=ParetoPreferencePolicy(
            priorities=(
                PreferencePriority(
                    metric=ObjectiveMetric.MEAN_LATENCY_MS,
                ),
            ),
        ),
    )

    assert (
        analysis.pareto_frontier_search_point_ids
        == (
            "high-throughput::cpu",
            "low-latency::cpu",
        )
    )

    assert (
        analysis.preferred_frontier_search_point_ids
        == ("low-latency::cpu",)
    )


def test_preference_tolerance_preserves_tie_for_next_priority() -> None:
    """Tolerance may preserve near-equal points for later priorities."""
    analysis = analyse_multi_objective_candidates(
        candidates=(
            _candidate(
                candidate_id="slightly-faster",
                latency_ms=1.00,
                throughput=100.0,
            ),
            _candidate(
                candidate_id="slightly-slower",
                latency_ms=1.05,
                throughput=200.0,
            ),
        ),
        objectives=(
            OptimisationObjective(
                metric=ObjectiveMetric.MEAN_LATENCY_MS,
                direction=ObjectiveDirection.MINIMIZE,
            ),
            OptimisationObjective(
                metric=(
                    ObjectiveMetric.MEAN_CASES_PER_SECOND
                ),
                direction=ObjectiveDirection.MAXIMIZE,
            ),
        ),
        preference_policy=ParetoPreferencePolicy(
            priorities=(
                PreferencePriority(
                    metric=ObjectiveMetric.MEAN_LATENCY_MS,
                    relative_tolerance=0.10,
                ),
                PreferencePriority(
                    metric=(
                        ObjectiveMetric.MEAN_CASES_PER_SECOND
                    ),
                ),
            ),
        ),
    )

    assert (
        analysis.pareto_frontier_search_point_ids
        == (
            "slightly-faster::cpu",
            "slightly-slower::cpu",
        )
    )

    assert (
        analysis.preferred_frontier_search_point_ids
        == ("slightly-slower::cpu",)
    )


def test_no_preference_policy_preserves_full_frontier() -> None:
    """Pareto evidence must remain unranked when no priorities are declared."""
    analysis = analyse_multi_objective_candidates(
        candidates=(
            _candidate(
                candidate_id="latency",
                latency_ms=1.0,
                throughput=100.0,
            ),
            _candidate(
                candidate_id="throughput",
                latency_ms=2.0,
                throughput=200.0,
            ),
        ),
        objectives=(
            OptimisationObjective(
                metric=ObjectiveMetric.MEAN_LATENCY_MS,
                direction=ObjectiveDirection.MINIMIZE,
            ),
            OptimisationObjective(
                metric=(
                    ObjectiveMetric.MEAN_CASES_PER_SECOND
                ),
                direction=ObjectiveDirection.MAXIMIZE,
            ),
        ),
    )

    assert (
        analysis.preferred_frontier_search_point_ids
        == analysis.pareto_frontier_search_point_ids
    )


def test_preference_priority_must_be_declared_objective() -> None:
    """Preference policy cannot smuggle undeclared metrics into ranking."""
    with pytest.raises(
        ValueError,
        match="declared optimisation objectives",
    ):
        analyse_multi_objective_candidates(
            candidates=(
                _candidate(
                    candidate_id="candidate",
                    latency_ms=1.0,
                    throughput=100.0,
                ),
            ),
            objectives=(
                OptimisationObjective(
                    metric=ObjectiveMetric.MEAN_LATENCY_MS,
                    direction=ObjectiveDirection.MINIMIZE,
                ),
            ),
            preference_policy=ParetoPreferencePolicy(
                priorities=(
                    PreferencePriority(
                        metric=(
                            ObjectiveMetric.MODEL_SIZE_BYTES
                        ),
                    ),
                ),
            ),
        )


def test_all_constraint_rejected_is_valid_empty_frontier() -> None:
    """No feasible candidates is valid evidence, not an optimiser failure."""
    analysis = analyse_multi_objective_candidates(
        candidates=(
            _candidate(
                candidate_id="candidate-a",
                latency_ms=1.0,
                throughput=100.0,
                model_size_bytes=10_000,
            ),
            _candidate(
                candidate_id="candidate-b",
                latency_ms=2.0,
                throughput=200.0,
                model_size_bytes=20_000,
            ),
        ),
        objectives=(
            OptimisationObjective(
                metric=ObjectiveMetric.MEAN_LATENCY_MS,
                direction=ObjectiveDirection.MINIMIZE,
            ),
        ),
        constraints=(
            ObjectiveConstraint(
                metric=ObjectiveMetric.MODEL_SIZE_BYTES,
                operator=ConstraintOperator.AT_MOST,
                threshold=5_000.0,
            ),
        ),
        preference_policy=ParetoPreferencePolicy(
            priorities=(
                PreferencePriority(
                    metric=ObjectiveMetric.MEAN_LATENCY_MS,
                ),
            ),
        ),
    )

    assert (
        analysis.pareto_frontier_search_point_ids
        == ()
    )

    assert (
        analysis.preferred_frontier_search_point_ids
        == ()
    )

    assert (
        analysis.dominated_search_point_ids
        == ()
    )

    assert {
        rejection.search_point_id
        for rejection in analysis.constraint_rejections
    } == {
        "candidate-a::cpu",
        "candidate-b::cpu",
    }
