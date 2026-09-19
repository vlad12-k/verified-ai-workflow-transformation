"""Tests for admissibility-bound structured inference benchmarking."""

from datetime import UTC, datetime

import pytest
from pydantic import JsonValue

from vait.contracts.models import (
    Effect,
    RiskLevel,
    VerificationCase,
)
from vait.inference.models import (
    InferenceBenchmarkReport,
    InferenceConfiguration,
    InferenceEnvironment,
    InferenceLatencySummary,
    InferenceThroughputSummary,
)
from vait.optimisation.admissibility import (
    AdmissibilityIssue,
    AdmissibilityIssueCode,
    AdmissibilityStatus,
    SearchPointAdmissibility,
)
from vait.optimisation.search_space import (
    InferenceSearchPoint,
)
from vait.optimisation.structured_benchmark import (
    StructuredBenchmarkRun,
    run_structured_benchmark_series,
    summarize_structured_repetitions,
)
from vait.runners.python_runner import (
    PythonImplementationRunner,
)


def build_point(
    *,
    batch_size: int = 1,
    implementation_id: str = "candidate-v1",
) -> InferenceSearchPoint:
    """Build one structured search point."""
    return InferenceSearchPoint(
        search_point_id="candidate::cpu",
        search_space_id="structured-space",
        task_family="structured-decision",
        candidate_id="candidate",
        implementation_id=implementation_id,
        configuration_id="cpu",
        configuration=InferenceConfiguration(
            provider="local",
            runtime="python",
            device="cpu",
            dtype="native",
            batch_size=batch_size,
            model_id=None,
            model_revision=None,
        ),
        requires_verification=True,
    )


def admissible(
    point: InferenceSearchPoint,
) -> SearchPointAdmissibility:
    """Mark one test search point as already admissible."""
    return SearchPointAdmissibility(
        search_point=point,
        status=AdmissibilityStatus.ADMISSIBLE,
    )


def rejected(
    point: InferenceSearchPoint,
) -> SearchPointAdmissibility:
    """Build one rejected test search point."""
    return SearchPointAdmissibility(
        search_point=point,
        status=(
            AdmissibilityStatus.NOT_ADMISSIBLE
        ),
        issues=(
            AdmissibilityIssue(
                code=(
                    AdmissibilityIssueCode
                    .VERIFICATION_EVIDENCE_MISSING
                ),
                subject=point.search_point_id,
                message="Verification evidence is missing.",
            ),
        ),
    )


def candidate_function(
    data: dict[str, JsonValue],
) -> JsonValue:
    """Return one deterministic structured decision."""
    del data

    return {
        "decision": "REVIEW",
    }


def test_rejected_point_cannot_be_benchmarked() -> None:
    """M4-D must preserve VERIFY FIRST -> OPTIMISE SECOND."""
    point = build_point()

    runner = PythonImplementationRunner(
        implementation_id=(
            point.implementation_id
        ),
        function=candidate_function,
        declared_effects=frozenset(
            {Effect.NONE}
        ),
    )

    with pytest.raises(
        ValueError,
        match="admissible search point",
    ):
        run_structured_benchmark_series(
            admissibility=rejected(
                point
            ),
            runner=runner,
            cases=(
                VerificationCase(
                    id="case-1",
                    input_data={},
                    risk_level=RiskLevel.LOW,
                ),
            ),
            repetitions=1,
            warmup_rounds=0,
            measured_rounds=1,
        )


def test_runner_identity_must_match_admissible_point() -> None:
    """Evidence cannot be generated under another implementation identity."""
    point = build_point()

    runner = PythonImplementationRunner(
        implementation_id="other-v1",
        function=candidate_function,
    )

    with pytest.raises(
        ValueError,
        match="implementation_id",
    ):
        run_structured_benchmark_series(
            admissibility=admissible(
                point
            ),
            runner=runner,
            cases=(
                VerificationCase(
                    id="case-1",
                    input_data={},
                ),
            ),
            repetitions=1,
            warmup_rounds=0,
            measured_rounds=1,
        )


def test_scalar_path_rejects_fake_batch_configuration() -> None:
    """Scalar execution must not masquerade as native batching."""
    point = build_point(
        batch_size=8
    )

    runner = PythonImplementationRunner(
        implementation_id=(
            point.implementation_id
        ),
        function=candidate_function,
    )

    with pytest.raises(
        ValueError,
        match="batch_size=1",
    ):
        run_structured_benchmark_series(
            admissibility=admissible(
                point
            ),
            runner=runner,
            cases=(
                VerificationCase(
                    id="case-1",
                    input_data={},
                ),
            ),
            repetitions=1,
            warmup_rounds=0,
            measured_rounds=1,
        )


def test_admissible_point_runs_repeated_scalar_benchmark() -> None:
    """Repeated runs preserve admissibility and search-point provenance."""
    point = build_point()

    runner = PythonImplementationRunner(
        implementation_id=(
            point.implementation_id
        ),
        function=candidate_function,
    )

    cases = (
        VerificationCase(
            id="case-1",
            input_data={"amount": 10},
        ),
        VerificationCase(
            id="case-2",
            input_data={"amount": 20},
        ),
    )

    series = run_structured_benchmark_series(
        admissibility=admissible(
            point
        ),
        runner=runner,
        cases=cases,
        repetitions=2,
        warmup_rounds=0,
        measured_rounds=1,
    )

    assert len(series.runs) == 2

    assert (
        series.repeatability.repetitions
        == 2
    )

    assert (
        series.repeatability.measured_executions
        == 4
    )

    assert all(
        run.report.evidence_metadata[
            "search_point_id"
        ]
        == point.search_point_id
        for run in series.runs
    )

    assert all(
        run.report.evidence_metadata[
            "admissibility_gate"
        ]
        == "passed"
        for run in series.runs
    )

    assert all(
        run.report.evidence_metadata[
            "execution_mode"
        ]
        == "scalar-per-case"
        for run in series.runs
    )


def build_report(
    *,
    run_id: str,
    mean_ms: float,
    p95_ms: float,
    p99_ms: float,
    max_ms: float,
    cases_per_second: float,
) -> InferenceBenchmarkReport:
    """Build deterministic synthetic benchmark evidence."""
    return InferenceBenchmarkReport(
        run_id=run_id,
        created_at=datetime.now(
            UTC
        ),
        candidate_implementation_id=(
            "candidate-v1"
        ),
        task_family="structured-decision",
        warmup_iterations=0,
        measured_iterations=2,
        configuration=InferenceConfiguration(
            provider="local",
            runtime="python",
            device="cpu",
            dtype="native",
            batch_size=1,
            model_id=None,
            model_revision=None,
        ),
        latency=InferenceLatencySummary(
            count=2,
            mean_ms=mean_ms,
            p50_ms=mean_ms,
            p95_ms=p95_ms,
            p99_ms=p99_ms,
            max_ms=max_ms,
        ),
        throughput=InferenceThroughputSummary(
            cases_per_second=(
                cases_per_second
            ),
            requests_per_second=(
                cases_per_second
            ),
        ),
        environment=InferenceEnvironment(
            python_version="3.12.0",
            platform="test-platform",
            system="test-system",
            machine="test-machine",
            processor="test-processor",
        ),
    )


def test_repeatability_summary_aggregates_runs() -> None:
    """Cross-run evidence must expose latency and throughput dispersion."""
    runs = (
        StructuredBenchmarkRun(
            repetition_index=0,
            report=build_report(
                run_id="run-1",
                mean_ms=1.0,
                p95_ms=2.0,
                p99_ms=3.0,
                max_ms=3.0,
                cases_per_second=100.0,
            ),
        ),
        StructuredBenchmarkRun(
            repetition_index=1,
            report=build_report(
                run_id="run-2",
                mean_ms=3.0,
                p95_ms=4.0,
                p99_ms=5.0,
                max_ms=5.0,
                cases_per_second=200.0,
            ),
        ),
    )

    summary = (
        summarize_structured_repetitions(
            runs
        )
    )

    assert summary.repetitions == 2
    assert summary.measured_executions == 4
    assert (
        summary.weighted_mean_latency_ms
        == pytest.approx(2.0)
    )
    assert (
        summary.run_mean_latency_stdev_ms
        == pytest.approx(1.0)
    )
    assert (
        summary.run_mean_latency_cv
        == pytest.approx(0.5)
    )
    assert (
        summary.mean_cases_per_second
        == pytest.approx(150.0)
    )
    assert (
        summary.throughput_stdev
        == pytest.approx(50.0)
    )
    assert (
        summary.throughput_cv
        == pytest.approx(
            1.0 / 3.0
        )
    )
    assert (
        summary.worst_p95_latency_ms
        == 4.0
    )
    assert (
        summary.worst_p99_latency_ms
        == 5.0
    )
    assert (
        summary.worst_max_latency_ms
        == 5.0
    )


def test_single_repetition_does_not_invent_dispersion() -> None:
    """One run cannot establish cross-run repeatability dispersion."""
    runs = (
        StructuredBenchmarkRun(
            repetition_index=0,
            report=build_report(
                run_id="run-1",
                mean_ms=1.0,
                p95_ms=2.0,
                p99_ms=3.0,
                max_ms=3.0,
                cases_per_second=100.0,
            ),
        ),
    )

    summary = (
        summarize_structured_repetitions(
            runs
        )
    )

    assert (
        summary.run_mean_latency_stdev_ms
        is None
    )
    assert summary.run_mean_latency_cv is None
    assert summary.throughput_stdev is None
    assert summary.throughput_cv is None
