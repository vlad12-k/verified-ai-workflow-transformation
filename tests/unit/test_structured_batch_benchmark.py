"""Tests for native structured batch inference benchmarking."""

import pytest
from pydantic import JsonValue

from vait.contracts.models import (
    RiskLevel,
    VerificationCase,
)
from vait.inference.models import (
    InferenceConfiguration,
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
from vait.optimisation.structured_batch_benchmark import (
    run_structured_native_batch_series,
)


def _point(
    *,
    batch_size: int = 2,
) -> InferenceSearchPoint:
    """Build one native-batch structured search point."""
    return InferenceSearchPoint(
        search_point_id=(
            f"candidate::batch-{batch_size}"
        ),
        search_space_id="m4d-test",
        task_family="structured-decision",
        candidate_id="candidate",
        implementation_id="candidate-v1",
        configuration_id=(
            f"batch-{batch_size}"
        ),
        configuration=InferenceConfiguration(
            provider="local",
            runtime="native-test",
            device="cpu",
            dtype="native",
            batch_size=batch_size,
            model_id=None,
            model_revision=None,
        ),
        requires_verification=False,
    )


def _admissible(
    *,
    batch_size: int = 2,
) -> SearchPointAdmissibility:
    """Build an admissible native-batch search point."""
    return SearchPointAdmissibility(
        search_point=_point(
            batch_size=batch_size
        ),
        status=(
            AdmissibilityStatus.ADMISSIBLE
        ),
    )


def _cases() -> tuple[
    VerificationCase,
    ...,
]:
    """Build five deterministic structured cases."""
    return tuple(
        VerificationCase(
            id=f"case-{index}",
            input_data={
                "value": index,
            },
            risk_level=RiskLevel.LOW,
        )
        for index in range(5)
    )


def test_native_batch_series_records_scaling_evidence() -> None:
    """Native batching must record batch and case throughput evidence."""
    observed_batch_sizes: list[
        int
    ] = []

    def operation(
        batch,
    ) -> tuple[
        JsonValue,
        ...,
    ]:
        observed_batch_sizes.append(
            len(batch)
        )

        return tuple(
            {
                "decision": "REVIEW",
            }
            for _ in batch
        )

    series = (
        run_structured_native_batch_series(
            admissibility=_admissible(
                batch_size=2
            ),
            implementation_id=(
                "candidate-v1"
            ),
            operation=operation,
            cases=_cases(),
            repetitions=1,
            warmup_rounds=1,
            measured_rounds=2,
        )
    )

    assert len(
        series.runs
    ) == 1

    report = (
        series.runs[0].report
    )

    assert (
        report.configuration.batch_size
        == 2
    )

    assert (
        report.measured_iterations
        == 6
    )

    assert (
        report.evidence_metadata[
            "execution_mode"
        ]
        == "native-batch"
    )

    assert (
        report.evidence_metadata[
            "batch_invocations_per_round"
        ]
        == 3
    )

    assert (
        report.evidence_metadata[
            "measured_cases"
        ]
        == 10
    )

    assert (
        report.evidence_metadata[
            "tail_batch_size"
        ]
        == 1
    )

    assert (
        report.throughput.cases_per_second
        is not None
    )

    assert (
        report.throughput.requests_per_second
        is not None
    )

    assert set(
        observed_batch_sizes
    ) == {
        1,
        2,
    }


def test_native_batch_requires_admissible_point() -> None:
    """Rejected optimisation points must never enter batch benchmarking."""
    point = _point()

    rejected = (
        SearchPointAdmissibility(
            search_point=point,
            status=(
                AdmissibilityStatus
                .NOT_ADMISSIBLE
            ),
            issues=(
                AdmissibilityIssue(
                    code=(
                        AdmissibilityIssueCode
                        .VERIFICATION_REJECTED
                    ),
                    subject=(
                        point.search_point_id
                    ),
                    message=(
                        "Verification rejected."
                    ),
                ),
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="admissible",
    ):
        run_structured_native_batch_series(
            admissibility=rejected,
            implementation_id=(
                "candidate-v1"
            ),
            operation=lambda batch: (
                tuple(
                    None
                    for _ in batch
                )
            ),
            cases=_cases(),
            repetitions=1,
            warmup_rounds=0,
            measured_rounds=1,
        )


def test_native_batch_identity_must_match_search_point() -> None:
    """Batch execution cannot benchmark a different implementation."""
    with pytest.raises(
        ValueError,
        match="implementation_id",
    ):
        run_structured_native_batch_series(
            admissibility=_admissible(),
            implementation_id=(
                "different-v1"
            ),
            operation=lambda batch: (
                tuple(
                    None
                    for _ in batch
                )
            ),
            cases=_cases(),
            repetitions=1,
            warmup_rounds=0,
            measured_rounds=1,
        )


def test_native_batch_rejects_output_count_mismatch() -> None:
    """A batch operation must return one result per input case."""
    with pytest.raises(
        RuntimeError,
        match="different number",
    ):
        run_structured_native_batch_series(
            admissibility=_admissible(),
            implementation_id=(
                "candidate-v1"
            ),
            operation=lambda batch: (),
            cases=_cases(),
            repetitions=1,
            warmup_rounds=0,
            measured_rounds=1,
        )
