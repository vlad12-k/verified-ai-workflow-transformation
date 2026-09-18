"""Tests for executable verified inference search plans."""

import pytest

from vait.decision.models import Decision
from vait.inference.models import (
    InferenceConfiguration,
)
from vait.optimisation.admissibility import (
    AdmissibilityIssueCode,
    SearchPointVerificationEvidence,
    VerificationEvidenceKind,
)
from vait.optimisation.compatibility import (
    CompatibilityIssueCode,
    InferenceCompatibilityContext,
)
from vait.optimisation.search_plan import (
    build_executable_search_plan,
)
from vait.optimisation.search_space import (
    CandidateConfiguration,
    InferenceCandidate,
    InferenceSearchSpace,
    enumerate_search_points,
)


def build_configuration(
    *,
    device: str = "cpu",
) -> InferenceConfiguration:
    """Build one local structured-inference configuration."""
    return InferenceConfiguration(
        provider="local",
        runtime="python",
        device=device,
        dtype="native",
        batch_size=1,
        model_id=None,
        model_revision=None,
    )


def build_candidate(
    *,
    candidate_id: str,
    device: str = "cpu",
    requires_verification: bool = True,
) -> InferenceCandidate:
    """Build one candidate with one explicitly allowed configuration."""
    return InferenceCandidate(
        candidate_id=candidate_id,
        implementation_id=(
            f"{candidate_id}-v1"
        ),
        allowed_configurations=(
            CandidateConfiguration(
                configuration_id="default",
                configuration=(
                    build_configuration(
                        device=device
                    )
                ),
            ),
        ),
        requires_verification=(
            requires_verification
        ),
    )


def build_context() -> InferenceCompatibilityContext:
    """Build a deterministic CPU-only execution context."""
    return InferenceCompatibilityContext(
        available_providers=frozenset(
            {"local"}
        ),
        available_runtimes=frozenset(
            {"python"}
        ),
        available_devices=frozenset(
            {"cpu"}
        ),
        available_dtypes=frozenset(
            {"native"}
        ),
        max_batch_size=8,
    )


def exact_evidence(
    *,
    search_point_id: str,
    implementation_id: str,
    decision: Decision = Decision.EXACT,
) -> SearchPointVerificationEvidence:
    """Build deterministic verification evidence."""
    return SearchPointVerificationEvidence(
        search_point_id=search_point_id,
        implementation_id=implementation_id,
        contract_id=(
            f"contract-{search_point_id}"
        ),
        decision=decision,
        evidence_kind=(
            VerificationEvidenceKind.DETERMINISTIC
        ),
    )


def test_plan_partitions_search_space_deterministically() -> None:
    """Only compatible verified points may enter the executable partition."""
    search_space = InferenceSearchSpace(
        search_space_id="structured-v01",
        version="0.1.0",
        task_family="structured-decision",
        candidates=(
            build_candidate(
                candidate_id="d-cuda",
                device="cuda",
            ),
            build_candidate(
                candidate_id="c-missing",
            ),
            build_candidate(
                candidate_id="b-free",
                requires_verification=False,
            ),
            build_candidate(
                candidate_id="a-exact",
            ),
        ),
    )

    points = {
        point.candidate_id: point
        for point in enumerate_search_points(
            search_space
        )
    }

    exact_point = points["a-exact"]

    plan = build_executable_search_plan(
        search_space=search_space,
        compatibility_context=build_context(),
        verification_evidence_by_search_point={
            exact_point.search_point_id: (
                exact_evidence(
                    search_point_id=(
                        exact_point.search_point_id
                    ),
                    implementation_id=(
                        exact_point.implementation_id
                    ),
                )
            ),
        },
    )

    assert plan.total_registered_points == 4

    assert [
        point.search_point_id
        for point in plan.admissible_points
    ] == [
        "a-exact::default",
        "b-free::default",
    ]

    assert [
        result.search_point.search_point_id
        for result
        in plan.compatibility_rejections
    ] == [
        "d-cuda::default",
    ]

    assert [
        result.search_point.search_point_id
        for result
        in plan.admissibility_rejections
    ] == [
        "c-missing::default",
    ]

    assert (
        plan.compatibility_rejections[
            0
        ].issues[0].code
        is CompatibilityIssueCode.DEVICE_UNAVAILABLE
    )

    assert (
        plan.admissibility_rejections[
            0
        ].issues[0].code
        is (
            AdmissibilityIssueCode
            .VERIFICATION_EVIDENCE_MISSING
        )
    )


def test_rejected_verification_never_enters_plan() -> None:
    """A verification REJECT must remain outside benchmarking."""
    search_space = InferenceSearchSpace(
        search_space_id="reject-space",
        version="0.1.0",
        task_family="structured-decision",
        candidates=(
            build_candidate(
                candidate_id="rejected",
            ),
        ),
    )

    point = enumerate_search_points(
        search_space
    )[0]

    plan = build_executable_search_plan(
        search_space=search_space,
        compatibility_context=build_context(),
        verification_evidence_by_search_point={
            point.search_point_id: (
                exact_evidence(
                    search_point_id=(
                        point.search_point_id
                    ),
                    implementation_id=(
                        point.implementation_id
                    ),
                    decision=Decision.REJECT,
                )
            ),
        },
    )

    assert plan.admissible_points == ()
    assert (
        len(
            plan.admissibility_rejections
        )
        == 1
    )

    assert (
        plan.admissibility_rejections[
            0
        ].issues[0].code
        is (
            AdmissibilityIssueCode
            .VERIFICATION_REJECTED
        )
    )


def test_bounded_verified_point_is_executable() -> None:
    """BOUNDED statistical evidence may survive into execution."""
    search_space = InferenceSearchSpace(
        search_space_id="bounded-space",
        version="0.1.0",
        task_family="structured-decision",
        candidates=(
            build_candidate(
                candidate_id="bounded",
            ),
        ),
    )

    point = enumerate_search_points(
        search_space
    )[0]

    evidence = SearchPointVerificationEvidence(
        search_point_id=(
            point.search_point_id
        ),
        implementation_id=(
            point.implementation_id
        ),
        contract_id="contract-bounded",
        decision=Decision.BOUNDED,
        evidence_kind=(
            VerificationEvidenceKind
            .BOUNDED_STATISTICAL
        ),
        statistical_evidence_present=True,
    )

    plan = build_executable_search_plan(
        search_space=search_space,
        compatibility_context=build_context(),
        verification_evidence_by_search_point={
            point.search_point_id: evidence,
        },
    )

    assert (
        len(plan.admissible)
        == 1
    )
    assert (
        plan.admissible[
            0
        ].verification_evidence
        == evidence
    )


def test_unknown_verification_evidence_is_rejected() -> None:
    """Stale evidence may not silently attach to a search plan."""
    search_space = InferenceSearchSpace(
        search_space_id="known-space",
        version="0.1.0",
        task_family="structured-decision",
        candidates=(
            build_candidate(
                candidate_id="known",
            ),
        ),
    )

    evidence = SearchPointVerificationEvidence(
        search_point_id="unknown::default",
        implementation_id="unknown-v1",
        contract_id="contract-unknown",
        decision=Decision.EXACT,
        evidence_kind=(
            VerificationEvidenceKind.DETERMINISTIC
        ),
    )

    with pytest.raises(
        ValueError,
        match="unknown search points",
    ):
        build_executable_search_plan(
            search_space=search_space,
            compatibility_context=build_context(),
            verification_evidence_by_search_point={
                "unknown::default": evidence,
            },
        )


def test_incompatible_point_stops_before_admissibility() -> None:
    """Compatibility failure must short-circuit verification admission."""
    search_space = InferenceSearchSpace(
        search_space_id="cuda-space",
        version="0.1.0",
        task_family="structured-decision",
        candidates=(
            build_candidate(
                candidate_id="cuda-only",
                device="cuda",
            ),
        ),
    )

    plan = build_executable_search_plan(
        search_space=search_space,
        compatibility_context=build_context(),
    )

    assert plan.admissible == ()
    assert (
        len(
            plan.compatibility_rejections
        )
        == 1
    )
    assert (
        plan.admissibility_rejections
        == ()
    )
