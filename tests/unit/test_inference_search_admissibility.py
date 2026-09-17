"""Tests for verification-backed search-point admissibility."""

from vait.decision.models import (
    Decision,
    VerificationResult,
)
from vait.inference.models import (
    InferenceConfiguration,
)
from vait.optimisation.admissibility import (
    AdmissibilityIssueCode,
    AdmissibilityStatus,
    SearchPointVerificationEvidence,
    VerificationEvidenceKind,
    bounded_verification_evidence,
    deterministic_verification_evidence,
    evaluate_search_point_admissibility,
)
from vait.optimisation.compatibility import (
    InferenceCompatibilityContext,
    evaluate_search_point_compatibility,
)
from vait.optimisation.search_space import (
    InferenceSearchPoint,
)
from vait.verification.models import (
    BoundedVerificationResult,
    StatisticalEvidence,
)


def build_point(
    *,
    requires_verification: bool = True,
    device: str = "cpu",
) -> InferenceSearchPoint:
    """Build one optimisation search point."""
    return InferenceSearchPoint(
        search_point_id="candidate::cpu",
        search_space_id="test-space",
        task_family="structured-decision",
        candidate_id="candidate",
        implementation_id="candidate-v1",
        configuration_id="cpu",
        configuration=InferenceConfiguration(
            provider="local",
            runtime="python",
            device=device,
            dtype="native",
            batch_size=1,
            model_id=None,
            model_revision=None,
        ),
        requires_verification=requires_verification,
    )


def compatibility_for(
    point: InferenceSearchPoint,
):
    """Evaluate one point against a deterministic CPU context."""
    return evaluate_search_point_compatibility(
        point,
        InferenceCompatibilityContext(
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
        ),
    )


def build_statistical_evidence() -> StatisticalEvidence:
    """Build minimal accepted bounded statistical evidence."""
    return StatisticalEvidence(
        cases_evaluated=100,
        disagreements=1,
        disagreement_rate=0.01,
        disagreement_upper_bound=0.04,
        confidence_level=0.95,
        high_risk_cases_evaluated=20,
        high_risk_disagreements=0,
        high_risk_disagreement_rate=0.0,
        high_risk_disagreement_upper_bound=0.14,
    )


def test_exact_verified_point_is_admissible() -> None:
    """EXACT deterministic evidence may enter optimisation."""
    point = build_point()

    evidence = deterministic_verification_evidence(
        search_point=point,
        result=VerificationResult(
            contract_id="contract-exact",
            decision=Decision.EXACT,
            cases_evaluated=20,
            exact_matches=20,
        ),
    )

    result = evaluate_search_point_admissibility(
        compatibility=compatibility_for(
            point
        ),
        verification_evidence=evidence,
    )

    assert (
        result.status
        is AdmissibilityStatus.ADMISSIBLE
    )
    assert result.is_admissible is True
    assert result.issues == ()


def test_bounded_with_statistical_evidence_is_admissible() -> None:
    """Accepted bounded evidence may enter optimisation."""
    point = build_point()

    evidence = bounded_verification_evidence(
        search_point=point,
        result=BoundedVerificationResult(
            contract_id="contract-bounded",
            decision=Decision.BOUNDED,
            statistical_evidence=(
                build_statistical_evidence()
            ),
        ),
    )

    result = evaluate_search_point_admissibility(
        compatibility=compatibility_for(
            point
        ),
        verification_evidence=evidence,
    )

    assert result.is_admissible is True
    assert (
        evidence.statistical_evidence_present
        is True
    )


def test_bounded_without_statistical_evidence_is_not_admissible() -> None:
    """A BOUNDED label alone is insufficient evidence."""
    point = build_point()

    evidence = bounded_verification_evidence(
        search_point=point,
        result=BoundedVerificationResult(
            contract_id="contract-bounded",
            decision=Decision.BOUNDED,
            statistical_evidence=None,
        ),
    )

    result = evaluate_search_point_admissibility(
        compatibility=compatibility_for(
            point
        ),
        verification_evidence=evidence,
    )

    assert result.is_admissible is False
    assert (
        result.issues[0].code
        is AdmissibilityIssueCode.BOUNDED_EVIDENCE_MISSING
    )


def test_reject_decision_is_never_admissible() -> None:
    """Performance must never override verification REJECT."""
    point = build_point()

    evidence = deterministic_verification_evidence(
        search_point=point,
        result=VerificationResult(
            contract_id="contract-reject",
            decision=Decision.REJECT,
            cases_evaluated=20,
            exact_matches=10,
        ),
    )

    result = evaluate_search_point_admissibility(
        compatibility=compatibility_for(
            point
        ),
        verification_evidence=evidence,
    )

    assert result.is_admissible is False
    assert (
        result.issues[0].code
        is AdmissibilityIssueCode.VERIFICATION_REJECTED
    )


def test_required_verification_cannot_be_missing() -> None:
    """Required evidence must exist before benchmarking."""
    point = build_point(
        requires_verification=True
    )

    result = evaluate_search_point_admissibility(
        compatibility=compatibility_for(
            point
        )
    )

    assert result.is_admissible is False
    assert (
        result.issues[0].code
        is (
            AdmissibilityIssueCode
            .VERIFICATION_EVIDENCE_MISSING
        )
    )


def test_optional_verification_can_be_admissible_without_evidence() -> None:
    """Explicitly verification-free points need no invented evidence."""
    point = build_point(
        requires_verification=False
    )

    result = evaluate_search_point_admissibility(
        compatibility=compatibility_for(
            point
        )
    )

    assert result.is_admissible is True
    assert result.verification_evidence is None


def test_incompatible_point_is_never_admissible() -> None:
    """Compatibility rejection happens before verification/performance."""
    point = build_point(
        device="cuda"
    )

    evidence = deterministic_verification_evidence(
        search_point=point,
        result=VerificationResult(
            contract_id="contract-exact",
            decision=Decision.EXACT,
            cases_evaluated=20,
            exact_matches=20,
        ),
    )

    result = evaluate_search_point_admissibility(
        compatibility=compatibility_for(
            point
        ),
        verification_evidence=evidence,
    )

    assert result.is_admissible is False
    assert (
        result.issues[0].code
        is AdmissibilityIssueCode.INCOMPATIBLE
    )


def test_verification_identity_must_match_search_point() -> None:
    """Evidence from another search point cannot be reused."""
    point = build_point()

    evidence = SearchPointVerificationEvidence(
        search_point_id="other::cpu",
        implementation_id="other-v1",
        contract_id="contract-exact",
        decision=Decision.EXACT,
        evidence_kind=(
            VerificationEvidenceKind.DETERMINISTIC
        ),
    )

    result = evaluate_search_point_admissibility(
        compatibility=compatibility_for(
            point
        ),
        verification_evidence=evidence,
    )

    assert result.is_admissible is False

    assert all(
        issue.code
        is (
            AdmissibilityIssueCode
            .VERIFICATION_IDENTITY_MISMATCH
        )
        for issue in result.issues
    )

    assert len(result.issues) == 2


def test_decision_must_match_evidence_kind() -> None:
    """EXACT cannot be backed by bounded-statistical evidence."""
    point = build_point()

    evidence = SearchPointVerificationEvidence(
        search_point_id=point.search_point_id,
        implementation_id=point.implementation_id,
        contract_id="contract-mismatch",
        decision=Decision.EXACT,
        evidence_kind=(
            VerificationEvidenceKind.BOUNDED_STATISTICAL
        ),
        statistical_evidence_present=True,
    )

    result = evaluate_search_point_admissibility(
        compatibility=compatibility_for(
            point
        ),
        verification_evidence=evidence,
    )

    assert result.is_admissible is False
    assert (
        result.issues[0].code
        is (
            AdmissibilityIssueCode
            .DECISION_EVIDENCE_MISMATCH
        )
    )
