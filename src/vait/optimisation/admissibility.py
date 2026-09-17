"""Verification-backed admissibility for inference search points."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, model_validator

from vait.decision.models import (
    Decision,
    FailureCode,
    VerificationResult,
)
from vait.optimisation.compatibility import (
    SearchPointCompatibility,
)
from vait.optimisation.search_space import (
    InferenceSearchPoint,
)
from vait.verification.models import (
    BoundedVerificationResult,
)


class VerificationEvidenceKind(StrEnum):
    """Verification regime that produced search-point evidence."""

    DETERMINISTIC = "deterministic"
    BOUNDED_STATISTICAL = "bounded_statistical"


class SearchPointVerificationEvidence(BaseModel):
    """Normalised verification evidence attached to one search point."""

    search_point_id: str = Field(min_length=1)
    implementation_id: str = Field(min_length=1)
    contract_id: str = Field(min_length=1)

    decision: Decision
    evidence_kind: VerificationEvidenceKind

    statistical_evidence_present: bool = False

    failure_codes: tuple[
        FailureCode,
        ...,
    ] = ()


class AdmissibilityStatus(StrEnum):
    """Whether a search point may enter performance optimisation."""

    ADMISSIBLE = "admissible"
    NOT_ADMISSIBLE = "not_admissible"


class AdmissibilityIssueCode(StrEnum):
    """Machine-readable reasons a search point is not admissible."""

    INCOMPATIBLE = "incompatible"
    VERIFICATION_EVIDENCE_MISSING = (
        "verification_evidence_missing"
    )
    VERIFICATION_IDENTITY_MISMATCH = (
        "verification_identity_mismatch"
    )
    VERIFICATION_REJECTED = "verification_rejected"
    DECISION_EVIDENCE_MISMATCH = (
        "decision_evidence_mismatch"
    )
    BOUNDED_EVIDENCE_MISSING = (
        "bounded_evidence_missing"
    )


class AdmissibilityIssue(BaseModel):
    """One explicit reason optimisation must exclude a search point."""

    code: AdmissibilityIssueCode
    subject: str = Field(min_length=1)
    message: str = Field(min_length=1)


class SearchPointAdmissibility(BaseModel):
    """Auditable admissibility decision for one search point."""

    search_point: InferenceSearchPoint

    status: AdmissibilityStatus

    verification_evidence: (
        SearchPointVerificationEvidence | None
    ) = None

    issues: tuple[
        AdmissibilityIssue,
        ...,
    ] = ()

    @model_validator(mode="after")
    def validate_status_consistency(
        self,
    ) -> Self:
        """Prevent contradictory admissibility evidence."""
        if (
            self.status
            is AdmissibilityStatus.ADMISSIBLE
            and self.issues
        ):
            raise ValueError(
                "An admissible search point cannot contain issues."
            )

        if (
            self.status
            is AdmissibilityStatus.NOT_ADMISSIBLE
            and not self.issues
        ):
            raise ValueError(
                "A not-admissible search point must contain "
                "at least one issue."
            )

        return self

    @property
    def is_admissible(self) -> bool:
        """Return whether the point may enter benchmarking."""
        return (
            self.status
            is AdmissibilityStatus.ADMISSIBLE
        )


def deterministic_verification_evidence(
    *,
    search_point: InferenceSearchPoint,
    result: VerificationResult,
) -> SearchPointVerificationEvidence:
    """Normalise deterministic verifier output for optimisation."""
    return SearchPointVerificationEvidence(
        search_point_id=search_point.search_point_id,
        implementation_id=search_point.implementation_id,
        contract_id=result.contract_id,
        decision=result.decision,
        evidence_kind=(
            VerificationEvidenceKind.DETERMINISTIC
        ),
        statistical_evidence_present=False,
        failure_codes=tuple(
            failure.code
            for failure in result.failures
        ),
    )


def bounded_verification_evidence(
    *,
    search_point: InferenceSearchPoint,
    result: BoundedVerificationResult,
) -> SearchPointVerificationEvidence:
    """Normalise bounded statistical verifier output for optimisation."""
    return SearchPointVerificationEvidence(
        search_point_id=search_point.search_point_id,
        implementation_id=search_point.implementation_id,
        contract_id=result.contract_id,
        decision=result.decision,
        evidence_kind=(
            VerificationEvidenceKind.BOUNDED_STATISTICAL
        ),
        statistical_evidence_present=(
            result.statistical_evidence is not None
        ),
        failure_codes=tuple(
            failure.code
            for failure in result.failures
        ),
    )


def evaluate_search_point_admissibility(
    *,
    compatibility: SearchPointCompatibility,
    verification_evidence: (
        SearchPointVerificationEvidence | None
    ) = None,
) -> SearchPointAdmissibility:
    """Gate optimisation on compatibility and accepted verification."""
    search_point = compatibility.search_point

    if not compatibility.is_compatible:
        issue_codes = ",".join(
            issue.code.value
            for issue in compatibility.issues
        )

        return SearchPointAdmissibility(
            search_point=search_point,
            status=(
                AdmissibilityStatus.NOT_ADMISSIBLE
            ),
            verification_evidence=verification_evidence,
            issues=(
                AdmissibilityIssue(
                    code=(
                        AdmissibilityIssueCode.INCOMPATIBLE
                    ),
                    subject=search_point.search_point_id,
                    message=(
                        "Search point failed compatibility "
                        f"checks: {issue_codes}."
                    ),
                ),
            ),
        )

    if (
        verification_evidence is None
        and not search_point.requires_verification
    ):
        return SearchPointAdmissibility(
            search_point=search_point,
            status=AdmissibilityStatus.ADMISSIBLE,
        )

    if verification_evidence is None:
        return SearchPointAdmissibility(
            search_point=search_point,
            status=(
                AdmissibilityStatus.NOT_ADMISSIBLE
            ),
            issues=(
                AdmissibilityIssue(
                    code=(
                        AdmissibilityIssueCode
                        .VERIFICATION_EVIDENCE_MISSING
                    ),
                    subject=search_point.search_point_id,
                    message=(
                        "Search point requires verification "
                        "before performance optimisation."
                    ),
                ),
            ),
        )

    identity_issues: list[
        AdmissibilityIssue
    ] = []

    if (
        verification_evidence.search_point_id
        != search_point.search_point_id
    ):
        identity_issues.append(
            AdmissibilityIssue(
                code=(
                    AdmissibilityIssueCode
                    .VERIFICATION_IDENTITY_MISMATCH
                ),
                subject=(
                    verification_evidence.search_point_id
                ),
                message=(
                    "Verification evidence search_point_id "
                    "does not match the evaluated search point."
                ),
            )
        )

    if (
        verification_evidence.implementation_id
        != search_point.implementation_id
    ):
        identity_issues.append(
            AdmissibilityIssue(
                code=(
                    AdmissibilityIssueCode
                    .VERIFICATION_IDENTITY_MISMATCH
                ),
                subject=(
                    verification_evidence.implementation_id
                ),
                message=(
                    "Verification evidence implementation_id "
                    "does not match the evaluated search point."
                ),
            )
        )

    if identity_issues:
        return SearchPointAdmissibility(
            search_point=search_point,
            status=(
                AdmissibilityStatus.NOT_ADMISSIBLE
            ),
            verification_evidence=verification_evidence,
            issues=tuple(identity_issues),
        )

    if (
        verification_evidence.decision
        is Decision.REJECT
    ):
        return SearchPointAdmissibility(
            search_point=search_point,
            status=(
                AdmissibilityStatus.NOT_ADMISSIBLE
            ),
            verification_evidence=verification_evidence,
            issues=(
                AdmissibilityIssue(
                    code=(
                        AdmissibilityIssueCode
                        .VERIFICATION_REJECTED
                    ),
                    subject=(
                        verification_evidence.contract_id
                    ),
                    message=(
                        "Verification decision is REJECT; "
                        "performance cannot make this search "
                        "point admissible."
                    ),
                ),
            ),
        )

    if (
        verification_evidence.decision
        is Decision.EXACT
        and verification_evidence.evidence_kind
        is not VerificationEvidenceKind.DETERMINISTIC
    ):
        return SearchPointAdmissibility(
            search_point=search_point,
            status=(
                AdmissibilityStatus.NOT_ADMISSIBLE
            ),
            verification_evidence=verification_evidence,
            issues=(
                AdmissibilityIssue(
                    code=(
                        AdmissibilityIssueCode
                        .DECISION_EVIDENCE_MISMATCH
                    ),
                    subject=(
                        verification_evidence.evidence_kind.value
                    ),
                    message=(
                        "EXACT admissibility requires "
                        "deterministic verification evidence."
                    ),
                ),
            ),
        )

    if (
        verification_evidence.decision
        is Decision.BOUNDED
        and verification_evidence.evidence_kind
        is not VerificationEvidenceKind.BOUNDED_STATISTICAL
    ):
        return SearchPointAdmissibility(
            search_point=search_point,
            status=(
                AdmissibilityStatus.NOT_ADMISSIBLE
            ),
            verification_evidence=verification_evidence,
            issues=(
                AdmissibilityIssue(
                    code=(
                        AdmissibilityIssueCode
                        .DECISION_EVIDENCE_MISMATCH
                    ),
                    subject=(
                        verification_evidence.evidence_kind.value
                    ),
                    message=(
                        "BOUNDED admissibility requires "
                        "bounded statistical verification evidence."
                    ),
                ),
            ),
        )

    if (
        verification_evidence.decision
        is Decision.BOUNDED
        and not (
            verification_evidence
            .statistical_evidence_present
        )
    ):
        return SearchPointAdmissibility(
            search_point=search_point,
            status=(
                AdmissibilityStatus.NOT_ADMISSIBLE
            ),
            verification_evidence=verification_evidence,
            issues=(
                AdmissibilityIssue(
                    code=(
                        AdmissibilityIssueCode
                        .BOUNDED_EVIDENCE_MISSING
                    ),
                    subject=(
                        verification_evidence.contract_id
                    ),
                    message=(
                        "BOUNDED decision lacks statistical "
                        "evidence required for admissibility."
                    ),
                ),
            ),
        )

    return SearchPointAdmissibility(
        search_point=search_point,
        status=AdmissibilityStatus.ADMISSIBLE,
        verification_evidence=verification_evidence,
    )
