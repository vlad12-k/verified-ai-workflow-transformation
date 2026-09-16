"""Applicability evaluation for VAIT workflow transformations."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, model_validator

from vait.contracts.models import RiskLevel
from vait.transformations.models import TransformationDescriptor


class ApplicabilityStatus(StrEnum):
    """Outcome of transformation applicability evaluation."""

    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"


class ApplicabilityIssueCode(StrEnum):
    """Machine-readable reasons why a transformation is not applicable."""

    UNSUPPORTED_RISK_LEVEL = "unsupported_risk_level"
    MISSING_CAPABILITY = "missing_capability"


class ApplicabilityContext(BaseModel):
    """Declared workflow context used for applicability checks."""

    risk_level: RiskLevel
    available_capabilities: frozenset[str] = frozenset()


class ApplicabilityIssue(BaseModel):
    """One machine-readable applicability failure."""

    code: ApplicabilityIssueCode
    message: str
    subject: str


class ApplicabilityResult(BaseModel):
    """Result of evaluating a transformation against a workflow context."""

    status: ApplicabilityStatus
    issues: tuple[ApplicabilityIssue, ...] = ()

    @model_validator(mode="after")
    def validate_status_consistency(self) -> Self:
        """Ensure status and issue collection cannot contradict each other."""
        if (
            self.status is ApplicabilityStatus.APPLICABLE
            and self.issues
        ):
            raise ValueError(
                "An applicable result cannot contain applicability issues."
            )

        if (
            self.status is ApplicabilityStatus.NOT_APPLICABLE
            and not self.issues
        ):
            raise ValueError(
                "A not-applicable result must contain at least one issue."
            )

        return self

    @property
    def is_applicable(self) -> bool:
        """Return whether the transformation may be attempted."""
        return self.status is ApplicabilityStatus.APPLICABLE


def evaluate_applicability(
    descriptor: TransformationDescriptor,
    context: ApplicabilityContext,
) -> ApplicabilityResult:
    """Evaluate descriptor-level applicability requirements."""
    issues: list[ApplicabilityIssue] = []

    if context.risk_level not in descriptor.supported_risk_levels:
        issues.append(
            ApplicabilityIssue(
                code=ApplicabilityIssueCode.UNSUPPORTED_RISK_LEVEL,
                subject=context.risk_level.value,
                message=(
                    f"Risk level '{context.risk_level.value}' is not "
                    f"supported by transformation "
                    f"'{descriptor.canonical_id}'."
                ),
            )
        )

    missing_capabilities = sorted(
        descriptor.required_capabilities
        - context.available_capabilities
    )

    for capability in missing_capabilities:
        issues.append(
            ApplicabilityIssue(
                code=ApplicabilityIssueCode.MISSING_CAPABILITY,
                subject=capability,
                message=(
                    f"Required capability '{capability}' is not available."
                ),
            )
        )

    if not issues:
        return ApplicabilityResult(
            status=ApplicabilityStatus.APPLICABLE,
        )

    return ApplicabilityResult(
        status=ApplicabilityStatus.NOT_APPLICABLE,
        issues=tuple(issues),
    )
