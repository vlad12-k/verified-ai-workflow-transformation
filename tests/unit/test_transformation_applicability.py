"""Tests for VAIT transformation applicability evaluation."""

import pytest
from pydantic import ValidationError

from vait.contracts.models import RiskLevel
from vait.transformations.applicability import (
    ApplicabilityContext,
    ApplicabilityIssue,
    ApplicabilityIssueCode,
    ApplicabilityResult,
    ApplicabilityStatus,
    evaluate_applicability,
)
from vait.transformations.models import (
    TransformationCategory,
    TransformationDescriptor,
)


def build_descriptor(
    *,
    supported_risk_levels: frozenset[RiskLevel] | None = None,
    required_capabilities: frozenset[str] | None = None,
) -> TransformationDescriptor:
    """Create a descriptor for applicability tests."""
    return TransformationDescriptor(
        transformation_id="test-transform",
        version="1.0.0",
        name="Test transformation",
        description="Transformation used by applicability tests.",
        category=TransformationCategory.OPTIMIZATION,
        supported_risk_levels=(
            supported_risk_levels
            if supported_risk_levels is not None
            else frozenset(RiskLevel)
        ),
        required_capabilities=(
            required_capabilities
            if required_capabilities is not None
            else frozenset()
        ),
    )


def test_transformation_is_applicable_when_requirements_are_met() -> None:
    """Satisfied risk and capability requirements should allow evaluation."""
    descriptor = build_descriptor(
        supported_risk_levels=frozenset(
            {
                RiskLevel.LOW,
                RiskLevel.MEDIUM,
            }
        ),
        required_capabilities=frozenset(
            {
                "typed-inputs",
                "deterministic-output",
            }
        ),
    )

    context = ApplicabilityContext(
        risk_level=RiskLevel.MEDIUM,
        available_capabilities=frozenset(
            {
                "typed-inputs",
                "deterministic-output",
                "extra-capability",
            }
        ),
    )

    result = evaluate_applicability(
        descriptor,
        context,
    )

    assert result.status is ApplicabilityStatus.APPLICABLE
    assert result.is_applicable is True
    assert result.issues == ()


def test_unsupported_risk_level_is_not_applicable() -> None:
    """Unsupported business risk should block a transformation attempt."""
    descriptor = build_descriptor(
        supported_risk_levels=frozenset(
            {
                RiskLevel.LOW,
                RiskLevel.MEDIUM,
            }
        )
    )

    result = evaluate_applicability(
        descriptor,
        ApplicabilityContext(
            risk_level=RiskLevel.CRITICAL,
        ),
    )

    assert result.status is ApplicabilityStatus.NOT_APPLICABLE
    assert result.is_applicable is False
    assert len(result.issues) == 1
    assert (
        result.issues[0].code
        is ApplicabilityIssueCode.UNSUPPORTED_RISK_LEVEL
    )
    assert result.issues[0].subject == "critical"


def test_missing_capability_is_not_applicable() -> None:
    """Missing required capabilities should block applicability."""
    descriptor = build_descriptor(
        required_capabilities=frozenset(
            {
                "model-inference",
            }
        )
    )

    result = evaluate_applicability(
        descriptor,
        ApplicabilityContext(
            risk_level=RiskLevel.LOW,
        ),
    )

    assert result.status is ApplicabilityStatus.NOT_APPLICABLE
    assert len(result.issues) == 1
    assert (
        result.issues[0].code
        is ApplicabilityIssueCode.MISSING_CAPABILITY
    )
    assert result.issues[0].subject == "model-inference"


def test_all_missing_capabilities_are_reported_deterministically() -> None:
    """Missing capability issues should have deterministic ordering."""
    descriptor = build_descriptor(
        required_capabilities=frozenset(
            {
                "z-capability",
                "a-capability",
                "m-capability",
            }
        )
    )

    result = evaluate_applicability(
        descriptor,
        ApplicabilityContext(
            risk_level=RiskLevel.LOW,
        ),
    )

    assert [
        issue.subject
        for issue in result.issues
    ] == [
        "a-capability",
        "m-capability",
        "z-capability",
    ]


def test_multiple_applicability_failures_are_preserved() -> None:
    """Independent applicability failures should be reported together."""
    descriptor = build_descriptor(
        supported_risk_levels=frozenset(
            {
                RiskLevel.LOW,
            }
        ),
        required_capabilities=frozenset(
            {
                "typed-inputs",
            }
        ),
    )

    result = evaluate_applicability(
        descriptor,
        ApplicabilityContext(
            risk_level=RiskLevel.HIGH,
        ),
    )

    assert result.status is ApplicabilityStatus.NOT_APPLICABLE
    assert {
        issue.code
        for issue in result.issues
    } == {
        ApplicabilityIssueCode.UNSUPPORTED_RISK_LEVEL,
        ApplicabilityIssueCode.MISSING_CAPABILITY,
    }


def test_applicable_result_cannot_contain_issues() -> None:
    """Applicability result models should reject contradictory state."""
    with pytest.raises(
        ValidationError,
        match="cannot contain applicability issues",
    ):
        ApplicabilityResult(
            status=ApplicabilityStatus.APPLICABLE,
            issues=(
                ApplicabilityIssue(
                    code=ApplicabilityIssueCode.MISSING_CAPABILITY,
                    subject="model-inference",
                    message="Capability is unavailable.",
                ),
            ),
        )


def test_not_applicable_result_requires_an_issue() -> None:
    """A failed applicability result must explain why it failed."""
    with pytest.raises(
        ValidationError,
        match="must contain at least one issue",
    ):
        ApplicabilityResult(
            status=ApplicabilityStatus.NOT_APPLICABLE,
        )
