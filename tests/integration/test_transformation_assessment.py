"""Integration tests for verification-aware transformation assessment."""

from pathlib import Path

from pydantic import JsonValue

from vait.benchmark.loader import load_benchmark
from vait.benchmark.transformation_assessment import (
    assess_python_rule_transformations,
)
from vait.contracts.models import (
    BoundedVerificationPolicy,
    Effect,
    RiskLevel,
)
from vait.decision.models import Decision
from vait.transformations.applicability import ApplicabilityContext
from vait.transformations.library.synthetic_ap import (
    build_synthetic_ap_transformation,
    synthetic_ap_policy,
)
from vait.transformations.models import (
    TransformationCategory,
    TransformationDescriptor,
)
from vait.transformations.python_rule import PythonRuleTransformation


def always_approve(_: dict[str, JsonValue]) -> JsonValue:
    """Return an intentionally unsafe benchmark candidate."""
    return {"decision": "RECOMMEND_APPROVE"}


def build_not_applicable_transformation() -> PythonRuleTransformation:
    """Build a transformation that excludes critical-risk workflows."""
    return PythonRuleTransformation(
        descriptor=TransformationDescriptor(
            transformation_id="aaa-low-risk-only",
            version="1.0.0",
            name="Low-risk-only AP rule",
            description="Synthetic applicability test transformation.",
            category=TransformationCategory.OPTIMIZATION,
            declared_effects=frozenset({Effect.NONE}),
            supported_risk_levels=frozenset(
                {
                    RiskLevel.LOW,
                    RiskLevel.MEDIUM,
                }
            ),
            required_capabilities=frozenset({"typed-inputs"}),
        ),
        implementation_id="low-risk-only-ap-policy",
        function=synthetic_ap_policy,
    )


def build_unsafe_transformation() -> PythonRuleTransformation:
    """Build an intentionally unsafe candidate for rejection testing."""
    return PythonRuleTransformation(
        descriptor=TransformationDescriptor(
            transformation_id="zzz-always-approve",
            version="1.0.0",
            name="Always approve",
            description="Intentionally unsafe benchmark test candidate.",
            category=TransformationCategory.OPTIMIZATION,
            declared_effects=frozenset({Effect.NONE}),
            supported_risk_levels=frozenset(RiskLevel),
            required_capabilities=frozenset({"typed-inputs"}),
        ),
        implementation_id="always-approve-ap-policy",
        function=always_approve,
    )


def test_candidates_are_assessed_without_ranking() -> None:
    """Candidates should retain independent applicability and verification."""
    dataset = load_benchmark(
        Path("datasets/ap_invoice_exceptions/v0.2/cases.yaml")
    )

    report = assess_python_rule_transformations(
        dataset=dataset,
        transformations=(
            build_unsafe_transformation(),
            build_synthetic_ap_transformation(),
            build_not_applicable_transformation(),
        ),
        context=ApplicabilityContext(
            risk_level=RiskLevel.CRITICAL,
            available_capabilities=frozenset({"typed-inputs"}),
        ),
        policy=BoundedVerificationPolicy(
            max_overall_disagreement_rate=0.15,
            max_high_risk_disagreement_rate=0.25,
            confidence_level=0.95,
            min_total_cases=20,
            min_high_risk_cases=11,
        ),
    )

    assert report.benchmark_id == "ap-invoice-exceptions"
    assert report.benchmark_version == "0.2"
    assert len(report.assessments) == 3

    assert [
        assessment.canonical_id
        for assessment in report.assessments
    ] == [
        "aaa-low-risk-only@1.0.0",
        "synthetic-ap-rule-replacement@1.0.0",
        "zzz-always-approve@1.0.0",
    ]

    not_applicable, accepted, rejected = report.assessments

    assert not_applicable.applicability.is_applicable is False
    assert not_applicable.verification is None
    assert not_applicable.decision is None

    assert accepted.applicability.is_applicable is True
    assert accepted.decision is Decision.BOUNDED
    assert accepted.verification is not None
    assert accepted.verification.failures == []

    assert rejected.applicability.is_applicable is True
    assert rejected.decision is Decision.REJECT
    assert rejected.verification is not None
    assert rejected.verification.failures
