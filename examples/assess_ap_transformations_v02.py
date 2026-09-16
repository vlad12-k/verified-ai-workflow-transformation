"""Assess multiple supplied AP transformations without ranking them."""

from pathlib import Path

from pydantic import JsonValue

from vait.benchmark.loader import load_benchmark
from vait.benchmark.reporting import write_benchmark_report
from vait.benchmark.transformation_assessment import (
    assess_python_rule_transformations,
)
from vait.contracts.models import (
    BoundedVerificationPolicy,
    Effect,
    RiskLevel,
)
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


def build_low_risk_only_transformation() -> PythonRuleTransformation:
    """Build a candidate that is not applicable to critical-risk workflows."""
    return PythonRuleTransformation(
        descriptor=TransformationDescriptor(
            transformation_id="aaa-low-risk-only",
            version="1.0.0",
            name="Low-risk-only AP rule",
            description=(
                "Synthetic supplied candidate restricted to low and "
                "medium risk."
            ),
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
    """Build an intentionally unsafe supplied candidate."""
    return PythonRuleTransformation(
        descriptor=TransformationDescriptor(
            transformation_id="zzz-always-approve",
            version="1.0.0",
            name="Always approve",
            description=(
                "Intentionally unsafe synthetic candidate used to "
                "exercise rejection."
            ),
            category=TransformationCategory.OPTIMIZATION,
            declared_effects=frozenset({Effect.NONE}),
            supported_risk_levels=frozenset(RiskLevel),
            required_capabilities=frozenset({"typed-inputs"}),
        ),
        implementation_id="always-approve-ap-policy",
        function=always_approve,
    )


dataset = load_benchmark(
    Path("datasets/ap_invoice_exceptions/v0.2/cases.yaml")
)

report = assess_python_rule_transformations(
    dataset=dataset,
    transformations=(
        build_unsafe_transformation(),
        build_synthetic_ap_transformation(),
        build_low_risk_only_transformation(),
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

report_path = write_benchmark_report(
    report,
    Path(
        "artifacts/benchmarks/"
        "ap-v0.2-transformation-assessment.json"
    ),
)

print(f"Benchmark: {report.benchmark_id}")
print(f"Version: {report.benchmark_version}")
print(f"Transformations assessed: {len(report.assessments)}")

for assessment in report.assessments:
    if assessment.decision is None:
        issue_codes = ", ".join(
            issue.code.value
            for issue in assessment.applicability.issues
        )

        print(
            f"{assessment.canonical_id}: "
            f"NOT_APPLICABLE ({issue_codes})"
        )
        continue

    print(
        f"{assessment.canonical_id}: "
        f"{assessment.decision.value}"
    )

print(f"Report: {report_path}")
