"""Evaluate the AP benchmark against its declared quality baseline."""

from vait.benchmark.loader import load_benchmark
from vait.benchmark.quality import (
    BenchmarkQualityPolicy,
    evaluate_benchmark_quality,
)
from vait.contracts.models import RiskLevel

dataset = load_benchmark(
    "datasets/ap_invoice_exceptions/v0.1/cases.yaml"
)

policy = BenchmarkQualityPolicy(
    min_total_cases=12,
    min_cases_per_risk={
        RiskLevel.LOW: 2,
        RiskLevel.MEDIUM: 4,
        RiskLevel.HIGH: 4,
        RiskLevel.CRITICAL: 2,
    },
    required_tags=frozenset(
        {
            "adversarial",
            "boundary",
            "amount-mismatch",
            "fraud-indicator",
            "compliance",
        }
    ),
    min_adversarial_cases=2,
    min_boundary_cases=2,
)

report = evaluate_benchmark_quality(
    dataset=dataset,
    policy=policy,
)

print(f"Benchmark: {report.benchmark_id}")
print(f"Version: {report.benchmark_version}")
print(f"Cases: {report.total_cases}")
print(f"Adversarial cases: {report.adversarial_cases}")
print(f"Boundary cases: {report.boundary_cases}")
print(f"Duplicate groups: {len(report.duplicate_case_groups)}")
print(f"Quality gate: {'PASS' if report.passed else 'FAIL'}")

for issue in report.issues:
    print(f"- {issue.code.value}: {issue.message}")
