"""Evaluate AP benchmark v0.2 against its expanded quality baseline."""

from vait.benchmark.loader import load_benchmark
from vait.benchmark.quality import (
    BenchmarkQualityPolicy,
    evaluate_benchmark_quality,
)
from vait.contracts.models import RiskLevel

dataset = load_benchmark(
    "datasets/ap_invoice_exceptions/v0.2/cases.yaml"
)

report = evaluate_benchmark_quality(
    dataset=dataset,
    policy=BenchmarkQualityPolicy(
        min_total_cases=20,
        min_cases_per_risk={
            RiskLevel.LOW: 3,
            RiskLevel.MEDIUM: 6,
            RiskLevel.HIGH: 7,
            RiskLevel.CRITICAL: 4,
        },
        required_tags=frozenset(
            {
                "adversarial",
                "boundary",
                "precision",
                "inconsistent-input",
                "precedence",
                "compliance",
                "fraud-indicator",
            }
        ),
        min_adversarial_cases=9,
        min_boundary_cases=5,
    ),
)

print(f"Benchmark: {report.benchmark_id}")
print(f"Version: {report.benchmark_version}")
print(f"Cases: {report.total_cases}")
print(f"Risk counts: {report.risk_counts}")
print(f"Adversarial cases: {report.adversarial_cases}")
print(f"Boundary cases: {report.boundary_cases}")
print(f"Duplicate groups: {len(report.duplicate_case_groups)}")
print(f"Quality gate: {'PASS' if report.passed else 'FAIL'}")

for issue in report.issues:
    print(f"- {issue.code.value}: {issue.message}")
