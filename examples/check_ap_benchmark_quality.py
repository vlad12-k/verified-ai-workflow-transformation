"""Evaluate AP benchmark v0.1 against its versioned quality policy."""

from pathlib import Path

from vait.benchmark.loader import load_benchmark
from vait.benchmark.quality_policy import (
    evaluate_versioned_benchmark_quality,
    load_quality_policy,
)
from vait.benchmark.reporting import write_benchmark_report

dataset = load_benchmark(
    "datasets/ap_invoice_exceptions/v0.1/cases.yaml"
)

policy_document = load_quality_policy(
    "datasets/ap_invoice_exceptions/v0.1/quality-policy.yaml"
)

artifact = evaluate_versioned_benchmark_quality(
    dataset=dataset,
    document=policy_document,
)

report_path = write_benchmark_report(
    artifact,
    Path("artifacts/benchmarks/ap-v0.1-quality.json"),
)

quality = artifact.quality

print(f"Benchmark: {artifact.benchmark_id}")
print(f"Version: {artifact.benchmark_version}")
print(f"Cases: {quality.total_cases}")
print(f"Adversarial cases: {quality.adversarial_cases}")
print(f"Boundary cases: {quality.boundary_cases}")
print(f"Duplicate groups: {len(quality.duplicate_case_groups)}")
print(f"Quality gate: {'PASS' if quality.passed else 'FAIL'}")
print(f"Dataset SHA-256: {artifact.dataset_fingerprint_sha256}")
print(
    "Policy SHA-256: "
    f"{artifact.quality_policy_fingerprint_sha256}"
)
print(f"Report: {report_path}")

for issue in quality.issues:
    print(f"- {issue.code.value}: {issue.message}")
