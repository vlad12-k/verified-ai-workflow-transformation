"""Load and inspect the versioned VAIT AP benchmark."""

from pathlib import Path

from vait.benchmark.loader import load_benchmark
from vait.contracts.models import RiskLevel

dataset = load_benchmark(
    Path("datasets/ap_invoice_exceptions/v0.1/cases.yaml")
)

high_risk_cases = [
    case
    for case in dataset.cases
    if case.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}
]

print(f"Benchmark: {dataset.benchmark_id}")
print(f"Version: {dataset.version}")
print(f"Cases: {len(dataset.cases)}")
print(f"High-risk cases: {len(high_risk_cases)}")
