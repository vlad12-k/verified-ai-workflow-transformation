"""Verify a deterministic candidate against AP benchmark gold labels."""

from pathlib import Path

from pydantic import JsonValue

from vait.benchmark.loader import load_benchmark
from vait.benchmark.reporting import write_benchmark_report
from vait.benchmark.verification import verify_benchmark_bounded
from vait.contracts.models import BoundedVerificationPolicy
from vait.runners.python_runner import PythonImplementationRunner


def candidate_policy(data: dict[str, JsonValue]) -> JsonValue:
    """Apply the synthetic AP benchmark decision policy."""
    invoice_amount = data["invoice_amount"]
    purchase_order_amount = data["purchase_order_amount"]

    purchase_order_present = data["purchase_order_present"]
    supplier_known = data["supplier_known"]
    duplicate_invoice = data["duplicate_invoice"]
    bank_details_changed = data["bank_details_changed"]
    tax_id_valid = data["tax_id_valid"]

    if not isinstance(invoice_amount, (int, float)) or isinstance(
        invoice_amount,
        bool,
    ):
        raise ValueError("invoice_amount must be numeric")

    if bank_details_changed is True or duplicate_invoice is True:
        return {"decision": "HOLD"}

    if invoice_amount <= 0:
        return {"decision": "REVIEW"}

    if tax_id_valid is not True:
        return {"decision": "REVIEW"}

    if purchase_order_present is not True:
        return {"decision": "REVIEW"}

    if supplier_known is not True:
        return {"decision": "REVIEW"}

    if not isinstance(
        purchase_order_amount,
        (int, float),
    ) or isinstance(purchase_order_amount, bool):
        return {"decision": "REVIEW"}

    difference = abs(invoice_amount - purchase_order_amount)

    if difference >= 1000:
        return {"decision": "HOLD"}

    if difference > 0:
        return {"decision": "REVIEW"}

    return {"decision": "RECOMMEND_APPROVE"}


dataset = load_benchmark(
    Path("datasets/ap_invoice_exceptions/v0.1/cases.yaml")
)

report = verify_benchmark_bounded(
    dataset=dataset,
    candidate=PythonImplementationRunner(
        implementation_id="synthetic-ap-policy-v0.1",
        function=candidate_policy,
    ),
    policy=BoundedVerificationPolicy(
        max_overall_disagreement_rate=0.25,
        max_high_risk_disagreement_rate=0.40,
        confidence_level=0.95,
        min_total_cases=12,
        min_high_risk_cases=6,
    ),
)

report_path = write_benchmark_report(
    report,
    Path("artifacts/benchmarks/ap-v0.1-verification.json"),
)

print(f"Benchmark: {report.benchmark_id}")
print(f"Version: {report.benchmark_version}")
print(f"Candidate: {report.candidate_implementation_id}")
print(f"Cases evaluated: {report.cases_evaluated}")
print(f"Decision: {report.decision.value}")

if report.statistical_evidence is not None:
    evidence = report.statistical_evidence

    print(
        "Observed disagreement: "
        f"{evidence.disagreement_rate:.1%}"
    )
    print(
        "95% overall upper bound: "
        f"{evidence.disagreement_upper_bound:.1%}"
    )

    high_risk_upper = evidence.high_risk_disagreement_upper_bound

    if high_risk_upper is not None:
        print(
            "95% high-risk upper bound: "
            f"{high_risk_upper:.1%}"
        )

print(f"Failures: {len(report.failures)}")
print(f"Report: {report_path}")
