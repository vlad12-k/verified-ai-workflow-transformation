"""Tests for benchmark-to-verification integration."""

import pytest
from pydantic import JsonValue

from vait.benchmark.models import BenchmarkCase, BenchmarkDataset
from vait.benchmark.verification import verify_benchmark_bounded
from vait.contracts.models import (
    BoundedVerificationPolicy,
    RiskLevel,
)
from vait.decision.models import Decision, FailureCode
from vait.economics.metrics import (
    CostEstimate,
    CostEvidenceKind,
)
from vait.runners.python_runner import PythonImplementationRunner


def candidate(data: dict[str, JsonValue]) -> JsonValue:
    """Return the benchmark decision, optionally introducing a mismatch."""
    decision = data["decision"]
    fail = data.get("fail", False)

    if not isinstance(decision, str):
        raise ValueError("decision must be a string")

    if fail is True:
        return {"decision": "WRONG"}

    return {"decision": decision}


def build_dataset(
    *,
    high_risk_failure: bool = False,
) -> BenchmarkDataset:
    """Create a compact gold-labelled benchmark."""
    return BenchmarkDataset(
        benchmark_id="integration-test",
        version="0.1",
        description="Benchmark verification integration fixture.",
        cases=[
            BenchmarkCase(
                case_id="low-1",
                input_data={"decision": "APPROVE"},
                gold_outcome={"decision": "APPROVE"},
                risk_level=RiskLevel.LOW,
                rationale="Low-risk fixture.",
            ),
            BenchmarkCase(
                case_id="low-2",
                input_data={"decision": "APPROVE"},
                gold_outcome={"decision": "APPROVE"},
                risk_level=RiskLevel.LOW,
                rationale="Low-risk fixture.",
            ),
            BenchmarkCase(
                case_id="high-1",
                input_data={
                    "decision": "HOLD",
                    "fail": high_risk_failure,
                },
                gold_outcome={"decision": "HOLD"},
                risk_level=RiskLevel.HIGH,
                rationale="High-risk fixture.",
            ),
            BenchmarkCase(
                case_id="critical-1",
                input_data={"decision": "HOLD"},
                gold_outcome={"decision": "HOLD"},
                risk_level=RiskLevel.CRITICAL,
                rationale="Critical-risk fixture.",
            ),
        ],
    )


def build_candidate() -> PythonImplementationRunner:
    """Create the benchmark candidate runner."""
    return PythonImplementationRunner(
        implementation_id="candidate",
        function=candidate,
    )


def test_gold_benchmark_can_produce_bounded_decision() -> None:
    """Gold-labelled benchmark evidence should feed bounded verification."""
    report = verify_benchmark_bounded(
        dataset=build_dataset(),
        candidate=build_candidate(),
        policy=BoundedVerificationPolicy(
            max_overall_disagreement_rate=0.60,
            max_high_risk_disagreement_rate=0.80,
            confidence_level=0.95,
            min_total_cases=4,
            min_high_risk_cases=2,
        ),
    )

    assert report.decision is Decision.BOUNDED
    assert report.statistical_evidence is not None
    assert report.statistical_evidence.disagreements == 0
    assert report.cases_evaluated == 4


def test_high_risk_gold_mismatch_can_reject_candidate() -> None:
    """A high-risk gold disagreement must be able to reject a candidate."""
    report = verify_benchmark_bounded(
        dataset=build_dataset(high_risk_failure=True),
        candidate=build_candidate(),
        policy=BoundedVerificationPolicy(
            max_overall_disagreement_rate=0.90,
            max_high_risk_disagreement_rate=0.70,
            confidence_level=0.95,
            min_total_cases=4,
            min_high_risk_cases=2,
        ),
    )

    assert report.decision is Decision.REJECT
    assert report.statistical_evidence is not None
    assert report.statistical_evidence.high_risk_disagreements == 1
    assert any(
        failure.code is FailureCode.RISK_THRESHOLD_EXCEEDED
        for failure in report.failures
    )


def test_bounded_benchmark_report_contains_provenance() -> None:
    """Bounded benchmark verification should preserve run provenance."""
    report = verify_benchmark_bounded(
        dataset=build_dataset(),
        candidate=build_candidate(),
        policy=BoundedVerificationPolicy(
            max_overall_disagreement_rate=0.60,
            max_high_risk_disagreement_rate=0.80,
            confidence_level=0.95,
            min_total_cases=4,
            min_high_risk_cases=2,
        ),
        candidate_configuration={
            "policy_version": "test-v1",
        },
    )

    assert report.provenance is not None
    assert report.provenance.candidate_configuration == {
        "policy_version": "test-v1",
    }

    assert report.reference_latency is None

    assert report.candidate_latency is not None
    assert report.candidate_latency.count == 4

    assert report.economic_evidence is None

def build_cost(amount: float) -> CostEstimate:
    """Create controlled development cost evidence."""
    return CostEstimate(
        amount_per_case=amount,
        currency="USD",
        evidence_kind=CostEvidenceKind.DECLARED,
        source="controlled-m2-development-scenario",
        pricing_version="scenario-v0.1",
    )


def test_benchmark_report_can_include_cost_evidence() -> None:
    """Cost comparison should be attached without changing verification."""
    report = verify_benchmark_bounded(
        dataset=build_dataset(),
        candidate=build_candidate(),
        policy=BoundedVerificationPolicy(
            max_overall_disagreement_rate=0.60,
            max_high_risk_disagreement_rate=0.80,
            confidence_level=0.95,
            min_total_cases=4,
            min_high_risk_cases=2,
        ),
        reference_cost=build_cost(0.10),
        candidate_cost=build_cost(0.02),
    )

    assert report.decision is Decision.BOUNDED
    assert report.economic_evidence is not None

    cost = report.economic_evidence.cost

    assert cost is not None
    assert cost.delta_per_case == pytest.approx(-0.08)
    assert cost.relative_delta == pytest.approx(-0.8)
    assert report.reference_latency is None
    assert report.candidate_latency is not None
    assert report.economic_evidence.latency is None


def test_cheaper_candidate_cannot_override_reject() -> None:
    """Economic evidence must not override a behavioural rejection."""
    report = verify_benchmark_bounded(
        dataset=build_dataset(
            high_risk_failure=True,
        ),
        candidate=build_candidate(),
        policy=BoundedVerificationPolicy(
            max_overall_disagreement_rate=0.90,
            max_high_risk_disagreement_rate=0.70,
            confidence_level=0.95,
            min_total_cases=4,
            min_high_risk_cases=2,
        ),
        reference_cost=build_cost(0.10),
        candidate_cost=build_cost(0.01),
    )

    assert report.decision is Decision.REJECT
    assert report.economic_evidence is not None

    cost = report.economic_evidence.cost

    assert cost is not None
    assert cost.delta_per_case == pytest.approx(-0.09)
    assert cost.relative_delta == pytest.approx(-0.9)

    assert any(
        failure.code is FailureCode.RISK_THRESHOLD_EXCEEDED
        for failure in report.failures
    )
