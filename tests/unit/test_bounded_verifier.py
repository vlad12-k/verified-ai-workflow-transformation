"""Tests for risk-aware bounded statistical verification."""

from pydantic import JsonValue

from vait.contracts.models import (
    BoundedVerificationPolicy,
    Effect,
    RiskConstraint,
    RiskLevel,
    TransformationContract,
    VerificationCase,
)
from vait.decision.models import Decision, FailureCode
from vait.runners.python_runner import PythonImplementationRunner
from vait.verification.bounded import verify_bounded


def reference(data: dict[str, JsonValue]) -> JsonValue:
    """Reference decision implementation."""
    decision = data["decision"]

    if not isinstance(decision, str):
        raise ValueError("decision must be a string")

    return {"decision": decision}


def candidate(data: dict[str, JsonValue]) -> JsonValue:
    """Candidate with an optional deliberate disagreement."""
    decision = data["decision"]
    flip = data.get("flip", False)

    if not isinstance(decision, str):
        raise ValueError("decision must be a string")

    if not isinstance(flip, bool):
        raise ValueError("flip must be a boolean")

    if flip:
        return {"decision": f"{decision}-different"}

    return {"decision": decision}


def build_contract(
    *,
    max_overall: float = 0.05,
    max_high_risk: float = 0.05,
    min_total: int = 200,
    min_high_risk: int = 100,
) -> TransformationContract:
    """Build a bounded statistical transformation contract."""
    return TransformationContract(
        id="bounded-test",
        reference_implementation_id="reference",
        candidate_implementation_id="candidate",
        risk=RiskConstraint(level=RiskLevel.HIGH),
        bounded_verification=BoundedVerificationPolicy(
            max_overall_disagreement_rate=max_overall,
            max_high_risk_disagreement_rate=max_high_risk,
            confidence_level=0.95,
            min_total_cases=min_total,
            min_high_risk_cases=min_high_risk,
        ),
        allowed_effects={Effect.NONE},
    )


def build_cases(
    *,
    high_risk_disagreements: int = 0,
) -> list[VerificationCase]:
    """Build 100 low-risk and 100 high-risk evaluation cases."""
    low_risk_cases = [
        VerificationCase(
            id=f"low-{index}",
            input_data={"decision": "APPROVE"},
            risk_level=RiskLevel.LOW,
        )
        for index in range(100)
    ]

    high_risk_cases = [
        VerificationCase(
            id=f"high-{index}",
            input_data={
                "decision": "HOLD",
                "flip": index < high_risk_disagreements,
            },
            risk_level=RiskLevel.HIGH,
        )
        for index in range(100)
    ]

    return low_risk_cases + high_risk_cases


def build_reference() -> PythonImplementationRunner:
    """Create the reference runner."""
    return PythonImplementationRunner(
        implementation_id="reference",
        function=reference,
    )


def build_candidate() -> PythonImplementationRunner:
    """Create the candidate runner."""
    return PythonImplementationRunner(
        implementation_id="candidate",
        function=candidate,
    )


def test_candidate_is_bounded_when_all_declared_bounds_pass() -> None:
    """Sufficient evidence within all bounds should produce BOUNDED."""
    result = verify_bounded(
        contract=build_contract(),
        reference=build_reference(),
        candidate=build_candidate(),
        cases=build_cases(),
    )

    assert result.decision is Decision.BOUNDED
    assert result.statistical_evidence is not None
    assert result.statistical_evidence.disagreements == 0
    assert result.failures == []


def test_high_risk_bound_failure_rejects_candidate() -> None:
    """High-risk evidence must override acceptable overall performance."""
    result = verify_bounded(
        contract=build_contract(
            max_overall=0.05,
            max_high_risk=0.03,
        ),
        reference=build_reference(),
        candidate=build_candidate(),
        cases=build_cases(
            high_risk_disagreements=1,
        ),
    )

    assert result.decision is Decision.REJECT
    assert result.statistical_evidence is not None
    assert any(
        failure.code is FailureCode.RISK_THRESHOLD_EXCEEDED
        for failure in result.failures
    )


def test_insufficient_sample_is_rejected() -> None:
    """A small clean sample must not be accepted as bounded evidence."""
    cases = [
        VerificationCase(
            id=f"low-{index}",
            input_data={"decision": "APPROVE"},
            risk_level=RiskLevel.LOW,
        )
        for index in range(20)
    ]

    result = verify_bounded(
        contract=build_contract(
            min_total=100,
            min_high_risk=0,
        ),
        reference=build_reference(),
        candidate=build_candidate(),
        cases=cases,
    )

    assert result.decision is Decision.REJECT
    assert any(
        failure.code is FailureCode.INSUFFICIENT_EVIDENCE
        for failure in result.failures
    )
