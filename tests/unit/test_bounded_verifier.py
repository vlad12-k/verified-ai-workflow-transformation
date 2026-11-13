"""Tests for risk-aware bounded statistical verification."""

import pytest
from pydantic import JsonValue

from vait.contracts.models import (
    BoundedVerificationPolicy,
    Effect,
    Invariant,
    InvariantOperator,
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


def build_single_case(
    *,
    flip: bool = False,
) -> list[VerificationCase]:
    """Build one high-risk case for fail-closed verifier tests."""
    return [
        VerificationCase(
            id="single-high-risk",
            input_data={
                "decision": "HOLD",
                "flip": flip,
            },
            risk_level=RiskLevel.HIGH,
        )
    ]


def test_empty_case_set_is_rejected() -> None:
    """Bounded verification requires explicit evaluation evidence."""
    with pytest.raises(
        ValueError,
        match="At least one verification case is required",
    ):
        verify_bounded(
            contract=build_contract(),
            reference=build_reference(),
            candidate=build_candidate(),
            cases=[],
        )


def test_missing_bounded_policy_is_rejected() -> None:
    """A bounded decision cannot be made without a declared policy."""
    contract = build_contract().model_copy(
        update={
            "bounded_verification": None,
        }
    )

    with pytest.raises(
        ValueError,
        match="requires a bounded verification policy",
    ):
        verify_bounded(
            contract=contract,
            reference=build_reference(),
            candidate=build_candidate(),
            cases=build_single_case(),
        )


def test_reference_identity_mismatch_is_rejected() -> None:
    """Evidence from the wrong reference implementation is inadmissible."""
    wrong_reference = PythonImplementationRunner(
        implementation_id="wrong-reference",
        function=reference,
    )

    with pytest.raises(
        ValueError,
        match="Reference implementation does not match",
    ):
        verify_bounded(
            contract=build_contract(),
            reference=wrong_reference,
            candidate=build_candidate(),
            cases=build_single_case(),
        )


def test_candidate_identity_mismatch_is_rejected() -> None:
    """Evidence from the wrong candidate implementation is inadmissible."""
    wrong_candidate = PythonImplementationRunner(
        implementation_id="wrong-candidate",
        function=candidate,
    )

    with pytest.raises(
        ValueError,
        match="Candidate implementation does not match",
    ):
        verify_bounded(
            contract=build_contract(),
            reference=build_reference(),
            candidate=wrong_candidate,
            cases=build_single_case(),
        )


def test_forbidden_candidate_effect_rejects_before_statistical_acceptance() -> None:
    """Forbidden side effects must remain terminal verification failures."""
    network_candidate = PythonImplementationRunner(
        implementation_id="candidate",
        function=candidate,
        declared_effects=frozenset(
            {
                Effect.NETWORK,
            }
        ),
    )

    result = verify_bounded(
        contract=build_contract(),
        reference=build_reference(),
        candidate=network_candidate,
        cases=build_single_case(),
    )

    assert result.decision is Decision.REJECT
    assert result.statistical_evidence is None

    assert any(
        failure.code is FailureCode.FORBIDDEN_EFFECT
        for failure in result.failures
    )


def test_reference_execution_failure_is_terminal() -> None:
    """A failed reference execution cannot produce bounded evidence."""

    def failing_reference(
        data: dict[str, JsonValue],
    ) -> JsonValue:
        del data
        raise RuntimeError("reference failed")

    reference_runner = PythonImplementationRunner(
        implementation_id="reference",
        function=failing_reference,
    )

    result = verify_bounded(
        contract=build_contract(),
        reference=reference_runner,
        candidate=build_candidate(),
        cases=build_single_case(),
    )

    assert result.decision is Decision.REJECT
    assert result.statistical_evidence is None

    assert any(
        failure.code is FailureCode.EXECUTION_ERROR
        for failure in result.failures
    )


def test_candidate_execution_failure_is_terminal() -> None:
    """A failed candidate execution cannot produce bounded evidence."""

    def failing_candidate(
        data: dict[str, JsonValue],
    ) -> JsonValue:
        del data
        raise RuntimeError("candidate failed")

    candidate_runner = PythonImplementationRunner(
        implementation_id="candidate",
        function=failing_candidate,
    )

    result = verify_bounded(
        contract=build_contract(),
        reference=build_reference(),
        candidate=candidate_runner,
        cases=build_single_case(),
    )

    assert result.decision is Decision.REJECT
    assert result.statistical_evidence is None

    assert any(
        failure.code is FailureCode.EXECUTION_ERROR
        for failure in result.failures
    )


def test_invariant_violation_is_terminal_before_statistical_acceptance() -> None:
    """Candidate invariant failure must override otherwise matching output."""
    contract = build_contract().model_copy(
        update={
            "invariants": [
                Invariant(
                    id="decision-remains-hold",
                    path="decision",
                    operator=InvariantOperator.EQUALS,
                    expected="HOLD",
                ),
                Invariant(
                    id="decision-must-not-be-hold",
                    path="decision",
                    operator=InvariantOperator.EQUALS,
                    expected="APPROVE",
                ),
            ],
        }
    )

    result = verify_bounded(
        contract=contract,
        reference=build_reference(),
        candidate=build_candidate(),
        cases=build_single_case(),
    )

    assert result.decision is Decision.REJECT
    assert result.statistical_evidence is None

    violations = [
        failure
        for failure in result.failures
        if failure.code is FailureCode.INVARIANT_VIOLATION
    ]

    assert len(violations) == 1
    assert violations[0].case_id == "single-high-risk"


def test_insufficient_high_risk_sample_is_rejected() -> None:
    """High-risk sample requirements must be enforced independently."""
    result = verify_bounded(
        contract=build_contract(
            min_total=200,
            min_high_risk=101,
        ),
        reference=build_reference(),
        candidate=build_candidate(),
        cases=build_cases(),
    )

    assert result.decision is Decision.REJECT
    assert result.statistical_evidence is not None

    assert any(
        failure.code is FailureCode.INSUFFICIENT_EVIDENCE
        for failure in result.failures
    )


def test_overall_statistical_threshold_failure_rejects_candidate() -> None:
    """Overall disagreement risk must independently force REJECT."""
    result = verify_bounded(
        contract=build_contract(
            max_overall=0.0,
            max_high_risk=1.0,
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
        failure.code
        is FailureCode.STATISTICAL_THRESHOLD_EXCEEDED
        for failure in result.failures
    )

    assert not any(
        failure.code is FailureCode.RISK_THRESHOLD_EXCEEDED
        for failure in result.failures
    )
