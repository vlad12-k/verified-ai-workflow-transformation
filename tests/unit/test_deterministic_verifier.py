"""Tests for the deterministic VAIT verifier."""

import pytest
from pydantic import JsonValue

from vait.contracts.models import (
    Effect,
    ExactVerificationScope,
    Invariant,
    InvariantOperator,
    RiskConstraint,
    RiskLevel,
    TransformationContract,
    VerificationCase,
)
from vait.decision.models import Decision, FailureCode
from vait.runners.python_runner import PythonImplementationRunner
from vait.verification.deterministic import verify_deterministic


def reference_rule(data: dict[str, JsonValue]) -> JsonValue:
    """Reference invoice rule."""
    amount = data["amount"]

    if not isinstance(amount, (int, float)) or isinstance(amount, bool):
        raise ValueError("amount must be numeric")

    return {
        "decision": "HOLD" if amount > 1000 else "RECOMMEND_APPROVE",
    }


def equivalent_candidate(data: dict[str, JsonValue]) -> JsonValue:
    """Equivalent candidate implementation."""
    amount = data["amount"]

    if not isinstance(amount, (int, float)) or isinstance(amount, bool):
        raise ValueError("amount must be numeric")

    if amount <= 1000:
        return {"decision": "RECOMMEND_APPROVE"}

    return {"decision": "HOLD"}


def unsafe_candidate(data: dict[str, JsonValue]) -> JsonValue:
    """Incorrect candidate that approves every case."""
    return {"decision": "RECOMMEND_APPROVE"}


def failing_candidate(data: dict[str, JsonValue]) -> JsonValue:
    """Candidate that fails during execution."""
    raise RuntimeError("candidate failure")


def build_contract(
    exhaustive_case_ids: frozenset[str] | None = None,
) -> TransformationContract:
    """Create a deterministic test contract."""
    exact_scope = (
        ExactVerificationScope(
            exhaustive_case_ids=exhaustive_case_ids,
        )
        if exhaustive_case_ids is not None
        else None
    )

    return TransformationContract(
        id="invoice-decision-v1",
        reference_implementation_id="reference-rules",
        candidate_implementation_id="candidate-rules",
        risk=RiskConstraint(level=RiskLevel.HIGH),
        exact_verification_scope=exact_scope,
        allowed_effects={Effect.NONE},
        invariants=[
            Invariant(
                id="decision-exists",
                path="decision",
                operator=InvariantOperator.EXISTS,
            )
        ],
    )


def build_reference() -> PythonImplementationRunner:
    """Create the deterministic reference runner."""
    return PythonImplementationRunner(
        implementation_id="reference-rules",
        function=reference_rule,
    )


def test_equivalent_candidate_is_exact() -> None:
    """Equivalent behavior over the declared finite scope should be EXACT."""
    result = verify_deterministic(
        contract=build_contract(
            frozenset({"low", "high"}),
        ),
        reference=build_reference(),
        candidate=PythonImplementationRunner(
            implementation_id="candidate-rules",
            function=equivalent_candidate,
        ),
        cases=[
            VerificationCase(id="low", input_data={"amount": 500}),
            VerificationCase(id="high", input_data={"amount": 1500}),
        ],
    )

    assert result.decision is Decision.EXACT
    assert result.exact_matches == 2
    assert result.failures == []


def test_matching_sample_without_exact_scope_is_not_exact() -> None:
    """Matching observations alone must not justify an EXACT decision."""
    result = verify_deterministic(
        contract=build_contract(),
        reference=build_reference(),
        candidate=PythonImplementationRunner(
            implementation_id="candidate-rules",
            function=equivalent_candidate,
        ),
        cases=[
            VerificationCase(id="low", input_data={"amount": 500}),
        ],
    )

    assert result.decision is Decision.REJECT
    assert any(
        failure.code is FailureCode.INSUFFICIENT_EVIDENCE
        for failure in result.failures
    )


def test_mismatch_is_rejected() -> None:
    """A behavioral mismatch must prevent an EXACT decision."""
    result = verify_deterministic(
        contract=build_contract(),
        reference=build_reference(),
        candidate=PythonImplementationRunner(
            implementation_id="candidate-rules",
            function=unsafe_candidate,
        ),
        cases=[
            VerificationCase(id="high", input_data={"amount": 1500}),
        ],
    )

    assert result.decision is Decision.REJECT
    assert result.exact_matches == 0
    assert any(
        failure.code is FailureCode.OUTPUT_MISMATCH
        for failure in result.failures
    )


def test_forbidden_effect_is_rejected() -> None:
    """A candidate may not declare effects outside the contract."""
    result = verify_deterministic(
        contract=build_contract(),
        reference=build_reference(),
        candidate=PythonImplementationRunner(
            implementation_id="candidate-rules",
            function=equivalent_candidate,
            declared_effects=frozenset({Effect.NETWORK}),
        ),
        cases=[
            VerificationCase(id="low", input_data={"amount": 500}),
        ],
    )

    assert result.decision is Decision.REJECT
    assert any(
        failure.code is FailureCode.FORBIDDEN_EFFECT
        for failure in result.failures
    )


def test_candidate_execution_failure_is_rejected() -> None:
    """Execution failures must result in rejection."""
    result = verify_deterministic(
        contract=build_contract(),
        reference=build_reference(),
        candidate=PythonImplementationRunner(
            implementation_id="candidate-rules",
            function=failing_candidate,
        ),
        cases=[
            VerificationCase(id="failure", input_data={"amount": 500}),
        ],
    )

    assert result.decision is Decision.REJECT
    assert any(
        failure.code is FailureCode.EXECUTION_ERROR
        for failure in result.failures
    )


def test_empty_evaluation_set_is_rejected() -> None:
    """Verification requires evidence."""
    with pytest.raises(
        ValueError,
        match="At least one verification case",
    ):
        verify_deterministic(
            contract=build_contract(),
            reference=build_reference(),
            candidate=PythonImplementationRunner(
                implementation_id="candidate-rules",
                function=equivalent_candidate,
            ),
            cases=[],
        )


def test_wrong_candidate_id_is_rejected() -> None:
    """The executed candidate must match the transformation contract."""
    with pytest.raises(
        ValueError,
        match="Candidate implementation does not match",
    ):
        verify_deterministic(
            contract=build_contract(),
            reference=build_reference(),
            candidate=PythonImplementationRunner(
                implementation_id="wrong-candidate",
                function=equivalent_candidate,
            ),
            cases=[
                VerificationCase(id="low", input_data={"amount": 500}),
            ],
        )
