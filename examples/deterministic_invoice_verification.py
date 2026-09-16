"""Run a deterministic VAIT invoice-decision verification example."""

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
from vait.runners.python_runner import PythonImplementationRunner
from vait.verification.deterministic import verify_deterministic


def reference(data: dict[str, JsonValue]) -> JsonValue:
    """Reference invoice approval policy."""
    amount = data["amount"]

    if not isinstance(amount, (int, float)) or isinstance(amount, bool):
        raise ValueError("amount must be numeric")

    if amount > 1000:
        return {"decision": "HOLD"}

    return {"decision": "RECOMMEND_APPROVE"}


def candidate(data: dict[str, JsonValue]) -> JsonValue:
    """Equivalent deterministic candidate."""
    amount = data["amount"]

    if not isinstance(amount, (int, float)) or isinstance(amount, bool):
        raise ValueError("amount must be numeric")

    decision = "HOLD" if amount > 1000 else "RECOMMEND_APPROVE"

    return {"decision": decision}


case_ids = frozenset(
    {
        "invoice-001",
        "invoice-002",
        "invoice-003",
        "invoice-004",
    }
)

contract = TransformationContract(
    id="invoice-decision-example",
    reference_implementation_id="reference-policy",
    candidate_implementation_id="candidate-policy",
    risk=RiskConstraint(level=RiskLevel.HIGH),
    exact_verification_scope=ExactVerificationScope(
        exhaustive_case_ids=case_ids,
    ),
    allowed_effects={Effect.NONE},
    invariants=[
        Invariant(
            id="decision-present",
            path="decision",
            operator=InvariantOperator.EXISTS,
        )
    ],
)

cases = [
    VerificationCase(id="invoice-001", input_data={"amount": 200}),
    VerificationCase(id="invoice-002", input_data={"amount": 1000}),
    VerificationCase(id="invoice-003", input_data={"amount": 1001}),
    VerificationCase(id="invoice-004", input_data={"amount": 5000}),
]

result = verify_deterministic(
    contract=contract,
    reference=PythonImplementationRunner(
        implementation_id="reference-policy",
        function=reference,
    ),
    candidate=PythonImplementationRunner(
        implementation_id="candidate-policy",
        function=candidate,
    ),
    cases=cases,
)

print(f"Contract: {result.contract_id}")
print(f"Cases evaluated: {result.cases_evaluated}")
print(f"Exact matches: {result.exact_matches}")
print(f"Decision: {result.decision.value}")
