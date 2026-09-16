"""Property tests for deterministic transformation verification."""

from hypothesis import given
from hypothesis import strategies as st
from pydantic import JsonValue

from vait.contracts.models import (
    Effect,
    ExactVerificationScope,
    RiskConstraint,
    RiskLevel,
    TransformationContract,
    VerificationCase,
)
from vait.decision.models import Decision
from vait.runners.python_runner import PythonImplementationRunner
from vait.verification.deterministic import verify_deterministic


def reference(data: dict[str, JsonValue]) -> JsonValue:
    """Reference threshold implementation."""
    value = data["value"]

    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("value must be an integer")

    return {"accepted": value >= 0}


def candidate(data: dict[str, JsonValue]) -> JsonValue:
    """Equivalent threshold implementation."""
    value = data["value"]

    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("value must be an integer")

    if value < 0:
        return {"accepted": False}

    return {"accepted": True}


@given(st.integers(min_value=-1_000_000, max_value=1_000_000))
def test_equivalent_implementations_remain_exact(value: int) -> None:
    """Equivalent implementations should agree across generated inputs."""
    contract = TransformationContract(
        id="property-equivalence",
        reference_implementation_id="reference",
        candidate_implementation_id="candidate",
        risk=RiskConstraint(level=RiskLevel.LOW),
        exact_verification_scope=ExactVerificationScope(
            exhaustive_case_ids=frozenset({"generated-case"}),
        ),
        allowed_effects={Effect.NONE},
    )

    result = verify_deterministic(
        contract=contract,
        reference=PythonImplementationRunner(
            implementation_id="reference",
            function=reference,
        ),
        candidate=PythonImplementationRunner(
            implementation_id="candidate",
            function=candidate,
        ),
        cases=[
            VerificationCase(
                id="generated-case",
                input_data={"value": value},
            )
        ],
    )

    assert result.decision is Decision.EXACT
