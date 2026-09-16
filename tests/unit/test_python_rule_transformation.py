"""Tests for supplied deterministic Python rule transformations."""

import pytest
from pydantic import JsonValue

from vait.contracts.models import Effect, RiskLevel, VerificationCase
from vait.transformations.applicability import (
    ApplicabilityContext,
    ApplicabilityIssueCode,
)
from vait.transformations.models import (
    Transformation,
    TransformationCategory,
    TransformationDescriptor,
)
from vait.transformations.python_rule import PythonRuleTransformation
from vait.transformations.registry import TransformationRegistry


def example_rule(data: dict[str, JsonValue]) -> JsonValue:
    """Return a deterministic decision for a numeric amount."""
    amount = data["amount"]

    if not isinstance(amount, (int, float)) or isinstance(amount, bool):
        raise ValueError("amount must be numeric")

    if amount >= 100:
        return {"decision": "review"}

    return {"decision": "approve"}


def build_transformation() -> PythonRuleTransformation:
    """Create a deterministic rule transformation fixture."""
    descriptor = TransformationDescriptor(
        transformation_id="amount-rule",
        version="1.0.0",
        name="Amount rule",
        description="Replace a decision step with a supplied amount rule.",
        category=TransformationCategory.OPTIMIZATION,
        declared_effects=frozenset({Effect.NONE}),
        supported_risk_levels=frozenset(
            {
                RiskLevel.LOW,
                RiskLevel.MEDIUM,
            }
        ),
        required_capabilities=frozenset(
            {
                "typed-inputs",
            }
        ),
    )

    return PythonRuleTransformation(
        descriptor=descriptor,
        implementation_id="amount-rule-python-v1",
        function=example_rule,
    )


def test_python_rule_transformation_satisfies_protocol() -> None:
    """Concrete rule transformations should satisfy the registry protocol."""
    transformation = build_transformation()

    assert isinstance(transformation, Transformation)


def test_applicable_candidate_is_prepared_for_execution() -> None:
    """Applicable supplied candidates should produce an executable runner."""
    transformation = build_transformation()

    preparation = transformation.prepare_candidate(
        ApplicabilityContext(
            risk_level=RiskLevel.LOW,
            available_capabilities=frozenset(
                {
                    "typed-inputs",
                }
            ),
        )
    )

    assert preparation.ready is True
    assert preparation.runner is not None

    observation = preparation.runner.execute(
        VerificationCase(
            id="case-1",
            input_data={"amount": 50},
            risk_level=RiskLevel.LOW,
        )
    )

    assert observation.error is None
    assert observation.output == {"decision": "approve"}
    assert observation.implementation_id == "amount-rule-python-v1"


def test_not_applicable_candidate_is_not_prepared() -> None:
    """Failed applicability must prevent candidate preparation."""
    transformation = build_transformation()

    preparation = transformation.prepare_candidate(
        ApplicabilityContext(
            risk_level=RiskLevel.CRITICAL,
            available_capabilities=frozenset(),
        )
    )

    assert preparation.ready is False
    assert preparation.runner is None
    assert preparation.applicability.is_applicable is False

    issue_codes = {
        issue.code
        for issue in preparation.applicability.issues
    }

    assert issue_codes == {
        ApplicabilityIssueCode.UNSUPPORTED_RISK_LEVEL,
        ApplicabilityIssueCode.MISSING_CAPABILITY,
    }


def test_transformation_can_be_registered_and_retrieved() -> None:
    """Concrete supplied transformations should integrate with the registry."""
    transformation = build_transformation()
    registry = TransformationRegistry()

    registry.register(transformation)

    assert registry.get(
        "amount-rule",
        "1.0.0",
    ) is transformation


def test_empty_implementation_id_is_rejected() -> None:
    """Concrete transformations require an implementation identity."""
    descriptor = build_transformation().descriptor

    with pytest.raises(
        ValueError,
        match="implementation_id must not be empty",
    ):
        PythonRuleTransformation(
            descriptor=descriptor,
            implementation_id="   ",
            function=example_rule,
        )
