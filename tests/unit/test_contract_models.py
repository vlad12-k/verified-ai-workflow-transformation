"""Tests for VAIT transformation contract models."""

import pytest
from pydantic import ValidationError

from vait.contracts.models import (
    Invariant,
    InvariantOperator,
    RiskConstraint,
    RiskLevel,
    TransformationContract,
)


def test_contract_requires_different_implementations() -> None:
    """Reference and candidate IDs must not be identical."""
    with pytest.raises(ValidationError):
        TransformationContract(
            id="contract-1",
            reference_implementation_id="rules-v1",
            candidate_implementation_id="rules-v1",
            risk=RiskConstraint(level=RiskLevel.LOW),
        )


def test_contract_rejects_duplicate_invariant_ids() -> None:
    """Invariant identifiers must be unique within a contract."""
    invariant = Invariant(
        id="blocked-vendor",
        path="decision",
        operator=InvariantOperator.NOT_EQUALS,
        expected="RECOMMEND_APPROVE",
    )

    with pytest.raises(ValidationError):
        TransformationContract(
            id="contract-1",
            reference_implementation_id="reference",
            candidate_implementation_id="candidate",
            risk=RiskConstraint(level=RiskLevel.HIGH),
            invariants=[invariant, invariant],
        )
