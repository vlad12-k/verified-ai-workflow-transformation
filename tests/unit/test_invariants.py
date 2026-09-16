"""Tests for declarative invariant evaluation."""

from vait.contracts.invariants import evaluate_invariant
from vait.contracts.models import Invariant, InvariantOperator


def test_nested_equality_invariant_passes() -> None:
    """Nested output paths should be resolvable."""
    invariant = Invariant(
        id="decision-is-hold",
        path="result.decision",
        operator=InvariantOperator.EQUALS,
        expected="HOLD",
    )

    output = {
        "result": {
            "decision": "HOLD",
        }
    }

    assert evaluate_invariant(output, invariant)


def test_missing_path_fails_comparison() -> None:
    """A missing path must fail a comparison invariant."""
    invariant = Invariant(
        id="decision-is-hold",
        path="decision",
        operator=InvariantOperator.EQUALS,
        expected="HOLD",
    )

    assert not evaluate_invariant({}, invariant)


def test_exists_operator_detects_present_value() -> None:
    """EXISTS should succeed when the declared path is present."""
    invariant = Invariant(
        id="decision-exists",
        path="decision",
        operator=InvariantOperator.EXISTS,
    )

    assert evaluate_invariant({"decision": "HOLD"}, invariant)


def test_numeric_comparison_passes() -> None:
    """Ordered numeric invariants should support int and float values."""
    invariant = Invariant(
        id="confidence-threshold",
        path="confidence",
        operator=InvariantOperator.GREATER_THAN_OR_EQUAL,
        expected=0.9,
    )

    assert evaluate_invariant({"confidence": 0.95}, invariant)


def test_numeric_comparison_rejects_boolean_values() -> None:
    """Booleans must not be treated as integers in numeric invariants."""
    invariant = Invariant(
        id="minimum-score",
        path="score",
        operator=InvariantOperator.GREATER_THAN,
        expected=0,
    )

    assert not evaluate_invariant({"score": True}, invariant)
