"""Tests for declarative invariant evaluation."""

from typing import cast

import pytest
from pydantic import JsonValue

from vait.contracts.invariants import (
    _ordered_compare,
    evaluate_invariant,
)
from vait.contracts.models import Invariant, InvariantOperator


def test_nested_equality_invariant_passes() -> None:
    """Nested output paths should be resolvable."""
    invariant = Invariant(
        id="decision-is-hold",
        path="result.decision",
        operator=InvariantOperator.EQUALS,
        expected="HOLD",
    )

    output: JsonValue = {
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


def test_nested_path_fails_when_intermediate_value_is_not_mapping() -> None:
    """Nested traversal must fail safely when an intermediate value is scalar."""
    invariant = Invariant(
        id="nested-decision",
        path="result.decision",
        operator=InvariantOperator.EQUALS,
        expected="HOLD",
    )

    output: JsonValue = {
        "result": "not-a-mapping",
    }

    assert not evaluate_invariant(
        output,
        invariant,
    )


def test_exists_operator_rejects_missing_value() -> None:
    """EXISTS must distinguish a missing path from a present null value."""
    invariant = Invariant(
        id="decision-exists",
        path="decision",
        operator=InvariantOperator.EXISTS,
    )

    assert not evaluate_invariant(
        {},
        invariant,
    )

    assert evaluate_invariant(
        {"decision": None},
        invariant,
    )


@pytest.mark.parametrize(
    (
        "operator",
        "actual",
        "expected",
        "expected_result",
    ),
    [
        (
            InvariantOperator.EQUALS,
            "HOLD",
            "HOLD",
            True,
        ),
        (
            InvariantOperator.NOT_EQUALS,
            "HOLD",
            "APPROVE",
            True,
        ),
        (
            InvariantOperator.IN,
            "HOLD",
            ["HOLD", "REVIEW"],
            True,
        ),
        (
            InvariantOperator.IN,
            "HOLD",
            "HOLD",
            False,
        ),
        (
            InvariantOperator.NOT_IN,
            "HOLD",
            ["APPROVE", "REVIEW"],
            True,
        ),
        (
            InvariantOperator.NOT_IN,
            "HOLD",
            "HOLD",
            False,
        ),
        (
            InvariantOperator.GREATER_THAN,
            3,
            2,
            True,
        ),
        (
            InvariantOperator.GREATER_THAN_OR_EQUAL,
            2,
            2,
            True,
        ),
        (
            InvariantOperator.LESS_THAN,
            1,
            2,
            True,
        ),
        (
            InvariantOperator.LESS_THAN_OR_EQUAL,
            2,
            2,
            True,
        ),
    ],
)
def test_all_supported_invariant_operators(
    operator: InvariantOperator,
    actual: JsonValue,
    expected: JsonValue,
    expected_result: bool,
) -> None:
    """Every declared invariant operator must execute its intended semantics."""
    invariant = Invariant(
        id=f"operator-{operator.value}",
        path="value",
        operator=operator,
        expected=expected,
    )

    assert (
        evaluate_invariant(
            {"value": actual},
            invariant,
        )
        is expected_result
    )


@pytest.mark.parametrize(
    (
        "actual",
        "expected",
    ),
    [
        (
            True,
            0,
        ),
        (
            1,
            False,
        ),
        (
            "1",
            0,
        ),
        (
            1,
            "0",
        ),
    ],
)
def test_ordered_comparison_rejects_non_numeric_or_boolean_values(
    actual: JsonValue,
    expected: JsonValue,
) -> None:
    """Ordered comparisons must reject booleans and non-numeric operands."""
    invariant = Invariant(
        id="ordered-type-safety",
        path="value",
        operator=InvariantOperator.GREATER_THAN,
        expected=expected,
    )

    assert not evaluate_invariant(
        {"value": actual},
        invariant,
    )


@pytest.mark.parametrize(
    (
        "operator",
        "actual",
        "expected",
    ),
    [
        (
            InvariantOperator.GREATER_THAN,
            1,
            2,
        ),
        (
            InvariantOperator.GREATER_THAN_OR_EQUAL,
            1,
            2,
        ),
        (
            InvariantOperator.LESS_THAN,
            2,
            1,
        ),
        (
            InvariantOperator.LESS_THAN_OR_EQUAL,
            2,
            1,
        ),
    ],
)
def test_ordered_comparison_false_results_are_preserved(
    operator: InvariantOperator,
    actual: JsonValue,
    expected: JsonValue,
) -> None:
    """Valid numeric comparisons must also preserve negative outcomes."""
    invariant = Invariant(
        id=f"ordered-false-{operator.value}",
        path="value",
        operator=operator,
        expected=expected,
    )

    assert not evaluate_invariant(
        {"value": actual},
        invariant,
    )


def test_unknown_invariant_operator_fails_closed() -> None:
    """A bypassed model validator must not make an unknown operator succeed."""
    invariant = Invariant.model_construct(
        id="unsupported-operator",
        path="value",
        operator=cast(
            InvariantOperator,
            "unsupported",
        ),
        expected=1,
    )

    assert not evaluate_invariant(
        {"value": 1},
        invariant,
    )


def test_ordered_compare_unknown_selector_fails_closed() -> None:
    """The numeric helper must reject unknown comparison selectors."""
    assert not _ordered_compare(
        2,
        1,
        "unsupported",
    )
