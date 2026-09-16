"""Evaluation of declarative VAIT invariants."""

from collections.abc import Mapping
from typing import cast

from pydantic import JsonValue

from vait.contracts.models import Invariant, InvariantOperator

_MISSING = object()


def resolve_path(value: JsonValue, path: str) -> object:
    """Resolve a dotted path from a JSON-compatible value."""
    current: object = value

    for component in path.split("."):
        if not isinstance(current, Mapping):
            return _MISSING

        mapping = cast(Mapping[str, object], current)

        if component not in mapping:
            return _MISSING

        current = mapping[component]

    return current


def evaluate_invariant(output: JsonValue, invariant: Invariant) -> bool:
    """Return whether an output satisfies an invariant."""
    actual = resolve_path(output, invariant.path)

    if invariant.operator is InvariantOperator.EXISTS:
        return actual is not _MISSING

    if actual is _MISSING:
        return False

    expected: object = invariant.expected

    if invariant.operator is InvariantOperator.EQUALS:
        return actual == expected

    if invariant.operator is InvariantOperator.NOT_EQUALS:
        return actual != expected

    if invariant.operator is InvariantOperator.IN:
        return isinstance(expected, list) and actual in expected

    if invariant.operator is InvariantOperator.NOT_IN:
        return isinstance(expected, list) and actual not in expected

    if invariant.operator is InvariantOperator.GREATER_THAN:
        return _ordered_compare(actual, expected, "gt")

    if invariant.operator is InvariantOperator.GREATER_THAN_OR_EQUAL:
        return _ordered_compare(actual, expected, "gte")

    if invariant.operator is InvariantOperator.LESS_THAN:
        return _ordered_compare(actual, expected, "lt")

    if invariant.operator is InvariantOperator.LESS_THAN_OR_EQUAL:
        return _ordered_compare(actual, expected, "lte")

    return False


def _ordered_compare(actual: object, expected: object, operator: str) -> bool:
    """Safely perform an ordered numeric comparison."""
    if (
        isinstance(actual, bool)
        or isinstance(expected, bool)
        or not isinstance(actual, (int, float))
        or not isinstance(expected, (int, float))
    ):
        return False

    if operator == "gt":
        return actual > expected

    if operator == "gte":
        return actual >= expected

    if operator == "lt":
        return actual < expected

    if operator == "lte":
        return actual <= expected

    return False
