"""Execution runner for deterministic Python implementations."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter_ns

from pydantic import JsonValue

from vait.contracts.models import Effect, VerificationCase
from vait.decision.models import ExecutionObservation

ImplementationFunction = Callable[[dict[str, JsonValue]], JsonValue]


@dataclass(frozen=True, slots=True)
class PythonImplementationRunner:
    """Execute a supplied Python implementation."""

    implementation_id: str
    function: ImplementationFunction
    declared_effects: frozenset[Effect] = frozenset({Effect.NONE})

    def execute(self, case: VerificationCase) -> ExecutionObservation:
        """Execute one case and capture basic evidence."""
        started = perf_counter_ns()

        try:
            output = self.function(deepcopy(case.input_data))
            error = None
        except Exception as exc:
            output = None
            error = f"{type(exc).__name__}: {exc}"

        latency_ms = (perf_counter_ns() - started) / 1_000_000

        return ExecutionObservation(
            implementation_id=self.implementation_id,
            case_id=case.id,
            output=output,
            latency_ms=latency_ms,
            declared_effects=set(self.declared_effects),
            error=error,
        )
