"""Runner exposing benchmark gold labels as verification observations."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

from pydantic import JsonValue

from vait.contracts.models import Effect, VerificationCase
from vait.decision.models import ExecutionObservation


@dataclass(frozen=True, slots=True)
class GoldBenchmarkRunner:
    """Expose benchmark gold outcomes through the runner abstraction."""

    implementation_id: str
    gold_outcomes: Mapping[str, dict[str, JsonValue]]
    declared_effects: frozenset[Effect] = frozenset({Effect.NONE})

    def execute(self, case: VerificationCase) -> ExecutionObservation:
        """Return the declared gold outcome for one benchmark case."""
        if case.id not in self.gold_outcomes:
            return ExecutionObservation(
                implementation_id=self.implementation_id,
                case_id=case.id,
                latency_ms=0.0,
                declared_effects=set(self.declared_effects),
                error=f"Missing gold outcome for case '{case.id}'.",
            )

        return ExecutionObservation(
            implementation_id=self.implementation_id,
            case_id=case.id,
            output=deepcopy(self.gold_outcomes[case.id]),
            latency_ms=0.0,
            declared_effects=set(self.declared_effects),
        )
