"""Provider-neutral execution abstractions for VAIT."""

from typing import Protocol

from vait.contracts.models import Effect, VerificationCase
from vait.decision.models import ExecutionObservation


class ImplementationRunner(Protocol):
    """Execution interface implemented by all VAIT runners."""

    @property
    def implementation_id(self) -> str:
        """Return the implementation identifier."""
        ...

    @property
    def declared_effects(self) -> frozenset[Effect]:
        """Return effects declared by the implementation."""
        ...

    def execute(self, case: VerificationCase) -> ExecutionObservation:
        """Execute one verification case."""
        ...
