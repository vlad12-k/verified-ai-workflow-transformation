"""Readiness contracts for VAIT platform dependencies."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ReadinessCheck:
    """Result of one required platform dependency check."""

    name: str
    ready: bool


class ReadinessProbe(Protocol):
    """Callable contract used to evaluate required platform dependencies."""

    def __call__(self) -> Sequence[ReadinessCheck]:
        """Return current readiness results."""


def ready_without_dependencies() -> tuple[ReadinessCheck, ...]:
    """Report readiness when no external dependencies are required yet."""
    return ()
