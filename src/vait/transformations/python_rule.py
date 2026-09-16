"""Backward-compatible deterministic Python rule transformation API."""

from dataclasses import dataclass

from vait.transformations.python_callable import (
    CandidatePreparation,
    PythonCallableTransformation,
)


@dataclass(frozen=True, slots=True)
class PythonRuleTransformation(PythonCallableTransformation):
    """A deterministic Python rule candidate transformation."""


__all__ = [
    "CandidatePreparation",
    "PythonRuleTransformation",
]
