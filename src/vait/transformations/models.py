"""Typed models for VAIT workflow transformations."""

from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field, model_validator

from vait.contracts.models import Effect, RiskLevel


class TransformationCategory(StrEnum):
    """High-level category of a workflow transformation."""

    STRUCTURAL = "structural"
    CONTROL_FLOW = "control-flow"
    MODEL = "model"
    DATA = "data"
    OPTIMIZATION = "optimization"


class TransformationDescriptor(BaseModel):
    """Machine-readable declaration of one transformation."""

    transformation_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    )
    version: str = Field(
        min_length=1,
        pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$",
    )

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)

    category: TransformationCategory

    declared_effects: frozenset[Effect] = Field(
        default_factory=lambda: frozenset({Effect.NONE})
    )

    supported_risk_levels: frozenset[RiskLevel] = Field(
        default_factory=lambda: frozenset(RiskLevel),
        min_length=1,
    )

    required_capabilities: frozenset[str] = Field(
        default_factory=frozenset
    )

    @model_validator(mode="after")
    def validate_effects(self) -> "TransformationDescriptor":
        """Reject contradictory effect declarations."""
        if (
            Effect.NONE in self.declared_effects
            and len(self.declared_effects) > 1
        ):
            raise ValueError(
                "Effect.NONE cannot be combined with other effects."
            )

        return self

    @property
    def canonical_id(self) -> str:
        """Return a version-qualified transformation identifier."""
        return f"{self.transformation_id}@{self.version}"


@runtime_checkable
class Transformation(Protocol):
    """Minimum interface implemented by registered transformations."""

    @property
    def descriptor(self) -> TransformationDescriptor:
        """Return the transformation declaration."""
        ...
