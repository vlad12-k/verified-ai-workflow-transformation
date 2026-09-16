"""Domain models for VAIT transformation contracts."""

from enum import StrEnum

from pydantic import BaseModel, Field, JsonValue, model_validator


class RiskLevel(StrEnum):
    """Business-risk classification for a transformation or evaluation case."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Effect(StrEnum):
    """Effects that an implementation may declare."""

    NONE = "none"
    READ = "read"
    NETWORK = "network"
    WRITE = "write"
    EXTERNAL_ACTION = "external_action"


class InvariantOperator(StrEnum):
    """Supported invariant comparison operators."""

    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    IN = "in"
    NOT_IN = "not_in"
    GREATER_THAN = "greater_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    LESS_THAN = "less_than"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    EXISTS = "exists"


class Invariant(BaseModel):
    """A declarative condition that a candidate output must satisfy."""

    id: str = Field(min_length=1)
    description: str = ""
    path: str = Field(min_length=1)
    operator: InvariantOperator
    expected: JsonValue | None = None
    critical: bool = True


class RiskConstraint(BaseModel):
    """Risk constraints attached to a transformation."""

    level: RiskLevel
    max_invariant_violations: int = Field(default=0, ge=0)
    max_output_mismatches: int = Field(default=0, ge=0)


class ExactVerificationScope(BaseModel):
    """Declared finite scope over which EXACT may be established."""

    exhaustive_case_ids: frozenset[str] = Field(min_length=1)


class BoundedVerificationPolicy(BaseModel):
    """Pre-declared thresholds for statistical bounded verification."""

    max_overall_disagreement_rate: float = Field(ge=0.0, le=1.0)
    max_high_risk_disagreement_rate: float = Field(ge=0.0, le=1.0)

    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)

    min_total_cases: int = Field(default=1, ge=1)
    min_high_risk_cases: int = Field(default=0, ge=0)


class TransformationContract(BaseModel):
    """Contract governing whether a candidate transformation is admissible."""

    id: str = Field(min_length=1)
    description: str = ""

    reference_implementation_id: str = Field(min_length=1)
    candidate_implementation_id: str = Field(min_length=1)

    risk: RiskConstraint

    exact_verification_scope: ExactVerificationScope | None = None
    bounded_verification: BoundedVerificationPolicy | None = None

    invariants: list[Invariant] = Field(default_factory=list)
    allowed_effects: set[Effect] = Field(default_factory=lambda: {Effect.NONE})

    @model_validator(mode="after")
    def validate_contract(self) -> "TransformationContract":
        """Validate cross-field contract rules."""
        if self.reference_implementation_id == self.candidate_implementation_id:
            raise ValueError(
                "Reference and candidate implementation IDs must be different."
            )

        invariant_ids = [invariant.id for invariant in self.invariants]
        if len(invariant_ids) != len(set(invariant_ids)):
            raise ValueError("Invariant IDs must be unique within a contract.")

        return self


class VerificationCase(BaseModel):
    """A single verification input with declared business risk."""

    id: str = Field(min_length=1)
    input_data: dict[str, JsonValue]
    risk_level: RiskLevel = RiskLevel.LOW
