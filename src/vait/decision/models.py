"""Decision and evidence models produced by VAIT verification."""

from enum import StrEnum

from pydantic import BaseModel, Field, JsonValue

from vait.contracts.models import Effect


class Decision(StrEnum):
    """VAIT transformation decision."""

    EXACT = "EXACT"
    BOUNDED = "BOUNDED"
    REJECT = "REJECT"


class FailureCode(StrEnum):
    """Machine-readable reasons why verification failed."""

    OUTPUT_MISMATCH = "output_mismatch"
    INVARIANT_VIOLATION = "invariant_violation"
    FORBIDDEN_EFFECT = "forbidden_effect"
    EXECUTION_ERROR = "execution_error"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    STATISTICAL_THRESHOLD_EXCEEDED = "statistical_threshold_exceeded"
    RISK_THRESHOLD_EXCEEDED = "risk_threshold_exceeded"


class ExecutionObservation(BaseModel):
    """Observed execution evidence for one implementation and case."""

    implementation_id: str
    case_id: str
    output: JsonValue | None = None
    latency_ms: float = Field(ge=0.0)
    declared_effects: set[Effect] = Field(default_factory=set)
    error: str | None = None


class VerificationFailure(BaseModel):
    """A concrete verification failure."""

    code: FailureCode
    case_id: str | None = None
    message: str


class VerificationResult(BaseModel):
    """Evidence and decision produced by a verifier."""

    contract_id: str
    decision: Decision
    cases_evaluated: int = Field(ge=0)
    exact_matches: int = Field(ge=0)

    reference_observations: list[ExecutionObservation] = Field(default_factory=list)
    candidate_observations: list[ExecutionObservation] = Field(default_factory=list)
    failures: list[VerificationFailure] = Field(default_factory=list)
