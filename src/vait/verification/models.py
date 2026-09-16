"""Evidence models produced by statistical VAIT verification."""

from pydantic import BaseModel, Field

from vait.contracts.models import RiskLevel
from vait.decision.models import (
    Decision,
    ExecutionObservation,
    VerificationFailure,
)


class RiskStratumEvidence(BaseModel):
    """Behavioural evidence for one business-risk stratum."""

    risk_level: RiskLevel
    cases_evaluated: int = Field(ge=0)
    disagreements: int = Field(ge=0)
    disagreement_rate: float = Field(ge=0.0, le=1.0)
    disagreement_upper_bound: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


class StatisticalEvidence(BaseModel):
    """Statistical evidence for a candidate transformation."""

    cases_evaluated: int = Field(ge=0)
    disagreements: int = Field(ge=0)

    disagreement_rate: float = Field(ge=0.0, le=1.0)
    disagreement_upper_bound: float = Field(ge=0.0, le=1.0)

    confidence_level: float = Field(gt=0.5, lt=1.0)

    high_risk_cases_evaluated: int = Field(ge=0)
    high_risk_disagreements: int = Field(ge=0)
    high_risk_disagreement_rate: float = Field(ge=0.0, le=1.0)
    high_risk_disagreement_upper_bound: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    risk_strata: list[RiskStratumEvidence] = Field(default_factory=list)


class BoundedVerificationResult(BaseModel):
    """Decision and evidence from bounded statistical verification."""

    contract_id: str
    decision: Decision
    statistical_evidence: StatisticalEvidence | None = None

    reference_observations: list[ExecutionObservation] = Field(default_factory=list)
    candidate_observations: list[ExecutionObservation] = Field(default_factory=list)
    failures: list[VerificationFailure] = Field(default_factory=list)
