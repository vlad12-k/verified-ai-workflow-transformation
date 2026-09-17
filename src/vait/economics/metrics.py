"""Provider-neutral cost and latency comparison evidence."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, JsonValue, model_validator

from vait.benchmark.models import LatencySummary


class CostEvidenceKind(StrEnum):
    """How a per-case monetary cost value was obtained."""

    DECLARED = "declared"
    ESTIMATED = "estimated"
    MEASURED = "measured"


class CostEstimate(BaseModel):
    """One provider-neutral monetary execution-cost estimate."""

    amount_per_case: float = Field(
        ge=0.0,
        allow_inf_nan=False,
    )
    currency: str = Field(
        min_length=3,
        max_length=3,
        pattern=r"^[A-Z]{3}$",
    )
    evidence_kind: CostEvidenceKind
    source: str = Field(min_length=1)

    pricing_version: str | None = None
    observed_at: datetime | None = None

    metadata: dict[str, JsonValue] = Field(
        default_factory=dict
    )


class CostComparison(BaseModel):
    """Reference-versus-candidate per-case cost evidence."""

    reference: CostEstimate
    candidate: CostEstimate

    delta_per_case: float = Field(
        allow_inf_nan=False
    )
    relative_delta: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )


class LatencyComparison(BaseModel):
    """Reference-versus-candidate latency evidence."""

    reference: LatencySummary
    candidate: LatencySummary

    mean_delta_ms: float = Field(
        allow_inf_nan=False
    )
    p50_delta_ms: float = Field(
        allow_inf_nan=False
    )
    p95_delta_ms: float = Field(
        allow_inf_nan=False
    )
    max_delta_ms: float = Field(
        allow_inf_nan=False
    )


class EconomicEvidence(BaseModel):
    """Optional cost and performance evidence for a transformation."""

    cost: CostComparison | None = None
    latency: LatencyComparison | None = None
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_evidence(self) -> "EconomicEvidence":
        """Require at least one economic or performance comparison."""
        if self.cost is None and self.latency is None:
            raise ValueError(
                "Economic evidence requires cost or latency evidence."
            )

        return self


def compare_cost_estimates(
    reference: CostEstimate,
    candidate: CostEstimate,
) -> CostComparison:
    """Compare candidate cost with reference cost.

    A negative delta means that the candidate is cheaper.
    """
    if reference.currency != candidate.currency:
        raise ValueError(
            "Reference and candidate cost estimates must "
            "use the same currency."
        )

    delta = (
        candidate.amount_per_case
        - reference.amount_per_case
    )

    relative_delta = (
        delta / reference.amount_per_case
        if reference.amount_per_case > 0.0
        else None
    )

    return CostComparison(
        reference=reference,
        candidate=candidate,
        delta_per_case=delta,
        relative_delta=relative_delta,
    )


def compare_latency_summaries(
    reference: LatencySummary,
    candidate: LatencySummary,
) -> LatencyComparison:
    """Compare candidate latency with reference latency.

    Negative deltas mean that the candidate was faster for the
    corresponding statistic.
    """
    return LatencyComparison(
        reference=reference,
        candidate=candidate,
        mean_delta_ms=(
            candidate.mean_ms - reference.mean_ms
        ),
        p50_delta_ms=(
            candidate.p50_ms - reference.p50_ms
        ),
        p95_delta_ms=(
            candidate.p95_ms - reference.p95_ms
        ),
        max_delta_ms=(
            candidate.max_ms - reference.max_ms
        ),
    )
