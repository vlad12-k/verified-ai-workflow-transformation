"""Integration between benchmark evidence and VAIT verification."""

from collections.abc import Iterable, Mapping

from pydantic import BaseModel, Field, JsonValue

from vait.benchmark.gold_runner import GoldBenchmarkRunner
from vait.benchmark.models import (
    BenchmarkDataset,
    ExperimentProvenance,
    LatencySummary,
)
from vait.benchmark.provenance import (
    build_experiment_provenance,
    summarize_latencies,
)
from vait.contracts.models import (
    BoundedVerificationPolicy,
    Effect,
    RiskConstraint,
    RiskLevel,
    TransformationContract,
    VerificationCase,
)
from vait.decision.models import Decision, VerificationFailure
from vait.runners.base import ImplementationRunner
from vait.verification.bounded import verify_bounded
from vait.verification.models import StatisticalEvidence

_RISK_ORDER = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}


class BenchmarkVerificationReport(BaseModel):
    """Benchmark metadata plus a bounded VAIT verification decision."""

    benchmark_id: str
    benchmark_version: str
    candidate_implementation_id: str

    cases_evaluated: int = Field(ge=0)
    decision: Decision

    statistical_evidence: StatisticalEvidence | None = None
    candidate_latency: LatencySummary | None = None
    provenance: ExperimentProvenance | None = None

    failures: list[VerificationFailure] = Field(default_factory=list)


def verify_benchmark_bounded(
    dataset: BenchmarkDataset,
    candidate: ImplementationRunner,
    policy: BoundedVerificationPolicy,
    allowed_effects: Iterable[Effect] = (Effect.NONE,),
    candidate_configuration: Mapping[str, JsonValue] | None = None,
) -> BenchmarkVerificationReport:
    """Verify a candidate against benchmark gold labels under a bounded policy."""
    gold_implementation_id = (
        f"benchmark-gold:{dataset.benchmark_id}:{dataset.version}"
    )

    gold_outcomes = {
        case.case_id: case.gold_outcome
        for case in dataset.cases
    }

    reference = GoldBenchmarkRunner(
        implementation_id=gold_implementation_id,
        gold_outcomes=gold_outcomes,
    )

    cases = [
        VerificationCase(
            id=case.case_id,
            input_data=case.input_data,
            risk_level=case.risk_level,
        )
        for case in dataset.cases
    ]

    overall_risk = max(
        (case.risk_level for case in dataset.cases),
        key=_RISK_ORDER.__getitem__,
    )

    contract = TransformationContract(
        id=(
            f"benchmark:{dataset.benchmark_id}:{dataset.version}:"
            f"{candidate.implementation_id}"
        ),
        reference_implementation_id=gold_implementation_id,
        candidate_implementation_id=candidate.implementation_id,
        risk=RiskConstraint(level=overall_risk),
        bounded_verification=policy,
        allowed_effects=set(allowed_effects),
    )

    result = verify_bounded(
        contract=contract,
        reference=reference,
        candidate=candidate,
        cases=cases,
    )

    return BenchmarkVerificationReport(
        benchmark_id=dataset.benchmark_id,
        benchmark_version=dataset.version,
        candidate_implementation_id=candidate.implementation_id,
        cases_evaluated=len(cases),
        decision=result.decision,
        statistical_evidence=result.statistical_evidence,
        candidate_latency=summarize_latencies(
            observation.latency_ms
            for observation in result.candidate_observations
        ),
        provenance=build_experiment_provenance(
            dataset=dataset,
            candidate=candidate,
            candidate_configuration=candidate_configuration,
        ),
        failures=result.failures,
    )
