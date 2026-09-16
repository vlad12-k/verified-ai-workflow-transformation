"""Behavioural comparison for bounded VAIT verification."""

from collections.abc import Sequence

from vait.contracts.models import RiskLevel, VerificationCase
from vait.decision.models import ExecutionObservation
from vait.verification.models import (
    RiskStratumEvidence,
    StatisticalEvidence,
)
from vait.verification.statistics import binomial_upper_bound

_HIGH_RISK_LEVELS = frozenset({RiskLevel.HIGH, RiskLevel.CRITICAL})


def build_statistical_evidence(
    cases: Sequence[VerificationCase],
    reference_observations: Sequence[ExecutionObservation],
    candidate_observations: Sequence[ExecutionObservation],
    confidence_level: float,
) -> StatisticalEvidence:
    """Compare reference and candidate outputs and build statistical evidence."""
    if not cases:
        raise ValueError("At least one verification case is required.")

    if len(reference_observations) != len(cases):
        raise ValueError("Reference observation count must match case count.")

    if len(candidate_observations) != len(cases):
        raise ValueError("Candidate observation count must match case count.")

    totals = {risk_level: 0 for risk_level in RiskLevel}
    disagreements = {risk_level: 0 for risk_level in RiskLevel}

    overall_disagreements = 0
    high_risk_cases = 0
    high_risk_disagreements = 0

    for case, reference, candidate in zip(
        cases,
        reference_observations,
        candidate_observations,
        strict=True,
    ):
        if reference.case_id != case.id or candidate.case_id != case.id:
            raise ValueError("Observation case IDs must match evaluation case IDs.")

        if reference.error is not None or candidate.error is not None:
            raise ValueError(
                "Statistical evidence requires successful execution observations."
            )

        is_disagreement = reference.output != candidate.output

        totals[case.risk_level] += 1

        if is_disagreement:
            disagreements[case.risk_level] += 1
            overall_disagreements += 1

        if case.risk_level in _HIGH_RISK_LEVELS:
            high_risk_cases += 1

            if is_disagreement:
                high_risk_disagreements += 1

    total_cases = len(cases)

    risk_strata: list[RiskStratumEvidence] = []

    for risk_level in RiskLevel:
        stratum_cases = totals[risk_level]

        if stratum_cases == 0:
            continue

        stratum_disagreements = disagreements[risk_level]

        risk_strata.append(
            RiskStratumEvidence(
                risk_level=risk_level,
                cases_evaluated=stratum_cases,
                disagreements=stratum_disagreements,
                disagreement_rate=stratum_disagreements / stratum_cases,
                disagreement_upper_bound=binomial_upper_bound(
                    events=stratum_disagreements,
                    trials=stratum_cases,
                    confidence_level=confidence_level,
                ),
            )
        )

    high_risk_upper_bound = (
        binomial_upper_bound(
            events=high_risk_disagreements,
            trials=high_risk_cases,
            confidence_level=confidence_level,
        )
        if high_risk_cases > 0
        else None
    )

    return StatisticalEvidence(
        cases_evaluated=total_cases,
        disagreements=overall_disagreements,
        disagreement_rate=overall_disagreements / total_cases,
        disagreement_upper_bound=binomial_upper_bound(
            events=overall_disagreements,
            trials=total_cases,
            confidence_level=confidence_level,
        ),
        confidence_level=confidence_level,
        high_risk_cases_evaluated=high_risk_cases,
        high_risk_disagreements=high_risk_disagreements,
        high_risk_disagreement_rate=(
            high_risk_disagreements / high_risk_cases
            if high_risk_cases > 0
            else 0.0
        ),
        high_risk_disagreement_upper_bound=high_risk_upper_bound,
        risk_strata=risk_strata,
    )
