"""Risk-aware bounded statistical transformation verification."""

from collections.abc import Sequence

from vait.contracts.invariants import evaluate_invariant
from vait.contracts.models import TransformationContract, VerificationCase
from vait.decision.engine import decide_bounded_or_reject
from vait.decision.models import (
    ExecutionObservation,
    FailureCode,
    VerificationFailure,
)
from vait.runners.base import ImplementationRunner
from vait.verification.behaviour import build_statistical_evidence
from vait.verification.models import BoundedVerificationResult


def verify_bounded(
    contract: TransformationContract,
    reference: ImplementationRunner,
    candidate: ImplementationRunner,
    cases: Sequence[VerificationCase],
) -> BoundedVerificationResult:
    """Evaluate a candidate under declared bounded statistical constraints."""
    if not cases:
        raise ValueError("At least one verification case is required.")

    policy = contract.bounded_verification

    if policy is None:
        raise ValueError(
            "Bounded verification requires a bounded verification policy."
        )

    if reference.implementation_id != contract.reference_implementation_id:
        raise ValueError("Reference implementation does not match the contract.")

    if candidate.implementation_id != contract.candidate_implementation_id:
        raise ValueError("Candidate implementation does not match the contract.")

    failures: list[VerificationFailure] = []
    reference_observations: list[ExecutionObservation] = []
    candidate_observations: list[ExecutionObservation] = []

    forbidden_effects = candidate.declared_effects.difference(
        contract.allowed_effects
    )

    if forbidden_effects:
        effects = ", ".join(
            sorted(effect.value for effect in forbidden_effects)
        )
        failures.append(
            VerificationFailure(
                code=FailureCode.FORBIDDEN_EFFECT,
                message=f"Candidate declares forbidden effects: {effects}.",
            )
        )

    for case in cases:
        reference_observation = reference.execute(case)
        candidate_observation = candidate.execute(case)

        reference_observations.append(reference_observation)
        candidate_observations.append(candidate_observation)

        if reference_observation.error is not None:
            failures.append(
                VerificationFailure(
                    code=FailureCode.EXECUTION_ERROR,
                    case_id=case.id,
                    message=(
                        "Reference implementation failed: "
                        f"{reference_observation.error}"
                    ),
                )
            )
            continue

        if candidate_observation.error is not None:
            failures.append(
                VerificationFailure(
                    code=FailureCode.EXECUTION_ERROR,
                    case_id=case.id,
                    message=(
                        "Candidate implementation failed: "
                        f"{candidate_observation.error}"
                    ),
                )
            )
            continue

        for invariant in contract.invariants:
            if not evaluate_invariant(
                candidate_observation.output,
                invariant,
            ):
                failures.append(
                    VerificationFailure(
                        code=FailureCode.INVARIANT_VIOLATION,
                        case_id=case.id,
                        message=f"Invariant '{invariant.id}' was violated.",
                    )
                )

    if failures:
        return BoundedVerificationResult(
            contract_id=contract.id,
            decision=decide_bounded_or_reject(failures),
            reference_observations=reference_observations,
            candidate_observations=candidate_observations,
            failures=failures,
        )

    evidence = build_statistical_evidence(
        cases=cases,
        reference_observations=reference_observations,
        candidate_observations=candidate_observations,
        confidence_level=policy.confidence_level,
    )

    if evidence.cases_evaluated < policy.min_total_cases:
        failures.append(
            VerificationFailure(
                code=FailureCode.INSUFFICIENT_EVIDENCE,
                message=(
                    "Total evaluation sample is smaller than the declared "
                    f"minimum of {policy.min_total_cases} cases."
                ),
            )
        )

    if evidence.high_risk_cases_evaluated < policy.min_high_risk_cases:
        failures.append(
            VerificationFailure(
                code=FailureCode.INSUFFICIENT_EVIDENCE,
                message=(
                    "High-risk evaluation sample is smaller than the declared "
                    f"minimum of {policy.min_high_risk_cases} cases."
                ),
            )
        )

    if (
        evidence.disagreement_upper_bound
        > policy.max_overall_disagreement_rate
    ):
        failures.append(
            VerificationFailure(
                code=FailureCode.STATISTICAL_THRESHOLD_EXCEEDED,
                message=(
                    "Overall disagreement upper bound exceeds the declared "
                    "maximum."
                ),
            )
        )

    high_risk_upper_bound = evidence.high_risk_disagreement_upper_bound

    if (
        high_risk_upper_bound is not None
        and high_risk_upper_bound
        > policy.max_high_risk_disagreement_rate
    ):
        failures.append(
            VerificationFailure(
                code=FailureCode.RISK_THRESHOLD_EXCEEDED,
                message=(
                    "High-risk disagreement upper bound exceeds the declared "
                    "maximum."
                ),
            )
        )

    return BoundedVerificationResult(
        contract_id=contract.id,
        decision=decide_bounded_or_reject(failures),
        statistical_evidence=evidence,
        reference_observations=reference_observations,
        candidate_observations=candidate_observations,
        failures=failures,
    )
