"""Deterministic transformation verification."""

from collections.abc import Sequence

from vait.contracts.invariants import evaluate_invariant
from vait.contracts.models import TransformationContract, VerificationCase
from vait.decision.engine import decide_exact_or_reject
from vait.decision.models import (
    ExecutionObservation,
    FailureCode,
    VerificationFailure,
    VerificationResult,
)
from vait.runners.base import ImplementationRunner


def verify_deterministic(
    contract: TransformationContract,
    reference: ImplementationRunner,
    candidate: ImplementationRunner,
    cases: Sequence[VerificationCase],
) -> VerificationResult:
    """Verify deterministic equivalence under a declared contract."""
    if not cases:
        raise ValueError("At least one verification case is required.")

    if reference.implementation_id != contract.reference_implementation_id:
        raise ValueError("Reference implementation does not match the contract.")

    if candidate.implementation_id != contract.candidate_implementation_id:
        raise ValueError("Candidate implementation does not match the contract.")

    failures: list[VerificationFailure] = []
    reference_observations: list[ExecutionObservation] = []
    candidate_observations: list[ExecutionObservation] = []
    exact_matches = 0

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

        if reference_observation.output == candidate_observation.output:
            exact_matches += 1
        else:
            failures.append(
                VerificationFailure(
                    code=FailureCode.OUTPUT_MISMATCH,
                    case_id=case.id,
                    message="Candidate output differs from reference output.",
                )
            )

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

    if not failures:
        scope = contract.exact_verification_scope

        if scope is None:
            failures.append(
                VerificationFailure(
                    code=FailureCode.INSUFFICIENT_EVIDENCE,
                    message=(
                        "EXACT requires a declared finite verification scope."
                    ),
                )
            )
        else:
            provided_case_ids = frozenset(case.id for case in cases)

            if not scope.exhaustive_case_ids.issubset(provided_case_ids):
                missing_case_ids = sorted(
                    scope.exhaustive_case_ids.difference(provided_case_ids)
                )
                missing_cases = ", ".join(missing_case_ids)

                failures.append(
                    VerificationFailure(
                        code=FailureCode.INSUFFICIENT_EVIDENCE,
                        message=(
                            "Declared exhaustive verification cases are missing: "
                            f"{missing_cases}."
                        ),
                    )
                )

    decision = decide_exact_or_reject(failures)

    return VerificationResult(
        contract_id=contract.id,
        decision=decision,
        cases_evaluated=len(cases),
        exact_matches=exact_matches,
        reference_observations=reference_observations,
        candidate_observations=candidate_observations,
        failures=failures,
    )
