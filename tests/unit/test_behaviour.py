"""Tests for VAIT behavioural statistical evidence."""

from vait.contracts.models import RiskLevel, VerificationCase
from vait.decision.models import ExecutionObservation
from vait.verification.behaviour import build_statistical_evidence


def observation(
    implementation_id: str,
    case_id: str,
    output: str,
) -> ExecutionObservation:
    """Create a simple successful observation."""
    return ExecutionObservation(
        implementation_id=implementation_id,
        case_id=case_id,
        output={"decision": output},
        latency_ms=1.0,
    )


def test_behavioural_evidence_is_risk_stratified() -> None:
    """Disagreements should be counted by business-risk stratum."""
    cases = [
        VerificationCase(
            id="low",
            input_data={},
            risk_level=RiskLevel.LOW,
        ),
        VerificationCase(
            id="high",
            input_data={},
            risk_level=RiskLevel.HIGH,
        ),
        VerificationCase(
            id="critical",
            input_data={},
            risk_level=RiskLevel.CRITICAL,
        ),
    ]

    reference = [
        observation("reference", "low", "A"),
        observation("reference", "high", "A"),
        observation("reference", "critical", "A"),
    ]

    candidate = [
        observation("candidate", "low", "A"),
        observation("candidate", "high", "B"),
        observation("candidate", "critical", "A"),
    ]

    evidence = build_statistical_evidence(
        cases=cases,
        reference_observations=reference,
        candidate_observations=candidate,
        confidence_level=0.95,
    )

    assert evidence.cases_evaluated == 3
    assert evidence.disagreements == 1
    assert evidence.high_risk_cases_evaluated == 2
    assert evidence.high_risk_disagreements == 1
    assert len(evidence.risk_strata) == 3
