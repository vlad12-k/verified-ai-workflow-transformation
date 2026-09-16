"""End-to-end integration test for the applied AP transformation."""

from pathlib import Path

from vait.benchmark.evaluator import evaluate_benchmark
from vait.benchmark.loader import load_benchmark
from vait.benchmark.verification import verify_benchmark_bounded
from vait.contracts.models import (
    BoundedVerificationPolicy,
    RiskLevel,
)
from vait.decision.models import Decision
from vait.transformations.applicability import ApplicabilityContext
from vait.transformations.library.synthetic_ap import (
    build_synthetic_ap_transformation,
)
from vait.transformations.registry import TransformationRegistry


def test_ap_transformation_runs_end_to_end() -> None:
    """Applied transformation should pass the complete VAIT evaluation path."""
    dataset = load_benchmark(
        Path("datasets/ap_invoice_exceptions/v0.2/cases.yaml")
    )

    transformation = build_synthetic_ap_transformation()

    registry = TransformationRegistry()
    registry.register(transformation)

    registered = registry.get(
        "synthetic-ap-rule-replacement",
        "1.0.0",
    )

    preparation = registered.prepare_candidate(
        ApplicabilityContext(
            risk_level=RiskLevel.CRITICAL,
            available_capabilities=frozenset(
                {
                    "typed-inputs",
                }
            ),
        )
    )

    assert preparation.ready is True
    assert preparation.runner is not None

    candidate_configuration = {
        "transformation_id": transformation.descriptor.transformation_id,
        "transformation_version": transformation.descriptor.version,
    }

    evaluation = evaluate_benchmark(
        dataset=dataset,
        candidate=preparation.runner,
        candidate_configuration=candidate_configuration,
    )

    assert evaluation.benchmark_version == "0.2"
    assert evaluation.cases_evaluated == 20
    assert evaluation.gold_matches == 20
    assert evaluation.gold_agreement_rate == 1.0
    assert evaluation.high_risk_failures == 0
    assert evaluation.execution_errors == 0

    policy = BoundedVerificationPolicy(
        max_overall_disagreement_rate=0.15,
        max_high_risk_disagreement_rate=0.25,
        confidence_level=0.95,
        min_total_cases=20,
        min_high_risk_cases=11,
    )

    verification = verify_benchmark_bounded(
        dataset=dataset,
        candidate=preparation.runner,
        policy=policy,
        candidate_configuration=candidate_configuration,
    )

    assert verification.decision is Decision.BOUNDED
    assert verification.cases_evaluated == 20
    assert verification.failures == []
    assert verification.statistical_evidence is not None

    evidence = verification.statistical_evidence

    assert evidence.disagreement_rate == 0.0
    assert evidence.disagreement_upper_bound <= 0.15
    assert evidence.high_risk_disagreement_upper_bound is not None
    assert evidence.high_risk_disagreement_upper_bound <= 0.25
