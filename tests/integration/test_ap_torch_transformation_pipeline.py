"""End-to-end verification of the AP PyTorch transformation."""

from vait.benchmark.loader import load_benchmark
from vait.benchmark.transformation_assessment import (
    assess_python_transformations,
)
from vait.contracts.models import (
    BoundedVerificationPolicy,
    RiskLevel,
)
from vait.decision.models import Decision
from vait.transformations.applicability import ApplicabilityContext
from vait.transformations.library.ap_torch import (
    build_pytorch_transformation,
)
from vait.transformations.library.ap_training import (
    build_synthetic_ap_training_corpus,
)


def test_ap_pytorch_candidate_is_verified_on_v02_validation_benchmark() -> None:
    """Verify the PyTorch candidate on the v0.2 validation benchmark."""
    dataset = load_benchmark(
        "datasets/ap_invoice_exceptions/v0.2/cases.yaml"
    )

    corpus = build_synthetic_ap_training_corpus(
        sample_count=600,
        seed=20260916,
        excluded_inputs=(
            case.input_data
            for case in dataset.cases
        ),
    )

    transformation = build_pytorch_transformation(
        corpus
    )

    report = assess_python_transformations(
        dataset=dataset,
        transformations=(transformation,),
        context=ApplicabilityContext(
            risk_level=RiskLevel.CRITICAL,
            available_capabilities=frozenset(
                {
                    "typed-inputs",
                    "tabular-features",
                    "trained-model",
                    "pytorch-inference",
                }
            ),
        ),
        policy=BoundedVerificationPolicy(
            max_overall_disagreement_rate=0.15,
            max_high_risk_disagreement_rate=0.25,
            confidence_level=0.95,
            min_total_cases=20,
            min_high_risk_cases=11,
        ),
    )

    assert len(report.assessments) == 1

    assessment = report.assessments[0]

    assert (
        assessment.candidate_implementation_id
        == "synthetic-ap-pytorch-mlp-v1"
    )
    assert assessment.applicability.is_applicable is True
    assert assessment.verification is not None
    assert assessment.verification.cases_evaluated == 20

    assert assessment.decision is Decision.BOUNDED

    evidence = assessment.verification.statistical_evidence

    assert evidence is not None
    assert evidence.disagreements == 0
    assert evidence.high_risk_disagreements == 0

    provenance = assessment.verification.provenance

    assert provenance is not None
    assert provenance.candidate_configuration["model_family"] == "pytorch_mlp"
    assert provenance.candidate_configuration["training_seed"] == 20260916
    assert provenance.candidate_configuration["training_cases"] == 600
    assert provenance.candidate_configuration["feature_count"] == 11
    assert provenance.candidate_configuration["training_device"] == "cpu"
    assert provenance.package_versions["torch"] != "not-installed"
