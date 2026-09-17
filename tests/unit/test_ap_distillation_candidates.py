"""Tests for AP distillation candidate transformations."""

from vait.contracts.models import RiskLevel
from vait.transformations.applicability import (
    ApplicabilityContext,
)
from vait.transformations.library.ap_distillation_candidates import (
    build_ap_distillation_transformations,
)
from vait.transformations.library.ap_training import (
    build_synthetic_ap_training_corpus,
)


def test_builds_two_distillation_candidates() -> None:
    """Both compact training strategies should be exposed."""
    corpus = build_synthetic_ap_training_corpus()

    hard, distilled = (
        build_ap_distillation_transformations(
            corpus
        )
    )

    assert (
        hard.implementation_id
        == "synthetic-ap-pytorch-student-hard-label-v1"
    )
    assert (
        distilled.implementation_id
        == "synthetic-ap-pytorch-student-distilled-v1"
    )

    assert (
        hard.configuration["training_method"]
        == "hard_labels"
    )
    assert (
        distilled.configuration["training_method"]
        == "knowledge_distillation"
    )


def test_distillation_candidates_are_applicable() -> None:
    """Compact students should support the AP benchmark context."""
    corpus = build_synthetic_ap_training_corpus()

    hard, distilled = (
        build_ap_distillation_transformations(
            corpus
        )
    )

    context = ApplicabilityContext(
        risk_level=RiskLevel.CRITICAL,
        available_capabilities=frozenset(
            {
                "typed-inputs",
                "tabular-features",
                "trained-model",
                "pytorch-inference",
            }
        ),
    )

    assert hard.assess_applicability(
        context
    ).is_applicable

    assert distilled.assess_applicability(
        context
    ).is_applicable


def test_distillation_candidates_return_decisions() -> None:
    """Both candidates should expose the VAIT decision schema."""
    corpus = build_synthetic_ap_training_corpus()

    hard, distilled = (
        build_ap_distillation_transformations(
            corpus
        )
    )

    sample = corpus.cases[0].input_data

    hard_result = hard.function(
        sample
    )

    distilled_result = distilled.function(
        sample
    )

    assert hard_result["decision"] in {
        "HOLD",
        "RECOMMEND_APPROVE",
        "REVIEW",
    }

    assert distilled_result["decision"] in {
        "HOLD",
        "RECOMMEND_APPROVE",
        "REVIEW",
    }
