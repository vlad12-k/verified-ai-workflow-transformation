"""Tests for the Keras AP candidate."""

from vait.transformations.library.ap_keras import (
    build_keras_model,
    build_keras_transformation,
    train_keras_model,
)
from vait.transformations.library.ap_training import (
    build_synthetic_ap_training_corpus,
)


def test_keras_model_matches_pytorch_parameter_count() -> None:
    """Equivalent framework architectures should have equal parameter counts."""
    model = build_keras_model(
        seed=20260916,
    )

    assert model.count_params() == 963


def test_keras_training_produces_finite_loss() -> None:
    """A short deterministic corpus should train successfully."""
    corpus = build_synthetic_ap_training_corpus(
        sample_count=30,
    )

    model, final_loss = train_keras_model(
        corpus
    )

    assert model.count_params() == 963
    assert final_loss >= 0.0


def test_keras_transformation_returns_ap_decision() -> None:
    """The fitted Keras candidate should expose the VAIT decision schema."""
    corpus = build_synthetic_ap_training_corpus(
        sample_count=30,
    )

    transformation = build_keras_transformation(
        corpus
    )

    result = transformation.function(
        corpus.cases[0].input_data
    )

    assert result["decision"] in {
        "HOLD",
        "RECOMMEND_APPROVE",
        "REVIEW",
    }


def test_keras_transformation_records_configuration() -> None:
    """The candidate should record reproducibility metadata."""
    corpus = build_synthetic_ap_training_corpus(
        sample_count=30,
    )

    transformation = build_keras_transformation(
        corpus
    )

    assert (
        transformation.configuration[
            "model_family"
        ]
        == "keras_mlp"
    )

    assert (
        transformation.configuration[
            "parameter_count"
        ]
        == 963
    )
