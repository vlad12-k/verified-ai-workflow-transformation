"""Tests for synthetic AP classical ML candidates."""

import pytest

from vait.contracts.models import RiskLevel, VerificationCase
from vait.transformations.applicability import ApplicabilityContext
from vait.transformations.library.ap_ml import (
    build_random_forest_transformation,
    build_xgboost_transformation,
)
from vait.transformations.library.ap_training import (
    AP_DECISIONS,
    build_synthetic_ap_training_corpus,
)
from vait.transformations.models import TransformationCategory


@pytest.fixture(scope="module")
def training_corpus():
    """Create one reproducible corpus for model tests."""
    return build_synthetic_ap_training_corpus(
        sample_count=600,
        seed=20260916,
    )


@pytest.mark.parametrize(
    "builder",
    [
        build_random_forest_transformation,
        build_xgboost_transformation,
    ],
)
def test_ml_candidate_has_model_descriptor(
    builder,
    training_corpus,
) -> None:
    """ML candidates should expose model-specific VAIT metadata."""
    transformation = builder(training_corpus)

    assert transformation.descriptor.category is TransformationCategory.MODEL
    assert transformation.configuration["training_cases"] == 600
    assert transformation.configuration["training_seed"] == 20260916
    assert transformation.configuration["feature_count"] == 11


@pytest.mark.parametrize(
    "builder",
    [
        build_random_forest_transformation,
        build_xgboost_transformation,
    ],
)
def test_ml_candidate_executes_through_vait_runner(
    builder,
    training_corpus,
) -> None:
    """Fitted ML candidates should execute through the common runner."""
    transformation = builder(training_corpus)

    preparation = transformation.prepare_candidate(
        ApplicabilityContext(
            risk_level=RiskLevel.HIGH,
            available_capabilities=frozenset(
                {
                    "typed-inputs",
                    "tabular-features",
                    "trained-model",
                }
            ),
        )
    )

    assert preparation.ready is True
    assert preparation.runner is not None

    observation = preparation.runner.execute(
        VerificationCase(
            id="nominal-test",
            input_data={
                "invoice_amount": 500.0,
                "purchase_order_amount": 500.0,
                "purchase_order_present": True,
                "supplier_known": True,
                "duplicate_invoice": False,
                "bank_details_changed": False,
                "tax_id_valid": True,
            },
            risk_level=RiskLevel.LOW,
        )
    )

    assert observation.error is None
    assert isinstance(observation.output, dict)
    assert observation.output["decision"] in AP_DECISIONS


@pytest.mark.parametrize(
    "builder",
    [
        build_random_forest_transformation,
        build_xgboost_transformation,
    ],
)
def test_ml_candidate_requires_training_capabilities(
    builder,
    training_corpus,
) -> None:
    """Missing model capabilities should block candidate preparation."""
    transformation = builder(training_corpus)

    preparation = transformation.prepare_candidate(
        ApplicabilityContext(
            risk_level=RiskLevel.HIGH,
            available_capabilities=frozenset(
                {
                    "typed-inputs",
                }
            ),
        )
    )

    assert preparation.ready is False
    assert preparation.runner is None
    assert preparation.applicability.is_applicable is False


@pytest.mark.parametrize(
    "builder",
    [
        pytest.param(
            "random_forest",
            id="random-forest",
        ),
        pytest.param(
            "xgboost",
            id="xgboost",
        ),
    ],
)
def test_ml_native_batch_matches_scalar_predictions(
    builder,
    training_corpus,
) -> None:
    """Vectorised ML inference must preserve scalar candidate behaviour."""
    from vait.transformations.library.ap_ml import (
        build_random_forest_native_batch_candidate,
        build_xgboost_native_batch_candidate,
    )

    native_builder = (
        build_random_forest_native_batch_candidate
        if builder == "random_forest"
        else build_xgboost_native_batch_candidate
    )

    transformation, predict_batch = (
        native_builder(
            training_corpus
        )
    )

    preparation = (
        transformation.prepare_candidate(
            ApplicabilityContext(
                risk_level=RiskLevel.HIGH,
                available_capabilities=frozenset(
                    {
                        "typed-inputs",
                        "tabular-features",
                        "trained-model",
                    }
                ),
            )
        )
    )

    assert preparation.runner is not None

    cases = tuple(
        VerificationCase(
            id=case.case_id,
            input_data=case.input_data,
            risk_level=RiskLevel.LOW,
        )
        for case in training_corpus.cases[
            :8
        ]
    )

    batch_outputs = predict_batch(
        tuple(
            case.input_data
            for case in cases
        )
    )

    scalar_outputs = tuple(
        preparation.runner.execute(
            case
        ).output
        for case in cases
    )

    assert len(
        batch_outputs
    ) == len(cases)

    assert (
        batch_outputs
        == scalar_outputs
    )
