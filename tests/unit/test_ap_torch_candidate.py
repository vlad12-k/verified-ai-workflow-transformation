"""Tests for the synthetic AP PyTorch candidate."""

import pytest

from vait.contracts.models import RiskLevel, VerificationCase
from vait.transformations.applicability import ApplicabilityContext
from vait.transformations.library.ap_torch import (
    build_pytorch_transformation,
)
from vait.transformations.library.ap_training import (
    AP_DECISIONS,
    build_synthetic_ap_training_corpus,
)
from vait.transformations.models import TransformationCategory


@pytest.fixture(scope="module")
def training_corpus():
    """Create one reproducible corpus for PyTorch tests."""
    return build_synthetic_ap_training_corpus(
        sample_count=600,
        seed=20260916,
    )


@pytest.fixture(scope="module")
def transformation(training_corpus):
    """Train one PyTorch candidate for the module."""
    return build_pytorch_transformation(
        training_corpus
    )


def test_pytorch_candidate_has_model_descriptor(
    transformation,
) -> None:
    """PyTorch candidate should expose reproducibility metadata."""
    assert (
        transformation.descriptor.category
        is TransformationCategory.MODEL
    )

    assert transformation.configuration["model_family"] == "pytorch_mlp"
    assert transformation.configuration["training_cases"] == 600
    assert transformation.configuration["training_seed"] == 20260916
    assert transformation.configuration["feature_count"] == 11
    assert transformation.configuration["training_device"] == "cpu"
    assert transformation.configuration["parameter_count"] > 0


def test_pytorch_candidate_executes_through_vait_runner(
    transformation,
) -> None:
    """PyTorch inference should execute through the common VAIT runner."""
    preparation = transformation.prepare_candidate(
        ApplicabilityContext(
            risk_level=RiskLevel.HIGH,
            available_capabilities=frozenset(
                {
                    "typed-inputs",
                    "tabular-features",
                    "trained-model",
                    "pytorch-inference",
                }
            ),
        )
    )

    assert preparation.ready is True
    assert preparation.runner is not None

    observation = preparation.runner.execute(
        VerificationCase(
            id="pytorch-nominal-test",
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


def test_pytorch_candidate_requires_pytorch_capability(
    transformation,
) -> None:
    """Missing PyTorch capability should block candidate preparation."""
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

    assert preparation.ready is False
    assert preparation.runner is None
    assert preparation.applicability.is_applicable is False


def test_pytorch_training_is_reproducible(
    training_corpus,
) -> None:
    """Fixed training seed should reproduce predictions."""
    first = build_pytorch_transformation(
        training_corpus
    )
    second = build_pytorch_transformation(
        training_corpus
    )

    context = ApplicabilityContext(
        risk_level=RiskLevel.LOW,
        available_capabilities=frozenset(
            {
                "typed-inputs",
                "tabular-features",
                "trained-model",
                "pytorch-inference",
            }
        ),
    )

    first_runner = first.prepare_candidate(
        context
    ).runner
    second_runner = second.prepare_candidate(
        context
    ).runner

    assert first_runner is not None
    assert second_runner is not None

    case = VerificationCase(
        id="reproducibility-test",
        input_data={
            "invoice_amount": 1000.01,
            "purchase_order_amount": 1000.0,
            "purchase_order_present": True,
            "supplier_known": True,
            "duplicate_invoice": False,
            "bank_details_changed": False,
            "tax_id_valid": True,
        },
        risk_level=RiskLevel.MEDIUM,
    )

    first_observation = first_runner.execute(case)
    second_observation = second_runner.execute(case)

    assert first_observation.output == second_observation.output


def test_pytorch_feature_transform_separates_tiny_mismatch() -> None:
    """Exact and tiny positive mismatches should have a clear model margin."""
    from vait.transformations.library.ap_torch import (
        _transform_features,
    )

    exact = _transform_features(
        (
            1000.0,
            1000.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        )
    )

    tiny_mismatch = _transform_features(
        (
            1000.000002,
            1000.0,
            0.0,
            0.000002,
            1.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        )
    )

    assert exact[3] == 0.0
    assert tiny_mismatch[3] > 1.0
