"""Tests for reproducible synthetic AP training data."""

import json

import pytest

from vait.benchmark.loader import load_benchmark
from vait.transformations.library.ap_tabular import (
    AP_TABULAR_FEATURE_NAMES,
    extract_ap_tabular_features,
)
from vait.transformations.library.ap_training import (
    AP_DECISIONS,
    build_synthetic_ap_training_corpus,
)
from vait.transformations.library.synthetic_ap import synthetic_ap_policy


def _signature(data: object) -> str:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
    )


def test_training_corpus_is_reproducible() -> None:
    """The same seed must produce byte-equivalent supervised examples."""
    first = build_synthetic_ap_training_corpus(
        sample_count=30,
        seed=42,
    )
    second = build_synthetic_ap_training_corpus(
        sample_count=30,
        seed=42,
    )

    assert first.model_dump() == second.model_dump()


def test_training_corpus_is_balanced() -> None:
    """A divisible sample count should balance all supported decisions."""
    corpus = build_synthetic_ap_training_corpus(
        sample_count=90,
        seed=7,
    )

    assert corpus.class_counts == {
        "HOLD": 30,
        "RECOMMEND_APPROVE": 30,
        "REVIEW": 30,
    }


def test_training_labels_match_declared_policy() -> None:
    """Every supervised label must agree with the synthetic policy."""
    corpus = build_synthetic_ap_training_corpus(
        sample_count=60,
        seed=11,
    )

    for case in corpus.cases:
        result = synthetic_ap_policy(case.input_data)

        assert isinstance(result, dict)
        assert result["decision"] == case.label
        assert case.label in AP_DECISIONS


def test_training_cases_use_shared_feature_contract() -> None:
    """Generated cases must be consumable by the shared tabular encoder."""
    corpus = build_synthetic_ap_training_corpus(
        sample_count=30,
        seed=21,
    )

    for case in corpus.cases:
        features = extract_ap_tabular_features(
            case.input_data
        )

        assert len(features) == len(
            AP_TABULAR_FEATURE_NAMES
        )


def test_v02_evaluation_inputs_are_excluded() -> None:
    """Training data must not reproduce exact AP v0.2 evaluation inputs."""
    benchmark = load_benchmark(
        "datasets/ap_invoice_exceptions/v0.2/cases.yaml"
    )

    evaluation_inputs = [
        case.input_data
        for case in benchmark.cases
    ]

    corpus = build_synthetic_ap_training_corpus(
        sample_count=300,
        seed=20260916,
        excluded_inputs=evaluation_inputs,
    )

    evaluation_signatures = {
        _signature(input_data)
        for input_data in evaluation_inputs
    }

    training_signatures = {
        _signature(case.input_data)
        for case in corpus.cases
    }

    assert evaluation_signatures.isdisjoint(
        training_signatures
    )
    assert len(training_signatures) == 300


def test_sample_count_below_minimum_is_rejected() -> None:
    """At least one training example per decision class is required."""
    with pytest.raises(
        ValueError,
        match="sample_count must be at least 3",
    ):
        build_synthetic_ap_training_corpus(
            sample_count=2,
        )
