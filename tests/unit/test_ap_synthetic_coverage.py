"""Tests for synthetic AP coverage and leakage checks."""

from pydantic import JsonValue

from vait.benchmark.loader import load_benchmark
from vait.transformations.library.ap_synthetic_coverage import (
    evaluate_synthetic_ap_coverage,
)
from vait.transformations.library.ap_training import (
    APTrainingCase,
    APTrainingCorpus,
    build_synthetic_ap_training_corpus,
)


def test_generated_corpus_covers_declared_risk_regions() -> None:
    """The canonical corpus should cover required AP regions."""
    benchmark = load_benchmark(
        "datasets/ap_invoice_exceptions/v0.2/cases.yaml"
    )

    evaluation_inputs = [
        case.input_data
        for case in benchmark.cases
    ]

    corpus = build_synthetic_ap_training_corpus(
        sample_count=600,
        seed=20260916,
        excluded_inputs=evaluation_inputs,
    )

    report = evaluate_synthetic_ap_coverage(
        corpus,
        excluded_inputs=evaluation_inputs,
    )

    assert report.passed is True
    assert report.evaluation_overlap_count == 0
    assert report.precision_boundary_review_count > 0
    assert report.escalation_boundary_hold_count > 0
    assert report.bank_change_missing_po_count > 0
    assert report.duplicate_negative_amount_count > 0
    assert report.inconsistent_purchase_order_count > 0


def test_coverage_detects_evaluation_overlap() -> None:
    """Exact evaluation-input leakage must be reported."""
    corpus = build_synthetic_ap_training_corpus(
        sample_count=30,
        seed=42,
    )

    leaked_input = corpus.cases[0].input_data

    report = evaluate_synthetic_ap_coverage(
        corpus,
        excluded_inputs=(
            leaked_input,
        ),
    )

    assert report.evaluation_overlap_count == 1
    assert report.passed is False


def test_incomplete_corpus_fails_required_coverage() -> None:
    """Class presence alone must not imply risk-region coverage."""
    base_input: dict[str, JsonValue] = {
        "invoice_amount": 100.0,
        "purchase_order_amount": 100.0,
        "purchase_order_present": True,
        "supplier_known": True,
        "duplicate_invoice": False,
        "bank_details_changed": False,
        "tax_id_valid": True,
    }

    corpus = APTrainingCorpus(
        seed=1,
        cases=(
            APTrainingCase(
                case_id="approve",
                input_data=base_input,
                label="RECOMMEND_APPROVE",
            ),
            APTrainingCase(
                case_id="review",
                input_data={
                    **base_input,
                    "supplier_known": False,
                },
                label="REVIEW",
            ),
            APTrainingCase(
                case_id="hold",
                input_data={
                    **base_input,
                    "duplicate_invoice": True,
                },
                label="HOLD",
            ),
        ),
    )

    report = evaluate_synthetic_ap_coverage(
        corpus
    )

    assert report.passed is False
    assert report.precision_boundary_review_count == 0
    assert report.escalation_boundary_hold_count == 0
    assert report.bank_change_missing_po_count == 0
    assert report.duplicate_negative_amount_count == 0
    assert report.inconsistent_purchase_order_count == 0
