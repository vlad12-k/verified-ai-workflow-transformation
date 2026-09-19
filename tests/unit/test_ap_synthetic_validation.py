"""Tests for synthetic AP corpus validation."""

from pydantic import JsonValue

from vait.transformations.library.ap_synthetic_validation import (
    validate_synthetic_ap_corpus,
)
from vait.transformations.library.ap_training import (
    APTrainingCase,
    APTrainingCorpus,
    build_synthetic_ap_training_corpus,
)


def test_generated_corpus_passes_validation() -> None:
    """Current deterministic corpus should pass bounded checks."""
    corpus = build_synthetic_ap_training_corpus(
        sample_count=600,
        seed=20260916,
    )

    report = validate_synthetic_ap_corpus(
        corpus
    )

    assert report.passed is True
    assert report.case_count == 600
    assert report.unique_input_count == 600
    assert report.duplicate_input_count == 0
    assert report.policy_label_mismatch_count == 0
    assert report.missing_decision_classes == ()
    assert report.policy_label_agreement_rate == 1.0
    assert report.duplicate_rate == 0.0


def test_validation_detects_duplicate_inputs() -> None:
    """Exact duplicate synthetic inputs should be reported."""
    input_data: dict[str, JsonValue] = {
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
                case_id="case-1",
                input_data=input_data,
                label="RECOMMEND_APPROVE",
            ),
            APTrainingCase(
                case_id="case-2",
                input_data=input_data,
                label="RECOMMEND_APPROVE",
            ),
            APTrainingCase(
                case_id="case-3",
                input_data={
                    **input_data,
                    "supplier_known": False,
                },
                label="REVIEW",
            ),
        ),
    )

    report = validate_synthetic_ap_corpus(
        corpus
    )

    assert report.passed is False
    assert report.duplicate_input_count == 1
    assert report.unique_input_count == 2


def test_validation_detects_policy_label_mismatch() -> None:
    """Incorrect synthetic supervision should be rejected."""
    corpus = APTrainingCorpus(
        seed=1,
        cases=(
            APTrainingCase(
                case_id="approve",
                input_data={
                    "invoice_amount": 100.0,
                    "purchase_order_amount": 100.0,
                    "purchase_order_present": True,
                    "supplier_known": True,
                    "duplicate_invoice": False,
                    "bank_details_changed": False,
                    "tax_id_valid": True,
                },
                label="HOLD",
            ),
            APTrainingCase(
                case_id="review",
                input_data={
                    "invoice_amount": 100.0,
                    "purchase_order_amount": None,
                    "purchase_order_present": False,
                    "supplier_known": True,
                    "duplicate_invoice": False,
                    "bank_details_changed": False,
                    "tax_id_valid": True,
                },
                label="REVIEW",
            ),
            APTrainingCase(
                case_id="hold",
                input_data={
                    "invoice_amount": 100.0,
                    "purchase_order_amount": 100.0,
                    "purchase_order_present": True,
                    "supplier_known": True,
                    "duplicate_invoice": True,
                    "bank_details_changed": False,
                    "tax_id_valid": True,
                },
                label="HOLD",
            ),
        ),
    )

    report = validate_synthetic_ap_corpus(
        corpus
    )

    assert report.passed is False
    assert report.policy_label_mismatch_count == 1


def test_validation_reports_missing_class() -> None:
    """Corpus validation should expose absent decision classes."""
    corpus = APTrainingCorpus(
        seed=1,
        cases=(
            APTrainingCase(
                case_id="approve-1",
                input_data={
                    "invoice_amount": 100.0,
                    "purchase_order_amount": 100.0,
                    "purchase_order_present": True,
                    "supplier_known": True,
                    "duplicate_invoice": False,
                    "bank_details_changed": False,
                    "tax_id_valid": True,
                },
                label="RECOMMEND_APPROVE",
            ),
            APTrainingCase(
                case_id="approve-2",
                input_data={
                    "invoice_amount": 200.0,
                    "purchase_order_amount": 200.0,
                    "purchase_order_present": True,
                    "supplier_known": True,
                    "duplicate_invoice": False,
                    "bank_details_changed": False,
                    "tax_id_valid": True,
                },
                label="RECOMMEND_APPROVE",
            ),
            APTrainingCase(
                case_id="review",
                input_data={
                    "invoice_amount": 100.0,
                    "purchase_order_amount": None,
                    "purchase_order_present": False,
                    "supplier_known": True,
                    "duplicate_invoice": False,
                    "bank_details_changed": False,
                    "tax_id_valid": True,
                },
                label="REVIEW",
            ),
        ),
    )

    report = validate_synthetic_ap_corpus(
        corpus
    )

    assert report.passed is False
    assert report.missing_decision_classes == (
        "HOLD",
    )
