"""Tests for synthetic AP tabular feature extraction."""

import pytest

from vait.transformations.library.ap_tabular import (
    AP_TABULAR_FEATURE_NAMES,
    extract_ap_tabular_features,
)


def test_exact_match_is_encoded_deterministically() -> None:
    """Nominal AP input should produce the documented feature order."""
    features = extract_ap_tabular_features(
        {
            "invoice_amount": 500.0,
            "purchase_order_amount": 500.0,
            "purchase_order_present": True,
            "supplier_known": True,
            "duplicate_invoice": False,
            "bank_details_changed": False,
            "tax_id_valid": True,
        }
    )

    assert len(features) == len(AP_TABULAR_FEATURE_NAMES)

    assert features == (
        500.0,
        500.0,
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


def test_missing_purchase_order_is_encoded_explicitly() -> None:
    """Missing PO amount should not be confused with a numeric zero."""
    features = extract_ap_tabular_features(
        {
            "invoice_amount": 800.0,
            "purchase_order_amount": None,
            "purchase_order_present": False,
            "supplier_known": True,
            "duplicate_invoice": False,
            "bank_details_changed": False,
            "tax_id_valid": True,
        }
    )

    assert features[1] == 0.0
    assert features[2] == 1.0
    assert features[3] == 0.0
    assert features[10] == 0.0


def test_inconsistent_purchase_order_state_is_encoded() -> None:
    """Conflicting PO presence and amount should receive its own feature."""
    features = extract_ap_tabular_features(
        {
            "invoice_amount": 750.0,
            "purchase_order_amount": None,
            "purchase_order_present": True,
            "supplier_known": True,
            "duplicate_invoice": False,
            "bank_details_changed": False,
            "tax_id_valid": True,
        }
    )

    assert features[10] == 1.0


def test_non_positive_invoice_is_encoded() -> None:
    """Non-positive invoices should expose an explicit boundary feature."""
    features = extract_ap_tabular_features(
        {
            "invoice_amount": -250.0,
            "purchase_order_amount": -250.0,
            "purchase_order_present": True,
            "supplier_known": True,
            "duplicate_invoice": False,
            "bank_details_changed": False,
            "tax_id_valid": True,
        }
    )

    assert features[9] == 1.0


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("invoice_amount", "500", "invoice_amount must be numeric"),
        (
            "purchase_order_amount",
            "500",
            "purchase_order_amount must be numeric or null",
        ),
        (
            "purchase_order_present",
            1,
            "purchase_order_present must be boolean",
        ),
        ("supplier_known", 1, "supplier_known must be boolean"),
        ("duplicate_invoice", 0, "duplicate_invoice must be boolean"),
        (
            "bank_details_changed",
            0,
            "bank_details_changed must be boolean",
        ),
        ("tax_id_valid", 1, "tax_id_valid must be boolean"),
    ],
)
def test_invalid_feature_values_are_rejected(
    key: str,
    value: object,
    message: str,
) -> None:
    """Invalid AP feature types should fail before model execution."""
    data = {
        "invoice_amount": 500.0,
        "purchase_order_amount": 500.0,
        "purchase_order_present": True,
        "supplier_known": True,
        "duplicate_invoice": False,
        "bank_details_changed": False,
        "tax_id_valid": True,
    }

    data[key] = value  # type: ignore[assignment]

    with pytest.raises(
        ValueError,
        match=message,
    ):
        extract_ap_tabular_features(data)
