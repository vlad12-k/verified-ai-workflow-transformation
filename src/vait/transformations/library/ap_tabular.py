"""Typed tabular feature extraction for synthetic AP model candidates."""

from pydantic import JsonValue

AP_TABULAR_FEATURE_NAMES: tuple[str, ...] = (
    "invoice_amount",
    "purchase_order_amount",
    "purchase_order_amount_missing",
    "absolute_amount_difference",
    "purchase_order_present",
    "supplier_known",
    "duplicate_invoice",
    "bank_details_changed",
    "tax_id_valid",
    "non_positive_invoice",
    "inconsistent_purchase_order_state",
)


def _require_number(
    data: dict[str, JsonValue],
    key: str,
) -> float:
    """Return a required numeric field as float."""
    value = data.get(key)

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")

    return float(value)


def _require_bool(
    data: dict[str, JsonValue],
    key: str,
) -> bool:
    """Return a required boolean field."""
    value = data.get(key)

    if not isinstance(value, bool):
        raise ValueError(f"{key} must be boolean")

    return value


def extract_ap_tabular_features(
    data: dict[str, JsonValue],
) -> tuple[float, ...]:
    """Convert one synthetic AP input into a stable numeric feature vector."""
    invoice_amount = _require_number(
        data,
        "invoice_amount",
    )

    purchase_order_raw = data.get("purchase_order_amount")

    if purchase_order_raw is None:
        purchase_order_amount = 0.0
        purchase_order_amount_missing = 1.0
        absolute_amount_difference = 0.0
    else:
        if not isinstance(
            purchase_order_raw,
            (int, float),
        ) or isinstance(purchase_order_raw, bool):
            raise ValueError(
                "purchase_order_amount must be numeric or null"
            )

        purchase_order_amount = float(purchase_order_raw)
        purchase_order_amount_missing = 0.0
        absolute_amount_difference = abs(
            invoice_amount - purchase_order_amount
        )

    purchase_order_present = _require_bool(
        data,
        "purchase_order_present",
    )
    supplier_known = _require_bool(
        data,
        "supplier_known",
    )
    duplicate_invoice = _require_bool(
        data,
        "duplicate_invoice",
    )
    bank_details_changed = _require_bool(
        data,
        "bank_details_changed",
    )
    tax_id_valid = _require_bool(
        data,
        "tax_id_valid",
    )

    inconsistent_purchase_order_state = (
        purchase_order_present
        and purchase_order_amount_missing == 1.0
    )

    return (
        invoice_amount,
        purchase_order_amount,
        purchase_order_amount_missing,
        absolute_amount_difference,
        float(purchase_order_present),
        float(supplier_known),
        float(duplicate_invoice),
        float(bank_details_changed),
        float(tax_id_valid),
        float(invoice_amount <= 0.0),
        float(inconsistent_purchase_order_state),
    )
