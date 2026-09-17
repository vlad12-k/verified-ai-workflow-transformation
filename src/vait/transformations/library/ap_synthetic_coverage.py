"""Coverage and leakage checks for synthetic AP training corpora."""

import json
from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import JsonValue

from vait.transformations.library.ap_training import (
    APTrainingCorpus,
)


@dataclass(frozen=True)
class APSyntheticCoverageReport:
    """Coverage summary for one synthetic AP corpus."""

    evaluation_overlap_count: int
    precision_boundary_review_count: int
    escalation_boundary_hold_count: int
    bank_change_missing_po_count: int
    duplicate_negative_amount_count: int
    inconsistent_purchase_order_count: int

    @property
    def passed(self) -> bool:
        """Return whether required development coverage is present."""
        return (
            self.evaluation_overlap_count == 0
            and self.precision_boundary_review_count > 0
            and self.escalation_boundary_hold_count > 0
            and self.bank_change_missing_po_count > 0
            and self.duplicate_negative_amount_count > 0
            and self.inconsistent_purchase_order_count > 0
        )


def evaluate_synthetic_ap_coverage(
    corpus: APTrainingCorpus,
    *,
    excluded_inputs: Iterable[
        dict[str, JsonValue]
    ] = (),
) -> APSyntheticCoverageReport:
    """Evaluate leakage and declared AP risk-region coverage."""
    excluded_signatures = {
        _input_signature(input_data)
        for input_data in excluded_inputs
    }

    overlap_count = 0
    precision_reviews = 0
    escalation_holds = 0
    bank_change_missing_po = 0
    duplicate_negative = 0
    inconsistent_po = 0

    for case in corpus.cases:
        input_data = case.input_data

        if (
            _input_signature(input_data)
            in excluded_signatures
        ):
            overlap_count += 1

        invoice_amount = input_data.get(
            "invoice_amount"
        )
        purchase_order_amount = input_data.get(
            "purchase_order_amount"
        )

        if (
            isinstance(invoice_amount, (int, float))
            and not isinstance(invoice_amount, bool)
            and isinstance(
                purchase_order_amount,
                (int, float),
            )
            and not isinstance(
                purchase_order_amount,
                bool,
            )
        ):
            difference = abs(
                float(invoice_amount)
                - float(purchase_order_amount)
            )

            if (
                case.label == "REVIEW"
                and 0.0 < difference <= 0.01
            ):
                precision_reviews += 1

            if (
                case.label == "HOLD"
                and 1000.0
                <= difference
                <= 1000.01
            ):
                escalation_holds += 1

        if (
            case.label == "HOLD"
            and input_data.get(
                "bank_details_changed"
            )
            is True
            and input_data.get(
                "purchase_order_present"
            )
            is False
        ):
            bank_change_missing_po += 1

        if (
            case.label == "HOLD"
            and input_data.get(
                "duplicate_invoice"
            )
            is True
            and isinstance(
                invoice_amount,
                (int, float),
            )
            and not isinstance(
                invoice_amount,
                bool,
            )
            and float(invoice_amount) < 0.0
        ):
            duplicate_negative += 1

        if (
            case.label == "REVIEW"
            and input_data.get(
                "purchase_order_present"
            )
            is True
            and purchase_order_amount is None
        ):
            inconsistent_po += 1

    return APSyntheticCoverageReport(
        evaluation_overlap_count=overlap_count,
        precision_boundary_review_count=(
            precision_reviews
        ),
        escalation_boundary_hold_count=(
            escalation_holds
        ),
        bank_change_missing_po_count=(
            bank_change_missing_po
        ),
        duplicate_negative_amount_count=(
            duplicate_negative
        ),
        inconsistent_purchase_order_count=(
            inconsistent_po
        ),
    )


def _input_signature(
    input_data: dict[str, JsonValue],
) -> str:
    """Return deterministic identity for one AP input."""
    return json.dumps(
        input_data,
        sort_keys=True,
        separators=(",", ":"),
    )
