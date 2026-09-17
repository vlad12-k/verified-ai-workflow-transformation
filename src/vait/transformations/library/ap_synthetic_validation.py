"""Validation metrics for synthetic AP training corpora."""

import json
from dataclasses import dataclass

from vait.transformations.library.ap_training import (
    AP_DECISIONS,
    APTrainingCorpus,
)
from vait.transformations.library.synthetic_ap import (
    synthetic_ap_policy,
)


@dataclass(frozen=True)
class APSyntheticValidationReport:
    """Validation summary for one synthetic AP corpus."""

    case_count: int
    unique_input_count: int
    duplicate_input_count: int
    class_counts: dict[str, int]
    policy_label_mismatch_count: int
    missing_decision_classes: tuple[str, ...]
    invalid_purchase_order_state_count: int

    @property
    def duplicate_rate(self) -> float:
        """Return the fraction of duplicated synthetic inputs."""
        if self.case_count == 0:
            return 0.0

        return (
            self.duplicate_input_count
            / self.case_count
        )

    @property
    def policy_label_agreement_rate(self) -> float:
        """Return agreement between corpus labels and policy oracle."""
        if self.case_count == 0:
            return 0.0

        return (
            self.case_count
            - self.policy_label_mismatch_count
        ) / self.case_count

    @property
    def passed(self) -> bool:
        """Return whether bounded corpus validation checks pass."""
        return (
            self.duplicate_input_count == 0
            and self.policy_label_mismatch_count == 0
            and not self.missing_decision_classes
        )


def validate_synthetic_ap_corpus(
    corpus: APTrainingCorpus,
) -> APSyntheticValidationReport:
    """Evaluate bounded integrity checks for one synthetic corpus."""
    signatures: set[str] = set()
    duplicate_count = 0
    mismatch_count = 0
    invalid_po_state_count = 0

    for case in corpus.cases:
        signature = json.dumps(
            case.input_data,
            sort_keys=True,
            separators=(",", ":"),
        )

        if signature in signatures:
            duplicate_count += 1
        else:
            signatures.add(signature)

        result = synthetic_ap_policy(
            case.input_data
        )

        if not isinstance(result, dict):
            mismatch_count += 1
        else:
            decision = result.get("decision")

            if decision != case.label:
                mismatch_count += 1

        purchase_order_present = (
            case.input_data.get(
                "purchase_order_present"
            )
        )
        purchase_order_amount = (
            case.input_data.get(
                "purchase_order_amount"
            )
        )

        if (
            purchase_order_present is False
            and purchase_order_amount is not None
        ):
            invalid_po_state_count += 1

    class_counts = corpus.class_counts

    missing_classes = tuple(
        decision
        for decision in sorted(AP_DECISIONS)
        if class_counts.get(decision, 0) == 0
    )

    return APSyntheticValidationReport(
        case_count=len(corpus.cases),
        unique_input_count=len(signatures),
        duplicate_input_count=duplicate_count,
        class_counts=class_counts,
        policy_label_mismatch_count=mismatch_count,
        missing_decision_classes=missing_classes,
        invalid_purchase_order_state_count=(
            invalid_po_state_count
        ),
    )
