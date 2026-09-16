"""Reproducible synthetic training data for AP model candidates."""

import json
from collections.abc import Iterable
from random import Random

from pydantic import BaseModel, Field, JsonValue

from vait.transformations.library.synthetic_ap import synthetic_ap_policy

AP_DECISIONS = frozenset(
    {
        "RECOMMEND_APPROVE",
        "REVIEW",
        "HOLD",
    }
)

_REVIEW_MISMATCH_DIFFERENCES = (
    0.000001,
    0.01,
    0.10,
    1.00,
    25.00,
    250.00,
    999.99,
)

_HOLD_MISMATCH_DIFFERENCES = (
    1000.00,
    1000.01,
    1250.00,
    2500.00,
    10_000.00,
)


class APTrainingCase(BaseModel):
    """One supervised synthetic AP training example."""

    case_id: str = Field(min_length=1)
    input_data: dict[str, JsonValue]
    label: str = Field(min_length=1)


class APTrainingCorpus(BaseModel):
    """Reproducible collection of supervised AP training examples."""

    seed: int
    cases: tuple[APTrainingCase, ...] = Field(min_length=3)

    @property
    def class_counts(self) -> dict[str, int]:
        """Return deterministic label counts."""
        counts = {
            decision: 0
            for decision in sorted(AP_DECISIONS)
        }

        for case in self.cases:
            counts[case.label] = counts.get(case.label, 0) + 1

        return counts


def build_synthetic_ap_training_corpus(
    *,
    sample_count: int = 600,
    seed: int = 20260916,
    excluded_inputs: Iterable[dict[str, JsonValue]] = (),
) -> APTrainingCorpus:
    """Build a balanced synthetic corpus without evaluation-case leakage."""
    if sample_count < 3:
        raise ValueError("sample_count must be at least 3")

    rng = Random(seed)

    excluded_signatures = {
        _input_signature(input_data)
        for input_data in excluded_inputs
    }

    seen_signatures = set(excluded_signatures)
    cases: list[APTrainingCase] = []

    attempts = 0
    max_attempts = sample_count * 50

    while len(cases) < sample_count:
        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                "Unable to generate enough unique AP training cases."
            )

        target_class = len(cases) % 3
        input_data = _generate_input(
            rng=rng,
            target_class=target_class,
        )

        signature = _input_signature(input_data)

        if signature in seen_signatures:
            continue

        label = _policy_label(input_data)

        expected_label = (
            "RECOMMEND_APPROVE"
            if target_class == 0
            else "REVIEW"
            if target_class == 1
            else "HOLD"
        )

        if label != expected_label:
            raise RuntimeError(
                "Synthetic AP generator produced an unexpected label: "
                f"expected {expected_label}, got {label}."
            )

        seen_signatures.add(signature)

        cases.append(
            APTrainingCase(
                case_id=f"synthetic-train-{len(cases) + 1:05d}",
                input_data=input_data,
                label=label,
            )
        )

    return APTrainingCorpus(
        seed=seed,
        cases=tuple(cases),
    )


def _generate_input(
    *,
    rng: Random,
    target_class: int,
) -> dict[str, JsonValue]:
    """Generate one valid AP input for a requested decision class."""
    invoice_amount = round(
        rng.uniform(25.0, 50_000.0),
        2,
    )

    base: dict[str, JsonValue] = {
        "invoice_amount": invoice_amount,
        "purchase_order_amount": invoice_amount,
        "purchase_order_present": True,
        "supplier_known": True,
        "duplicate_invoice": False,
        "bank_details_changed": False,
        "tax_id_valid": True,
    }

    if target_class == 0:
        return base

    if target_class == 1:
        review_scenario = rng.randrange(6)

        if review_scenario == 0:
            base["purchase_order_present"] = False
            base["purchase_order_amount"] = None

        elif review_scenario == 1:
            base["supplier_known"] = False

        elif review_scenario == 2:
            base["tax_id_valid"] = False

        elif review_scenario == 3:
            difference = rng.choice(
                _REVIEW_MISMATCH_DIFFERENCES
            )
            base["purchase_order_amount"] = round(
                invoice_amount + difference,
                6,
            )

        elif review_scenario == 4:
            if rng.random() < 0.5:
                non_positive_amount = 0.0
            else:
                non_positive_amount = -round(
                    rng.uniform(0.01, 5000.0),
                    2,
                )

            base["invoice_amount"] = non_positive_amount
            base["purchase_order_amount"] = non_positive_amount

        else:
            base["purchase_order_present"] = True
            base["purchase_order_amount"] = None

        return base

    if target_class == 2:
        hold_scenario = rng.randrange(6)

        if hold_scenario == 0:
            base["duplicate_invoice"] = True

        elif hold_scenario == 1:
            base["bank_details_changed"] = True

        elif hold_scenario == 2:
            difference = rng.choice(
                _HOLD_MISMATCH_DIFFERENCES
            )
            base["purchase_order_amount"] = round(
                invoice_amount + difference,
                6,
            )

        elif hold_scenario == 3:
            negative_amount = -round(
                rng.uniform(0.01, 5000.0),
                2,
            )
            base["invoice_amount"] = negative_amount
            base["purchase_order_amount"] = negative_amount
            base["duplicate_invoice"] = True

        elif hold_scenario == 4:
            base["purchase_order_present"] = False
            base["purchase_order_amount"] = None
            base["bank_details_changed"] = True

        else:
            base["supplier_known"] = False
            base["duplicate_invoice"] = True
            base["tax_id_valid"] = False

        return base

    raise ValueError("target_class must be 0, 1, or 2")


def _policy_label(
    input_data: dict[str, JsonValue],
) -> str:
    """Return the supervised label assigned by the synthetic AP policy."""
    result = synthetic_ap_policy(input_data)

    if not isinstance(result, dict):
        raise RuntimeError(
            "Synthetic AP policy must return a mapping."
        )

    decision = result.get("decision")

    if not isinstance(decision, str):
        raise RuntimeError(
            "Synthetic AP policy must return a string decision."
        )

    if decision not in AP_DECISIONS:
        raise RuntimeError(
            f"Unsupported AP decision: {decision}"
        )

    return decision


def _input_signature(
    input_data: dict[str, JsonValue],
) -> str:
    """Return a deterministic identity for one AP input."""
    return json.dumps(
        input_data,
        sort_keys=True,
        separators=(",", ":"),
    )
