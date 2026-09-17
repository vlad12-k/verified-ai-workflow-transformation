"""Generate and validate the canonical synthetic AP training corpus."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from vait.benchmark.loader import load_benchmark
from vait.transformations.library.ap_synthetic_coverage import (
    evaluate_synthetic_ap_coverage,
)
from vait.transformations.library.ap_synthetic_validation import (
    validate_synthetic_ap_corpus,
)
from vait.transformations.library.ap_training import (
    build_synthetic_ap_training_corpus,
)

SAMPLE_COUNT = 600
SEED = 20260916

EVALUATION_DATASET_PATH = Path(
    "datasets/ap_invoice_exceptions/v0.2/cases.yaml"
)

OUTPUT_PATH = Path(
    "artifacts/benchmarks/"
    "ap-synthetic-training-corpus-v0.1.json"
)


benchmark = load_benchmark(
    EVALUATION_DATASET_PATH
)

evaluation_inputs = [
    case.input_data
    for case in benchmark.cases
]

corpus = build_synthetic_ap_training_corpus(
    sample_count=SAMPLE_COUNT,
    seed=SEED,
    excluded_inputs=evaluation_inputs,
)

validation = validate_synthetic_ap_corpus(
    corpus
)

coverage = evaluate_synthetic_ap_coverage(
    corpus,
    excluded_inputs=evaluation_inputs,
)

canonical_corpus = json.dumps(
    corpus.model_dump(mode="json"),
    sort_keys=True,
    separators=(",", ":"),
)

corpus_fingerprint = hashlib.sha256(
    canonical_corpus.encode("utf-8")
).hexdigest()

if not validation.passed:
    raise RuntimeError(
        "Synthetic AP integrity validation failed."
    )

if not coverage.passed:
    raise RuntimeError(
        "Synthetic AP coverage validation failed."
    )

report = {
    "artifact_id": (
        "ap-synthetic-training-corpus-v0.1"
    ),
    "development_evidence": True,
    "generator": {
        "sample_count": SAMPLE_COUNT,
        "seed": SEED,
    },
    "evaluation_exclusion": {
        "dataset": str(
            EVALUATION_DATASET_PATH
        ),
        "evaluation_case_count": len(
            evaluation_inputs
        ),
        "exact_input_overlap_count": (
            coverage.evaluation_overlap_count
        ),
    },
    "corpus": {
        "case_count": validation.case_count,
        "unique_input_count": (
            validation.unique_input_count
        ),
        "duplicate_input_count": (
            validation.duplicate_input_count
        ),
        "duplicate_rate": (
            validation.duplicate_rate
        ),
        "class_counts": (
            validation.class_counts
        ),
        "sha256": corpus_fingerprint,
    },
    "integrity": {
        "passed": validation.passed,
        "policy_label_mismatch_count": (
            validation.policy_label_mismatch_count
        ),
        "policy_label_agreement_rate": (
            validation.policy_label_agreement_rate
        ),
        "missing_decision_classes": list(
            validation.missing_decision_classes
        ),
        "invalid_purchase_order_state_count": (
            validation.invalid_purchase_order_state_count
        ),
    },
    "coverage": {
        **asdict(coverage),
        "passed": coverage.passed,
    },
}

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

print(
    "Synthetic AP training corpus evidence"
)
print()
print(f"Cases: {validation.case_count}")
print(
    "Unique inputs:",
    validation.unique_input_count,
)
print(
    "Class counts:",
    validation.class_counts,
)
print(
    "Policy-label agreement:",
    f"{validation.policy_label_agreement_rate:.3f}",
)
print(
    "Duplicate rate:",
    f"{validation.duplicate_rate:.3f}",
)
print(
    "Evaluation overlap:",
    coverage.evaluation_overlap_count,
)
print(
    "Precision-boundary REVIEW:",
    coverage.precision_boundary_review_count,
)
print(
    "Escalation-boundary HOLD:",
    coverage.escalation_boundary_hold_count,
)
print(
    "Bank-change + missing-PO:",
    coverage.bank_change_missing_po_count,
)
print(
    "Duplicate + negative amount:",
    coverage.duplicate_negative_amount_count,
)
print(
    "Inconsistent PO state:",
    coverage.inconsistent_purchase_order_count,
)
print(
    "Corpus SHA256:",
    corpus_fingerprint,
)
print(
    "Integrity validation:",
    "PASS" if validation.passed else "FAIL",
)
print(
    "Coverage validation:",
    "PASS" if coverage.passed else "FAIL",
)
print()
print(f"Evidence report: {OUTPUT_PATH}")
