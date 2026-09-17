"""Evaluate deterministic semantic-drift guard on strict-v2 Qwen outputs."""

import json
from pathlib import Path

from vait.rag.faithfulness import (
    FaithfulnessVerdict,
    answer_sha256,
    load_faithfulness_annotations,
)
from vait.rag.semantic_guard import assess_semantic_drift

ARTIFACT_PATH = Path(
    "artifacts/benchmarks/"
    "ap-policy-rag-qwen-strict-v0.2.json"
)

ANNOTATION_PATH = Path(
    "datasets/ap_policy_rag/v0.1/"
    "qwen_strict_v02_faithfulness_annotations.json"
)

artifact = json.loads(
    ARTIFACT_PATH.read_text(
        encoding="utf-8"
    )
)

annotations = load_faithfulness_annotations(
    ANNOTATION_PATH
)

annotations_by_case_id = {
    annotation.case_id: annotation
    for annotation in annotations.annotations
}

true_positive = 0
false_positive = 0
false_negative = 0
true_negative = 0

print("Semantic-drift guard evaluation")
print()

for result, trace in zip(
    artifact["report"]["results"],
    artifact["traces"],
    strict=True,
):
    if trace["abstained"]:
        continue

    case_id = result["case_id"]

    annotation = annotations_by_case_id[
        case_id
    ]

    answer = trace["answer"]

    observed_hash = answer_sha256(
        answer
    )

    if observed_hash != annotation.answer_sha256:
        raise ValueError(
            f"Generated answer hash changed for {case_id}."
        )

    context_documents = trace["context"]

    context_text = "\n\n".join(
        document["text"]
        for document in context_documents
    )

    assessment = assess_semantic_drift(
        context_text=context_text,
        answer_text=answer,
    )

    manual_risky = (
        annotation.verdict
        is not FaithfulnessVerdict.SUPPORTED
    )

    guard_flagged = not assessment.passed

    if manual_risky and guard_flagged:
        true_positive += 1
        outcome = "TP"
    elif not manual_risky and guard_flagged:
        false_positive += 1
        outcome = "FP"
    elif manual_risky and not guard_flagged:
        false_negative += 1
        outcome = "FN"
    else:
        true_negative += 1
        outcome = "TN"

    issues = ", ".join(
        issue.risk.value
        for issue in assessment.issues
    )

    if not issues:
        issues = "-"

    print(
        f"{case_id:<20} "
        f"manual={annotation.verdict.value:<11} "
        f"guard={'FLAG':<4} "
        if guard_flagged
        else (
            f"{case_id:<20} "
            f"manual={annotation.verdict.value:<11} "
            f"guard={'PASS':<4} "
        ),
        end="",
    )

    print(
        f"outcome={outcome} "
        f"issues={issues}"
    )

risky_total = (
    true_positive
    + false_negative
)

safe_total = (
    true_negative
    + false_positive
)

risk_recall = (
    true_positive / risky_total
    if risky_total
    else 0.0
)

false_positive_rate = (
    false_positive / safe_total
    if safe_total
    else 0.0
)

print()
print("Confusion counts")
print(f"TP: {true_positive}")
print(f"FP: {false_positive}")
print(f"FN: {false_negative}")
print(f"TN: {true_negative}")
print(f"Risk recall: {risk_recall:.3f}")
print(
    "False-positive rate:",
    f"{false_positive_rate:.3f}",
)
