"""Evaluate semantic-drift guard on adversarial development cases."""

import json
from pathlib import Path

from vait.rag.semantic_guard import assess_semantic_drift

DATASET_PATH = Path(
    "datasets/ap_policy_rag/v0.1/"
    "semantic_guard_adversarial_v01.json"
)

payload = json.loads(
    DATASET_PATH.read_text(encoding="utf-8")
)

true_positive = 0
false_positive = 0
false_negative = 0
true_negative = 0

print(
    "Semantic-drift guard adversarial evaluation"
)
print()

for case in payload["cases"]:
    assessment = assess_semantic_drift(
        context_text=case["context"],
        answer_text=case["answer"],
    )

    observed_flag = not assessment.passed
    expected_flag = case["expected_flag"]

    if expected_flag and observed_flag:
        outcome = "TP"
        true_positive += 1
    elif not expected_flag and observed_flag:
        outcome = "FP"
        false_positive += 1
    elif expected_flag and not observed_flag:
        outcome = "FN"
        false_negative += 1
    else:
        outcome = "TN"
        true_negative += 1

    issues = ", ".join(
        issue.risk.value
        for issue in assessment.issues
    )

    if not issues:
        issues = "-"

    print(
        f"{case['case_id']:<32} "
        f"expected={'FLAG' if expected_flag else 'PASS':<4} "
        f"observed={'FLAG' if observed_flag else 'PASS':<4} "
        f"outcome={outcome:<2} "
        f"issues={issues}"
    )

risky_count = (
    true_positive + false_negative
)

safe_count = (
    true_negative + false_positive
)

recall = (
    true_positive / risky_count
    if risky_count
    else 0.0
)

precision = (
    true_positive
    / (true_positive + false_positive)
    if true_positive + false_positive
    else 0.0
)

false_positive_rate = (
    false_positive / safe_count
    if safe_count
    else 0.0
)

print()
print("Confusion counts")
print(f"TP: {true_positive}")
print(f"FP: {false_positive}")
print(f"FN: {false_negative}")
print(f"TN: {true_negative}")
print(f"Risk recall: {recall:.3f}")
print(f"Flag precision: {precision:.3f}")
print(
    "False-positive rate:",
    f"{false_positive_rate:.3f}",
)
