"""Evaluate reviewed faithfulness for guarded Qwen RAG v0.3."""

import json
from pathlib import Path

from vait.rag.faithfulness import (
    evaluate_faithfulness_annotations,
    load_faithfulness_annotations,
)

ARTIFACT_PATH = Path(
    "artifacts/benchmarks/"
    "ap-policy-rag-qwen-strict-guarded-v0.3.json"
)

ANNOTATION_PATH = Path(
    "datasets/ap_policy_rag/v0.1/"
    "qwen_guarded_v03_faithfulness_annotations.json"
)

artifact = json.loads(
    ARTIFACT_PATH.read_text(
        encoding="utf-8"
    )
)

annotations = load_faithfulness_annotations(
    ANNOTATION_PATH
)

answers_by_case_id: dict[str, str] = {}

for result, trace in zip(
    artifact["report"]["results"],
    artifact["traces"],
    strict=True,
):
    if trace["abstained"]:
        continue

    answers_by_case_id[
        result["case_id"]
    ] = trace["answer"]

report = evaluate_faithfulness_annotations(
    annotations,
    answers_by_case_id=answers_by_case_id,
)

print(f"Experiment: {report.experiment_id}")
print(f"Reviewed: {report.reviewed_count}")
print(f"Supported: {report.supported_count}")
print(f"Partial: {report.partial_count}")
print(f"Unsupported: {report.unsupported_count}")
print(
    f"Supported rate: "
    f"{report.supported_rate:.3f}"
)
print(
    f"Non-unsupported rate: "
    f"{report.non_unsupported_rate:.3f}"
)
