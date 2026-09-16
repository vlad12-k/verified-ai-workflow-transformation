"""Evaluate approved faithfulness annotations for Qwen RAG v0.1."""

import json
from pathlib import Path

from vait.rag.faithfulness import (
    evaluate_faithfulness_annotations,
    load_faithfulness_annotations,
)

artifact_path = Path(
    "artifacts/benchmarks/"
    "ap-policy-rag-qwen-v0.1.json"
)

annotation_path = Path(
    "datasets/ap_policy_rag/v0.1/"
    "qwen_faithfulness_annotations.json"
)

artifact = json.loads(
    artifact_path.read_text(
        encoding="utf-8"
    )
)

annotations = load_faithfulness_annotations(
    annotation_path
)

answerable_traces = [
    trace
    for trace in artifact["traces"]
    if not trace["abstained"]
]

answers_by_case_id = {}

report_results = artifact["report"]["results"]

for result, trace in zip(
    report_results,
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
