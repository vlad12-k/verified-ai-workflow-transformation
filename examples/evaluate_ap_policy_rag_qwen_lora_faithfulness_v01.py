"""Evaluate hash-bound faithfulness annotations for the LoRA specialist."""

import json
from pathlib import Path

from vait.rag.faithfulness import (
    evaluate_faithfulness_annotations,
    load_faithfulness_annotations,
)

EXPERIMENT_PATH = Path(
    "artifacts/benchmarks/"
    "ap-policy-rag-qwen-lora-v0.1.json"
)

ANNOTATION_PATH = Path(
    "datasets/ap_policy_rag/v0.1/"
    "qwen_lora_v01_faithfulness_annotations.json"
)

experiment = json.loads(
    EXPERIMENT_PATH.read_text(
        encoding="utf-8"
    )
)

answers_by_case_id = {
    result["case_id"]: trace["answer"]
    for result, trace in zip(
        experiment["report"]["results"],
        experiment["traces"],
        strict=True,
    )
}

annotations = load_faithfulness_annotations(
    ANNOTATION_PATH
)

report = evaluate_faithfulness_annotations(
    annotations,
    answers_by_case_id=answers_by_case_id,
)

print("LoRA specialist faithfulness")
print()
print(f"Experiment: {report.experiment_id}")
print(f"Reviewed: {report.reviewed_count}")
print(f"SUPPORTED: {report.supported_count}")
print(f"PARTIAL: {report.partial_count}")
print(f"UNSUPPORTED: {report.unsupported_count}")
print(
    f"Supported rate: "
    f"{report.supported_rate:.3f}"
)
print(
    f"Non-unsupported rate: "
    f"{report.non_unsupported_rate:.3f}"
)
print()
print("Answer hashes: PASS")
