"""Compare base strict-v2 Qwen with the LoRA AP policy specialist."""

import json
from pathlib import Path

from vait.rag.faithfulness import (
    evaluate_faithfulness_annotations,
    load_faithfulness_annotations,
)

BASE_EXPERIMENT_PATH = Path(
    "artifacts/benchmarks/"
    "ap-policy-rag-qwen-strict-v0.2.json"
)

LORA_EXPERIMENT_PATH = Path(
    "artifacts/benchmarks/"
    "ap-policy-rag-qwen-lora-v0.1.json"
)

BASE_ANNOTATION_PATH = Path(
    "datasets/ap_policy_rag/v0.1/"
    "qwen_strict_v02_faithfulness_annotations.json"
)

LORA_ANNOTATION_PATH = Path(
    "datasets/ap_policy_rag/v0.1/"
    "qwen_lora_v01_faithfulness_annotations.json"
)

OUTPUT_PATH = Path(
    "artifacts/benchmarks/"
    "ap-policy-rag-qwen-lora-comparison-v0.1.json"
)


def load_answers(
    path: Path,
) -> dict[str, str]:
    """Load generated answers keyed by benchmark case ID."""
    experiment = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    return {
        result["case_id"]: trace["answer"]
        for result, trace in zip(
            experiment["report"]["results"],
            experiment["traces"],
            strict=True,
        )
    }


base_annotations = load_faithfulness_annotations(
    BASE_ANNOTATION_PATH
)

lora_annotations = load_faithfulness_annotations(
    LORA_ANNOTATION_PATH
)

base_report = evaluate_faithfulness_annotations(
    base_annotations,
    answers_by_case_id=load_answers(
        BASE_EXPERIMENT_PATH
    ),
)

lora_report = evaluate_faithfulness_annotations(
    lora_annotations,
    answers_by_case_id=load_answers(
        LORA_EXPERIMENT_PATH
    ),
)

base_by_case = {
    annotation.case_id: annotation
    for annotation in base_annotations.annotations
}

lora_by_case = {
    annotation.case_id: annotation
    for annotation in lora_annotations.annotations
}

if set(base_by_case) != set(lora_by_case):
    raise RuntimeError(
        "Baseline and LoRA reviews must cover the same cases."
    )

transitions: dict[str, int] = {}

case_comparisons = []

for case_id in sorted(base_by_case):
    base = base_by_case[case_id]
    lora = lora_by_case[case_id]

    transition = (
        f"{base.verdict.value}"
        f"->{lora.verdict.value}"
    )

    transitions[transition] = (
        transitions.get(
            transition,
            0,
        )
        + 1
    )

    case_comparisons.append(
        {
            "case_id": case_id,
            "base_verdict": base.verdict.value,
            "lora_verdict": lora.verdict.value,
            "transition": transition,
        }
    )

supported_delta = (
    lora_report.supported_rate
    - base_report.supported_rate
)

non_unsupported_delta = (
    lora_report.non_unsupported_rate
    - base_report.non_unsupported_rate
)

unsupported_delta = (
    lora_report.unsupported_count
    - base_report.unsupported_count
)

comparison = {
    "comparison_id": (
        "ap-policy-rag-qwen-lora-comparison-v0.1"
    ),
    "development_evidence": True,
    "benchmark_id": (
        base_annotations.benchmark_id
    ),
    "benchmark_version": (
        base_annotations.benchmark_version
    ),
    "reviewed_cases": (
        base_report.reviewed_count
    ),
    "baseline": (
        base_report.model_dump(
            mode="json"
        )
    ),
    "lora_specialist": (
        lora_report.model_dump(
            mode="json"
        )
    ),
    "delta": {
        "supported_rate": (
            supported_delta
        ),
        "non_unsupported_rate": (
            non_unsupported_delta
        ),
        "unsupported_count": (
            unsupported_delta
        ),
    },
    "verdict_transitions": (
        transitions
    ),
    "cases": case_comparisons,
    "interpretation_scope": (
        "Development evidence only. "
        "The same reviewed synthetic benchmark was used "
        "during prior prompt and guard development, so these "
        "results do not establish unseen generalisation."
    ),
}

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH.write_text(
    json.dumps(
        comparison,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

print("Base Qwen vs LoRA specialist")
print()
print(
    "Base:",
    f"{base_report.supported_count} SUPPORTED,",
    f"{base_report.partial_count} PARTIAL,",
    f"{base_report.unsupported_count} UNSUPPORTED",
)
print(
    "LoRA:",
    f"{lora_report.supported_count} SUPPORTED,",
    f"{lora_report.partial_count} PARTIAL,",
    f"{lora_report.unsupported_count} UNSUPPORTED",
)
print()
print(
    "Supported-rate delta:",
    f"{supported_delta:+.3f}",
)
print(
    "Non-unsupported-rate delta:",
    f"{non_unsupported_delta:+.3f}",
)
print(
    "Unsupported-count delta:",
    f"{unsupported_delta:+d}",
)
print()
print("Verdict transitions:")

for transition, count in sorted(
    transitions.items()
):
    print(
        f"{transition}: {count}"
    )

print()
print(
    f"Evidence report: {OUTPUT_PATH}"
)
