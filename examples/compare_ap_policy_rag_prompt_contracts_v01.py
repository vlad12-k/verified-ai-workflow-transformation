"""Compare baseline-v1 and strict-v2 Qwen RAG prompt contracts."""

import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from vait.rag.faithfulness import (
    FaithfulnessEvaluationReport,
    evaluate_faithfulness_annotations,
    load_faithfulness_annotations,
)

_ARTIFACT_ADAPTER = TypeAdapter(
    dict[str, Any]
)

BASELINE_ARTIFACT = Path(
    "artifacts/benchmarks/ap-policy-rag-qwen-v0.1.json"
)

STRICT_ARTIFACT = Path(
    "artifacts/benchmarks/ap-policy-rag-qwen-strict-v0.2.json"
)

BASELINE_ANNOTATIONS = Path(
    "datasets/ap_policy_rag/v0.1/"
    "qwen_faithfulness_annotations.json"
)

STRICT_ANNOTATIONS = Path(
    "datasets/ap_policy_rag/v0.1/"
    "qwen_strict_v02_faithfulness_annotations.json"
)


def load_artifact(
    path: Path,
) -> dict[str, Any]:
    """Load one experiment artifact."""
    decoded: object = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    return _ARTIFACT_ADAPTER.validate_python(
        decoded
    )


def answers_by_case_id(
    artifact: dict[str, Any],
) -> dict[str, str]:
    """Map answerable case IDs to exact generated answers."""
    answers: dict[str, str] = {}

    for result, trace in zip(
        artifact["report"]["results"],
        artifact["traces"],
        strict=True,
    ):
        if trace["abstained"]:
            continue

        answers[result["case_id"]] = trace["answer"]

    return answers


def validate_controls(
    baseline: dict[str, Any],
    strict: dict[str, Any],
) -> None:
    """Require the experiments to differ only by prompt contract."""
    invariant_fields = (
        "embedding_model_id",
        "embedding_revision",
        "generator_model_id",
        "generator_revision",
    )

    for field in invariant_fields:
        if baseline[field] != strict[field]:
            raise ValueError(
                f"Experiment control changed: {field}."
            )

    baseline_config = dict(
        baseline["configuration"]
    )
    strict_config = dict(
        strict["configuration"]
    )

    baseline_prompt = baseline_config.pop(
        "prompt_contract"
    )
    strict_prompt = strict_config.pop(
        "prompt_contract"
    )

    if baseline_prompt != "baseline-v1":
        raise ValueError(
            "Baseline experiment must use baseline-v1."
        )

    if strict_prompt != "strict-v2":
        raise ValueError(
            "Candidate experiment must use strict-v2."
        )

    if baseline_config != strict_config:
        raise ValueError(
            "Experiment configuration changed beyond "
            "the prompt contract."
        )

    structural_metrics = (
        "evidence_coverage",
        "citation_accuracy",
        "abstention_accuracy",
        "answerable_success_rate",
        "overall_success_rate",
    )

    for metric in structural_metrics:
        baseline_value = baseline["report"][metric]
        strict_value = strict["report"][metric]

        if baseline_value != strict_value:
            raise ValueError(
                f"Structural metric changed: {metric}."
            )


def evaluate_artifact(
    artifact: dict[str, Any],
    annotation_path: Path,
) -> FaithfulnessEvaluationReport:
    """Validate exact outputs against reviewed annotations."""
    annotations = load_faithfulness_annotations(
        annotation_path
    )

    return evaluate_faithfulness_annotations(
        annotations,
        answers_by_case_id=answers_by_case_id(
            artifact
        ),
    )


baseline = load_artifact(
    BASELINE_ARTIFACT
)

strict = load_artifact(
    STRICT_ARTIFACT
)

validate_controls(
    baseline,
    strict,
)

baseline_report = evaluate_artifact(
    baseline,
    BASELINE_ANNOTATIONS,
)

strict_report = evaluate_artifact(
    strict,
    STRICT_ANNOTATIONS,
)

baseline_unsupported_rate = (
    baseline_report.unsupported_count
    / baseline_report.reviewed_count
)

strict_unsupported_rate = (
    strict_report.unsupported_count
    / strict_report.reviewed_count
)

unsupported_delta = (
    strict_unsupported_rate
    - baseline_unsupported_rate
)

print("Controlled prompt-contract comparison")
print()
print(
    f"{'Metric':<24}"
    f"{'baseline-v1':>14}"
    f"{'strict-v2':>14}"
)
print("-" * 52)

print(
    f"{'SUPPORTED':<24}"
    f"{baseline_report.supported_count:>14}"
    f"{strict_report.supported_count:>14}"
)

print(
    f"{'PARTIAL':<24}"
    f"{baseline_report.partial_count:>14}"
    f"{strict_report.partial_count:>14}"
)

print(
    f"{'UNSUPPORTED':<24}"
    f"{baseline_report.unsupported_count:>14}"
    f"{strict_report.unsupported_count:>14}"
)

print(
    f"{'Supported rate':<24}"
    f"{baseline_report.supported_rate:>14.3f}"
    f"{strict_report.supported_rate:>14.3f}"
)

print(
    f"{'Non-unsupported rate':<24}"
    f"{baseline_report.non_unsupported_rate:>14.3f}"
    f"{strict_report.non_unsupported_rate:>14.3f}"
)

print(
    f"{'Unsupported rate':<24}"
    f"{baseline_unsupported_rate:>14.3f}"
    f"{strict_unsupported_rate:>14.3f}"
)

print()
print(
    "Unsupported-rate delta:",
    f"{unsupported_delta:+.3f}",
)

output_path = Path(
    "artifacts/benchmarks/"
    "ap-policy-rag-prompt-contract-comparison-v0.1.json"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

output_path.write_text(
    json.dumps(
        {
            "comparison_id": (
                "ap-policy-rag-prompt-contract-comparison-v0.1"
            ),
            "controlled_variable": "prompt_contract",
            "baseline_experiment_id": (
                baseline_report.experiment_id
            ),
            "candidate_experiment_id": (
                strict_report.experiment_id
            ),
            "baseline": baseline_report.model_dump(
                mode="json"
            ),
            "candidate": strict_report.model_dump(
                mode="json"
            ),
            "unsupported_rate": {
                "baseline": baseline_unsupported_rate,
                "candidate": strict_unsupported_rate,
                "delta": unsupported_delta,
            },
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print()
print(f"Comparison report: {output_path}")
