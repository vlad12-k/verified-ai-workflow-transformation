"""Compare baseline, strict, and guarded RAG faithfulness evidence."""

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
GUARDED_ARTIFACT = Path(
    "artifacts/benchmarks/"
    "ap-policy-rag-qwen-strict-guarded-v0.3.json"
)

BASELINE_ANNOTATIONS = Path(
    "datasets/ap_policy_rag/v0.1/"
    "qwen_faithfulness_annotations.json"
)
STRICT_ANNOTATIONS = Path(
    "datasets/ap_policy_rag/v0.1/"
    "qwen_strict_v02_faithfulness_annotations.json"
)
GUARDED_ANNOTATIONS = Path(
    "datasets/ap_policy_rag/v0.1/"
    "qwen_guarded_v03_faithfulness_annotations.json"
)

COMMON_CONFIGURATION_KEYS = (
    "device",
    "minimum_score",
    "retrieval_k",
    "max_context_documents",
    "max_context_chars",
    "max_new_tokens",
    "do_sample",
    "local_files_only",
)


def load_artifact(
    path: Path,
) -> dict[str, Any]:
    """Load one versioned local experiment artifact."""
    if not path.exists():
        raise FileNotFoundError(
            f"Experiment artifact not found: {path}"
        )

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
    """Map answerable benchmark cases to surfaced answers."""
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


def evaluate_artifact(
    artifact: dict[str, Any],
    annotation_path: Path,
) -> FaithfulnessEvaluationReport:
    """Evaluate exact surfaced answers against reviewed annotations."""
    annotations = load_faithfulness_annotations(
        annotation_path
    )

    return evaluate_faithfulness_annotations(
        annotations,
        answers_by_case_id=answers_by_case_id(
            artifact
        ),
    )


def validate_experiment_controls(
    baseline: dict[str, Any],
    strict: dict[str, Any],
    guarded: dict[str, Any],
) -> None:
    """Validate shared model and retrieval controls."""
    invariant_fields = (
        "embedding_model_id",
        "embedding_revision",
        "generator_model_id",
        "generator_revision",
    )

    for field in invariant_fields:
        values = {
            baseline[field],
            strict[field],
            guarded[field],
        }

        if len(values) != 1:
            raise ValueError(
                f"Experiment identity changed: {field}."
            )

    for key in COMMON_CONFIGURATION_KEYS:
        values = {
            baseline["configuration"][key],
            strict["configuration"][key],
            guarded["configuration"][key],
        }

        if len(values) != 1:
            raise ValueError(
                f"Experiment control changed: {key}."
            )

    if (
        baseline["configuration"]["prompt_contract"]
        != "baseline-v1"
    ):
        raise ValueError(
            "Baseline experiment must use baseline-v1."
        )

    if (
        strict["configuration"]["prompt_contract"]
        != "strict-v2"
    ):
        raise ValueError(
            "Strict experiment must use strict-v2."
        )

    if (
        guarded["configuration"]["prompt_contract"]
        != "strict-v2"
    ):
        raise ValueError(
            "Guarded experiment must use strict-v2."
        )

    if not guarded["configuration"].get(
        "semantic_guard",
        False,
    ):
        raise ValueError(
            "Guarded experiment must enable semantic_guard."
        )

    structural_metrics = (
        "evidence_coverage",
        "citation_accuracy",
        "abstention_accuracy",
        "answerable_success_rate",
        "overall_success_rate",
    )

    for metric in structural_metrics:
        values = {
            baseline["report"][metric],
            strict["report"][metric],
            guarded["report"][metric],
        }

        if len(values) != 1:
            raise ValueError(
                f"Structural metric changed: {metric}."
            )


baseline = load_artifact(BASELINE_ARTIFACT)
strict = load_artifact(STRICT_ARTIFACT)
guarded = load_artifact(GUARDED_ARTIFACT)

validate_experiment_controls(
    baseline,
    strict,
    guarded,
)

baseline_report = evaluate_artifact(
    baseline,
    BASELINE_ANNOTATIONS,
)
strict_report = evaluate_artifact(
    strict,
    STRICT_ANNOTATIONS,
)
guarded_report = evaluate_artifact(
    guarded,
    GUARDED_ANNOTATIONS,
)

reports = (
    ("baseline-v1", baseline_report),
    ("strict-v2", strict_report),
    ("strict-v2+guard", guarded_report),
)

print("RAG faithfulness progression")
print()
print(
    f"{'Metric':<24}"
    f"{'baseline-v1':>14}"
    f"{'strict-v2':>14}"
    f"{'guarded':>14}"
)
print("-" * 66)

print(
    f"{'SUPPORTED':<24}"
    f"{baseline_report.supported_count:>14}"
    f"{strict_report.supported_count:>14}"
    f"{guarded_report.supported_count:>14}"
)

print(
    f"{'PARTIAL':<24}"
    f"{baseline_report.partial_count:>14}"
    f"{strict_report.partial_count:>14}"
    f"{guarded_report.partial_count:>14}"
)

print(
    f"{'UNSUPPORTED':<24}"
    f"{baseline_report.unsupported_count:>14}"
    f"{strict_report.unsupported_count:>14}"
    f"{guarded_report.unsupported_count:>14}"
)

print(
    f"{'Supported rate':<24}"
    f"{baseline_report.supported_rate:>14.3f}"
    f"{strict_report.supported_rate:>14.3f}"
    f"{guarded_report.supported_rate:>14.3f}"
)

print(
    f"{'Non-unsupported rate':<24}"
    f"{baseline_report.non_unsupported_rate:>14.3f}"
    f"{strict_report.non_unsupported_rate:>14.3f}"
    f"{guarded_report.non_unsupported_rate:>14.3f}"
)

routing = guarded["routing"]

print()
print("Guarded routing")
print(
    "Answerable surfaced outputs:",
    routing["answerable_surfaced_outputs"],
)
print(
    "Primary surfaced:",
    routing["primary_surfaced"],
)
print(
    "Fallback surfaced:",
    routing["fallback_surfaced"],
)

output_path = Path(
    "artifacts/benchmarks/"
    "ap-policy-rag-faithfulness-progression-v0.1.json"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

output_path.write_text(
    json.dumps(
        {
            "comparison_id": (
                "ap-policy-rag-faithfulness-progression-v0.1"
            ),
            "development_evidence": True,
            "experiments": {
                name: report.model_dump(mode="json")
                for name, report in reports
            },
            "guarded_routing": routing,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print()
print(f"Comparison report: {output_path}")
