"""Build unified heterogeneous M3 development evidence."""

import json
from pathlib import Path
from typing import Any

ARTIFACT_ROOT = Path(
    "artifacts/benchmarks"
)

OUTPUT_PATH = (
    ARTIFACT_ROOT
    / "m3-heterogeneous-evidence-v0.1.json"
)

SOURCE_ARTIFACTS = (
    "ap-v0.2-transformation-assessment.json",
    "ap-v0.2-ml-transformation-assessment.json",
    "ap-v0.2-pytorch-transformation-assessment.json",
    "ap-distillation-training-v0.1.json",
    "ap-distillation-verification-v0.1.json",
    "ap-distillation-comparison-v0.1.json",
    "ap-keras-verification-v0.1.json",
    "ap-pytorch-vs-keras-v0.1.json",
    "ap-policy-context-v0.1-retrieval.json",
    "ap-policy-rag-faithfulness-progression-v0.1.json",
    "ap-policy-rag-qwen-strict-guarded-v0.3.json",
    "ap-policy-rag-qwen-lora-comparison-v0.1.json",
    "ap-policy-specialist-lora-training-v0.1.json",
    "ap-policy-specialist-tokenization-v0.1.json",
    "ap-policy-specialist-qlora-feasibility-v0.1.json",
    "ap-synthetic-training-corpus-v0.1.json",
)


def load_artifact(
    name: str,
) -> dict[str, Any]:
    """Load one required JSON evidence artifact."""
    path = ARTIFACT_ROOT / name

    if not path.exists():
        raise FileNotFoundError(
            f"Required M3 evidence artifact is missing: {path}"
        )

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        data,
        dict,
    ):
        raise RuntimeError(
            f"Expected JSON object in {path}"
        )

    return data


def summarize_verification(
    verification: dict[str, Any],
) -> dict[str, Any]:
    """Extract comparable bounded-verification fields."""
    summary: dict[str, Any] = {
        "decision": verification.get(
            "decision"
        ),
        "cases_evaluated": verification.get(
            "cases_evaluated"
        ),
    }

    statistical = verification.get(
        "statistical_evidence"
    )

    if isinstance(
        statistical,
        dict,
    ):
        summary["statistical_evidence"] = {
            "disagreements": statistical.get(
                "disagreements"
            ),
            "disagreement_rate": statistical.get(
                "disagreement_rate"
            ),
            "disagreement_upper_bound": statistical.get(
                "disagreement_upper_bound"
            ),
            "high_risk_cases_evaluated": statistical.get(
                "high_risk_cases_evaluated"
            ),
            "high_risk_disagreements": statistical.get(
                "high_risk_disagreements"
            ),
            "high_risk_disagreement_rate": statistical.get(
                "high_risk_disagreement_rate"
            ),
            "high_risk_disagreement_upper_bound": (
                statistical.get(
                    "high_risk_disagreement_upper_bound"
                )
            ),
        }

    latency = verification.get(
        "candidate_latency"
    )

    if isinstance(
        latency,
        dict,
    ):
        summary["latency_ms"] = {
            "mean": latency.get(
                "mean_ms"
            ),
            "p50": latency.get(
                "p50_ms"
            ),
            "p95": latency.get(
                "p95_ms"
            ),
            "max": latency.get(
                "max_ms"
            ),
        }

    failures = verification.get(
        "failures"
    )

    if isinstance(
        failures,
        list,
    ):
        summary["failure_count"] = len(
            failures
        )

    return summary


def summarize_assessment_report(
    report: dict[str, Any],
) -> dict[str, Any]:
    """Summarize a transformation assessment report."""
    assessments = report.get(
        "assessments"
    )

    if not isinstance(
        assessments,
        list,
    ):
        raise RuntimeError(
            "Transformation assessment report "
            "does not contain an assessments list."
        )

    summarized = []

    for assessment in assessments:
        if not isinstance(
            assessment,
            dict,
        ):
            raise RuntimeError(
                "Transformation assessment entry "
                "is not a JSON object."
            )

        applicability = assessment.get(
            "applicability"
        )

        applicability_status = None

        if isinstance(
            applicability,
            dict,
        ):
            applicability_status = (
                applicability.get(
                    "status"
                )
            )

        item: dict[str, Any] = {
            "canonical_id": assessment.get(
                "canonical_id"
            ),
            "candidate_implementation_id": (
                assessment.get(
                    "candidate_implementation_id"
                )
            ),
            "applicability": (
                applicability_status
            ),
        }

        verification = assessment.get(
            "verification"
        )

        if isinstance(
            verification,
            dict,
        ):
            item["verification"] = (
                summarize_verification(
                    verification
                )
            )
        else:
            item["verification"] = None

        summarized.append(
            item
        )

    return {
        "benchmark_id": report.get(
            "benchmark_id"
        ),
        "benchmark_version": report.get(
            "benchmark_version"
        ),
        "assessments": summarized,
    }


rules = load_artifact(
    "ap-v0.2-transformation-assessment.json"
)

classical_ml = load_artifact(
    "ap-v0.2-ml-transformation-assessment.json"
)

pytorch = load_artifact(
    "ap-v0.2-pytorch-transformation-assessment.json"
)

distillation_training = load_artifact(
    "ap-distillation-training-v0.1.json"
)

distillation_verification = load_artifact(
    "ap-distillation-verification-v0.1.json"
)

distillation_comparison = load_artifact(
    "ap-distillation-comparison-v0.1.json"
)

keras_verification = load_artifact(
    "ap-keras-verification-v0.1.json"
)

framework_comparison = load_artifact(
    "ap-pytorch-vs-keras-v0.1.json"
)

retrieval = load_artifact(
    "ap-policy-context-v0.1-retrieval.json"
)

rag_progression = load_artifact(
    "ap-policy-rag-faithfulness-progression-v0.1.json"
)

guarded_rag = load_artifact(
    "ap-policy-rag-qwen-strict-guarded-v0.3.json"
)

lora_comparison = load_artifact(
    "ap-policy-rag-qwen-lora-comparison-v0.1.json"
)

lora_training = load_artifact(
    "ap-policy-specialist-lora-training-v0.1.json"
)

specialist_tokenization = load_artifact(
    "ap-policy-specialist-tokenization-v0.1.json"
)

qlora = load_artifact(
    "ap-policy-specialist-qlora-feasibility-v0.1.json"
)

synthetic_data = load_artifact(
    "ap-synthetic-training-corpus-v0.1.json"
)

guarded_report = guarded_rag.get(
    "report"
)

if not isinstance(
    guarded_report,
    dict,
):
    raise RuntimeError(
        "Guarded RAG artifact is missing its report."
    )

guarded_configuration = guarded_rag.get(
    "configuration"
)

if not isinstance(
    guarded_configuration,
    dict,
):
    raise RuntimeError(
        "Guarded RAG artifact is missing configuration."
    )

report = {
    "artifact_id": (
        "m3-heterogeneous-evidence-v0.1"
    ),
    "development_evidence": True,
    "milestone": "M3 AI breadth",
    "evidence_model": {
        "cross_task_ranking": False,
        "candidate_selection": False,
        "principle": (
            "Evidence is grouped by task family and "
            "verification semantics. Metrics from incompatible "
            "task families are not combined into an overall score."
        ),
    },
    "tracks": {
        "structured_decision": {
            "deterministic_transformations": (
                summarize_assessment_report(
                    rules
                )
            ),
            "classical_ml": (
                summarize_assessment_report(
                    classical_ml
                )
            ),
            "pytorch": (
                summarize_assessment_report(
                    pytorch
                )
            ),
            "distillation": {
                "training": {
                    "corpus": (
                        distillation_training[
                            "corpus"
                        ]
                    ),
                    "teacher": (
                        distillation_training[
                            "teacher"
                        ]
                    ),
                    "hard_label_student": (
                        distillation_training[
                            "hard_label_student"
                        ]
                    ),
                    "distilled_student": (
                        distillation_training[
                            "distilled_student"
                        ]
                    ),
                    "parameter_reduction": (
                        distillation_training[
                            "parameter_reduction"
                        ]
                    ),
                },
                "verification": (
                    summarize_assessment_report(
                        distillation_verification
                    )
                ),
                "observations": (
                    distillation_comparison[
                        "observations"
                    ]
                ),
            },
            "keras": (
                summarize_assessment_report(
                    keras_verification
                )
            ),
            "pytorch_vs_keras": {
                "controlled_dimensions": (
                    framework_comparison[
                        "controlled_dimensions"
                    ]
                ),
                "pytorch": (
                    framework_comparison[
                        "pytorch"
                    ]
                ),
                "keras": (
                    framework_comparison[
                        "keras"
                    ]
                ),
                "observed_boundary_failure": (
                    framework_comparison[
                        "observed_boundary_failure"
                    ]
                ),
                "interpretation": (
                    framework_comparison[
                        "interpretation"
                    ]
                ),
            },
        },
        "retrieval_and_rag": {
            "retrieval": {
                "benchmark_id": retrieval[
                    "benchmark_id"
                ],
                "benchmark_version": retrieval[
                    "benchmark_version"
                ],
                "encoder_implementation_id": (
                    retrieval[
                        "encoder_implementation_id"
                    ]
                ),
                "query_count": retrieval[
                    "query_count"
                ],
                "recall_at_1": retrieval[
                    "recall_at_1"
                ],
                "recall_at_3": retrieval[
                    "recall_at_3"
                ],
                "mean_reciprocal_rank": (
                    retrieval[
                        "mean_reciprocal_rank"
                    ]
                ),
            },
            "faithfulness_progression": {
                "experiments": (
                    rag_progression[
                        "experiments"
                    ]
                ),
                "guarded_routing": (
                    rag_progression[
                        "guarded_routing"
                    ]
                ),
            },
            "guarded_local_slm": {
                "generator_model_id": (
                    guarded_rag[
                        "generator_model_id"
                    ]
                ),
                "generator_revision": (
                    guarded_rag[
                        "generator_revision"
                    ]
                ),
                "embedding_model_id": (
                    guarded_rag[
                        "embedding_model_id"
                    ]
                ),
                "embedding_revision": (
                    guarded_rag[
                        "embedding_revision"
                    ]
                ),
                "device": (
                    guarded_configuration.get(
                        "device"
                    )
                ),
                "prompt_contract": (
                    guarded_configuration.get(
                        "prompt_contract"
                    )
                ),
                "semantic_guard": (
                    guarded_configuration.get(
                        "semantic_guard"
                    )
                ),
                "routing": guarded_rag[
                    "routing"
                ],
                "case_count": guarded_report[
                    "case_count"
                ],
                "answerable_case_count": (
                    guarded_report[
                        "answerable_case_count"
                    ]
                ),
                "abstention_case_count": (
                    guarded_report[
                        "abstention_case_count"
                    ]
                ),
                "evidence_coverage": (
                    guarded_report[
                        "evidence_coverage"
                    ]
                ),
                "citation_accuracy": (
                    guarded_report[
                        "citation_accuracy"
                    ]
                ),
                "abstention_accuracy": (
                    guarded_report[
                        "abstention_accuracy"
                    ]
                ),
                "answerable_success_rate": (
                    guarded_report[
                        "answerable_success_rate"
                    ]
                ),
                "overall_success_rate": (
                    guarded_report[
                        "overall_success_rate"
                    ]
                ),
            },
        },
        "specialist_adaptation": {
            "tokenization": {
                "corpus_id": (
                    specialist_tokenization[
                        "corpus_id"
                    ]
                ),
                "corpus_version": (
                    specialist_tokenization[
                        "corpus_version"
                    ]
                ),
                "corpus_sha256": (
                    specialist_tokenization[
                        "corpus_sha256"
                    ]
                ),
                "example_count": (
                    specialist_tokenization[
                        "example_count"
                    ]
                ),
                "train_count": (
                    specialist_tokenization[
                        "train_count"
                    ]
                ),
                "dev_count": (
                    specialist_tokenization[
                        "dev_count"
                    ]
                ),
                "prompt_masking_valid": (
                    specialist_tokenization[
                        "prompt_masking_valid"
                    ]
                ),
                "assistant_supervision_valid": (
                    specialist_tokenization[
                        "assistant_supervision_valid"
                    ]
                ),
                "max_length_valid": (
                    specialist_tokenization[
                        "max_length_valid"
                    ]
                ),
            },
            "lora_training": {
                "model_id": lora_training[
                    "model_id"
                ],
                "model_revision": (
                    lora_training[
                        "model_revision"
                    ]
                ),
                "device": lora_training[
                    "device"
                ],
                "corpus": lora_training[
                    "corpus"
                ],
                "lora": lora_training[
                    "lora"
                ],
                "best_epoch": (
                    lora_training[
                        "training"
                    ][
                        "best_epoch"
                    ]
                ),
                "initial_dev_loss": (
                    lora_training[
                        "training"
                    ][
                        "initial_dev_loss"
                    ]
                ),
                "best_dev_loss": (
                    lora_training[
                        "training"
                    ][
                        "best_dev_loss"
                    ]
                ),
            },
            "lora_semantic_comparison": {
                "reviewed_cases": (
                    lora_comparison[
                        "reviewed_cases"
                    ]
                ),
                "baseline": (
                    lora_comparison[
                        "baseline"
                    ]
                ),
                "lora_specialist": (
                    lora_comparison[
                        "lora_specialist"
                    ]
                ),
                "delta": lora_comparison[
                    "delta"
                ],
                "verdict_transitions": (
                    lora_comparison[
                        "verdict_transitions"
                    ]
                ),
                "interpretation_scope": (
                    lora_comparison[
                        "interpretation_scope"
                    ]
                ),
            },
            "qlora_feasibility": {
                "model": qlora[
                    "model"
                ],
                "quantization": (
                    qlora[
                        "quantization"
                    ]
                ),
                "lora": qlora[
                    "lora"
                ],
                "gradient_evidence": (
                    qlora[
                        "gradient_evidence"
                    ]
                ),
                "checks": qlora[
                    "checks"
                ],
                "scope": qlora[
                    "scope"
                ],
            },
        },
        "synthetic_data": {
            "corpus": synthetic_data[
                "corpus"
            ],
            "coverage": synthetic_data[
                "coverage"
            ],
            "evaluation_exclusion": (
                synthetic_data[
                    "evaluation_exclusion"
                ]
            ),
            "generator": synthetic_data[
                "generator"
            ],
            "integrity": synthetic_data[
                "integrity"
            ],
        },
    },
    "scope_and_limitations": {
        "ap_benchmark": (
            "AP v0.2 is development evidence, not an "
            "independent blind benchmark."
        ),
        "synthetic_labels": (
            "Synthetic structured labels are policy-derived "
            "and are not independent ground truth."
        ),
        "rag_evaluation": (
            "Reviewed RAG faithfulness evidence is bounded "
            "to the declared synthetic development benchmark."
        ),
        "lora": (
            "LoRA semantic comparison is development evidence "
            "on a small specialist corpus and reviewed cases."
        ),
        "qlora": (
            "QLoRA evidence establishes local technical "
            "feasibility of quantized loading and gradient "
            "flow, not model quality or production readiness."
        ),
        "latency": (
            "Local latency measurements do not establish "
            "universal framework or model performance."
        ),
        "verification": (
            "BOUNDED denotes statistical evidence under the "
            "declared benchmark, risk strata, thresholds, and "
            "confidence level. It is not universal semantic "
            "equivalence."
        ),
    },
    "source_artifacts": list(
        SOURCE_ARTIFACTS
    ),
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

print("M3 heterogeneous evidence")
print()
print("Artifact:", report["artifact_id"])
print(
    "Tracks:",
    len(report["tracks"]),
)
print(
    "Source artifacts:",
    len(report["source_artifacts"]),
)

structured = report[
    "tracks"
]["structured_decision"]

print()
print("Structured decision assessments:")
print(
    "  deterministic:",
    len(
        structured[
            "deterministic_transformations"
        ][
            "assessments"
        ]
    ),
)
print(
    "  classical ML:",
    len(
        structured[
            "classical_ml"
        ][
            "assessments"
        ]
    ),
)
print(
    "  PyTorch:",
    len(
        structured[
            "pytorch"
        ][
            "assessments"
        ]
    ),
)
print(
    "  distillation:",
    len(
        structured[
            "distillation"
        ][
            "verification"
        ][
            "assessments"
        ]
    ),
)
print(
    "  Keras:",
    len(
        structured[
            "keras"
        ][
            "assessments"
        ]
    ),
)

print()
print(
    "Cross-task ranking:",
    report[
        "evidence_model"
    ][
        "cross_task_ranking"
    ],
)
print(
    "Candidate selection:",
    report[
        "evidence_model"
    ][
        "candidate_selection"
    ],
)
print()
print(
    "Evidence report:",
    OUTPUT_PATH,
)
