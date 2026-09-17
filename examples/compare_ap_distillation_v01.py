"""Compare AP teacher, hard-label student, and distilled student evidence."""

import json
from pathlib import Path

TRAINING_PATH = Path(
    "artifacts/benchmarks/"
    "ap-distillation-training-v0.1.json"
)

VERIFICATION_PATH = Path(
    "artifacts/benchmarks/"
    "ap-distillation-verification-v0.1.json"
)

OUTPUT_PATH = Path(
    "artifacts/benchmarks/"
    "ap-distillation-comparison-v0.1.json"
)

TEACHER_ID = "synthetic-ap-pytorch-mlp-v1"

HARD_STUDENT_ID = (
    "synthetic-ap-pytorch-student-hard-label-v1"
)

DISTILLED_STUDENT_ID = (
    "synthetic-ap-pytorch-student-distilled-v1"
)


training = json.loads(
    TRAINING_PATH.read_text(
        encoding="utf-8"
    )
)

verification = json.loads(
    VERIFICATION_PATH.read_text(
        encoding="utf-8"
    )
)

verification_by_id = {
    assessment["candidate_implementation_id"]: assessment
    for assessment in verification["assessments"]
}


def verification_payload(
    candidate_id: str,
) -> dict[str, object]:
    """Return one verified candidate payload."""
    assessment = verification_by_id[
        candidate_id
    ]

    payload = assessment["verification"]

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            f"Missing verification payload for {candidate_id}."
        )

    return payload


teacher_verification = verification_payload(
    TEACHER_ID
)

hard_verification = verification_payload(
    HARD_STUDENT_ID
)

distilled_verification = verification_payload(
    DISTILLED_STUDENT_ID
)


def evidence_summary(
    payload: dict[str, object],
) -> dict[str, object]:
    """Extract bounded verification and latency evidence."""
    evidence = payload.get(
        "statistical_evidence"
    )

    latency = payload.get(
        "candidate_latency"
    )

    if not isinstance(
        evidence,
        dict,
    ):
        raise RuntimeError(
            "Missing statistical evidence."
        )

    if not isinstance(
        latency,
        dict,
    ):
        raise RuntimeError(
            "Missing candidate latency evidence."
        )

    return {
        "decision": payload["decision"],
        "cases_evaluated": payload[
            "cases_evaluated"
        ],
        "disagreements": evidence[
            "disagreements"
        ],
        "disagreement_rate": evidence[
            "disagreement_rate"
        ],
        "disagreement_upper_bound": evidence[
            "disagreement_upper_bound"
        ],
        "high_risk_cases_evaluated": evidence[
            "high_risk_cases_evaluated"
        ],
        "high_risk_disagreements": evidence[
            "high_risk_disagreements"
        ],
        "high_risk_disagreement_rate": evidence[
            "high_risk_disagreement_rate"
        ],
        "high_risk_disagreement_upper_bound": evidence[
            "high_risk_disagreement_upper_bound"
        ],
        "latency": {
            "mean_ms": latency[
                "mean_ms"
            ],
            "p50_ms": latency[
                "p50_ms"
            ],
            "p95_ms": latency[
                "p95_ms"
            ],
            "max_ms": latency[
                "max_ms"
            ],
        },
        "failures": len(
            payload["failures"]
        ),
    }


teacher_summary = evidence_summary(
    teacher_verification
)

hard_summary = evidence_summary(
    hard_verification
)

distilled_summary = evidence_summary(
    distilled_verification
)

comparison = {
    "comparison_id": (
        "ap-distillation-comparison-v0.1"
    ),
    "development_evidence": True,
    "corpus": training[
        "corpus"
    ],
    "training_configuration": training[
        "training"
    ],
    "parameter_reduction": training[
        "parameter_reduction"
    ],
    "models": {
        "teacher": {
            **training["teacher"],
            "verification": teacher_summary,
        },
        "hard_label_student": {
            **training["hard_label_student"],
            "verification": hard_summary,
        },
        "distilled_student": {
            **training["distilled_student"],
            "verification": distilled_summary,
        },
    },
    "observations": {
        "teacher_to_student_compression": (
            "Both compact students use substantially fewer "
            "trainable parameters than the teacher while "
            "preserving the same observed AP v0.2 benchmark "
            "decision under the bounded policy."
        ),
        "distillation_vs_hard_labels": (
            "The distilled and hard-label students achieved "
            "the same observed training accuracy, teacher "
            "agreement, benchmark disagreement count, and "
            "bounded verification decision in this development "
            "experiment."
        ),
        "latency_scope": (
            "Latency values are local single-run development "
            "measurements and do not establish universal "
            "performance superiority."
        ),
        "generality_scope": training[
            "scope"
        ],
    },
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

print("AP distillation controlled comparison")
print()
print(
    "Corpus cases:",
    training["corpus"]["cases"],
)
print(
    "Parameter reduction:",
    f"{training['parameter_reduction']:.2%}",
)
print()

print(
    "Teacher accuracy:",
    f"{training['teacher']['training_accuracy']:.4f}",
)
print(
    "Hard student accuracy:",
    f"{training['hard_label_student']['training_accuracy']:.4f}",
)
print(
    "Distilled student accuracy:",
    f"{training['distilled_student']['training_accuracy']:.4f}",
)
print()

print(
    "Hard student teacher agreement:",
    f"{training['hard_label_student']['teacher_agreement']:.4f}",
)
print(
    "Distilled student teacher agreement:",
    f"{training['distilled_student']['teacher_agreement']:.4f}",
)
print()

for label, summary in (
    ("Teacher", teacher_summary),
    ("Hard student", hard_summary),
    ("Distilled student", distilled_summary),
):
    print(label)
    print(
        "  decision:",
        summary["decision"],
    )
    print(
        "  disagreements:",
        summary["disagreements"],
    )
    print(
        "  high-risk disagreements:",
        summary[
            "high_risk_disagreements"
        ],
    )
    print(
        "  p50 latency:",
        (
            f"{summary['latency']['p50_ms']:.4f} ms"
        ),
    )
    print(
        "  p95 latency:",
        (
            f"{summary['latency']['p95_ms']:.4f} ms"
        ),
    )
    print()

print(
    "Evidence report:",
    OUTPUT_PATH,
)
