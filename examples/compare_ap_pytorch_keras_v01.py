"""Compare PyTorch and Keras AP candidates under the same development setup."""

import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

_JSON_OBJECT_ADAPTER = TypeAdapter(
    dict[str, Any]
)

PYTORCH_TRAINING_PATH = Path(
    "artifacts/benchmarks/"
    "ap-distillation-training-v0.1.json"
)

PYTORCH_VERIFICATION_PATH = Path(
    "artifacts/benchmarks/"
    "ap-distillation-verification-v0.1.json"
)

KERAS_TRAINING_PATH = Path(
    "artifacts/benchmarks/"
    "ap-keras-training-v0.1.json"
)

KERAS_VERIFICATION_PATH = Path(
    "artifacts/benchmarks/"
    "ap-keras-verification-v0.1.json"
)

OUTPUT_PATH = Path(
    "artifacts/benchmarks/"
    "ap-pytorch-vs-keras-v0.1.json"
)

PYTORCH_ID = "synthetic-ap-pytorch-mlp-v1"
KERAS_ID = "synthetic-ap-keras-mlp-v1"


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON evidence artifact."""
    decoded: object = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    return _JSON_OBJECT_ADAPTER.validate_python(
        decoded
    )


def find_verification(
    report: dict[str, Any],
    candidate_id: str,
) -> dict[str, Any]:
    """Return verification payload for one candidate."""
    assessments = report["assessments"]

    if not isinstance(
        assessments,
        list,
    ):
        raise RuntimeError(
            "Verification report assessments are invalid."
        )

    for assessment in assessments:
        if not isinstance(
            assessment,
            dict,
        ):
            continue

        if (
            assessment.get(
                "candidate_implementation_id"
            )
            != candidate_id
        ):
            continue

        payload = assessment.get(
            "verification"
        )

        if not isinstance(
            payload,
            dict,
        ):
            raise RuntimeError(
                f"Missing verification payload for {candidate_id}."
            )

        return _JSON_OBJECT_ADAPTER.validate_python(
            payload
        )

    raise RuntimeError(
        f"Candidate not found: {candidate_id}"
    )


pytorch_training = load_json(
    PYTORCH_TRAINING_PATH
)

pytorch_verification_report = load_json(
    PYTORCH_VERIFICATION_PATH
)

keras_training = load_json(
    KERAS_TRAINING_PATH
)

keras_verification_report = load_json(
    KERAS_VERIFICATION_PATH
)

pytorch_verification = find_verification(
    pytorch_verification_report,
    PYTORCH_ID,
)

keras_verification = find_verification(
    keras_verification_report,
    KERAS_ID,
)

pytorch_evidence = pytorch_verification[
    "statistical_evidence"
]

keras_evidence = keras_verification[
    "statistical_evidence"
]

pytorch_latency = pytorch_verification[
    "candidate_latency"
]

keras_latency = keras_verification[
    "candidate_latency"
]

if not isinstance(
    pytorch_evidence,
    dict,
):
    raise RuntimeError(
        "Missing PyTorch statistical evidence."
    )

if not isinstance(
    keras_evidence,
    dict,
):
    raise RuntimeError(
        "Missing Keras statistical evidence."
    )

if not isinstance(
    pytorch_latency,
    dict,
):
    raise RuntimeError(
        "Missing PyTorch latency evidence."
    )

if not isinstance(
    keras_latency,
    dict,
):
    raise RuntimeError(
        "Missing Keras latency evidence."
    )

comparison: dict[str, Any] = {
    "comparison_id": (
        "ap-pytorch-vs-keras-v0.1"
    ),
    "development_evidence": True,
    "controlled_dimensions": {
        "feature_count": 11,
        "architecture": [11, 32, 16, 3],
        "parameter_count": 963,
        "training_cases": 600,
        "benchmark": "ap-invoice-exceptions@0.2",
        "verification_policy": {
            "max_overall_disagreement_rate": 0.15,
            "max_high_risk_disagreement_rate": 0.25,
            "confidence_level": 0.95,
            "min_total_cases": 20,
            "min_high_risk_cases": 11,
        },
    },
    "pytorch": {
        "training_accuracy": (
            pytorch_training[
                "teacher"
            ][
                "training_accuracy"
            ]
        ),
        "final_training_loss": (
            pytorch_training[
                "teacher"
            ][
                "final_training_loss"
            ]
        ),
        "decision": pytorch_verification[
            "decision"
        ],
        "disagreements": pytorch_evidence[
            "disagreements"
        ],
        "high_risk_disagreements": (
            pytorch_evidence[
                "high_risk_disagreements"
            ]
        ),
        "overall_upper_bound": (
            pytorch_evidence[
                "disagreement_upper_bound"
            ]
        ),
        "high_risk_upper_bound": (
            pytorch_evidence[
                "high_risk_disagreement_upper_bound"
            ]
        ),
        "p50_latency_ms": pytorch_latency[
            "p50_ms"
        ],
        "p95_latency_ms": pytorch_latency[
            "p95_ms"
        ],
    },
    "keras": {
        "training_accuracy": (
            keras_training[
                "model"
            ][
                "training_accuracy"
            ]
        ),
        "final_training_loss": (
            keras_training[
                "model"
            ][
                "final_training_loss"
            ]
        ),
        "decision": keras_verification[
            "decision"
        ],
        "disagreements": keras_evidence[
            "disagreements"
        ],
        "high_risk_disagreements": (
            keras_evidence[
                "high_risk_disagreements"
            ]
        ),
        "overall_upper_bound": (
            keras_evidence[
                "disagreement_upper_bound"
            ]
        ),
        "high_risk_upper_bound": (
            keras_evidence[
                "high_risk_disagreement_upper_bound"
            ]
        ),
        "p50_latency_ms": keras_latency[
            "p50_ms"
        ],
        "p95_latency_ms": keras_latency[
            "p95_ms"
        ],
    },
    "observed_boundary_failure": {
        "case_id": "amount-mismatch-large",
        "risk_level": "high",
        "input_summary": {
            "invoice_amount": 5000.0,
            "purchase_order_amount": 4000.0,
            "absolute_difference": 1000.0,
        },
        "gold_decision": "HOLD",
        "pytorch_decision": "HOLD",
        "keras_decision": "REVIEW",
        "reproduced_runs": 3,
    },
    "interpretation": {
        "training_metric_scope": (
            "The two framework implementations achieved nearly "
            "identical training accuracy on the synthetic development "
            "corpus, but this did not imply equivalent bounded "
            "verification behaviour."
        ),
        "boundary_scope": (
            "The Keras candidate reproducibly diverged on a high-risk "
            "amount-mismatch boundary case despite training examples "
            "on both sides of the 1000-unit policy boundary."
        ),
        "latency_scope": (
            "Latency measurements are local eager-inference development "
            "measurements and do not establish universal framework "
            "performance superiority."
        ),
        "generality_scope": (
            "This is development evidence on a synthetic policy-derived "
            "corpus and AP v0.2 benchmark. It does not establish unseen "
            "generalisation or production safety."
        ),
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

print("AP PyTorch vs Keras controlled comparison")
print()
print(
    "PyTorch training accuracy:",
    f"{comparison['pytorch']['training_accuracy']:.4f}",
)
print(
    "Keras training accuracy:",
    f"{comparison['keras']['training_accuracy']:.4f}",
)
print()
print(
    "PyTorch decision:",
    comparison["pytorch"]["decision"],
)
print(
    "Keras decision:",
    comparison["keras"]["decision"],
)
print()
print(
    "PyTorch disagreements:",
    comparison["pytorch"]["disagreements"],
)
print(
    "Keras disagreements:",
    comparison["keras"]["disagreements"],
)
print()
print(
    "PyTorch high-risk disagreements:",
    comparison["pytorch"]["high_risk_disagreements"],
)
print(
    "Keras high-risk disagreements:",
    comparison["keras"]["high_risk_disagreements"],
)
print()
print(
    "Evidence report:",
    OUTPUT_PATH,
)
