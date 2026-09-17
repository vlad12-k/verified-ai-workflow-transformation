"""Compare teacher, hard-label student, and distilled AP student."""

import json
from pathlib import Path

import torch

from vait.transformations.library.ap_distillation import (
    train_distilled_student,
    train_hard_label_student,
    trainable_parameter_count,
)
from vait.transformations.library.ap_torch import (
    _train_model,
    _training_arrays,
)
from vait.transformations.library.ap_training import (
    build_synthetic_ap_training_corpus,
)

SEED = 20260916
SAMPLE_COUNT = 600
EPOCHS = 500
LEARNING_RATE = 0.01
TEMPERATURE = 2.0
HARD_LABEL_WEIGHT = 0.5

OUTPUT_PATH = Path(
    "artifacts/benchmarks/"
    "ap-distillation-training-v0.1.json"
)


corpus = build_synthetic_ap_training_corpus(
    sample_count=SAMPLE_COUNT,
    seed=SEED,
)

teacher, teacher_final_loss = _train_model(
    corpus
)

features_array, labels_array = _training_arrays(
    corpus
)

features = torch.from_numpy(
    features_array
)

hard_labels = torch.from_numpy(
    labels_array
)

teacher.eval()

with torch.inference_mode():
    teacher_logits = teacher(
        features
    )

hard_student_result = train_hard_label_student(
    features=features,
    hard_labels=hard_labels,
    seed=SEED,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
)

distilled_student_result = train_distilled_student(
    features=features,
    hard_labels=hard_labels,
    teacher_logits=teacher_logits,
    seed=SEED,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    temperature=TEMPERATURE,
    hard_label_weight=HARD_LABEL_WEIGHT,
)

with torch.inference_mode():
    teacher_predictions = torch.argmax(
        teacher_logits,
        dim=1,
    )

    hard_student_predictions = torch.argmax(
        hard_student_result.model(
            features
        ),
        dim=1,
    )

    distilled_predictions = torch.argmax(
        distilled_student_result.model(
            features
        ),
        dim=1,
    )


def agreement_rate(
    first: torch.Tensor,
    second: torch.Tensor,
) -> float:
    """Return exact class agreement rate."""
    return float(
        (
            first == second
        )
        .float()
        .mean()
        .item()
    )


teacher_accuracy = agreement_rate(
    teacher_predictions,
    hard_labels,
)

hard_student_accuracy = agreement_rate(
    hard_student_predictions,
    hard_labels,
)

distilled_student_accuracy = agreement_rate(
    distilled_predictions,
    hard_labels,
)

hard_teacher_agreement = agreement_rate(
    hard_student_predictions,
    teacher_predictions,
)

distilled_teacher_agreement = agreement_rate(
    distilled_predictions,
    teacher_predictions,
)

teacher_parameters = trainable_parameter_count(
    teacher
)

hard_student_parameters = trainable_parameter_count(
    hard_student_result.model
)

distilled_parameters = trainable_parameter_count(
    distilled_student_result.model
)

if hard_student_parameters != distilled_parameters:
    raise RuntimeError(
        "Hard-label and distilled students must have "
        "identical architecture size."
    )

parameter_reduction = (
    1.0
    - distilled_parameters
    / teacher_parameters
)

report = {
    "experiment_id": (
        "ap-distillation-training-v0.1"
    ),
    "development_evidence": True,
    "corpus": {
        "seed": corpus.seed,
        "cases": len(
            corpus.cases
        ),
        "class_counts": (
            corpus.class_counts
        ),
    },
    "training": {
        "epochs": EPOCHS,
        "learning_rate": (
            LEARNING_RATE
        ),
        "temperature": (
            TEMPERATURE
        ),
        "hard_label_weight": (
            HARD_LABEL_WEIGHT
        ),
    },
    "teacher": {
        "parameters": (
            teacher_parameters
        ),
        "final_training_loss": (
            teacher_final_loss
        ),
        "training_accuracy": (
            teacher_accuracy
        ),
    },
    "hard_label_student": {
        "parameters": (
            hard_student_parameters
        ),
        "final_training_loss": (
            hard_student_result.final_loss
        ),
        "training_accuracy": (
            hard_student_accuracy
        ),
        "teacher_agreement": (
            hard_teacher_agreement
        ),
    },
    "distilled_student": {
        "parameters": (
            distilled_parameters
        ),
        "final_training_loss": (
            distilled_student_result.final_loss
        ),
        "training_accuracy": (
            distilled_student_accuracy
        ),
        "teacher_agreement": (
            distilled_teacher_agreement
        ),
    },
    "parameter_reduction": (
        parameter_reduction
    ),
    "scope": (
        "Training-distribution development evidence only. "
        "This experiment does not establish benchmark "
        "generalisation."
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

print("AP knowledge-distillation training experiment")
print()
print(f"Cases: {len(corpus.cases)}")
print(
    "Class counts:",
    corpus.class_counts,
)
print()
print(
    "Teacher parameters:",
    f"{teacher_parameters:,}",
)
print(
    "Student parameters:",
    f"{distilled_parameters:,}",
)
print(
    "Parameter reduction:",
    f"{100 * parameter_reduction:.2f}%",
)
print()
print(
    "Teacher training accuracy:",
    f"{teacher_accuracy:.4f}",
)
print(
    "Hard student training accuracy:",
    f"{hard_student_accuracy:.4f}",
)
print(
    "Distilled student training accuracy:",
    f"{distilled_student_accuracy:.4f}",
)
print()
print(
    "Hard student teacher agreement:",
    f"{hard_teacher_agreement:.4f}",
)
print(
    "Distilled student teacher agreement:",
    f"{distilled_teacher_agreement:.4f}",
)
print()
print(
    "Teacher final loss:",
    f"{teacher_final_loss:.6f}",
)
print(
    "Hard student final loss:",
    f"{hard_student_result.final_loss:.6f}",
)
print(
    "Distilled student final loss:",
    f"{distilled_student_result.final_loss:.6f}",
)
print()
print(
    f"Evidence report: {OUTPUT_PATH}"
)
