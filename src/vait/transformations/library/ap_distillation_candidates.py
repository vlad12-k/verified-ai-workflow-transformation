"""VAIT candidate wrappers for AP knowledge-distillation students."""

from collections.abc import Callable

import torch
from pydantic import JsonValue

from vait.contracts.models import Effect, RiskLevel
from vait.transformations.library.ap_distillation import (
    APDistillationStudent,
    train_distilled_student,
    train_hard_label_student,
)
from vait.transformations.library.ap_tabular import (
    AP_TABULAR_FEATURE_NAMES,
    extract_ap_tabular_features,
)
from vait.transformations.library.ap_torch import (
    _train_model,
    _training_arrays,
    _transform_features,
)
from vait.transformations.library.ap_training import (
    APTrainingCorpus,
)
from vait.transformations.models import (
    TransformationCategory,
    TransformationDescriptor,
)
from vait.transformations.python_callable import (
    PythonCallableTransformation,
)

_INDEX_TO_DECISION = {
    0: "HOLD",
    1: "RECOMMEND_APPROVE",
    2: "REVIEW",
}

_STUDENT_EPOCHS = 500
_STUDENT_LEARNING_RATE = 0.01
_DISTILLATION_TEMPERATURE = 2.0
_HARD_LABEL_WEIGHT = 0.5


def build_ap_distillation_transformations(
    corpus: APTrainingCorpus,
) -> tuple[
    PythonCallableTransformation,
    PythonCallableTransformation,
]:
    """Build hard-label and distilled compact AP candidates."""
    features_array, labels_array = _training_arrays(
        corpus
    )

    features = torch.from_numpy(
        features_array
    )

    labels = torch.from_numpy(
        labels_array
    )

    teacher, _ = _train_model(
        corpus
    )

    teacher.eval()

    with torch.inference_mode():
        teacher_logits = teacher(
            features
        )

    hard_result = train_hard_label_student(
        features=features,
        hard_labels=labels,
        seed=corpus.seed,
        epochs=_STUDENT_EPOCHS,
        learning_rate=_STUDENT_LEARNING_RATE,
    )

    distilled_result = train_distilled_student(
        features=features,
        hard_labels=labels,
        teacher_logits=teacher_logits,
        seed=corpus.seed,
        epochs=_STUDENT_EPOCHS,
        learning_rate=_STUDENT_LEARNING_RATE,
        temperature=_DISTILLATION_TEMPERATURE,
        hard_label_weight=_HARD_LABEL_WEIGHT,
    )

    hard_transformation = PythonCallableTransformation(
        descriptor=_descriptor(
            transformation_id=(
                "synthetic-ap-pytorch-student-hard-label"
            ),
            name=(
                "Synthetic AP compact PyTorch "
                "hard-label student"
            ),
            description=(
                "Compact PyTorch AP classifier trained "
                "directly from synthetic hard labels."
            ),
        ),
        implementation_id=(
            "synthetic-ap-pytorch-student-hard-label-v1"
        ),
        function=_prediction_function(
            hard_result.model
        ),
        configuration={
            "model_family": "pytorch_mlp_student",
            "training_method": "hard_labels",
            "training_seed": corpus.seed,
            "training_cases": len(corpus.cases),
            "feature_count": len(
                AP_TABULAR_FEATURE_NAMES
            ),
            "hidden_dimensions": [8],
            "training_epochs": hard_result.epochs,
            "learning_rate": hard_result.learning_rate,
            "final_training_loss": hard_result.final_loss,
        },
    )

    distilled_transformation = PythonCallableTransformation(
        descriptor=_descriptor(
            transformation_id=(
                "synthetic-ap-pytorch-student-distilled"
            ),
            name=(
                "Synthetic AP compact PyTorch "
                "distilled student"
            ),
            description=(
                "Compact PyTorch AP classifier trained "
                "with teacher soft targets and hard labels."
            ),
        ),
        implementation_id=(
            "synthetic-ap-pytorch-student-distilled-v1"
        ),
        function=_prediction_function(
            distilled_result.model
        ),
        configuration={
            "model_family": "pytorch_mlp_student",
            "training_method": "knowledge_distillation",
            "training_seed": corpus.seed,
            "training_cases": len(corpus.cases),
            "feature_count": len(
                AP_TABULAR_FEATURE_NAMES
            ),
            "hidden_dimensions": [8],
            "training_epochs": distilled_result.epochs,
            "learning_rate": (
                distilled_result.learning_rate
            ),
            "temperature": (
                distilled_result.temperature
            ),
            "hard_label_weight": (
                distilled_result.hard_label_weight
            ),
            "final_training_loss": (
                distilled_result.final_loss
            ),
        },
    )

    return (
        hard_transformation,
        distilled_transformation,
    )


def _descriptor(
    *,
    transformation_id: str,
    name: str,
    description: str,
) -> TransformationDescriptor:
    """Build shared metadata for compact AP students."""
    return TransformationDescriptor(
        transformation_id=transformation_id,
        version="1.0.0",
        name=name,
        description=description,
        category=TransformationCategory.OPTIMIZATION,
        declared_effects=frozenset(
            {Effect.NONE}
        ),
        supported_risk_levels=frozenset(
            RiskLevel
        ),
        required_capabilities=frozenset(
            {
                "typed-inputs",
                "tabular-features",
                "trained-model",
                "pytorch-inference",
            }
        ),
    )


def _prediction_function(
    model: APDistillationStudent,
) -> Callable[[dict[str, JsonValue]], JsonValue]:
    """Wrap a compact fitted student as a VAIT implementation."""

    def predict(
        data: dict[str, JsonValue],
    ) -> JsonValue:
        features = _transform_features(
            extract_ap_tabular_features(
                data
            )
        )

        tensor = torch.from_numpy(
            features.reshape(
                1,
                -1,
            )
        )

        with torch.inference_mode():
            logits = model(
                tensor
            )

            predicted_index = int(
                torch.argmax(
                    logits,
                    dim=1,
                ).item()
            )

        try:
            decision = _INDEX_TO_DECISION[
                predicted_index
            ]
        except KeyError as exc:
            raise RuntimeError(
                "Unsupported distilled-student "
                f"class index: {predicted_index}"
            ) from exc

        return {
            "decision": decision,
        }

    return predict
