"""PyTorch tabular candidate for the synthetic AP benchmark."""

from collections.abc import Callable, Sequence
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import JsonValue
from torch import Tensor, nn
from torch.nn import functional as F

from vait.contracts.models import Effect, RiskLevel
from vait.transformations.library.ap_tabular import (
    AP_TABULAR_FEATURE_NAMES,
    extract_ap_tabular_features,
)
from vait.transformations.library.ap_training import APTrainingCorpus
from vait.transformations.models import (
    TransformationCategory,
    TransformationDescriptor,
)
from vait.transformations.python_callable import PythonCallableTransformation

_DECISION_TO_INDEX = {
    "HOLD": 0,
    "RECOMMEND_APPROVE": 1,
    "REVIEW": 2,
}

_INDEX_TO_DECISION = {
    index: decision
    for decision, index in _DECISION_TO_INDEX.items()
}

_SIGNED_LOG_FEATURE_INDICES = (
    0,
    1,
)

_AMOUNT_DIFFERENCE_INDEX = 3

_HIDDEN_DIMENSIONS = (
    32,
    16,
)

_TRAINING_EPOCHS = 500
_LEARNING_RATE = 0.01


class APTabularNetwork(nn.Module):
    """Small feed-forward network for synthetic AP decisions."""

    def __init__(self) -> None:
        """Initialize the tabular classifier."""
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                len(AP_TABULAR_FEATURE_NAMES),
                _HIDDEN_DIMENSIONS[0],
            ),
            nn.ReLU(),
            nn.Linear(
                _HIDDEN_DIMENSIONS[0],
                _HIDDEN_DIMENSIONS[1],
            ),
            nn.ReLU(),
            nn.Linear(
                _HIDDEN_DIMENSIONS[1],
                len(_DECISION_TO_INDEX),
            ),
        )

    def forward(
        self,
        features: Tensor,
    ) -> Tensor:
        """Return decision logits for a batch of AP features."""
        return cast(
            Tensor,
            self.network(features),
        )


def build_pytorch_transformation(
    corpus: APTrainingCorpus,
) -> PythonCallableTransformation:
    """Train and expose a reproducible PyTorch AP candidate."""
    model, final_loss = _train_model(corpus)

    descriptor = TransformationDescriptor(
        transformation_id="synthetic-ap-pytorch-mlp-replacement",
        version="1.0.0",
        name="Synthetic AP PyTorch MLP replacement",
        description=(
            "Feed-forward PyTorch candidate trained on the "
            "synthetic AP training corpus."
        ),
        category=TransformationCategory.MODEL,
        declared_effects=frozenset({Effect.NONE}),
        supported_risk_levels=frozenset(RiskLevel),
        required_capabilities=frozenset(
            {
                "typed-inputs",
                "tabular-features",
                "trained-model",
                "pytorch-inference",
            }
        ),
    )

    return PythonCallableTransformation(
        descriptor=descriptor,
        implementation_id="synthetic-ap-pytorch-mlp-v1",
        function=_prediction_function(model),
        configuration={
            "model_family": "pytorch_mlp",
            "training_seed": corpus.seed,
            "training_cases": len(corpus.cases),
            "feature_count": len(AP_TABULAR_FEATURE_NAMES),
            "hidden_dimensions": list(_HIDDEN_DIMENSIONS),
            "training_epochs": _TRAINING_EPOCHS,
            "learning_rate": _LEARNING_RATE,
            "training_device": "cpu",
            "parameter_count": int(
                sum(
                    parameter.numel()
                    for parameter in model.parameters()
                )
            ),
            "final_training_loss": final_loss,
        },
    )


def _train_model(
    corpus: APTrainingCorpus,
) -> tuple[APTabularNetwork, float]:
    """Train the AP network using deterministic full-batch optimisation."""
    torch.manual_seed(corpus.seed)

    features, labels = _training_arrays(corpus)

    feature_tensor = torch.from_numpy(features)
    label_tensor = torch.from_numpy(labels)

    model = APTabularNetwork()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=_LEARNING_RATE,
        weight_decay=1e-4,
    )

    model.train()
    final_loss = 0.0

    for _ in range(_TRAINING_EPOCHS):
        optimizer.zero_grad(set_to_none=True)

        logits = model(feature_tensor)
        loss = F.cross_entropy(
            logits,
            label_tensor,
        )

        torch.autograd.backward(loss)
        optimizer.step()

        final_loss = float(loss.detach().item())

    model.eval()

    return model, final_loss


def _training_arrays(
    corpus: APTrainingCorpus,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.int64],
]:
    """Convert the supervised corpus into model-ready arrays."""
    features = np.asarray(
        [
            _transform_features(
                extract_ap_tabular_features(case.input_data)
            )
            for case in corpus.cases
        ],
        dtype=np.float32,
    )

    labels = np.asarray(
        [
            _decision_index(case.label)
            for case in corpus.cases
        ],
        dtype=np.int64,
    )

    return features, labels


def _transform_features(
    features: tuple[float, ...],
) -> NDArray[np.float32]:
    """Compress numeric scale while preserving the shared feature contract."""
    transformed = np.asarray(
        features,
        dtype=np.float32,
    ).copy()

    for index in _SIGNED_LOG_FEATURE_INDICES:
        value = transformed[index]

        transformed[index] = (
            np.sign(value)
            * np.log1p(np.abs(value))
        )

    amount_difference = transformed[
        _AMOUNT_DIFFERENCE_INDEX
    ]

    transformed[_AMOUNT_DIFFERENCE_INDEX] = (
        0.0
        if amount_difference == 0.0
        else 1.0 + np.log1p(amount_difference)
    )

    return transformed


def _decision_index(
    decision: str,
) -> int:
    """Map an AP decision to its stable numeric class."""
    try:
        return _DECISION_TO_INDEX[decision]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported AP training decision: {decision}"
        ) from exc


def _prediction_function(
    model: APTabularNetwork,
) -> Callable[[dict[str, JsonValue]], JsonValue]:
    """Wrap the fitted PyTorch network as a VAIT implementation."""

    def predict_decision(
        data: dict[str, JsonValue],
    ) -> JsonValue:
        features = _transform_features(
            extract_ap_tabular_features(data)
        )

        feature_tensor = torch.from_numpy(
            features.reshape(1, -1)
        )

        with torch.inference_mode():
            logits = model(feature_tensor)
            prediction = int(
                torch.argmax(
                    logits,
                    dim=1,
                ).item()
            )

        try:
            decision = _INDEX_TO_DECISION[prediction]
        except KeyError as exc:
            raise RuntimeError(
                f"Unsupported predicted class index: {prediction}"
            ) from exc

        return {
            "decision": decision,
        }

    return predict_decision


def build_pytorch_native_batch_candidate(
    corpus: APTrainingCorpus,
) -> tuple[
    PythonCallableTransformation,
    Callable[
        [Sequence[dict[str, JsonValue]]],
        tuple[JsonValue, ...],
    ],
]:
    """Build one PyTorch candidate with scalar and native-batch paths."""
    model, final_loss = _train_model(
        corpus
    )

    parameter_count = int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
        )
    )

    descriptor = TransformationDescriptor(
        transformation_id=(
            "synthetic-ap-pytorch-mlp-replacement"
        ),
        version="1.0.0",
        name="Synthetic AP PyTorch MLP replacement",
        description=(
            "Feed-forward PyTorch candidate trained on the "
            "synthetic AP training corpus."
        ),
        category=TransformationCategory.MODEL,
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

    transformation = PythonCallableTransformation(
        descriptor=descriptor,
        implementation_id=(
            "synthetic-ap-pytorch-mlp-v1"
        ),
        function=_prediction_function(
            model
        ),
        configuration={
            "model_family": "pytorch_mlp",
            "training_seed": corpus.seed,
            "training_cases": len(
                corpus.cases
            ),
            "feature_count": len(
                AP_TABULAR_FEATURE_NAMES
            ),
            "hidden_dimensions": list(
                _HIDDEN_DIMENSIONS
            ),
            "training_epochs": (
                _TRAINING_EPOCHS
            ),
            "learning_rate": (
                _LEARNING_RATE
            ),
            "training_device": "cpu",
            "parameter_count": (
                parameter_count
            ),
            "final_training_loss": (
                final_loss
            ),
        },
    )

    return (
        transformation,
        _native_batch_prediction_function(
            model
        ),
    )


def _native_batch_prediction_function(
    model: APTabularNetwork,
) -> Callable[
    [Sequence[dict[str, JsonValue]]],
    tuple[JsonValue, ...],
]:
    """Build genuine vectorised PyTorch inference over one input batch."""

    def predict_batch(
        items: Sequence[
            dict[str, JsonValue]
        ],
    ) -> tuple[
        JsonValue,
        ...,
    ]:
        if not items:
            return ()

        features = np.asarray(
            [
                _transform_features(
                    extract_ap_tabular_features(
                        item
                    )
                )
                for item in items
            ],
            dtype=np.float32,
        )

        tensor = torch.from_numpy(
            features
        )

        with torch.inference_mode():
            logits = model(
                tensor
            )

            predictions = (
                torch.argmax(
                    logits,
                    dim=1,
                )
                .cpu()
                .tolist()
            )

        outputs: list[
            JsonValue
        ] = []

        for prediction in predictions:
            class_index = int(
                prediction
            )

            try:
                decision = (
                    _INDEX_TO_DECISION[
                        class_index
                    ]
                )
            except KeyError as exc:
                raise RuntimeError(
                    "Unsupported predicted "
                    f"class index: {class_index}"
                ) from exc

            outputs.append(
                {
                    "decision": decision,
                }
            )

        return tuple(
            outputs
        )

    return predict_batch
