"""Keras tabular candidate for the synthetic AP benchmark."""

from collections.abc import Callable
from typing import cast

import keras
import numpy as np
from pydantic import JsonValue

from vait.contracts.models import Effect, RiskLevel
from vait.transformations.library.ap_tabular import (
    AP_TABULAR_FEATURE_NAMES,
    extract_ap_tabular_features,
)
from vait.transformations.library.ap_torch import (
    _training_arrays,
    _transform_features,
)
from vait.transformations.library.ap_training import APTrainingCorpus
from vait.transformations.models import (
    TransformationCategory,
    TransformationDescriptor,
)
from vait.transformations.python_callable import PythonCallableTransformation

_INDEX_TO_DECISION = {
    0: "HOLD",
    1: "RECOMMEND_APPROVE",
    2: "REVIEW",
}

_HIDDEN_DIMENSIONS = (
    32,
    16,
)

_TRAINING_EPOCHS = 500
_LEARNING_RATE = 0.01


def build_keras_model(
    *,
    seed: int,
) -> keras.Model:
    """Build the Keras AP classifier."""
    keras.utils.set_random_seed(seed)

    model = keras.Sequential(
        [
            keras.layers.Input(
                shape=(len(AP_TABULAR_FEATURE_NAMES),),
            ),
            keras.layers.Dense(
                _HIDDEN_DIMENSIONS[0],
                activation="relu",
            ),
            keras.layers.Dense(
                _HIDDEN_DIMENSIONS[1],
                activation="relu",
            ),
            keras.layers.Dense(
                len(_INDEX_TO_DECISION),
            ),
        ]
    )

    return cast(
        keras.Model,
        model,
    )


def train_keras_model(
    corpus: APTrainingCorpus,
) -> tuple[keras.Model, float]:
    """Train the AP model using deterministic full-batch optimisation."""
    features, labels = _training_arrays(
        corpus
    )

    model = build_keras_model(
        seed=corpus.seed,
    )

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=_LEARNING_RATE,
            weight_decay=1e-4,
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
        ),
    )

    history = model.fit(
        features,
        labels,
        batch_size=len(corpus.cases),
        epochs=_TRAINING_EPOCHS,
        shuffle=False,
        verbose=0,
    )

    losses = history.history.get(
        "loss"
    )

    if not losses:
        raise RuntimeError(
            "Keras training did not produce loss history."
        )

    final_loss = float(
        losses[-1]
    )

    return model, final_loss


def build_keras_transformation(
    corpus: APTrainingCorpus,
) -> PythonCallableTransformation:
    """Train and expose a reproducible Keras AP candidate."""
    model, final_loss = train_keras_model(
        corpus
    )

    descriptor = TransformationDescriptor(
        transformation_id=(
            "synthetic-ap-keras-mlp-replacement"
        ),
        version="1.0.0",
        name="Synthetic AP Keras MLP replacement",
        description=(
            "Feed-forward Keras candidate trained on the "
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
                "keras-inference",
            }
        ),
    )

    return PythonCallableTransformation(
        descriptor=descriptor,
        implementation_id=(
            "synthetic-ap-keras-mlp-v1"
        ),
        function=_prediction_function(
            model
        ),
        configuration={
            "model_family": "keras_mlp",
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
            "training_epochs": _TRAINING_EPOCHS,
            "learning_rate": _LEARNING_RATE,
            "training_device": "cpu",
            "parameter_count": int(
                model.count_params()
            ),
            "final_training_loss": final_loss,
        },
    )


def _prediction_function(
    model: keras.Model,
) -> Callable[[dict[str, JsonValue]], JsonValue]:
    """Wrap the fitted Keras network as a VAIT implementation."""

    def predict_decision(
        data: dict[str, JsonValue],
    ) -> JsonValue:
        features = _transform_features(
            extract_ap_tabular_features(
                data
            )
        )

        logits = model(
            features.reshape(
                1,
                -1,
            ),
            training=False,
        )

        prediction = int(
            np.argmax(
                np.asarray(logits),
                axis=1,
            )[0]
        )

        try:
            decision = _INDEX_TO_DECISION[
                prediction
            ]
        except KeyError as exc:
            raise RuntimeError(
                "Unsupported predicted class index: "
                f"{prediction}"
            ) from exc

        return {
            "decision": decision,
        }

    return predict_decision
