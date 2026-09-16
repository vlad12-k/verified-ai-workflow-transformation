"""Classical ML candidates for the synthetic AP benchmark."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from pydantic import JsonValue
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

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

PredictionFunction = Callable[
    [NDArray[np.float64]],
    NDArray[np.int64],
]


def build_random_forest_transformation(
    corpus: APTrainingCorpus,
) -> PythonCallableTransformation:
    """Train and expose a deterministic Random Forest AP candidate."""
    features, labels = _training_matrix(corpus)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        random_state=corpus.seed,
        n_jobs=1,
    )

    model.fit(features, labels)

    descriptor = TransformationDescriptor(
        transformation_id="synthetic-ap-random-forest-replacement",
        version="1.0.0",
        name="Synthetic AP Random Forest replacement",
        description=(
            "Supervised Random Forest candidate trained on a separate "
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
            }
        ),
    )

    return PythonCallableTransformation(
        descriptor=descriptor,
        implementation_id="synthetic-ap-random-forest-v1",
        function=_prediction_function(model.predict),
        configuration={
            "model_family": "random_forest",
            "training_seed": corpus.seed,
            "training_cases": len(corpus.cases),
            "feature_count": len(AP_TABULAR_FEATURE_NAMES),
            "n_estimators": 300,
            "n_jobs": 1,
        },
    )


def build_xgboost_transformation(
    corpus: APTrainingCorpus,
) -> PythonCallableTransformation:
    """Train and expose a deterministic XGBoost AP candidate."""
    features, labels = _training_matrix(corpus)

    model = XGBClassifier(
        n_estimators=160,
        max_depth=5,
        learning_rate=0.08,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="multi:softmax",
        num_class=len(_DECISION_TO_INDEX),
        eval_metric="mlogloss",
        random_state=corpus.seed,
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
    )

    model.fit(features, labels)

    descriptor = TransformationDescriptor(
        transformation_id="synthetic-ap-xgboost-replacement",
        version="1.0.0",
        name="Synthetic AP XGBoost replacement",
        description=(
            "Supervised XGBoost candidate trained on a separate "
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
            }
        ),
    )

    return PythonCallableTransformation(
        descriptor=descriptor,
        implementation_id="synthetic-ap-xgboost-v1",
        function=_prediction_function(model.predict),
        configuration={
            "model_family": "xgboost",
            "training_seed": corpus.seed,
            "training_cases": len(corpus.cases),
            "feature_count": len(AP_TABULAR_FEATURE_NAMES),
            "n_estimators": 160,
            "max_depth": 5,
            "learning_rate": 0.08,
            "n_jobs": 1,
        },
    )


def _training_matrix(
    corpus: APTrainingCorpus,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
]:
    """Convert the supervised corpus into deterministic model arrays."""
    features = np.asarray(
        [
            extract_ap_tabular_features(case.input_data)
            for case in corpus.cases
        ],
        dtype=np.float64,
    )

    labels = np.asarray(
        [
            _decision_index(case.label)
            for case in corpus.cases
        ],
        dtype=np.int64,
    )

    return features, labels


def _decision_index(decision: str) -> int:
    """Map a benchmark decision to its stable numeric class."""
    try:
        return _DECISION_TO_INDEX[decision]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported AP training decision: {decision}"
        ) from exc


def _prediction_function(
    predict: PredictionFunction,
) -> Callable[[dict[str, JsonValue]], JsonValue]:
    """Wrap a fitted classifier as a VAIT implementation function."""

    def predict_decision(
        data: dict[str, JsonValue],
    ) -> JsonValue:
        features = np.asarray(
            [extract_ap_tabular_features(data)],
            dtype=np.float64,
        )

        prediction = predict(features)

        if prediction.size != 1:
            raise RuntimeError(
                "AP classifier must return exactly one prediction."
            )

        class_index = int(prediction[0])

        try:
            decision = _INDEX_TO_DECISION[class_index]
        except KeyError as exc:
            raise RuntimeError(
                f"Unsupported predicted class index: {class_index}"
            ) from exc

        return {
            "decision": decision,
        }

    return predict_decision
