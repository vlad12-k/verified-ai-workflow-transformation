"""Evaluate Keras AP training on the full synthetic development corpus."""

import json
from pathlib import Path

import numpy as np

from vait.transformations.library.ap_keras import (
    train_keras_model,
)
from vait.transformations.library.ap_torch import (
    _training_arrays,
)
from vait.transformations.library.ap_training import (
    build_synthetic_ap_training_corpus,
)

OUTPUT_PATH = Path(
    "artifacts/benchmarks/"
    "ap-keras-training-v0.1.json"
)

corpus = build_synthetic_ap_training_corpus()

features, labels = _training_arrays(
    corpus
)

model, final_loss = train_keras_model(
    corpus
)

logits = model(
    features,
    training=False,
)

predictions = np.argmax(
    np.asarray(logits),
    axis=1,
)

training_accuracy = float(
    np.mean(
        predictions == labels
    )
)

report = {
    "experiment_id": (
        "ap-keras-training-v0.1"
    ),
    "development_evidence": True,
    "framework": {
        "name": "keras",
        "backend": "tensorflow",
    },
    "corpus": {
        "cases": len(
            corpus.cases
        ),
        "class_counts": corpus.class_counts,
        "seed": corpus.seed,
    },
    "model": {
        "parameters": int(
            model.count_params()
        ),
        "training_accuracy": (
            training_accuracy
        ),
        "final_training_loss": (
            final_loss
        ),
    },
    "scope": (
        "Development evidence on a synthetic policy-derived "
        "training corpus. This does not establish unseen "
        "generalisation or production safety."
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

print("AP Keras training experiment")
print()
print(
    "Cases:",
    len(corpus.cases),
)
print(
    "Class counts:",
    corpus.class_counts,
)
print(
    "Parameters:",
    model.count_params(),
)
print(
    "Training accuracy:",
    f"{training_accuracy:.4f}",
)
print(
    "Final loss:",
    f"{final_loss:.6f}",
)
print()
print(
    "Evidence report:",
    OUTPUT_PATH,
)
