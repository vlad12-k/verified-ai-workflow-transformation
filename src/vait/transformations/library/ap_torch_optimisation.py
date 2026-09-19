"""PyTorch precision and compression candidates for M4 optimisation."""

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import version as package_version
from io import BytesIO
from time import perf_counter_ns
from typing import NamedTuple, cast

import numpy as np
import torch
from pydantic import JsonValue
from torch import Tensor, nn
from torchao.quantization import (
    Int8DynamicActivationInt8WeightConfig,
    quantize_,
)

from vait.contracts.models import Effect, RiskLevel
from vait.transformations.library.ap_tabular import (
    AP_TABULAR_FEATURE_NAMES,
    extract_ap_tabular_features,
)
from vait.transformations.library.ap_torch import (
    _INDEX_TO_DECISION,
    _train_model,
    _transform_features,
)
from vait.transformations.library.ap_training import APTrainingCorpus
from vait.transformations.models import (
    TransformationCategory,
    TransformationDescriptor,
)
from vait.transformations.python_callable import (
    PythonCallableTransformation,
)

BatchPredictionFunction = Callable[
    [Sequence[dict[str, JsonValue]]],
    tuple[JsonValue, ...],
]


class TorchOptimisationCandidate(NamedTuple):
    """One executable PyTorch optimisation candidate."""

    transformation: PythonCallableTransformation
    batch_operation: BatchPredictionFunction


def build_pytorch_precision_candidates(
    corpus: APTrainingCorpus,
) -> tuple[
    TorchOptimisationCandidate,
    TorchOptimisationCandidate,
]:
    """Build matched FP32 and TorchAO INT8 candidates."""
    trained_model, final_loss = _train_model(
        corpus
    )

    trained_model.eval()

    parameter_count = int(
        sum(
            parameter.numel()
            for parameter in trained_model.parameters()
        )
    )

    fp32_model = deepcopy(
        trained_model
    )

    int8_model = deepcopy(
        trained_model
    )

    fp32_serialized_bytes = (
        _serialized_state_bytes(
            fp32_model
        )
    )

    quantize_(
        int8_model,
        Int8DynamicActivationInt8WeightConfig(),
    )

    int8_model.eval()

    int8_serialized_bytes = (
        _serialized_state_bytes(
            int8_model
        )
    )

    fp32_transformation = (
        PythonCallableTransformation(
            descriptor=_descriptor(
                transformation_id=(
                    "synthetic-ap-pytorch-mlp-fp32-baseline"
                ),
                name=(
                    "Synthetic AP PyTorch MLP "
                    "FP32 baseline"
                ),
                description=(
                    "Matched FP32 baseline for controlled "
                    "precision and compression experiments."
                ),
                category=(
                    TransformationCategory.MODEL
                ),
                required_capabilities=frozenset(
                    {
                        "typed-inputs",
                        "tabular-features",
                        "trained-model",
                        "pytorch-inference",
                    }
                ),
            ),
            implementation_id=(
                "synthetic-ap-pytorch-mlp-fp32-v1"
            ),
            function=_prediction_function(
                fp32_model
            ),
            configuration={
                "model_family": "pytorch_mlp",
                "optimisation": "none",
                "precision": "float32",
                "training_seed": corpus.seed,
                "training_cases": len(
                    corpus.cases
                ),
                "feature_count": len(
                    AP_TABULAR_FEATURE_NAMES
                ),
                "training_device": "cpu",
                "parameter_count": (
                    parameter_count
                ),
                "serialized_state_bytes": (
                    fp32_serialized_bytes
                ),
                "final_training_loss": (
                    final_loss
                ),
            },
        )
    )

    int8_transformation = (
        PythonCallableTransformation(
            descriptor=_descriptor(
                transformation_id=(
                    "synthetic-ap-pytorch-mlp-int8-dynamic"
                ),
                name=(
                    "Synthetic AP PyTorch MLP "
                    "TorchAO INT8 dynamic"
                ),
                description=(
                    "TorchAO dynamic INT8 activation and "
                    "weight quantisation candidate derived "
                    "from the matched FP32 model."
                ),
                category=(
                    TransformationCategory.OPTIMIZATION
                ),
                required_capabilities=frozenset(
                    {
                        "typed-inputs",
                        "tabular-features",
                        "trained-model",
                        "pytorch-inference",
                        "torchao-quantization",
                    }
                ),
            ),
            implementation_id=(
                "synthetic-ap-pytorch-mlp-int8-dynamic-v1"
            ),
            function=_prediction_function(
                int8_model
            ),
            configuration={
                "model_family": "pytorch_mlp",
                "optimisation": (
                    "torchao-int8-dynamic"
                ),
                "source_precision": "float32",
                "activation_quantisation": (
                    "dynamic-int8"
                ),
                "weight_dtype": "int8",
                "training_seed": corpus.seed,
                "training_cases": len(
                    corpus.cases
                ),
                "feature_count": len(
                    AP_TABULAR_FEATURE_NAMES
                ),
                "training_device": "cpu",
                "parameter_count": (
                    parameter_count
                ),
                "serialized_state_bytes": (
                    int8_serialized_bytes
                ),
                "baseline_serialized_state_bytes": (
                    fp32_serialized_bytes
                ),
                "serialized_state_size_ratio": (
                    int8_serialized_bytes
                    / fp32_serialized_bytes
                ),
                "torchao_version": (
                    package_version(
                        "torchao"
                    )
                ),
                "quantization_config": (
                    "Int8DynamicActivationInt8WeightConfig"
                ),
                "final_training_loss": (
                    final_loss
                ),
            },
        )
    )

    return (
        TorchOptimisationCandidate(
            transformation=(
                fp32_transformation
            ),
            batch_operation=(
                _batch_prediction_function(
                    fp32_model
                )
            ),
        ),
        TorchOptimisationCandidate(
            transformation=(
                int8_transformation
            ),
            batch_operation=(
                _batch_prediction_function(
                    int8_model
                )
            ),
        ),
    )


def _descriptor(
    *,
    transformation_id: str,
    name: str,
    description: str,
    category: TransformationCategory,
    required_capabilities: frozenset[str],
) -> TransformationDescriptor:
    """Build one precision-optimisation transformation descriptor."""
    return TransformationDescriptor(
        transformation_id=(
            transformation_id
        ),
        version="1.0.0",
        name=name,
        description=description,
        category=category,
        declared_effects=frozenset(
            {Effect.NONE}
        ),
        supported_risk_levels=frozenset(
            RiskLevel
        ),
        required_capabilities=(
            required_capabilities
        ),
    )


def _prediction_function(
    model: nn.Module,
) -> Callable[
    [dict[str, JsonValue]],
    JsonValue,
]:
    """Wrap one PyTorch precision candidate for scalar execution."""

    def predict(
        data: dict[
            str,
            JsonValue,
        ],
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
            logits = cast(
                Tensor,
                model(
                    tensor
                ),
            )

            predicted_index = int(
                torch.argmax(
                    logits,
                    dim=1,
                ).item()
            )

        return {
            "decision": (
                _decision_from_index(
                    predicted_index
                )
            ),
        }

    return predict


def _batch_prediction_function(
    model: nn.Module,
) -> BatchPredictionFunction:
    """Wrap one PyTorch precision candidate for native batch execution."""

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
            logits = cast(
                Tensor,
                model(
                    tensor
                ),
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            ).cpu()

        outputs: list[
            JsonValue
        ] = []

        for index in range(
            predictions.numel()
        ):
            predicted_index = int(
                predictions[
                    index
                ].item()
            )

            outputs.append(
                {
                    "decision": (
                        _decision_from_index(
                            predicted_index
                        )
                    ),
                }
            )

        return tuple(
            outputs
        )

    return predict_batch


def _decision_from_index(
    index: int,
) -> str:
    """Return the stable AP decision for one class index."""
    try:
        return _INDEX_TO_DECISION[
            index
        ]
    except KeyError as exc:
        raise RuntimeError(
            "Unsupported predicted "
            f"class index: {index}"
        ) from exc


def _serialized_state_bytes(
    model: nn.Module,
) -> int:
    """Measure serialized state-dict size without writing a temporary file."""
    buffer = BytesIO()

    torch.save(
        model.state_dict(),
        buffer,
    )

    return len(
        buffer.getvalue()
    )


@dataclass(frozen=True, slots=True)
class TorchCompileSetup:
    """One static-shape TorchInductor compilation setup measurement."""

    batch_size: int
    wrapper_creation_ms: float
    first_call_compile_ms: float

    @property
    def total_setup_ms(self) -> float:
        """Return wrapper plus first-call compilation cost."""
        return (
            self.wrapper_creation_ms
            + self.first_call_compile_ms
        )


@dataclass(frozen=True, slots=True)
class TorchCompilationCandidate:
    """One compiled candidate with fixed-shape batch operations."""

    transformation: PythonCallableTransformation
    batch_operations: dict[
        int,
        BatchPredictionFunction,
    ]
    setup_by_batch_size: dict[
        int,
        TorchCompileSetup,
    ]


def build_pytorch_compilation_candidate(
    corpus: APTrainingCorpus,
    *,
    batch_sizes: Sequence[int],
) -> TorchCompilationCandidate:
    """Build one verify-first TorchInductor FP32 candidate."""
    declared_batch_sizes = tuple(
        dict.fromkeys(
            int(batch_size)
            for batch_size in batch_sizes
        )
    )

    if not declared_batch_sizes:
        raise ValueError(
            "At least one compiled batch size is required."
        )

    if any(
        batch_size < 1
        for batch_size in declared_batch_sizes
    ):
        raise ValueError(
            "Compiled batch sizes must be positive."
        )

    if 1 not in declared_batch_sizes:
        raise ValueError(
            "Compiled candidate requires batch_size=1 "
            "for scalar verification."
        )

    trained_model, final_loss = _train_model(
        corpus
    )

    trained_model.eval()

    parameter_count = int(
        sum(
            parameter.numel()
            for parameter
            in trained_model.parameters()
        )
    )

    serialized_state_bytes = (
        _serialized_state_bytes(
            trained_model
        )
    )

    compiled_models: dict[
        int,
        nn.Module,
    ] = {}

    setup_by_batch_size: dict[
        int,
        TorchCompileSetup,
    ] = {}

    for batch_size in declared_batch_sizes:
        source_model = deepcopy(
            trained_model
        ).eval()

        wrapper_started = (
            perf_counter_ns()
        )

        compiled_model = cast(
            nn.Module,
            torch.compile(
                source_model,
                backend="inductor",
                fullgraph=True,
                dynamic=False,
            ),
        )

        wrapper_creation_ms = (
            perf_counter_ns()
            - wrapper_started
        ) / 1_000_000

        example_input = torch.zeros(
            (
                batch_size,
                len(
                    AP_TABULAR_FEATURE_NAMES
                ),
            ),
            dtype=torch.float32,
        )

        first_call_started = (
            perf_counter_ns()
        )

        with torch.inference_mode():
            first_output = cast(
                Tensor,
                compiled_model(
                    example_input
                ),
            )

        first_call_compile_ms = (
            perf_counter_ns()
            - first_call_started
        ) / 1_000_000

        if not bool(
            torch.isfinite(
                first_output
            ).all()
        ):
            raise RuntimeError(
                "Compiled model produced non-finite "
                "values during setup."
            )

        compiled_models[
            batch_size
        ] = compiled_model

        setup_by_batch_size[
            batch_size
        ] = TorchCompileSetup(
            batch_size=batch_size,
            wrapper_creation_ms=(
                wrapper_creation_ms
            ),
            first_call_compile_ms=(
                first_call_compile_ms
            ),
        )

    compile_setup_metadata: dict[
        str,
        JsonValue,
    ] = {}

    for (
        batch_size,
        setup,
    ) in setup_by_batch_size.items():
        setup_metadata: dict[
            str,
            JsonValue,
        ] = {
            "wrapper_creation_ms": (
                setup.wrapper_creation_ms
            ),
            "first_call_compile_ms": (
                setup.first_call_compile_ms
            ),
            "total_setup_ms": (
                setup.total_setup_ms
            ),
        }

        compile_setup_metadata[
            str(batch_size)
        ] = setup_metadata

    transformation = (
        PythonCallableTransformation(
            descriptor=_descriptor(
                transformation_id=(
                    "synthetic-ap-pytorch-mlp-inductor"
                ),
                name=(
                    "Synthetic AP PyTorch MLP "
                    "TorchInductor FP32"
                ),
                description=(
                    "Static-shape TorchInductor FP32 "
                    "candidate for controlled compilation "
                    "experiments."
                ),
                category=(
                    TransformationCategory.OPTIMIZATION
                ),
                required_capabilities=frozenset(
                    {
                        "typed-inputs",
                        "tabular-features",
                        "trained-model",
                        "pytorch-inference",
                        "torch-compile",
                    }
                ),
            ),
            implementation_id=(
                "synthetic-ap-pytorch-mlp-inductor-v1"
            ),
            function=_prediction_function(
                compiled_models[1]
            ),
            configuration={
                "model_family": "pytorch_mlp",
                "optimisation": (
                    "torch-compile-inductor"
                ),
                "precision": "float32",
                "compile_backend": "inductor",
                "compile_fullgraph": True,
                "compile_dynamic": False,
                "compiled_batch_sizes": list(
                    declared_batch_sizes
                ),
                "compile_setup_by_batch_size": (
                    compile_setup_metadata
                ),
                "training_seed": corpus.seed,
                "training_cases": len(
                    corpus.cases
                ),
                "feature_count": len(
                    AP_TABULAR_FEATURE_NAMES
                ),
                "training_device": "cpu",
                "parameter_count": (
                    parameter_count
                ),
                "serialized_state_bytes": (
                    serialized_state_bytes
                ),
                "final_training_loss": (
                    final_loss
                ),
            },
        )
    )

    batch_operations = {
        batch_size: (
            _fixed_batch_prediction_function(
                model=compiled_models[
                    batch_size
                ],
                compiled_batch_size=(
                    batch_size
                ),
            )
        )
        for batch_size
        in declared_batch_sizes
    }

    return TorchCompilationCandidate(
        transformation=transformation,
        batch_operations=(
            batch_operations
        ),
        setup_by_batch_size=(
            setup_by_batch_size
        ),
    )


def _fixed_batch_prediction_function(
    *,
    model: nn.Module,
    compiled_batch_size: int,
) -> BatchPredictionFunction:
    """Build one fixed-shape compiled batch operation with tail padding."""

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

        if len(items) > compiled_batch_size:
            raise ValueError(
                "Input batch exceeds the compiled "
                f"batch size {compiled_batch_size}."
            )

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

        padding_count = (
            compiled_batch_size
            - len(items)
        )

        if padding_count:
            padding = np.zeros(
                (
                    padding_count,
                    len(
                        AP_TABULAR_FEATURE_NAMES
                    ),
                ),
                dtype=np.float32,
            )

            features = np.concatenate(
                (
                    features,
                    padding,
                ),
                axis=0,
            )

        tensor = torch.from_numpy(
            features
        )

        with torch.inference_mode():
            logits = cast(
                Tensor,
                model(
                    tensor
                ),
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

        for prediction in predictions[
            : len(items)
        ]:
            outputs.append(
                {
                    "decision": (
                        _decision_from_index(
                            int(
                                prediction
                            )
                        )
                    ),
                }
            )

        return tuple(
            outputs
        )

    return predict_batch



def build_pytorch_lower_precision_candidates(
    corpus: APTrainingCorpus,
) -> tuple[
    TorchOptimisationCandidate,
    TorchOptimisationCandidate,
]:
    """Build matched FP16 and BF16 candidates from one FP32 source model."""
    trained_model, final_loss = _train_model(
        corpus
    )

    trained_model.eval()

    parameter_count = int(
        sum(
            parameter.numel()
            for parameter
            in trained_model.parameters()
        )
    )

    baseline_serialized_bytes = (
        _serialized_state_bytes(
            trained_model
        )
    )

    fp16_model = (
        deepcopy(
            trained_model
        )
        .to(
            dtype=torch.float16
        )
        .eval()
    )

    bf16_model = (
        deepcopy(
            trained_model
        )
        .to(
            dtype=torch.bfloat16
        )
        .eval()
    )

    fp16_bytes = (
        _serialized_state_bytes(
            fp16_model
        )
    )

    bf16_bytes = (
        _serialized_state_bytes(
            bf16_model
        )
    )

    fp16_transformation = (
        PythonCallableTransformation(
            descriptor=_descriptor(
                transformation_id=(
                    "synthetic-ap-pytorch-mlp-fp16"
                ),
                name=(
                    "Synthetic AP PyTorch MLP FP16"
                ),
                description=(
                    "FP16 lower-precision candidate "
                    "derived from the matched FP32 model."
                ),
                category=(
                    TransformationCategory.OPTIMIZATION
                ),
                required_capabilities=frozenset(
                    {
                        "typed-inputs",
                        "tabular-features",
                        "trained-model",
                        "pytorch-inference",
                        "float16-inference",
                    }
                ),
            ),
            implementation_id=(
                "synthetic-ap-pytorch-mlp-fp16-v1"
            ),
            function=(
                _lower_precision_prediction_function(
                    model=fp16_model,
                    dtype=torch.float16,
                )
            ),
            configuration={
                "model_family": "pytorch_mlp",
                "optimisation": (
                    "lower-precision-fp16"
                ),
                "source_precision": "float32",
                "precision": "float16",
                "training_seed": corpus.seed,
                "training_cases": len(
                    corpus.cases
                ),
                "feature_count": len(
                    AP_TABULAR_FEATURE_NAMES
                ),
                "training_device": "cpu",
                "parameter_count": (
                    parameter_count
                ),
                "serialized_state_bytes": (
                    fp16_bytes
                ),
                "baseline_serialized_state_bytes": (
                    baseline_serialized_bytes
                ),
                "serialized_state_size_ratio": (
                    fp16_bytes
                    / baseline_serialized_bytes
                ),
                "final_training_loss": (
                    final_loss
                ),
            },
        )
    )

    bf16_transformation = (
        PythonCallableTransformation(
            descriptor=_descriptor(
                transformation_id=(
                    "synthetic-ap-pytorch-mlp-bf16"
                ),
                name=(
                    "Synthetic AP PyTorch MLP BF16"
                ),
                description=(
                    "BF16 lower-precision candidate "
                    "derived from the matched FP32 model."
                ),
                category=(
                    TransformationCategory.OPTIMIZATION
                ),
                required_capabilities=frozenset(
                    {
                        "typed-inputs",
                        "tabular-features",
                        "trained-model",
                        "pytorch-inference",
                        "bfloat16-inference",
                    }
                ),
            ),
            implementation_id=(
                "synthetic-ap-pytorch-mlp-bf16-v1"
            ),
            function=(
                _lower_precision_prediction_function(
                    model=bf16_model,
                    dtype=torch.bfloat16,
                )
            ),
            configuration={
                "model_family": "pytorch_mlp",
                "optimisation": (
                    "lower-precision-bf16"
                ),
                "source_precision": "float32",
                "precision": "bfloat16",
                "training_seed": corpus.seed,
                "training_cases": len(
                    corpus.cases
                ),
                "feature_count": len(
                    AP_TABULAR_FEATURE_NAMES
                ),
                "training_device": "cpu",
                "parameter_count": (
                    parameter_count
                ),
                "serialized_state_bytes": (
                    bf16_bytes
                ),
                "baseline_serialized_state_bytes": (
                    baseline_serialized_bytes
                ),
                "serialized_state_size_ratio": (
                    bf16_bytes
                    / baseline_serialized_bytes
                ),
                "final_training_loss": (
                    final_loss
                ),
            },
        )
    )

    return (
        TorchOptimisationCandidate(
            transformation=(
                fp16_transformation
            ),
            batch_operation=(
                _lower_precision_batch_prediction_function(
                    model=fp16_model,
                    dtype=torch.float16,
                )
            ),
        ),
        TorchOptimisationCandidate(
            transformation=(
                bf16_transformation
            ),
            batch_operation=(
                _lower_precision_batch_prediction_function(
                    model=bf16_model,
                    dtype=torch.bfloat16,
                )
            ),
        ),
    )


def _lower_precision_prediction_function(
    *,
    model: nn.Module,
    dtype: torch.dtype,
) -> Callable[
    [dict[str, JsonValue]],
    JsonValue,
]:
    """Build scalar inference for one lower-precision candidate."""

    def predict(
        data: dict[
            str,
            JsonValue,
        ],
    ) -> JsonValue:
        features = _transform_features(
            extract_ap_tabular_features(
                data
            )
        )

        tensor = (
            torch.from_numpy(
                features.reshape(
                    1,
                    -1,
                )
            )
            .to(
                dtype=dtype
            )
        )

        with torch.inference_mode():
            logits = cast(
                Tensor,
                model(
                    tensor
                ),
            )

            predicted_index = int(
                torch.argmax(
                    logits,
                    dim=1,
                ).item()
            )

        return {
            "decision": (
                _decision_from_index(
                    predicted_index
                )
            ),
        }

    return predict


def _lower_precision_batch_prediction_function(
    *,
    model: nn.Module,
    dtype: torch.dtype,
) -> BatchPredictionFunction:
    """Build native batch inference for one lower-precision candidate."""

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

        tensor = (
            torch.from_numpy(
                features
            )
            .to(
                dtype=dtype
            )
        )

        with torch.inference_mode():
            logits = cast(
                Tensor,
                model(
                    tensor
                ),
            )

            predictions = (
                torch.argmax(
                    logits,
                    dim=1,
                )
                .cpu()
                .tolist()
            )

        return tuple(
            {
                "decision": (
                    _decision_from_index(
                        int(
                            prediction
                        )
                    )
                ),
            }
            for prediction in predictions
        )

    return predict_batch
