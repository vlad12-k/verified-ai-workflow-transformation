"""Tests for M4 PyTorch precision and compression candidates."""

import pytest

from vait.contracts.models import (
    RiskLevel,
    VerificationCase,
)
from vait.transformations.applicability import (
    ApplicabilityContext,
)
from vait.transformations.library.ap_torch_optimisation import (
    TorchOptimisationCandidate,
    build_pytorch_precision_candidates,
)
from vait.transformations.library.ap_training import (
    AP_DECISIONS,
    APTrainingCorpus,
    build_synthetic_ap_training_corpus,
)
from vait.transformations.models import (
    TransformationCategory,
)


@pytest.fixture(scope="module")
def optimisation_candidates() -> tuple[
    APTrainingCorpus,
    tuple[
        TorchOptimisationCandidate,
        TorchOptimisationCandidate,
    ],
]:
    """Build one matched FP32/INT8 candidate pair."""
    corpus = (
        build_synthetic_ap_training_corpus(
            sample_count=600,
            seed=20260916,
        )
    )

    return (
        corpus,
        build_pytorch_precision_candidates(
            corpus
        ),
    )


def test_precision_candidates_have_distinct_identities(
    optimisation_candidates: tuple[
        APTrainingCorpus,
        tuple[
            TorchOptimisationCandidate,
            TorchOptimisationCandidate,
        ],
    ],
) -> None:
    """Baseline and quantised variants must remain auditable candidates."""
    _, (
        fp32,
        int8,
    ) = optimisation_candidates

    assert (
        fp32.transformation.implementation_id
        == "synthetic-ap-pytorch-mlp-fp32-v1"
    )

    assert (
        int8.transformation.implementation_id
        == "synthetic-ap-pytorch-mlp-int8-dynamic-v1"
    )

    assert (
        fp32.transformation.descriptor.category
        is TransformationCategory.MODEL
    )

    assert (
        int8.transformation.descriptor.category
        is TransformationCategory.OPTIMIZATION
    )


def test_precision_candidates_record_size_evidence(
    optimisation_candidates: tuple[
        APTrainingCorpus,
        tuple[
            TorchOptimisationCandidate,
            TorchOptimisationCandidate,
        ],
    ],
) -> None:
    """Compression evidence must be measured rather than assumed."""
    _, (
        fp32,
        int8,
    ) = optimisation_candidates

    fp32_bytes = (
        fp32.transformation.configuration[
            "serialized_state_bytes"
        ]
    )

    int8_bytes = (
        int8.transformation.configuration[
            "serialized_state_bytes"
        ]
    )

    ratio = (
        int8.transformation.configuration[
            "serialized_state_size_ratio"
        ]
    )

    assert isinstance(
        fp32_bytes,
        int,
    )
    assert isinstance(
        int8_bytes,
        int,
    )
    assert isinstance(
        ratio,
        float,
    )

    assert fp32_bytes > 0
    assert int8_bytes > 0
    assert ratio > 0.0

    assert ratio == pytest.approx(
        int8_bytes
        / fp32_bytes
    )

    assert (
        int8.transformation.configuration[
            "quantization_config"
        ]
        == "Int8DynamicActivationInt8WeightConfig"
    )


@pytest.mark.parametrize(
    "candidate_index",
    [
        pytest.param(
            0,
            id="fp32",
        ),
        pytest.param(
            1,
            id="int8",
        ),
    ],
)
def test_precision_candidate_executes_scalar_and_batch_consistently(
    optimisation_candidates: tuple[
        APTrainingCorpus,
        tuple[
            TorchOptimisationCandidate,
            TorchOptimisationCandidate,
        ],
    ],
    candidate_index: int,
) -> None:
    """Each optimisation candidate must preserve its scalar/batch behaviour."""
    corpus, candidates = (
        optimisation_candidates
    )

    candidate = candidates[
        candidate_index
    ]

    transformation = (
        candidate.transformation
    )

    preparation = (
        transformation.prepare_candidate(
            ApplicabilityContext(
                risk_level=RiskLevel.HIGH,
                available_capabilities=(
                    transformation
                    .descriptor
                    .required_capabilities
                ),
            )
        )
    )

    assert preparation.ready is True
    assert preparation.runner is not None

    source_cases = corpus.cases[
        :8
    ]

    verification_cases = tuple(
        VerificationCase(
            id=f"optimisation-{index}",
            input_data=case.input_data,
            risk_level=RiskLevel.LOW,
        )
        for index, case in enumerate(
            source_cases
        )
    )

    scalar_outputs = tuple(
        preparation.runner.execute(
            case
        ).output
        for case in verification_cases
    )

    batch_outputs = (
        candidate.batch_operation(
            tuple(
                case.input_data
                for case
                in verification_cases
            )
        )
    )

    assert (
        batch_outputs
        == scalar_outputs
    )

    assert all(
        isinstance(
            output,
            dict,
        )
        and output["decision"]
        in AP_DECISIONS
        for output in batch_outputs
    )


def test_int8_candidate_requires_torchao_capability(
    optimisation_candidates: tuple[
        APTrainingCorpus,
        tuple[
            TorchOptimisationCandidate,
            TorchOptimisationCandidate,
        ],
    ],
) -> None:
    """TorchAO candidate must not be applicable without its runtime capability."""
    _, (
        _,
        int8,
    ) = optimisation_candidates

    transformation = (
        int8.transformation
    )

    preparation = (
        transformation.prepare_candidate(
            ApplicabilityContext(
                risk_level=RiskLevel.HIGH,
                available_capabilities=frozenset(
                    {
                        "typed-inputs",
                        "tabular-features",
                        "trained-model",
                        "pytorch-inference",
                    }
                ),
            )
        )
    )

    assert preparation.ready is False
    assert preparation.runner is None
    assert (
        preparation.applicability.is_applicable
        is False
    )
