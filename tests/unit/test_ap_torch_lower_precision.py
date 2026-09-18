"""Tests for M4-E PyTorch FP16 and BF16 candidates."""

import pytest

from vait.contracts.models import (
    RiskLevel,
    VerificationCase,
)
from vait.transformations.applicability import (
    ApplicabilityContext,
)
from vait.transformations.library.ap_torch_optimisation import (
    build_pytorch_lower_precision_candidates,
)
from vait.transformations.library.ap_training import (
    AP_DECISIONS,
    build_synthetic_ap_training_corpus,
)
from vait.transformations.models import (
    TransformationCategory,
)


@pytest.fixture(scope="module")
def lower_precision_candidates():
    """Build one matched FP16/BF16 pair."""
    corpus = (
        build_synthetic_ap_training_corpus(
            sample_count=600,
            seed=20260916,
        )
    )

    return (
        corpus,
        build_pytorch_lower_precision_candidates(
            corpus
        ),
    )


def test_lower_precision_candidates_have_distinct_identities(
    lower_precision_candidates,
) -> None:
    """FP16 and BF16 must remain separate auditable candidates."""
    _, (
        fp16,
        bf16,
    ) = lower_precision_candidates

    assert (
        fp16.transformation.implementation_id
        == "synthetic-ap-pytorch-mlp-fp16-v1"
    )

    assert (
        bf16.transformation.implementation_id
        == "synthetic-ap-pytorch-mlp-bf16-v1"
    )

    assert (
        fp16.transformation.descriptor.category
        is TransformationCategory.OPTIMIZATION
    )

    assert (
        bf16.transformation.descriptor.category
        is TransformationCategory.OPTIMIZATION
    )


@pytest.mark.parametrize(
    (
        "candidate_index",
        "precision",
    ),
    [
        pytest.param(
            0,
            "float16",
            id="fp16",
        ),
        pytest.param(
            1,
            "bfloat16",
            id="bf16",
        ),
    ],
)
def test_lower_precision_candidates_record_size_evidence(
    lower_precision_candidates,
    candidate_index,
    precision,
) -> None:
    """Lower precision must expose measured footprint evidence."""
    _, candidates = (
        lower_precision_candidates
    )

    transformation = (
        candidates[
            candidate_index
        ].transformation
    )

    configuration = (
        transformation.configuration
    )

    assert (
        configuration[
            "precision"
        ]
        == precision
    )

    baseline_bytes = (
        configuration[
            "baseline_serialized_state_bytes"
        ]
    )

    candidate_bytes = (
        configuration[
            "serialized_state_bytes"
        ]
    )

    ratio = (
        configuration[
            "serialized_state_size_ratio"
        ]
    )

    assert isinstance(
        baseline_bytes,
        int,
    )

    assert isinstance(
        candidate_bytes,
        int,
    )

    assert isinstance(
        ratio,
        float,
    )

    assert baseline_bytes > 0
    assert candidate_bytes > 0

    assert ratio == pytest.approx(
        candidate_bytes
        / baseline_bytes
    )


@pytest.mark.parametrize(
    "candidate_index",
    [
        pytest.param(
            0,
            id="fp16",
        ),
        pytest.param(
            1,
            id="bf16",
        ),
    ],
)
def test_lower_precision_scalar_and_batch_are_equivalent(
    lower_precision_candidates,
    candidate_index,
) -> None:
    """Native batch path must preserve scalar candidate behaviour."""
    corpus, candidates = (
        lower_precision_candidates
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
                risk_level=(
                    RiskLevel.HIGH
                ),
                available_capabilities=(
                    transformation
                    .descriptor
                    .required_capabilities
                ),
            )
        )
    )

    assert preparation.runner is not None

    cases = tuple(
        VerificationCase(
            id=f"lower-precision-{index}",
            input_data=case.input_data,
            risk_level=RiskLevel.LOW,
        )
        for index, case in enumerate(
            corpus.cases[
                :8
            ]
        )
    )

    scalar_outputs = tuple(
        preparation.runner.execute(
            case
        ).output
        for case in cases
    )

    batch_outputs = (
        candidate.batch_operation(
            tuple(
                case.input_data
                for case in cases
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
        and output[
            "decision"
        ]
        in AP_DECISIONS
        for output in batch_outputs
    )


def test_fp16_requires_float16_capability(
    lower_precision_candidates,
) -> None:
    """FP16 candidate must declare its runtime capability."""
    _, (
        fp16,
        _,
    ) = lower_precision_candidates

    preparation = (
        fp16.transformation.prepare_candidate(
            ApplicabilityContext(
                risk_level=(
                    RiskLevel.HIGH
                ),
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


def test_bf16_requires_bfloat16_capability(
    lower_precision_candidates,
) -> None:
    """BF16 candidate must declare its runtime capability."""
    _, (
        _,
        bf16,
    ) = lower_precision_candidates

    preparation = (
        bf16.transformation.prepare_candidate(
            ApplicabilityContext(
                risk_level=(
                    RiskLevel.HIGH
                ),
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
