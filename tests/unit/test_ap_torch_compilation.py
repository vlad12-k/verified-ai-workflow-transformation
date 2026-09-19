"""Tests for M4 TorchInductor optimisation candidates."""

import pytest
import torch
from torch import nn

from vait.contracts.models import (
    RiskLevel,
    VerificationCase,
)
from vait.transformations.applicability import (
    ApplicabilityContext,
)
from vait.transformations.library.ap_torch_optimisation import (
    TorchCompilationCandidate,
    build_pytorch_compilation_candidate,
)
from vait.transformations.library.ap_training import (
    APTrainingCorpus,
    build_synthetic_ap_training_corpus,
)
from vait.transformations.models import (
    TransformationCategory,
)


@pytest.fixture(scope="module")
def training_corpus() -> APTrainingCorpus:
    """Create one deterministic AP corpus."""
    return (
        build_synthetic_ap_training_corpus(
            sample_count=100,
            seed=20260916,
        )
    )


@pytest.fixture()
def compiled_candidate(
    monkeypatch: pytest.MonkeyPatch,
    training_corpus: APTrainingCorpus,
) -> TorchCompilationCandidate:
    """Build candidate with compiler mocked to identity."""
    def fake_compile(
        model: nn.Module,
        **_: object,
    ) -> nn.Module:
        return model

    monkeypatch.setattr(
        torch,
        "compile",
        fake_compile,
    )

    return (
        build_pytorch_compilation_candidate(
            training_corpus,
            batch_sizes=(
                1,
                4,
            ),
        )
    )


def test_compilation_candidate_has_distinct_identity(
    compiled_candidate: TorchCompilationCandidate,
) -> None:
    """Compiled execution must remain an auditable candidate."""
    transformation = (
        compiled_candidate.transformation
    )

    assert (
        transformation.implementation_id
        == "synthetic-ap-pytorch-mlp-inductor-v1"
    )

    assert (
        transformation.descriptor.category
        is TransformationCategory.OPTIMIZATION
    )

    assert (
        transformation.configuration[
            "compile_backend"
        ]
        == "inductor"
    )

    assert (
        transformation.configuration[
            "compile_dynamic"
        ]
        is False
    )

    assert (
        transformation.configuration[
            "compiled_batch_sizes"
        ]
        == [
            1,
            4,
        ]
    )


def test_compilation_candidate_records_setup_evidence(
    compiled_candidate: TorchCompilationCandidate,
) -> None:
    """Each static search point must retain setup-cost evidence."""
    setup = (
        compiled_candidate
        .setup_by_batch_size
    )

    assert set(
        setup
    ) == {
        1,
        4,
    }

    for batch_size in (
        1,
        4,
    ):
        item = setup[
            batch_size
        ]

        assert (
            item.batch_size
            == batch_size
        )

        assert (
            item.wrapper_creation_ms
            >= 0.0
        )

        assert (
            item.first_call_compile_ms
            >= 0.0
        )

        assert (
            item.total_setup_ms
            >= 0.0
        )


def test_compiled_scalar_and_fixed_batch_are_equivalent(
    compiled_candidate: TorchCompilationCandidate,
    training_corpus: APTrainingCorpus,
) -> None:
    """Compiled batch execution must preserve scalar behaviour."""
    transformation = (
        compiled_candidate.transformation
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
            id=f"compile-{index}",
            input_data=case.input_data,
            risk_level=RiskLevel.LOW,
        )
        for index, case in enumerate(
            training_corpus.cases[
                :3
            ]
        )
    )

    scalar_outputs = tuple(
        preparation.runner.execute(
            case
        ).output
        for case in cases
    )

    batch_operation = (
        compiled_candidate
        .batch_operations[
            4
        ]
    )

    batch_outputs = (
        batch_operation(
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

    assert len(
        batch_outputs
    ) == 3


def test_compiled_batch_rejects_oversized_input(
    compiled_candidate: TorchCompilationCandidate,
    training_corpus: APTrainingCorpus,
) -> None:
    """Static compiled path must not silently exceed its shape."""
    operation = (
        compiled_candidate
        .batch_operations[
            4
        ]
    )

    inputs = tuple(
        case.input_data
        for case
        in training_corpus.cases[
            :5
        ]
    )

    with pytest.raises(
        ValueError,
        match=(
            "exceeds the compiled batch size"
        ),
    ):
        operation(
            inputs
        )


def test_compilation_candidate_requires_compile_capability(
    compiled_candidate: TorchCompilationCandidate,
) -> None:
    """Missing torch-compile capability must block preparation."""
    transformation = (
        compiled_candidate.transformation
    )

    preparation = (
        transformation.prepare_candidate(
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
