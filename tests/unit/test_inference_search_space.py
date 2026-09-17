"""Tests for bounded declarative inference search spaces."""

import pytest

from vait.inference.models import (
    InferenceConfiguration,
)
from vait.optimisation.search_space import (
    CandidateConfiguration,
    InferenceCandidate,
    InferenceSearchSpace,
    enumerate_search_points,
)


def build_configuration(
    *,
    provider: str,
    runtime: str,
    device: str,
    dtype: str,
    batch_size: int,
    model_id: str | None = None,
) -> InferenceConfiguration:
    """Build one inference configuration used by search-space tests."""
    return InferenceConfiguration(
        provider=provider,
        runtime=runtime,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        model_id=model_id,
        model_revision=None,
    )


def test_search_space_enumerates_only_declared_pairs() -> None:
    """Enumeration must not synthesize an arbitrary Cartesian product."""
    rules = InferenceCandidate(
        candidate_id="rules",
        implementation_id="synthetic-ap-policy-v0.2",
        allowed_configurations=(
            CandidateConfiguration(
                configuration_id="cpu-b1",
                configuration=build_configuration(
                    provider="local",
                    runtime="python",
                    device="cpu",
                    dtype="native",
                    batch_size=1,
                ),
            ),
        ),
    )

    pytorch = InferenceCandidate(
        candidate_id="pytorch-mlp",
        implementation_id="synthetic-ap-pytorch-mlp-v1",
        allowed_configurations=(
            CandidateConfiguration(
                configuration_id="cpu-fp32-b1",
                configuration=build_configuration(
                    provider="local",
                    runtime="pytorch",
                    device="cpu",
                    dtype="float32",
                    batch_size=1,
                ),
            ),
            CandidateConfiguration(
                configuration_id="cpu-fp32-b8",
                configuration=build_configuration(
                    provider="local",
                    runtime="pytorch",
                    device="cpu",
                    dtype="float32",
                    batch_size=8,
                ),
            ),
        ),
    )

    search_space = InferenceSearchSpace(
        search_space_id="ap-structured-v01",
        version="0.1.0",
        task_family="structured-decision",
        candidates=(
            pytorch,
            rules,
        ),
    )

    points = enumerate_search_points(
        search_space
    )

    assert len(points) == 3

    assert [
        point.search_point_id
        for point in points
    ] == [
        "pytorch-mlp::cpu-fp32-b1",
        "pytorch-mlp::cpu-fp32-b8",
        "rules::cpu-b1",
    ]

    assert [
        point.configuration.batch_size
        for point in points
    ] == [
        1,
        8,
        1,
    ]

    assert all(
        point.task_family
        == "structured-decision"
        for point in points
    )


def test_search_space_preserves_verification_requirement() -> None:
    """Each search point must preserve whether verification is required."""
    candidate = InferenceCandidate(
        candidate_id="verified-candidate",
        implementation_id="candidate-v1",
        allowed_configurations=(
            CandidateConfiguration(
                configuration_id="default",
                configuration=build_configuration(
                    provider="local",
                    runtime="python",
                    device="cpu",
                    dtype="native",
                    batch_size=1,
                ),
            ),
        ),
        requires_verification=True,
    )

    search_space = InferenceSearchSpace(
        search_space_id="verification-space",
        version="0.1.0",
        task_family="structured-decision",
        candidates=(candidate,),
    )

    point = enumerate_search_points(
        search_space
    )[0]

    assert point.requires_verification is True
    assert point.candidate_id == "verified-candidate"
    assert point.implementation_id == "candidate-v1"


def test_candidate_rejects_duplicate_configuration_ids() -> None:
    """A candidate cannot contain ambiguous configuration identities."""
    configuration = build_configuration(
        provider="local",
        runtime="python",
        device="cpu",
        dtype="native",
        batch_size=1,
    )

    with pytest.raises(
        ValueError,
        match="configuration_id",
    ):
        InferenceCandidate(
            candidate_id="duplicate-config",
            implementation_id="candidate-v1",
            allowed_configurations=(
                CandidateConfiguration(
                    configuration_id="same",
                    configuration=configuration,
                ),
                CandidateConfiguration(
                    configuration_id="same",
                    configuration=configuration,
                ),
            ),
        )


def test_search_space_rejects_duplicate_candidate_ids() -> None:
    """Candidate identity must be unique within one compatible search space."""
    configuration = CandidateConfiguration(
        configuration_id="default",
        configuration=build_configuration(
            provider="local",
            runtime="python",
            device="cpu",
            dtype="native",
            batch_size=1,
        ),
    )

    candidate_one = InferenceCandidate(
        candidate_id="duplicate",
        implementation_id="candidate-v1",
        allowed_configurations=(
            configuration,
        ),
    )

    candidate_two = InferenceCandidate(
        candidate_id="duplicate",
        implementation_id="candidate-v2",
        allowed_configurations=(
            configuration,
        ),
    )

    with pytest.raises(
        ValueError,
        match="candidate_id",
    ):
        InferenceSearchSpace(
            search_space_id="duplicate-space",
            version="0.1.0",
            task_family="structured-decision",
            candidates=(
                candidate_one,
                candidate_two,
            ),
        )


def test_search_point_keeps_candidate_and_configuration_metadata() -> None:
    """Candidate and runtime metadata must remain separately attributable."""
    candidate = InferenceCandidate(
        candidate_id="qwen-local",
        implementation_id="qwen-provider-v1",
        allowed_configurations=(
            CandidateConfiguration(
                configuration_id="cpu-b1",
                configuration=build_configuration(
                    provider="local-huggingface",
                    runtime="transformers-causal-lm",
                    device="cpu",
                    dtype="framework-default",
                    batch_size=1,
                    model_id=(
                        "Qwen/Qwen2.5-0.5B-Instruct"
                    ),
                ),
                metadata={
                    "max_output_tokens": 32,
                },
            ),
        ),
        metadata={
            "candidate_family": "slm",
        },
    )

    search_space = InferenceSearchSpace(
        search_space_id="qwen-space",
        version="0.1.0",
        task_family="provider-generative-inference",
        candidates=(candidate,),
    )

    point = enumerate_search_points(
        search_space
    )[0]

    assert (
        point.candidate_metadata["candidate_family"]
        == "slm"
    )
    assert (
        point.configuration_metadata[
            "max_output_tokens"
        ]
        == 32
    )
