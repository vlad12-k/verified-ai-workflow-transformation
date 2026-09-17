"""Declarative bounded candidate and configuration search spaces."""

from typing import Self

from pydantic import BaseModel, Field, JsonValue, model_validator

from vait.inference.models import InferenceConfiguration


class CandidateConfiguration(BaseModel):
    """One explicitly allowed inference configuration for a candidate."""

    configuration_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    )

    configuration: InferenceConfiguration

    metadata: dict[str, JsonValue] = Field(
        default_factory=dict,
    )


class InferenceCandidate(BaseModel):
    """One registered candidate and only the configurations it may use."""

    candidate_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    )

    implementation_id: str = Field(
        min_length=1,
    )

    allowed_configurations: tuple[
        CandidateConfiguration,
        ...,
    ] = Field(
        min_length=1,
    )

    requires_verification: bool = True

    metadata: dict[str, JsonValue] = Field(
        default_factory=dict,
    )

    @model_validator(mode="after")
    def validate_configuration_ids(
        self,
    ) -> Self:
        """Require deterministic unique configuration identities."""
        configuration_ids = [
            item.configuration_id
            for item in self.allowed_configurations
        ]

        if len(configuration_ids) != len(
            set(configuration_ids)
        ):
            raise ValueError(
                "Candidate configuration_id values must be unique."
            )

        return self


class InferenceSearchSpace(BaseModel):
    """A bounded search space containing compatible candidates only."""

    search_space_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    )

    version: str = Field(
        min_length=1,
        pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$",
    )

    task_family: str = Field(
        min_length=1,
    )

    candidates: tuple[
        InferenceCandidate,
        ...,
    ] = Field(
        min_length=1,
    )

    metadata: dict[str, JsonValue] = Field(
        default_factory=dict,
    )

    @model_validator(mode="after")
    def validate_candidate_ids(
        self,
    ) -> Self:
        """Require unique candidate identities within one task family."""
        candidate_ids = [
            candidate.candidate_id
            for candidate in self.candidates
        ]

        if len(candidate_ids) != len(
            set(candidate_ids)
        ):
            raise ValueError(
                "Search-space candidate_id values must be unique."
            )

        return self


class InferenceSearchPoint(BaseModel):
    """One explicit candidate/configuration pair eligible for evaluation."""

    search_point_id: str = Field(
        min_length=1,
    )

    search_space_id: str = Field(
        min_length=1,
    )

    task_family: str = Field(
        min_length=1,
    )

    candidate_id: str = Field(
        min_length=1,
    )

    implementation_id: str = Field(
        min_length=1,
    )

    configuration_id: str = Field(
        min_length=1,
    )

    configuration: InferenceConfiguration

    requires_verification: bool

    candidate_metadata: dict[str, JsonValue] = Field(
        default_factory=dict,
    )

    configuration_metadata: dict[str, JsonValue] = Field(
        default_factory=dict,
    )


def enumerate_search_points(
    search_space: InferenceSearchSpace,
) -> tuple[InferenceSearchPoint, ...]:
    """Enumerate only explicitly registered candidate/configuration pairs."""
    points: list[InferenceSearchPoint] = []

    for candidate in sorted(
        search_space.candidates,
        key=lambda item: item.candidate_id,
    ):
        for candidate_configuration in sorted(
            candidate.allowed_configurations,
            key=lambda item: item.configuration_id,
        ):
            points.append(
                InferenceSearchPoint(
                    search_point_id=(
                        f"{candidate.candidate_id}::"
                        f"{candidate_configuration.configuration_id}"
                    ),
                    search_space_id=(
                        search_space.search_space_id
                    ),
                    task_family=search_space.task_family,
                    candidate_id=candidate.candidate_id,
                    implementation_id=(
                        candidate.implementation_id
                    ),
                    configuration_id=(
                        candidate_configuration.configuration_id
                    ),
                    configuration=(
                        candidate_configuration.configuration
                    ),
                    requires_verification=(
                        candidate.requires_verification
                    ),
                    candidate_metadata=dict(
                        candidate.metadata
                    ),
                    configuration_metadata=dict(
                        candidate_configuration.metadata
                    ),
                )
            )

    return tuple(points)
