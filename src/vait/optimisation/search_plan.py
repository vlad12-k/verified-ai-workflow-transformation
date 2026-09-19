"""Executable plans for bounded verified inference search spaces."""

from collections.abc import Mapping
from typing import Self

from pydantic import BaseModel, Field, model_validator

from vait.optimisation.admissibility import (
    SearchPointAdmissibility,
    SearchPointVerificationEvidence,
    evaluate_search_point_admissibility,
)
from vait.optimisation.compatibility import (
    InferenceCompatibilityContext,
    SearchPointCompatibility,
    evaluate_search_point_compatibility,
)
from vait.optimisation.search_space import (
    InferenceSearchPoint,
    InferenceSearchSpace,
    enumerate_search_points,
)


class ExecutableSearchPlan(BaseModel):
    """Deterministic partition of one bounded inference search space."""

    search_space_id: str = Field(min_length=1)
    search_space_version: str = Field(min_length=1)
    task_family: str = Field(min_length=1)

    total_registered_points: int = Field(
        ge=1,
    )

    admissible: tuple[
        SearchPointAdmissibility,
        ...,
    ] = ()

    compatibility_rejections: tuple[
        SearchPointCompatibility,
        ...,
    ] = ()

    admissibility_rejections: tuple[
        SearchPointAdmissibility,
        ...,
    ] = ()

    @model_validator(mode="after")
    def validate_partition(
        self,
    ) -> Self:
        """Require a complete non-overlapping partition of search points."""
        admissible_ids = [
            result.search_point.search_point_id
            for result in self.admissible
        ]

        compatibility_rejection_ids = [
            result.search_point.search_point_id
            for result in self.compatibility_rejections
        ]

        admissibility_rejection_ids = [
            result.search_point.search_point_id
            for result in self.admissibility_rejections
        ]

        all_ids = (
            admissible_ids
            + compatibility_rejection_ids
            + admissibility_rejection_ids
        )

        if len(all_ids) != self.total_registered_points:
            raise ValueError(
                "Search-plan partition does not cover every "
                "registered search point."
            )

        if len(all_ids) != len(set(all_ids)):
            raise ValueError(
                "A search point cannot appear in multiple "
                "search-plan partitions."
            )

        if any(
            not result.is_admissible
            for result in self.admissible
        ):
            raise ValueError(
                "The admissible partition may contain only "
                "admissible search points."
            )

        if any(
            result.is_admissible
            for result in self.admissibility_rejections
        ):
            raise ValueError(
                "The admissibility-rejection partition may "
                "contain only rejected search points."
            )

        if any(
            result.is_compatible
            for result in self.compatibility_rejections
        ):
            raise ValueError(
                "The compatibility-rejection partition may "
                "contain only incompatible search points."
            )

        return self

    @property
    def admissible_points(
        self,
    ) -> tuple[InferenceSearchPoint, ...]:
        """Return only points permitted to enter benchmarking."""
        return tuple(
            result.search_point
            for result in self.admissible
        )


def build_executable_search_plan(
    *,
    search_space: InferenceSearchSpace,
    compatibility_context: InferenceCompatibilityContext,
    verification_evidence_by_search_point: Mapping[
        str,
        SearchPointVerificationEvidence,
    ]
    | None = None,
) -> ExecutableSearchPlan:
    """Build a deterministic VERIFY-FIRST executable search plan."""
    search_points = enumerate_search_points(
        search_space
    )

    evidence_by_id = dict(
        verification_evidence_by_search_point
        or {}
    )

    registered_ids = {
        point.search_point_id
        for point in search_points
    }

    unknown_evidence_ids = sorted(
        set(evidence_by_id)
        - registered_ids
    )

    if unknown_evidence_ids:
        joined_ids = ", ".join(
            unknown_evidence_ids
        )

        raise ValueError(
            "Verification evidence references unknown "
            f"search points: {joined_ids}."
        )

    admissible: list[
        SearchPointAdmissibility
    ] = []

    compatibility_rejections: list[
        SearchPointCompatibility
    ] = []

    admissibility_rejections: list[
        SearchPointAdmissibility
    ] = []

    for search_point in search_points:
        compatibility = (
            evaluate_search_point_compatibility(
                search_point,
                compatibility_context,
            )
        )

        if not compatibility.is_compatible:
            compatibility_rejections.append(
                compatibility
            )
            continue

        admissibility = (
            evaluate_search_point_admissibility(
                compatibility=compatibility,
                verification_evidence=(
                    evidence_by_id.get(
                        search_point.search_point_id
                    )
                ),
            )
        )

        if admissibility.is_admissible:
            admissible.append(
                admissibility
            )
        else:
            admissibility_rejections.append(
                admissibility
            )

    return ExecutableSearchPlan(
        search_space_id=(
            search_space.search_space_id
        ),
        search_space_version=(
            search_space.version
        ),
        task_family=(
            search_space.task_family
        ),
        total_registered_points=len(
            search_points
        ),
        admissible=tuple(
            admissible
        ),
        compatibility_rejections=tuple(
            compatibility_rejections
        ),
        admissibility_rejections=tuple(
            admissibility_rejections
        ),
    )
