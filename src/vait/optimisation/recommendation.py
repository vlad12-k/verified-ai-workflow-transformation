"""Auditable recommendation planning over verified M4-I evidence."""

from collections.abc import Mapping
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, model_validator

from vait.decision.models import Decision
from vait.optimisation.admissibility import (
    SearchPointAdmissibility,
)
from vait.optimisation.multi_objective import (
    EvidenceComparabilityKey,
    MultiObjectiveAnalysis,
    ObjectiveConstraint,
    ObjectiveMetric,
    OptimisationObjective,
    ParetoPreferencePolicy,
)


class RecommendationStatus(StrEnum):
    """Outcome of recommendation planning after M4-I optimisation."""

    READY = "ready"
    NO_FEASIBLE_CANDIDATE = (
        "no_feasible_candidate"
    )
    UNRESOLVED_PREFERENCE_TIE = (
        "unresolved_preference_tie"
    )


class RecommendationAction(StrEnum):
    """Explicit non-executing action encoded by the recommendation plan."""

    APPLY_SELECTED_SEARCH_POINT = (
        "apply_selected_search_point"
    )

    HOLD_NO_FEASIBLE_CANDIDATE = (
        "hold_no_feasible_candidate"
    )

    HOLD_UNRESOLVED_PREFERENCE_TIE = (
        "hold_unresolved_preference_tie"
    )


class RecommendationNextGate(StrEnum):
    """Required gate before the recommendation may progress."""

    REVERIFY_AFTER_TRANSFORMATION = (
        "reverify_after_transformation"
    )

    REVISE_FEASIBILITY = (
        "revise_feasibility"
    )

    RESOLVE_PREFERENCE_TIE = (
        "resolve_preference_tie"
    )


class RecommendationReasonCode(StrEnum):
    """Machine-readable explanation of recommendation status."""

    UNIQUE_PREFERRED_FRONTIER = (
        "unique_preferred_frontier"
    )
    EMPTY_PREFERRED_FRONTIER = (
        "empty_preferred_frontier"
    )
    MULTIPLE_PREFERRED_FRONTIER = (
        "multiple_preferred_frontier"
    )


class TransformationRecommendationPlan(BaseModel):
    """Auditable transformation recommendation derived from M4-I."""

    schema_version: str = "0.1"

    task_family: str = Field(
        min_length=1,
    )

    comparability: EvidenceComparabilityKey

    status: RecommendationStatus

    action: RecommendationAction
    required_next_gate: RecommendationNextGate

    automatic_execution_allowed: bool = False

    selected_search_point_id: str | None = None
    selected_implementation_id: str | None = None
    selected_verification_decision: (
        Decision | None
    ) = None

    selected_objective_values: (
        dict[
            ObjectiveMetric,
            float,
        ]
        | None
    ) = None

    pareto_frontier_search_point_ids: tuple[
        str,
        ...,
    ] = ()

    preferred_frontier_search_point_ids: tuple[
        str,
        ...,
    ] = ()

    tradeoff_frontier_search_point_ids: tuple[
        str,
        ...,
    ] = ()

    objectives: tuple[
        OptimisationObjective,
        ...,
    ] = Field(
        min_length=1,
    )

    constraints: tuple[
        ObjectiveConstraint,
        ...,
    ] = ()

    preference_policy: (
        ParetoPreferencePolicy | None
    ) = None

    reason_codes: tuple[
        RecommendationReasonCode,
        ...,
    ] = Field(
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_recommendation_consistency(
        self,
    ) -> Self:
        """Prevent recommendation state from contradicting M4-I evidence."""
        frontier = set(
            self.pareto_frontier_search_point_ids
        )

        preferred = set(
            self.preferred_frontier_search_point_ids
        )

        tradeoffs = set(
            self.tradeoff_frontier_search_point_ids
        )

        if len(frontier) != len(
            self.pareto_frontier_search_point_ids
        ):
            raise ValueError(
                "Pareto frontier search-point IDs "
                "must be unique."
            )

        if len(preferred) != len(
            self.preferred_frontier_search_point_ids
        ):
            raise ValueError(
                "Preferred frontier search-point IDs "
                "must be unique."
            )

        if not preferred <= frontier:
            raise ValueError(
                "Preferred recommendation candidates "
                "must remain inside the Pareto frontier."
            )

        if not tradeoffs <= frontier:
            raise ValueError(
                "Trade-off candidates must remain "
                "inside the Pareto frontier."
            )

        selected_fields_present = (
            self.selected_search_point_id
            is not None
            or self.selected_implementation_id
            is not None
            or self.selected_objective_values
            is not None
        )

        if self.automatic_execution_allowed:
            raise ValueError(
                "Recommendation plans are evidence artifacts; "
                "automatic execution is not allowed."
            )

        if (
            self.status
            is RecommendationStatus.READY
        ):
            if (
                self.action
                is not RecommendationAction
                .APPLY_SELECTED_SEARCH_POINT
            ):
                raise ValueError(
                    "READY recommendation requires the "
                    "apply-selected-search-point action."
                )

            if (
                self.required_next_gate
                is not RecommendationNextGate
                .REVERIFY_AFTER_TRANSFORMATION
            ):
                raise ValueError(
                    "READY recommendation requires "
                    "post-transformation reverification."
                )

            if len(
                self.preferred_frontier_search_point_ids
            ) != 1:
                raise ValueError(
                    "READY recommendation requires "
                    "exactly one preferred search point."
                )

            if (
                self.selected_search_point_id
                is None
                or self.selected_implementation_id
                is None
                or self.selected_objective_values
                is None
            ):
                raise ValueError(
                    "READY recommendation requires "
                    "complete selected-candidate evidence."
                )

            if (
                self.selected_search_point_id
                != self.preferred_frontier_search_point_ids[
                    0
                ]
            ):
                raise ValueError(
                    "Selected search point must equal "
                    "the unique preferred search point."
                )

            if (
                self.selected_search_point_id
                not in frontier
            ):
                raise ValueError(
                    "Selected search point must remain "
                    "inside the Pareto frontier."
                )

            if (
                RecommendationReasonCode
                .UNIQUE_PREFERRED_FRONTIER
                not in self.reason_codes
            ):
                raise ValueError(
                    "READY recommendation requires "
                    "unique-frontier rationale."
                )

        elif (
            self.status
            is RecommendationStatus
            .NO_FEASIBLE_CANDIDATE
        ):
            if (
                self.action
                is not RecommendationAction
                .HOLD_NO_FEASIBLE_CANDIDATE
            ):
                raise ValueError(
                    "NO_FEASIBLE_CANDIDATE requires "
                    "the feasibility hold action."
                )

            if (
                self.required_next_gate
                is not RecommendationNextGate
                .REVISE_FEASIBILITY
            ):
                raise ValueError(
                    "NO_FEASIBLE_CANDIDATE requires "
                    "feasibility revision before progress."
                )

            if preferred:
                raise ValueError(
                    "NO_FEASIBLE_CANDIDATE requires "
                    "an empty preferred frontier."
                )

            if selected_fields_present:
                raise ValueError(
                    "NO_FEASIBLE_CANDIDATE cannot "
                    "contain a selected candidate."
                )

            if (
                self.selected_verification_decision
                is not None
            ):
                raise ValueError(
                    "NO_FEASIBLE_CANDIDATE cannot "
                    "contain a verification decision."
                )

            if (
                RecommendationReasonCode
                .EMPTY_PREFERRED_FRONTIER
                not in self.reason_codes
            ):
                raise ValueError(
                    "Empty preferred frontier requires "
                    "explicit rationale."
                )

        else:
            if (
                self.action
                is not RecommendationAction
                .HOLD_UNRESOLVED_PREFERENCE_TIE
            ):
                raise ValueError(
                    "UNRESOLVED_PREFERENCE_TIE requires "
                    "the preference-tie hold action."
                )

            if (
                self.required_next_gate
                is not RecommendationNextGate
                .RESOLVE_PREFERENCE_TIE
            ):
                raise ValueError(
                    "UNRESOLVED_PREFERENCE_TIE requires "
                    "preference resolution before progress."
                )

            if len(
                self.preferred_frontier_search_point_ids
            ) < 2:
                raise ValueError(
                    "UNRESOLVED_PREFERENCE_TIE requires "
                    "multiple preferred search points."
                )

            if selected_fields_present:
                raise ValueError(
                    "Unresolved preference tie cannot "
                    "contain a selected candidate."
                )

            if (
                self.selected_verification_decision
                is not None
            ):
                raise ValueError(
                    "Unresolved preference tie cannot "
                    "contain a verification decision."
                )

            if (
                RecommendationReasonCode
                .MULTIPLE_PREFERRED_FRONTIER
                not in self.reason_codes
            ):
                raise ValueError(
                    "Preference tie requires "
                    "explicit rationale."
                )

        return self


def build_transformation_recommendation_plan(
    *,
    analysis: MultiObjectiveAnalysis,
    admissibility_by_search_point: Mapping[
        str,
        SearchPointAdmissibility,
    ],
) -> TransformationRecommendationPlan:
    """Build a recommendation only from verified M4-I evidence."""
    vectors_by_id = {
        vector.search_point_id: vector
        for vector in analysis.vectors
    }

    vector_ids = set(
        vectors_by_id
    )

    admissibility_ids = set(
        admissibility_by_search_point
    )

    if vector_ids != admissibility_ids:
        missing = sorted(
            vector_ids
            - admissibility_ids
        )

        unknown = sorted(
            admissibility_ids
            - vector_ids
        )

        raise ValueError(
            "Recommendation admissibility evidence "
            "must exactly match M4-I candidates; "
            f"missing={missing}, unknown={unknown}."
        )

    validated_admissibility: dict[
        str,
        SearchPointAdmissibility,
    ] = {}

    for search_point_id in sorted(
        vector_ids
    ):
        admissibility = (
            admissibility_by_search_point[
                search_point_id
            ]
        )

        point = admissibility.search_point
        vector = vectors_by_id[
            search_point_id
        ]

        if (
            point.search_point_id
            != search_point_id
        ):
            raise ValueError(
                "Recommendation admissibility key "
                "does not match search-point identity."
            )

        if (
            point.implementation_id
            != vector.implementation_id
        ):
            raise ValueError(
                "Recommendation implementation identity "
                "does not match M4-I evidence."
            )

        if (
            point.task_family
            != analysis.task_family
        ):
            raise ValueError(
                "Recommendation task family does not "
                "match M4-I evidence."
            )

        if not admissibility.is_admissible:
            raise ValueError(
                "Recommendation planning may consume "
                "only admissible search points."
            )

        verification = (
            admissibility.verification_evidence
        )

        if (
            point.requires_verification
            and verification is None
        ):
            raise ValueError(
                "Recommendation candidate requires "
                "verification evidence."
            )

        if (
            verification is not None
            and verification.decision
            is Decision.REJECT
        ):
            raise ValueError(
                "Verification REJECT can never become "
                "a recommendation."
            )

        validated_admissibility[
            search_point_id
        ] = admissibility

    preferred = (
        analysis
        .preferred_frontier_search_point_ids
    )

    frontier = (
        analysis
        .pareto_frontier_search_point_ids
    )

    if not preferred:
        return TransformationRecommendationPlan(
            task_family=analysis.task_family,
            comparability=analysis.comparability,
            status=(
                RecommendationStatus
                .NO_FEASIBLE_CANDIDATE
            ),
            action=(
                RecommendationAction
                .HOLD_NO_FEASIBLE_CANDIDATE
            ),
            required_next_gate=(
                RecommendationNextGate
                .REVISE_FEASIBILITY
            ),
            pareto_frontier_search_point_ids=(
                frontier
            ),
            preferred_frontier_search_point_ids=(
                preferred
            ),
            objectives=analysis.objectives,
            constraints=analysis.constraints,
            preference_policy=(
                analysis.preference_policy
            ),
            tradeoff_frontier_search_point_ids=(
                frontier
            ),
            reason_codes=(
                RecommendationReasonCode
                .EMPTY_PREFERRED_FRONTIER,
            ),
        )

    if len(preferred) > 1:
        return TransformationRecommendationPlan(
            task_family=analysis.task_family,
            comparability=analysis.comparability,
            status=(
                RecommendationStatus
                .UNRESOLVED_PREFERENCE_TIE
            ),
            action=(
                RecommendationAction
                .HOLD_UNRESOLVED_PREFERENCE_TIE
            ),
            required_next_gate=(
                RecommendationNextGate
                .RESOLVE_PREFERENCE_TIE
            ),
            pareto_frontier_search_point_ids=(
                frontier
            ),
            preferred_frontier_search_point_ids=(
                preferred
            ),
            objectives=analysis.objectives,
            constraints=analysis.constraints,
            preference_policy=(
                analysis.preference_policy
            ),
            tradeoff_frontier_search_point_ids=(
                frontier
            ),
            reason_codes=(
                RecommendationReasonCode
                .MULTIPLE_PREFERRED_FRONTIER,
            ),
        )

    selected_id = preferred[
        0
    ]

    selected_vector = vectors_by_id[
        selected_id
    ]

    selected_admissibility = (
        validated_admissibility[
            selected_id
        ]
    )

    verification = (
        selected_admissibility
        .verification_evidence
    )

    selected_decision = (
        verification.decision
        if verification is not None
        else None
    )

    tradeoffs = tuple(
        search_point_id
        for search_point_id in frontier
        if search_point_id != selected_id
    )

    return TransformationRecommendationPlan(
        task_family=analysis.task_family,
        comparability=analysis.comparability,
        status=RecommendationStatus.READY,
        action=(
            RecommendationAction
            .APPLY_SELECTED_SEARCH_POINT
        ),
        required_next_gate=(
            RecommendationNextGate
            .REVERIFY_AFTER_TRANSFORMATION
        ),
        pareto_frontier_search_point_ids=(
            frontier
        ),
        preferred_frontier_search_point_ids=(
            preferred
        ),
        objectives=analysis.objectives,
        constraints=analysis.constraints,
        preference_policy=(
            analysis.preference_policy
        ),
        selected_search_point_id=selected_id,
        selected_implementation_id=(
            selected_vector.implementation_id
        ),
        selected_verification_decision=(
            selected_decision
        ),
        selected_objective_values=dict(
            selected_vector.values
        ),
        tradeoff_frontier_search_point_ids=(
            tradeoffs
        ),
        reason_codes=(
            RecommendationReasonCode
            .UNIQUE_PREFERRED_FRONTIER,
        ),
    )
