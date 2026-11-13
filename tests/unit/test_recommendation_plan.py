"""Tests for M4-J auditable transformation recommendation planning."""

import pytest

from vait.decision.models import Decision
from vait.inference.models import (
    InferenceConfiguration,
)
from vait.optimisation.admissibility import (
    AdmissibilityIssue,
    AdmissibilityIssueCode,
    AdmissibilityStatus,
    SearchPointAdmissibility,
    SearchPointVerificationEvidence,
    VerificationEvidenceKind,
)
from vait.optimisation.multi_objective import (
    CandidateObjectiveVector,
    EvidenceComparabilityKey,
    MultiObjectiveAnalysis,
    ObjectiveDirection,
    ObjectiveMetric,
    OptimisationObjective,
    ParetoPreferencePolicy,
    PreferencePriority,
)
from vait.optimisation.recommendation import (
    RecommendationAction,
    RecommendationNextGate,
    RecommendationReasonCode,
    RecommendationStatus,
    TransformationRecommendationPlan,
    build_transformation_recommendation_plan,
)
from vait.optimisation.search_space import (
    InferenceSearchPoint,
)


def _point(
    candidate_id: str,
) -> InferenceSearchPoint:
    """Build one verified synthetic M4-J search point."""
    return InferenceSearchPoint(
        search_point_id=(
            f"{candidate_id}::cpu"
        ),
        search_space_id="m4j-test",
        task_family="structured-decision",
        candidate_id=candidate_id,
        implementation_id=(
            f"{candidate_id}-v1"
        ),
        configuration_id="cpu",
        configuration=InferenceConfiguration(
            provider="local",
            runtime="python",
            device="cpu",
            dtype="native",
            batch_size=1,
        ),
        requires_verification=True,
    )


def _admissible(
    point: InferenceSearchPoint,
    *,
    decision: Decision = Decision.EXACT,
) -> SearchPointAdmissibility:
    """Build explicit verification-backed admissibility evidence."""
    return SearchPointAdmissibility(
        search_point=point,
        status=AdmissibilityStatus.ADMISSIBLE,
        verification_evidence=(
            SearchPointVerificationEvidence(
                search_point_id=(
                    point.search_point_id
                ),
                implementation_id=(
                    point.implementation_id
                ),
                contract_id=(
                    f"contract:{point.search_point_id}"
                ),
                decision=decision,
                evidence_kind=(
                    VerificationEvidenceKind
                    .DETERMINISTIC
                ),
            )
        ),
    )


def _analysis(
    *,
    preferred: tuple[str, ...],
    frontier: tuple[str, ...],
    dominated: tuple[str, ...] = (),
) -> MultiObjectiveAnalysis:
    """Build one deterministic M4-I analysis for M4-J tests."""
    all_ids = (
        frontier
        + dominated
    )

    vectors = tuple(
        CandidateObjectiveVector(
            search_point_id=search_point_id,
            implementation_id=(
                search_point_id.split(
                    "::",
                    maxsplit=1,
                )[0]
                + "-v1"
            ),
            task_family="structured-decision",
            values={
                ObjectiveMetric.MEAN_LATENCY_MS: (
                    float(index + 1)
                ),
            },
        )
        for index, search_point_id
        in enumerate(all_ids)
    )

    preference_policy = (
        ParetoPreferencePolicy(
            priorities=(
                PreferencePriority(
                    metric=(
                        ObjectiveMetric
                        .MEAN_LATENCY_MS
                    ),
                ),
            ),
        )
        if preferred != frontier
        else None
    )

    return MultiObjectiveAnalysis(
        task_family="structured-decision",
        comparability=EvidenceComparabilityKey(
            workload_id="m4j-workload",
            workload_version="0.1",
            workload_fingerprint_sha256=(
                "a" * 64
            ),
            measurement_protocol_id=(
                "structured-scalar-v1"
            ),
        ),
        objectives=(
            OptimisationObjective(
                metric=(
                    ObjectiveMetric
                    .MEAN_LATENCY_MS
                ),
                direction=(
                    ObjectiveDirection.MINIMIZE
                ),
            ),
        ),
        preference_policy=preference_policy,
        vectors=vectors,
        pareto_frontier_search_point_ids=(
            frontier
        ),
        preferred_frontier_search_point_ids=(
            preferred
        ),
        dominated_search_point_ids=(
            dominated
        ),
    )


def _admissibility_map(
    *candidate_ids: str,
) -> dict[
    str,
    SearchPointAdmissibility,
]:
    """Build admissibility evidence keyed by search-point ID."""
    result: dict[
        str,
        SearchPointAdmissibility,
    ] = {}

    for candidate_id in candidate_ids:
        point = _point(
            candidate_id
        )

        result[
            point.search_point_id
        ] = _admissible(
            point
        )

    return result


def test_unique_preferred_candidate_builds_ready_plan() -> None:
    """A single preferred Pareto point may become recommendation-ready."""
    analysis = _analysis(
        frontier=(
            "balanced::cpu",
            "compact::cpu",
        ),
        preferred=(
            "compact::cpu",
        ),
    )

    plan = (
        build_transformation_recommendation_plan(
            analysis=analysis,
            admissibility_by_search_point=(
                _admissibility_map(
                    "balanced",
                    "compact",
                )
            ),
        )
    )

    assert (
        plan.status
        is RecommendationStatus.READY
    )

    assert (
        plan.selected_search_point_id
        == "compact::cpu"
    )

    assert (
        plan.selected_implementation_id
        == "compact-v1"
    )

    assert (
        plan.selected_verification_decision
        is Decision.EXACT
    )

    assert (
        plan.tradeoff_frontier_search_point_ids
        == ("balanced::cpu",)
    )

    assert (
        plan.reason_codes
        == (
            RecommendationReasonCode
            .UNIQUE_PREFERRED_FRONTIER,
        )
    )


def test_empty_preferred_frontier_produces_no_feasible_plan() -> None:
    """No feasible M4-I point must not be converted into a recommendation."""
    analysis = _analysis(
        frontier=(),
        preferred=(),
        dominated=(
            "dominated::cpu",
        ),
    )

    plan = (
        build_transformation_recommendation_plan(
            analysis=analysis,
            admissibility_by_search_point=(
                _admissibility_map(
                    "dominated",
                )
            ),
        )
    )

    assert (
        plan.status
        is RecommendationStatus
        .NO_FEASIBLE_CANDIDATE
    )

    assert (
        plan.selected_search_point_id
        is None
    )

    assert (
        plan.selected_implementation_id
        is None
    )

    assert (
        plan.selected_objective_values
        is None
    )


def test_multiple_preferred_candidates_remain_unresolved() -> None:
    """M4-J must not silently break a remaining M4-I preference tie."""
    analysis = _analysis(
        frontier=(
            "balanced::cpu",
            "compact::cpu",
        ),
        preferred=(
            "balanced::cpu",
            "compact::cpu",
        ),
    )

    plan = (
        build_transformation_recommendation_plan(
            analysis=analysis,
            admissibility_by_search_point=(
                _admissibility_map(
                    "balanced",
                    "compact",
                )
            ),
        )
    )

    assert (
        plan.status
        is RecommendationStatus
        .UNRESOLVED_PREFERENCE_TIE
    )

    assert (
        plan.selected_search_point_id
        is None
    )

    assert (
        plan.selected_verification_decision
        is None
    )


def test_forged_admissible_reject_can_never_be_recommended() -> None:
    """Defense in depth blocks REJECT even under forged admissibility."""
    point = _point(
        "rejected"
    )

    forged = _admissible(
        point,
        decision=Decision.REJECT,
    )

    analysis = _analysis(
        frontier=(
            "rejected::cpu",
        ),
        preferred=(
            "rejected::cpu",
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "REJECT can never become "
            "a recommendation"
        ),
    ):
        build_transformation_recommendation_plan(
            analysis=analysis,
            admissibility_by_search_point={
                point.search_point_id: forged,
            },
        )


def test_recommendation_requires_exact_candidate_evidence_set() -> None:
    """Stale or incomplete admissibility maps must not drive planning."""
    analysis = _analysis(
        frontier=(
            "balanced::cpu",
            "compact::cpu",
        ),
        preferred=(
            "compact::cpu",
        ),
    )

    with pytest.raises(
        ValueError,
        match="must exactly match M4-I candidates",
    ):
        build_transformation_recommendation_plan(
            analysis=analysis,
            admissibility_by_search_point=(
                _admissibility_map(
                    "compact",
                )
            ),
        )


def test_recommendation_actions_and_execution_boundary() -> None:
    """Each M4-J status has an explicit non-executing next action."""
    ready_analysis = _analysis(
        frontier=(
            "compact::cpu",
        ),
        preferred=(
            "compact::cpu",
        ),
    )

    ready = (
        build_transformation_recommendation_plan(
            analysis=ready_analysis,
            admissibility_by_search_point=(
                _admissibility_map(
                    "compact",
                )
            ),
        )
    )

    assert (
        ready.action
        is RecommendationAction
        .APPLY_SELECTED_SEARCH_POINT
    )
    assert (
        ready.required_next_gate
        is RecommendationNextGate
        .REVERIFY_AFTER_TRANSFORMATION
    )
    assert (
        ready.automatic_execution_allowed
        is False
    )

    empty_analysis = _analysis(
        frontier=(),
        preferred=(),
        dominated=(
            "dominated::cpu",
        ),
    )

    empty = (
        build_transformation_recommendation_plan(
            analysis=empty_analysis,
            admissibility_by_search_point=(
                _admissibility_map(
                    "dominated",
                )
            ),
        )
    )

    assert (
        empty.action
        is RecommendationAction
        .HOLD_NO_FEASIBLE_CANDIDATE
    )
    assert (
        empty.required_next_gate
        is RecommendationNextGate
        .REVISE_FEASIBILITY
    )

    tie_analysis = _analysis(
        frontier=(
            "balanced::cpu",
            "compact::cpu",
        ),
        preferred=(
            "balanced::cpu",
            "compact::cpu",
        ),
    )

    tie = (
        build_transformation_recommendation_plan(
            analysis=tie_analysis,
            admissibility_by_search_point=(
                _admissibility_map(
                    "balanced",
                    "compact",
                )
            ),
        )
    )

    assert (
        tie.action
        is RecommendationAction
        .HOLD_UNRESOLVED_PREFERENCE_TIE
    )
    assert (
        tie.required_next_gate
        is RecommendationNextGate
        .RESOLVE_PREFERENCE_TIE
    )

    payload = ready.model_dump(
        mode="python"
    )
    payload[
        "automatic_execution_allowed"
    ] = True

    with pytest.raises(
        ValueError,
        match="automatic execution is not allowed",
    ):
        TransformationRecommendationPlan.model_validate(
            payload
        )


def _ready_plan() -> TransformationRecommendationPlan:
    """Build one valid READY recommendation used for mutation tests."""
    analysis = _analysis(
        frontier=(
            "compact::cpu",
        ),
        preferred=(
            "compact::cpu",
        ),
    )

    return build_transformation_recommendation_plan(
        analysis=analysis,
        admissibility_by_search_point=(
            _admissibility_map(
                "compact",
            )
        ),
    )


def _empty_plan() -> TransformationRecommendationPlan:
    """Build one valid NO_FEASIBLE_CANDIDATE recommendation."""
    analysis = _analysis(
        frontier=(),
        preferred=(),
        dominated=(
            "dominated::cpu",
        ),
    )

    return build_transformation_recommendation_plan(
        analysis=analysis,
        admissibility_by_search_point=(
            _admissibility_map(
                "dominated",
            )
        ),
    )


def _tie_plan() -> TransformationRecommendationPlan:
    """Build one valid unresolved preference-tie recommendation."""
    analysis = _analysis(
        frontier=(
            "balanced::cpu",
            "compact::cpu",
        ),
        preferred=(
            "balanced::cpu",
            "compact::cpu",
        ),
    )

    return build_transformation_recommendation_plan(
        analysis=analysis,
        admissibility_by_search_point=(
            _admissibility_map(
                "balanced",
                "compact",
            )
        ),
    )


def test_plan_rejects_duplicate_pareto_ids() -> None:
    """Pareto frontier identity must remain unique."""
    payload = _ready_plan().model_dump(
        mode="python"
    )

    payload[
        "pareto_frontier_search_point_ids"
    ] = (
        "compact::cpu",
        "compact::cpu",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Pareto frontier search-point IDs "
            "must be unique"
        ),
    ):
        TransformationRecommendationPlan.model_validate(
            payload
        )


def test_plan_rejects_duplicate_preferred_ids() -> None:
    """Preferred frontier identity must remain unique."""
    payload = _ready_plan().model_dump(
        mode="python"
    )

    payload[
        "preferred_frontier_search_point_ids"
    ] = (
        "compact::cpu",
        "compact::cpu",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Preferred frontier search-point IDs "
            "must be unique"
        ),
    ):
        TransformationRecommendationPlan.model_validate(
            payload
        )


def test_preferred_candidates_must_remain_on_pareto_frontier() -> None:
    """A recommendation preference cannot escape the Pareto frontier."""
    payload = _ready_plan().model_dump(
        mode="python"
    )

    payload[
        "preferred_frontier_search_point_ids"
    ] = (
        "outside::cpu",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Preferred recommendation candidates "
            "must remain inside the Pareto frontier"
        ),
    ):
        TransformationRecommendationPlan.model_validate(
            payload
        )


def test_tradeoffs_must_remain_on_pareto_frontier() -> None:
    """Trade-off evidence cannot name candidates outside the frontier."""
    payload = _ready_plan().model_dump(
        mode="python"
    )

    payload[
        "tradeoff_frontier_search_point_ids"
    ] = (
        "outside::cpu",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Trade-off candidates must remain "
            "inside the Pareto frontier"
        ),
    ):
        TransformationRecommendationPlan.model_validate(
            payload
        )


@pytest.mark.parametrize(
    (
        "updates",
        "message",
    ),
    [
        (
            {
                "action": (
                    RecommendationAction
                    .HOLD_NO_FEASIBLE_CANDIDATE
                ),
            },
            "READY recommendation requires the "
            "apply-selected-search-point action",
        ),
        (
            {
                "required_next_gate": (
                    RecommendationNextGate
                    .REVISE_FEASIBILITY
                ),
            },
            "READY recommendation requires "
            "post-transformation reverification",
        ),
        (
            {
                "pareto_frontier_search_point_ids": (
                    "compact::cpu",
                    "balanced::cpu",
                ),
                "preferred_frontier_search_point_ids": (
                    "compact::cpu",
                    "balanced::cpu",
                ),
            },
            "READY recommendation requires exactly "
            "one preferred search point",
        ),
        (
            {
                "selected_objective_values": None,
            },
            "READY recommendation requires complete "
            "selected-candidate evidence",
        ),
        (
            {
                "selected_search_point_id": (
                    "different::cpu"
                ),
            },
            "Selected search point must equal "
            "the unique preferred search point",
        ),
        (
            {
                "reason_codes": (
                    RecommendationReasonCode
                    .EMPTY_PREFERRED_FRONTIER,
                ),
            },
            "READY recommendation requires "
            "unique-frontier rationale",
        ),
    ],
)
def test_ready_plan_rejects_contradictory_state(
    updates: dict[str, object],
    message: str,
) -> None:
    """READY state must remain consistent with its evidence contract."""
    payload = _ready_plan().model_dump(
        mode="python"
    )

    payload.update(updates)

    with pytest.raises(
        ValueError,
        match=message,
    ):
        TransformationRecommendationPlan.model_validate(
            payload
        )


@pytest.mark.parametrize(
    (
        "updates",
        "message",
    ),
    [
        (
            {
                "action": (
                    RecommendationAction
                    .APPLY_SELECTED_SEARCH_POINT
                ),
            },
            "NO_FEASIBLE_CANDIDATE requires "
            "the feasibility hold action",
        ),
        (
            {
                "required_next_gate": (
                    RecommendationNextGate
                    .REVERIFY_AFTER_TRANSFORMATION
                ),
            },
            "NO_FEASIBLE_CANDIDATE requires "
            "feasibility revision before progress",
        ),
        (
            {
                "pareto_frontier_search_point_ids": (
                    "compact::cpu",
                ),
                "preferred_frontier_search_point_ids": (
                    "compact::cpu",
                ),
            },
            "NO_FEASIBLE_CANDIDATE requires "
            "an empty preferred frontier",
        ),
        (
            {
                "selected_search_point_id": (
                    "compact::cpu"
                ),
            },
            "NO_FEASIBLE_CANDIDATE cannot "
            "contain a selected candidate",
        ),
        (
            {
                "selected_verification_decision": (
                    Decision.EXACT
                ),
            },
            "NO_FEASIBLE_CANDIDATE cannot "
            "contain a verification decision",
        ),
        (
            {
                "reason_codes": (
                    RecommendationReasonCode
                    .MULTIPLE_PREFERRED_FRONTIER,
                ),
            },
            "Empty preferred frontier requires "
            "explicit rationale",
        ),
    ],
)
def test_no_feasible_plan_rejects_contradictory_state(
    updates: dict[str, object],
    message: str,
) -> None:
    """No-feasible state must never smuggle in a selected candidate."""
    payload = _empty_plan().model_dump(
        mode="python"
    )

    payload.update(updates)

    with pytest.raises(
        ValueError,
        match=message,
    ):
        TransformationRecommendationPlan.model_validate(
            payload
        )


@pytest.mark.parametrize(
    (
        "updates",
        "message",
    ),
    [
        (
            {
                "action": (
                    RecommendationAction
                    .APPLY_SELECTED_SEARCH_POINT
                ),
            },
            "UNRESOLVED_PREFERENCE_TIE requires "
            "the preference-tie hold action",
        ),
        (
            {
                "required_next_gate": (
                    RecommendationNextGate
                    .REVERIFY_AFTER_TRANSFORMATION
                ),
            },
            "UNRESOLVED_PREFERENCE_TIE requires "
            "preference resolution before progress",
        ),
        (
            {
                "preferred_frontier_search_point_ids": (
                    "compact::cpu",
                ),
            },
            "UNRESOLVED_PREFERENCE_TIE requires "
            "multiple preferred search points",
        ),
        (
            {
                "selected_search_point_id": (
                    "compact::cpu"
                ),
            },
            "Unresolved preference tie cannot "
            "contain a selected candidate",
        ),
        (
            {
                "selected_verification_decision": (
                    Decision.EXACT
                ),
            },
            "Unresolved preference tie cannot "
            "contain a verification decision",
        ),
        (
            {
                "reason_codes": (
                    RecommendationReasonCode
                    .EMPTY_PREFERRED_FRONTIER,
                ),
            },
            "Preference tie requires explicit rationale",
        ),
    ],
)
def test_preference_tie_rejects_contradictory_state(
    updates: dict[str, object],
    message: str,
) -> None:
    """Unresolved ties must remain non-executing and unselected."""
    payload = _tie_plan().model_dump(
        mode="python"
    )

    payload.update(updates)

    with pytest.raises(
        ValueError,
        match=message,
    ):
        TransformationRecommendationPlan.model_validate(
            payload
        )


def test_builder_rejects_admissibility_key_identity_mismatch() -> None:
    """Mapping keys must identify the same search point as their evidence."""
    point = _point(
        "compact"
    ).model_copy(
        update={
            "search_point_id": "forged::cpu",
        }
    )

    analysis = _analysis(
        frontier=(
            "compact::cpu",
        ),
        preferred=(
            "compact::cpu",
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "admissibility key does not match "
            "search-point identity"
        ),
    ):
        build_transformation_recommendation_plan(
            analysis=analysis,
            admissibility_by_search_point={
                "compact::cpu": _admissible(
                    point
                ),
            },
        )


def test_builder_rejects_implementation_identity_mismatch() -> None:
    """Recommendation evidence must preserve implementation identity."""
    point = _point(
        "compact"
    ).model_copy(
        update={
            "implementation_id": "forged-v1",
        }
    )

    analysis = _analysis(
        frontier=(
            "compact::cpu",
        ),
        preferred=(
            "compact::cpu",
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "implementation identity does not "
            "match M4-I evidence"
        ),
    ):
        build_transformation_recommendation_plan(
            analysis=analysis,
            admissibility_by_search_point={
                "compact::cpu": _admissible(
                    point
                ),
            },
        )


def test_builder_rejects_task_family_mismatch() -> None:
    """Recommendation evidence cannot cross task-family boundaries."""
    point = _point(
        "compact"
    ).model_copy(
        update={
            "task_family": "other-task-family",
        }
    )

    analysis = _analysis(
        frontier=(
            "compact::cpu",
        ),
        preferred=(
            "compact::cpu",
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "task family does not match "
            "M4-I evidence"
        ),
    ):
        build_transformation_recommendation_plan(
            analysis=analysis,
            admissibility_by_search_point={
                "compact::cpu": _admissible(
                    point
                ),
            },
        )


def test_builder_rejects_non_admissible_search_point() -> None:
    """Performance planning must not consume a rejected admissibility state."""
    point = _point(
        "compact"
    )

    not_admissible = SearchPointAdmissibility(
        search_point=point,
        status=(
            AdmissibilityStatus
            .NOT_ADMISSIBLE
        ),
        issues=(
            AdmissibilityIssue(
                code=(
                    AdmissibilityIssueCode
                    .VERIFICATION_REJECTED
                ),
                subject=point.search_point_id,
                message="rejected for test",
            ),
        ),
    )

    analysis = _analysis(
        frontier=(
            "compact::cpu",
        ),
        preferred=(
            "compact::cpu",
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "only admissible search points"
        ),
    ):
        build_transformation_recommendation_plan(
            analysis=analysis,
            admissibility_by_search_point={
                point.search_point_id: (
                    not_admissible
                ),
            },
        )


def test_builder_requires_verification_evidence_when_declared() -> None:
    """A verification-required point cannot enter recommendation bare."""
    point = _point(
        "compact"
    )

    forged_admissible = SearchPointAdmissibility(
        search_point=point,
        status=AdmissibilityStatus.ADMISSIBLE,
    )

    analysis = _analysis(
        frontier=(
            "compact::cpu",
        ),
        preferred=(
            "compact::cpu",
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "candidate requires verification evidence"
        ),
    ):
        build_transformation_recommendation_plan(
            analysis=analysis,
            admissibility_by_search_point={
                point.search_point_id: (
                    forged_admissible
                ),
            },
        )


def test_ready_plan_without_required_verification_records_no_decision() -> None:
    """Optional verification must not invent a verification decision."""
    point = _point(
        "compact"
    ).model_copy(
        update={
            "requires_verification": False,
        }
    )

    admissibility = SearchPointAdmissibility(
        search_point=point,
        status=AdmissibilityStatus.ADMISSIBLE,
    )

    analysis = _analysis(
        frontier=(
            "compact::cpu",
        ),
        preferred=(
            "compact::cpu",
        ),
    )

    plan = build_transformation_recommendation_plan(
        analysis=analysis,
        admissibility_by_search_point={
            point.search_point_id: admissibility,
        },
    )

    assert plan.status is RecommendationStatus.READY
    assert plan.selected_verification_decision is None
