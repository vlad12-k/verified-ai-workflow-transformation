"""Build synthetic auditable evidence for M4-J recommendation planning."""

from pathlib import Path

from pydantic import BaseModel

from vait.decision.models import Decision
from vait.inference.benchmark import (
    capture_inference_environment,
)
from vait.inference.models import (
    InferenceConfiguration,
    InferenceEnvironment,
)
from vait.optimisation.admissibility import (
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
    TransformationRecommendationPlan,
    build_transformation_recommendation_plan,
)
from vait.optimisation.search_space import (
    InferenceSearchPoint,
)

ARTIFACT_PATH = Path(
    "artifacts/benchmarks/"
    "m4j-recommendation-plan-v0.1.json"
)


class M4JRecommendationArtifact(BaseModel):
    """Versioned synthetic evidence for M4-J planning semantics."""

    schema_version: str = "0.1"

    artifact_id: str
    evidence_scope: str
    performance_claim: bool

    environment: InferenceEnvironment

    ready_plan: TransformationRecommendationPlan
    unresolved_tie_plan: TransformationRecommendationPlan
    no_feasible_plan: TransformationRecommendationPlan


def build_point(
    candidate_id: str,
) -> InferenceSearchPoint:
    """Build one synthetic verified recommendation candidate."""
    return InferenceSearchPoint(
        search_point_id=(
            f"{candidate_id}::cpu"
        ),
        search_space_id="m4j-synthetic-space",
        task_family="structured-decision",
        candidate_id=candidate_id,
        implementation_id=(
            f"{candidate_id}-v1"
        ),
        configuration_id="cpu",
        configuration=InferenceConfiguration(
            provider="local",
            runtime="synthetic",
            device="cpu",
            dtype="float32",
            batch_size=1,
        ),
        requires_verification=True,
    )


def build_admissibility(
    point: InferenceSearchPoint,
) -> SearchPointAdmissibility:
    """Build explicit deterministic verification-backed admission."""
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
                    f"synthetic-contract:"
                    f"{point.search_point_id}"
                ),
                decision=Decision.EXACT,
                evidence_kind=(
                    VerificationEvidenceKind
                    .DETERMINISTIC
                ),
            )
        ),
    )


def build_analysis(
    *,
    points: tuple[
        InferenceSearchPoint,
        ...,
    ],
    frontier: tuple[str, ...],
    preferred: tuple[str, ...],
    dominated: tuple[str, ...] = (),
) -> MultiObjectiveAnalysis:
    """Build synthetic M4-I evidence consumed by M4-J."""
    vectors = tuple(
        CandidateObjectiveVector(
            search_point_id=(
                point.search_point_id
            ),
            implementation_id=(
                point.implementation_id
            ),
            task_family=point.task_family,
            values={
                ObjectiveMetric.MEAN_LATENCY_MS: (
                    float(index + 1)
                ),
            },
        )
        for index, point in enumerate(
            points
        )
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
            workload_id=(
                "m4j-synthetic-workload"
            ),
            workload_version="0.1",
            workload_fingerprint_sha256=(
                "b" * 64
            ),
            measurement_protocol_id=(
                "m4j-synthetic-protocol-v1"
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
        preference_policy=(
            preference_policy
        ),
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


def main() -> None:
    """Build recommendation plans without executing transformations."""
    balanced = build_point(
        "balanced"
    )
    compact = build_point(
        "compact"
    )

    admissibility = {
        point.search_point_id: (
            build_admissibility(
                point
            )
        )
        for point in (
            balanced,
            compact,
        )
    }

    ready_analysis = build_analysis(
        points=(
            balanced,
            compact,
        ),
        frontier=(
            balanced.search_point_id,
            compact.search_point_id,
        ),
        preferred=(
            compact.search_point_id,
        ),
    )

    ready_plan = (
        build_transformation_recommendation_plan(
            analysis=ready_analysis,
            admissibility_by_search_point=(
                admissibility
            ),
        )
    )

    tie_analysis = build_analysis(
        points=(
            balanced,
            compact,
        ),
        frontier=(
            balanced.search_point_id,
            compact.search_point_id,
        ),
        preferred=(
            balanced.search_point_id,
            compact.search_point_id,
        ),
    )

    unresolved_tie_plan = (
        build_transformation_recommendation_plan(
            analysis=tie_analysis,
            admissibility_by_search_point=(
                admissibility
            ),
        )
    )

    dominated = build_point(
        "dominated"
    )

    dominated_admissibility = {
        dominated.search_point_id: (
            build_admissibility(
                dominated
            )
        ),
    }

    no_feasible_analysis = build_analysis(
        points=(
            dominated,
        ),
        frontier=(),
        preferred=(),
        dominated=(
            dominated.search_point_id,
        ),
    )

    no_feasible_plan = (
        build_transformation_recommendation_plan(
            analysis=(
                no_feasible_analysis
            ),
            admissibility_by_search_point=(
                dominated_admissibility
            ),
        )
    )

    for plan in (
        ready_plan,
        unresolved_tie_plan,
        no_feasible_plan,
    ):
        if plan.automatic_execution_allowed:
            raise RuntimeError(
                "M4-J artifact unexpectedly allows "
                "automatic execution."
            )

    environment = (
        capture_inference_environment()
    )

    artifact = M4JRecommendationArtifact(
        artifact_id=(
            "m4j-recommendation-plan-v0.1"
        ),
        evidence_scope=(
            "synthetic-recommendation-planning"
        ),
        performance_claim=False,
        environment=environment,
        ready_plan=ready_plan,
        unresolved_tie_plan=(
            unresolved_tie_plan
        ),
        no_feasible_plan=(
            no_feasible_plan
        ),
    )

    ARTIFACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT_PATH.write_text(
        artifact.model_dump_json(
            indent=2
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "M4-J recommendation planning evidence"
    )
    print()

    print(
        "READY:",
        ready_plan.status.value,
        ready_plan.action.value,
        ready_plan.required_next_gate.value,
    )

    print(
        "Selected:",
        ready_plan.selected_search_point_id,
    )

    print(
        "Tie:",
        unresolved_tie_plan.status.value,
        unresolved_tie_plan.action.value,
        unresolved_tie_plan.required_next_gate.value,
    )

    print(
        "No feasible:",
        no_feasible_plan.status.value,
        no_feasible_plan.action.value,
        no_feasible_plan.required_next_gate.value,
    )

    print()
    print(
        "Automatic execution:",
        False,
    )
    print(
        "Performance claim:",
        artifact.performance_claim,
    )
    print(
        "Source revision:",
        environment.source_revision,
    )
    print(
        "Source dirty:",
        environment.source_dirty,
    )
    print(
        "Evidence artifact:",
        ARTIFACT_PATH,
    )


if __name__ == "__main__":
    main()
