"""Pareto multi-objective analysis over verified inference evidence."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, model_validator

from vait.optimisation.admissibility import (
    SearchPointAdmissibility,
)
from vait.optimisation.structured_benchmark import (
    StructuredBenchmarkSeries,
)


class ObjectiveDirection(StrEnum):
    """Whether lower or higher values are preferred."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ConstraintOperator(StrEnum):
    """Hard feasibility comparison applied before Pareto analysis."""

    AT_MOST = "at_most"
    AT_LEAST = "at_least"


class ObjectiveMetric(StrEnum):
    """Supported evidence dimensions for M4-I optimisation."""

    MEAN_LATENCY_MS = "mean_latency_ms"
    MEAN_CASES_PER_SECOND = "mean_cases_per_second"
    MEAN_PROCESS_CPU_TIME_MS = (
        "mean_process_cpu_time_ms"
    )
    MAX_PROCESS_PEAK_RSS_BYTES = (
        "max_process_peak_rss_bytes"
    )
    MODEL_SIZE_BYTES = "model_size_bytes"


class OptimisationObjective(BaseModel):
    """One explicit optimisation objective."""

    metric: ObjectiveMetric
    direction: ObjectiveDirection


class EvidenceComparabilityKey(BaseModel):
    """Workload and measurement protocol required for fair comparison."""

    workload_id: str = Field(
        min_length=1,
    )

    workload_version: str = Field(
        min_length=1,
    )

    workload_fingerprint_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    measurement_protocol_id: str = Field(
        min_length=1,
    )


class ObjectiveConstraint(BaseModel):
    """One declared hard feasibility constraint."""

    metric: ObjectiveMetric
    operator: ConstraintOperator

    threshold: float = Field(
        ge=0.0,
        allow_inf_nan=False,
    )


class ConstraintViolation(BaseModel):
    """Observed evidence that violated one declared constraint."""

    metric: ObjectiveMetric
    operator: ConstraintOperator

    threshold: float = Field(
        ge=0.0,
        allow_inf_nan=False,
    )

    observed_value: float = Field(
        ge=0.0,
        allow_inf_nan=False,
    )


class ConstraintRejectedCandidate(BaseModel):
    """Candidate excluded from Pareto analysis by hard constraints."""

    search_point_id: str = Field(
        min_length=1,
    )

    violations: tuple[
        ConstraintViolation,
        ...,
    ] = Field(
        min_length=1,
    )


class PreferencePriority(BaseModel):
    """One ordered preference applied only within the Pareto frontier."""

    metric: ObjectiveMetric

    relative_tolerance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )


class ParetoPreferencePolicy(BaseModel):
    """Explicit lexicographic priorities that preserve tolerated ties."""

    priorities: tuple[
        PreferencePriority,
        ...,
    ] = Field(
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_unique_priorities(
        self,
    ) -> Self:
        """A metric may appear only once in ordered preferences."""
        metrics = [
            priority.metric
            for priority in self.priorities
        ]

        if len(metrics) != len(
            set(metrics)
        ):
            raise ValueError(
                "Preference-priority metrics must be unique."
            )

        return self


class MultiObjectiveCandidateEvidence(BaseModel):
    """Verified benchmark evidence eligible for optimisation."""

    admissibility: SearchPointAdmissibility
    benchmark: StructuredBenchmarkSeries
    comparability: EvidenceComparabilityKey

    @model_validator(mode="after")
    def validate_evidence_identity(
        self,
    ) -> Self:
        """Require admissible, identity-aligned benchmark evidence."""
        if not self.admissibility.is_admissible:
            raise ValueError(
                "Multi-objective optimisation requires "
                "an admissible search point."
            )

        point = (
            self.admissibility.search_point
        )

        if (
            self.benchmark.search_point_id
            != point.search_point_id
        ):
            raise ValueError(
                "Benchmark search_point_id does not "
                "match admissibility evidence."
            )

        if (
            self.benchmark.implementation_id
            != point.implementation_id
        ):
            raise ValueError(
                "Benchmark implementation_id does not "
                "match admissibility evidence."
            )

        if (
            self.benchmark.task_family
            != point.task_family
        ):
            raise ValueError(
                "Benchmark task family does not match "
                "admissibility evidence."
            )

        return self


class CandidateObjectiveVector(BaseModel):
    """Comparable objective values for one admissible search point."""

    search_point_id: str = Field(
        min_length=1,
    )
    implementation_id: str = Field(
        min_length=1,
    )
    task_family: str = Field(
        min_length=1,
    )

    values: dict[
        ObjectiveMetric,
        float,
    ] = Field(
        min_length=1,
    )


class MultiObjectiveAnalysis(BaseModel):
    """Deterministic Pareto analysis without automatic recommendation."""

    task_family: str = Field(
        min_length=1,
    )

    comparability: EvidenceComparabilityKey

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

    preference_policy: ParetoPreferencePolicy | None = None

    vectors: tuple[
        CandidateObjectiveVector,
        ...,
    ] = Field(
        min_length=1,
    )

    pareto_frontier_search_point_ids: tuple[
        str,
        ...,
    ] = ()

    preferred_frontier_search_point_ids: tuple[
        str,
        ...,
    ] = ()

    dominated_search_point_ids: tuple[
        str,
        ...,
    ] = ()

    constraint_rejections: tuple[
        ConstraintRejectedCandidate,
        ...,
    ] = ()

    @model_validator(mode="after")
    def validate_partition(
        self,
    ) -> Self:
        """Require a complete, non-overlapping frontier partition."""
        vector_ids = {
            vector.search_point_id
            for vector in self.vectors
        }

        frontier_ids = set(
            self.pareto_frontier_search_point_ids
        )

        dominated_ids = set(
            self.dominated_search_point_ids
        )

        constraint_rejected_ids = {
            rejection.search_point_id
            for rejection in self.constraint_rejections
        }

        partitions = (
            frontier_ids,
            dominated_ids,
            constraint_rejected_ids,
        )

        for index, left in enumerate(
            partitions
        ):
            for right in partitions[
                index + 1 :
            ]:
                if left & right:
                    raise ValueError(
                        "A search point cannot appear in "
                        "multiple optimisation partitions."
                    )

        if (
            frontier_ids
            | dominated_ids
            | constraint_rejected_ids
            != vector_ids
        ):
            raise ValueError(
                "Multi-objective analysis must partition every "
                "candidate objective vector."
            )

        preferred_ids = set(
            self.preferred_frontier_search_point_ids
        )

        if not preferred_ids <= frontier_ids:
            raise ValueError(
                "Preferred candidates must remain inside "
                "the Pareto frontier."
            )

        if (
            self.preference_policy is None
            and preferred_ids != frontier_ids
        ):
            raise ValueError(
                "Without a preference policy, the preferred "
                "set must equal the full Pareto frontier."
            )

        return self


def analyse_multi_objective_candidates(
    *,
    candidates: tuple[
        MultiObjectiveCandidateEvidence,
        ...,
    ],
    objectives: tuple[
        OptimisationObjective,
        ...,
    ],
    constraints: tuple[
        ObjectiveConstraint,
        ...,
    ] = (),
    preference_policy: ParetoPreferencePolicy | None = None,
) -> MultiObjectiveAnalysis:
    """Build a Pareto frontier from already-admissible evidence."""
    if not candidates:
        raise ValueError(
            "At least one admissible optimisation candidate "
            "is required."
        )

    if not objectives:
        raise ValueError(
            "At least one optimisation objective is required."
        )

    objective_metrics = [
        objective.metric
        for objective in objectives
    ]

    if len(objective_metrics) != len(
        set(objective_metrics)
    ):
        raise ValueError(
            "Optimisation objective metrics must be unique."
        )

    constraint_keys = [
        (
            constraint.metric,
            constraint.operator,
        )
        for constraint in constraints
    ]

    if len(constraint_keys) != len(
        set(constraint_keys)
    ):
        raise ValueError(
            "Duplicate optimisation constraints are not allowed."
        )

    if preference_policy is not None:
        undeclared_preferences = [
            priority.metric
            for priority in preference_policy.priorities
            if priority.metric not in objective_metrics
        ]

        if undeclared_preferences:
            joined = ", ".join(
                metric.value
                for metric in undeclared_preferences
            )

            raise ValueError(
                "Preference priorities must reference declared "
                f"optimisation objectives: {joined}."
            )

    ordered_candidates = tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.benchmark.search_point_id
            ),
        )
    )

    task_families = {
        candidate.benchmark.task_family
        for candidate in ordered_candidates
    }

    if len(task_families) != 1:
        raise ValueError(
            "Multi-objective optimisation cannot compare "
            "different task families."
        )

    requested_metrics = {
        *objective_metrics,
        *(
            constraint.metric
            for constraint in constraints
        ),
    }

    comparability = _validate_comparability(
        candidates=ordered_candidates,
        requested_metrics=requested_metrics,
    )

    search_point_ids = [
        candidate.benchmark.search_point_id
        for candidate in ordered_candidates
    ]

    if len(search_point_ids) != len(
        set(search_point_ids)
    ):
        raise ValueError(
            "Duplicate search_point_id values cannot enter "
            "multi-objective optimisation."
        )

    vectors = tuple(
        _build_objective_vector(
            candidate=candidate,
            objectives=objectives,
        )
        for candidate in ordered_candidates
    )

    constraint_rejections = tuple(
        rejection
        for candidate in ordered_candidates
        if (
            rejection := _constraint_rejection(
                candidate=candidate,
                constraints=constraints,
            )
        )
        is not None
    )

    constraint_rejected_ids = {
        rejection.search_point_id
        for rejection in constraint_rejections
    }

    feasible_vectors = tuple(
        vector
        for vector in vectors
        if (
            vector.search_point_id
            not in constraint_rejected_ids
        )
    )

    frontier_ids: list[str] = []
    dominated_ids: list[str] = []

    for candidate in feasible_vectors:
        is_dominated = any(
            other.search_point_id
            != candidate.search_point_id
            and _dominates(
                left=other,
                right=candidate,
                objectives=objectives,
            )
            for other in feasible_vectors
        )

        if is_dominated:
            dominated_ids.append(
                candidate.search_point_id
            )
        else:
            frontier_ids.append(
                candidate.search_point_id
            )

    frontier_id_set = set(
        frontier_ids
    )

    frontier_vectors = tuple(
        vector
        for vector in feasible_vectors
        if vector.search_point_id in frontier_id_set
    )

    preferred_ids = (
        _apply_preference_policy(
            frontier=frontier_vectors,
            objectives=objectives,
            policy=preference_policy,
        )
    )

    return MultiObjectiveAnalysis(
        task_family=next(
            iter(task_families)
        ),
        comparability=comparability,
        objectives=objectives,
        constraints=constraints,
        preference_policy=preference_policy,
        vectors=vectors,
        pareto_frontier_search_point_ids=tuple(
            frontier_ids
        ),
        preferred_frontier_search_point_ids=(
            preferred_ids
        ),
        dominated_search_point_ids=tuple(
            dominated_ids
        ),
        constraint_rejections=(
            constraint_rejections
        ),
    )


def _validate_comparability(
    *,
    candidates: tuple[
        MultiObjectiveCandidateEvidence,
        ...,
    ],
    requested_metrics: set[
        ObjectiveMetric
    ],
) -> EvidenceComparabilityKey:
    """Reject incomparable workloads, protocols, or partial resources."""
    first = candidates[0].comparability

    for candidate in candidates[1:]:
        if candidate.comparability != first:
            raise ValueError(
                "Multi-objective optimisation requires the "
                "same workload fingerprint and measurement "
                "protocol for every compared candidate."
            )

    resource_metrics = {
        ObjectiveMetric.MEAN_PROCESS_CPU_TIME_MS,
        ObjectiveMetric.MAX_PROCESS_PEAK_RSS_BYTES,
    }

    if requested_metrics & resource_metrics:
        for candidate in candidates:
            summary = (
                candidate
                .benchmark
                .repeatability
            )

            if (
                summary.resource_observations
                != summary.repetitions
            ):
                raise ValueError(
                    "Resource optimisation requires complete "
                    "resource evidence for every benchmark "
                    "repetition; incomplete evidence for "
                    f"'{candidate.benchmark.search_point_id}'."
                )

    return first


def _build_objective_vector(
    *,
    candidate: MultiObjectiveCandidateEvidence,
    objectives: tuple[
        OptimisationObjective,
        ...,
    ],
) -> CandidateObjectiveVector:
    """Extract only explicitly requested evidence dimensions."""
    benchmark = candidate.benchmark

    values = {
        objective.metric: _metric_value(
            benchmark=benchmark,
            metric=objective.metric,
        )
        for objective in objectives
    }

    return CandidateObjectiveVector(
        search_point_id=benchmark.search_point_id,
        implementation_id=benchmark.implementation_id,
        task_family=benchmark.task_family,
        values=values,
    )


def _metric_value(
    *,
    benchmark: StructuredBenchmarkSeries,
    metric: ObjectiveMetric,
) -> float:
    """Return one observed metric or reject incomplete evidence."""
    summary = benchmark.repeatability

    if metric is ObjectiveMetric.MEAN_LATENCY_MS:
        return (
            summary.weighted_mean_latency_ms
        )

    if (
        metric
        is ObjectiveMetric.MEAN_CASES_PER_SECOND
    ):
        return summary.mean_cases_per_second

    if (
        metric
        is ObjectiveMetric.MEAN_PROCESS_CPU_TIME_MS
    ):
        value = (
            summary.mean_process_cpu_time_ms
        )

        if value is None:
            raise ValueError(
                "Missing mean process CPU evidence for "
                f"'{benchmark.search_point_id}'."
            )

        return value

    if (
        metric
        is ObjectiveMetric.MAX_PROCESS_PEAK_RSS_BYTES
    ):
        value = (
            summary.max_process_peak_rss_bytes
        )

        if value is None:
            raise ValueError(
                "Missing process peak RSS evidence for "
                f"'{benchmark.search_point_id}'."
            )

        return float(value)

    if metric is ObjectiveMetric.MODEL_SIZE_BYTES:
        footprint = benchmark.footprint

        if (
            footprint is None
            or footprint.model_size_bytes is None
        ):
            raise ValueError(
                "Missing model-size evidence for "
                f"'{benchmark.search_point_id}'."
            )

        return float(
            footprint.model_size_bytes
        )

    raise ValueError(
        f"Unsupported optimisation metric: {metric}."
    )


def _apply_preference_policy(
    *,
    frontier: tuple[
        CandidateObjectiveVector,
        ...,
    ],
    objectives: tuple[
        OptimisationObjective,
        ...,
    ],
    policy: ParetoPreferencePolicy | None,
) -> tuple[str, ...]:
    """Narrow only the Pareto frontier using explicit ordered priorities."""
    if policy is None:
        return tuple(
            vector.search_point_id
            for vector in frontier
        )

    directions = {
        objective.metric: objective.direction
        for objective in objectives
    }

    active = frontier

    for priority in policy.priorities:
        if len(active) <= 1:
            break

        metric = priority.metric
        direction = directions[
            metric
        ]

        observed = [
            vector.values[metric]
            for vector in active
        ]

        if (
            direction
            is ObjectiveDirection.MINIMIZE
        ):
            best = min(
                observed
            )
            threshold = (
                best
                + abs(best)
                * priority.relative_tolerance
            )

            active = tuple(
                vector
                for vector in active
                if (
                    vector.values[metric]
                    <= threshold
                )
            )
        else:
            best = max(
                observed
            )
            threshold = (
                best
                - abs(best)
                * priority.relative_tolerance
            )

            active = tuple(
                vector
                for vector in active
                if (
                    vector.values[metric]
                    >= threshold
                )
            )

    return tuple(
        vector.search_point_id
        for vector in active
    )


def _constraint_rejection(
    *,
    candidate: MultiObjectiveCandidateEvidence,
    constraints: tuple[
        ObjectiveConstraint,
        ...,
    ],
) -> ConstraintRejectedCandidate | None:
    """Return explicit hard-constraint violations for one candidate."""
    violations: list[
        ConstraintViolation
    ] = []

    for constraint in constraints:
        observed = _metric_value(
            benchmark=candidate.benchmark,
            metric=constraint.metric,
        )

        violated = (
            observed > constraint.threshold
            if (
                constraint.operator
                is ConstraintOperator.AT_MOST
            )
            else observed < constraint.threshold
        )

        if violated:
            violations.append(
                ConstraintViolation(
                    metric=constraint.metric,
                    operator=constraint.operator,
                    threshold=constraint.threshold,
                    observed_value=observed,
                )
            )

    if not violations:
        return None

    return ConstraintRejectedCandidate(
        search_point_id=(
            candidate.benchmark.search_point_id
        ),
        violations=tuple(
            violations
        ),
    )


def _dominates(
    *,
    left: CandidateObjectiveVector,
    right: CandidateObjectiveVector,
    objectives: tuple[
        OptimisationObjective,
        ...,
    ],
) -> bool:
    """Return whether left is Pareto-superior to right."""
    no_worse_on_all = True
    strictly_better_on_any = False

    for objective in objectives:
        left_value = left.values[
            objective.metric
        ]
        right_value = right.values[
            objective.metric
        ]

        if (
            objective.direction
            is ObjectiveDirection.MINIMIZE
        ):
            if left_value > right_value:
                no_worse_on_all = False
                break

            if left_value < right_value:
                strictly_better_on_any = True

        else:
            if left_value < right_value:
                no_worse_on_all = False
                break

            if left_value > right_value:
                strictly_better_on_any = True

    return (
        no_worse_on_all
        and strictly_better_on_any
    )
