"""Structured inference benchmarking after VAIT admissibility."""

from collections.abc import Mapping, Sequence
from statistics import fmean, pstdev

from pydantic import BaseModel, Field, JsonValue, model_validator

from vait.contracts.models import VerificationCase
from vait.inference.benchmark import (
    capture_inference_environment,
    run_controlled_inference_benchmark,
)
from vait.inference.models import (
    InferenceBenchmarkReport,
    InferenceEnvironment,
    InferenceSetupEvidence,
)
from vait.optimisation.admissibility import (
    SearchPointAdmissibility,
)
from vait.runners.base import ImplementationRunner


class StructuredModelFootprint(BaseModel):
    """Optional model-size evidence attached to a structured candidate."""

    parameter_count: int | None = Field(
        default=None,
        ge=0,
    )

    model_size_bytes: int | None = Field(
        default=None,
        ge=0,
    )

    metadata: dict[str, JsonValue] = Field(
        default_factory=dict,
    )


class StructuredBenchmarkRun(BaseModel):
    """One controlled repetition for an admissible structured candidate."""

    repetition_index: int = Field(
        ge=0,
    )

    report: InferenceBenchmarkReport

    @model_validator(mode="after")
    def validate_structured_report(
        self,
    ) -> "StructuredBenchmarkRun":
        """Require non-generative case-throughput evidence."""
        if self.report.generative is not None:
            raise ValueError(
                "Structured benchmark runs cannot contain "
                "generative inference evidence."
            )

        if (
            self.report.throughput.cases_per_second
            is None
        ):
            raise ValueError(
                "Structured benchmark runs require "
                "cases_per_second throughput evidence."
            )

        return self


class StructuredRepeatabilitySummary(BaseModel):
    """Cross-run repeatability evidence for one search point."""

    repetitions: int = Field(
        ge=1,
    )

    measured_executions: int = Field(
        ge=1,
    )

    weighted_mean_latency_ms: float = Field(
        ge=0.0,
        allow_inf_nan=False,
    )

    run_mean_latency_stdev_ms: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
    )

    run_mean_latency_cv: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
    )

    mean_cases_per_second: float = Field(
        ge=0.0,
        allow_inf_nan=False,
    )

    throughput_stdev: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
    )

    throughput_cv: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
    )

    worst_p95_latency_ms: float = Field(
        ge=0.0,
        allow_inf_nan=False,
    )

    worst_p99_latency_ms: float = Field(
        ge=0.0,
        allow_inf_nan=False,
    )

    worst_max_latency_ms: float = Field(
        ge=0.0,
        allow_inf_nan=False,
    )


class StructuredBenchmarkSeries(BaseModel):
    """Repeated controlled evidence for one admissible search point."""

    search_point_id: str = Field(
        min_length=1,
    )

    implementation_id: str = Field(
        min_length=1,
    )

    task_family: str = Field(
        min_length=1,
    )

    runs: tuple[
        StructuredBenchmarkRun,
        ...,
    ] = Field(
        min_length=1,
    )

    repeatability: StructuredRepeatabilitySummary

    footprint: StructuredModelFootprint | None = None

    @model_validator(mode="after")
    def validate_series_consistency(
        self,
    ) -> "StructuredBenchmarkSeries":
        """Require one candidate/configuration across every repetition."""
        expected_indices = list(
            range(len(self.runs))
        )

        actual_indices = [
            run.repetition_index
            for run in self.runs
        ]

        if actual_indices != expected_indices:
            raise ValueError(
                "Structured benchmark repetitions must be "
                "ordered contiguously from zero."
            )

        first_configuration = (
            self.runs[0].report.configuration
        )

        for run in self.runs:
            report = run.report

            if (
                report.candidate_implementation_id
                != self.implementation_id
            ):
                raise ValueError(
                    "Structured benchmark candidate identity "
                    "changed across repetitions."
                )

            if report.task_family != self.task_family:
                raise ValueError(
                    "Structured benchmark task family changed "
                    "across repetitions."
                )

            if (
                report.configuration
                != first_configuration
            ):
                raise ValueError(
                    "Structured benchmark configuration changed "
                    "across repetitions."
                )

        if (
            self.repeatability.repetitions
            != len(self.runs)
        ):
            raise ValueError(
                "Repeatability repetition count does not match "
                "the benchmark runs."
            )

        measured_executions = sum(
            run.report.latency.count
            for run in self.runs
        )

        if (
            self.repeatability.measured_executions
            != measured_executions
        ):
            raise ValueError(
                "Repeatability measured-execution count does "
                "not match the benchmark runs."
            )

        return self


def summarize_structured_repetitions(
    runs: Sequence[
        StructuredBenchmarkRun
    ],
) -> StructuredRepeatabilitySummary:
    """Summarise repeated structured inference measurements."""
    if not runs:
        raise ValueError(
            "At least one structured benchmark run is required."
        )

    measured_executions = sum(
        run.report.latency.count
        for run in runs
    )

    weighted_latency_total = sum(
        (
            run.report.latency.mean_ms
            * run.report.latency.count
        )
        for run in runs
    )

    weighted_mean_latency_ms = (
        weighted_latency_total
        / measured_executions
    )

    run_mean_latencies = [
        run.report.latency.mean_ms
        for run in runs
    ]

    throughput_values: list[float] = []

    for run in runs:
        cases_per_second = (
            run.report.throughput.cases_per_second
        )

        if cases_per_second is None:
            raise ValueError(
                "Structured benchmark run lacks "
                "cases_per_second evidence."
            )

        throughput_values.append(
            cases_per_second
        )

    latency_stdev, latency_cv = (
        _repeatability_dispersion(
            run_mean_latencies
        )
    )

    throughput_stdev, throughput_cv = (
        _repeatability_dispersion(
            throughput_values
        )
    )

    return StructuredRepeatabilitySummary(
        repetitions=len(runs),
        measured_executions=measured_executions,
        weighted_mean_latency_ms=(
            weighted_mean_latency_ms
        ),
        run_mean_latency_stdev_ms=(
            latency_stdev
        ),
        run_mean_latency_cv=latency_cv,
        mean_cases_per_second=fmean(
            throughput_values
        ),
        throughput_stdev=throughput_stdev,
        throughput_cv=throughput_cv,
        worst_p95_latency_ms=max(
            run.report.latency.p95_ms
            for run in runs
        ),
        worst_p99_latency_ms=max(
            run.report.latency.p99_ms
            for run in runs
        ),
        worst_max_latency_ms=max(
            run.report.latency.max_ms
            for run in runs
        ),
    )


def run_structured_benchmark_series(
    *,
    admissibility: SearchPointAdmissibility,
    runner: ImplementationRunner,
    cases: Sequence[VerificationCase],
    repetitions: int,
    warmup_rounds: int,
    measured_rounds: int,
    setup: InferenceSetupEvidence | None = None,
    environment: InferenceEnvironment | None = None,
    footprint: StructuredModelFootprint | None = None,
    evidence_metadata: Mapping[
        str,
        JsonValue,
    ]
    | None = None,
) -> StructuredBenchmarkSeries:
    """Benchmark only an already-admissible structured search point."""
    if not admissibility.is_admissible:
        raise ValueError(
            "Structured benchmarking requires an "
            "admissible search point."
        )

    search_point = admissibility.search_point

    if (
        runner.implementation_id
        != search_point.implementation_id
    ):
        raise ValueError(
            "Runner implementation_id does not match "
            "the admissible search point."
        )

    if repetitions < 1:
        raise ValueError(
            "At least one benchmark repetition is required."
        )

    if search_point.configuration.batch_size != 1:
        raise ValueError(
            "Scalar structured benchmarking requires "
            "batch_size=1; use the native batch benchmark "
            "path for batch scaling."
        )

    shared_environment = (
        environment
        or capture_inference_environment()
    )

    runs: list[
        StructuredBenchmarkRun
    ] = []

    for repetition_index in range(
        repetitions
    ):
        metadata: dict[
            str,
            JsonValue,
        ] = dict(
            evidence_metadata
            or {}
        )

        metadata.update(
            {
                "search_point_id": (
                    search_point.search_point_id
                ),
                "admissibility_gate": "passed",
                "execution_mode": "scalar-per-case",
                "repetition_index": (
                    repetition_index
                ),
            }
        )

        report = (
            run_controlled_inference_benchmark(
                runner=runner,
                cases=cases,
                task_family=(
                    search_point.task_family
                ),
                configuration=(
                    search_point.configuration
                ),
                warmup_rounds=warmup_rounds,
                measured_rounds=measured_rounds,
                setup=setup,
                measure_cold_start=False,
                environment=shared_environment,
                evidence_metadata=metadata,
            )
        )

        runs.append(
            StructuredBenchmarkRun(
                repetition_index=(
                    repetition_index
                ),
                report=report,
            )
        )

    repeatability = (
        summarize_structured_repetitions(
            runs
        )
    )

    return StructuredBenchmarkSeries(
        search_point_id=(
            search_point.search_point_id
        ),
        implementation_id=(
            search_point.implementation_id
        ),
        task_family=(
            search_point.task_family
        ),
        runs=tuple(runs),
        repeatability=repeatability,
        footprint=footprint,
    )


def _repeatability_dispersion(
    values: Sequence[float],
) -> tuple[
    float | None,
    float | None,
]:
    """Return population stdev and CV when repetition evidence exists."""
    if len(values) < 2:
        return None, None

    standard_deviation = pstdev(
        values
    )

    mean_value = fmean(
        values
    )

    coefficient_of_variation = (
        standard_deviation / mean_value
        if mean_value > 0.0
        else None
    )

    return (
        standard_deviation,
        coefficient_of_variation,
    )
