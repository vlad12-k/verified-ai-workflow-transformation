"""Controlled inference benchmark evidence utilities."""

import platform
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from math import ceil, isfinite
from statistics import fmean
from time import perf_counter_ns
from uuid import uuid4

from pydantic import JsonValue

from vait.contracts.models import VerificationCase
from vait.inference.models import (
    GenerativeInferenceSample,
    GenerativeInferenceSummary,
    InferenceBenchmarkReport,
    InferenceConfiguration,
    InferenceEnvironment,
    InferenceLatencySummary,
    InferenceSetupEvidence,
    InferenceThroughputSummary,
    InferenceWorkload,
)
from vait.runners.base import ImplementationRunner

_TRACKED_PACKAGES = (
    "verified-ai-workflow-transformation",
    "numpy",
    "scipy",
    "scikit-learn",
    "xgboost",
    "torch",
    "tensorflow",
    "sentence-transformers",
    "transformers",
    "peft",
    "bitsandbytes",
)


def summarize_inference_latencies(
    latencies_ms: Iterable[float],
) -> InferenceLatencySummary:
    """Summarize measured inference latency observations."""
    values = sorted(float(value) for value in latencies_ms)

    if not values:
        raise ValueError("At least one latency observation is required.")

    if any(value < 0.0 or not isfinite(value) for value in values):
        raise ValueError("Latency observations must be finite and non-negative.")

    return InferenceLatencySummary(
        count=len(values),
        mean_ms=fmean(values),
        p50_ms=_nearest_rank(values, 0.50),
        p95_ms=_nearest_rank(values, 0.95),
        p99_ms=_nearest_rank(values, 0.99),
        max_ms=values[-1],
    )


def build_throughput_summary(
    *,
    elapsed_seconds: float,
    cases_processed: int | None = None,
    requests_processed: int | None = None,
    output_tokens: int | None = None,
) -> InferenceThroughputSummary:
    """Calculate normalised throughput rates for a measured interval."""
    if elapsed_seconds <= 0.0 or not isfinite(elapsed_seconds):
        raise ValueError("Elapsed time must be finite and greater than zero.")

    counts = (
        cases_processed,
        requests_processed,
        output_tokens,
    )

    if all(value is None for value in counts):
        raise ValueError("At least one throughput count is required.")

    for value in counts:
        if value is not None and value < 0:
            raise ValueError("Throughput counts must be non-negative.")

    return InferenceThroughputSummary(
        cases_per_second=_rate(cases_processed, elapsed_seconds),
        requests_per_second=_rate(requests_processed, elapsed_seconds),
        tokens_per_second=_rate(output_tokens, elapsed_seconds),
    )


def summarize_generative_samples(
    samples: Sequence[GenerativeInferenceSample],
) -> GenerativeInferenceSummary:
    """Aggregate token and latency evidence for generative inference."""
    if not samples:
        raise ValueError("At least one generative sample is required.")

    ttft_values = [
        sample.time_to_first_token_ms
        for sample in samples
        if sample.time_to_first_token_ms is not None
    ]
    generation_values = [
        sample.generation_latency_ms
        for sample in samples
        if sample.generation_latency_ms is not None
    ]

    return GenerativeInferenceSummary(
        sample_count=len(samples),
        time_to_first_token=(
            summarize_inference_latencies(ttft_values)
            if ttft_values
            else None
        ),
        generation_latency=(
            summarize_inference_latencies(generation_values)
            if generation_values
            else None
        ),
        input_tokens=sum(sample.input_tokens for sample in samples),
        output_tokens=sum(sample.output_tokens for sample in samples),
    )


def capture_inference_environment() -> InferenceEnvironment:
    """Capture portable environment metadata for inference evidence."""
    processor = platform.processor().strip() or "unknown"

    return InferenceEnvironment(
        python_version=platform.python_version(),
        platform=platform.platform(),
        system=platform.system() or "unknown",
        machine=platform.machine() or "unknown",
        processor=processor,
        package_versions=_installed_package_versions(),
    )


def build_inference_benchmark_report(
    *,
    candidate_implementation_id: str,
    task_family: str,
    configuration: InferenceConfiguration,
    warmup_iterations: int,
    workload: InferenceWorkload | None = None,
    setup: InferenceSetupEvidence | None = None,
    latency_samples_ms: Sequence[float],
    throughput: InferenceThroughputSummary,
    cold_start_latency_ms: float | None = None,
    generative: GenerativeInferenceSummary | None = None,
    environment: InferenceEnvironment | None = None,
    evidence_metadata: Mapping[str, JsonValue] | None = None,
) -> InferenceBenchmarkReport:
    """Build one versioned controlled inference evidence report."""
    latency = summarize_inference_latencies(latency_samples_ms)

    return InferenceBenchmarkReport(
        run_id=str(uuid4()),
        created_at=datetime.now(UTC),
        candidate_implementation_id=candidate_implementation_id,
        task_family=task_family,
        workload=workload,
        setup=setup,
        warmup_iterations=warmup_iterations,
        measured_iterations=len(latency_samples_ms),
        cold_start_latency_ms=cold_start_latency_ms,
        configuration=configuration,
        latency=latency,
        throughput=throughput,
        generative=generative,
        environment=environment or capture_inference_environment(),
        evidence_metadata=dict(evidence_metadata or {}),
    )



def run_controlled_inference_benchmark(
    *,
    runner: ImplementationRunner,
    cases: Sequence[VerificationCase],
    task_family: str,
    configuration: InferenceConfiguration,
    warmup_rounds: int,
    measured_rounds: int,
    workload: InferenceWorkload | None = None,
    setup: InferenceSetupEvidence | None = None,
    measure_cold_start: bool = True,
    environment: InferenceEnvironment | None = None,
    evidence_metadata: Mapping[str, JsonValue] | None = None,
) -> InferenceBenchmarkReport:
    """Execute a controlled warm-up and measured inference benchmark."""
    if not cases:
        raise ValueError("At least one inference case is required.")

    if warmup_rounds < 0:
        raise ValueError("Warm-up rounds must be non-negative.")

    if measured_rounds < 1:
        raise ValueError("At least one measured round is required.")

    cold_start_latency_ms: float | None = None

    if measure_cold_start:
        cold_start_latency_ms = _execute_timed_case(
            runner=runner,
            case=cases[0],
            phase="cold-start",
        )

    for _ in range(warmup_rounds):
        for case in cases:
            _execute_timed_case(
                runner=runner,
                case=case,
                phase="warm-up",
            )

    latency_samples_ms: list[float] = []

    measurement_started = perf_counter_ns()

    for _ in range(measured_rounds):
        for case in cases:
            latency_samples_ms.append(
                _execute_timed_case(
                    runner=runner,
                    case=case,
                    phase="measured",
                )
            )

    elapsed_seconds = (
        perf_counter_ns() - measurement_started
    ) / 1_000_000_000

    measured_executions = len(latency_samples_ms)

    throughput = build_throughput_summary(
        elapsed_seconds=elapsed_seconds,
        cases_processed=measured_executions,
        requests_processed=measured_executions,
    )

    metadata: dict[str, JsonValue] = dict(
        evidence_metadata or {}
    )
    metadata.update(
        {
            "case_count": len(cases),
            "warmup_rounds": warmup_rounds,
            "measured_rounds": measured_rounds,
            "measurement_scope": "runner_execute_wall_clock",
        }
    )

    return build_inference_benchmark_report(
        candidate_implementation_id=runner.implementation_id,
        task_family=task_family,
        configuration=configuration,
        workload=workload,
        setup=setup,
        warmup_iterations=warmup_rounds * len(cases),
        latency_samples_ms=latency_samples_ms,
        throughput=throughput,
        cold_start_latency_ms=cold_start_latency_ms,
        environment=environment,
        evidence_metadata=metadata,
    )



def run_controlled_generative_benchmark(
    *,
    candidate_implementation_id: str,
    operation: Callable[[], GenerativeInferenceSample],
    task_family: str,
    configuration: InferenceConfiguration,
    warmup_iterations: int,
    measured_iterations: int,
    workload: InferenceWorkload | None = None,
    setup: InferenceSetupEvidence | None = None,
    environment: InferenceEnvironment | None = None,
    evidence_metadata: Mapping[str, JsonValue] | None = None,
) -> InferenceBenchmarkReport:
    """Execute controlled generative inference measurements."""
    if warmup_iterations < 0:
        raise ValueError("Warm-up iterations must be non-negative.")

    if measured_iterations < 1:
        raise ValueError("At least one measured iteration is required.")

    for _ in range(warmup_iterations):
        operation()

    samples: list[GenerativeInferenceSample] = []
    latency_samples_ms: list[float] = []

    measurement_started = perf_counter_ns()

    for _ in range(measured_iterations):
        started = perf_counter_ns()
        sample = operation()
        elapsed_ms = (
            perf_counter_ns() - started
        ) / 1_000_000

        samples.append(sample)
        latency_samples_ms.append(elapsed_ms)

    elapsed_seconds = (
        perf_counter_ns() - measurement_started
    ) / 1_000_000_000

    generative = summarize_generative_samples(samples)

    throughput = build_throughput_summary(
        elapsed_seconds=elapsed_seconds,
        requests_processed=measured_iterations,
        output_tokens=generative.output_tokens,
    )

    ttft_observed_samples = sum(
        sample.time_to_first_token_ms is not None
        for sample in samples
    )
    generation_latency_observed_samples = sum(
        sample.generation_latency_ms is not None
        for sample in samples
    )

    metadata: dict[str, JsonValue] = dict(
        evidence_metadata or {}
    )
    metadata.update(
        {
            "measurement_scope": (
                "generative_operation_wall_clock"
            ),
            "ttft_observed_samples": (
                ttft_observed_samples
            ),
            "generation_latency_observed_samples": (
                generation_latency_observed_samples
            ),
        }
    )

    return build_inference_benchmark_report(
        candidate_implementation_id=candidate_implementation_id,
        task_family=task_family,
        configuration=configuration,
        workload=workload,
        setup=setup,
        warmup_iterations=warmup_iterations,
        latency_samples_ms=latency_samples_ms,
        throughput=throughput,
        generative=generative,
        environment=environment,
        evidence_metadata=metadata,
    )


def _execute_timed_case(
    *,
    runner: ImplementationRunner,
    case: VerificationCase,
    phase: str,
) -> float:
    """Execute one case and return outer wall-clock latency in milliseconds."""
    started = perf_counter_ns()
    observation = runner.execute(case)
    latency_ms = (perf_counter_ns() - started) / 1_000_000

    if observation.error is not None:
        raise RuntimeError(
            f"Inference execution failed during {phase} "
            f"for case '{case.id}': {observation.error}"
        )

    return latency_ms

def _nearest_rank(
    sorted_values: Sequence[float],
    percentile: float,
) -> float:
    """Return the nearest-rank percentile for sorted measurements."""
    index = ceil(percentile * len(sorted_values)) - 1
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def _rate(
    count: int | None,
    elapsed_seconds: float,
) -> float | None:
    """Convert an optional count into a per-second rate."""
    if count is None:
        return None

    return count / elapsed_seconds


def _installed_package_versions() -> dict[str, str]:
    """Return versions of packages relevant to inference reproduction."""
    versions: dict[str, str] = {}

    for package_name in _TRACKED_PACKAGES:
        try:
            versions[package_name] = package_version(package_name)
        except PackageNotFoundError:
            versions[package_name] = "not-installed"

    return versions
