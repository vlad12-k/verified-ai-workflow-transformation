"""Tests for controlled inference benchmark evidence utilities."""

import pytest

from vait.contracts.models import VerificationCase
from vait.inference.benchmark import (
    build_inference_benchmark_report,
    build_throughput_summary,
    capture_inference_environment,
    run_controlled_inference_benchmark,
    summarize_generative_samples,
    summarize_inference_latencies,
)
from vait.inference.models import (
    GenerativeInferenceSample,
    InferenceConfiguration,
    InferenceEnvironment,
)
from vait.runners.python_runner import PythonImplementationRunner


def test_latency_summary_includes_p99() -> None:
    """Controlled latency evidence should include the full M4 distribution."""
    summary = summarize_inference_latencies(
        [float(value) for value in range(1, 101)]
    )

    assert summary.count == 100
    assert summary.mean_ms == 50.5
    assert summary.p50_ms == 50.0
    assert summary.p95_ms == 95.0
    assert summary.p99_ms == 99.0
    assert summary.max_ms == 100.0


@pytest.mark.parametrize(
    "values",
    [
        [],
        [-1.0],
        [float("inf")],
        [float("nan")],
    ],
)
def test_invalid_latency_samples_are_rejected(
    values: list[float],
) -> None:
    """Invalid measurements must not enter optimisation evidence."""
    with pytest.raises(ValueError):
        summarize_inference_latencies(values)


def test_throughput_summary_normalises_rates() -> None:
    """Throughput should be represented using comparable per-second rates."""
    summary = build_throughput_summary(
        elapsed_seconds=2.0,
        cases_processed=20,
        requests_processed=10,
        output_tokens=200,
    )

    assert summary.cases_per_second == 10.0
    assert summary.requests_per_second == 5.0
    assert summary.tokens_per_second == 100.0


def test_generative_summary_tracks_ttft_generation_and_tokens() -> None:
    """Generative evidence should keep timing stages and token counts separate."""
    summary = summarize_generative_samples(
        [
            GenerativeInferenceSample(
                time_to_first_token_ms=10.0,
                generation_latency_ms=40.0,
                input_tokens=20,
                output_tokens=10,
            ),
            GenerativeInferenceSample(
                time_to_first_token_ms=20.0,
                generation_latency_ms=60.0,
                input_tokens=30,
                output_tokens=15,
            ),
        ]
    )

    assert summary.sample_count == 2
    assert summary.time_to_first_token is not None
    assert summary.time_to_first_token.p50_ms == 10.0
    assert summary.generation_latency is not None
    assert summary.generation_latency.p95_ms == 60.0
    assert summary.input_tokens == 50
    assert summary.output_tokens == 25


def test_environment_capture_contains_reproducibility_metadata() -> None:
    """Inference evidence should identify its runtime environment."""
    environment = capture_inference_environment()

    assert environment.python_version
    assert environment.platform
    assert environment.system
    assert environment.machine
    assert environment.processor
    assert "torch" in environment.package_versions
    assert "transformers" in environment.package_versions


def test_report_is_versioned_and_links_configuration() -> None:
    """The M4 report should preserve configuration and evidence provenance."""
    configuration = InferenceConfiguration(
        provider="local",
        runtime="python",
        device="cpu",
        dtype="float32",
        batch_size=1,
        model_id="test-model",
        model_revision="revision-1",
    )
    environment = InferenceEnvironment(
        python_version="3.12.0",
        platform="test-platform",
        system="test-system",
        machine="test-machine",
        processor="test-processor",
        package_versions={"torch": "test-version"},
    )
    throughput = build_throughput_summary(
        elapsed_seconds=1.0,
        cases_processed=3,
        requests_processed=3,
    )

    report = build_inference_benchmark_report(
        candidate_implementation_id="candidate-v1",
        task_family="structured-decision",
        configuration=configuration,
        warmup_iterations=2,
        latency_samples_ms=[1.0, 2.0, 3.0],
        throughput=throughput,
        cold_start_latency_ms=5.0,
        environment=environment,
        evidence_metadata={"benchmark": "controlled-test"},
    )

    assert report.schema_version == "0.1"
    assert report.measured_iterations == 3
    assert report.latency.p95_ms == 3.0
    assert report.configuration.model_revision == "revision-1"
    assert report.throughput.cases_per_second == 3.0
    assert report.evidence_metadata == {
        "benchmark": "controlled-test",
    }
    assert report.created_at.tzinfo is not None
    assert report.run_id


def test_controlled_benchmark_runs_warmup_and_measured_rounds() -> None:
    """Controlled execution should produce measured inference evidence."""
    calls: list[int] = []

    def candidate(data: dict[str, object]) -> object:
        calls.append(1)
        return data

    runner = PythonImplementationRunner(
        implementation_id="controlled-candidate-v1",
        function=candidate,
    )
    cases = [
        VerificationCase(
            id="case-1",
            input_data={"value": 1},
        ),
        VerificationCase(
            id="case-2",
            input_data={"value": 2},
        ),
    ]
    configuration = InferenceConfiguration(
        provider="local",
        runtime="python",
        device="cpu",
        dtype="native",
        batch_size=1,
    )

    report = run_controlled_inference_benchmark(
        runner=runner,
        cases=cases,
        task_family="structured-decision",
        configuration=configuration,
        warmup_rounds=2,
        measured_rounds=3,
        measure_cold_start=True,
    )

    assert len(calls) == 11
    assert report.candidate_implementation_id == "controlled-candidate-v1"
    assert report.warmup_iterations == 4
    assert report.measured_iterations == 6
    assert report.cold_start_latency_ms is not None
    assert report.latency.count == 6
    assert report.throughput.cases_per_second is not None
    assert report.throughput.cases_per_second > 0.0
    assert report.throughput.requests_per_second is not None
    assert report.evidence_metadata["case_count"] == 2
    assert report.evidence_metadata["warmup_rounds"] == 2
    assert report.evidence_metadata["measured_rounds"] == 3
    assert (
        report.evidence_metadata["measurement_scope"]
        == "runner_execute_wall_clock"
    )


def test_controlled_benchmark_can_skip_cold_start() -> None:
    """Cold-start evidence should be optional when it is not meaningful."""

    def candidate(data: dict[str, object]) -> object:
        return data

    runner = PythonImplementationRunner(
        implementation_id="warm-only-candidate",
        function=candidate,
    )

    report = run_controlled_inference_benchmark(
        runner=runner,
        cases=[
            VerificationCase(
                id="case-1",
                input_data={"value": 1},
            )
        ],
        task_family="structured-decision",
        configuration=InferenceConfiguration(
            provider="local",
            runtime="python",
            device="cpu",
            dtype="native",
            batch_size=1,
        ),
        warmup_rounds=1,
        measured_rounds=2,
        measure_cold_start=False,
    )

    assert report.cold_start_latency_ms is None
    assert report.warmup_iterations == 1
    assert report.measured_iterations == 2


def test_controlled_benchmark_rejects_execution_errors() -> None:
    """Failed inference must not be represented as valid performance evidence."""

    def candidate(data: dict[str, object]) -> object:
        raise RuntimeError("synthetic inference failure")

    runner = PythonImplementationRunner(
        implementation_id="failing-candidate",
        function=candidate,
    )

    with pytest.raises(
        RuntimeError,
        match="synthetic inference failure",
    ):
        run_controlled_inference_benchmark(
            runner=runner,
            cases=[
                VerificationCase(
                    id="case-1",
                    input_data={"value": 1},
                )
            ],
            task_family="structured-decision",
            configuration=InferenceConfiguration(
                provider="local",
                runtime="python",
                device="cpu",
                dtype="native",
                batch_size=1,
            ),
            warmup_rounds=0,
            measured_rounds=1,
        )


@pytest.mark.parametrize(
    ("warmup_rounds", "measured_rounds"),
    [
        (-1, 1),
        (0, 0),
    ],
)
def test_controlled_benchmark_rejects_invalid_round_counts(
    warmup_rounds: int,
    measured_rounds: int,
) -> None:
    """Invalid benchmark execution plans should fail before measurement."""

    def candidate(data: dict[str, object]) -> object:
        return data

    runner = PythonImplementationRunner(
        implementation_id="candidate-v1",
        function=candidate,
    )

    with pytest.raises(ValueError):
        run_controlled_inference_benchmark(
            runner=runner,
            cases=[
                VerificationCase(
                    id="case-1",
                    input_data={"value": 1},
                )
            ],
            task_family="structured-decision",
            configuration=InferenceConfiguration(
                provider="local",
                runtime="python",
                device="cpu",
                dtype="native",
                batch_size=1,
            ),
            warmup_rounds=warmup_rounds,
            measured_rounds=measured_rounds,
        )
