"""Tests for benchmark experiment provenance."""

from pydantic import JsonValue

from vait.benchmark.models import BenchmarkCase, BenchmarkDataset
from vait.benchmark.provenance import (
    benchmark_fingerprint,
    build_experiment_provenance,
    summarize_latencies,
)
from vait.contracts.models import RiskLevel
from vait.runners.python_runner import PythonImplementationRunner


def candidate(data: dict[str, JsonValue]) -> JsonValue:
    """Return a deterministic test result."""
    return data


def build_dataset() -> BenchmarkDataset:
    """Create a compact benchmark fixture."""
    return BenchmarkDataset(
        benchmark_id="provenance-test",
        version="0.1",
        description="Dataset fingerprint fixture.",
        cases=[
            BenchmarkCase(
                case_id="case-1",
                input_data={"value": 1},
                gold_outcome={"value": 1},
                risk_level=RiskLevel.LOW,
                tags=frozenset({"nominal", "test"}),
                rationale="Provenance test case.",
            )
        ],
    )


def test_dataset_fingerprint_is_deterministic() -> None:
    """Identical benchmark content should produce the same fingerprint."""
    dataset = build_dataset()

    first = benchmark_fingerprint(dataset)
    second = benchmark_fingerprint(dataset)

    assert first == second
    assert len(first) == 64


def test_dataset_change_changes_fingerprint() -> None:
    """Changing benchmark truth must change its fingerprint."""
    original = build_dataset()
    modified = original.model_copy(deep=True)

    modified.cases[0].gold_outcome = {"value": 2}

    assert benchmark_fingerprint(original) != benchmark_fingerprint(modified)


def test_provenance_records_candidate_and_environment() -> None:
    """A run should capture candidate and reproducibility metadata."""
    runner = PythonImplementationRunner(
        implementation_id="candidate-v1",
        function=candidate,
    )

    provenance = build_experiment_provenance(
        dataset=build_dataset(),
        candidate=runner,
        candidate_configuration={
            "mode": "deterministic",
        },
    )

    assert provenance.candidate_implementation_id == "candidate-v1"
    assert provenance.candidate_configuration == {
        "mode": "deterministic",
    }
    assert provenance.created_at.tzinfo is not None
    assert len(provenance.dataset_fingerprint_sha256) == 64
    assert provenance.python_version
    assert provenance.platform
    assert provenance.package_versions


def test_latency_summary_reports_distribution() -> None:
    """Latency evidence should include basic distribution statistics."""
    summary = summarize_latencies(
        [1.0, 2.0, 3.0, 4.0]
    )

    assert summary is not None
    assert summary.count == 4
    assert summary.mean_ms == 2.5
    assert summary.p50_ms == 2.0
    assert summary.p95_ms == 4.0
    assert summary.max_ms == 4.0


def test_empty_latency_sequence_has_no_summary() -> None:
    """No latency observations should produce no summary."""
    assert summarize_latencies([]) is None
