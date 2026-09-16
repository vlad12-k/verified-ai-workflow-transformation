"""Tests for VAIT benchmark evaluation."""

from pydantic import JsonValue

from vait.benchmark.evaluator import evaluate_benchmark
from vait.benchmark.models import BenchmarkCase, BenchmarkDataset
from vait.contracts.models import RiskLevel
from vait.runners.python_runner import PythonImplementationRunner


def identity_candidate(data: dict[str, JsonValue]) -> JsonValue:
    """Return the declared decision."""
    return {"decision": data["decision"]}


def wrong_candidate(data: dict[str, JsonValue]) -> JsonValue:
    """Return an intentionally incorrect decision."""
    return {"decision": "WRONG"}


def failing_candidate(data: dict[str, JsonValue]) -> JsonValue:
    """Raise an execution failure."""
    raise RuntimeError("candidate failure")


def build_dataset() -> BenchmarkDataset:
    """Create a compact benchmark fixture."""
    return BenchmarkDataset(
        benchmark_id="unit-benchmark",
        version="0.1",
        description="Benchmark evaluator test fixture.",
        cases=[
            BenchmarkCase(
                case_id="low",
                input_data={"decision": "APPROVE"},
                gold_outcome={"decision": "APPROVE"},
                risk_level=RiskLevel.LOW,
                rationale="Low-risk fixture.",
            ),
            BenchmarkCase(
                case_id="high",
                input_data={"decision": "HOLD"},
                gold_outcome={"decision": "HOLD"},
                risk_level=RiskLevel.HIGH,
                rationale="High-risk fixture.",
            ),
            BenchmarkCase(
                case_id="critical",
                input_data={"decision": "REVIEW"},
                gold_outcome={"decision": "REVIEW"},
                risk_level=RiskLevel.CRITICAL,
                rationale="Critical-risk fixture.",
            ),
        ],
    )


def test_perfect_candidate_matches_gold() -> None:
    """A correct candidate should match every gold outcome."""
    report = evaluate_benchmark(
        dataset=build_dataset(),
        candidate=PythonImplementationRunner(
            implementation_id="perfect",
            function=identity_candidate,
        ),
    )

    assert report.cases_evaluated == 3
    assert report.gold_matches == 3
    assert report.gold_agreement_rate == 1.0
    assert report.high_risk_failures == 0
    assert report.execution_errors == 0


def test_high_risk_mismatches_are_counted() -> None:
    """Incorrect high-risk decisions must be visible in the report."""
    report = evaluate_benchmark(
        dataset=build_dataset(),
        candidate=PythonImplementationRunner(
            implementation_id="wrong",
            function=wrong_candidate,
        ),
    )

    assert report.gold_matches == 0
    assert report.high_risk_cases_evaluated == 2
    assert report.high_risk_failures == 2
    assert report.high_risk_agreement_rate == 0.0


def test_execution_errors_are_reported() -> None:
    """Candidate execution errors must remain visible as evidence."""
    report = evaluate_benchmark(
        dataset=build_dataset(),
        candidate=PythonImplementationRunner(
            implementation_id="failing",
            function=failing_candidate,
        ),
    )

    assert report.execution_errors == 3
    assert report.gold_matches == 0
    assert all(
        result.error is not None
        for result in report.case_results
    )
