"""Tests for benchmark gold-label execution."""

from vait.benchmark.gold_runner import GoldBenchmarkRunner
from vait.contracts.models import VerificationCase


def test_gold_runner_returns_declared_outcome() -> None:
    """Known benchmark cases should return their gold outcome."""
    runner = GoldBenchmarkRunner(
        implementation_id="gold",
        gold_outcomes={
            "case-1": {"decision": "HOLD"},
        },
    )

    observation = runner.execute(
        VerificationCase(
            id="case-1",
            input_data={},
        )
    )

    assert observation.error is None
    assert observation.output == {"decision": "HOLD"}


def test_gold_runner_reports_missing_case() -> None:
    """Unknown cases must not silently receive fabricated gold labels."""
    runner = GoldBenchmarkRunner(
        implementation_id="gold",
        gold_outcomes={},
    )

    observation = runner.execute(
        VerificationCase(
            id="missing",
            input_data={},
        )
    )

    assert observation.output is None
    assert observation.error is not None
    assert "Missing gold outcome" in observation.error
