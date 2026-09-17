"""Tests for provider-neutral economic evidence."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from vait.benchmark.models import LatencySummary
from vait.economics.metrics import (
    CostEstimate,
    CostEvidenceKind,
    EconomicEvidence,
    compare_cost_estimates,
    compare_latency_summaries,
)


def build_cost(
    amount: float,
    *,
    currency: str = "USD",
) -> CostEstimate:
    """Create one deterministic cost fixture."""
    return CostEstimate(
        amount_per_case=amount,
        currency=currency,
        evidence_kind=CostEvidenceKind.ESTIMATED,
        source="controlled-development-scenario",
        pricing_version="scenario-v0.1",
        observed_at=datetime(
            2026,
            9,
            17,
            tzinfo=UTC,
        ),
    )


def test_cost_comparison_reports_candidate_delta() -> None:
    """Candidate-minus-reference deltas should be explicit."""
    comparison = compare_cost_estimates(
        reference=build_cost(0.10),
        candidate=build_cost(0.02),
    )

    assert comparison.delta_per_case == pytest.approx(
        -0.08
    )
    assert comparison.relative_delta == pytest.approx(
        -0.8
    )


def test_zero_reference_cost_has_no_relative_delta() -> None:
    """Relative savings are undefined for a zero-cost reference."""
    comparison = compare_cost_estimates(
        reference=build_cost(0.0),
        candidate=build_cost(0.01),
    )

    assert comparison.delta_per_case == pytest.approx(
        0.01
    )
    assert comparison.relative_delta is None


def test_cost_comparison_rejects_currency_mismatch() -> None:
    """Unnormalised currencies must not be compared."""
    with pytest.raises(
        ValueError,
        match="same currency",
    ):
        compare_cost_estimates(
            reference=build_cost(
                0.10,
                currency="USD",
            ),
            candidate=build_cost(
                0.08,
                currency="GBP",
            ),
        )


def test_cost_estimate_requires_normalised_currency() -> None:
    """Currency codes should use explicit uppercase notation."""
    with pytest.raises(ValidationError):
        build_cost(
            0.10,
            currency="usd",
        )


def test_latency_comparison_reports_deltas() -> None:
    """Latency comparison should report candidate-minus-reference."""
    reference = LatencySummary(
        count=10,
        mean_ms=100.0,
        p50_ms=90.0,
        p95_ms=150.0,
        max_ms=180.0,
    )
    candidate = LatencySummary(
        count=10,
        mean_ms=40.0,
        p50_ms=35.0,
        p95_ms=60.0,
        max_ms=80.0,
    )

    comparison = compare_latency_summaries(
        reference=reference,
        candidate=candidate,
    )

    assert comparison.mean_delta_ms == pytest.approx(
        -60.0
    )
    assert comparison.p50_delta_ms == pytest.approx(
        -55.0
    )
    assert comparison.p95_delta_ms == pytest.approx(
        -90.0
    )
    assert comparison.max_delta_ms == pytest.approx(
        -100.0
    )


def test_economic_evidence_requires_content() -> None:
    """Empty evidence containers should be invalid."""
    with pytest.raises(
        ValidationError,
        match="requires cost or latency evidence",
    ):
        EconomicEvidence()
