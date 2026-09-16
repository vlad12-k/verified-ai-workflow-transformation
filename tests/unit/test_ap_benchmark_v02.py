"""Tests for the expanded AP benchmark v0.2."""

from vait.benchmark.loader import load_benchmark
from vait.benchmark.quality import (
    BenchmarkQualityPolicy,
    evaluate_benchmark_quality,
)
from vait.contracts.models import RiskLevel


def test_ap_v02_loads_with_expected_expansion() -> None:
    """AP benchmark v0.2 should preserve v0.1 and add eight cases."""
    baseline = load_benchmark(
        "datasets/ap_invoice_exceptions/v0.1/cases.yaml"
    )
    expanded = load_benchmark(
        "datasets/ap_invoice_exceptions/v0.2/cases.yaml"
    )

    assert baseline.version == "0.1"
    assert expanded.version == "0.2"

    assert len(baseline.cases) == 12
    assert len(expanded.cases) == 20

    baseline_ids = {
        case.case_id
        for case in baseline.cases
    }
    expanded_ids = {
        case.case_id
        for case in expanded.cases
    }

    assert baseline_ids < expanded_ids


def test_ap_v02_has_stronger_adversarial_coverage() -> None:
    """The expanded benchmark should satisfy a stricter quality policy."""
    dataset = load_benchmark(
        "datasets/ap_invoice_exceptions/v0.2/cases.yaml"
    )

    report = evaluate_benchmark_quality(
        dataset=dataset,
        policy=BenchmarkQualityPolicy(
            min_total_cases=20,
            min_cases_per_risk={
                RiskLevel.LOW: 3,
                RiskLevel.MEDIUM: 6,
                RiskLevel.HIGH: 7,
                RiskLevel.CRITICAL: 4,
            },
            required_tags=frozenset(
                {
                    "adversarial",
                    "boundary",
                    "precision",
                    "inconsistent-input",
                    "precedence",
                    "compliance",
                    "fraud-indicator",
                }
            ),
            min_adversarial_cases=9,
            min_boundary_cases=5,
        ),
    )

    assert report.passed is True
    assert report.total_cases == 20
    assert report.adversarial_cases == 9
    assert report.boundary_cases == 5
    assert report.duplicate_case_groups == []
    assert report.issues == []


def test_ap_v02_contains_precedence_cases() -> None:
    """The expanded benchmark should explicitly test rule precedence."""
    dataset = load_benchmark(
        "datasets/ap_invoice_exceptions/v0.2/cases.yaml"
    )

    precedence_cases = [
        case
        for case in dataset.cases
        if "precedence" in case.tags
    ]

    assert len(precedence_cases) >= 4
    assert any(
        case.risk_level is RiskLevel.CRITICAL
        for case in precedence_cases
    )
