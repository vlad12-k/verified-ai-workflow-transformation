"""Tests for VAIT benchmark quality evaluation."""

from vait.benchmark.loader import load_benchmark
from vait.benchmark.models import BenchmarkCase, BenchmarkDataset
from vait.benchmark.quality import (
    BenchmarkQualityIssueCode,
    BenchmarkQualityPolicy,
    evaluate_benchmark_quality,
)
from vait.contracts.models import RiskLevel


def test_ap_benchmark_passes_declared_quality_policy() -> None:
    """The v0.1 AP benchmark should satisfy its current quality baseline."""
    dataset = load_benchmark(
        "datasets/ap_invoice_exceptions/v0.1/cases.yaml"
    )

    report = evaluate_benchmark_quality(
        dataset=dataset,
        policy=BenchmarkQualityPolicy(
            min_total_cases=12,
            min_cases_per_risk={
                RiskLevel.LOW: 2,
                RiskLevel.MEDIUM: 4,
                RiskLevel.HIGH: 4,
                RiskLevel.CRITICAL: 2,
            },
            required_tags=frozenset(
                {
                    "adversarial",
                    "boundary",
                    "amount-mismatch",
                    "fraud-indicator",
                    "compliance",
                }
            ),
            min_adversarial_cases=2,
            min_boundary_cases=2,
        ),
    )

    assert report.passed is True
    assert report.total_cases == 12
    assert report.adversarial_cases >= 2
    assert report.boundary_cases >= 2
    assert report.duplicate_case_groups == []
    assert report.issues == []


def test_missing_risk_coverage_fails_quality_gate() -> None:
    """Missing declared risk coverage should fail benchmark adequacy."""
    dataset = BenchmarkDataset(
        benchmark_id="risk-gap",
        version="0.1",
        description="Risk coverage failure fixture.",
        cases=[
            BenchmarkCase(
                case_id="low-only",
                input_data={"value": 1},
                gold_outcome={"decision": "A"},
                risk_level=RiskLevel.LOW,
                rationale="Only low-risk coverage.",
            )
        ],
    )

    report = evaluate_benchmark_quality(
        dataset=dataset,
        policy=BenchmarkQualityPolicy(
            min_cases_per_risk={
                RiskLevel.HIGH: 1,
            }
        ),
    )

    assert report.passed is False
    assert any(
        issue.code
        is BenchmarkQualityIssueCode.INSUFFICIENT_RISK_COVERAGE
        for issue in report.issues
    )


def test_duplicate_semantic_cases_are_detected() -> None:
    """Duplicate benchmark semantics should be visible to reviewers."""
    cases = [
        BenchmarkCase(
            case_id="duplicate-a",
            input_data={"value": 1},
            gold_outcome={"decision": "A"},
            risk_level=RiskLevel.HIGH,
            rationale="First duplicate.",
        ),
        BenchmarkCase(
            case_id="duplicate-b",
            input_data={"value": 1},
            gold_outcome={"decision": "A"},
            risk_level=RiskLevel.HIGH,
            rationale="Second duplicate.",
        ),
    ]

    report = evaluate_benchmark_quality(
        dataset=BenchmarkDataset(
            benchmark_id="duplicate-test",
            version="0.1",
            description="Duplicate detection fixture.",
            cases=cases,
        ),
        policy=BenchmarkQualityPolicy(),
    )

    assert report.passed is False
    assert report.duplicate_case_groups == [
        ["duplicate-a", "duplicate-b"]
    ]
    assert any(
        issue.code is BenchmarkQualityIssueCode.DUPLICATE_CASE_CONTENT
        for issue in report.issues
    )


def test_missing_required_tag_fails_quality_gate() -> None:
    """Declared benchmark dimensions must be represented."""
    dataset = BenchmarkDataset(
        benchmark_id="tag-gap",
        version="0.1",
        description="Tag coverage fixture.",
        cases=[
            BenchmarkCase(
                case_id="case-1",
                input_data={},
                gold_outcome={"decision": "A"},
                risk_level=RiskLevel.LOW,
                tags=frozenset({"nominal"}),
                rationale="Tag coverage test.",
            )
        ],
    )

    report = evaluate_benchmark_quality(
        dataset=dataset,
        policy=BenchmarkQualityPolicy(
            required_tags=frozenset({"adversarial"})
        ),
    )

    assert report.passed is False
    assert any(
        issue.code is BenchmarkQualityIssueCode.MISSING_REQUIRED_TAG
        for issue in report.issues
    )
