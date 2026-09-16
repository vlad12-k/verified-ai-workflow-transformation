"""Quality and adequacy checks for VAIT benchmark datasets."""

import json
from collections import Counter
from enum import StrEnum

from pydantic import BaseModel, Field

from vait.benchmark.models import BenchmarkCase, BenchmarkDataset
from vait.contracts.models import RiskLevel


class BenchmarkQualityIssueCode(StrEnum):
    """Machine-readable benchmark quality issue."""

    INSUFFICIENT_TOTAL_CASES = "insufficient_total_cases"
    INSUFFICIENT_RISK_COVERAGE = "insufficient_risk_coverage"
    MISSING_REQUIRED_TAG = "missing_required_tag"
    INSUFFICIENT_ADVERSARIAL_CASES = "insufficient_adversarial_cases"
    INSUFFICIENT_BOUNDARY_CASES = "insufficient_boundary_cases"
    DUPLICATE_CASE_CONTENT = "duplicate_case_content"


class BenchmarkQualityPolicy(BaseModel):
    """Declared benchmark adequacy requirements."""

    min_total_cases: int = Field(default=1, ge=1)

    min_cases_per_risk: dict[RiskLevel, int] = Field(
        default_factory=dict
    )

    required_tags: frozenset[str] = Field(default_factory=frozenset)

    min_adversarial_cases: int = Field(default=0, ge=0)
    min_boundary_cases: int = Field(default=0, ge=0)

    allow_exact_duplicates: bool = False


class BenchmarkQualityIssue(BaseModel):
    """One benchmark quality failure."""

    code: BenchmarkQualityIssueCode
    message: str


class BenchmarkQualityReport(BaseModel):
    """Machine-readable benchmark quality report."""

    benchmark_id: str
    benchmark_version: str

    passed: bool

    total_cases: int = Field(ge=0)
    risk_counts: dict[RiskLevel, int]
    tag_counts: dict[str, int]

    adversarial_cases: int = Field(ge=0)
    boundary_cases: int = Field(ge=0)

    duplicate_case_groups: list[list[str]] = Field(default_factory=list)
    issues: list[BenchmarkQualityIssue] = Field(default_factory=list)


def evaluate_benchmark_quality(
    dataset: BenchmarkDataset,
    policy: BenchmarkQualityPolicy,
) -> BenchmarkQualityReport:
    """Evaluate benchmark composition against declared quality requirements."""
    risk_counts = Counter(case.risk_level for case in dataset.cases)

    tag_counts: Counter[str] = Counter(
        tag
        for case in dataset.cases
        for tag in case.tags
    )

    duplicate_groups = _find_duplicate_case_groups(dataset.cases)

    issues: list[BenchmarkQualityIssue] = []

    if len(dataset.cases) < policy.min_total_cases:
        issues.append(
            BenchmarkQualityIssue(
                code=BenchmarkQualityIssueCode.INSUFFICIENT_TOTAL_CASES,
                message=(
                    f"Benchmark contains {len(dataset.cases)} cases; "
                    f"at least {policy.min_total_cases} are required."
                ),
            )
        )

    for risk_level, minimum in policy.min_cases_per_risk.items():
        actual = risk_counts[risk_level]

        if actual < minimum:
            issues.append(
                BenchmarkQualityIssue(
                    code=(
                        BenchmarkQualityIssueCode.INSUFFICIENT_RISK_COVERAGE
                    ),
                    message=(
                        f"Risk stratum '{risk_level.value}' contains "
                        f"{actual} cases; at least {minimum} are required."
                    ),
                )
            )

    for required_tag in sorted(policy.required_tags):
        if tag_counts[required_tag] == 0:
            issues.append(
                BenchmarkQualityIssue(
                    code=BenchmarkQualityIssueCode.MISSING_REQUIRED_TAG,
                    message=(
                        f"Required benchmark tag '{required_tag}' "
                        "is not represented."
                    ),
                )
            )

    adversarial_cases = tag_counts["adversarial"]
    boundary_cases = tag_counts["boundary"]

    if adversarial_cases < policy.min_adversarial_cases:
        issues.append(
            BenchmarkQualityIssue(
                code=(
                    BenchmarkQualityIssueCode.INSUFFICIENT_ADVERSARIAL_CASES
                ),
                message=(
                    f"Benchmark contains {adversarial_cases} adversarial "
                    f"cases; at least {policy.min_adversarial_cases} "
                    "are required."
                ),
            )
        )

    if boundary_cases < policy.min_boundary_cases:
        issues.append(
            BenchmarkQualityIssue(
                code=BenchmarkQualityIssueCode.INSUFFICIENT_BOUNDARY_CASES,
                message=(
                    f"Benchmark contains {boundary_cases} boundary cases; "
                    f"at least {policy.min_boundary_cases} are required."
                ),
            )
        )

    if duplicate_groups and not policy.allow_exact_duplicates:
        issues.append(
            BenchmarkQualityIssue(
                code=BenchmarkQualityIssueCode.DUPLICATE_CASE_CONTENT,
                message=(
                    "Benchmark contains duplicate semantic case content."
                ),
            )
        )

    return BenchmarkQualityReport(
        benchmark_id=dataset.benchmark_id,
        benchmark_version=dataset.version,
        passed=not issues,
        total_cases=len(dataset.cases),
        risk_counts=dict(risk_counts),
        tag_counts=dict(sorted(tag_counts.items())),
        adversarial_cases=adversarial_cases,
        boundary_cases=boundary_cases,
        duplicate_case_groups=duplicate_groups,
        issues=issues,
    )


def _find_duplicate_case_groups(
    cases: list[BenchmarkCase],
) -> list[list[str]]:
    """Find cases with identical input, gold outcome, and risk semantics."""
    signatures: dict[str, list[str]] = {}

    for case in cases:
        payload = {
            "input_data": case.input_data,
            "gold_outcome": case.gold_outcome,
            "risk_level": case.risk_level.value,
        }

        signature = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        signatures.setdefault(signature, []).append(case.case_id)

    return [
        sorted(case_ids)
        for case_ids in signatures.values()
        if len(case_ids) > 1
    ]
