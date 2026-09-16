"""Tests for versioned VAIT benchmark loading."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from vait.benchmark.loader import load_benchmark
from vait.contracts.models import RiskLevel

_DATASET_PATH = Path(
    "datasets/ap_invoice_exceptions/v0.1/cases.yaml"
)


def test_ap_benchmark_loads() -> None:
    """The versioned AP benchmark should load and validate."""
    dataset = load_benchmark(_DATASET_PATH)

    assert dataset.benchmark_id == "ap-invoice-exceptions"
    assert dataset.version == "0.1"
    assert len(dataset.cases) == 12

    case_ids = {case.case_id for case in dataset.cases}

    assert "standard-po-match" in case_ids
    assert "changed-bank-details" in case_ids
    assert "multiple-risk-indicators" in case_ids


def test_ap_benchmark_contains_high_risk_cases() -> None:
    """The benchmark must contain explicit high-risk evaluation coverage."""
    dataset = load_benchmark(_DATASET_PATH)

    high_risk_cases = [
        case
        for case in dataset.cases
        if case.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}
    ]

    assert high_risk_cases
    assert any(
        case.risk_level is RiskLevel.CRITICAL
        for case in high_risk_cases
    )


def test_gold_outcome_is_explicit() -> None:
    """Benchmark truth must be represented independently of model output."""
    dataset = load_benchmark(_DATASET_PATH)

    duplicate_case = next(
        case
        for case in dataset.cases
        if case.case_id == "duplicate-invoice"
    )

    assert duplicate_case.gold_outcome == {"decision": "HOLD"}


def test_duplicate_case_ids_are_rejected(tmp_path: Path) -> None:
    """Duplicate benchmark identifiers must fail dataset validation."""
    benchmark = tmp_path / "duplicate.yaml"

    benchmark.write_text(
        """
benchmark_id: duplicate-test
version: "0.1"
description: Duplicate ID validation test.
cases:
  - case_id: duplicate
    input_data: {}
    gold_outcome:
      decision: A
    risk_level: low
    rationale: First case.
  - case_id: duplicate
    input_data: {}
    gold_outcome:
      decision: B
    risk_level: high
    rationale: Second case.
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ValidationError,
        match="Benchmark case IDs must be unique",
    ):
        load_benchmark(benchmark)


def test_missing_benchmark_file_is_rejected(
    tmp_path: Path,
) -> None:
    """Loading a missing benchmark path must fail explicitly."""
    missing = tmp_path / "missing.yaml"

    with pytest.raises(
        FileNotFoundError,
        match="Benchmark dataset does not exist",
    ):
        load_benchmark(missing)
