"""Tests for versioned benchmark quality policies."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from vait.benchmark.loader import load_benchmark
from vait.benchmark.quality_policy import (
    evaluate_versioned_benchmark_quality,
    load_quality_policy,
    quality_policy_fingerprint,
)
from vait.benchmark.reporting import write_benchmark_report


def test_v02_quality_policy_loads() -> None:
    """The AP v0.2 quality policy should be independently versioned."""
    document = load_quality_policy(
        "datasets/ap_invoice_exceptions/v0.2/quality-policy.yaml"
    )

    assert document.benchmark_id == "ap-invoice-exceptions"
    assert document.benchmark_version == "0.2"
    assert document.policy.min_total_cases == 20
    assert document.policy.min_adversarial_cases == 9
    assert document.policy.min_boundary_cases == 5


def test_quality_policy_fingerprint_is_deterministic() -> None:
    """Unchanged policy content should retain the same fingerprint."""
    document = load_quality_policy(
        "datasets/ap_invoice_exceptions/v0.2/quality-policy.yaml"
    )

    first = quality_policy_fingerprint(document)
    second = quality_policy_fingerprint(document)

    assert first == second
    assert len(first) == 64


def test_policy_version_mismatch_is_rejected() -> None:
    """A quality policy must not silently target another dataset version."""
    dataset = load_benchmark(
        "datasets/ap_invoice_exceptions/v0.1/cases.yaml"
    )
    document = load_quality_policy(
        "datasets/ap_invoice_exceptions/v0.2/quality-policy.yaml"
    )

    with pytest.raises(
        ValueError,
        match="version does not match",
    ):
        evaluate_versioned_benchmark_quality(
            dataset=dataset,
            document=document,
        )


def test_versioned_quality_artifact_contains_fingerprints() -> None:
    """Quality evidence should identify both dataset and policy content."""
    dataset = load_benchmark(
        "datasets/ap_invoice_exceptions/v0.2/cases.yaml"
    )
    document = load_quality_policy(
        "datasets/ap_invoice_exceptions/v0.2/quality-policy.yaml"
    )

    artifact = evaluate_versioned_benchmark_quality(
        dataset=dataset,
        document=document,
    )

    assert artifact.quality.passed is True
    assert len(artifact.dataset_fingerprint_sha256) == 64
    assert len(artifact.quality_policy_fingerprint_sha256) == 64


def test_quality_artifact_is_written_as_json(
    tmp_path: Path,
) -> None:
    """A versioned quality artifact should be machine-readable."""
    dataset = load_benchmark(
        "datasets/ap_invoice_exceptions/v0.2/cases.yaml"
    )
    document = load_quality_policy(
        "datasets/ap_invoice_exceptions/v0.2/quality-policy.yaml"
    )

    artifact = evaluate_versioned_benchmark_quality(
        dataset=dataset,
        document=document,
    )

    output_path = write_benchmark_report(
        artifact,
        tmp_path / "quality.json",
    )

    payload = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert payload["benchmark_version"] == "0.2"
    assert payload["quality"]["passed"] is True
    assert payload["policy"]["min_adversarial_cases"] == 9
    assert len(payload["dataset_fingerprint_sha256"]) == 64
    assert len(payload["quality_policy_fingerprint_sha256"]) == 64


def test_quality_policy_fingerprint_is_stable_across_hash_seeds() -> None:
    """Policy fingerprints must remain stable across Python processes."""
    script = (
        "from vait.benchmark.quality_policy import "
        "load_quality_policy, quality_policy_fingerprint; "
        "document = load_quality_policy("
        "'datasets/ap_invoice_exceptions/v0.2/quality-policy.yaml'); "
        "print(quality_policy_fingerprint(document))"
    )

    fingerprints: list[str] = []

    for seed in ("1", "2", "3", "42"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed

        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )

        fingerprints.append(completed.stdout.strip())

    assert len(set(fingerprints)) == 1
