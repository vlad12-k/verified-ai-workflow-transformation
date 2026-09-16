"""Reproducibility metadata for VAIT benchmark experiments."""

import hashlib
import json
import platform
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from math import ceil
from statistics import fmean
from uuid import uuid4

from pydantic import JsonValue

from vait.benchmark.models import (
    BenchmarkDataset,
    ExperimentProvenance,
    LatencySummary,
)
from vait.runners.base import ImplementationRunner

_TRACKED_PACKAGES = (
    "verified-ai-workflow-transformation",
    "pydantic",
    "numpy",
    "scipy",
    "pandas",
    "PyYAML",
    "scikit-learn",
    "xgboost",
)


def benchmark_fingerprint(dataset: BenchmarkDataset) -> str:
    """Return a deterministic SHA-256 fingerprint for benchmark content."""
    canonical_cases = [
        {
            "case_id": case.case_id,
            "input_data": case.input_data,
            "gold_outcome": case.gold_outcome,
            "risk_level": case.risk_level.value,
            "tags": sorted(case.tags),
            "rationale": case.rationale,
        }
        for case in sorted(
            dataset.cases,
            key=lambda item: item.case_id,
        )
    ]

    payload = {
        "benchmark_id": dataset.benchmark_id,
        "version": dataset.version,
        "description": dataset.description,
        "cases": canonical_cases,
    }

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def summarize_latencies(
    latencies_ms: Iterable[float],
) -> LatencySummary | None:
    """Summarize observed execution latencies."""
    values = sorted(float(value) for value in latencies_ms)

    if not values:
        return None

    return LatencySummary(
        count=len(values),
        mean_ms=fmean(values),
        p50_ms=_nearest_rank(values, 0.50),
        p95_ms=_nearest_rank(values, 0.95),
        max_ms=values[-1],
    )


def build_experiment_provenance(
    dataset: BenchmarkDataset,
    candidate: ImplementationRunner,
    candidate_configuration: Mapping[str, JsonValue] | None = None,
) -> ExperimentProvenance:
    """Create reproducibility metadata for one benchmark execution."""
    return ExperimentProvenance(
        run_id=str(uuid4()),
        created_at=datetime.now(UTC),
        dataset_fingerprint_sha256=benchmark_fingerprint(dataset),
        candidate_implementation_id=candidate.implementation_id,
        candidate_runner_type=(
            f"{candidate.__class__.__module__}."
            f"{candidate.__class__.__qualname__}"
        ),
        candidate_configuration=dict(candidate_configuration or {}),
        python_version=platform.python_version(),
        platform=platform.platform(),
        package_versions=_installed_package_versions(),
    )


def _nearest_rank(
    sorted_values: list[float],
    percentile: float,
) -> float:
    """Return a nearest-rank percentile from sorted observations."""
    index = ceil(percentile * len(sorted_values)) - 1
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def _installed_package_versions() -> dict[str, str]:
    """Return versions of packages relevant to benchmark reproduction."""
    versions: dict[str, str] = {}

    for package_name in _TRACKED_PACKAGES:
        try:
            versions[package_name] = package_version(package_name)
        except PackageNotFoundError:
            versions[package_name] = "not-installed"

    return versions
