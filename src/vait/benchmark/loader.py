"""Loading utilities for VAIT benchmark datasets."""

from pathlib import Path

import yaml

from vait.benchmark.models import BenchmarkDataset


def load_benchmark(path: str | Path) -> BenchmarkDataset:
    """Load and validate a benchmark dataset from YAML."""
    benchmark_path = Path(path)

    if not benchmark_path.is_file():
        raise FileNotFoundError(
            f"Benchmark dataset does not exist: {benchmark_path}"
        )

    raw: object = yaml.safe_load(
        benchmark_path.read_text(encoding="utf-8")
    )

    return BenchmarkDataset.model_validate(raw)
