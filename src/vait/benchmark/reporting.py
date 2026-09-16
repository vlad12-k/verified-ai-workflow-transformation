"""Persistence of VAIT benchmark reports."""

from pathlib import Path

from vait.benchmark.models import BenchmarkReport


def write_benchmark_report(
    report: BenchmarkReport,
    path: str | Path,
) -> Path:
    """Write a benchmark report as formatted JSON."""
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    return report_path
