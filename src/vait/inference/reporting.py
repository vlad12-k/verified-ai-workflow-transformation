"""Persistence helpers for VAIT inference benchmark evidence."""

from pathlib import Path

from vait.inference.models import InferenceBenchmarkReport


def write_inference_report(
    report: InferenceBenchmarkReport,
    path: str | Path,
) -> Path:
    """Write a controlled inference report as formatted JSON."""
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    return report_path
