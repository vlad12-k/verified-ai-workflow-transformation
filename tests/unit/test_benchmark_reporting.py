"""Tests for VAIT benchmark report persistence."""

import json
from pathlib import Path

from vait.benchmark.models import BenchmarkReport
from vait.benchmark.reporting import write_benchmark_report


def test_benchmark_report_is_written_as_json(
    tmp_path: Path,
) -> None:
    """Benchmark reports should be persisted as machine-readable JSON."""
    report = BenchmarkReport(
        benchmark_id="report-test",
        benchmark_version="0.1",
        implementation_id="candidate",
        cases_evaluated=10,
        gold_matches=9,
        gold_agreement_rate=0.9,
        high_risk_cases_evaluated=4,
        high_risk_failures=1,
        high_risk_agreement_rate=0.75,
        execution_errors=0,
    )

    output_path = write_benchmark_report(
        report,
        tmp_path / "nested" / "report.json",
    )

    payload = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert output_path.is_file()
    assert payload["benchmark_id"] == "report-test"
    assert payload["gold_matches"] == 9
    assert payload["high_risk_failures"] == 1
