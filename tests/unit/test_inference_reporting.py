"""Tests for inference evidence persistence."""

import json
from pathlib import Path

from vait.inference.benchmark import (
    build_inference_benchmark_report,
    build_throughput_summary,
)
from vait.inference.models import (
    InferenceConfiguration,
    InferenceEnvironment,
)
from vait.inference.reporting import write_inference_report


def test_write_inference_report_persists_versioned_json(
    tmp_path: Path,
) -> None:
    """Inference evidence should persist as machine-readable JSON."""
    report = build_inference_benchmark_report(
        candidate_implementation_id="candidate-v1",
        task_family="structured-decision",
        configuration=InferenceConfiguration(
            provider="local",
            runtime="python",
            device="cpu",
            dtype="native",
            batch_size=1,
        ),
        warmup_iterations=2,
        latency_samples_ms=[1.0, 2.0, 3.0],
        throughput=build_throughput_summary(
            elapsed_seconds=1.0,
            cases_processed=3,
            requests_processed=3,
        ),
        environment=InferenceEnvironment(
            python_version="3.12.0",
            platform="test-platform",
            system="test-system",
            machine="test-machine",
            processor="test-processor",
        ),
    )

    output = tmp_path / "inference-evidence.json"

    written = write_inference_report(
        report=report,
        path=output,
    )

    payload = json.loads(
        written.read_text(encoding="utf-8")
    )

    assert written == output
    assert payload["schema_version"] == "0.1"
    assert payload["candidate_implementation_id"] == "candidate-v1"
    assert payload["task_family"] == "structured-decision"
    assert payload["measured_iterations"] == 3
