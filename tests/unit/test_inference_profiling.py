"""Tests for controlled inference process-resource profiling."""

import pytest

from vait.inference.profiling import (
    ProcessResourceSnapshot,
    build_process_resource_evidence,
    capture_process_resource_snapshot,
)


def test_resource_evidence_calculates_cpu_and_rss_growth() -> None:
    """Resource evidence should preserve interval CPU and RSS semantics."""
    started = ProcessResourceSnapshot(
        process_cpu_time_ns=1_000_000_000,
        process_peak_rss_bytes=100_000,
    )
    ended = ProcessResourceSnapshot(
        process_cpu_time_ns=1_250_000_000,
        process_peak_rss_bytes=160_000,
    )

    evidence = build_process_resource_evidence(
        started=started,
        ended=ended,
        elapsed_seconds=0.5,
        measurement_scope="controlled-inference-measured-region",
    )

    assert evidence.wall_time_ms == pytest.approx(500.0)
    assert evidence.process_cpu_time_ms == pytest.approx(250.0)

    assert (
        evidence.cpu_time_to_wall_time_ratio
        == pytest.approx(0.5)
    )

    assert (
        evidence.process_peak_rss_baseline_bytes
        == 100_000
    )
    assert evidence.process_peak_rss_bytes == 160_000
    assert (
        evidence.process_peak_rss_growth_bytes
        == 60_000
    )

    assert (
        evidence.peak_rss_scope
        == "process-lifetime-high-water-mark"
    )


def test_resource_evidence_preserves_unavailable_rss() -> None:
    """Unsupported RSS collection must remain explicit rather than invented."""
    started = ProcessResourceSnapshot(
        process_cpu_time_ns=10,
        process_peak_rss_bytes=None,
    )
    ended = ProcessResourceSnapshot(
        process_cpu_time_ns=20,
        process_peak_rss_bytes=None,
    )

    evidence = build_process_resource_evidence(
        started=started,
        ended=ended,
        elapsed_seconds=0.001,
        measurement_scope="test",
    )

    assert evidence.process_peak_rss_baseline_bytes is None
    assert evidence.process_peak_rss_bytes is None
    assert evidence.process_peak_rss_growth_bytes is None


def test_zero_wall_time_does_not_invent_cpu_ratio() -> None:
    """A zero-duration synthetic interval cannot support a CPU/wall ratio."""
    snapshot = ProcessResourceSnapshot(
        process_cpu_time_ns=100,
        process_peak_rss_bytes=None,
    )

    evidence = build_process_resource_evidence(
        started=snapshot,
        ended=snapshot,
        elapsed_seconds=0.0,
        measurement_scope="test",
    )

    assert evidence.wall_time_ms == 0.0
    assert evidence.cpu_time_to_wall_time_ratio is None


def test_resource_evidence_rejects_decreasing_cpu_time() -> None:
    """Invalid snapshot ordering must fail instead of producing evidence."""
    with pytest.raises(
        ValueError,
        match="CPU time cannot decrease",
    ):
        build_process_resource_evidence(
            started=ProcessResourceSnapshot(
                process_cpu_time_ns=20,
                process_peak_rss_bytes=None,
            ),
            ended=ProcessResourceSnapshot(
                process_cpu_time_ns=10,
                process_peak_rss_bytes=None,
            ),
            elapsed_seconds=1.0,
            measurement_scope="test",
        )


def test_resource_evidence_rejects_decreasing_peak_rss() -> None:
    """Process-lifetime peak RSS is monotonic within one process."""
    with pytest.raises(
        ValueError,
        match="peak RSS cannot decrease",
    ):
        build_process_resource_evidence(
            started=ProcessResourceSnapshot(
                process_cpu_time_ns=10,
                process_peak_rss_bytes=200,
            ),
            ended=ProcessResourceSnapshot(
                process_cpu_time_ns=20,
                process_peak_rss_bytes=100,
            ),
            elapsed_seconds=1.0,
            measurement_scope="test",
        )


def test_capture_process_resource_snapshot_is_valid() -> None:
    """Real process snapshots should expose non-negative portable evidence."""
    snapshot = capture_process_resource_snapshot()

    assert snapshot.process_cpu_time_ns >= 0

    if snapshot.process_peak_rss_bytes is not None:
        assert snapshot.process_peak_rss_bytes > 0
