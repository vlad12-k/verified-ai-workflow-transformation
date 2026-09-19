"""Portable process-resource evidence for controlled inference profiling."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from math import isfinite
from time import process_time_ns

try:
    import resource
except ImportError:  # pragma: no cover - unavailable on non-POSIX platforms
    resource = None  # type: ignore[assignment]

from vait.inference.models import InferenceResourceEvidence


@dataclass(frozen=True, slots=True)
class ProcessResourceSnapshot:
    """One process-resource snapshot around a controlled measurement interval."""

    process_cpu_time_ns: int
    process_peak_rss_bytes: int | None


def capture_process_resource_snapshot() -> ProcessResourceSnapshot:
    """Capture process CPU time and process-lifetime peak resident memory."""
    return ProcessResourceSnapshot(
        process_cpu_time_ns=process_time_ns(),
        process_peak_rss_bytes=_process_peak_rss_bytes(),
    )


def build_process_resource_evidence(
    *,
    started: ProcessResourceSnapshot,
    ended: ProcessResourceSnapshot,
    elapsed_seconds: float,
    measurement_scope: str,
) -> InferenceResourceEvidence:
    """Build normalised resource evidence from two process snapshots."""
    if elapsed_seconds < 0.0 or not isfinite(elapsed_seconds):
        raise ValueError(
            "Elapsed resource measurement time must be finite "
            "and non-negative."
        )

    cpu_delta_ns = (
        ended.process_cpu_time_ns
        - started.process_cpu_time_ns
    )

    if cpu_delta_ns < 0:
        raise ValueError(
            "Process CPU time cannot decrease between snapshots."
        )

    rss_growth_bytes: int | None = None

    if (
        started.process_peak_rss_bytes is not None
        and ended.process_peak_rss_bytes is not None
    ):
        if (
            ended.process_peak_rss_bytes
            < started.process_peak_rss_bytes
        ):
            raise ValueError(
                "Process peak RSS cannot decrease between snapshots."
            )

        rss_growth_bytes = (
            ended.process_peak_rss_bytes
            - started.process_peak_rss_bytes
        )

    process_cpu_time_ms = cpu_delta_ns / 1_000_000
    wall_time_ms = elapsed_seconds * 1_000

    ratio = (
        (process_cpu_time_ms / wall_time_ms)
        if wall_time_ms > 0.0
        else None
    )

    return InferenceResourceEvidence(
        measurement_scope=measurement_scope,
        wall_time_ms=wall_time_ms,
        process_cpu_time_ms=process_cpu_time_ms,
        cpu_time_to_wall_time_ratio=ratio,
        process_peak_rss_baseline_bytes=(
            started.process_peak_rss_bytes
        ),
        process_peak_rss_bytes=(
            ended.process_peak_rss_bytes
        ),
        process_peak_rss_growth_bytes=rss_growth_bytes,
    )


def _process_peak_rss_bytes() -> int | None:
    """Return process-lifetime peak RSS normalised to bytes when supported."""
    if resource is None:
        return None

    usage = resource.getrusage(
        resource.RUSAGE_SELF
    )
    raw_peak_rss = int(
        usage.ru_maxrss
    )

    if raw_peak_rss <= 0:
        return None

    system = platform.system()

    if system == "Darwin":
        return raw_peak_rss

    if system == "Linux":
        return raw_peak_rss * 1024

    return None
