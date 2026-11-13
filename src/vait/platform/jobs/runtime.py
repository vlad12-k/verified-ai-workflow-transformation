"""Controlled polling lifecycle for a durable worker."""

from collections.abc import Callable
from datetime import datetime
from threading import Event
from typing import Protocol

from vait.platform.jobs.models import utc_now
from vait.platform.jobs.worker import WorkerCycleResult


class WorkerRunner(Protocol):
    """Execute one durable job polling cycle."""

    def run_once(self, *, now: datetime) -> WorkerCycleResult:
        ...


def run_worker_loop(
    *,
    service: WorkerRunner,
    stop_event: Event,
    poll_interval_seconds: float = 1.0,
    clock: Callable[[], datetime] = utc_now,
) -> None:
    """Poll until shutdown; never interrupt an active cycle."""

    if not 0.01 <= poll_interval_seconds <= 60.0:
        raise ValueError(
            "poll_interval_seconds must be between 0.01 and 60"
        )

    while not stop_event.is_set():
        result = service.run_once(now=clock())

        if stop_event.is_set():
            break

        if not result.claimed:
            stop_event.wait(poll_interval_seconds)
