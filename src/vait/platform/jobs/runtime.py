"""Controlled polling lifecycle for a durable worker."""

import logging
from collections.abc import Callable
from datetime import datetime
from threading import Event
from typing import Protocol

from sqlalchemy.exc import InterfaceError, OperationalError

from vait.platform.jobs.models import utc_now
from vait.platform.jobs.worker import WorkerCycleResult

_LOGGER = logging.getLogger("vait")


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
    max_consecutive_database_errors: int = 3,
) -> None:
    """Poll until shutdown; never interrupt an active cycle."""

    if not 0.01 <= poll_interval_seconds <= 60.0:
        raise ValueError(
            "poll_interval_seconds must be between 0.01 and 60"
        )

    if max_consecutive_database_errors < 1:
        raise ValueError(
            "max_consecutive_database_errors must be positive"
        )

    consecutive_database_errors = 0

    while not stop_event.is_set():
        try:
            result = service.run_once(now=clock())
        except (OperationalError, InterfaceError) as exc:
            consecutive_database_errors += 1
            _LOGGER.warning("worker_database_unavailable")

            if (
                consecutive_database_errors
                >= max_consecutive_database_errors
            ):
                raise RuntimeError(
                    "Worker database unavailable after bounded retries"
                ) from exc

            stop_event.wait(poll_interval_seconds)
            continue

        consecutive_database_errors = 0

        if stop_event.is_set():
            break

        if not result.claimed:
            stop_event.wait(poll_interval_seconds)
