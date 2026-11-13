"""Unit tests for the bounded worker polling lifecycle."""

from datetime import UTC, datetime
from threading import Event

import pytest

from vait.platform.jobs.runtime import run_worker_loop
from vait.platform.jobs.worker import WorkerCycleResult


class FakeWorker:
    def __init__(self, stop_event: Event) -> None:
        self.stop_event = stop_event
        self.calls = 0

    def run_once(self, *, now: datetime) -> WorkerCycleResult:
        assert now.tzinfo is not None
        self.calls += 1
        self.stop_event.set()
        return WorkerCycleResult(claimed=True, job=None)


def test_shutdown_before_poll_does_not_claim() -> None:
    stop = Event()
    stop.set()
    worker = FakeWorker(stop)

    run_worker_loop(service=worker, stop_event=stop)

    assert worker.calls == 0


def test_shutdown_after_active_cycle() -> None:
    stop = Event()
    worker = FakeWorker(stop)

    run_worker_loop(
        service=worker,
        stop_event=stop,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert worker.calls == 1


@pytest.mark.parametrize("interval", [0.0, -1.0, 60.01])
def test_invalid_poll_interval_is_rejected(
    interval: float,
) -> None:
    stop = Event()
    worker = FakeWorker(stop)

    with pytest.raises(ValueError):
        run_worker_loop(
            service=worker,
            stop_event=stop,
            poll_interval_seconds=interval,
        )

    assert worker.calls == 0
