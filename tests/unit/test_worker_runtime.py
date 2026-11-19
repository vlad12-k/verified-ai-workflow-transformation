"""Unit tests for the bounded worker polling lifecycle."""

from datetime import UTC, datetime
from threading import Event

import pytest
from sqlalchemy.exc import OperationalError

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



def test_worker_recovers_after_transient_database_error() -> None:
    """A transient DB error can recover without executing work twice."""
    stop = Event()

    class IntermittentWorker:
        def __init__(self) -> None:
            self.calls = 0

        def run_once(self, *, now: datetime) -> WorkerCycleResult:
            self.calls += 1

            if self.calls == 1:
                raise OperationalError(
                    "SELECT 1", {}, ConnectionError("offline")
                )

            stop.set()
            return WorkerCycleResult(claimed=False, job=None)

    worker = IntermittentWorker()

    run_worker_loop(
        service=worker,
        stop_event=stop,
        poll_interval_seconds=0.01,
    )

    assert worker.calls == 2


def test_worker_stops_after_bounded_database_errors() -> None:
    """Persistent database loss must not spin forever."""
    stop = Event()

    class OfflineWorker:
        def __init__(self) -> None:
            self.calls = 0

        def run_once(self, *, now: datetime) -> WorkerCycleResult:
            self.calls += 1
            raise OperationalError(
                "SELECT 1", {}, ConnectionError("offline")
            )

    worker = OfflineWorker()

    with pytest.raises(RuntimeError, match="bounded retries"):
        run_worker_loop(
            service=worker,
            stop_event=stop,
            poll_interval_seconds=0.01,
            max_consecutive_database_errors=3,
        )

    assert worker.calls == 3


def test_database_error_budget_resets_after_success() -> None:
    """An intervening healthy cycle resets the failure counter."""
    stop = Event()

    class IntermittentWorker:
        def __init__(self) -> None:
            self.calls = 0

        def run_once(self, *, now: datetime) -> WorkerCycleResult:
            self.calls += 1

            if self.calls in (1, 3):
                raise OperationalError(
                    "SELECT 1", {}, ConnectionError("offline")
                )

            if self.calls == 4:
                stop.set()

            return WorkerCycleResult(claimed=False, job=None)

    worker = IntermittentWorker()

    run_worker_loop(
        service=worker,
        stop_event=stop,
        poll_interval_seconds=0.01,
        max_consecutive_database_errors=2,
    )

    assert worker.calls == 4


def test_non_database_worker_error_still_propagates() -> None:
    """Programming failures must not be mistaken for DB outages."""
    stop = Event()

    class BrokenWorker:
        def run_once(self, *, now: datetime) -> WorkerCycleResult:
            raise ValueError("invalid executor")

    with pytest.raises(ValueError, match="invalid executor"):
        run_worker_loop(
            service=BrokenWorker(),
            stop_event=stop,
            poll_interval_seconds=0.01,
        )
