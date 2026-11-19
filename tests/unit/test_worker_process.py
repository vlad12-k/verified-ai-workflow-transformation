"""Tests for signal-managed worker lifecycle."""

import signal
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Event
from unittest.mock import patch

import pytest

from vait.platform.jobs.process import run_worker_process
from vait.platform.jobs.worker import WorkerCycleResult


class OneCycleWorker:
    def __init__(self, stop: Event) -> None:
        self.stop = stop
        self.calls = 0

    def run_once(self, *, now: datetime) -> WorkerCycleResult:
        assert now.tzinfo is not None
        self.calls += 1
        self.stop.set()
        return WorkerCycleResult(claimed=False, job=None)


class FailingWorker:
    def run_once(self, *, now: datetime) -> WorkerCycleResult:
        raise RuntimeError("Simulated worker failure")


def test_signal_handlers_are_restored() -> None:
    stop = Event()
    worker = OneCycleWorker(stop)

    with patch("vait.platform.jobs.process.signal.signal") as mocked:
        run_worker_process(service=worker, stop_event=stop)

    assert worker.calls == 1
    assert [call.args[0] for call in mocked.call_args_list] == [
        signal.SIGINT,
        signal.SIGTERM,
        signal.SIGTERM,
        signal.SIGINT,
    ]


def test_shutdown_handler_sets_stop_event() -> None:
    stop = Event()
    worker = OneCycleWorker(stop)

    with patch("vait.platform.jobs.process.signal.signal") as mocked:
        run_worker_process(service=worker, stop_event=stop)

    handler = mocked.call_args_list[0].args[1]

    stop.clear()
    handler(signal.SIGINT, None)

    assert stop.is_set()


def test_handlers_restored_after_failure() -> None:
    with patch("vait.platform.jobs.process.signal.signal") as mocked, pytest.raises(
        RuntimeError,
        match="Simulated worker failure",
    ):
        run_worker_process(service=FailingWorker())

    assert mocked.call_count == 4


def test_worker_process_requires_main_thread() -> None:
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            run_worker_process,
            service=FailingWorker(),
        )

        with pytest.raises(RuntimeError, match="main thread"):
            future.result(timeout=5)


def test_preexisting_shutdown_prevents_polling() -> None:
    stop = Event()
    stop.set()
    worker = OneCycleWorker(stop)

    with patch("vait.platform.jobs.process.signal.signal"):
        run_worker_process(service=worker, stop_event=stop)

    assert worker.calls == 0
