"""Signal-managed lifecycle for a durable worker process."""

import signal
from collections.abc import Callable
from datetime import datetime
from threading import Event, current_thread, main_thread
from types import FrameType

from vait.platform.jobs.models import utc_now
from vait.platform.jobs.runtime import WorkerRunner, run_worker_loop


def run_worker_process(
    *,
    service: WorkerRunner,
    poll_interval_seconds: float = 1.0,
    clock: Callable[[], datetime] = utc_now,
    stop_event: Event | None = None,
) -> None:
    """Run until shutdown, restoring original signal handlers on exit."""

    if current_thread() is not main_thread():
        raise RuntimeError(
            "Worker signal handlers require the main thread."
        )

    shutdown = stop_event if stop_event is not None else Event()

    originals = {
        signal.SIGINT: signal.getsignal(signal.SIGINT),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }

    installed: list[signal.Signals] = []

    def request_shutdown(
        _signum: int,
        _frame: FrameType | None,
    ) -> None:
        shutdown.set()

    try:
        for signum in originals:
            signal.signal(signum, request_shutdown)
            installed.append(signum)

        run_worker_loop(
            service=service,
            stop_event=shutdown,
            poll_interval_seconds=poll_interval_seconds,
            clock=clock,
        )
    finally:
        for signum in reversed(installed):
            signal.signal(signum, originals[signum])
