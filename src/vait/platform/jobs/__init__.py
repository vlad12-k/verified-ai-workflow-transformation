"""Durable background-job contracts and state machine."""

from vait.platform.jobs.models import (
    JobRecord,
    JobStatus,
    utc_now,
)
from vait.platform.jobs.state_machine import (
    TERMINAL_JOB_STATUSES,
    InvalidJobTransition,
    can_transition,
    is_terminal,
    require_transition,
)
from vait.platform.jobs.worker import (
    JobExecutor,
    WorkerCycleResult,
    WorkerService,
)

__all__ = [
    "TERMINAL_JOB_STATUSES",
    "InvalidJobTransition",
    "JobExecutor",
    "JobRecord",
    "JobStatus",
    "WorkerCycleResult",
    "WorkerService",
    "can_transition",
    "is_terminal",
    "require_transition",
    "utc_now",
]
