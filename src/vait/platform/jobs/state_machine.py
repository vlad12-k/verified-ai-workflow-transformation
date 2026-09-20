"""Explicit state transitions for durable background jobs."""

from vait.platform.jobs.models import JobStatus

_ALLOWED_TRANSITIONS: dict[
    JobStatus,
    frozenset[JobStatus],
] = {
    "PENDING": frozenset(
        {
            "RUNNING",
            "CANCELLED",
        }
    ),
    "RUNNING": frozenset(
        {
            "SUCCEEDED",
            "FAILED",
            "RETRY_PENDING",
            "CANCELLED",
        }
    ),
    "RETRY_PENDING": frozenset(
        {
            "PENDING",
            "CANCELLED",
        }
    ),
    "SUCCEEDED": frozenset(),
    "FAILED": frozenset(),
    "CANCELLED": frozenset(),
}

TERMINAL_JOB_STATUSES: frozenset[JobStatus] = frozenset(
    {
        "SUCCEEDED",
        "FAILED",
        "CANCELLED",
    }
)


class InvalidJobTransition(ValueError):
    """Raised when a durable job transition violates the state machine."""


def can_transition(
    current: JobStatus,
    target: JobStatus,
) -> bool:
    """Return whether a state transition is explicitly permitted."""
    return target in _ALLOWED_TRANSITIONS[current]


def require_transition(
    current: JobStatus,
    target: JobStatus,
) -> None:
    """Fail closed when a requested job transition is invalid."""
    if not can_transition(
        current,
        target,
    ):
        raise InvalidJobTransition(
            f"Invalid job transition: {current} -> {target}"
        )


def is_terminal(status: JobStatus) -> bool:
    """Return whether a job state is terminal."""
    return status in TERMINAL_JOB_STATUSES
