"""Tests for the M5-F durable job state machine."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from vait.platform.jobs import (
    InvalidJobTransition,
    JobRecord,
    JobStatus,
    can_transition,
    is_terminal,
    require_transition,
)


def _pending_job() -> JobRecord:
    return JobRecord(
        job_id=uuid4(),
        experiment_id=uuid4(),
        operation="verification",
        idempotency_key="experiment:verification:001",
        created_at=datetime.now(UTC),
    )


def test_new_job_defaults_to_pending() -> None:
    """New durable jobs begin unclaimed and pending."""
    job = _pending_job()

    assert job.status == "PENDING"
    assert job.attempt == 0
    assert job.max_attempts == 3
    assert job.claimed_by is None
    assert job.finished_at is None


@pytest.mark.parametrize(
    ("current", "target"),
    [
        ("PENDING", "RUNNING"),
        ("PENDING", "CANCELLED"),
        ("RUNNING", "SUCCEEDED"),
        ("RUNNING", "FAILED"),
        ("RUNNING", "RETRY_PENDING"),
        ("RUNNING", "CANCELLED"),
        ("RETRY_PENDING", "PENDING"),
        ("RETRY_PENDING", "CANCELLED"),
    ],
)
def test_expected_job_transitions_are_allowed(
    current: JobStatus,
    target: JobStatus,
) -> None:
    """Only explicitly modelled lifecycle transitions are allowed."""
    assert can_transition(current, target) is True


@pytest.mark.parametrize(
    ("current", "target"),
    [
        ("PENDING", "SUCCEEDED"),
        ("PENDING", "FAILED"),
        ("SUCCEEDED", "RUNNING"),
        ("FAILED", "PENDING"),
        ("CANCELLED", "RUNNING"),
        ("RETRY_PENDING", "SUCCEEDED"),
    ],
)
def test_invalid_job_transitions_fail_closed(
    current: JobStatus,
    target: JobStatus,
) -> None:
    """Illegal state changes must fail instead of being inferred."""
    with pytest.raises(InvalidJobTransition):
        require_transition(
            current,
            target,
        )


def test_terminal_states_are_explicit() -> None:
    """Completed states cannot silently re-enter execution."""
    assert is_terminal("SUCCEEDED") is True
    assert is_terminal("FAILED") is True
    assert is_terminal("CANCELLED") is True

    assert is_terminal("PENDING") is False
    assert is_terminal("RUNNING") is False
    assert is_terminal("RETRY_PENDING") is False


def test_running_job_requires_worker_lease_metadata() -> None:
    """RUNNING cannot exist without ownership and lease timestamps."""
    with pytest.raises(ValidationError):
        JobRecord(
            job_id=uuid4(),
            experiment_id=uuid4(),
            operation="verification",
            idempotency_key="missing-lease",
            status="RUNNING",
            attempt=1,
            created_at=datetime.now(UTC),
        )


def test_valid_running_job_has_bounded_lease() -> None:
    """A claimed job records ownership, heartbeat, and lease expiry."""
    now = datetime.now(UTC)

    job = JobRecord(
        job_id=uuid4(),
        experiment_id=uuid4(),
        operation="verification",
        idempotency_key="claimed-job",
        status="RUNNING",
        attempt=1,
        max_attempts=3,
        claimed_by="worker-01",
        created_at=now,
        started_at=now,
        heartbeat_at=now,
        lease_expires_at=now + timedelta(seconds=60),
    )

    assert job.claimed_by == "worker-01"
    assert job.lease_expires_at is not None


def test_retry_pending_requires_remaining_attempts() -> None:
    """Retry state cannot exceed the configured retry budget."""
    with pytest.raises(ValidationError):
        JobRecord(
            job_id=uuid4(),
            experiment_id=uuid4(),
            operation="verification",
            idempotency_key="retry-exhausted",
            status="RETRY_PENDING",
            attempt=3,
            max_attempts=3,
            created_at=datetime.now(UTC),
        )


def test_failed_job_requires_error_code_and_finished_at() -> None:
    """Terminal failure must preserve machine-readable failure metadata."""
    with pytest.raises(ValidationError):
        JobRecord(
            job_id=uuid4(),
            experiment_id=uuid4(),
            operation="verification",
            idempotency_key="failed-job",
            status="FAILED",
            attempt=1,
            max_attempts=3,
            created_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
        )


def test_terminal_success_requires_finished_at() -> None:
    """Successful completion must preserve its completion timestamp."""
    with pytest.raises(ValidationError):
        JobRecord(
            job_id=uuid4(),
            experiment_id=uuid4(),
            operation="verification",
            idempotency_key="success-without-finish",
            status="SUCCEEDED",
            attempt=1,
            created_at=datetime.now(UTC),
        )
