"""SQLAlchemy persistence model for durable background jobs."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Uuid

from vait.platform.persistence.models import Base


class JobRow(Base):
    """Persist durable background execution state."""

    __tablename__ = "jobs"
    __table_args__ = (
        UniqueConstraint(
            "experiment_id",
            "idempotency_key",
            name="uq_jobs_experiment_idempotency",
        ),
        CheckConstraint(
            "length(operation) > 0",
            name="ck_jobs_operation_nonempty",
        ),
        CheckConstraint(
            "length(idempotency_key) > 0",
            name="ck_jobs_idempotency_nonempty",
        ),
        CheckConstraint(
            (
                "status IN ("
                "'PENDING', "
                "'RUNNING', "
                "'RETRY_PENDING', "
                "'SUCCEEDED', "
                "'FAILED', "
                "'CANCELLED'"
                ")"
            ),
            name="ck_jobs_status",
        ),
        CheckConstraint(
            "attempt >= 0",
            name="ck_jobs_attempt_nonnegative",
        ),
        CheckConstraint(
            "max_attempts >= 1",
            name="ck_jobs_max_attempts_positive",
        ),
        CheckConstraint(
            "attempt <= max_attempts",
            name="ck_jobs_attempt_budget",
        ),
        CheckConstraint(
            (
                "claimed_by IS NULL "
                "OR length(claimed_by) > 0"
            ),
            name="ck_jobs_claimed_by_nonempty",
        ),
        CheckConstraint(
            (
                "error_code IS NULL "
                "OR length(error_code) > 0"
            ),
            name="ck_jobs_error_code_nonempty",
        ),
        CheckConstraint(
            (
                "status <> 'PENDING' OR ("
                "claimed_by IS NULL "
                "AND started_at IS NULL "
                "AND heartbeat_at IS NULL "
                "AND lease_expires_at IS NULL "
                "AND finished_at IS NULL"
                ")"
            ),
            name="ck_jobs_pending_state",
        ),
        CheckConstraint(
            (
                "status <> 'RUNNING' OR ("
                "claimed_by IS NOT NULL "
                "AND started_at IS NOT NULL "
                "AND heartbeat_at IS NOT NULL "
                "AND lease_expires_at IS NOT NULL "
                "AND finished_at IS NULL"
                ")"
            ),
            name="ck_jobs_running_state",
        ),
        CheckConstraint(
            (
                "status <> 'RETRY_PENDING' OR ("
                "attempt < max_attempts "
                "AND finished_at IS NULL"
                ")"
            ),
            name="ck_jobs_retry_state",
        ),
        CheckConstraint(
            (
                "status NOT IN ("
                "'SUCCEEDED', 'FAILED', 'CANCELLED'"
                ") OR finished_at IS NOT NULL"
            ),
            name="ck_jobs_terminal_finished",
        ),
        CheckConstraint(
            (
                "status <> 'FAILED' "
                "OR error_code IS NOT NULL"
            ),
            name="ck_jobs_failed_error",
        ),
        Index(
            "ix_jobs_experiment_id",
            "experiment_id",
        ),
        Index(
            "ix_jobs_status_created_at",
            "status",
            "created_at",
        ),
        Index(
            "ix_jobs_lease_expires_at",
            "lease_expires_at",
        ),
    )

    job_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
    )
    experiment_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey(
            "experiment_runs.experiment_id",
            name="fk_jobs_experiment_id_experiment_runs",
        ),
        nullable=False,
    )
    operation: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )
    payload: Mapped[dict[str, object]] = mapped_column(
        JSON,
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    attempt: Mapped[int] = mapped_column(
        nullable=False,
    )
    max_attempts: Mapped[int] = mapped_column(
        nullable=False,
    )
    idempotency_key: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    claimed_by: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    heartbeat_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    error_code: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )
