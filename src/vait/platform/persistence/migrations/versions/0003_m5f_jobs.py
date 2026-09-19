"""Add durable M5-F background jobs.

Revision ID: 0003_m5f_jobs
Revises: 0002_m5e_registry
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_m5f_jobs"
down_revision: str | None = "0002_m5e_registry"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create durable background job state."""
    op.create_table(
        "jobs",
        sa.Column(
            "job_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "experiment_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "operation",
            sa.String(length=128),
            nullable=False,
        ),
        sa.Column(
            "payload",
            sa.JSON(),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column(
            "attempt",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "max_attempts",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "idempotency_key",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "claimed_by",
            sa.String(length=255),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "heartbeat_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "lease_expires_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "finished_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "error_code",
            sa.String(length=128),
            nullable=True,
        ),
        sa.CheckConstraint(
            "length(operation) > 0",
            name="ck_jobs_operation_nonempty",
        ),
        sa.CheckConstraint(
            "length(idempotency_key) > 0",
            name="ck_jobs_idempotency_nonempty",
        ),
        sa.CheckConstraint(
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
        sa.CheckConstraint(
            "attempt >= 0",
            name="ck_jobs_attempt_nonnegative",
        ),
        sa.CheckConstraint(
            "max_attempts >= 1",
            name="ck_jobs_max_attempts_positive",
        ),
        sa.CheckConstraint(
            "attempt <= max_attempts",
            name="ck_jobs_attempt_budget",
        ),
        sa.CheckConstraint(
            (
                "claimed_by IS NULL "
                "OR length(claimed_by) > 0"
            ),
            name="ck_jobs_claimed_by_nonempty",
        ),
        sa.CheckConstraint(
            (
                "error_code IS NULL "
                "OR length(error_code) > 0"
            ),
            name="ck_jobs_error_code_nonempty",
        ),
        sa.CheckConstraint(
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
        sa.CheckConstraint(
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
        sa.CheckConstraint(
            (
                "status <> 'RETRY_PENDING' OR ("
                "attempt < max_attempts "
                "AND finished_at IS NULL"
                ")"
            ),
            name="ck_jobs_retry_state",
        ),
        sa.CheckConstraint(
            (
                "status NOT IN ("
                "'SUCCEEDED', 'FAILED', 'CANCELLED'"
                ") OR finished_at IS NOT NULL"
            ),
            name="ck_jobs_terminal_finished",
        ),
        sa.CheckConstraint(
            (
                "status <> 'FAILED' "
                "OR error_code IS NOT NULL"
            ),
            name="ck_jobs_failed_error",
        ),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiment_runs.experiment_id"],
            name="fk_jobs_experiment_id_experiment_runs",
        ),
        sa.PrimaryKeyConstraint(
            "job_id",
            name="pk_jobs",
        ),
        sa.UniqueConstraint(
            "experiment_id",
            "idempotency_key",
            name="uq_jobs_experiment_idempotency",
        ),
    )

    op.create_index(
        "ix_jobs_experiment_id",
        "jobs",
        ["experiment_id"],
        unique=False,
    )
    op.create_index(
        "ix_jobs_status_created_at",
        "jobs",
        ["status", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_jobs_lease_expires_at",
        "jobs",
        ["lease_expires_at"],
        unique=False,
    )


def downgrade() -> None:
    """Remove durable background job state."""
    op.drop_index(
        "ix_jobs_lease_expires_at",
        table_name="jobs",
    )
    op.drop_index(
        "ix_jobs_status_created_at",
        table_name="jobs",
    )
    op.drop_index(
        "ix_jobs_experiment_id",
        table_name="jobs",
    )
    op.drop_table("jobs")
