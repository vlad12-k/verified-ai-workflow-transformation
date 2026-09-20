"""Tests for the M5-F durable job persistence schema."""

from typing import cast

from sqlalchemy import CheckConstraint, Table, UniqueConstraint

from vait.platform.persistence.job_models import JobRow


def _job_table() -> Table:
    """Return the concrete SQLAlchemy job table."""
    return cast(
        Table,
        JobRow.__table__,
    )


def test_job_model_registers_expected_columns() -> None:
    """Durable jobs must retain execution and lease state."""
    columns = set(
        _job_table().columns.keys()
    )

    assert columns == {
        "job_id",
        "experiment_id",
        "operation",
        "payload",
        "status",
        "attempt",
        "max_attempts",
        "idempotency_key",
        "claimed_by",
        "created_at",
        "started_at",
        "heartbeat_at",
        "lease_expires_at",
        "finished_at",
        "error_code",
    }


def test_job_foreign_key_preserves_experiment_lineage() -> None:
    """Every durable job must belong to an experiment."""
    targets = {
        foreign_key.target_fullname
        for foreign_key in _job_table().foreign_keys
    }

    assert targets == {
        "experiment_runs.experiment_id",
    }


def test_job_idempotency_key_is_unique_per_experiment() -> None:
    """Duplicate submission must be guardable at the database boundary."""
    unique_sets = {
        tuple(constraint.columns.keys())
        for constraint in _job_table().constraints
        if isinstance(
            constraint,
            UniqueConstraint,
        )
    }

    assert (
        "experiment_id",
        "idempotency_key",
    ) in unique_sets


def test_job_schema_contains_state_machine_checks() -> None:
    """Persistence must enforce core durable-state invariants."""
    names = {
        str(constraint.name)
        for constraint in _job_table().constraints
        if isinstance(
            constraint,
            CheckConstraint,
        )
    }

    required = {
        "ck_jobs_status",
        "ck_jobs_attempt_nonnegative",
        "ck_jobs_max_attempts_positive",
        "ck_jobs_attempt_budget",
        "ck_jobs_pending_state",
        "ck_jobs_running_state",
        "ck_jobs_retry_state",
        "ck_jobs_terminal_finished",
        "ck_jobs_failed_error",
    }

    assert required <= names


def test_job_schema_has_claim_and_recovery_indexes() -> None:
    """Queue claiming and lease recovery require explicit indexes."""
    names = {
        index.name
        for index in _job_table().indexes
    }

    assert names == {
        "ix_jobs_experiment_id",
        "ix_jobs_status_created_at",
        "ix_jobs_lease_expires_at",
    }


def test_job_identifiers_fit_postgresql_limit() -> None:
    """Every explicit job identifier must fit PostgreSQL's 63-byte limit."""
    table = _job_table()

    assert len(table.name.encode("utf-8")) <= 63

    for constraint in table.constraints:
        assert constraint.name is not None
        assert len(
            str(constraint.name).encode("utf-8")
        ) <= 63

    for index in table.indexes:
        assert index.name is not None
        assert len(
            index.name.encode("utf-8")
        ) <= 63
