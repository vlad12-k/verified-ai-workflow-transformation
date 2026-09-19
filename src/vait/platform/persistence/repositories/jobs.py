"""PostgreSQL repository for durable background jobs."""

from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from vait.platform.jobs.models import JobRecord
from vait.platform.persistence.job_models import JobRow
from vait.platform.persistence.repositories.base import Repository


def _job_record(row: JobRow) -> JobRecord:
    """Map a persistence row to the typed durable-job contract."""
    return JobRecord.model_validate(
        {
            "job_id": row.job_id,
            "experiment_id": row.experiment_id,
            "operation": row.operation,
            "payload": row.payload,
            "status": row.status,
            "attempt": row.attempt,
            "max_attempts": row.max_attempts,
            "idempotency_key": row.idempotency_key,
            "claimed_by": row.claimed_by,
            "created_at": row.created_at,
            "started_at": row.started_at,
            "heartbeat_at": row.heartbeat_at,
            "lease_expires_at": row.lease_expires_at,
            "finished_at": row.finished_at,
            "error_code": row.error_code,
        }
    )


class JobRepository(Repository[JobRecord, UUID]):
    """Persist and atomically claim durable platform jobs."""

    def __init__(self, session: Session) -> None:
        self._session = session

    @property
    def session(self) -> Session:
        """Return the SQLAlchemy session owned by this repository."""
        return self._session

    def add(self, entity: JobRecord) -> None:
        """Persist a new durable job."""
        row = JobRow(
            job_id=entity.job_id,
            experiment_id=entity.experiment_id,
            operation=entity.operation,
            payload=dict(entity.payload),
            status=entity.status,
            attempt=entity.attempt,
            max_attempts=entity.max_attempts,
            idempotency_key=entity.idempotency_key,
            claimed_by=entity.claimed_by,
            created_at=entity.created_at,
            started_at=entity.started_at,
            heartbeat_at=entity.heartbeat_at,
            lease_expires_at=entity.lease_expires_at,
            finished_at=entity.finished_at,
            error_code=entity.error_code,
        )

        self.session.add(row)
        self.session.flush()

    def get(self, entity_id: UUID) -> JobRecord | None:
        """Return one durable job by identifier."""
        row = self.session.get(
            JobRow,
            entity_id,
        )

        if row is None:
            return None

        return _job_record(row)

    def claim_next(
        self,
        *,
        worker_id: str,
        now: datetime,
        lease_seconds: int,
    ) -> JobRecord | None:
        """Atomically claim the oldest eligible pending job.

        PostgreSQL SKIP LOCKED allows competing workers to skip
        jobs already locked by another worker instead of blocking.
        """
        if not worker_id:
            raise ValueError(
                "worker_id must not be empty"
            )

        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError(
                "now must be timezone-aware"
            )

        if lease_seconds < 1:
            raise ValueError(
                "lease_seconds must be positive"
            )

        statement = (
            select(JobRow)
            .where(
                JobRow.status == "PENDING",
                JobRow.attempt < JobRow.max_attempts,
            )
            .order_by(
                JobRow.created_at,
                JobRow.job_id,
            )
            .limit(1)
            .with_for_update(
                skip_locked=True,
            )
        )

        row = self.session.scalars(
            statement
        ).first()

        if row is None:
            return None

        row.status = "RUNNING"
        row.attempt += 1
        row.claimed_by = worker_id
        row.started_at = now
        row.heartbeat_at = now
        row.lease_expires_at = (
            now
            + timedelta(
                seconds=lease_seconds,
            )
        )
        row.finished_at = None
        row.error_code = None

        self.session.flush()

        return _job_record(row)

    def heartbeat(
        self,
        *,
        job_id: UUID,
        worker_id: str,
        now: datetime,
        lease_seconds: int,
    ) -> JobRecord | None:
        """Renew the lease for a running job owned by this worker."""
        if not worker_id:
            raise ValueError(
                "worker_id must not be empty"
            )

        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError(
                "now must be timezone-aware"
            )

        if lease_seconds < 1:
            raise ValueError(
                "lease_seconds must be positive"
            )

        statement = (
            select(JobRow)
            .where(
                JobRow.job_id == job_id,
                JobRow.status == "RUNNING",
                JobRow.claimed_by == worker_id,
            )
            .with_for_update()
        )

        row = self.session.scalars(
            statement
        ).first()

        if row is None:
            return None

        row.heartbeat_at = now
        row.lease_expires_at = (
            now
            + timedelta(
                seconds=lease_seconds,
            )
        )

        self.session.flush()

        return _job_record(row)

    def mark_succeeded(
        self,
        *,
        job_id: UUID,
        worker_id: str,
        now: datetime,
    ) -> JobRecord | None:
        """Complete a running job successfully."""
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError(
                "now must be timezone-aware"
            )

        statement = (
            select(JobRow)
            .where(
                JobRow.job_id == job_id,
                JobRow.status == "RUNNING",
                JobRow.claimed_by == worker_id,
            )
            .with_for_update()
        )

        row = self.session.scalars(
            statement
        ).first()

        if row is None:
            return None

        row.status = "SUCCEEDED"
        row.finished_at = now

        self.session.flush()

        return _job_record(row)

    def mark_failed_or_retry(
        self,
        *,
        job_id: UUID,
        worker_id: str,
        now: datetime,
        error_code: str,
    ) -> JobRecord | None:
        """Record worker failure and either retry or finish permanently."""
        if not error_code:
            raise ValueError(
                "error_code must not be empty"
            )

        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError(
                "now must be timezone-aware"
            )

        statement = (
            select(JobRow)
            .where(
                JobRow.job_id == job_id,
                JobRow.status == "RUNNING",
                JobRow.claimed_by == worker_id,
            )
            .with_for_update()
        )

        row = self.session.scalars(
            statement
        ).first()

        if row is None:
            return None

        row.error_code = error_code

        if row.attempt < row.max_attempts:
            row.status = "RETRY_PENDING"
            row.claimed_by = None
            row.started_at = None
            row.heartbeat_at = None
            row.lease_expires_at = None
            row.finished_at = None
        else:
            row.status = "FAILED"
            row.finished_at = now

        self.session.flush()

        return _job_record(row)

    def requeue_retry(
        self,
        *,
        job_id: UUID,
    ) -> JobRecord | None:
        """Move a retryable job back into the pending queue."""
        statement = (
            select(JobRow)
            .where(
                JobRow.job_id == job_id,
                JobRow.status == "RETRY_PENDING",
            )
            .with_for_update()
        )

        row = self.session.scalars(
            statement
        ).first()

        if row is None:
            return None

        row.status = "PENDING"

        self.session.flush()

        return _job_record(row)

    def recover_expired_leases(
        self,
        *,
        now: datetime,
        limit: int = 100,
    ) -> list[JobRecord]:
        """Recover running jobs whose worker lease has expired."""
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError(
                "now must be timezone-aware"
            )

        if limit < 1:
            raise ValueError(
                "limit must be positive"
            )

        statement = (
            select(JobRow)
            .where(
                JobRow.status == "RUNNING",
                JobRow.lease_expires_at < now,
            )
            .order_by(
                JobRow.lease_expires_at,
                JobRow.job_id,
            )
            .limit(limit)
            .with_for_update(
                skip_locked=True,
            )
        )

        rows = list(
            self.session.scalars(statement)
        )

        recovered: list[JobRecord] = []

        for row in rows:
            row.error_code = "LEASE_EXPIRED"

            if row.attempt < row.max_attempts:
                row.status = "RETRY_PENDING"
                row.claimed_by = None
                row.started_at = None
                row.heartbeat_at = None
                row.lease_expires_at = None
                row.finished_at = None
            else:
                row.status = "FAILED"
                row.finished_at = now

            recovered.append(
                _job_record(row)
            )

        self.session.flush()

        return recovered
