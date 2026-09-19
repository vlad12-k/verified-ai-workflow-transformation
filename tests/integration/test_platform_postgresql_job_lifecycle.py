"""PostgreSQL lifecycle tests for M5-F durable jobs."""

import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text

from vait.platform.jobs import JobRecord
from vait.platform.persistence.engine import (
    DatabaseRuntime,
    create_database_runtime,
)
from vait.platform.persistence.repositories.jobs import (
    JobRepository,
)
from vait.platform.persistence.repositories.registry import (
    ExperimentRunRepository,
)
from vait.platform.persistence.session import (
    transactional_session,
)
from vait.platform.registry import ExperimentRunRecord
from vait.platform.settings import PlatformSettings

_ROOT = Path(__file__).resolve().parents[2]
_ALEMBIC_INI = _ROOT / "alembic.ini"


@pytest.fixture
def jobs_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[DatabaseRuntime]:
    """Provide a clean migrated PostgreSQL runtime."""
    database_url = os.getenv(
        "VAIT_TEST_JOBS_DATABASE_URL"
    )

    if database_url is None:
        pytest.skip(
            "VAIT_TEST_JOBS_DATABASE_URL is not configured"
        )

    monkeypatch.setenv(
        "VAIT_DATABASE_URL",
        database_url,
    )

    command.upgrade(
        Config(str(_ALEMBIC_INI)),
        "head",
    )

    runtime = create_database_runtime(
        PlatformSettings()
    )

    with runtime.engine.begin() as connection:
        connection.execute(
            text(
                "TRUNCATE TABLE "
                "jobs, artifacts, evidence_records, "
                "candidates, experiment_runs CASCADE"
            )
        )

    try:
        yield runtime
    finally:
        with runtime.engine.begin() as connection:
            connection.execute(
                text(
                    "TRUNCATE TABLE "
                    "jobs, artifacts, evidence_records, "
                    "candidates, experiment_runs CASCADE"
                )
            )

        runtime.engine.dispose()


def _create_experiment(
    runtime: DatabaseRuntime,
) -> ExperimentRunRecord:
    now = datetime.now(UTC)

    experiment = ExperimentRunRecord(
        experiment_id=uuid4(),
        source_revision="a" * 40,
        dataset_fingerprint="b" * 64,
        dataset_partition="validation-v1",
        transformation_contract_id="worker-lifecycle-test",
        transformation_contract_version="1.0",
        benchmark_configuration={
            "deterministic": True,
        },
        created_at=now,
    )

    with transactional_session(runtime) as session:
        ExperimentRunRepository(
            session
        ).add(experiment)

    return experiment


def _create_pending_job(
    runtime: DatabaseRuntime,
    *,
    experiment_id: UUID,
    max_attempts: int = 3,
) -> JobRecord:
    job = JobRecord(
        job_id=uuid4(),
        experiment_id=experiment_id,
        operation="verification",
        payload={},
        idempotency_key=str(uuid4()),
        max_attempts=max_attempts,
        created_at=datetime.now(UTC),
    )

    with transactional_session(runtime) as session:
        JobRepository(session).add(job)

    return job


def test_heartbeat_renews_only_owner_lease(
    jobs_runtime: DatabaseRuntime,
) -> None:
    """Only the owning worker may renew a running job lease."""
    experiment = _create_experiment(
        jobs_runtime
    )
    job = _create_pending_job(
        jobs_runtime,
        experiment_id=experiment.experiment_id,
    )

    claimed_at = datetime.now(UTC)

    with transactional_session(jobs_runtime) as session:
        claimed = JobRepository(
            session
        ).claim_next(
            worker_id="worker-01",
            now=claimed_at,
            lease_seconds=60,
        )

    assert claimed is not None

    heartbeat_at = claimed_at + timedelta(
        seconds=30,
    )

    with transactional_session(jobs_runtime) as session:
        renewed = JobRepository(
            session
        ).heartbeat(
            job_id=job.job_id,
            worker_id="worker-01",
            now=heartbeat_at,
            lease_seconds=60,
        )

    assert renewed is not None
    assert renewed.heartbeat_at == heartbeat_at
    assert renewed.lease_expires_at == (
        heartbeat_at + timedelta(seconds=60)
    )

    with transactional_session(jobs_runtime) as session:
        rejected = JobRepository(
            session
        ).heartbeat(
            job_id=job.job_id,
            worker_id="worker-02",
            now=heartbeat_at,
            lease_seconds=60,
        )

    assert rejected is None


def test_running_job_can_complete_successfully(
    jobs_runtime: DatabaseRuntime,
) -> None:
    """A worker may terminally complete only its running job."""
    experiment = _create_experiment(
        jobs_runtime
    )
    job = _create_pending_job(
        jobs_runtime,
        experiment_id=experiment.experiment_id,
    )

    claimed_at = datetime.now(UTC)

    with transactional_session(jobs_runtime) as session:
        claimed = JobRepository(
            session
        ).claim_next(
            worker_id="worker-01",
            now=claimed_at,
            lease_seconds=60,
        )

    assert claimed is not None

    finished_at = claimed_at + timedelta(
        seconds=10,
    )

    with transactional_session(jobs_runtime) as session:
        completed = JobRepository(
            session
        ).mark_succeeded(
            job_id=job.job_id,
            worker_id="worker-01",
            now=finished_at,
        )

    assert completed is not None
    assert completed.status == "SUCCEEDED"
    assert completed.finished_at == finished_at


def test_failure_can_retry_then_be_claimed_again(
    jobs_runtime: DatabaseRuntime,
) -> None:
    """Retryable failure must return to the queue with incremented attempt."""
    experiment = _create_experiment(
        jobs_runtime
    )
    job = _create_pending_job(
        jobs_runtime,
        experiment_id=experiment.experiment_id,
        max_attempts=2,
    )

    first_claim_at = datetime.now(UTC)

    with transactional_session(jobs_runtime) as session:
        first_claim = JobRepository(
            session
        ).claim_next(
            worker_id="worker-01",
            now=first_claim_at,
            lease_seconds=60,
        )

    assert first_claim is not None
    assert first_claim.attempt == 1

    with transactional_session(jobs_runtime) as session:
        retry = JobRepository(
            session
        ).mark_failed_or_retry(
            job_id=job.job_id,
            worker_id="worker-01",
            now=first_claim_at + timedelta(
                seconds=10,
            ),
            error_code="TRANSIENT_FAILURE",
        )

    assert retry is not None
    assert retry.status == "RETRY_PENDING"

    with transactional_session(jobs_runtime) as session:
        pending = JobRepository(
            session
        ).requeue_retry(
            job_id=job.job_id,
        )

    assert pending is not None
    assert pending.status == "PENDING"

    with transactional_session(jobs_runtime) as session:
        second_claim = JobRepository(
            session
        ).claim_next(
            worker_id="worker-02",
            now=first_claim_at + timedelta(
                seconds=20,
            ),
            lease_seconds=60,
        )

    assert second_claim is not None
    assert second_claim.job_id == job.job_id
    assert second_claim.attempt == 2
    assert second_claim.claimed_by == "worker-02"


def test_exhausted_failure_becomes_terminal(
    jobs_runtime: DatabaseRuntime,
) -> None:
    """The final permitted attempt must end in FAILED."""
    experiment = _create_experiment(
        jobs_runtime
    )
    job = _create_pending_job(
        jobs_runtime,
        experiment_id=experiment.experiment_id,
        max_attempts=1,
    )

    claimed_at = datetime.now(UTC)

    with transactional_session(jobs_runtime) as session:
        claimed = JobRepository(
            session
        ).claim_next(
            worker_id="worker-01",
            now=claimed_at,
            lease_seconds=60,
        )

    assert claimed is not None

    failed_at = claimed_at + timedelta(
        seconds=10,
    )

    with transactional_session(jobs_runtime) as session:
        failed = JobRepository(
            session
        ).mark_failed_or_retry(
            job_id=job.job_id,
            worker_id="worker-01",
            now=failed_at,
            error_code="PERMANENT_FAILURE",
        )

    assert failed is not None
    assert failed.status == "FAILED"
    assert failed.error_code == "PERMANENT_FAILURE"
    assert failed.finished_at == failed_at


def test_expired_lease_is_recovered_for_retry(
    jobs_runtime: DatabaseRuntime,
) -> None:
    """A crashed worker lease must become recoverable work."""
    experiment = _create_experiment(
        jobs_runtime
    )
    job = _create_pending_job(
        jobs_runtime,
        experiment_id=experiment.experiment_id,
        max_attempts=2,
    )

    claimed_at = datetime.now(UTC)

    with transactional_session(jobs_runtime) as session:
        claimed = JobRepository(
            session
        ).claim_next(
            worker_id="worker-crashed",
            now=claimed_at,
            lease_seconds=1,
        )

    assert claimed is not None

    recovery_at = claimed_at + timedelta(
        seconds=2,
    )

    with transactional_session(jobs_runtime) as session:
        recovered = JobRepository(
            session
        ).recover_expired_leases(
            now=recovery_at,
        )

    assert len(recovered) == 1
    assert recovered[0].job_id == job.job_id
    assert recovered[0].status == "RETRY_PENDING"
    assert recovered[0].error_code == "LEASE_EXPIRED"
    assert recovered[0].claimed_by is None
    assert recovered[0].lease_expires_at is None
