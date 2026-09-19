"""PostgreSQL integration tests for durable M5-F jobs."""

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


def _alembic_config() -> Config:
    return Config(str(_ALEMBIC_INI))


@pytest.fixture
def jobs_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[DatabaseRuntime]:
    """Provide a migrated real PostgreSQL runtime."""
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
        _alembic_config(),
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
                "candidates, experiment_runs "
                "CASCADE"
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
                    "candidates, experiment_runs "
                    "CASCADE"
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
        transformation_contract_id="worker-test",
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


def _pending_job(
    *,
    experiment_id: UUID,
    created_at: datetime,
    idempotency_key: str,
) -> JobRecord:
    return JobRecord(
        job_id=uuid4(),
        experiment_id=experiment_id,
        operation="verification",
        payload={
            "source": "integration-test",
        },
        idempotency_key=idempotency_key,
        created_at=created_at,
    )


def test_claim_next_transitions_pending_job_to_running(
    jobs_runtime: DatabaseRuntime,
) -> None:
    """Claiming must atomically establish worker ownership and lease."""
    experiment = _create_experiment(
        jobs_runtime
    )

    created_at = datetime.now(UTC)

    job = _pending_job(
        experiment_id=experiment.experiment_id,
        created_at=created_at,
        idempotency_key="claim-one",
    )

    with transactional_session(jobs_runtime) as session:
        JobRepository(session).add(job)

    claimed_at = created_at + timedelta(
        seconds=5,
    )

    with transactional_session(jobs_runtime) as session:
        claimed = JobRepository(
            session
        ).claim_next(
            worker_id="worker-01",
            now=claimed_at,
            lease_seconds=60,
        )

    assert claimed is not None
    assert claimed.job_id == job.job_id
    assert claimed.status == "RUNNING"
    assert claimed.attempt == 1
    assert claimed.claimed_by == "worker-01"
    assert claimed.started_at == claimed_at
    assert claimed.heartbeat_at == claimed_at
    assert claimed.lease_expires_at == (
        claimed_at + timedelta(seconds=60)
    )


def test_skip_locked_allows_competing_workers_to_claim_different_jobs(
    jobs_runtime: DatabaseRuntime,
) -> None:
    """A locked queue row must not block another PostgreSQL worker."""
    experiment = _create_experiment(
        jobs_runtime
    )

    created_at = datetime.now(UTC)

    first_job = _pending_job(
        experiment_id=experiment.experiment_id,
        created_at=created_at,
        idempotency_key="worker-race-1",
    )
    second_job = _pending_job(
        experiment_id=experiment.experiment_id,
        created_at=created_at + timedelta(
            microseconds=1,
        ),
        idempotency_key="worker-race-2",
    )

    with transactional_session(jobs_runtime) as session:
        repository = JobRepository(session)
        repository.add(first_job)
        repository.add(second_job)

    session_one = jobs_runtime.session_factory()
    session_two = jobs_runtime.session_factory()

    try:
        first_claim = JobRepository(
            session_one
        ).claim_next(
            worker_id="worker-01",
            now=created_at + timedelta(
                seconds=5,
            ),
            lease_seconds=60,
        )

        assert first_claim is not None

        second_claim = JobRepository(
            session_two
        ).claim_next(
            worker_id="worker-02",
            now=created_at + timedelta(
                seconds=6,
            ),
            lease_seconds=60,
        )

        assert second_claim is not None

        assert {
            first_claim.job_id,
            second_claim.job_id,
        } == {
            first_job.job_id,
            second_job.job_id,
        }

        assert (
            first_claim.job_id
            != second_claim.job_id
        )

        session_one.commit()
        session_two.commit()
    finally:
        session_one.close()
        session_two.close()


def test_claim_next_returns_none_when_queue_is_empty(
    jobs_runtime: DatabaseRuntime,
) -> None:
    """An empty queue must not fabricate background work."""
    _create_experiment(
        jobs_runtime
    )

    with transactional_session(jobs_runtime) as session:
        claimed = JobRepository(
            session
        ).claim_next(
            worker_id="worker-01",
            now=datetime.now(UTC),
            lease_seconds=60,
        )

    assert claimed is None


def test_claim_next_does_not_exceed_attempt_budget(
    jobs_runtime: DatabaseRuntime,
) -> None:
    """Jobs with no remaining attempts must not be claimed."""
    experiment = _create_experiment(
        jobs_runtime
    )

    created_at = datetime.now(UTC)

    exhausted = JobRecord(
        job_id=uuid4(),
        experiment_id=experiment.experiment_id,
        operation="verification",
        idempotency_key="exhausted-job",
        status="PENDING",
        attempt=1,
        max_attempts=1,
        created_at=created_at,
    )

    with transactional_session(jobs_runtime) as session:
        JobRepository(session).add(
            exhausted
        )

    with transactional_session(jobs_runtime) as session:
        claimed = JobRepository(
            session
        ).claim_next(
            worker_id="worker-01",
            now=created_at + timedelta(
                seconds=5,
            ),
            lease_seconds=60,
        )

    assert claimed is None
