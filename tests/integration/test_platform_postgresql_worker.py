"""PostgreSQL integration tests for the M5-F worker service."""

import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text

from vait.platform.jobs import JobRecord, WorkerService
from vait.platform.persistence.engine import (
    DatabaseRuntime,
    create_database_runtime,
)
from vait.platform.persistence.repositories.jobs import JobRepository
from vait.platform.persistence.repositories.registry import (
    ExperimentRunRepository,
)
from vait.platform.persistence.session import transactional_session
from vait.platform.registry import ExperimentRunRecord
from vait.platform.settings import PlatformSettings

_ROOT = Path(__file__).resolve().parents[2]
_ALEMBIC_INI = _ROOT / "alembic.ini"


@pytest.fixture
def worker_runtime(
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
    """Persist one experiment owning worker-test jobs."""
    experiment = ExperimentRunRecord(
        experiment_id=uuid4(),
        source_revision="a" * 40,
        dataset_fingerprint="b" * 64,
        dataset_partition="validation-v1",
        transformation_contract_id="worker-service-test",
        transformation_contract_version="1.0",
        benchmark_configuration={
            "deterministic": True,
        },
        created_at=datetime.now(UTC),
    )

    with transactional_session(runtime) as session:
        ExperimentRunRepository(session).add(
            experiment
        )

    return experiment


def _create_job(
    runtime: DatabaseRuntime,
    *,
    experiment_id: UUID,
    max_attempts: int = 3,
) -> JobRecord:
    """Persist one pending worker-test job."""
    job = JobRecord(
        job_id=uuid4(),
        experiment_id=experiment_id,
        operation="verification",
        payload={
            "source": "worker-integration-test",
        },
        idempotency_key=str(uuid4()),
        max_attempts=max_attempts,
        created_at=datetime.now(UTC),
    )

    with transactional_session(runtime) as session:
        JobRepository(session).add(job)

    return job


class SuccessfulExecutor:
    """Observe durable state while executing successfully."""

    def __init__(
        self,
        runtime: DatabaseRuntime,
    ) -> None:
        self._runtime = runtime
        self.seen_job_ids: list[UUID] = []
        self.observed_statuses: list[str] = []

    def execute(
        self,
        job: JobRecord,
    ) -> None:
        """Verify the claim is committed before execution begins."""
        self.seen_job_ids.append(
            job.job_id
        )

        with transactional_session(
            self._runtime
        ) as session:
            persisted = JobRepository(
                session
            ).get(job.job_id)

        assert persisted is not None

        self.observed_statuses.append(
            persisted.status
        )


class FailingExecutor:
    """Raise a deterministic execution failure."""

    def execute(
        self,
        job: JobRecord,
    ) -> None:
        """Fail after a job has been durably claimed."""
        del job
        raise RuntimeError(
            "worker execution failed"
        )


def test_worker_returns_no_work_for_empty_queue(
    worker_runtime: DatabaseRuntime,
) -> None:
    """An empty queue must produce an idle worker cycle."""
    executor = SuccessfulExecutor(
        worker_runtime
    )

    worker = WorkerService(
        runtime=worker_runtime,
        executor=executor,
        worker_id="worker-01",
        lease_seconds=60,
    )

    result = worker.run_once(
        now=datetime.now(UTC),
    )

    assert result.claimed is False
    assert result.job is None
    assert executor.seen_job_ids == []


def test_worker_commits_claim_before_execution_and_then_succeeds(
    worker_runtime: DatabaseRuntime,
) -> None:
    """Executor must observe RUNNING before terminal success."""
    experiment = _create_experiment(
        worker_runtime
    )
    job = _create_job(
        worker_runtime,
        experiment_id=experiment.experiment_id,
    )

    executor = SuccessfulExecutor(
        worker_runtime
    )

    worker = WorkerService(
        runtime=worker_runtime,
        executor=executor,
        worker_id="worker-01",
        lease_seconds=60,
    )

    result = worker.run_once(
        now=datetime.now(UTC),
    )

    assert result.claimed is True
    assert result.job is not None
    assert result.job.job_id == job.job_id
    assert result.job.status == "SUCCEEDED"

    assert executor.seen_job_ids == [
        job.job_id,
    ]
    assert executor.observed_statuses == [
        "RUNNING",
    ]

    with transactional_session(
        worker_runtime
    ) as session:
        persisted = JobRepository(
            session
        ).get(job.job_id)

    assert persisted is not None
    assert persisted.status == "SUCCEEDED"


def test_worker_failure_becomes_retry_pending_when_budget_remains(
    worker_runtime: DatabaseRuntime,
) -> None:
    """Transient executor failure must preserve retryability."""
    experiment = _create_experiment(
        worker_runtime
    )
    job = _create_job(
        worker_runtime,
        experiment_id=experiment.experiment_id,
        max_attempts=2,
    )

    worker = WorkerService(
        runtime=worker_runtime,
        executor=FailingExecutor(),
        worker_id="worker-01",
        lease_seconds=60,
    )

    result = worker.run_once(
        now=datetime.now(UTC),
    )

    assert result.claimed is True
    assert result.job is not None
    assert result.job.job_id == job.job_id
    assert result.job.status == "RETRY_PENDING"
    assert result.job.attempt == 1
    assert result.job.error_code == "RuntimeError"
    assert result.job.claimed_by is None


def test_worker_failure_becomes_terminal_when_budget_is_exhausted(
    worker_runtime: DatabaseRuntime,
) -> None:
    """Final executor failure must persist terminal FAILED state."""
    experiment = _create_experiment(
        worker_runtime
    )
    job = _create_job(
        worker_runtime,
        experiment_id=experiment.experiment_id,
        max_attempts=1,
    )

    worker = WorkerService(
        runtime=worker_runtime,
        executor=FailingExecutor(),
        worker_id="worker-01",
        lease_seconds=60,
    )

    result = worker.run_once(
        now=datetime.now(UTC),
    )

    assert result.claimed is True
    assert result.job is not None
    assert result.job.job_id == job.job_id
    assert result.job.status == "FAILED"
    assert result.job.attempt == 1
    assert result.job.error_code == "RuntimeError"
    assert result.job.finished_at is not None
