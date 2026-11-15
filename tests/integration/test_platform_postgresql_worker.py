"""PostgreSQL integration tests for the M5-F worker service."""

import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import cast
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from alembic import command
from alembic.config import Config
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from sqlalchemy import text

from vait.inference.providers.models import (
    ProviderInferenceResponse,
    ProviderTiming,
)
from vait.platform.jobs import JobRecord, WorkerService
from vait.platform.jobs.executors import BudgetedProviderJobExecutor
from vait.platform.jobs.provider_evidence import (
    PostgreSQLProviderEvidenceWriter,
)
from vait.platform.observability import (
    configure_platform_logging,
    create_tracing_runtime,
    current_log_context,
)
from vait.platform.persistence.engine import (
    DatabaseRuntime,
    create_database_runtime,
)
from vait.platform.persistence.repositories.jobs import JobRepository
from vait.platform.persistence.repositories.registry import (
    EvidenceRepository,
    ExperimentRunRepository,
)
from vait.platform.persistence.session import transactional_session
from vait.platform.provider_execution import BudgetedJobProviderInvocation
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
        self.observed_contexts: list[dict[str, str]] = []

    def execute(
        self,
        job: JobRecord,
    ) -> None:
        """Verify the claim is committed before execution begins."""
        self.seen_job_ids.append(
            job.job_id
        )
        self.observed_contexts.append(
            current_log_context()
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
    assert executor.observed_contexts == [
        {
            "experiment_id": str(
                experiment.experiment_id
            ),
            "job_id": str(
                job.job_id
            ),
            "worker_id": "worker-01",
        }
    ]
    assert current_log_context() == {}

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
    assert current_log_context() == {}


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
    assert current_log_context() == {}

def test_worker_emits_correlated_success_events(
    worker_runtime: DatabaseRuntime,
) -> None:
    """Successful work must emit correlated structured lifecycle events."""
    experiment = _create_experiment(
        worker_runtime
    )
    job = _create_job(
        worker_runtime,
        experiment_id=experiment.experiment_id,
    )

    stream = StringIO()
    configure_platform_logging(
        stream=stream,
    )

    worker = WorkerService(
        runtime=worker_runtime,
        executor=SuccessfulExecutor(
            worker_runtime
        ),
        worker_id="worker-log-success",
        lease_seconds=60,
    )

    result = worker.run_once(
        now=datetime.now(UTC),
    )

    assert result.job is not None
    assert result.job.status == "SUCCEEDED"
    assert current_log_context() == {}

    serialized = stream.getvalue()

    assert '"event":"job_claimed"' in serialized
    assert '"event":"job_succeeded"' in serialized

    assert (
        f'"experiment_id":"{experiment.experiment_id}"'
        in serialized
    )
    assert (
        f'"job_id":"{job.job_id}"'
        in serialized
    )
    assert (
        '"worker_id":"worker-log-success"'
        in serialized
    )


def test_worker_failure_logs_type_without_exception_message(
    worker_runtime: DatabaseRuntime,
) -> None:
    """Worker failure logs must remain correlated and sanitised."""
    experiment = _create_experiment(
        worker_runtime
    )
    job = _create_job(
        worker_runtime,
        experiment_id=experiment.experiment_id,
        max_attempts=1,
    )

    stream = StringIO()
    configure_platform_logging(
        stream=stream,
    )

    worker = WorkerService(
        runtime=worker_runtime,
        executor=FailingExecutor(),
        worker_id="worker-log-failure",
        lease_seconds=60,
    )

    result = worker.run_once(
        now=datetime.now(UTC),
    )

    assert result.job is not None
    assert result.job.status == "FAILED"
    assert current_log_context() == {}

    serialized = stream.getvalue()

    assert '"event":"job_claimed"' in serialized
    assert (
        '"event":"job_execution_failed"'
        in serialized
    )
    assert '"event":"job_failed"' in serialized
    assert (
        '"exception_type":"RuntimeError"'
        in serialized
    )

    assert (
        f'"experiment_id":"{experiment.experiment_id}"'
        in serialized
    )
    assert (
        f'"job_id":"{job.job_id}"'
        in serialized
    )
    assert (
        '"worker_id":"worker-log-failure"'
        in serialized
    )

    assert "worker execution failed" not in serialized
    assert "Traceback" not in serialized

def test_worker_success_events_share_one_trace_span(
    worker_runtime: DatabaseRuntime,
) -> None:
    """Successful worker lifecycle events must share one OTel span."""
    experiment = _create_experiment(
        worker_runtime
    )
    job = _create_job(
        worker_runtime,
        experiment_id=experiment.experiment_id,
    )

    stream = StringIO()
    configure_platform_logging(
        stream=stream,
    )

    exporter = InMemorySpanExporter()
    tracing_runtime = create_tracing_runtime(
        PlatformSettings(
            environment="test",
            service_name="vait-worker-test",
        ),
        span_exporter=exporter,
    )

    executor = SuccessfulExecutor(
        worker_runtime
    )

    try:
        worker = WorkerService(
            runtime=worker_runtime,
            executor=executor,
            worker_id="worker-traced-success",
            lease_seconds=60,
            tracing_runtime=tracing_runtime,
        )

        result = worker.run_once(
            now=datetime.now(UTC),
        )

        tracing_runtime.provider.force_flush()

        assert result.job is not None
        assert result.job.status == "SUCCEEDED"
        assert current_log_context() == {}

        spans = exporter.get_finished_spans()

        assert len(spans) == 1
        assert spans[0].name == "worker.job"

        payloads = [
            json.loads(line)
            for line in stream.getvalue().splitlines()
            if line
        ]

        lifecycle = [
            payload
            for payload in payloads
            if payload["event"]
            in {
                "job_claimed",
                "job_succeeded",
            }
        ]

        assert len(lifecycle) == 2

        trace_ids = {
            payload["trace_id"]
            for payload in lifecycle
        }
        span_ids = {
            payload["span_id"]
            for payload in lifecycle
        }

        assert len(trace_ids) == 1
        assert len(span_ids) == 1

        trace_id = next(iter(trace_ids))
        span_id = next(iter(span_ids))

        assert isinstance(trace_id, str)
        assert isinstance(span_id, str)
        assert len(trace_id) == 32
        assert len(span_id) == 16
        assert int(trace_id, 16) > 0
        assert int(span_id, 16) > 0

        assert len(
            executor.observed_contexts
        ) == 1

        observed = executor.observed_contexts[0]

        assert observed["trace_id"] == trace_id
        assert observed["span_id"] == span_id
        assert (
            observed["experiment_id"]
            == str(experiment.experiment_id)
        )
        assert (
            observed["job_id"]
            == str(job.job_id)
        )
        assert (
            observed["worker_id"]
            == "worker-traced-success"
        )
    finally:
        tracing_runtime.shutdown()


def test_worker_failure_events_share_one_trace_span(
    worker_runtime: DatabaseRuntime,
) -> None:
    """Failed worker lifecycle events must remain trace-correlated."""
    experiment = _create_experiment(
        worker_runtime
    )
    _create_job(
        worker_runtime,
        experiment_id=experiment.experiment_id,
        max_attempts=1,
    )

    stream = StringIO()
    configure_platform_logging(
        stream=stream,
    )

    exporter = InMemorySpanExporter()
    tracing_runtime = create_tracing_runtime(
        PlatformSettings(
            environment="test",
            service_name="vait-worker-test",
        ),
        span_exporter=exporter,
    )

    try:
        worker = WorkerService(
            runtime=worker_runtime,
            executor=FailingExecutor(),
            worker_id="worker-traced-failure",
            lease_seconds=60,
            tracing_runtime=tracing_runtime,
        )

        result = worker.run_once(
            now=datetime.now(UTC),
        )

        tracing_runtime.provider.force_flush()

        assert result.job is not None
        assert result.job.status == "FAILED"
        assert current_log_context() == {}

        spans = exporter.get_finished_spans()

        assert len(spans) == 1
        assert spans[0].name == "worker.job"

        payloads = [
            json.loads(line)
            for line in stream.getvalue().splitlines()
            if line
        ]

        lifecycle = [
            payload
            for payload in payloads
            if payload["event"]
            in {
                "job_claimed",
                "job_execution_failed",
                "job_failed",
            }
        ]

        assert len(lifecycle) == 3

        trace_ids = {
            payload["trace_id"]
            for payload in lifecycle
        }
        span_ids = {
            payload["span_id"]
            for payload in lifecycle
        }

        assert len(trace_ids) == 1
        assert len(span_ids) == 1

        serialized = stream.getvalue()

        assert "worker execution failed" not in serialized
        assert "Traceback" not in serialized
    finally:
        tracing_runtime.shutdown()



def test_worker_cannot_report_success_after_lease_expiry(
    worker_runtime: DatabaseRuntime,
) -> None:
    """Execution completion must not revive an expired claim."""
    experiment = _create_experiment(worker_runtime)
    job = _create_job(
        worker_runtime,
        experiment_id=experiment.experiment_id,
    )
    executor = SuccessfulExecutor(worker_runtime)

    worker = WorkerService(
        runtime=worker_runtime,
        executor=executor,
        worker_id="expired-worker",
        lease_seconds=60,
    )

    with pytest.raises(
        RuntimeError,
        match="Claimed job lost worker ownership",
    ):
        worker.run_once(
            now=datetime.now(UTC) - timedelta(minutes=2),
        )

    with transactional_session(worker_runtime) as session:
        persisted = JobRepository(session).get(job.job_id)

    assert executor.seen_job_ids == [job.job_id]
    assert persisted is not None
    assert persisted.status == "RUNNING"
    assert persisted.finished_at is None



def _create_provider_worker_job(
    runtime: DatabaseRuntime,
) -> JobRecord:
    """Create one allowlisted single-attempt provider job."""
    experiment = _create_experiment(runtime)
    job = JobRecord(
        job_id=uuid4(),
        experiment_id=experiment.experiment_id,
        operation="provider-inference",
        payload={},
        idempotency_key=str(uuid4()),
        max_attempts=1,
        created_at=datetime.now(UTC),
    )

    with transactional_session(runtime) as session:
        JobRepository(session).add(job)

    return job


def test_worker_provider_atomic_completion_has_one_success(
    worker_runtime: DatabaseRuntime,
) -> None:
    """Provider worker must not perform a second success transition."""
    job = _create_provider_worker_job(worker_runtime)

    invocation = Mock(spec=BudgetedJobProviderInvocation)
    invocation.invoke.return_value = ProviderInferenceResponse(
        request_id="worker-atomic-request",
        provider_id="fake-provider",
        runtime_id="fake-runtime",
        model_id="fake-model",
        output_text="synthetic-success",
        timing=ProviderTiming(total_latency_ms=1.0),
    )

    executor = BudgetedProviderJobExecutor(
        invocation=cast(BudgetedJobProviderInvocation, invocation),
        evidence_writer=PostgreSQLProviderEvidenceWriter(
            runtime=worker_runtime,
            worker_id="atomic-worker",
        ),
    )

    worker = WorkerService(
        runtime=worker_runtime,
        executor=executor,
        worker_id="atomic-worker",
        lease_seconds=60,
        allowed_operations=("provider-inference",),
    )

    result = worker.run_once(now=datetime.now(UTC))

    assert result.job is not None
    assert result.job.job_id == job.job_id
    assert result.job.status == "SUCCEEDED"
    assert invocation.invoke.call_count == 1

    with transactional_session(worker_runtime) as session:
        persisted = JobRepository(session).get(job.job_id)
        records = EvidenceRepository(session).list_for_experiment(
            job.experiment_id
        )

    assert persisted is not None
    assert persisted.status == "SUCCEEDED"
    assert len(records) == 1
    assert records[0].payload["job_id"] == str(job.job_id)


def test_provider_worker_refuses_success_without_atomic_evidence(
    worker_runtime: DatabaseRuntime,
) -> None:
    """An executor returning normally is not enough for provider success."""
    job = _create_provider_worker_job(worker_runtime)

    worker = WorkerService(
        runtime=worker_runtime,
        executor=SuccessfulExecutor(worker_runtime),
        worker_id="incomplete-worker",
        lease_seconds=60,
        allowed_operations=("provider-inference",),
    )

    with pytest.raises(
        RuntimeError,
        match="lacks durable atomic completion",
    ):
        worker.run_once(now=datetime.now(UTC))

    with transactional_session(worker_runtime) as session:
        persisted = JobRepository(session).get(job.job_id)
        records = EvidenceRepository(session).list_for_experiment(
            job.experiment_id
        )

    assert persisted is not None
    assert persisted.status == "RUNNING"
    assert persisted.finished_at is None
    assert records == ()



def test_worker_post_commit_crash_does_not_replay(
    worker_runtime: DatabaseRuntime,
) -> None:
    """A crash after atomic commit cannot re-execute a completed job."""
    job = _create_provider_worker_job(worker_runtime)

    writer = PostgreSQLProviderEvidenceWriter(
        runtime=worker_runtime,
        worker_id="crash-worker",
    )

    class PostCommitCrashExecutor:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, claimed: JobRecord) -> None:
            self.calls += 1
            completed = writer.persist_and_complete(
                job=claimed,
                response=ProviderInferenceResponse(
                    request_id="post-commit-crash",
                    provider_id="fake-provider",
                    runtime_id="fake-runtime",
                    model_id="fake-model",
                    output_text="durable-result",
                    timing=ProviderTiming(total_latency_ms=1.0),
                ),
            )
            assert completed.status == "SUCCEEDED"
            raise RuntimeError("simulated crash after commit")

    executor = PostCommitCrashExecutor()

    worker = WorkerService(
        runtime=worker_runtime,
        executor=executor,
        worker_id="crash-worker",
        lease_seconds=60,
        allowed_operations=("provider-inference",),
    )

    with pytest.raises(
        RuntimeError,
        match="Claimed job lost worker ownership",
    ):
        worker.run_once(now=datetime.now(UTC))

    with transactional_session(worker_runtime) as session:
        persisted = JobRepository(session).get(job.job_id)
        evidence = EvidenceRepository(session).list_for_experiment(
            job.experiment_id
        )

    assert persisted is not None
    assert persisted.status == "SUCCEEDED"
    assert persisted.finished_at is not None
    assert len(evidence) == 1
    assert executor.calls == 1

    next_cycle = worker.run_once(now=datetime.now(UTC))

    assert next_cycle.claimed is False
    assert next_cycle.job is None
    assert executor.calls == 1


def test_provider_transport_error_is_terminal_without_evidence(
    worker_runtime: DatabaseRuntime,
) -> None:
    """A single-attempt provider job cannot replay uncertain I/O."""
    job = _create_provider_worker_job(worker_runtime)

    invocation = Mock(spec=BudgetedJobProviderInvocation)
    invocation.invoke.side_effect = ConnectionError(
        "simulated provider connection failure"
    )

    executor = BudgetedProviderJobExecutor(
        invocation=cast(BudgetedJobProviderInvocation, invocation),
        evidence_writer=PostgreSQLProviderEvidenceWriter(
            runtime=worker_runtime,
            worker_id="failure-worker",
        ),
    )

    worker = WorkerService(
        runtime=worker_runtime,
        executor=executor,
        worker_id="failure-worker",
        lease_seconds=60,
        allowed_operations=("provider-inference",),
    )

    result = worker.run_once(now=datetime.now(UTC))

    assert result.job is not None
    assert result.job.status == "FAILED"
    assert result.job.attempt == 1
    assert result.job.error_code == "ConnectionError"

    with transactional_session(worker_runtime) as session:
        persisted = JobRepository(session).get(job.job_id)
        evidence = EvidenceRepository(session).list_for_experiment(
            job.experiment_id
        )

    assert persisted is not None
    assert persisted.status == "FAILED"
    assert evidence == ()

    next_cycle = worker.run_once(now=datetime.now(UTC))
    assert next_cycle.claimed is False

    invocation.invoke.assert_called_once()



def test_worker_recovers_expired_retry_after_restart(
    worker_runtime: DatabaseRuntime,
) -> None:
    """A replacement worker can resume an eligible expired job."""
    experiment = _create_experiment(worker_runtime)
    job = _create_job(
        worker_runtime,
        experiment_id=experiment.experiment_id,
        max_attempts=2,
    )

    with transactional_session(worker_runtime) as session:
        first = JobRepository(session).claim_next(
            worker_id="dead-worker",
            now=datetime.now(UTC) - timedelta(minutes=2),
            lease_seconds=60,
            allowed_operations=("verification",),
        )

    assert first is not None
    assert first.attempt == 1

    executor = SuccessfulExecutor(worker_runtime)
    replacement = WorkerService(
        runtime=worker_runtime,
        executor=executor,
        worker_id="replacement-worker",
        lease_seconds=60,
        allowed_operations=("verification",),
    )

    result = replacement.run_once(now=datetime.now(UTC))

    assert result.claimed is True
    assert result.job is not None
    assert result.job.job_id == job.job_id
    assert result.job.status == "SUCCEEDED"
    assert result.job.attempt == 2
    assert result.job.claimed_by == "replacement-worker"
    assert executor.seen_job_ids == [job.job_id]


def test_worker_recovery_does_not_replay_provider_job(
    worker_runtime: DatabaseRuntime,
) -> None:
    """An expired single-attempt provider job becomes FAILED, not retried."""
    job = _create_provider_worker_job(worker_runtime)

    with transactional_session(worker_runtime) as session:
        first = JobRepository(session).claim_next(
            worker_id="dead-provider",
            now=datetime.now(UTC) - timedelta(minutes=2),
            lease_seconds=60,
            allowed_operations=("provider-inference",),
        )

    assert first is not None
    assert first.attempt == 1

    executor = SuccessfulExecutor(worker_runtime)
    replacement = WorkerService(
        runtime=worker_runtime,
        executor=executor,
        worker_id="replacement-provider",
        lease_seconds=60,
        allowed_operations=("provider-inference",),
    )

    result = replacement.run_once(now=datetime.now(UTC))

    assert result.claimed is False
    assert executor.seen_job_ids == []

    with transactional_session(worker_runtime) as session:
        persisted = JobRepository(session).get(job.job_id)
        records = EvidenceRepository(session).list_for_experiment(
            job.experiment_id
        )

    assert persisted is not None
    assert persisted.status == "FAILED"
    assert persisted.attempt == 1
    assert persisted.error_code == "LEASE_EXPIRED"
    assert records == ()


def test_worker_recovery_respects_operation_allowlist(
    worker_runtime: DatabaseRuntime,
) -> None:
    """A verification worker cannot recover unrelated provider work."""
    provider_job = _create_provider_worker_job(worker_runtime)

    with transactional_session(worker_runtime) as session:
        claimed = JobRepository(session).claim_next(
            worker_id="old-provider",
            now=datetime.now(UTC) - timedelta(minutes=2),
            lease_seconds=60,
            allowed_operations=("provider-inference",),
        )

    assert claimed is not None

    experiment = _create_experiment(worker_runtime)
    verification_job = _create_job(
        worker_runtime,
        experiment_id=experiment.experiment_id,
    )

    executor = SuccessfulExecutor(worker_runtime)
    worker = WorkerService(
        runtime=worker_runtime,
        executor=executor,
        worker_id="verification-only",
        lease_seconds=60,
        allowed_operations=("verification",),
    )

    result = worker.run_once(now=datetime.now(UTC))

    assert result.job is not None
    assert result.job.job_id == verification_job.job_id
    assert result.job.status == "SUCCEEDED"

    with transactional_session(worker_runtime) as session:
        untouched = JobRepository(session).get(provider_job.job_id)

    assert untouched is not None
    assert untouched.status == "RUNNING"
    assert untouched.claimed_by == "old-provider"
