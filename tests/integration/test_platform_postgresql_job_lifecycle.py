"""PostgreSQL lifecycle tests for M5-F durable jobs."""

import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from threading import Barrier, Lock
from uuid import UUID, uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text

from vait.contracts.budgeting import (
    BudgetDeniedError,
    BudgetEstimate,
    BudgetPolicy,
    BudgetReservationRequest,
    BudgetReservationState,
)
from vait.inference.providers.cache import DeterministicCachingProvider
from vait.inference.providers.models import (
    InferenceMessage,
    InferenceRole,
    ProviderError,
    ProviderInferenceRequest,
    ProviderInferenceResponse,
    ProviderTiming,
    ProviderTokenUsage,
)
from vait.platform.jobs import JobRecord
from vait.platform.jobs.provider_evidence import PostgreSQLProviderEvidenceWriter
from vait.platform.persistence.engine import (
    DatabaseRuntime,
    create_database_runtime,
)
from vait.platform.persistence.repositories.budgets import (
    PostgreSQLBudgetReservationLedger,
)
from vait.platform.persistence.repositories.jobs import (
    JobRepository,
)
from vait.platform.persistence.repositories.registry import (
    EvidenceRepository,
    ExperimentRunRepository,
)
from vait.platform.persistence.session import (
    transactional_session,
)
from vait.platform.provider_execution import (
    BudgetedJobProviderInvocation,
    ProviderBudgetPlan,
    ProviderInvocationFailed,
)
from vait.platform.registry import ExperimentRunRecord
from vait.platform.settings import PlatformSettings

_PROVIDER_WORKER_ID = "provider-worker-test"

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
    operation: str = "verification",
) -> JobRecord:
    job = JobRecord(
        job_id=uuid4(),
        experiment_id=experiment_id,
        operation=operation,
        payload={},
        idempotency_key=str(uuid4()),
        max_attempts=max_attempts,
        created_at=datetime.now(UTC),
    )

    with transactional_session(runtime) as session:
        JobRepository(session).add(job)

    return job


def test_worker_claim_filters_unsupported_operations(
    jobs_runtime: DatabaseRuntime,
) -> None:
    """Unregistered jobs must not be claimed by a scoped worker."""
    experiment = _create_experiment(jobs_runtime)

    unsupported = _create_pending_job(
        jobs_runtime,
        experiment_id=experiment.experiment_id,
        operation="unregistered-operation",
    )
    supported = _create_pending_job(
        jobs_runtime,
        experiment_id=experiment.experiment_id,
        operation="verification",
    )

    with transactional_session(jobs_runtime) as session:
        repository = JobRepository(session)
        claimed = repository.claim_next(
            worker_id="scoped-worker",
            now=datetime.now(UTC),
            lease_seconds=60,
            allowed_operations=("verification",),
        )
        untouched = repository.get(unsupported.job_id)

    assert claimed is not None
    assert claimed.job_id == supported.job_id
    assert untouched is not None
    assert untouched.status == "PENDING"
    assert untouched.attempt == 0


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
            expected_attempt=claimed.attempt,
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
            expected_attempt=claimed.attempt,
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
            expected_attempt=claimed.attempt,
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
            expected_attempt=first_claim.attempt,
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
            expected_attempt=claimed.attempt,
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


class DatabaseCheckingProvider:
    """Record one fake provider call and observe its committed dispatch."""

    provider_id = "postgres-test-provider"
    runtime_id = "postgres-test-runtime"

    def __init__(
        self,
        runtime: DatabaseRuntime,
        *,
        reservation_ids: tuple[UUID, ...] = (),
        raise_failure: bool = False,
        return_failure: bool = False,
    ) -> None:
        self.runtime = runtime
        self.reservation_ids = list(reservation_ids)
        self.raise_failure = raise_failure
        self.return_failure = return_failure
        self.requests: list[ProviderInferenceRequest] = []
        self.dispatch_states: list[str | None] = []
        self._lock = Lock()

    def generate(
        self,
        request: ProviderInferenceRequest,
    ) -> ProviderInferenceResponse:
        with self._lock:
            self.requests.append(request)
            reservation_id = (
                self.reservation_ids.pop(0)
                if self.reservation_ids
                else None
            )

        if reservation_id is not None:
            with self.runtime.engine.connect() as connection:
                state = connection.scalar(
                    text(
                        "SELECT state FROM budget_reservations "
                        "WHERE reservation_id = :reservation_id"
                    ),
                    {"reservation_id": reservation_id},
                )
            self.dispatch_states.append(state)

        if self.raise_failure:
            raise ConnectionError("Simulated provider transport failure")

        if self.return_failure:
            return ProviderInferenceResponse(
                request_id=request.request_id,
                provider_id=self.provider_id,
                runtime_id=self.runtime_id,
                model_id=request.model_id,
                timing=ProviderTiming(total_latency_ms=1.0),
                error=ProviderError(
                    error_type="provider_error",
                    message="The provider rejected the request.",
                ),
            )

        return ProviderInferenceResponse(
            request_id=request.request_id,
            provider_id=self.provider_id,
            runtime_id=self.runtime_id,
            model_id=request.model_id,
            output_text="verified response",
            usage=ProviderTokenUsage(input_tokens=3, output_tokens=2),
            timing=ProviderTiming(total_latency_ms=1.0),
        )


class StaticProviderBudgetPlanResolver:
    """Provide trusted test plans while retaining claimed job identities."""

    def __init__(
        self,
        plan: ProviderBudgetPlan,
        *,
        barrier: Barrier | None = None,
    ) -> None:
        self.plan = plan
        self.barrier = barrier
        self.jobs: list[JobRecord] = []
        self._lock = Lock()

    def resolve(self, job: JobRecord) -> ProviderBudgetPlan:
        with self._lock:
            self.jobs.append(job)

        if self.barrier is not None:
            self.barrier.wait(timeout=10)

        return self.plan


class LeaseRecoveringProviderBudgetPlanResolver:
    """Recover the claimed job during planning to exercise the dispatch fence."""

    def __init__(
        self,
        runtime: DatabaseRuntime,
        plan: ProviderBudgetPlan,
    ) -> None:
        self.runtime = runtime
        self.plan = plan
        self.jobs: list[JobRecord] = []
        self.recovered: list[JobRecord] = []

    def resolve(self, job: JobRecord) -> ProviderBudgetPlan:
        self.jobs.append(job)
        assert job.lease_expires_at is not None
        with transactional_session(self.runtime) as session:
            self.recovered = JobRepository(session).recover_expired_leases(
                now=job.lease_expires_at + timedelta(seconds=1),
            )
        return self.plan


def _provider_policy(
    *,
    max_requests: int = 10,
    max_tokens: int = 1000,
) -> BudgetPolicy:
    return BudgetPolicy.model_validate(
        {
            "policy_id": "postgres-job-provider-test",
            "policy_version": 1,
            "scope": {
                "kind": "principal",
                "scope_id": str(uuid4()),
            },
            "window": {"kind": "lifetime"},
            "request_limit": {"max_requests": max_requests},
            "token_limit": {"max_tokens": max_tokens},
            "retry_budget": {"max_retries": 0},
            "provider_attempt_budget": {"max_attempts": max_requests},
        }
    )


def _provider_plan(
    policy: BudgetPolicy,
    *,
    request_id: str | None = None,
    prompt: str = "Verify this evidence.",
    temperature: float = 0.0,
) -> ProviderBudgetPlan:
    return ProviderBudgetPlan(
        request=ProviderInferenceRequest(
            request_id=request_id or str(uuid4()),
            model_id="postgres-test-model",
            messages=(
                InferenceMessage(role=InferenceRole.USER, content=prompt),
            ),
            max_output_tokens=16,
            temperature=temperature,
        ),
        policies=(policy,),
        estimate=BudgetEstimate(
            provider_requests=1,
            provider_attempts=1,
            retries=0,
            provider_tokens=20,
            token_estimator_id="postgres-test-estimator",
            token_estimator_version="1",
        ),
    )


def _claim_provider_job(
    runtime: DatabaseRuntime,
    *,
    experiment_id: UUID,
    max_attempts: int = 3,
) -> JobRecord:
    job = _create_pending_job(
        runtime,
        experiment_id=experiment_id,
        max_attempts=max_attempts,
    )
    with transactional_session(runtime) as session:
        claimed = JobRepository(session).claim_next(
            worker_id=_PROVIDER_WORKER_ID,
            now=datetime.now(UTC),
            lease_seconds=60,
        )

    assert claimed is not None
    assert claimed.job_id == job.job_id
    return claimed


def _budget_row(
    runtime: DatabaseRuntime,
    reservation_id: UUID,
) -> tuple[str, int | None, int | None, int | None]:
    with runtime.engine.connect() as connection:
        row = connection.execute(
            text(
                "SELECT state, consumed_provider_requests, "
                "consumed_provider_attempts, consumed_provider_tokens "
                "FROM budget_reservations "
                "WHERE reservation_id = :reservation_id"
            ),
            {"reservation_id": reservation_id},
        ).one()
    return (
        row.state,
        row.consumed_provider_requests,
        row.consumed_provider_attempts,
        row.consumed_provider_tokens,
    )


def test_budgeted_job_invocation_commits_admission_before_provider_io(
    jobs_runtime: DatabaseRuntime,
) -> None:
    """A claimed durable job reaches the fake provider only after dispatch commits."""
    experiment = _create_experiment(jobs_runtime)
    job = _claim_provider_job(
        jobs_runtime,
        experiment_id=experiment.experiment_id,
    )
    plan = _provider_plan(_provider_policy())
    provider = DatabaseCheckingProvider(
        jobs_runtime,
        reservation_ids=(job.job_id,),
    )
    invocation = BudgetedJobProviderInvocation(
        provider=provider,
        runtime=jobs_runtime,
        plan_resolver=StaticProviderBudgetPlanResolver(plan),
        worker_id=_PROVIDER_WORKER_ID,
    )

    response = invocation.invoke(job)

    assert response.succeeded
    assert provider.dispatch_states == [BudgetReservationState.DISPATCHED.value]
    assert provider.requests == [plan.request]
    assert _budget_row(jobs_runtime, job.job_id) == (
        BudgetReservationState.SETTLED.value,
        1,
        1,
        5,
    )


def test_budgeted_job_invocation_denies_before_provider_io(
    jobs_runtime: DatabaseRuntime,
) -> None:
    """A full durable request budget blocks provider I/O for a claimed job."""
    experiment = _create_experiment(jobs_runtime)
    job = _claim_provider_job(
        jobs_runtime,
        experiment_id=experiment.experiment_id,
    )
    policy = _provider_policy(max_requests=1)
    plan = _provider_plan(policy)
    seed = BudgetReservationRequest(
        reservation_id=uuid4(),
        policies=(policy,),
        estimate=plan.estimate,
    )
    with transactional_session(jobs_runtime) as session:
        PostgreSQLBudgetReservationLedger(session).reserve(seed)

    provider = DatabaseCheckingProvider(jobs_runtime)
    invocation = BudgetedJobProviderInvocation(
        provider=provider,
        runtime=jobs_runtime,
        plan_resolver=StaticProviderBudgetPlanResolver(plan),
        worker_id=_PROVIDER_WORKER_ID,
    )

    with pytest.raises(BudgetDeniedError):
        invocation.invoke(job)

    assert provider.requests == []
    with jobs_runtime.engine.connect() as connection:
        reservation_count = connection.scalar(
            text(
                "SELECT COUNT(*) FROM budget_reservations "
                "WHERE reservation_id = :reservation_id"
            ),
            {"reservation_id": job.job_id},
        )
    assert reservation_count == 0


def test_unpersisted_job_record_cannot_reach_provider_io(
    jobs_runtime: DatabaseRuntime,
) -> None:
    """A caller-created RUNNING record is not a durable worker claim."""
    experiment = _create_experiment(jobs_runtime)
    now = datetime.now(UTC)
    job = JobRecord(
        job_id=uuid4(),
        experiment_id=experiment.experiment_id,
        operation="provider-inference",
        payload={},
        status="RUNNING",
        attempt=1,
        max_attempts=3,
        idempotency_key="fabricated-job-key",
        claimed_by=_PROVIDER_WORKER_ID,
        created_at=now - timedelta(seconds=1),
        started_at=now,
        heartbeat_at=now,
        lease_expires_at=now + timedelta(minutes=1),
    )
    plan = _provider_plan(_provider_policy())
    provider = DatabaseCheckingProvider(jobs_runtime)
    resolver = StaticProviderBudgetPlanResolver(plan)
    invocation = BudgetedJobProviderInvocation(
        provider=provider,
        runtime=jobs_runtime,
        plan_resolver=resolver,
        worker_id=_PROVIDER_WORKER_ID,
    )

    with pytest.raises(ValueError, match="active durable job claim"):
        invocation.invoke(job)

    assert resolver.jobs == []
    assert provider.requests == []
    with jobs_runtime.engine.connect() as connection:
        reservation_count = connection.scalar(
            text(
                "SELECT COUNT(*) FROM budget_reservations "
                "WHERE reservation_id = :reservation_id"
            ),
            {"reservation_id": job.job_id},
        )
    assert reservation_count == 0


def test_recovered_job_claim_cannot_reach_provider_io(
    jobs_runtime: DatabaseRuntime,
) -> None:
    """A stale RUNNING snapshot is rejected after durable lease recovery."""
    experiment = _create_experiment(jobs_runtime)
    job = _claim_provider_job(
        jobs_runtime,
        experiment_id=experiment.experiment_id,
    )
    assert job.lease_expires_at is not None
    with transactional_session(jobs_runtime) as session:
        recovered = JobRepository(session).recover_expired_leases(
            now=job.lease_expires_at + timedelta(seconds=1),
        )
    assert [record.job_id for record in recovered] == [job.job_id]

    plan = _provider_plan(_provider_policy())
    provider = DatabaseCheckingProvider(jobs_runtime)
    resolver = StaticProviderBudgetPlanResolver(plan)
    invocation = BudgetedJobProviderInvocation(
        provider=provider,
        runtime=jobs_runtime,
        plan_resolver=resolver,
        worker_id=_PROVIDER_WORKER_ID,
    )

    with pytest.raises(ValueError, match="active durable job claim"):
        invocation.invoke(job)

    assert resolver.jobs == []
    assert provider.requests == []


def test_claim_recovered_during_plan_resolution_prevents_provider_io(
    jobs_runtime: DatabaseRuntime,
) -> None:
    """The provider boundary rechecks ownership after policy resolution."""
    experiment = _create_experiment(jobs_runtime)
    job = _claim_provider_job(
        jobs_runtime,
        experiment_id=experiment.experiment_id,
    )
    plan = _provider_plan(_provider_policy())
    provider = DatabaseCheckingProvider(
        jobs_runtime,
        reservation_ids=(job.job_id,),
    )
    resolver = LeaseRecoveringProviderBudgetPlanResolver(jobs_runtime, plan)
    invocation = BudgetedJobProviderInvocation(
        provider=provider,
        runtime=jobs_runtime,
        plan_resolver=resolver,
        worker_id=_PROVIDER_WORKER_ID,
    )

    with pytest.raises(ValueError, match="active durable job claim"):
        invocation.invoke(job)

    assert resolver.jobs == [job]
    assert [record.job_id for record in resolver.recovered] == [job.job_id]
    assert provider.requests == []
    assert _budget_row(jobs_runtime, job.job_id) == (
        BudgetReservationState.SETTLED.value,
        1,
        1,
        None,
    )


def test_budgeted_job_invocation_blocks_replay_for_the_same_job(
    jobs_runtime: DatabaseRuntime,
) -> None:
    """Replaying one claimed job cannot claim a second provider dispatch."""
    experiment = _create_experiment(jobs_runtime)
    job = _claim_provider_job(
        jobs_runtime,
        experiment_id=experiment.experiment_id,
    )
    plan = _provider_plan(_provider_policy())
    provider = DatabaseCheckingProvider(jobs_runtime)
    invocation = BudgetedJobProviderInvocation(
        provider=provider,
        runtime=jobs_runtime,
        plan_resolver=StaticProviderBudgetPlanResolver(plan),
        worker_id=_PROVIDER_WORKER_ID,
    )

    assert invocation.invoke(job).succeeded
    with pytest.raises(RuntimeError, match="Budget admission did not produce"):
        invocation.invoke(job)

    assert len(provider.requests) == 1


def test_concurrent_budgeted_job_replay_calls_provider_once(
    jobs_runtime: DatabaseRuntime,
) -> None:
    """Concurrent deliveries of one job have a single committed dispatch claim."""
    experiment = _create_experiment(jobs_runtime)
    job = _claim_provider_job(
        jobs_runtime,
        experiment_id=experiment.experiment_id,
    )
    plan = _provider_plan(_provider_policy())
    provider = DatabaseCheckingProvider(
        jobs_runtime,
        reservation_ids=(job.job_id,),
    )
    resolver = StaticProviderBudgetPlanResolver(plan, barrier=Barrier(2))
    invocation = BudgetedJobProviderInvocation(
        provider=provider,
        runtime=jobs_runtime,
        plan_resolver=resolver,
        worker_id=_PROVIDER_WORKER_ID,
    )

    def invoke_or_capture() -> tuple[ProviderInferenceResponse | None, Exception | None]:
        try:
            return invocation.invoke(job), None
        except Exception as exc:
            return None, exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(invoke_or_capture) for _ in range(2)]
        results = [future.result(timeout=15) for future in futures]

    assert sum(response is not None for response, _ in results) == 1
    assert sum(error is not None for _, error in results) == 1
    assert len(resolver.jobs) == 2
    assert provider.dispatch_states == [BudgetReservationState.DISPATCHED.value]
    assert len(provider.requests) == 1


def test_provider_exception_settles_unknown_consumption_for_the_job(
    jobs_runtime: DatabaseRuntime,
) -> None:
    """A failed provider call remains durably charged as unknown consumption."""
    experiment = _create_experiment(jobs_runtime)
    job = _claim_provider_job(
        jobs_runtime,
        experiment_id=experiment.experiment_id,
        max_attempts=1,
    )
    plan = _provider_plan(_provider_policy())
    provider = DatabaseCheckingProvider(
        jobs_runtime,
        reservation_ids=(job.job_id,),
        raise_failure=True,
    )
    invocation = BudgetedJobProviderInvocation(
        provider=provider,
        runtime=jobs_runtime,
        plan_resolver=StaticProviderBudgetPlanResolver(plan),
        worker_id=_PROVIDER_WORKER_ID,
    )

    with pytest.raises(ConnectionError, match="Simulated provider transport failure"):
        invocation.invoke(job)

    assert provider.dispatch_states == [BudgetReservationState.DISPATCHED.value]
    assert _budget_row(jobs_runtime, job.job_id) == (
        BudgetReservationState.SETTLED.value,
        1,
        1,
        None,
    )


def test_normalised_provider_failure_settles_unknown_consumption(
    jobs_runtime: DatabaseRuntime,
) -> None:
    """A normalised provider error is not mistaken for successful job work."""
    experiment = _create_experiment(jobs_runtime)
    job = _claim_provider_job(
        jobs_runtime,
        experiment_id=experiment.experiment_id,
        max_attempts=1,
    )
    plan = _provider_plan(_provider_policy())
    provider = DatabaseCheckingProvider(
        jobs_runtime,
        reservation_ids=(job.job_id,),
        return_failure=True,
    )
    invocation = BudgetedJobProviderInvocation(
        provider=provider,
        runtime=jobs_runtime,
        plan_resolver=StaticProviderBudgetPlanResolver(plan),
        worker_id=_PROVIDER_WORKER_ID,
    )

    with pytest.raises(ProviderInvocationFailed) as captured:
        invocation.invoke(job)

    assert captured.value.response.error is not None
    assert _budget_row(jobs_runtime, job.job_id) == (
        BudgetReservationState.SETTLED.value,
        1,
        1,
        None,
    )


def test_cache_hits_and_bypasses_still_require_budget_admission(
    jobs_runtime: DatabaseRuntime,
) -> None:
    """Cache hit, miss and bypass paths all cross the durable dispatch gate."""
    experiment = _create_experiment(jobs_runtime)
    policy = _provider_policy(max_requests=3)
    underlying = DatabaseCheckingProvider(jobs_runtime)
    cached_provider = DeterministicCachingProvider(provider=underlying)
    requests = (
        ("cache-miss", 0.0),
        ("cache-hit", 0.0),
        ("cache-bypass", 0.7),
        ("denied-cache-bypass", 0.9),
    )
    job_ids: list[UUID] = []

    for request_id, temperature in requests:
        job = _claim_provider_job(
            jobs_runtime,
            experiment_id=experiment.experiment_id,
        )
        job_ids.append(job.job_id)
        plan = _provider_plan(
            policy,
            request_id=request_id,
            temperature=temperature,
        )
        invocation = BudgetedJobProviderInvocation(
            provider=cached_provider,
            runtime=jobs_runtime,
            plan_resolver=StaticProviderBudgetPlanResolver(plan),
            worker_id=_PROVIDER_WORKER_ID,
        )

        if request_id == "denied-cache-bypass":
            with pytest.raises(BudgetDeniedError):
                invocation.invoke(job)
        else:
            assert invocation.invoke(job).succeeded

    assert len(underlying.requests) == 2
    assert cached_provider.cache_misses == 1
    assert cached_provider.cache_hits == 1
    assert cached_provider.cache_bypasses == 1
    for job_id in job_ids[:3]:
        assert _budget_row(jobs_runtime, job_id)[0] == BudgetReservationState.SETTLED.value
    with jobs_runtime.engine.connect() as connection:
        denied_reservation_count = connection.scalar(
            text(
                "SELECT COUNT(*) FROM budget_reservations "
                "WHERE reservation_id = :reservation_id"
            ),
            {"reservation_id": job_ids[3]},
        )
    assert denied_reservation_count == 0



@pytest.mark.parametrize(
    "action",
    ["heartbeat", "success", "failure"],
)
def test_expired_claim_cannot_mutate_job(
    jobs_runtime: DatabaseRuntime,
    action: str,
) -> None:
    """An expired lease blocks renewal and terminal writes."""
    experiment = _create_experiment(jobs_runtime)
    job = _create_pending_job(
        jobs_runtime,
        experiment_id=experiment.experiment_id,
    )

    claimed_at = datetime.now(UTC) - timedelta(minutes=2)

    with transactional_session(jobs_runtime) as session:
        claimed = JobRepository(session).claim_next(
            worker_id="expired-worker",
            now=claimed_at,
            lease_seconds=60,
        )

    assert claimed is not None
    assert claimed.status == "RUNNING"

    finished_at = datetime.now(UTC)

    with transactional_session(jobs_runtime) as session:
        repository = JobRepository(session)

        if action == "heartbeat":
            rejected = repository.heartbeat(
                expected_attempt=claimed.attempt,
                job_id=job.job_id,
                worker_id="expired-worker",
                now=finished_at,
                lease_seconds=60,
            )
        elif action == "success":
            rejected = repository.mark_succeeded(
                expected_attempt=claimed.attempt,
                job_id=job.job_id,
                worker_id="expired-worker",
                now=finished_at,
            )
        else:
            rejected = repository.mark_failed_or_retry(
                expected_attempt=claimed.attempt,
                job_id=job.job_id,
                worker_id="expired-worker",
                now=finished_at,
                error_code="EXPIRED_TEST",
            )

        persisted = repository.get(job.job_id)

    assert rejected is None
    assert persisted is not None
    assert persisted.status == "RUNNING"
    assert persisted.attempt == 1
    assert persisted.lease_expires_at == claimed.lease_expires_at



def test_old_attempt_cannot_mutate_reclaimed_same_worker(
    jobs_runtime: DatabaseRuntime,
) -> None:
    """Reusing worker_id cannot grant an old attempt ownership."""
    experiment = _create_experiment(jobs_runtime)
    job = _create_pending_job(
        jobs_runtime,
        experiment_id=experiment.experiment_id,
        max_attempts=2,
    )

    with transactional_session(jobs_runtime) as session:
        first = JobRepository(session).claim_next(
            worker_id="reused-worker",
            now=datetime.now(UTC) - timedelta(minutes=2),
            lease_seconds=60,
        )

    assert first is not None
    assert first.attempt == 1

    with transactional_session(jobs_runtime) as session:
        repository = JobRepository(session)
        recovered = repository.recover_expired_leases(
            now=datetime.now(UTC),
        )
        assert len(recovered) == 1
        requeued = repository.requeue_retry(job_id=job.job_id)

    assert requeued is not None
    assert requeued.status == "PENDING"

    with transactional_session(jobs_runtime) as session:
        second = JobRepository(session).claim_next(
            worker_id="reused-worker",
            now=datetime.now(UTC),
            lease_seconds=60,
        )

    assert second is not None
    assert second.attempt == 2

    with transactional_session(jobs_runtime) as session:
        repository = JobRepository(session)

        stale_success = repository.mark_succeeded(
            job_id=job.job_id,
            worker_id="reused-worker",
            expected_attempt=first.attempt,
            now=datetime.now(UTC),
        )
        stale_failure = repository.mark_failed_or_retry(
            job_id=job.job_id,
            worker_id="reused-worker",
            expected_attempt=first.attempt,
            now=datetime.now(UTC),
            error_code="STALE_ATTEMPT",
        )
        stale_heartbeat = repository.heartbeat(
            job_id=job.job_id,
            worker_id="reused-worker",
            expected_attempt=first.attempt,
            now=datetime.now(UTC),
            lease_seconds=60,
        )
        current = repository.get(job.job_id)

    assert stale_success is None
    assert stale_failure is None
    assert stale_heartbeat is None
    assert current is not None
    assert current.status == "RUNNING"
    assert current.attempt == 2
    assert current.claimed_by == "reused-worker"

    with transactional_session(jobs_runtime) as session:
        completed = JobRepository(session).mark_succeeded(
            job_id=job.job_id,
            worker_id="reused-worker",
            expected_attempt=second.attempt,
            now=datetime.now(UTC),
        )

    assert completed is not None
    assert completed.status == "SUCCEEDED"



def _provider_evidence_response() -> ProviderInferenceResponse:
    return ProviderInferenceResponse(
        request_id="evidence-request",
        provider_id="test-provider",
        runtime_id="test-runtime",
        model_id="test-model",
        output_text="api_key=must-never-be-persisted",
        usage=ProviderTokenUsage(input_tokens=3, output_tokens=2),
        timing=ProviderTiming(total_latency_ms=2.0),
    )


def _provider_evidence_claim(
    runtime: DatabaseRuntime,
    *,
    expired: bool = False,
) -> JobRecord:
    experiment = _create_experiment(runtime)
    job = _create_pending_job(
        runtime,
        experiment_id=experiment.experiment_id,
        operation="provider-inference",
        max_attempts=1,
    )
    claim_time = datetime.now(UTC)
    if expired:
        claim_time -= timedelta(minutes=2)

    with transactional_session(runtime) as session:
        claimed = JobRepository(session).claim_next(
            worker_id="evidence-worker",
            now=claim_time,
            lease_seconds=60,
            allowed_operations=("provider-inference",),
        )

    assert claimed is not None
    assert claimed.job_id == job.job_id
    return claimed


def test_provider_evidence_writer_commits_once(
    jobs_runtime: DatabaseRuntime,
) -> None:
    """Identical replay must not duplicate committed inference evidence."""
    job = _provider_evidence_claim(jobs_runtime)
    response = _provider_evidence_response()
    assert response.output_text is not None
    writer = PostgreSQLProviderEvidenceWriter(
        runtime=jobs_runtime,
        worker_id="evidence-worker",
    )

    writer.persist(job=job, response=response)
    writer.persist(job=job, response=response)

    with transactional_session(jobs_runtime) as session:
        records = EvidenceRepository(session).list_for_experiment(
            job.experiment_id
        )

    assert len(records) == 1
    assert records[0].kind == "inference"
    assert records[0].payload["job_id"] == str(job.job_id)
    assert records[0].payload["output_sha256"] == sha256(
        response.output_text.encode("utf-8")
    ).hexdigest()
    assert response.output_text not in str(records[0].payload)
    assert "output_text" not in records[0].payload
    assert "metadata" not in records[0].payload


def test_provider_evidence_writer_rejects_expired_claim(
    jobs_runtime: DatabaseRuntime,
) -> None:
    job = _provider_evidence_claim(jobs_runtime, expired=True)
    writer = PostgreSQLProviderEvidenceWriter(
        runtime=jobs_runtime,
        worker_id="evidence-worker",
    )

    with pytest.raises(RuntimeError, match="active first-attempt claim"):
        writer.persist(job=job, response=_provider_evidence_response())

    with transactional_session(jobs_runtime) as session:
        records = EvidenceRepository(session).list_for_experiment(
            job.experiment_id
        )

    assert records == ()


def test_provider_evidence_writer_rejects_other_worker(
    jobs_runtime: DatabaseRuntime,
) -> None:
    job = _provider_evidence_claim(jobs_runtime)
    writer = PostgreSQLProviderEvidenceWriter(
        runtime=jobs_runtime,
        worker_id="other-worker",
    )

    with pytest.raises(ValueError, match="not eligible"):
        writer.persist(job=job, response=_provider_evidence_response())

    with transactional_session(jobs_runtime) as session:
        records = EvidenceRepository(session).list_for_experiment(
            job.experiment_id
        )

    assert records == ()


def test_provider_evidence_writer_rejects_conflicting_replay(
    jobs_runtime: DatabaseRuntime,
) -> None:
    job = _provider_evidence_claim(jobs_runtime)
    response = _provider_evidence_response()
    assert response.output_text is not None
    writer = PostgreSQLProviderEvidenceWriter(
        runtime=jobs_runtime,
        worker_id="evidence-worker",
    )
    writer.persist(job=job, response=response)

    changed = response.model_copy(
        update={"output_text": "different-result"}
    )

    with pytest.raises(RuntimeError, match="Conflicting provider evidence"):
        writer.persist(job=job, response=changed)

    with transactional_session(jobs_runtime) as session:
        records = EvidenceRepository(session).list_for_experiment(
            job.experiment_id
        )

    assert len(records) == 1
    assert records[0].payload["output_sha256"] == sha256(
        response.output_text.encode("utf-8")
    ).hexdigest()
