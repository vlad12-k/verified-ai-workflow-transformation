"""Tests for durable, budgeted provider invocation from claimed jobs."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import uuid4

import pytest

import vait.platform.provider_execution as provider_execution
from vait.contracts.budgeting import BudgetEstimate, BudgetPolicy
from vait.inference.providers.models import (
    InferenceMessage,
    InferenceRole,
    ProviderError,
    ProviderInferenceRequest,
    ProviderInferenceResponse,
    ProviderTiming,
    ProviderTokenUsage,
)
from vait.platform.jobs.models import JobRecord
from vait.platform.persistence.engine import DatabaseRuntime


class RecordingProvider:
    """Represent one configured provider without external network access."""

    provider_id = "test-provider"
    runtime_id = "test-runtime"

    def __init__(self) -> None:
        self.requests: list[ProviderInferenceRequest] = []
        self.return_failure = False

    def generate(
        self,
        request: ProviderInferenceRequest,
    ) -> ProviderInferenceResponse:
        self.requests.append(request)
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
            output_text="approved response",
            usage=ProviderTokenUsage(input_tokens=3, output_tokens=2),
            timing=ProviderTiming(total_latency_ms=1.0),
        )


class RecordingResolver:
    """Return a fixed, trusted plan and record the durable job context."""

    def __init__(self, plan: provider_execution.ProviderBudgetPlan) -> None:
        self.plan = plan
        self.jobs: list[JobRecord] = []

    def resolve(
        self,
        job: JobRecord,
    ) -> provider_execution.ProviderBudgetPlan:
        self.jobs.append(job)
        return self.plan


class FakeJobSession:
    """Expose one durable job row to the repository without a database."""

    def __init__(self, job: JobRecord | None) -> None:
        self.job = job

    def get(self, model: type[object], job_id: object) -> JobRecord | None:
        del model
        if self.job is not None and self.job.job_id == job_id:
            return self.job
        return None

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        pass


class FakeDatabaseRuntime:
    """Provide repository lookups backed by an in-memory job record."""

    def __init__(self, job: JobRecord | None) -> None:
        self.job = job

    def session_factory(self) -> FakeJobSession:
        return FakeJobSession(self.job)


class RecordingDispatcher:
    """Capture the inputs delegated to durable budget enforcement."""

    def __init__(self, *, provider: RecordingProvider, runtime: DatabaseRuntime) -> None:
        self.provider = provider
        self.runtime = runtime
        self.arguments: dict[str, object] | None = None

    def execute(self, **arguments: object) -> ProviderInferenceResponse:
        self.arguments = arguments
        request = cast(ProviderInferenceRequest, arguments["request"])
        return self.provider.generate(request)


def _plan() -> provider_execution.ProviderBudgetPlan:
    request = ProviderInferenceRequest(
        request_id="client-correlation-id",
        model_id="test-model",
        messages=(
            InferenceMessage(role=InferenceRole.USER, content="Check the evidence."),
        ),
        max_output_tokens=16,
    )
    policy = BudgetPolicy.model_validate(
        {
            "policy_id": "trusted-test-policy",
            "policy_version": 1,
            "scope": {"kind": "principal", "scope_id": "test-principal"},
            "window": {"kind": "lifetime"},
            "request_limit": {"max_requests": 10},
            "token_limit": {"max_tokens": 100},
            "retry_budget": {"max_retries": 0},
            "provider_attempt_budget": {"max_attempts": 1},
        }
    )
    estimate = BudgetEstimate(
        provider_requests=1,
        provider_attempts=1,
        retries=0,
        provider_tokens=20,
        token_estimator_id="trusted-test-estimator",
        token_estimator_version="1",
    )
    return provider_execution.ProviderBudgetPlan(
        request=request,
        policies=(policy,),
        estimate=estimate,
    )


def _claimed_job(
    *,
    attempt: int = 1,
    lease_offset: timedelta = timedelta(minutes=1),
) -> JobRecord:
    now = datetime.now(UTC)
    return JobRecord(
        job_id=uuid4(),
        experiment_id=uuid4(),
        operation="provider-inference",
        payload={"prompt": "Check the evidence."},
        status="RUNNING",
        attempt=attempt,
        max_attempts=3,
        idempotency_key="caller-selected-idempotency-key",
        claimed_by="worker-01",
        created_at=now - timedelta(seconds=1),
        started_at=now,
        heartbeat_at=now,
        lease_expires_at=now + lease_offset,
    )


def _runtime_with_job(job: JobRecord | None) -> DatabaseRuntime:
    return cast(DatabaseRuntime, FakeDatabaseRuntime(job))


def test_invocation_uses_the_durable_job_id_as_dispatch_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A claimed job, not request metadata, owns its budget dispatch identity."""
    dispatchers: list[RecordingDispatcher] = []

    def build_dispatcher(
        *,
        provider: RecordingProvider,
        runtime: DatabaseRuntime,
    ) -> RecordingDispatcher:
        dispatcher = RecordingDispatcher(provider=provider, runtime=runtime)
        dispatchers.append(dispatcher)
        return dispatcher

    monkeypatch.setattr(
        provider_execution,
        "BudgetedProviderDispatcher",
        build_dispatcher,
    )
    provider = RecordingProvider()
    resolver = RecordingResolver(_plan())
    job = _claimed_job()
    invocation = provider_execution.BudgetedJobProviderInvocation(
        provider=provider,
        runtime=_runtime_with_job(job),
        plan_resolver=resolver,
        worker_id="worker-01",
    )

    response = invocation.invoke(job)

    assert response.succeeded
    assert resolver.jobs == [job]
    assert dispatchers[0].arguments is not None
    assert dispatchers[0].arguments["execution_id"] == job.job_id
    assert dispatchers[0].arguments["execution_id"] != "client-correlation-id"
    assert dispatchers[0].arguments["request"] == resolver.plan.request
    assert dispatchers[0].arguments["policies"] == resolver.plan.policies
    assert dispatchers[0].arguments["estimate"] == resolver.plan.estimate


def test_retry_attempt_is_rejected_before_resolving_or_calling_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retries cannot mint a fresh budget identity and reach the provider."""
    monkeypatch.setattr(
        provider_execution,
        "BudgetedProviderDispatcher",
        RecordingDispatcher,
    )
    provider = RecordingProvider()
    resolver = RecordingResolver(_plan())
    job = _claimed_job(attempt=2)
    invocation = provider_execution.BudgetedJobProviderInvocation(
        provider=provider,
        runtime=_runtime_with_job(job),
        plan_resolver=resolver,
        worker_id="worker-01",
    )

    with pytest.raises(ValueError, match="first durable job attempt"):
        invocation.invoke(job)

    assert resolver.jobs == []
    assert provider.requests == []


def test_unpersisted_job_record_is_rejected_before_resolving_or_provider_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller-created job metadata is not proof of a durable worker claim."""
    monkeypatch.setattr(
        provider_execution,
        "BudgetedProviderDispatcher",
        RecordingDispatcher,
    )
    provider = RecordingProvider()
    resolver = RecordingResolver(_plan())
    invocation = provider_execution.BudgetedJobProviderInvocation(
        provider=provider,
        runtime=_runtime_with_job(None),
        plan_resolver=resolver,
        worker_id="worker-01",
    )

    with pytest.raises(ValueError, match="durable job claim"):
        invocation.invoke(_claimed_job())

    assert resolver.jobs == []
    assert provider.requests == []


def test_expired_job_lease_is_rejected_before_resolving_or_provider_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale RUNNING snapshot cannot dispatch after its lease has expired."""
    monkeypatch.setattr(
        provider_execution,
        "BudgetedProviderDispatcher",
        RecordingDispatcher,
    )
    provider = RecordingProvider()
    resolver = RecordingResolver(_plan())
    expired_job = _claimed_job(lease_offset=timedelta(seconds=-1))
    invocation = provider_execution.BudgetedJobProviderInvocation(
        provider=provider,
        runtime=_runtime_with_job(expired_job),
        plan_resolver=resolver,
        worker_id="worker-01",
    )

    with pytest.raises(ValueError, match="durable job claim"):
        invocation.invoke(expired_job)

    assert resolver.jobs == []
    assert provider.requests == []


def test_claim_owned_by_another_worker_is_rejected_before_provider_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid durable claim is usable only by its configured worker."""
    monkeypatch.setattr(
        provider_execution,
        "BudgetedProviderDispatcher",
        RecordingDispatcher,
    )
    provider = RecordingProvider()
    resolver = RecordingResolver(_plan())
    job = _claimed_job()
    invocation = provider_execution.BudgetedJobProviderInvocation(
        provider=provider,
        runtime=_runtime_with_job(job),
        plan_resolver=resolver,
        worker_id="worker-02",
    )

    with pytest.raises(ValueError, match="active durable job claim"):
        invocation.invoke(job)

    assert resolver.jobs == []
    assert provider.requests == []


def test_normalised_provider_failure_is_not_reported_as_job_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider error response must enter the worker failure path."""
    monkeypatch.setattr(
        provider_execution,
        "BudgetedProviderDispatcher",
        RecordingDispatcher,
    )
    provider = RecordingProvider()
    provider.return_failure = True
    job = _claimed_job()
    invocation = provider_execution.BudgetedJobProviderInvocation(
        provider=provider,
        runtime=_runtime_with_job(job),
        plan_resolver=RecordingResolver(_plan()),
        worker_id="worker-01",
    )

    with pytest.raises(RuntimeError, match="unsuccessful provider response"):
        invocation.invoke(job)

    assert len(provider.requests) == 1
