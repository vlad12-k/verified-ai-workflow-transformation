"""Tests for durable, budgeted provider invocation from claimed jobs."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import uuid4

import pytest

import vait.platform.provider_execution as provider_execution
from vait.contracts.budgeting import BudgetEstimate, BudgetPolicy
from vait.inference.providers.base import InferenceProvider
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
from vait.inference.providers.openai_compatible import (
    OpenAICompatibleProvider,
    OpenAICompatibleTransport,
)
from vait.inference.providers.stdlib_http import StdlibJsonTransport
from vait.inference.providers.watsonx import WatsonxProvider
from vait.platform.jobs.models import JobRecord
from vait.platform.persistence.engine import DatabaseRuntime
from vait.platform.provider_dispatch_policy import (
    DispatchSafetyMode,
    ProviderDispatchPolicy,
    ProviderDispatchPolicyError,
    ProviderLocation,
)
from vait.platform.security.egress import ProviderEgressPolicy

_TEST_POLICY = ProviderDispatchPolicy(
    location=ProviderLocation.LOCAL,
    mode=DispatchSafetyMode.LOCAL_ONLY,
)


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

    def scalar(self, statement: object) -> datetime:
        del statement
        return datetime.now(UTC)

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


@pytest.fixture(autouse=True)
def _classify_recording_test_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trust only the existing in-memory fake in this test module."""
    original = provider_execution.classify_builtin_provider

    def classifier(provider: InferenceProvider) -> ProviderLocation:
        if type(provider) is RecordingProvider:
            return ProviderLocation.LOCAL
        return original(provider)

    monkeypatch.setattr(
        provider_execution,
        "classify_builtin_provider",
        classifier,
    )


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
        dispatch_policy=_TEST_POLICY,
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
        dispatch_policy=_TEST_POLICY,
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
        dispatch_policy=_TEST_POLICY,
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
        dispatch_policy=_TEST_POLICY,
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
        dispatch_policy=_TEST_POLICY,
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
        dispatch_policy=_TEST_POLICY,
    )

    with pytest.raises(RuntimeError, match="unsuccessful provider response"):
        invocation.invoke(job)

    assert len(provider.requests) == 1



def test_database_clock_rejects_expired_provider_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Database time, not worker time, decides lease expiry."""
    monkeypatch.setattr(
        provider_execution,
        "BudgetedProviderDispatcher",
        RecordingDispatcher,
    )

    def future_database_clock(
        self: FakeJobSession,
        statement: object,
    ) -> datetime:
        del self, statement
        return datetime.now(UTC) + timedelta(minutes=5)

    monkeypatch.setattr(
        FakeJobSession,
        "scalar",
        future_database_clock,
    )

    provider = RecordingProvider()
    resolver = RecordingResolver(_plan())
    job = _claimed_job(
        lease_offset=timedelta(minutes=1),
    )

    invocation = provider_execution.BudgetedJobProviderInvocation(
        provider=provider,
        runtime=_runtime_with_job(job),
        plan_resolver=resolver,
        worker_id="worker-01",
        dispatch_policy=_TEST_POLICY,
    )

    with pytest.raises(ValueError, match="durable job claim"):
        invocation.invoke(job)

    assert resolver.jobs == []
    assert provider.requests == []



def test_verification_job_cannot_dispatch_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only provider-inference jobs can authorise provider I/O."""
    monkeypatch.setattr(
        provider_execution,
        "BudgetedProviderDispatcher",
        RecordingDispatcher,
    )

    job = _claimed_job().model_copy(
        update={"operation": "verification"}
    )
    provider = RecordingProvider()
    resolver = RecordingResolver(_plan())

    invocation = provider_execution.BudgetedJobProviderInvocation(
        provider=provider,
        runtime=_runtime_with_job(job),
        plan_resolver=resolver,
        worker_id="worker-01",
        dispatch_policy=_TEST_POLICY,
    )

    with pytest.raises(ValueError, match="active durable job claim"):
        invocation.invoke(job)

    assert resolver.jobs == []
    assert provider.requests == []



@pytest.mark.parametrize(
    ("policy", "message"),
    [
        (
            ProviderDispatchPolicy(
                location=ProviderLocation.REMOTE,
                mode=DispatchSafetyMode.REMOTE_RISK_ACCEPTED,
                internal_retries_verified_disabled=True,
            ),
            "explicit risk acceptance",
        ),
        (
            ProviderDispatchPolicy(
                location=ProviderLocation.REMOTE,
                mode=DispatchSafetyMode.STRICT_NO_LATE_IO,
                residual_risk_accepted=True,
                internal_retries_verified_disabled=True,
            ),
            "receiver-side fencing",
        ),
    ],
)
def test_invocation_rejects_unsafe_dispatch_policy(
    policy: ProviderDispatchPolicy,
    message: str,
) -> None:
    """Unsafe policy cannot construct provider invocation."""
    provider = RecordingProvider()
    resolver = RecordingResolver(_plan())
    job = _claimed_job()

    with pytest.raises(ValueError, match=message):
        provider_execution.BudgetedJobProviderInvocation(
            provider=provider,
            runtime=_runtime_with_job(job),
            plan_resolver=resolver,
            worker_id="worker-01",
            dispatch_policy=policy,
        )

    assert provider.requests == []
    assert resolver.jobs == []



def test_local_policy_rejects_real_remote_adapter() -> None:
    """A remote HTTP adapter cannot claim a local-only permission."""
    job = _claimed_job()
    provider = object.__new__(OpenAICompatibleProvider)

    with pytest.raises(
        ProviderDispatchPolicyError,
        match="does not match adapter identity",
    ):
        provider_execution.BudgetedJobProviderInvocation(
            provider=provider,
            runtime=_runtime_with_job(job),
            plan_resolver=RecordingResolver(_plan()),
            worker_id="worker-01",
            dispatch_policy=_TEST_POLICY,
        )


def test_cached_remote_adapter_cannot_claim_local_identity() -> None:
    """A cache wrapper cannot disguise a remote HTTP provider."""
    job = _claimed_job()
    provider = DeterministicCachingProvider(
        provider=object.__new__(OpenAICompatibleProvider),
    )

    with pytest.raises(
        ProviderDispatchPolicyError,
        match="does not match adapter identity",
    ):
        provider_execution.BudgetedJobProviderInvocation(
            provider=provider,
            runtime=_runtime_with_job(job),
            plan_resolver=RecordingResolver(_plan()),
            worker_id="worker-01",
            dispatch_policy=_TEST_POLICY,
        )


def test_unknown_adapter_is_rejected_despite_spoofed_identity() -> None:
    """Matching provider identifiers do not establish adapter trust."""
    class SpoofedProvider(RecordingProvider):
        pass

    job = _claimed_job()

    with pytest.raises(
        ProviderDispatchPolicyError,
        match="Unknown provider adapter",
    ):
        provider_execution.BudgetedJobProviderInvocation(
            provider=SpoofedProvider(),
            runtime=_runtime_with_job(job),
            plan_resolver=RecordingResolver(_plan()),
            worker_id="worker-01",
            dispatch_policy=_TEST_POLICY,
        )



def _known_remote_provider() -> OpenAICompatibleProvider:
    """Build a configured HTTPS adapter without making network calls."""
    return OpenAICompatibleProvider(
        base_url="https://api.provider.test",
        model_id="test-model",
        transport=StdlibJsonTransport(
            url_validator=ProviderEgressPolicy(
                allowed_origins=frozenset(
                    {"https://api.provider.test"}
                ),
            ),
        ),
    )


def _remote_policy() -> ProviderDispatchPolicy:
    return ProviderDispatchPolicy(
        location=ProviderLocation.REMOTE,
        mode=DispatchSafetyMode.REMOTE_RISK_ACCEPTED,
        residual_risk_accepted=True,
        internal_retries_verified_disabled=True,
    )


def test_real_remote_adapter_accepts_explicit_remote_policy() -> None:
    """Verified remote type and explicit remote policy may be composed."""
    job = _claimed_job()

    policy = ProviderDispatchPolicy(
        location=ProviderLocation.REMOTE,
        mode=DispatchSafetyMode.REMOTE_RISK_ACCEPTED,
        residual_risk_accepted=True,
        internal_retries_verified_disabled=True,
    )

    invocation = provider_execution.BudgetedJobProviderInvocation(
        provider=_known_remote_provider(),
        runtime=_runtime_with_job(job),
        plan_resolver=RecordingResolver(_plan()),
        worker_id="worker-01",
        dispatch_policy=policy,
    )

    assert isinstance(
        invocation,
        provider_execution.BudgetedJobProviderInvocation,
    )



def test_remote_invocation_rejects_unknown_http_transport() -> None:
    """An unknown transport cannot be authorised by a policy flag."""
    job = _claimed_job()
    provider = _known_remote_provider()
    provider._transport = cast(OpenAICompatibleTransport, object())

    with pytest.raises(
        ProviderDispatchPolicyError,
        match="known HTTP transport",
    ):
        provider_execution.BudgetedJobProviderInvocation(
            provider=provider,
            runtime=_runtime_with_job(job),
            plan_resolver=RecordingResolver(_plan()),
            worker_id="worker-01",
            dispatch_policy=_remote_policy(),
        )


def test_remote_invocation_rejects_unaudited_watsonx() -> None:
    """Unknown Watsonx transport behavior must fail closed."""
    job = _claimed_job()

    with pytest.raises(
        ProviderDispatchPolicyError,
        match="has not been audited",
    ):
        provider_execution.BudgetedJobProviderInvocation(
            provider=object.__new__(WatsonxProvider),
            runtime=_runtime_with_job(job),
            plan_resolver=RecordingResolver(_plan()),
            worker_id="worker-01",
            dispatch_policy=_remote_policy(),
        )


def test_remote_cache_cannot_hide_unknown_http_transport() -> None:
    """A cache wrapper cannot bypass transport validation."""
    job = _claimed_job()
    provider = _known_remote_provider()
    provider._transport = cast(OpenAICompatibleTransport, object())

    cached = DeterministicCachingProvider(provider=provider)

    with pytest.raises(
        ProviderDispatchPolicyError,
        match="known HTTP transport",
    ):
        provider_execution.BudgetedJobProviderInvocation(
            provider=cached,
            runtime=_runtime_with_job(job),
            plan_resolver=RecordingResolver(_plan()),
            worker_id="worker-01",
            dispatch_policy=_remote_policy(),
        )
