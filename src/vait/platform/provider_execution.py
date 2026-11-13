"""Platform boundary for budgeted inference from durable job context."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from vait.contracts.budgeting import BudgetEstimate, BudgetPolicy
from vait.inference.providers.base import InferenceProvider
from vait.inference.providers.models import (
    ProviderInferenceRequest,
    ProviderInferenceResponse,
)
from vait.platform.budget_guard import BudgetedProviderDispatcher
from vait.platform.jobs.models import JobRecord
from vait.platform.persistence.engine import DatabaseRuntime
from vait.platform.persistence.repositories.jobs import JobRepository
from vait.platform.persistence.session import transactional_session


@dataclass(frozen=True, slots=True)
class ProviderBudgetPlan:
    """Trusted request, policy and estimate selected for one durable job."""

    request: ProviderInferenceRequest
    policies: tuple[BudgetPolicy, ...]
    estimate: BudgetEstimate


class ProviderBudgetPlanResolver(Protocol):
    """Resolve a provider plan from trusted platform configuration and a job."""

    def resolve(self, job: JobRecord) -> ProviderBudgetPlan:
        """Build a request and budget from trusted server-side policy."""
        ...


class ProviderInvocationFailed(RuntimeError):
    """Expose a normalised provider failure to the durable job failure path."""

    def __init__(self, response: ProviderInferenceResponse) -> None:
        self.response = response
        super().__init__("The provider returned an unsuccessful provider response.")


def _require_active_job_claim(
    *,
    runtime: DatabaseRuntime,
    job_id: UUID,
    worker_id: str,
) -> JobRecord:
    """Load and validate the current durable claim for a worker."""
    with transactional_session(runtime) as session:
        durable_job = JobRepository(session).get(job_id)

    if (
        durable_job is None
        or durable_job.status != "RUNNING"
        or durable_job.claimed_by != worker_id
        or durable_job.lease_expires_at is None
        or durable_job.lease_expires_at <= datetime.now(UTC)
    ):
        raise ValueError("Provider invocation requires an active durable job claim.")

    return durable_job


class _ClaimValidatedProvider:
    """Recheck durable claim ownership immediately before provider I/O."""

    def __init__(
        self,
        *,
        provider: InferenceProvider,
        runtime: DatabaseRuntime,
        job_id: UUID,
        worker_id: str,
    ) -> None:
        self._provider = provider
        self._runtime = runtime
        self._job_id = job_id
        self._worker_id = worker_id

    @property
    def provider_id(self) -> str:
        """Return the wrapped provider identity."""
        return self._provider.provider_id

    @property
    def runtime_id(self) -> str:
        """Return the wrapped runtime identity."""
        return self._provider.runtime_id

    def generate(
        self,
        request: ProviderInferenceRequest,
    ) -> ProviderInferenceResponse:
        """Fence provider I/O with a fresh durable ownership check."""
        job = _require_active_job_claim(
            runtime=self._runtime,
            job_id=self._job_id,
            worker_id=self._worker_id,
        )
        if job.attempt != 1:
            raise ValueError("Only the first durable job attempt is supported.")
        return self._provider.generate(request)


class BudgetedJobProviderInvocation:
    """Invoke one provider request using a durable job as its identity.

    The resolver is a trusted platform component. It must not treat caller-
    controlled job payload or inference metadata as budget-policy authority.
    """

    def __init__(
        self,
        *,
        provider: InferenceProvider,
        runtime: DatabaseRuntime,
        plan_resolver: ProviderBudgetPlanResolver,
        worker_id: str,
    ) -> None:
        if not worker_id:
            raise ValueError("worker_id must not be empty")

        self._provider = provider
        self._runtime = runtime
        self._plan_resolver = plan_resolver
        self._worker_id = worker_id

    def invoke(self, job: JobRecord) -> ProviderInferenceResponse:
        """Resolve and dispatch only from this worker's active durable claim."""
        durable_job = _require_active_job_claim(
            runtime=self._runtime,
            job_id=job.job_id,
            worker_id=self._worker_id,
        )

        if durable_job.attempt != 1:
            raise ValueError("Only the first durable job attempt is supported.")

        plan = self._plan_resolver.resolve(durable_job)
        dispatcher = BudgetedProviderDispatcher(
            provider=_ClaimValidatedProvider(
                provider=self._provider,
                runtime=self._runtime,
                job_id=durable_job.job_id,
                worker_id=self._worker_id,
            ),
            runtime=self._runtime,
        )
        response = dispatcher.execute(
            request=plan.request,
            policies=plan.policies,
            estimate=plan.estimate,
            execution_id=durable_job.job_id,
        )

        if not response.succeeded:
            raise ProviderInvocationFailed(response)

        return response
