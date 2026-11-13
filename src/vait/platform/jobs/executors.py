"""Fail-closed adapter between durable jobs and budgeted provider execution."""

from typing import Protocol

from vait.inference.providers.models import ProviderInferenceResponse
from vait.platform.jobs.models import JobRecord
from vait.platform.provider_execution import (
    BudgetedJobProviderInvocation,
    ProviderInvocationFailed,
)


class ProviderEvidenceWriter(Protocol):
    """Commit provider evidence durably before job success is recorded."""

    def persist(
        self,
        *,
        job: JobRecord,
        response: ProviderInferenceResponse,
    ) -> None:
        """Persist evidence or raise on failure."""
        ...


class BudgetedProviderJobExecutor:
    """Execute one allowlisted, single-attempt provider job.

    WorkerService may mark the job successful only after this
    executor returns. The evidence writer must durably commit
    evidence before returning.
    """

    OPERATION = "provider-inference"

    def __init__(
        self,
        *,
        invocation: BudgetedJobProviderInvocation,
        evidence_writer: ProviderEvidenceWriter,
    ) -> None:
        self._invocation = invocation
        self._evidence_writer = evidence_writer

    def execute(self, job: JobRecord) -> None:
        """Invoke a budgeted provider and persist its evidence."""

        if job.operation != self.OPERATION:
            raise ValueError("Unsupported provider job operation")

        if job.status != "RUNNING" or job.claimed_by is None:
            raise ValueError("Provider job requires a running claim")

        if job.attempt != 1 or job.max_attempts != 1:
            raise ValueError(
                "Provider jobs require exactly one permitted attempt"
            )

        response = self._invocation.invoke(job)

        if not response.succeeded:
            raise ProviderInvocationFailed(response)

        self._evidence_writer.persist(
            job=job,
            response=response,
        )
