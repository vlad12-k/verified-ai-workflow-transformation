"""Unit tests for safe durable provider-job execution."""

from datetime import UTC, datetime, timedelta
from typing import cast
from unittest.mock import Mock
from uuid import uuid4

import pytest

from vait.inference.providers.models import (
    ProviderError,
    ProviderInferenceResponse,
    ProviderTiming,
)
from vait.platform.jobs.executors import BudgetedProviderJobExecutor
from vait.platform.jobs.models import JobRecord
from vait.platform.provider_execution import (
    BudgetedJobProviderInvocation,
    ProviderInvocationFailed,
)


class RecordingEvidenceWriter:
    def __init__(self, *, fail: bool = False) -> None:
        self.saved: list[tuple[JobRecord, ProviderInferenceResponse]] = []
        self.fail = fail

    def persist(
        self,
        *,
        job: JobRecord,
        response: ProviderInferenceResponse,
    ) -> None:
        if self.fail:
            raise OSError("Evidence persistence failed")
        self.saved.append((job, response))


def make_job(
    *,
    operation: str = "provider-inference",
    attempt: int = 1,
    max_attempts: int = 1,
) -> JobRecord:
    now = datetime.now(UTC)
    return JobRecord(
        job_id=uuid4(),
        experiment_id=uuid4(),
        operation=operation,
        payload={},
        status="RUNNING",
        attempt=attempt,
        max_attempts=max_attempts,
        idempotency_key=str(uuid4()),
        claimed_by="test-worker",
        created_at=now,
        started_at=now,
        heartbeat_at=now,
        lease_expires_at=now + timedelta(seconds=60),
    )


def make_response(*, failed: bool = False) -> ProviderInferenceResponse:
    values: dict[str, object] = {
        "request_id": "test-request",
        "provider_id": "fake",
        "runtime_id": "test-runtime",
        "model_id": "test-model",
        "timing": ProviderTiming(total_latency_ms=1.0),
    }

    if failed:
        values["error"] = ProviderError(
            error_type="provider_failure",
            message="Test provider failed",
            retryable=False,
        )
    else:
        values["output_text"] = "Verified output"

    return ProviderInferenceResponse.model_validate(values)


def make_executor(
    response: ProviderInferenceResponse,
    writer: RecordingEvidenceWriter,
) -> tuple[BudgetedProviderJobExecutor, Mock]:
    invocation = Mock(spec=BudgetedJobProviderInvocation)
    invocation.invoke.return_value = response

    executor = BudgetedProviderJobExecutor(
        invocation=cast(BudgetedJobProviderInvocation, invocation),
        evidence_writer=writer,
    )
    return executor, invocation


def test_success_commits_evidence() -> None:
    job = make_job()
    response = make_response()
    writer = RecordingEvidenceWriter()
    executor, invocation = make_executor(response, writer)

    executor.execute(job)

    invocation.invoke.assert_called_once_with(job)
    assert writer.saved == [(job, response)]


def test_unsupported_operation_cannot_invoke_provider() -> None:
    writer = RecordingEvidenceWriter()
    executor, invocation = make_executor(make_response(), writer)

    with pytest.raises(ValueError, match="Unsupported"):
        executor.execute(make_job(operation="unsupported-operation"))

    invocation.invoke.assert_not_called()
    assert writer.saved == []


@pytest.mark.parametrize(
    ("attempt", "max_attempts"),
    [(2, 2), (1, 3)],
)
def test_retryable_job_configuration_fails_closed(
    attempt: int,
    max_attempts: int,
) -> None:
    writer = RecordingEvidenceWriter()
    executor, invocation = make_executor(make_response(), writer)

    with pytest.raises(ValueError, match="exactly one"):
        executor.execute(
            make_job(attempt=attempt, max_attempts=max_attempts)
        )

    invocation.invoke.assert_not_called()
    assert writer.saved == []


def test_normalised_provider_failure_does_not_save_success_evidence() -> None:
    writer = RecordingEvidenceWriter()
    executor, invocation = make_executor(
        make_response(failed=True),
        writer,
    )

    with pytest.raises(ProviderInvocationFailed):
        executor.execute(make_job())

    invocation.invoke.assert_called_once()
    assert writer.saved == []


def test_evidence_persistence_failure_propagates() -> None:
    writer = RecordingEvidenceWriter(fail=True)
    executor, invocation = make_executor(make_response(), writer)

    with pytest.raises(OSError, match="Evidence persistence failed"):
        executor.execute(make_job())

    invocation.invoke.assert_called_once()
    assert writer.saved == []
