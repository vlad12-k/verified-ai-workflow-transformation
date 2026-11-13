"""PostgreSQL-backed, idempotent provider inference evidence."""

from datetime import UTC, datetime
from hashlib import sha256
from uuid import NAMESPACE_URL, uuid5

from pydantic import JsonValue
from sqlalchemy import func, select, text

from vait.inference.providers.models import ProviderInferenceResponse
from vait.platform.jobs.models import JobRecord
from vait.platform.persistence.engine import DatabaseRuntime
from vait.platform.persistence.job_models import JobRow
from vait.platform.persistence.repositories.registry import EvidenceRepository
from vait.platform.persistence.session import transactional_session
from vait.platform.registry.contracts import EvidenceRecord


def _payload(
    job: JobRecord,
    response: ProviderInferenceResponse,
) -> dict[str, JsonValue]:
    """Retain safe, verifiable metadata rather than provider output text."""
    output = (
        response.output_text.encode("utf-8")
        if response.output_text is not None
        else None
    )
    usage = response.usage

    return {
        "schema_version": 1,
        "job_id": str(job.job_id),
        "attempt": job.attempt,
        "request_sha256": sha256(
            response.request_id.encode("utf-8")
        ).hexdigest(),
        "provider_id": response.provider_id,
        "runtime_id": response.runtime_id,
        "model_id": response.model_id,
        "model_revision": response.model_revision,
        "output_sha256": (
            sha256(output).hexdigest() if output is not None else None
        ),
        "output_size_bytes": len(output) if output is not None else None,
        "input_tokens": usage.input_tokens if usage is not None else None,
        "output_tokens": usage.output_tokens if usage is not None else None,
        "latency_ms": response.timing.total_latency_ms,
        "cost_usd": response.cost_usd,
    }


class PostgreSQLProviderEvidenceWriter:
    """Persist one inference record under a locked, live first-attempt claim."""

    def __init__(
        self,
        *,
        runtime: DatabaseRuntime,
        worker_id: str,
    ) -> None:
        if not worker_id or worker_id.strip() != worker_id:
            raise ValueError("worker_id must be nonempty and normalised")
        self._runtime = runtime
        self._worker_id = worker_id

    def persist(
        self,
        *,
        job: JobRecord,
        response: ProviderInferenceResponse,
    ) -> None:
        """Atomically check claim ownership and persist idempotent evidence."""
        if (
            job.operation != "provider-inference"
            or job.status != "RUNNING"
            or job.claimed_by != self._worker_id
            or job.attempt != 1
            or job.max_attempts != 1
        ):
            raise ValueError("Job is not eligible for provider evidence")

        if not response.succeeded:
            raise ValueError("Failed provider response is not success evidence")

        evidence = EvidenceRecord(
            evidence_id=uuid5(
                NAMESPACE_URL,
                f"vait:provider-inference:{job.job_id}",
            ),
            experiment_id=job.experiment_id,
            kind="inference",
            payload=_payload(job, response),
            created_at=datetime.now(UTC),
        )

        with transactional_session(self._runtime) as session:
            session.execute(text("SET LOCAL lock_timeout = '3000ms'"))

            active_claim = session.scalar(
                select(JobRow.job_id)
                .where(
                    JobRow.job_id == job.job_id,
                    JobRow.experiment_id == job.experiment_id,
                    JobRow.operation == "provider-inference",
                    JobRow.status == "RUNNING",
                    JobRow.claimed_by == self._worker_id,
                    JobRow.attempt == 1,
                    JobRow.max_attempts == 1,
                    JobRow.lease_expires_at > func.clock_timestamp(),
                )
                .with_for_update()
            )

            if active_claim is None:
                raise RuntimeError(
                    "Provider evidence requires an active first-attempt claim"
                )

            repository = EvidenceRepository(session)
            existing = repository.get(evidence.evidence_id)

            if existing is not None:
                if (
                    existing.experiment_id != evidence.experiment_id
                    or existing.kind != evidence.kind
                    or existing.candidate_id != evidence.candidate_id
                    or existing.payload != evidence.payload
                ):
                    raise RuntimeError("Conflicting provider evidence for job")
                return

            repository.add(evidence)
