"""Thin orchestration service for durable background jobs."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from vait.platform.jobs.models import JobRecord
from vait.platform.observability import (
    TracingRuntime,
    bind_log_context,
    reset_log_context,
    traced_span,
)
from vait.platform.persistence.engine import DatabaseRuntime
from vait.platform.persistence.repositories.jobs import JobRepository
from vait.platform.persistence.session import transactional_session

_LOGGER = logging.getLogger("vait")


class JobExecutor(Protocol):
    """Execute one claimed job outside the persistence transaction."""

    def execute(self, job: JobRecord) -> None:
        """Execute the durable job or raise on failure."""
        ...


@dataclass(frozen=True)
class WorkerCycleResult:
    """Observable result of one worker polling cycle."""

    claimed: bool
    job: JobRecord | None


class WorkerService:
    """Claim, execute, and persist the result of one durable job."""

    def __init__(
        self,
        *,
        runtime: DatabaseRuntime,
        executor: JobExecutor,
        worker_id: str,
        lease_seconds: int,
        tracing_runtime: TracingRuntime | None = None,
    ) -> None:
        if not worker_id:
            raise ValueError(
                "worker_id must not be empty"
            )

        if lease_seconds < 1:
            raise ValueError(
                "lease_seconds must be positive"
            )

        self._runtime = runtime
        self._executor = executor
        self._worker_id = worker_id
        self._lease_seconds = lease_seconds
        self._tracing_runtime = tracing_runtime

    def run_once(
        self,
        *,
        now: datetime,
    ) -> WorkerCycleResult:
        """Claim and execute at most one durable job."""
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError(
                "now must be timezone-aware"
            )

        with transactional_session(
            self._runtime
        ) as session:
            claimed = JobRepository(
                session
            ).claim_next(
                worker_id=self._worker_id,
                now=now,
                lease_seconds=self._lease_seconds,
            )

        if claimed is None:
            return WorkerCycleResult(
                claimed=False,
                job=None,
            )

        context_token = bind_log_context(
            experiment_id=str(
                claimed.experiment_id
            ),
            job_id=str(
                claimed.job_id
            ),
            worker_id=self._worker_id,
        )

        def execute_claimed() -> WorkerCycleResult:
            _LOGGER.info(
                "job_claimed"
            )

            try:
                self._executor.execute(
                    claimed
                )
            except Exception as exc:
                error_code = type(exc).__name__

                _LOGGER.exception(
                    "job_execution_failed"
                )

                with transactional_session(
                    self._runtime
                ) as session:
                    failed = JobRepository(
                        session
                    ).mark_failed_or_retry(
                        job_id=claimed.job_id,
                        worker_id=self._worker_id,
                        now=now,
                        error_code=error_code,
                    )

                if failed is None:
                    raise RuntimeError(
                        "Claimed job lost worker ownership "
                        "before failure persistence"
                    ) from exc

                if failed.status == "RETRY_PENDING":
                    _LOGGER.warning(
                        "job_retry_pending"
                    )
                elif failed.status == "FAILED":
                    _LOGGER.error(
                        "job_failed"
                    )
                else:
                    raise RuntimeError(
                        "Unexpected durable job state "
                        "after execution failure"
                    ) from exc

                return WorkerCycleResult(
                    claimed=True,
                    job=failed,
                )

            with transactional_session(
                self._runtime
            ) as session:
                completed = JobRepository(
                    session
                ).mark_succeeded(
                    job_id=claimed.job_id,
                    worker_id=self._worker_id,
                    now=now,
                )

            if completed is None:
                raise RuntimeError(
                    "Claimed job lost worker ownership "
                    "before success persistence"
                )

            _LOGGER.info(
                "job_succeeded"
            )

            return WorkerCycleResult(
                claimed=True,
                job=completed,
            )

        try:
            if self._tracing_runtime is None:
                return execute_claimed()

            with traced_span(
                self._tracing_runtime,
                "worker.job",
            ):
                return execute_claimed()
        finally:
            reset_log_context(
                context_token
            )
