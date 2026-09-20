"""Typed contracts for durable background jobs."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    model_validator,
)

JobStatus = Literal[
    "PENDING",
    "RUNNING",
    "RETRY_PENDING",
    "SUCCEEDED",
    "FAILED",
    "CANCELLED",
]


class JobRecord(BaseModel):
    """Durable state for one background platform operation."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    job_id: UUID
    experiment_id: UUID
    operation: str = Field(
        min_length=1,
        max_length=128,
    )
    payload: dict[str, JsonValue] = Field(
        default_factory=dict,
    )
    status: JobStatus = "PENDING"
    attempt: int = Field(
        default=0,
        ge=0,
    )
    max_attempts: int = Field(
        default=3,
        ge=1,
    )
    idempotency_key: str = Field(
        min_length=1,
        max_length=255,
    )
    claimed_by: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
    )
    created_at: AwareDatetime
    started_at: AwareDatetime | None = None
    heartbeat_at: AwareDatetime | None = None
    lease_expires_at: AwareDatetime | None = None
    finished_at: AwareDatetime | None = None
    error_code: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
    )

    @model_validator(mode="after")
    def validate_state_consistency(self) -> "JobRecord":
        """Reject internally inconsistent durable job states."""
        if self.attempt > self.max_attempts:
            raise ValueError(
                "Job attempt cannot exceed max_attempts"
            )

        running_fields = (
            self.claimed_by,
            self.started_at,
            self.heartbeat_at,
            self.lease_expires_at,
        )

        if self.status == "PENDING":
            if any(value is not None for value in running_fields):
                raise ValueError(
                    "PENDING jobs cannot hold an active worker lease"
                )

            if self.finished_at is not None:
                raise ValueError(
                    "PENDING jobs cannot be finished"
                )

        if self.status == "RUNNING":
            if (
                self.claimed_by is None
                or self.started_at is None
                or self.heartbeat_at is None
                or self.lease_expires_at is None
            ):
                raise ValueError(
                    "RUNNING jobs require worker and lease metadata"
                )

            if self.finished_at is not None:
                raise ValueError(
                    "RUNNING jobs cannot be finished"
                )

        if self.status == "RETRY_PENDING":
            if self.attempt >= self.max_attempts:
                raise ValueError(
                    "RETRY_PENDING requires remaining attempts"
                )

            if self.finished_at is not None:
                raise ValueError(
                    "RETRY_PENDING jobs cannot be finished"
                )

        if (
            self.status
            in {
                "SUCCEEDED",
                "FAILED",
                "CANCELLED",
            }
            and self.finished_at is None
        ):
            raise ValueError(
                "Terminal jobs require finished_at"
            )

        if self.status == "FAILED" and self.error_code is None:
            raise ValueError(
                "FAILED jobs require error_code"
            )

        return self


def utc_now() -> datetime:
    """Return an aware UTC timestamp for platform job operations."""
    from datetime import UTC

    return datetime.now(UTC)
