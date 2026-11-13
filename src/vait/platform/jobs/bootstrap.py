"""Safe composition and startup for the platform worker."""

import argparse
import os
import sys
from collections.abc import Sequence
from threading import Event

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from vait.platform.jobs.process import run_worker_process
from vait.platform.jobs.worker import JobExecutor, WorkerService
from vait.platform.persistence.engine import (
    DatabaseConfigurationError,
    create_database_runtime,
)
from vait.platform.settings import PlatformSettings


class WorkerRuntimeSettings(BaseSettings):
    """Strict operational parameters for a worker process."""

    model_config = SettingsConfigDict(
        env_prefix="VAIT_WORKER_",
        extra="ignore",
    )

    worker_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
    )
    lease_seconds: int = Field(default=60, ge=1, le=3600)
    poll_interval_seconds: float = Field(
        default=1.0,
        ge=0.01,
        le=60.0,
        allow_inf_nan=False,
    )


def load_worker_settings() -> WorkerRuntimeSettings:
    """Load required worker identity with Pydantic environment validation."""
    return WorkerRuntimeSettings(
        worker_id=os.environ.get("VAIT_WORKER_ID", "")
    )


def check_worker_configuration() -> None:
    """Validate configuration without claiming jobs or connecting to DB."""

    platform_settings = PlatformSettings()
    load_worker_settings()

    runtime = create_database_runtime(platform_settings)
    runtime.dispose()


def run_configured_worker(
    *,
    executor: JobExecutor,
    allowed_operations: tuple[str, ...],
    platform_settings: PlatformSettings | None = None,
    worker_settings: WorkerRuntimeSettings | None = None,
    stop_event: Event | None = None,
) -> None:
    """Run only explicitly allowlisted durable job operations."""

    if not allowed_operations or any(
        not operation
        or operation.strip() != operation
        or len(operation) > 128
        for operation in allowed_operations
    ):
        raise ValueError("Invalid allowed_operations")

    platform = (
        platform_settings
        if platform_settings is not None
        else PlatformSettings()
    )
    worker = (
        worker_settings
        if worker_settings is not None
        else load_worker_settings()
    )

    runtime = create_database_runtime(platform)

    try:
        service = WorkerService(
            runtime=runtime,
            executor=executor,
            worker_id=worker.worker_id,
            lease_seconds=worker.lease_seconds,
            allowed_operations=allowed_operations,
        )
        run_worker_process(
            service=service,
            poll_interval_seconds=worker.poll_interval_seconds,
            stop_event=stop_event,
        )
    finally:
        runtime.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    """Expose safe CLI preflight; refuse unconfigured job execution."""

    parser = argparse.ArgumentParser(
        description="VAIT durable worker bootstrap"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check-config", action="store_true")
    group.add_argument("--run", action="store_true")

    args = parser.parse_args(argv)

    if args.run:
        print(
            "ERROR: No trusted concrete JobExecutor is registered. "
            "Refusing to claim jobs.",
            file=sys.stderr,
        )
        return 2

    try:
        check_worker_configuration()
    except (ValidationError, DatabaseConfigurationError):
        print(
            "ERROR: Invalid worker or platform configuration. "
            "Check VAIT_WORKER_* and VAIT_DATABASE_URL.",
            file=sys.stderr,
        )
        return 2

    print(
        "PASS: Worker configuration and database URL validated. "
        "No database connection or job claim was performed."
    )
    return 0
