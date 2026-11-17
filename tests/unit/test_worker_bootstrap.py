"""Tests for safe worker configuration and composition."""

from datetime import UTC, datetime, timedelta
from typing import cast
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from pydantic import JsonValue, SecretStr, ValidationError

import vait.platform.jobs.bootstrap as bootstrap
from vait.platform.jobs.bootstrap import (
    WorkerRuntimeSettings,
    main,
    run_configured_worker,
)
from vait.platform.jobs.composition import (
    LocalWorkerOperation,
    TrustedLocalExecutorRegistry,
    create_trusted_local_executor_registry,
)
from vait.platform.jobs.models import JobRecord
from vait.platform.settings import PlatformSettings


class FakeExecutor:
    def __init__(self, operation: LocalWorkerOperation) -> None:
        self.operation = operation
        self.seen_job_ids: list[UUID] = []

    def execute(self, job: JobRecord) -> None:
        self.seen_job_ids.append(job.job_id)
        raise RuntimeError("Test executor must not complete a job")


def _composition() -> TrustedLocalExecutorRegistry:
    return create_trusted_local_executor_registry()


def _running_job(
    *,
    operation: str = "verification",
    payload: dict[str, JsonValue] | None = None,
) -> JobRecord:
    now = datetime.now(UTC)
    return JobRecord(
        job_id=uuid4(),
        experiment_id=uuid4(),
        operation=operation,
        payload=payload or {},
        status="RUNNING",
        attempt=1,
        max_attempts=1,
        idempotency_key=str(uuid4()),
        claimed_by="worker-test",
        created_at=now,
        started_at=now,
        heartbeat_at=now,
        lease_expires_at=now + timedelta(minutes=1),
    )


def test_cli_refusal_has_no_false_success_evidence(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unexpected_runtime(*_args: object, **_kwargs: object) -> None:
        pytest.fail("--run must refuse before creating worker runtime")

    monkeypatch.setattr(
        bootstrap,
        "create_database_runtime",
        unexpected_runtime,
    )

    assert main(["--run"]) == 2
    captured = capsys.readouterr()
    assert "Refusing to claim jobs" in captured.err
    assert "PASS" not in captured.out


def test_cli_check_config_validates_without_connecting(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("VAIT_ENVIRONMENT", "development")
    monkeypatch.setenv("VAIT_AUTHENTICATION_MODE", "disabled")
    monkeypatch.setenv("VAIT_AUTH_ROLE", "viewer")
    monkeypatch.delenv("VAIT_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("VAIT_WORKER_ID", "worker-test")
    monkeypatch.setenv("VAIT_WORKER_LEASE_SECONDS", "60")
    monkeypatch.setenv("VAIT_WORKER_POLL_INTERVAL_SECONDS", "1")
    monkeypatch.setenv(
        "VAIT_DATABASE_URL",
        "postgresql+psycopg://test:test@127.0.0.1:1/vait_test",
    )

    assert main(["--check-config"]) == 0
    assert "No database connection" in capsys.readouterr().out


def test_cli_rejects_missing_database_url(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("VAIT_ENVIRONMENT", "development")
    monkeypatch.setenv("VAIT_AUTHENTICATION_MODE", "disabled")
    monkeypatch.setenv("VAIT_AUTH_ROLE", "viewer")
    monkeypatch.delenv("VAIT_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("VAIT_WORKER_ID", "worker-test")
    monkeypatch.delenv("VAIT_DATABASE_URL", raising=False)

    assert main(["--check-config"]) == 2
    assert "Invalid worker" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("worker_id", "invalid worker"),
        ("lease_seconds", 0),
        ("poll_interval_seconds", 0.0),
    ],
)
def test_invalid_worker_settings_fail_closed(
    field: str,
    value: str | int | float,
) -> None:
    values: dict[str, str | int | float] = {
        "worker_id": "worker-test",
        field: value,
    }

    with pytest.raises(ValidationError):
        WorkerRuntimeSettings.model_validate(values)


def test_configured_worker_fails_closed_without_evidence_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_factory = Mock()
    monkeypatch.setenv("VAIT_WORKER_ID", "worker-test")
    monkeypatch.setattr(bootstrap, "create_database_runtime", runtime_factory)

    with pytest.raises(RuntimeError, match="evidence"):
        run_configured_worker(composition=_composition())

    runtime_factory.assert_not_called()



def test_empty_executor_registry_fails_closed() -> None:
    with pytest.raises(ValueError, match="registration"):
        TrustedLocalExecutorRegistry({})


def test_registry_rejects_non_mapping_registration_input() -> None:
    registrations = cast(dict[LocalWorkerOperation, None], [])

    with pytest.raises(TypeError, match="must be a mapping"):
        TrustedLocalExecutorRegistry(registrations)


def test_registry_rejects_non_enum_registration_keys() -> None:
    registrations = cast(
        dict[LocalWorkerOperation, None],
        {"verification": None},
    )

    with pytest.raises(TypeError, match="Registration keys"):
        TrustedLocalExecutorRegistry(registrations)


def test_registry_rejects_internal_mapping_reassignment() -> None:
    registry = _composition()

    attribute_name = "_registrations"
    with pytest.raises((AttributeError, TypeError)):
        setattr(registry, attribute_name, {})

    with pytest.raises(TypeError):
        cast(
            dict[LocalWorkerOperation, None],
            registry._registrations,
        )[LocalWorkerOperation.VERIFICATION] = None


def test_registry_defensively_copies_operation_registration_mapping() -> None:
    registrations = {LocalWorkerOperation.VERIFICATION: None}
    registry = TrustedLocalExecutorRegistry(registrations)

    registrations.clear()

    assert registry.allowed_operations == (LocalWorkerOperation.VERIFICATION,)


def test_registry_routes_typed_operation_without_using_job_payload() -> None:
    registry = _composition()
    job = _running_job(
        payload={
            "operation": "provider-inference",
            "implementation_id": "untrusted-import-path",
        }
    )

    assert registry.route(job) is LocalWorkerOperation.VERIFICATION

    assert registry.allowed_operations == (LocalWorkerOperation.VERIFICATION,)


def test_registry_rejects_unsupported_operation() -> None:
    registry = _composition()

    with pytest.raises(ValueError, match="Unsupported local operation"):
        registry.route(_running_job(operation="provider-inference"))


def test_registry_rejects_unclaimed_job() -> None:
    job = _running_job().model_copy(update={"claimed_by": None})

    with pytest.raises(ValueError, match="claimed RUNNING job"):
        _composition().route(job)


def test_registry_rejects_executor_instances_as_registration_values() -> None:
    registrations = cast(
        dict[LocalWorkerOperation, None],
        {
            LocalWorkerOperation.VERIFICATION: FakeExecutor(
                LocalWorkerOperation.VERIFICATION
            )
        },
    )

    with pytest.raises(TypeError, match="routing-only"):
        TrustedLocalExecutorRegistry(registrations)


def test_generic_executor_cannot_activate_configured_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_factory = Mock()
    monkeypatch.setattr(bootstrap, "create_database_runtime", runtime_factory)

    with pytest.raises(TypeError, match="TrustedLocalExecutorRegistry"):
        run_configured_worker(
            composition=cast(
                TrustedLocalExecutorRegistry,
                FakeExecutor(LocalWorkerOperation.VERIFICATION),
            )
        )

    runtime_factory.assert_not_called()


def test_untyped_composition_is_rejected_before_database_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_runtime(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Invalid composition must fail before database setup")

    monkeypatch.setattr(
        bootstrap,
        "create_database_runtime",
        unexpected_runtime,
    )

    with pytest.raises(TypeError, match="TrustedLocalExecutorRegistry"):
        run_configured_worker(
            composition=cast(TrustedLocalExecutorRegistry, object()),
            platform_settings=PlatformSettings(
                database_url=SecretStr(
                    "postgresql+psycopg://test:test@localhost/vait_test"
                )
            ),
            worker_settings=WorkerRuntimeSettings(worker_id="worker-test"),
        )
