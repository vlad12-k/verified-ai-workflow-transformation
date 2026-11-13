"""Tests for safe worker configuration and composition."""

from threading import Event
from unittest.mock import Mock

import pytest
from pydantic import SecretStr, ValidationError

import vait.platform.jobs.bootstrap as bootstrap
from vait.platform.jobs.bootstrap import (
    WorkerRuntimeSettings,
    main,
    run_configured_worker,
)
from vait.platform.jobs.models import JobRecord
from vait.platform.jobs.worker import WorkerService
from vait.platform.settings import PlatformSettings


class FakeExecutor:
    def execute(self, job: JobRecord) -> None:
        raise AssertionError("Fake executor must not execute a job")


def test_cli_rejects_run_without_trusted_executor(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--run"]) == 2
    assert "Refusing to claim jobs" in capsys.readouterr().err


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


def test_composition_uses_explicit_executor_and_disposes_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Mock()
    observed: list[float] = []

    monkeypatch.setattr(
        bootstrap,
        "create_database_runtime",
        lambda _settings: runtime,
    )

    def capture_process(
        *,
        service: WorkerService,
        poll_interval_seconds: float,
        stop_event: Event | None,
    ) -> None:
        assert isinstance(service, WorkerService)
        assert stop_event is not None
        observed.append(poll_interval_seconds)

    monkeypatch.setattr(
        bootstrap,
        "run_worker_process",
        capture_process,
    )

    run_configured_worker(
        executor=FakeExecutor(),
        allowed_operations=("verification",),
        platform_settings=PlatformSettings(
            database_url=SecretStr(
                "postgresql+psycopg://test:test@localhost/vait_test"
            )
        ),
        worker_settings=WorkerRuntimeSettings(
            worker_id="worker-test",
            lease_seconds=120,
            poll_interval_seconds=0.5,
        ),
        stop_event=Event(),
    )

    assert observed == [0.5]
    runtime.dispose.assert_called_once()


def test_composition_disposes_database_after_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Mock()

    monkeypatch.setattr(
        bootstrap,
        "create_database_runtime",
        lambda _settings: runtime,
    )

    def fail_process(
        *,
        service: WorkerService,
        poll_interval_seconds: float,
        stop_event: Event | None,
    ) -> None:
        raise RuntimeError("Simulated process failure")

    monkeypatch.setattr(
        bootstrap,
        "run_worker_process",
        fail_process,
    )

    with pytest.raises(RuntimeError, match="Simulated process failure"):
        run_configured_worker(
            executor=FakeExecutor(),
        allowed_operations=("verification",),
            platform_settings=PlatformSettings(
                database_url=SecretStr(
                    "postgresql+psycopg://test:test@localhost/vait_test"
                )
            ),
            worker_settings=WorkerRuntimeSettings(worker_id="worker-test"),
        )

    runtime.dispose.assert_called_once()



def test_empty_operation_allowlist_fails_closed() -> None:
    with pytest.raises(ValueError, match="allowed_operations"):
        run_configured_worker(
            executor=FakeExecutor(),
            allowed_operations=(),
        )
