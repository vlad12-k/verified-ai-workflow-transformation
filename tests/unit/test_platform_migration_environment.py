"""Direct tests for the executable Alembic migration environment."""

from contextlib import AbstractContextManager, nullcontext
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest
from alembic import context as alembic_context

import vait.platform.persistence as persistence_package
from vait.platform.settings import PlatformSettings

_ROOT = Path(__file__).resolve().parents[2]

_ENV_PATH = (
    _ROOT
    / "src"
    / "vait"
    / "platform"
    / "persistence"
    / "migrations"
    / "env.py"
)

_DATABASE_URL = (
    "postgresql+psycopg://"
    "vait:vait-test-password@localhost:5432/vait_test"
)


class _ContextRecorder:
    """Record Alembic environment interactions."""

    def __init__(
        self,
        *,
        offline: bool,
        migration_error: RuntimeError | None = None,
    ) -> None:
        self.offline = offline
        self.migration_error = migration_error
        self.configure_calls: list[
            dict[str, object]
        ] = []
        self.run_count = 0

    def is_offline_mode(self) -> bool:
        """Return the configured Alembic execution mode."""
        return self.offline

    def configure(
        self,
        **kwargs: object,
    ) -> None:
        """Record one Alembic context configuration."""
        self.configure_calls.append(
            kwargs
        )

    def begin_transaction(
        self,
    ) -> AbstractContextManager[None]:
        """Provide a deterministic transaction context."""
        return nullcontext()

    def run_migrations(self) -> None:
        """Record migration execution and optionally fail."""
        self.run_count += 1

        if self.migration_error is not None:
            raise self.migration_error


class _FakeEngine:
    """Minimal engine boundary required by env.py."""

    def __init__(self) -> None:
        self.connection = object()

    def connect(
        self,
    ) -> AbstractContextManager[object]:
        """Return one fake database connection."""
        return nullcontext(
            self.connection
        )


class _FakeRuntime:
    """Minimal disposable database runtime."""

    def __init__(self) -> None:
        self.engine = _FakeEngine()
        self.disposed = False

    def dispose(self) -> None:
        """Record runtime disposal."""
        self.disposed = True


def _install_context(
    monkeypatch: pytest.MonkeyPatch,
    recorder: _ContextRecorder,
) -> None:
    """Replace Alembic proxy functions with deterministic test doubles."""
    monkeypatch.setitem(
        alembic_context.__dict__,
        "config",
        object(),
    )
    monkeypatch.setitem(
        alembic_context.__dict__,
        "is_offline_mode",
        recorder.is_offline_mode,
    )
    monkeypatch.setitem(
        alembic_context.__dict__,
        "configure",
        recorder.configure,
    )
    monkeypatch.setitem(
        alembic_context.__dict__,
        "begin_transaction",
        recorder.begin_transaction,
    )
    monkeypatch.setitem(
        alembic_context.__dict__,
        "run_migrations",
        recorder.run_migrations,
    )


def _load_environment_module(
    suffix: str,
) -> ModuleType:
    """Execute env.py under a unique module identity."""
    module_name = (
        "_vait_test_migration_environment_"
        + suffix
    )

    spec = spec_from_file_location(
        module_name,
        _ENV_PATH,
    )

    if spec is None or spec.loader is None:
        raise AssertionError(
            f"Unable to load migration environment: {_ENV_PATH}"
        )

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def test_environment_fails_closed_without_database_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Migrations must fail before execution when DB configuration is absent."""
    monkeypatch.delenv(
        "VAIT_DATABASE_URL",
        raising=False,
    )

    recorder = _ContextRecorder(
        offline=True,
    )

    _install_context(
        monkeypatch,
        recorder,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "VAIT_DATABASE_URL is required "
            "for database migrations"
        ),
    ):
        _load_environment_module(
            "missing_database_url"
        )

    assert recorder.configure_calls == []
    assert recorder.run_count == 0


def test_offline_environment_uses_validated_database_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Offline migrations must configure Alembic from validated settings."""
    monkeypatch.setenv(
        "VAIT_DATABASE_URL",
        _DATABASE_URL,
    )

    recorder = _ContextRecorder(
        offline=True,
    )

    _install_context(
        monkeypatch,
        recorder,
    )

    module = _load_environment_module(
        "offline"
    )

    assert recorder.run_count == 1
    assert len(
        recorder.configure_calls
    ) == 1

    configured = (
        recorder.configure_calls[0]
    )

    assert configured["url"] == _DATABASE_URL
    assert (
        configured["target_metadata"]
        is module.target_metadata
    )
    assert configured["literal_binds"] is True
    assert configured["dialect_opts"] == {
        "paramstyle": "named",
    }
    assert configured["compare_type"] is True
    assert "connection" not in configured


def test_online_environment_uses_runtime_connection_and_disposes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Online migrations must use and dispose the validated DB runtime."""
    monkeypatch.setenv(
        "VAIT_DATABASE_URL",
        _DATABASE_URL,
    )

    recorder = _ContextRecorder(
        offline=False,
    )

    _install_context(
        monkeypatch,
        recorder,
    )

    runtime = _FakeRuntime()
    captured_settings: list[
        PlatformSettings
    ] = []

    def create_database_runtime(
        settings: PlatformSettings,
    ) -> _FakeRuntime:
        captured_settings.append(
            settings
        )
        return runtime

    monkeypatch.setitem(
        persistence_package.__dict__,
        "create_database_runtime",
        create_database_runtime,
    )

    module = _load_environment_module(
        "online"
    )

    assert len(captured_settings) == 1

    database_url = (
        captured_settings[0].database_url
    )

    assert database_url is not None
    assert (
        database_url.get_secret_value()
        == _DATABASE_URL
    )

    assert recorder.run_count == 1
    assert len(
        recorder.configure_calls
    ) == 1

    configured = (
        recorder.configure_calls[0]
    )

    assert (
        configured["connection"]
        is runtime.engine.connection
    )
    assert (
        configured["target_metadata"]
        is module.target_metadata
    )
    assert configured["compare_type"] is True
    assert "url" not in configured

    assert runtime.disposed is True


def test_online_environment_disposes_runtime_after_migration_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Migration failure must not leak the online database runtime."""
    monkeypatch.setenv(
        "VAIT_DATABASE_URL",
        _DATABASE_URL,
    )

    migration_error = RuntimeError(
        "simulated migration failure"
    )

    recorder = _ContextRecorder(
        offline=False,
        migration_error=migration_error,
    )

    _install_context(
        monkeypatch,
        recorder,
    )

    runtime = _FakeRuntime()

    def create_database_runtime(
        settings: PlatformSettings,
    ) -> _FakeRuntime:
        del settings
        return runtime

    monkeypatch.setitem(
        persistence_package.__dict__,
        "create_database_runtime",
        create_database_runtime,
    )

    with pytest.raises(
        RuntimeError,
        match="simulated migration failure",
    ):
        _load_environment_module(
            "online_failure"
        )

    assert recorder.run_count == 1
    assert runtime.disposed is True
