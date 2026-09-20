"""Tests for FastAPI ownership of database lifecycle and readiness."""

from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from vait.platform import PlatformSettings
from vait.platform.api import create_app
from vait.platform.persistence import DatabaseRuntime


def _runtime_for_engine(engine: Engine) -> DatabaseRuntime:
    """Build a test runtime around a supplied SQLAlchemy engine."""
    factory: sessionmaker[Session] = sessionmaker(
        bind=engine,
        class_=Session,
        autoflush=False,
        expire_on_commit=False,
    )

    return DatabaseRuntime(
        engine=engine,
        session_factory=factory,
    )


def _configured_settings() -> PlatformSettings:
    """Return settings that activate database lifecycle management."""
    return PlatformSettings(
        environment="test",
        database_url=SecretStr(
            "postgresql+psycopg://"
            "vait_user:test-password@localhost:5432/vait"
        ),
    )


def test_database_lifecycle_reports_ready_and_disposes_engine() -> None:
    """Configured database resources must be checked and disposed."""
    engine = create_engine("sqlite+pysqlite:///:memory:")
    runtime = _runtime_for_engine(engine)

    disposed = False

    def mark_disposed(disposed_engine: Engine) -> None:
        nonlocal disposed

        assert disposed_engine is engine
        disposed = True

    event.listen(
        engine,
        "engine_disposed",
        mark_disposed,
    )

    def runtime_factory(
        settings: PlatformSettings,
    ) -> DatabaseRuntime:
        assert settings.database_url is not None
        return runtime

    app = create_app(
        _configured_settings(),
        database_runtime_factory=runtime_factory,
    )

    with TestClient(app) as client:
        assert app.state.database_runtime is runtime

        response = client.get("/api/v1/health/ready")

        assert response.status_code == 200
        assert response.json() == {
            "status": "ready",
            "checks": [
                {
                    "name": "database",
                    "ready": True,
                }
            ],
        }

        assert disposed is False

    assert app.state.database_runtime is None
    assert disposed is True


def test_database_failure_makes_readiness_unavailable(
    tmp_path: Path,
) -> None:
    """Database connection failure must produce readiness 503."""
    database_path = (
        tmp_path
        / "missing-directory"
        / "vait.sqlite"
    )

    engine = create_engine(
        f"sqlite+pysqlite:///{database_path}"
    )
    runtime = _runtime_for_engine(engine)

    def runtime_factory(
        settings: PlatformSettings,
    ) -> DatabaseRuntime:
        assert settings.database_url is not None
        return runtime

    app = create_app(
        _configured_settings(),
        database_runtime_factory=runtime_factory,
    )

    with TestClient(app) as client:
        readiness_response = client.get(
            "/api/v1/health/ready"
        )
        liveness_response = client.get(
            "/api/v1/health/live"
        )

    assert readiness_response.status_code == 503
    assert readiness_response.json() == {
        "status": "not_ready",
        "checks": [
            {
                "name": "database",
                "ready": False,
            }
        ],
    }

    assert liveness_response.status_code == 200


def test_database_runtime_is_not_created_without_configuration() -> None:
    """Database resources must remain absent when DB config is absent."""

    def forbidden_factory(
        settings: PlatformSettings,
    ) -> DatabaseRuntime:
        raise AssertionError(
            f"Unexpected database creation for {settings.environment}"
        )

    app = create_app(
        PlatformSettings(
            environment="test",
        ),
        database_runtime_factory=forbidden_factory,
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/health/ready")

        assert app.state.database_runtime is None

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": [],
    }
    assert app.state.database_runtime is None
