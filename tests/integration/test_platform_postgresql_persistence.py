"""Real PostgreSQL integration tests for M5-C persistence."""

import os

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import text

from vait.platform import PlatformSettings
from vait.platform.api import create_app
from vait.platform.persistence import (
    create_database_runtime,
    transactional_session,
)

_DATABASE_ENV = "VAIT_TEST_DATABASE_URL"
_PROBE_TABLE = "m5c_transaction_probe"


@pytest.fixture
def postgresql_settings() -> PlatformSettings:
    """Load the explicitly configured disposable PostgreSQL test database."""
    database_url = os.getenv(_DATABASE_ENV)

    if database_url is None:
        pytest.skip(
            f"{_DATABASE_ENV} is not configured"
        )

    return PlatformSettings(
        environment="test",
        database_url=SecretStr(database_url),
    )


def test_real_postgresql_connection(
    postgresql_settings: PlatformSettings,
) -> None:
    """Persistence runtime must connect through the psycopg PostgreSQL driver."""
    runtime = create_database_runtime(postgresql_settings)

    try:
        with runtime.engine.connect() as connection:
            assert connection.dialect.name == "postgresql"
            assert connection.scalar(text("SELECT 1")) == 1

            database_name = connection.scalar(
                text("SELECT current_database()")
            )

        assert isinstance(database_name, str)
        assert database_name
    finally:
        runtime.dispose()


def test_real_postgresql_commit_and_rollback(
    postgresql_settings: PlatformSettings,
) -> None:
    """Transaction boundary must commit success and roll back failure."""
    runtime = create_database_runtime(postgresql_settings)

    try:
        with runtime.engine.begin() as connection:
            connection.execute(
                text(
                    f"DROP TABLE IF EXISTS {_PROBE_TABLE}"
                )
            )
            connection.execute(
                text(
                    f"CREATE TABLE {_PROBE_TABLE} ("
                    "id INTEGER PRIMARY KEY, "
                    "value TEXT NOT NULL"
                    ")"
                )
            )

        with transactional_session(runtime) as session:
            session.execute(
                text(
                    f"INSERT INTO {_PROBE_TABLE} "
                    "(id, value) "
                    "VALUES (1, 'committed')"
                )
            )

        with runtime.engine.connect() as connection:
            committed_count = connection.scalar(
                text(
                    f"SELECT COUNT(*) FROM {_PROBE_TABLE}"
                )
            )

        assert committed_count == 1

        with pytest.raises(
            RuntimeError,
            match="simulated PostgreSQL failure",
        ), transactional_session(runtime) as session:
            session.execute(
                text(
                    f"INSERT INTO {_PROBE_TABLE} "
                    "(id, value) "
                    "VALUES (2, 'must-roll-back')"
                )
            )
            raise RuntimeError(
                "simulated PostgreSQL failure"
            )

        with runtime.engine.connect() as connection:
            final_count = connection.scalar(
                text(
                    f"SELECT COUNT(*) FROM {_PROBE_TABLE}"
                )
            )

        assert final_count == 1
    finally:
        with runtime.engine.begin() as connection:
            connection.execute(
                text(
                    f"DROP TABLE IF EXISTS {_PROBE_TABLE}"
                )
            )

        runtime.dispose()


def test_real_postgresql_drives_api_readiness(
    postgresql_settings: PlatformSettings,
) -> None:
    """FastAPI readiness must reflect the real PostgreSQL dependency."""
    app = create_app(postgresql_settings)

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/health/ready"
        )

        runtime = app.state.database_runtime

        assert runtime is not None
        assert runtime.engine.dialect.name == "postgresql"

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

    assert app.state.database_runtime is None
