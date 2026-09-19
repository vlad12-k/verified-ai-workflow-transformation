"""Real PostgreSQL integration tests for M5-D migrations."""

import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text

_DATABASE_ENV = "VAIT_TEST_MIGRATION_DATABASE_URL"
_BASELINE_REVISION = "0001_m5d_baseline"
_ROOT = Path(__file__).resolve().parents[2]
_ALEMBIC_INI = _ROOT / "alembic.ini"


@pytest.fixture
def migration_database_url() -> str:
    """Load the explicitly configured disposable migration database."""
    database_url = os.getenv(_DATABASE_ENV)

    if database_url is None:
        pytest.skip(
            f"{_DATABASE_ENV} is not configured"
        )

    return database_url


def _alembic_config() -> Config:
    """Return repository Alembic configuration."""
    return Config(str(_ALEMBIC_INI))


def _version_table_exists(engine: Engine) -> bool:
    """Return whether Alembic's version table currently exists."""
    with engine.connect() as connection:
        result = connection.scalar(
            text(
                "SELECT "
                "to_regclass('public.alembic_version') "
                "IS NOT NULL"
            )
        )

    return bool(result)


def _current_revision(engine: Engine) -> str | None:
    """Read the current Alembic revision."""
    with engine.connect() as connection:
        value = connection.scalar(
            text(
                "SELECT version_num "
                "FROM alembic_version"
            )
        )

    return str(value) if value is not None else None


def _version_row_count(engine: Engine) -> int:
    """Return the number of current Alembic revision rows."""
    with engine.connect() as connection:
        value = connection.scalar(
            text(
                "SELECT COUNT(*) "
                "FROM alembic_version"
            )
        )

    assert isinstance(value, int)
    return value


def test_postgresql_upgrade_downgrade_upgrade_lifecycle(
    migration_database_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Migration history must work from empty DB through both directions."""
    monkeypatch.setenv(
        "VAIT_DATABASE_URL",
        migration_database_url,
    )

    config = _alembic_config()
    engine = create_engine(migration_database_url)

    try:
        # Make repeated executions deterministic.
        if _version_table_exists(engine):
            command.downgrade(config, "base")

        with engine.begin() as connection:
            connection.execute(
                text(
                    "DROP TABLE IF EXISTS alembic_version"
                )
            )

        assert _version_table_exists(engine) is False

        # Empty database -> current head.
        command.upgrade(config, "head")

        assert _version_table_exists(engine) is True
        assert _current_revision(engine) == _BASELINE_REVISION

        # Current head -> base.
        command.downgrade(config, "base")

        assert _version_table_exists(engine) is True
        assert _version_row_count(engine) == 0

        # Base -> current head again.
        command.upgrade(config, "head")

        assert _current_revision(engine) == _BASELINE_REVISION
    finally:
        engine.dispose()
