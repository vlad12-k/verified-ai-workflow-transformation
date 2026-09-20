"""Tests for the M5-C persistence foundation."""

import pytest
from pydantic import SecretStr
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from vait.platform import PlatformSettings
from vait.platform.persistence import (
    DatabaseConfigurationError,
    DatabaseRuntime,
    create_database_runtime,
    transactional_session,
)


def test_database_runtime_requires_database_url() -> None:
    """Persistence must fail closed without explicit database config."""
    settings = PlatformSettings(
        environment="test",
    )

    with pytest.raises(
        DatabaseConfigurationError,
        match="VAIT_DATABASE_URL is required",
    ):
        create_database_runtime(settings)


def test_database_runtime_rejects_non_postgresql_driver() -> None:
    """Platform persistence must not silently switch database backends."""
    settings = PlatformSettings(
        environment="test",
        database_url=SecretStr("sqlite+pysqlite:///:memory:"),
    )

    with pytest.raises(
        DatabaseConfigurationError,
        match=r"postgresql\+psycopg",
    ):
        create_database_runtime(settings)


def test_database_runtime_rejects_malformed_url() -> None:
    """Malformed configuration must fail without exposing its value."""
    settings = PlatformSettings(
        environment="test",
        database_url=SecretStr("not a valid database url"),
    )

    with pytest.raises(
        DatabaseConfigurationError,
        match="not a valid SQLAlchemy URL",
    ):
        create_database_runtime(settings)


def test_database_runtime_builds_lazy_postgresql_resources() -> None:
    """Creating persistence resources must not open a DB connection."""
    password = "database-secret-password"

    settings = PlatformSettings(
        environment="test",
        database_url=SecretStr(
            "postgresql+psycopg://"
            f"vait_user:{password}@localhost:5432/vait"
        ),
    )

    runtime = create_database_runtime(settings)

    try:
        assert runtime.engine.url.drivername == "postgresql+psycopg"
        assert runtime.engine.url.database == "vait"
        assert runtime.engine.url.username == "vait_user"

        assert password not in str(runtime.engine.url)
        assert password not in repr(runtime.engine.url)

        session = runtime.session_factory()

        try:
            assert session.autoflush is False
            assert session.expire_on_commit is False
        finally:
            session.close()
    finally:
        runtime.dispose()


def _sqlite_test_runtime() -> DatabaseRuntime:
    """Build an isolated SQLAlchemy runtime for transaction unit tests."""
    engine = create_engine("sqlite+pysqlite:///:memory:")

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


def test_transactional_session_commits_successful_work() -> None:
    """Successful unit-of-work execution must commit."""
    runtime = _sqlite_test_runtime()

    try:
        with runtime.engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE TABLE records "
                    "(id INTEGER PRIMARY KEY, value TEXT NOT NULL)"
                )
            )

        with transactional_session(runtime) as session:
            session.execute(
                text(
                    "INSERT INTO records (id, value) "
                    "VALUES (1, 'committed')"
                )
            )

        with runtime.engine.connect() as connection:
            count = connection.scalar(
                text("SELECT COUNT(*) FROM records")
            )

        assert count == 1
    finally:
        runtime.dispose()


def test_transactional_session_rolls_back_failed_work() -> None:
    """Failed unit-of-work execution must not persist partial work."""
    runtime = _sqlite_test_runtime()

    try:
        with runtime.engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE TABLE records "
                    "(id INTEGER PRIMARY KEY, value TEXT NOT NULL)"
                )
            )

        with (
            pytest.raises(RuntimeError, match="simulated failure"),
            transactional_session(runtime) as session,
        ):
            session.execute(
                text(
                    "INSERT INTO records (id, value) "
                    "VALUES (1, 'must-roll-back')"
                )
            )
            raise RuntimeError("simulated failure")

        with runtime.engine.connect() as connection:
            count = connection.scalar(
                text("SELECT COUNT(*) FROM records")
            )

        assert count == 0
    finally:
        runtime.dispose()
