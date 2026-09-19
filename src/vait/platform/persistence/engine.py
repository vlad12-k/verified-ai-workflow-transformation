"""SQLAlchemy engine and session-factory lifecycle."""

from dataclasses import dataclass

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.orm import Session, sessionmaker

from vait.platform.settings import PlatformSettings

POSTGRESQL_PSYCOPG_DRIVER = "postgresql+psycopg"


class DatabaseConfigurationError(ValueError):
    """Raised when database configuration is missing or unsupported."""


@dataclass(slots=True)
class DatabaseRuntime:
    """Owned SQLAlchemy resources for one platform process."""

    engine: Engine
    session_factory: sessionmaker[Session]

    def dispose(self) -> None:
        """Release database connection-pool resources."""
        self.engine.dispose()


def create_database_runtime(
    settings: PlatformSettings,
) -> DatabaseRuntime:
    """Create lazy PostgreSQL engine and session resources."""
    if settings.database_url is None:
        raise DatabaseConfigurationError(
            "VAIT_DATABASE_URL is required for database persistence"
        )

    raw_url = settings.database_url.get_secret_value()

    try:
        url = make_url(raw_url)
    except ArgumentError:
        raise DatabaseConfigurationError(
            "VAIT_DATABASE_URL is not a valid SQLAlchemy URL"
        ) from None

    if url.drivername != POSTGRESQL_PSYCOPG_DRIVER:
        raise DatabaseConfigurationError(
            "VAIT_DATABASE_URL must use postgresql+psycopg"
        )

    engine = create_engine(
        url,
        pool_pre_ping=True,
    )

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
