"""Alembic migration environment for VAIT PostgreSQL persistence."""

from importlib import import_module

from alembic import context

from vait.platform.persistence import (
    Base,
    create_database_runtime,
)
from vait.platform.settings import PlatformSettings

# Register persistence mappings before exposing metadata to Alembic.
import_module("vait.platform.persistence.registry_models")

config = context.config
target_metadata = Base.metadata


def _settings() -> PlatformSettings:
    """Load migration settings from the VAIT environment namespace."""
    settings = PlatformSettings()

    if settings.database_url is None:
        raise RuntimeError(
            "VAIT_DATABASE_URL is required for database migrations"
        )

    return settings


def run_migrations_offline() -> None:
    """Run migrations without creating a live database connection."""
    settings = _settings()

    assert settings.database_url is not None

    context.configure(
        url=settings.database_url.get_secret_value(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={
            "paramstyle": "named",
        },
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations using the validated VAIT database runtime."""
    runtime = create_database_runtime(_settings())

    try:
        with runtime.engine.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                compare_type=True,
            )

            with context.begin_transaction():
                context.run_migrations()
    finally:
        runtime.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
