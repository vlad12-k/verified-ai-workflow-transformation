"""Database integration at the HTTP platform boundary."""

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from vait.platform.api.readiness import (
    ReadinessCheck,
    ReadinessProbe,
)
from vait.platform.persistence import DatabaseRuntime


def create_database_readiness_probe(
    runtime: DatabaseRuntime,
) -> ReadinessProbe:
    """Create a probe that checks database connectivity without leaking errors."""

    def probe() -> tuple[ReadinessCheck, ...]:
        try:
            with runtime.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except SQLAlchemyError:
            return (
                ReadinessCheck(
                    name="database",
                    ready=False,
                ),
            )

        return (
            ReadinessCheck(
                name="database",
                ready=True,
            ),
        )

    return probe
