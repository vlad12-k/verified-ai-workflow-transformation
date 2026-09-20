"""Transactional SQLAlchemy session boundary."""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy.orm import Session

from vait.platform.persistence.engine import DatabaseRuntime


@contextmanager
def transactional_session(
    runtime: DatabaseRuntime,
) -> Iterator[Session]:
    """Commit successful work and roll back failed work."""
    session = runtime.session_factory()

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
