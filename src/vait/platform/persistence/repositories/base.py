"""Base contract for SQLAlchemy-backed repositories."""

from abc import ABC, abstractmethod

from sqlalchemy.orm import Session


class Repository[EntityT, EntityIdT](ABC):
    """Base class for repositories bound to one transaction session."""

    def __init__(self, session: Session) -> None:
        self._session = session

    @property
    def session(self) -> Session:
        """Return the transaction-scoped SQLAlchemy session."""
        return self._session

    @abstractmethod
    def add(self, entity: EntityT) -> None:
        """Add an entity to the current transaction."""

    @abstractmethod
    def get(self, entity_id: EntityIdT) -> EntityT | None:
        """Return an entity by identifier when it exists."""
