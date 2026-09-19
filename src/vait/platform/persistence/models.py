"""Shared SQLAlchemy declarative model base."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base for VAIT platform persistence models."""
