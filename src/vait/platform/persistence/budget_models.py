"""SQLAlchemy mappings for durable PostgreSQL budget reservations."""

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from vait.platform.persistence.models import Base


class BudgetReservationRow(Base):
    """Persist the stable identity of one provider-budget reservation."""

    __tablename__ = "budget_reservations"
    __table_args__ = (
        CheckConstraint(
            "state IN ('reserved', 'dispatched', 'settled', 'released')",
            name="ck_budget_reservations_state",
        ),
        CheckConstraint(
            "(state = 'reserved' AND dispatched_at IS NULL AND settled_at IS NULL "
            "AND released_at IS NULL) "
            "OR (state = 'dispatched' AND dispatched_at IS NOT NULL "
            "AND settled_at IS NULL AND released_at IS NULL) "
            "OR (state = 'settled' AND dispatched_at IS NOT NULL "
            "AND settled_at IS NOT NULL AND released_at IS NULL) "
            "OR (state = 'released' AND dispatched_at IS NULL "
            "AND settled_at IS NULL AND released_at IS NOT NULL)",
            name="ck_budget_reservations_lifecycle_timestamps",
        ),
        CheckConstraint(
            "(state = 'settled' AND consumption_fingerprint IS NOT NULL "
            "AND consumption_payload IS NOT NULL "
            "AND consumed_provider_requests IS NOT NULL "
            "AND consumed_provider_attempts IS NOT NULL "
            "AND consumed_retries IS NOT NULL "
            "AND consumed_provider_requests >= 0 "
            "AND consumed_provider_attempts >= 0 "
            "AND consumed_retries >= 0 "
            "AND consumed_provider_attempts = consumed_provider_requests + consumed_retries) "
            "OR (state != 'settled' AND consumption_fingerprint IS NULL "
            "AND consumption_payload IS NULL AND consumed_provider_requests IS NULL "
            "AND consumed_provider_attempts IS NULL AND consumed_retries IS NULL "
            "AND consumed_provider_tokens IS NULL AND consumed_spend_usd IS NULL)",
            name="ck_budget_reservations_consumption_state",
        ),
        CheckConstraint(
            "estimated_provider_requests > 0 AND estimated_provider_attempts > 0 "
            "AND estimated_retries >= 0 "
            "AND estimated_provider_attempts = estimated_provider_requests + estimated_retries",
            name="ck_budget_reservations_estimated_attempts",
        ),
    )

    reservation_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
    )
    request_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    request_payload: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    estimated_provider_requests: Mapped[int] = mapped_column(Integer, nullable=False)
    estimated_provider_attempts: Mapped[int] = mapped_column(Integer, nullable=False)
    estimated_retries: Mapped[int] = mapped_column(Integer, nullable=False)
    estimated_provider_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    estimated_spend_usd: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    dispatched_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    released_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    settled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    consumption_fingerprint: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    consumption_payload: Mapped[dict[str, object] | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    consumed_provider_requests: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    consumed_provider_attempts: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    consumed_retries: Mapped[int | None] = mapped_column(Integer, nullable=True)
    consumed_provider_tokens: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    consumed_spend_usd: Mapped[Decimal | None] = mapped_column(
        Numeric,
        nullable=True,
    )


class BudgetReservationScopeRow(Base):
    """Bind a reservation to each independently enforced budget scope."""

    __tablename__ = "budget_reservation_scopes"
    __table_args__ = (
        CheckConstraint(
            "(window_kind = 'fixed_window' AND interval_seconds > 0 "
            "AND window_start IS NOT NULL) "
            "OR (window_kind IN ('per_execution', 'lifetime') "
            "AND interval_seconds IS NULL AND window_start IS NULL)",
            name="ck_budget_reservation_scopes_window",
        ),
        Index(
            "ix_budget_reservation_scopes_scope_reservation",
            "scope_kind",
            "scope_id",
            "window_start",
            "reservation_id",
        ),
    )

    reservation_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey(
            "budget_reservations.reservation_id",
            ondelete="CASCADE",
            name="fk_budget_reservation_scopes_reservation_id_budget_reservations",
        ),
        primary_key=True,
    )
    scope_kind: Mapped[str] = mapped_column(String(16), primary_key=True)
    scope_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    policy_id: Mapped[str] = mapped_column(String(128), nullable=False)
    policy_version: Mapped[int] = mapped_column(Integer, nullable=False)
    window_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    interval_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    window_start: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
