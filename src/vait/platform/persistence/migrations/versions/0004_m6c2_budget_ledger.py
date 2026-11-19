"""Create the durable budget-reservation ledger tables.

Revision ID: 0004_m6c2_budget_ledger
Revises: 0003_m5f_jobs
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004_m6c2_budget_ledger"
down_revision: str | None = "0003_m5f_jobs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create primary and foreign keys for reservation scope ownership."""
    op.create_table(
        "budget_reservations",
        sa.Column("reservation_id", sa.Uuid(), nullable=False),
        sa.Column("request_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("request_payload", postgresql.JSONB(), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("estimated_provider_requests", sa.Integer(), nullable=False),
        sa.Column("estimated_provider_attempts", sa.Integer(), nullable=False),
        sa.Column("estimated_retries", sa.Integer(), nullable=False),
        sa.Column("estimated_provider_tokens", sa.Integer(), nullable=False),
        sa.Column("estimated_spend_usd", sa.Numeric(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("dispatched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("released_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("settled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("consumption_fingerprint", sa.String(length=64), nullable=True),
        sa.Column("consumption_payload", postgresql.JSONB(), nullable=True),
        sa.Column("consumed_provider_requests", sa.Integer(), nullable=True),
        sa.Column("consumed_provider_attempts", sa.Integer(), nullable=True),
        sa.Column("consumed_retries", sa.Integer(), nullable=True),
        sa.Column("consumed_provider_tokens", sa.Integer(), nullable=True),
        sa.Column("consumed_spend_usd", sa.Numeric(), nullable=True),
        sa.CheckConstraint(
            "state IN ('reserved', 'dispatched', 'settled', 'released')",
            name="ck_budget_reservations_state",
        ),
        sa.CheckConstraint(
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
        sa.CheckConstraint(
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
        sa.CheckConstraint(
            "estimated_provider_requests > 0 AND estimated_provider_attempts > 0 "
            "AND estimated_retries >= 0 "
            "AND estimated_provider_attempts = estimated_provider_requests + estimated_retries",
            name="ck_budget_reservations_estimated_attempts",
        ),
        sa.PrimaryKeyConstraint("reservation_id", name="pk_budget_reservations"),
    )
    op.create_table(
        "budget_reservation_scopes",
        sa.Column("reservation_id", sa.Uuid(), nullable=False),
        sa.Column("scope_kind", sa.String(length=16), nullable=False),
        sa.Column("scope_id", sa.String(length=255), nullable=False),
        sa.Column("policy_id", sa.String(length=128), nullable=False),
        sa.Column("policy_version", sa.Integer(), nullable=False),
        sa.Column("window_kind", sa.String(length=16), nullable=False),
        sa.Column("interval_seconds", sa.Integer(), nullable=True),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["reservation_id"],
            ["budget_reservations.reservation_id"],
            name="fk_budget_reservation_scopes_reservation_id_budget_reservations",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "reservation_id",
            "scope_kind",
            "scope_id",
            name="pk_budget_reservation_scopes",
        ),
        sa.CheckConstraint(
            "(window_kind = 'fixed_window' AND interval_seconds > 0 "
            "AND window_start IS NOT NULL) "
            "OR (window_kind IN ('per_execution', 'lifetime') "
            "AND interval_seconds IS NULL AND window_start IS NULL)",
            name="ck_budget_reservation_scopes_window",
        ),
    )
    op.create_index(
        "ix_budget_reservation_scopes_scope_reservation",
        "budget_reservation_scopes",
        ["scope_kind", "scope_id", "window_start", "reservation_id"],
        unique=False,
    )


def downgrade() -> None:
    """Remove the budget ledger tables in dependency order."""
    op.drop_index(
        "ix_budget_reservation_scopes_scope_reservation",
        table_name="budget_reservation_scopes",
    )
    op.drop_table("budget_reservation_scopes")
    op.drop_table("budget_reservations")
