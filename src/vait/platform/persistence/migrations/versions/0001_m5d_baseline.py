"""Establish the M5-D migration baseline.

Revision ID: 0001_m5d_baseline
Revises:
"""

from collections.abc import Sequence

revision: str = "0001_m5d_baseline"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Establish the versioned migration baseline."""


def downgrade() -> None:
    """Remove the versioned migration baseline."""
