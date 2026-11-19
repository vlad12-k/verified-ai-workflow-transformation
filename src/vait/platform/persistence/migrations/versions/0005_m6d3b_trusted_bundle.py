"""Create immutable trusted verification bundle storage.

Revision ID: 0005_m6d3b_trusted_bundle
Revises: 0004_m6c2_budget_ledger
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_m6d3b_trusted_bundle"
down_revision: str | None = "0004_m6c2_budget_ledger"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add one immutable canonical bundle row per persisted experiment."""
    op.create_table(
        "trusted_verification_bundles",
        sa.Column("experiment_id", sa.Uuid(), nullable=False),
        sa.Column("bundle_version", sa.String(length=32), nullable=False),
        sa.Column("contract_id", sa.String(length=255), nullable=False),
        sa.Column("contract_version", sa.String(length=128), nullable=False),
        sa.Column("source_revision", sa.String(length=40), nullable=False),
        sa.Column("fingerprint_sha256", sa.String(length=64), nullable=False),
        sa.Column("canonical_json", sa.Text(), nullable=False),
        sa.CheckConstraint(
            "length(bundle_version) > 0",
            name="ck_trusted_verification_bundles_bundle_version_nonempty",
        ),
        sa.CheckConstraint(
            "length(contract_id) > 0",
            name="ck_trusted_verification_bundles_contract_id_nonempty",
        ),
        sa.CheckConstraint(
            "length(contract_version) > 0",
            name="ck_trusted_verification_bundles_contract_version_nonempty",
        ),
        sa.CheckConstraint(
            "length(source_revision) = 40",
            name="ck_trusted_verification_bundles_source_revision_length",
        ),
        sa.CheckConstraint(
            "length(fingerprint_sha256) = 64",
            name="ck_trusted_verification_bundles_fingerprint_length",
        ),
        sa.CheckConstraint(
            "length(canonical_json) > 0",
            name="ck_trusted_verification_bundles_canonical_json_nonempty",
        ),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiment_runs.experiment_id"],
            name="fk_trusted_verification_bundles_experiment_id_experiment_runs",
        ),
        sa.PrimaryKeyConstraint(
            "experiment_id",
            name="pk_trusted_verification_bundles",
        ),
    )


def downgrade() -> None:
    """Remove only the bundle table introduced by this revision."""
    op.drop_table("trusted_verification_bundles")
