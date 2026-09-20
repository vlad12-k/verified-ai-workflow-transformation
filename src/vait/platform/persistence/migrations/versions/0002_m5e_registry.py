"""Add the M5-E experiment and evidence registry.

Revision ID: 0002_m5e_registry
Revises: 0001_m5d_baseline
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_m5e_registry"
down_revision: str | None = "0001_m5d_baseline"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create durable experiment/evidence registry tables."""
    op.create_table(
        "experiment_runs",
        sa.Column(
            "experiment_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "source_revision",
            sa.String(length=40),
            nullable=False,
        ),
        sa.Column(
            "dataset_fingerprint",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "dataset_partition",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "transformation_contract_id",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "transformation_contract_version",
            sa.String(length=128),
            nullable=False,
        ),
        sa.Column(
            "benchmark_configuration",
            sa.JSON(),
            nullable=False,
        ),
        sa.Column(
            "reproduction_of_experiment_id",
            sa.Uuid(),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.CheckConstraint(
            "length(source_revision) = 40",
            name="ck_experiment_runs_source_revision_length",
        ),
        sa.CheckConstraint(
            "length(dataset_fingerprint) = 64",
            name="ck_experiment_runs_dataset_fingerprint_length",
        ),
        sa.CheckConstraint(
            "length(dataset_partition) > 0",
            name="ck_experiment_runs_dataset_partition_nonempty",
        ),
        sa.CheckConstraint(
            "length(transformation_contract_id) > 0",
            name="ck_experiment_runs_contract_id_nonempty",
        ),
        sa.CheckConstraint(
            "length(transformation_contract_version) > 0",
            name="ck_experiment_runs_contract_version_nonempty",
        ),
        sa.ForeignKeyConstraint(
            ["reproduction_of_experiment_id"],
            ["experiment_runs.experiment_id"],
            name=(
                "fk_experiment_runs_reproduction_"
                "experiment_runs"
            ),
        ),
        sa.PrimaryKeyConstraint(
            "experiment_id",
            name="pk_experiment_runs",
        ),
    )
    op.create_index(
        "ix_experiment_runs_reproduction_of_experiment_id",
        "experiment_runs",
        ["reproduction_of_experiment_id"],
        unique=False,
    )

    op.create_table(
        "candidates",
        sa.Column(
            "candidate_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "experiment_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "candidate_identity",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "candidate_version",
            sa.String(length=128),
            nullable=False,
        ),
        sa.Column(
            "provider_identity",
            sa.String(length=255),
            nullable=True,
        ),
        sa.Column(
            "model_identity",
            sa.String(length=255),
            nullable=True,
        ),
        sa.Column(
            "runtime_identity",
            sa.String(length=255),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.CheckConstraint(
            "length(candidate_identity) > 0",
            name="ck_candidates_identity_nonempty",
        ),
        sa.CheckConstraint(
            "length(candidate_version) > 0",
            name="ck_candidates_version_nonempty",
        ),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiment_runs.experiment_id"],
            name="fk_candidates_experiment_id_experiment_runs",
        ),
        sa.PrimaryKeyConstraint(
            "candidate_id",
            name="pk_candidates",
        ),
        sa.UniqueConstraint(
            "experiment_id",
            "candidate_identity",
            "candidate_version",
            name="uq_candidates_experiment_identity_version",
        ),
    )
    op.create_index(
        "ix_candidates_experiment_id",
        "candidates",
        ["experiment_id"],
        unique=False,
    )

    op.create_table(
        "evidence_records",
        sa.Column(
            "evidence_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "experiment_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "candidate_id",
            sa.Uuid(),
            nullable=True,
        ),
        sa.Column(
            "kind",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column(
            "payload",
            sa.JSON(),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.CheckConstraint(
            (
                "kind IN ("
                "'verification', "
                "'inference', "
                "'resource', "
                "'economic', "
                "'optimisation', "
                "'recommendation'"
                ")"
            ),
            name="ck_evidence_records_kind",
        ),
        sa.ForeignKeyConstraint(
            ["candidate_id"],
            ["candidates.candidate_id"],
            name="fk_evidence_records_candidate_id_candidates",
        ),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiment_runs.experiment_id"],
            name="fk_evidence_records_experiment_id_experiment_runs",
        ),
        sa.PrimaryKeyConstraint(
            "evidence_id",
            name="pk_evidence_records",
        ),
    )
    op.create_index(
        "ix_evidence_records_candidate_id",
        "evidence_records",
        ["candidate_id"],
        unique=False,
    )
    op.create_index(
        "ix_evidence_records_experiment_id",
        "evidence_records",
        ["experiment_id"],
        unique=False,
    )
    op.create_index(
        "ix_evidence_records_kind",
        "evidence_records",
        ["kind"],
        unique=False,
    )

    op.create_table(
        "artifacts",
        sa.Column(
            "artifact_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "experiment_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "evidence_id",
            sa.Uuid(),
            nullable=True,
        ),
        sa.Column(
            "storage_location",
            sa.String(length=2048),
            nullable=False,
        ),
        sa.Column(
            "sha256",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "size_bytes",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "media_type",
            sa.String(length=255),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.CheckConstraint(
            "length(storage_location) > 0",
            name="ck_artifacts_storage_location_nonempty",
        ),
        sa.CheckConstraint(
            "length(sha256) = 64",
            name="ck_artifacts_sha256_length",
        ),
        sa.CheckConstraint(
            "size_bytes > 0",
            name="ck_artifacts_size_bytes_positive",
        ),
        sa.CheckConstraint(
            "media_type IS NULL OR length(media_type) > 0",
            name="ck_artifacts_media_type_nonempty",
        ),
        sa.ForeignKeyConstraint(
            ["evidence_id"],
            ["evidence_records.evidence_id"],
            name="fk_artifacts_evidence_id_evidence_records",
        ),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiment_runs.experiment_id"],
            name="fk_artifacts_experiment_id_experiment_runs",
        ),
        sa.PrimaryKeyConstraint(
            "artifact_id",
            name="pk_artifacts",
        ),
    )
    op.create_index(
        "ix_artifacts_evidence_id",
        "artifacts",
        ["evidence_id"],
        unique=False,
    )
    op.create_index(
        "ix_artifacts_experiment_id",
        "artifacts",
        ["experiment_id"],
        unique=False,
    )


def downgrade() -> None:
    """Remove the M5-E experiment/evidence registry tables."""
    op.drop_index(
        "ix_artifacts_experiment_id",
        table_name="artifacts",
    )
    op.drop_index(
        "ix_artifacts_evidence_id",
        table_name="artifacts",
    )
    op.drop_table("artifacts")

    op.drop_index(
        "ix_evidence_records_kind",
        table_name="evidence_records",
    )
    op.drop_index(
        "ix_evidence_records_experiment_id",
        table_name="evidence_records",
    )
    op.drop_index(
        "ix_evidence_records_candidate_id",
        table_name="evidence_records",
    )
    op.drop_table("evidence_records")

    op.drop_index(
        "ix_candidates_experiment_id",
        table_name="candidates",
    )
    op.drop_table("candidates")

    op.drop_index(
        "ix_experiment_runs_reproduction_of_experiment_id",
        table_name="experiment_runs",
    )
    op.drop_table("experiment_runs")
