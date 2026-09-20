"""SQLAlchemy persistence models for the experiment/evidence registry."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Uuid

from vait.platform.persistence.models import Base


class ExperimentRunRow(Base):
    """Persist reproducibility lineage for one experiment run."""

    __tablename__ = "experiment_runs"
    __table_args__ = (
        CheckConstraint(
            "length(source_revision) = 40",
            name="ck_experiment_runs_source_revision_length",
        ),
        CheckConstraint(
            "length(dataset_fingerprint) = 64",
            name="ck_experiment_runs_dataset_fingerprint_length",
        ),
        CheckConstraint(
            "length(dataset_partition) > 0",
            name="ck_experiment_runs_dataset_partition_nonempty",
        ),
        CheckConstraint(
            "length(transformation_contract_id) > 0",
            name="ck_experiment_runs_contract_id_nonempty",
        ),
        CheckConstraint(
            "length(transformation_contract_version) > 0",
            name="ck_experiment_runs_contract_version_nonempty",
        ),
    )

    experiment_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
    )
    source_revision: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
    )
    dataset_fingerprint: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    dataset_partition: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    transformation_contract_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    transformation_contract_version: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )
    benchmark_configuration: Mapped[dict[str, object]] = mapped_column(
        JSON,
        nullable=False,
    )
    reproduction_of_experiment_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey(
            "experiment_runs.experiment_id",
            name=(
                "fk_experiment_runs_reproduction_"
                "experiment_runs"
            ),
        ),
        nullable=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )


class CandidateRow(Base):
    """Persist one candidate or search point within an experiment."""

    __tablename__ = "candidates"
    __table_args__ = (
        UniqueConstraint(
            "experiment_id",
            "candidate_identity",
            "candidate_version",
            name="uq_candidates_experiment_identity_version",
        ),
        CheckConstraint(
            "length(candidate_identity) > 0",
            name="ck_candidates_identity_nonempty",
        ),
        CheckConstraint(
            "length(candidate_version) > 0",
            name="ck_candidates_version_nonempty",
        ),
    )

    candidate_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
    )
    experiment_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey(
            "experiment_runs.experiment_id",
            name="fk_candidates_experiment_id_experiment_runs",
        ),
        nullable=False,
        index=True,
    )
    candidate_identity: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    candidate_version: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )
    provider_identity: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    model_identity: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    runtime_identity: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )


class EvidenceRow(Base):
    """Persist one typed evidence payload and its lineage."""

    __tablename__ = "evidence_records"
    __table_args__ = (
        CheckConstraint(
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
    )

    evidence_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
    )
    experiment_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey(
            "experiment_runs.experiment_id",
            name="fk_evidence_records_experiment_id_experiment_runs",
        ),
        nullable=False,
        index=True,
    )
    candidate_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey(
            "candidates.candidate_id",
            name="fk_evidence_records_candidate_id_candidates",
        ),
        nullable=True,
        index=True,
    )
    kind: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
    )
    payload: Mapped[dict[str, object]] = mapped_column(
        JSON,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )


class ArtifactRow(Base):
    """Persist artifact metadata while keeping bytes outside PostgreSQL."""

    __tablename__ = "artifacts"
    __table_args__ = (
        CheckConstraint(
            "length(storage_location) > 0",
            name="ck_artifacts_storage_location_nonempty",
        ),
        CheckConstraint(
            "length(sha256) = 64",
            name="ck_artifacts_sha256_length",
        ),
        CheckConstraint(
            "size_bytes > 0",
            name="ck_artifacts_size_bytes_positive",
        ),
        CheckConstraint(
            "media_type IS NULL OR length(media_type) > 0",
            name="ck_artifacts_media_type_nonempty",
        ),
    )

    artifact_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
    )
    experiment_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey(
            "experiment_runs.experiment_id",
            name="fk_artifacts_experiment_id_experiment_runs",
        ),
        nullable=False,
        index=True,
    )
    evidence_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey(
            "evidence_records.evidence_id",
            name="fk_artifacts_evidence_id_evidence_records",
        ),
        nullable=True,
        index=True,
    )
    storage_location: Mapped[str] = mapped_column(
        String(2048),
        nullable=False,
    )
    sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    size_bytes: Mapped[int] = mapped_column(
        nullable=False,
    )
    media_type: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
