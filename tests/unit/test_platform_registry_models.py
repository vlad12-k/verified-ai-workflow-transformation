"""Tests for M5-E registry persistence schema."""

from typing import cast

from sqlalchemy import Table, UniqueConstraint

from vait.platform.persistence.registry_models import (
    ArtifactRow,
    CandidateRow,
    EvidenceRow,
    ExperimentRunRow,
)


def test_registry_models_register_expected_tables() -> None:
    """Registry metadata must contain the four durable M5-E tables."""
    tables = {
        cast(Table, ExperimentRunRow.__table__).name,
        cast(Table, CandidateRow.__table__).name,
        cast(Table, EvidenceRow.__table__).name,
        cast(Table, ArtifactRow.__table__).name,
    }

    assert tables == {
        "experiment_runs",
        "candidates",
        "evidence_records",
        "artifacts",
    }


def test_registry_foreign_keys_preserve_lineage() -> None:
    """Registry tables must retain experiment/candidate/evidence lineage."""
    candidate_targets = {
        foreign_key.target_fullname
        for foreign_key in CandidateRow.__table__.foreign_keys
    }
    evidence_targets = {
        foreign_key.target_fullname
        for foreign_key in EvidenceRow.__table__.foreign_keys
    }
    artifact_targets = {
        foreign_key.target_fullname
        for foreign_key in ArtifactRow.__table__.foreign_keys
    }

    assert candidate_targets == {
        "experiment_runs.experiment_id",
    }
    assert evidence_targets == {
        "experiment_runs.experiment_id",
        "candidates.candidate_id",
    }
    assert artifact_targets == {
        "experiment_runs.experiment_id",
        "evidence_records.evidence_id",
    }


def test_candidate_identity_is_unique_within_experiment() -> None:
    """One experiment cannot register the same candidate version twice."""
    candidate_table = cast(
        Table,
        CandidateRow.__table__,
    )

    unique_sets = {
        tuple(constraint.columns.keys())
        for constraint in candidate_table.constraints
        if isinstance(constraint, UniqueConstraint)
    }

    assert (
        "experiment_id",
        "candidate_identity",
        "candidate_version",
    ) in unique_sets


def test_registry_identifiers_fit_postgresql_limit() -> None:
    """Schema identifiers must fit PostgreSQL's 63-byte identifier limit."""
    tables = (
        cast(Table, ExperimentRunRow.__table__),
        cast(Table, CandidateRow.__table__),
        cast(Table, EvidenceRow.__table__),
        cast(Table, ArtifactRow.__table__),
    )

    for table in tables:
        assert len(table.name.encode("utf-8")) <= 63

        for constraint in table.constraints:
            assert constraint.name is not None
            assert len(str(constraint.name).encode("utf-8")) <= 63

        for index in table.indexes:
            assert index.name is not None
            assert len(index.name.encode("utf-8")) <= 63


def test_artifact_table_stores_metadata_not_binary_content() -> None:
    """Artifact bytes must remain behind the ArtifactStore boundary."""
    columns = set(ArtifactRow.__table__.columns.keys())

    assert columns == {
        "artifact_id",
        "experiment_id",
        "evidence_id",
        "storage_location",
        "sha256",
        "size_bytes",
        "media_type",
        "created_at",
    }
    assert "content" not in columns
    assert "data" not in columns
    assert "bytes" not in columns
