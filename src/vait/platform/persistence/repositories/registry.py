"""SQLAlchemy repositories for durable experiment/evidence lineage."""

from uuid import UUID

from sqlalchemy import select

from vait.platform.persistence.registry_models import (
    ArtifactRow,
    CandidateRow,
    EvidenceRow,
    ExperimentRunRow,
)
from vait.platform.persistence.repositories.base import Repository
from vait.platform.registry.contracts import (
    ArtifactRecord,
    CandidateRecord,
    EvidenceRecord,
    ExperimentRunRecord,
)


def _experiment_record(row: ExperimentRunRow) -> ExperimentRunRecord:
    """Validate a persisted experiment row back into its public contract."""
    return ExperimentRunRecord.model_validate(
        {
            "experiment_id": row.experiment_id,
            "source_revision": row.source_revision,
            "dataset_fingerprint": row.dataset_fingerprint,
            "dataset_partition": row.dataset_partition,
            "transformation_contract_id": row.transformation_contract_id,
            "transformation_contract_version": (
                row.transformation_contract_version
            ),
            "benchmark_configuration": row.benchmark_configuration,
            "reproduction_of_experiment_id": (
                row.reproduction_of_experiment_id
            ),
            "created_at": row.created_at,
        }
    )


def _candidate_record(row: CandidateRow) -> CandidateRecord:
    """Validate a persisted candidate row back into its public contract."""
    return CandidateRecord.model_validate(
        {
            "candidate_id": row.candidate_id,
            "experiment_id": row.experiment_id,
            "candidate_identity": row.candidate_identity,
            "candidate_version": row.candidate_version,
            "provider_identity": row.provider_identity,
            "model_identity": row.model_identity,
            "runtime_identity": row.runtime_identity,
            "created_at": row.created_at,
        }
    )


def _evidence_record(row: EvidenceRow) -> EvidenceRecord:
    """Validate a persisted evidence row back into its public contract."""
    return EvidenceRecord.model_validate(
        {
            "evidence_id": row.evidence_id,
            "experiment_id": row.experiment_id,
            "candidate_id": row.candidate_id,
            "kind": row.kind,
            "payload": row.payload,
            "created_at": row.created_at,
        }
    )


def _artifact_record(row: ArtifactRow) -> ArtifactRecord:
    """Validate persisted artifact metadata back into its public contract."""
    return ArtifactRecord.model_validate(
        {
            "artifact_id": row.artifact_id,
            "experiment_id": row.experiment_id,
            "evidence_id": row.evidence_id,
            "storage_location": row.storage_location,
            "sha256": row.sha256,
            "size_bytes": row.size_bytes,
            "media_type": row.media_type,
            "created_at": row.created_at,
        }
    )


class ExperimentRunRepository(
    Repository[ExperimentRunRecord, UUID],
):
    """Persist and retrieve experiment reproducibility lineage."""

    def add(
        self,
        entity: ExperimentRunRecord,
    ) -> None:
        """Add an experiment to the current transaction."""
        self.session.add(
            ExperimentRunRow(
                experiment_id=entity.experiment_id,
                source_revision=entity.source_revision,
                dataset_fingerprint=entity.dataset_fingerprint,
                dataset_partition=entity.dataset_partition,
                transformation_contract_id=(
                    entity.transformation_contract_id
                ),
                transformation_contract_version=(
                    entity.transformation_contract_version
                ),
                benchmark_configuration=dict(
                    entity.benchmark_configuration
                ),
                reproduction_of_experiment_id=(
                    entity.reproduction_of_experiment_id
                ),
                created_at=entity.created_at,
            )
        )
        self.session.flush()

    def get(
        self,
        entity_id: UUID,
    ) -> ExperimentRunRecord | None:
        """Return one experiment by identifier."""
        row = self.session.get(
            ExperimentRunRow,
            entity_id,
        )

        if row is None:
            return None

        return _experiment_record(row)


class CandidateRepository(
    Repository[CandidateRecord, UUID],
):
    """Persist and retrieve experiment candidates."""

    def add(
        self,
        entity: CandidateRecord,
    ) -> None:
        """Add a candidate to the current transaction."""
        self.session.add(
            CandidateRow(
                candidate_id=entity.candidate_id,
                experiment_id=entity.experiment_id,
                candidate_identity=entity.candidate_identity,
                candidate_version=entity.candidate_version,
                provider_identity=entity.provider_identity,
                model_identity=entity.model_identity,
                runtime_identity=entity.runtime_identity,
                created_at=entity.created_at,
            )
        )
        self.session.flush()

    def get(
        self,
        entity_id: UUID,
    ) -> CandidateRecord | None:
        """Return one candidate by identifier."""
        row = self.session.get(
            CandidateRow,
            entity_id,
        )

        if row is None:
            return None

        return _candidate_record(row)

    def list_for_experiment(
        self,
        experiment_id: UUID,
    ) -> tuple[CandidateRecord, ...]:
        """Return candidates for one experiment in deterministic order."""
        rows = self.session.scalars(
            select(CandidateRow)
            .where(
                CandidateRow.experiment_id == experiment_id
            )
            .order_by(
                CandidateRow.created_at,
                CandidateRow.candidate_id,
            )
        ).all()

        return tuple(
            _candidate_record(row)
            for row in rows
        )


class EvidenceRepository(
    Repository[EvidenceRecord, UUID],
):
    """Persist and retrieve typed evidence."""

    def add(
        self,
        entity: EvidenceRecord,
    ) -> None:
        """Add evidence to the current transaction."""
        self.session.add(
            EvidenceRow(
                evidence_id=entity.evidence_id,
                experiment_id=entity.experiment_id,
                candidate_id=entity.candidate_id,
                kind=entity.kind,
                payload=dict(entity.payload),
                created_at=entity.created_at,
            )
        )
        self.session.flush()

    def get(
        self,
        entity_id: UUID,
    ) -> EvidenceRecord | None:
        """Return one evidence record by identifier."""
        row = self.session.get(
            EvidenceRow,
            entity_id,
        )

        if row is None:
            return None

        return _evidence_record(row)

    def list_for_experiment(
        self,
        experiment_id: UUID,
    ) -> tuple[EvidenceRecord, ...]:
        """Return experiment evidence in deterministic order."""
        rows = self.session.scalars(
            select(EvidenceRow)
            .where(
                EvidenceRow.experiment_id == experiment_id
            )
            .order_by(
                EvidenceRow.created_at,
                EvidenceRow.evidence_id,
            )
        ).all()

        return tuple(
            _evidence_record(row)
            for row in rows
        )


class ArtifactMetadataRepository(
    Repository[ArtifactRecord, UUID],
):
    """Persist artifact metadata without storing artifact bytes."""

    def add(
        self,
        entity: ArtifactRecord,
    ) -> None:
        """Add artifact metadata to the current transaction."""
        self.session.add(
            ArtifactRow(
                artifact_id=entity.artifact_id,
                experiment_id=entity.experiment_id,
                evidence_id=entity.evidence_id,
                storage_location=entity.storage_location,
                sha256=entity.sha256,
                size_bytes=entity.size_bytes,
                media_type=entity.media_type,
                created_at=entity.created_at,
            )
        )
        self.session.flush()

    def get(
        self,
        entity_id: UUID,
    ) -> ArtifactRecord | None:
        """Return artifact metadata by identifier."""
        row = self.session.get(
            ArtifactRow,
            entity_id,
        )

        if row is None:
            return None

        return _artifact_record(row)

    def list_for_experiment(
        self,
        experiment_id: UUID,
    ) -> tuple[ArtifactRecord, ...]:
        """Return artifact metadata in deterministic order."""
        rows = self.session.scalars(
            select(ArtifactRow)
            .where(
                ArtifactRow.experiment_id == experiment_id
            )
            .order_by(
                ArtifactRow.created_at,
                ArtifactRow.artifact_id,
            )
        ).all()

        return tuple(
            _artifact_record(row)
            for row in rows
        )
