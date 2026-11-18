"""Repository for immutable trusted verification bundle snapshots."""

from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert

from vait.platform.persistence.registry_models import (
    ExperimentRunRow,
    TrustedVerificationBundleRow,
)
from vait.platform.persistence.repositories.base import Repository
from vait.platform.persistence.repositories.registry import ExperimentRunRepository
from vait.platform.registry.contracts import ExperimentRunRecord
from vait.verification.trusted_bundle import TrustedVerificationBundle


class TrustedVerificationBundleRepositoryError(ValueError):
    """Base error for a rejected trusted bundle persistence operation."""


class UnknownExperimentError(TrustedVerificationBundleRepositoryError):
    """Raised when a bundle refers to an experiment absent from persistence."""


class TrustedVerificationBundleLineageError(
    TrustedVerificationBundleRepositoryError,
):
    """Raised when bundle identity or provenance differs from its experiment."""


class TrustedVerificationBundleIntegrityError(
    TrustedVerificationBundleRepositoryError,
):
    """Raised when a bundle's canonical content or fingerprint is invalid."""


class PersistedTrustedVerificationBundleIntegrityError(
    TrustedVerificationBundleIntegrityError,
):
    """Raised when persisted bundle bytes or explicit columns disagree."""


class DuplicateTrustedVerificationBundleError(
    TrustedVerificationBundleRepositoryError,
):
    """Raised when the same experiment already has a registered bundle."""


class ConflictingTrustedVerificationBundleError(
    TrustedVerificationBundleRepositoryError,
):
    """Raised when a second bundle conflicts with the immutable registration."""


class TrustedVerificationBundleRepository(
    Repository[TrustedVerificationBundle, UUID],
):
    """Persist and retrieve canonical trusted bundles within one transaction."""

    def add(
        self,
        entity: TrustedVerificationBundle,
    ) -> None:
        """Insert one bundle row into the current transaction."""
        if not isinstance(entity, TrustedVerificationBundle):
            raise TypeError("entity must be a TrustedVerificationBundle")
        try:
            entity.verify_integrity()
        except (TypeError, ValueError) as error:
            raise TrustedVerificationBundleIntegrityError(
                "Trusted verification bundle failed integrity validation"
            ) from error

        experiment = self._get_experiment(entity.experiment_id, lock=True)
        self._validate_lineage(entity, experiment)
        contract = entity.contract
        canonical_json = entity.to_json()
        result = self.session.execute(
            postgresql_insert(TrustedVerificationBundleRow)
            .values(
                experiment_id=entity.experiment_id,
                bundle_version=entity.bundle_version,
                contract_id=contract.id,
                contract_version=entity.contract_version,
                source_revision=entity.source_revision,
                fingerprint_sha256=entity.fingerprint_sha256,
                canonical_json=canonical_json,
            )
            .on_conflict_do_nothing(
                index_elements=[TrustedVerificationBundleRow.experiment_id],
            )
            .returning(TrustedVerificationBundleRow.experiment_id)
        )
        if result.scalar_one_or_none() is not None:
            return

        existing = self._get_bundle_row(entity.experiment_id)
        if existing is None:
            raise ConflictingTrustedVerificationBundleError(
                "Trusted bundle registration conflicts with persisted state"
            )
        self._reconstruct(existing)
        if existing.canonical_json == canonical_json:
            raise DuplicateTrustedVerificationBundleError(
                "Trusted verification bundle is already registered"
            )
        raise ConflictingTrustedVerificationBundleError(
            "Conflicting trusted verification bundle is already registered"
        )

    def get(
        self,
        entity_id: UUID,
    ) -> TrustedVerificationBundle | None:
        """Return a reconstructed bundle by experiment identifier."""
        row = self._get_bundle_row(entity_id)
        if row is None:
            return None
        return self._reconstruct(row)

    def _get_experiment(
        self,
        experiment_id: UUID,
        *,
        lock: bool = False,
    ) -> ExperimentRunRecord:
        """Load authoritative experiment lineage from the current transaction."""
        statement = (
            select(ExperimentRunRow)
            .where(ExperimentRunRow.experiment_id == experiment_id)
            .execution_options(populate_existing=True)
        )
        if lock:
            statement = statement.with_for_update(read=True)
        row = self.session.scalar(statement)
        if row is None:
            raise UnknownExperimentError(
                "Unknown experiment for trusted verification bundle"
            )
        try:
            experiment = ExperimentRunRepository(self.session).get(experiment_id)
        except ValidationError as error:
            raise TrustedVerificationBundleLineageError(
                "Persisted experiment lineage is invalid"
            ) from error
        if experiment is None:
            raise UnknownExperimentError("Unknown persisted experiment lineage")
        return experiment

    def _validate_lineage(
        self,
        bundle: TrustedVerificationBundle,
        experiment: ExperimentRunRecord,
    ) -> None:
        """Bind canonical bundle identity and provenance to persisted lineage."""
        contract = bundle.contract
        if (
            bundle.experiment_id != experiment.experiment_id
            or contract.id != experiment.transformation_contract_id
            or bundle.contract_version
            != experiment.transformation_contract_version
            or bundle.source_revision != experiment.source_revision
        ):
            raise TrustedVerificationBundleLineageError(
                "Trusted bundle lineage does not match the persisted experiment"
            )

    def _get_bundle_row(
        self,
        experiment_id: UUID,
    ) -> TrustedVerificationBundleRow | None:
        """Read fresh persisted state rather than returning stale identity-map data."""
        return self.session.scalar(
            select(TrustedVerificationBundleRow)
            .where(TrustedVerificationBundleRow.experiment_id == experiment_id)
            .execution_options(populate_existing=True)
        )

    def _reconstruct(
        self,
        row: TrustedVerificationBundleRow,
    ) -> TrustedVerificationBundle:
        """Verify stored canonical text, indexed columns, and experiment lineage."""
        try:
            bundle = TrustedVerificationBundle.from_json(row.canonical_json)
        except (TypeError, ValueError) as error:
            raise PersistedTrustedVerificationBundleIntegrityError(
                "Stored trusted bundle canonical content failed validation"
            ) from error

        contract = bundle.contract
        if (
            row.experiment_id != bundle.experiment_id
            or row.bundle_version != bundle.bundle_version
            or row.contract_id != contract.id
            or row.contract_version != bundle.contract_version
            or row.source_revision != bundle.source_revision
            or row.fingerprint_sha256 != bundle.fingerprint_sha256
        ):
            raise PersistedTrustedVerificationBundleIntegrityError(
                "Stored trusted bundle metadata does not match canonical content"
            )

        experiment = self._get_experiment(row.experiment_id)
        self._validate_lineage(bundle, experiment)
        return bundle
