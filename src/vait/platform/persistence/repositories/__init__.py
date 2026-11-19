"""Repository contracts and implementations for durable platform state."""

from vait.platform.persistence.repositories.base import Repository
from vait.platform.persistence.repositories.budgets import (
    PostgreSQLBudgetReservationLedger,
)
from vait.platform.persistence.repositories.registry import (
    ArtifactMetadataRepository,
    CandidateRepository,
    EvidenceRepository,
    ExperimentRunRepository,
)
from vait.platform.persistence.repositories.trusted_bundles import (
    ConflictingTrustedVerificationBundleError,
    DuplicateTrustedVerificationBundleError,
    PersistedTrustedVerificationBundleIntegrityError,
    TrustedVerificationBundleIntegrityError,
    TrustedVerificationBundleLineageError,
    TrustedVerificationBundleRepository,
    TrustedVerificationBundleRepositoryError,
    UnknownExperimentError,
)

__all__ = [
    "ArtifactMetadataRepository",
    "CandidateRepository",
    "ConflictingTrustedVerificationBundleError",
    "DuplicateTrustedVerificationBundleError",
    "EvidenceRepository",
    "ExperimentRunRepository",
    "PersistedTrustedVerificationBundleIntegrityError",
    "PostgreSQLBudgetReservationLedger",
    "Repository",
    "TrustedVerificationBundleIntegrityError",
    "TrustedVerificationBundleLineageError",
    "TrustedVerificationBundleRepository",
    "TrustedVerificationBundleRepositoryError",
    "UnknownExperimentError",
]
