"""Unactivated, fail-closed boundary for future verification-job resolution."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from vait.platform.jobs.composition import LocalWorkerOperation
from vait.platform.jobs.models import JobRecord
from vait.platform.registry.contracts import CandidateRecord, ExperimentRunRecord
from vait.runners.base import ImplementationRunner
from vait.verification.trusted_bundle import TrustedVerificationBundle
from vait.verification.trusted_implementations import TrustedImplementationRegistry


class ExperimentRunReader(Protocol):
    """Read persisted experiment lineage by its stable identifier."""

    def get(self, entity_id: UUID) -> ExperimentRunRecord | None:
        """Return one persisted experiment or ``None``."""
        ...


class TrustedBundleReader(Protocol):
    """Read a persisted, integrity-checked bundle by experiment identifier."""

    def get(self, entity_id: UUID) -> TrustedVerificationBundle | None:
        """Return one persisted bundle or ``None``."""
        ...


class CandidateReader(Protocol):
    """Read candidate identity/version records for one experiment."""

    def list_for_experiment(self, experiment_id: UUID) -> tuple[CandidateRecord, ...]:
        """Return persisted candidate records for the experiment."""
        ...


class TrustedVerificationResolutionError(ValueError):
    """Base error for a rejected future verification resolution request."""


class UnsupportedVerificationJobReferenceError(TrustedVerificationResolutionError):
    """Raised when the job is not an admitted reference-only verification job."""


class UnknownTrustedExperimentError(TrustedVerificationResolutionError):
    """Raised when the referenced persisted experiment does not exist."""


class MissingTrustedVerificationBundleError(TrustedVerificationResolutionError):
    """Raised when a persisted experiment has no trusted bundle."""


class TrustedVerificationLineageError(TrustedVerificationResolutionError):
    """Raised when the persisted experiment and its bundle do not agree."""


class TrustedCandidateLineageError(TrustedVerificationResolutionError):
    """Raised when the candidate identity/version is absent or ambiguous."""


@dataclass(frozen=True, slots=True)
class ResolvedTrustedVerificationJob:
    """Validated experiment/bundle lineage and code-owned runner constructors."""

    experiment: ExperimentRunRecord
    bundle: TrustedVerificationBundle
    reference: ImplementationRunner
    reference_version: str
    candidate: ImplementationRunner
    candidate_version: str


def resolve_trusted_verification_job(
    job: JobRecord,
    *,
    experiment_repository: ExperimentRunReader,
    bundle_repository: TrustedBundleReader,
    candidate_repository: CandidateReader,
    implementation_registry: TrustedImplementationRegistry,
) -> ResolvedTrustedVerificationJob:
    """Resolve only persisted identities; never interpret payload as code.

    The persisted job's ``experiment_id`` is used as an untrusted lookup key.
    The verification operation currently has no payload contract, so any
    payload fields—including module, callable, or import instructions—are
    rejected. Experiment and bundle lineage, plus persisted candidate identity
    and version, are validated before any implementation factory is called.

    This function is a resolution boundary only. It is deliberately not wired
    into worker composition and does not prove job ownership, lease validity, or
    permission to execute the returned runners. The reader protocols are adapter
    seams, not proof of persistence or authenticity; production composition must
    supply trusted session-bound repositories and a consistent read boundary.
    """
    if not isinstance(job, JobRecord):
        raise TypeError("job must be a JobRecord")
    try:
        job = JobRecord.model_validate(job.model_dump(mode="python"))
    except (TypeError, ValueError) as error:
        raise UnsupportedVerificationJobReferenceError(
            "Verification job reference failed validation"
        ) from error
    if job.operation != LocalWorkerOperation.VERIFICATION.value or job.payload:
        raise UnsupportedVerificationJobReferenceError(
            "Verification resolution accepts only the verification operation and an empty payload"
        )

    try:
        experiment = experiment_repository.get(job.experiment_id)
    except (TypeError, ValueError) as error:
        raise TrustedVerificationLineageError(
            "Persisted experiment lineage could not be validated"
        ) from error
    if experiment is None:
        raise UnknownTrustedExperimentError("Unknown persisted experiment")
    if not isinstance(experiment, ExperimentRunRecord):
        raise TrustedVerificationLineageError(
            "Persisted experiment record is invalid"
        )
    try:
        experiment = ExperimentRunRecord.model_validate(
            experiment.model_dump(mode="python"),
        )
    except (TypeError, ValueError) as error:
        raise TrustedVerificationLineageError(
            "Persisted experiment record failed validation"
        ) from error
    if experiment.experiment_id != job.experiment_id:
        raise TrustedVerificationLineageError(
            "Persisted experiment identity does not match the job reference"
        )

    try:
        bundle = bundle_repository.get(job.experiment_id)
    except (TypeError, ValueError) as error:
        raise TrustedVerificationLineageError(
            "Persisted trusted bundle failed lineage validation"
        ) from error
    if bundle is None:
        raise MissingTrustedVerificationBundleError(
            "Persisted experiment has no trusted verification bundle"
        )
    if not isinstance(bundle, TrustedVerificationBundle):
        raise TrustedVerificationLineageError(
            "Persisted trusted bundle is invalid"
        )

    try:
        bundle.verify_integrity()
        contract = bundle.contract
    except (TypeError, ValueError) as error:
        raise TrustedVerificationLineageError(
            "Trusted verification bundle failed integrity validation"
        ) from error

    if (
        bundle.experiment_id != experiment.experiment_id
        or bundle.experiment_id != job.experiment_id
        or contract.id != experiment.transformation_contract_id
        or bundle.contract_version != experiment.transformation_contract_version
        or bundle.source_revision != experiment.source_revision
    ):
        raise TrustedVerificationLineageError(
            "Trusted verification bundle does not match persisted experiment lineage"
        )

    try:
        candidates = candidate_repository.list_for_experiment(job.experiment_id)
    except (TypeError, ValueError) as error:
        raise TrustedCandidateLineageError(
            "Persisted candidate lineage could not be validated"
        ) from error
    if not isinstance(candidates, Sequence):
        raise TrustedCandidateLineageError(
            "Persisted candidate records are invalid"
        )

    candidate_version = _candidate_version(
        candidates,
        experiment_id=experiment.experiment_id,
        implementation_id=contract.candidate_implementation_id,
    )

    reference_version, reference = implementation_registry.resolve_unique_id_with_version(
        implementation_id=contract.reference_implementation_id,
        source_revision=experiment.source_revision,
    )
    candidate = implementation_registry.resolve(
        implementation_id=contract.candidate_implementation_id,
        version=candidate_version,
        source_revision=experiment.source_revision,
    )
    return ResolvedTrustedVerificationJob(
        experiment=experiment,
        bundle=bundle,
        reference=reference,
        reference_version=reference_version,
        candidate=candidate,
        candidate_version=candidate_version,
    )


def _candidate_version(
    candidates: Sequence[CandidateRecord],
    *,
    experiment_id: UUID,
    implementation_id: str,
) -> str:
    """Require one persisted candidate row matching the trusted contract ID."""
    matches: list[CandidateRecord] = []
    for candidate in candidates:
        if not isinstance(candidate, CandidateRecord):
            raise TrustedCandidateLineageError(
                "Persisted candidate record is invalid"
            )
        try:
            candidate = CandidateRecord.model_validate(
                candidate.model_dump(mode="python"),
            )
        except (TypeError, ValueError) as error:
            raise TrustedCandidateLineageError(
                "Persisted candidate record failed validation"
            ) from error
        if candidate.experiment_id != experiment_id:
            raise TrustedCandidateLineageError(
                "Persisted candidate belongs to another experiment"
            )
        if candidate.candidate_identity == implementation_id:
            matches.append(candidate)

    if not matches:
        raise TrustedCandidateLineageError(
            "Trusted contract candidate has no persisted candidate version"
        )
    if len(matches) != 1:
        raise TrustedCandidateLineageError(
            "Trusted contract candidate version is ambiguous"
        )
    return matches[0].candidate_version
