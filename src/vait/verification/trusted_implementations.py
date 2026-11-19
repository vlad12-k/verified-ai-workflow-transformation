"""Immutable registry of explicitly code-owned Python implementations."""

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from vait.contracts.models import Effect
from vait.runners.base import ImplementationRunner
from vait.runners.python_runner import PythonImplementationRunner
from vait.transformations.library.synthetic_ap import (
    build_synthetic_ap_transformation,
)

_IMPLEMENTATION_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_IMPLEMENTATION_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_SOURCE_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")

ImplementationFactory = Callable[[], ImplementationRunner]
ImplementationKey = tuple[str, str]


class TrustedImplementationRegistryError(ValueError):
    """Base error for rejected trusted implementation resolution."""


class UnknownTrustedImplementationError(TrustedImplementationRegistryError):
    """Raised when no code-owned implementation has the requested ID."""


class UnsupportedTrustedImplementationVersionError(
    TrustedImplementationRegistryError,
):
    """Raised when an implementation ID has no registration for the version."""


class AmbiguousTrustedImplementationVersionError(
    TrustedImplementationRegistryError,
):
    """Raised when an ID-only reference matches multiple approved versions."""


class ConflictingTrustedImplementationRegistrationError(
    TrustedImplementationRegistryError,
):
    """Raised when code declares the same implementation key more than once."""


class TrustedImplementationSourceRevisionError(
    TrustedImplementationRegistryError,
):
    """Raised when a registration's pinned source revision does not match."""


class InvalidTrustedImplementationRunnerError(
    TrustedImplementationRegistryError,
):
    """Raised when a trusted factory returns an invalid or misidentified runner."""


@dataclass(frozen=True, slots=True)
class TrustedImplementationRegistration:
    """Code-owned factory binding for one immutable implementation version."""

    implementation_id: str
    version: str
    factory: ImplementationFactory = field(repr=False)
    source_revision: str | None = None

    def __post_init__(self) -> None:
        """Validate code-owned registration metadata at construction time."""
        if (
            not isinstance(self.implementation_id, str)
            or not _IMPLEMENTATION_ID_PATTERN.fullmatch(self.implementation_id)
        ):
            raise ValueError("Implementation ID is invalid")
        if (
            not isinstance(self.version, str)
            or not _IMPLEMENTATION_VERSION_PATTERN.fullmatch(self.version)
        ):
            raise ValueError("Implementation version must use numeric major.minor.patch")
        if not callable(self.factory):
            raise TypeError("Implementation factory must be callable")
        if self.source_revision is not None and (
            not isinstance(self.source_revision, str)
            or not _SOURCE_REVISION_PATTERN.fullmatch(self.source_revision)
        ):
            raise ValueError("Pinned source revision must be a full lowercase Git SHA-1")


@dataclass(frozen=True, slots=True, init=False)
class TrustedImplementationRegistry:
    """Immutable snapshot of application-owned implementation registrations.

    The registry has no mutation or deserialization API. Applications construct
    it from code-owned registration objects; persisted job data is never a
    registration source.
    """

    _registrations: Mapping[
        ImplementationKey,
        TrustedImplementationRegistration,
    ] = field(init=False, repr=False)

    def __init__(
        self,
        registrations: Sequence[TrustedImplementationRegistration],
    ) -> None:
        """Snapshot a non-empty sequence of reviewed code-owned registrations."""
        if isinstance(registrations, (str, bytes, bytearray)) or not isinstance(
            registrations,
            Sequence,
        ):
            raise TypeError("Trusted implementation registrations must be a sequence")

        snapshot: dict[ImplementationKey, TrustedImplementationRegistration] = {}
        for registration in registrations:
            if not isinstance(registration, TrustedImplementationRegistration):
                raise TypeError("Registration entries must be trusted registration objects")
            key = (registration.implementation_id, registration.version)
            if key in snapshot:
                raise ConflictingTrustedImplementationRegistrationError(
                    "Conflicting registration for implementation ID and version"
                )
            snapshot[key] = registration

        if not snapshot:
            raise ValueError("At least one trusted implementation registration is required")

        object.__setattr__(self, "_registrations", MappingProxyType(snapshot))

    def resolve(
        self,
        *,
        implementation_id: str,
        version: str,
        source_revision: str | None = None,
    ) -> ImplementationRunner:
        """Build the exact code-owned runner without executing a case."""
        registration = self._get_registration(implementation_id, version)
        self._validate_source_revision(registration, source_revision)
        runner = registration.factory()
        return self._validate_runner(runner, implementation_id)

    def resolve_unique_id(
        self,
        *,
        implementation_id: str,
        source_revision: str | None = None,
    ) -> ImplementationRunner:
        """Resolve an ID only when exactly one code-owned version is approved."""
        _, runner = self.resolve_unique_id_with_version(
            implementation_id=implementation_id,
            source_revision=source_revision,
        )
        return runner

    def resolve_unique_id_with_version(
        self,
        *,
        implementation_id: str,
        source_revision: str | None = None,
    ) -> tuple[str, ImplementationRunner]:
        """Return the unique registered version and its constructed runner."""
        matching_versions = tuple(
            version
            for registered_id, version in self._registrations
            if registered_id == implementation_id
        )
        if not matching_versions:
            raise UnknownTrustedImplementationError(
                "Unknown trusted implementation ID"
            )
        if len(matching_versions) != 1:
            raise AmbiguousTrustedImplementationVersionError(
                "Implementation ID has multiple approved versions; an exact version is required"
            )
        version = matching_versions[0]
        return (
            version,
            self.resolve(
                implementation_id=implementation_id,
                version=version,
                source_revision=source_revision,
            ),
        )

    def _get_registration(
        self,
        implementation_id: str,
        version: str,
    ) -> TrustedImplementationRegistration:
        """Find the exact allowlisted identity without consulting job data."""
        if not isinstance(implementation_id, str) or not implementation_id.strip():
            raise UnknownTrustedImplementationError("Unknown trusted implementation ID")
        if not isinstance(version, str) or not version.strip():
            raise UnsupportedTrustedImplementationVersionError(
                "Unsupported trusted implementation version"
            )
        registration = self._registrations.get((implementation_id, version))
        if registration is not None:
            return registration
        if any(registered_id == implementation_id for registered_id, _ in self._registrations):
            raise UnsupportedTrustedImplementationVersionError(
                "Unsupported trusted implementation version"
            )
        raise UnknownTrustedImplementationError("Unknown trusted implementation ID")

    @staticmethod
    def _validate_source_revision(
        registration: TrustedImplementationRegistration,
        source_revision: str | None,
    ) -> None:
        """Enforce an optional code-owned revision pin when one is declared."""
        if source_revision is not None and (
            not isinstance(source_revision, str)
            or not _SOURCE_REVISION_PATTERN.fullmatch(source_revision)
        ):
            raise TrustedImplementationSourceRevisionError(
                "Source revision is invalid"
            )
        if (
            registration.source_revision is not None
            and source_revision != registration.source_revision
        ):
            raise TrustedImplementationSourceRevisionError(
                "Implementation source revision does not match its code-owned registration"
            )

    @staticmethod
    def _validate_runner(
        runner: object,
        implementation_id: str,
    ) -> ImplementationRunner:
        """Enforce the runner protocol and the registered identity at runtime."""
        if not isinstance(runner, ImplementationRunner):
            raise InvalidTrustedImplementationRunnerError(
                "Trusted implementation factory did not return an ImplementationRunner"
            )

        if (
            not isinstance(runner.implementation_id, str)
            or runner.implementation_id != implementation_id
            or not isinstance(runner.declared_effects, frozenset)
            or not all(isinstance(effect, Effect) for effect in runner.declared_effects)
            or not callable(runner.execute)
        ):
            raise InvalidTrustedImplementationRunnerError(
                "Trusted implementation factory returned an invalid or misidentified runner"
            )
        return runner


def _build_synthetic_ap_policy_runner() -> ImplementationRunner:
    """Construct the existing deterministic synthetic AP rule runner."""
    transformation = build_synthetic_ap_transformation()
    return PythonImplementationRunner(
        implementation_id=transformation.implementation_id,
        function=transformation.function,
        declared_effects=transformation.descriptor.declared_effects,
    )


_CODE_OWNED_IMPLEMENTATIONS = (
    TrustedImplementationRegistration(
        implementation_id="synthetic-ap-policy-v0.2",
        version="1.0.0",
        factory=_build_synthetic_ap_policy_runner,
    ),
)


def create_trusted_implementation_registry() -> TrustedImplementationRegistry:
    """Create the immutable allowlist of reviewed code-owned implementations."""
    return TrustedImplementationRegistry(_CODE_OWNED_IMPLEMENTATIONS)
