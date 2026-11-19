"""Fail-closed routing metadata for local worker operations."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from vait.platform.jobs.models import JobRecord


class LocalWorkerOperation(StrEnum):
    """Local operation names admitted by the code-owned allowlist."""

    VERIFICATION = "verification"


@dataclass(frozen=True, slots=True, init=False)
class TrustedLocalExecutorRegistry:
    """Immutable operation routing only; it intentionally stores no executor.

    Registration values are ``None`` to represent allowlisted routing metadata,
    not an executable implementation. Executable verification remains disabled
    until trusted implementation identity and durable evidence contracts exist.
    """

    _registrations: Mapping[LocalWorkerOperation, None] = field(
        init=False,
        repr=False,
    )

    def __init__(
        self,
        registrations: Mapping[LocalWorkerOperation, None],
    ) -> None:
        if not isinstance(registrations, Mapping):
            raise TypeError("Local operation registrations must be a mapping")

        snapshot = dict(registrations)
        if not snapshot:
            raise ValueError("At least one local operation registration is required")

        for operation, registration in snapshot.items():
            if not isinstance(operation, LocalWorkerOperation):
                raise TypeError("Registration keys must be LocalWorkerOperation values")
            if registration is not None:
                raise TypeError(
                    "Executor instances are not accepted; local registration is routing-only"
                )

        object.__setattr__(
            self,
            "_registrations",
            MappingProxyType(snapshot),
        )

    @property
    def allowed_operations(self) -> tuple[LocalWorkerOperation, ...]:
        """Return the immutable typed allowlist used by claim filtering."""
        return tuple(self._registrations)

    def route(self, job: JobRecord) -> LocalWorkerOperation:
        """Validate a claimed job's operation without executing it or its payload."""
        if job.status != "RUNNING" or job.claimed_by is None:
            raise ValueError("Local routing requires a claimed RUNNING job")

        try:
            operation = LocalWorkerOperation(job.operation)
        except (TypeError, ValueError):
            raise ValueError("Unsupported local operation") from None

        if operation not in self._registrations:
            raise ValueError("No local registration exists for operation")

        return operation


_CODE_OWNED_REGISTRATIONS: Mapping[LocalWorkerOperation, None] = MappingProxyType(
    {LocalWorkerOperation.VERIFICATION: None}
)


def create_trusted_local_executor_registry() -> TrustedLocalExecutorRegistry:
    """Create the code-owned routing allowlist; this does not enable execution."""
    return TrustedLocalExecutorRegistry(_CODE_OWNED_REGISTRATIONS)
