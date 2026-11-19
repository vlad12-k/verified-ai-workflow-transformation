"""Explicit safety policy for platform-controlled provider dispatch."""

from dataclasses import dataclass
from enum import StrEnum


class ProviderLocation(StrEnum):
    LOCAL = "local"
    REMOTE = "remote"


class DispatchSafetyMode(StrEnum):
    LOCAL_ONLY = "local_only"
    REMOTE_RISK_ACCEPTED = "remote_risk_accepted"
    STRICT_NO_LATE_IO = "strict_no_late_io"


class ProviderDispatchPolicyError(ValueError):
    """Reject unsupported or unapproved provider dispatch."""


@dataclass(frozen=True, slots=True)
class ProviderDispatchPolicy:
    """Trusted server-side dispatch policy, not caller request metadata."""

    location: ProviderLocation
    mode: DispatchSafetyMode
    residual_risk_accepted: bool = False
    internal_retries_verified_disabled: bool = False

    def validate(self) -> None:
        """Fail closed when required provider guarantees are unavailable."""
        if self.mode is DispatchSafetyMode.STRICT_NO_LATE_IO:
            raise ProviderDispatchPolicyError(
                "Strict no-late-I/O requires verified receiver-side fencing."
            )

        if self.mode is DispatchSafetyMode.LOCAL_ONLY:
            if self.location is not ProviderLocation.LOCAL:
                raise ProviderDispatchPolicyError(
                    "Local-only mode rejects remote providers."
                )
            return

        if self.mode is DispatchSafetyMode.REMOTE_RISK_ACCEPTED:
            if self.location is not ProviderLocation.REMOTE:
                raise ProviderDispatchPolicyError(
                    "Remote mode requires a remote provider."
                )
            if not self.residual_risk_accepted:
                raise ProviderDispatchPolicyError(
                    "Remote dispatch requires explicit risk acceptance."
                )
            if not self.internal_retries_verified_disabled:
                raise ProviderDispatchPolicyError(
                    "Remote adapter retry behaviour must be audited."
                )
            return

        raise ProviderDispatchPolicyError(
            "Unsupported provider dispatch mode."
        )
