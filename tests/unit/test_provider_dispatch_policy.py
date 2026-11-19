"""Tests for explicit provider dispatch safety decisions."""

import pytest

from vait.platform.provider_dispatch_policy import (
    DispatchSafetyMode,
    ProviderDispatchPolicy,
    ProviderDispatchPolicyError,
    ProviderLocation,
)


def test_local_provider_is_allowed_in_local_mode() -> None:
    policy = ProviderDispatchPolicy(
        location=ProviderLocation.LOCAL,
        mode=DispatchSafetyMode.LOCAL_ONLY,
    )

    policy.validate()


def test_approved_remote_policy_passes() -> None:
    policy = ProviderDispatchPolicy(
        location=ProviderLocation.REMOTE,
        mode=DispatchSafetyMode.REMOTE_RISK_ACCEPTED,
        residual_risk_accepted=True,
        internal_retries_verified_disabled=True,
    )

    policy.validate()


@pytest.mark.parametrize(
    ("location", "mode", "accepted", "retries_audited"),
    [
        (
            ProviderLocation.REMOTE,
            DispatchSafetyMode.LOCAL_ONLY,
            False,
            False,
        ),
        (
            ProviderLocation.LOCAL,
            DispatchSafetyMode.REMOTE_RISK_ACCEPTED,
            True,
            True,
        ),
        (
            ProviderLocation.REMOTE,
            DispatchSafetyMode.REMOTE_RISK_ACCEPTED,
            False,
            True,
        ),
        (
            ProviderLocation.REMOTE,
            DispatchSafetyMode.REMOTE_RISK_ACCEPTED,
            True,
            False,
        ),
        (
            ProviderLocation.REMOTE,
            DispatchSafetyMode.STRICT_NO_LATE_IO,
            True,
            True,
        ),
        (
            ProviderLocation.LOCAL,
            DispatchSafetyMode.STRICT_NO_LATE_IO,
            True,
            True,
        ),
    ],
)
def test_unsupported_dispatch_fails_closed(
    location: ProviderLocation,
    mode: DispatchSafetyMode,
    accepted: bool,
    retries_audited: bool,
) -> None:
    policy = ProviderDispatchPolicy(
        location=location,
        mode=mode,
        residual_risk_accepted=accepted,
        internal_retries_verified_disabled=retries_audited,
    )

    with pytest.raises(ProviderDispatchPolicyError):
        policy.validate()
