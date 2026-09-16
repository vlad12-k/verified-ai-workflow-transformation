"""Tests for VAIT statistical utilities."""

import pytest

from vait.verification.statistics import binomial_upper_bound


def test_zero_events_still_has_nonzero_upper_bound() -> None:
    """Zero observed events must not imply zero population risk."""
    upper = binomial_upper_bound(
        events=0,
        trials=100,
        confidence_level=0.95,
    )

    assert 0.0 < upper < 0.05


def test_all_events_returns_one() -> None:
    """An all-event sample has an upper bound of one."""
    assert (
        binomial_upper_bound(
            events=10,
            trials=10,
            confidence_level=0.95,
        )
        == 1.0
    )


def test_more_evidence_tightens_zero_event_bound() -> None:
    """Larger clean samples should produce tighter upper bounds."""
    small_sample = binomial_upper_bound(
        events=0,
        trials=10,
        confidence_level=0.95,
    )
    large_sample = binomial_upper_bound(
        events=0,
        trials=1000,
        confidence_level=0.95,
    )

    assert large_sample < small_sample


@pytest.mark.parametrize(
    ("events", "trials"),
    [
        (-1, 10),
        (11, 10),
    ],
)
def test_invalid_event_counts_are_rejected(
    events: int,
    trials: int,
) -> None:
    """Invalid binomial observations must be rejected."""
    with pytest.raises(ValueError):
        binomial_upper_bound(
            events=events,
            trials=trials,
        )


def test_zero_trials_are_rejected() -> None:
    """A statistical bound requires observed trials."""
    with pytest.raises(
        ValueError,
        match="trials must be greater than zero",
    ):
        binomial_upper_bound(
            events=0,
            trials=0,
        )


@pytest.mark.parametrize(
    "confidence_level",
    [
        0.5,
        1.0,
    ],
)
def test_invalid_confidence_levels_are_rejected(
    confidence_level: float,
) -> None:
    """Confidence level must be strictly between 0.5 and 1."""
    with pytest.raises(ValueError):
        binomial_upper_bound(
            events=0,
            trials=10,
            confidence_level=confidence_level,
        )
