"""Tests for bounded statistical verification contracts."""

import pytest
from pydantic import ValidationError

from vait.contracts.models import BoundedVerificationPolicy


def test_valid_bounded_policy_parses() -> None:
    """A valid bounded verification policy should parse."""
    policy = BoundedVerificationPolicy(
        max_overall_disagreement_rate=0.05,
        max_high_risk_disagreement_rate=0.01,
        confidence_level=0.95,
        min_total_cases=100,
        min_high_risk_cases=20,
    )

    assert policy.max_overall_disagreement_rate == 0.05
    assert policy.confidence_level == 0.95


@pytest.mark.parametrize(
    "disagreement_rate",
    [
        -0.01,
        1.01,
    ],
)
def test_invalid_disagreement_threshold_is_rejected(
    disagreement_rate: float,
) -> None:
    """Disagreement thresholds must represent probabilities."""
    with pytest.raises(ValidationError):
        BoundedVerificationPolicy(
            max_overall_disagreement_rate=disagreement_rate,
            max_high_risk_disagreement_rate=0.0,
        )


def test_invalid_confidence_level_is_rejected() -> None:
    """Statistical confidence must be strictly between 0.5 and 1."""
    with pytest.raises(ValidationError):
        BoundedVerificationPolicy(
            max_overall_disagreement_rate=0.05,
            max_high_risk_disagreement_rate=0.0,
            confidence_level=1.0,
        )
