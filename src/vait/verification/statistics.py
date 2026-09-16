"""Statistical utilities for bounded VAIT verification."""

from scipy.stats import beta


def binomial_upper_bound(
    events: int,
    trials: int,
    confidence_level: float = 0.95,
) -> float:
    """Return a one-sided exact upper confidence bound for an event rate."""
    if trials <= 0:
        raise ValueError("trials must be greater than zero.")

    if events < 0 or events > trials:
        raise ValueError("events must satisfy 0 <= events <= trials.")

    if not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence_level must satisfy 0.5 < value < 1.0.")

    if events == trials:
        return 1.0

    upper = beta.ppf(
        confidence_level,
        events + 1,
        trials - events,
    )

    return float(upper)
