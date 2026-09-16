"""Decision rules for VAIT verification evidence."""

from collections.abc import Sequence

from vait.decision.models import Decision, VerificationFailure


def decide_exact_or_reject(
    failures: Sequence[VerificationFailure],
) -> Decision:
    """Classify deterministic verification evidence."""
    if failures:
        return Decision.REJECT

    return Decision.EXACT
