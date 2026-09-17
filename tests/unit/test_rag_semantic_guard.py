"""Tests for deterministic RAG semantic-drift checks."""

import pytest

from vait.rag.semantic_guard import (
    SemanticDriftRisk,
    assess_semantic_drift,
)


def test_guard_accepts_preserved_hard_policy_semantics() -> None:
    """Preserved obligation and timing should pass."""
    assessment = assess_semantic_drift(
        context_text=(
            "A supplier bank account change requires independent "
            "verification before any payment is released."
        ),
        answer_text=(
            "Independent verification must occur before payment "
            "is released."
        ),
    )

    assert assessment.passed is True
    assert assessment.issues == ()


def test_guard_flags_weakened_obligation() -> None:
    """Hard obligations must not silently become should."""
    assessment = assess_semantic_drift(
        context_text=(
            "Duplicate invoices must be held before approval."
        ),
        answer_text=(
            "Duplicate invoices should be held before approval."
        ),
    )

    assert assessment.passed is False
    assert SemanticDriftRisk.OBLIGATION_WEAKENED in {
        issue.risk
        for issue in assessment.issues
    }


def test_guard_accepts_before_only_after_equivalence() -> None:
    """Only-after form can preserve a before restriction."""
    assessment = assess_semantic_drift(
        context_text=(
            "Payment must not be released before independent "
            "verification is complete."
        ),
        answer_text=(
            "Payment may be released only after independent "
            "verification is complete."
        ),
    )

    assert assessment.passed is True


def test_guard_accepts_unless_except_when_equivalence() -> None:
    """Except-when can preserve an unless exception."""
    assessment = assess_semantic_drift(
        context_text=(
            "Invoices require manual review unless an approved "
            "exception applies."
        ),
        answer_text=(
            "Invoices require manual review except when an "
            "approved exception applies."
        ),
    )

    assert assessment.passed is True


def test_guard_accepts_may_can_equivalence() -> None:
    """Can can preserve bounded non-mandatory modality."""
    assessment = assess_semantic_drift(
        context_text=(
            "Large differences may require escalation."
        ),
        answer_text=(
            "Large differences can require escalation."
        ),
    )

    assert assessment.passed is True


def test_guard_accepts_nonzero_paraphrase() -> None:
    """Other-than-zero can preserve non-zero scope."""
    assessment = assess_semantic_drift(
        context_text=(
            "Any non-zero difference requires review."
        ),
        answer_text=(
            "Every difference other than zero requires review."
        ),
    )

    assert assessment.passed is True


def test_guard_accepts_prior_to_paraphrase() -> None:
    """Prior-to can preserve a before constraint."""
    assessment = assess_semantic_drift(
        context_text=(
            "Independent verification is required before payment."
        ),
        answer_text=(
            "Independent verification is required prior to payment."
        ),
    )

    assert assessment.passed is True


def test_guard_flags_universal_scope_narrowing() -> None:
    """Any must not silently become a narrower subset."""
    assessment = assess_semantic_drift(
        context_text=(
            "Any supplier bank account change requires "
            "independent verification."
        ),
        answer_text=(
            "Major supplier bank account changes require "
            "independent verification."
        ),
    )

    assert assessment.passed is False
    assert SemanticDriftRisk.UNIVERSAL_SCOPE_DROPPED in {
        issue.risk
        for issue in assessment.issues
    }


def test_guard_flags_added_causal_rationale() -> None:
    """Unsupported causal explanations should be flagged."""
    assessment = assess_semantic_drift(
        context_text=(
            "Zero-value invoices require review."
        ),
        answer_text=(
            "Zero-value invoices require review because "
            "they are fraudulent."
        ),
    )

    assert assessment.passed is False
    assert SemanticDriftRisk.CAUSAL_RATIONALE_ADDED in {
        issue.risk
        for issue in assessment.issues
    }


def test_guard_flags_scope_and_conditional_drift() -> None:
    """Scope and conditional semantics should remain visible."""
    assessment = assess_semantic_drift(
        context_text=(
            "Any non-zero difference requires review. "
            "Large differences may require escalation."
        ),
        answer_text=(
            "Review the invoice for any significant discrepancy."
        ),
    )

    risks = {
        issue.risk
        for issue in assessment.issues
    }

    assert assessment.passed is False
    assert SemanticDriftRisk.OBLIGATION_WEAKENED in risks
    assert SemanticDriftRisk.SCOPE_QUALIFIER_DROPPED in risks


def test_guard_flags_conditional_strengthening() -> None:
    """May must not silently become a mandatory obligation."""
    assessment = assess_semantic_drift(
        context_text=(
            "Large differences may require escalation."
        ),
        answer_text=(
            "Large differences must be escalated."
        ),
    )

    risks = {
        issue.risk
        for issue in assessment.issues
    }

    assert assessment.passed is False
    assert (
        SemanticDriftRisk.CONDITIONAL_MODALITY_DROPPED
        in risks
    )


@pytest.mark.parametrize(
    ("context_text", "answer_text"),
    (
        ("", "answer"),
        ("context", ""),
        ("   ", "answer"),
        ("context", "   "),
    ),
)
def test_guard_rejects_empty_inputs(
    context_text: str,
    answer_text: str,
) -> None:
    """The guard requires both evidence and generated text."""
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        assess_semantic_drift(
            context_text=context_text,
            answer_text=answer_text,
        )
