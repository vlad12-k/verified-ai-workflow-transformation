"""Tests for the deterministic extractive RAG generator."""

import pytest

from vait.rag.extractive import ExtractiveGroundedGenerator
from vait.rag.models import (
    RAGContextDocument,
    RAGGenerationRequest,
)


def build_request(
    *,
    top_score: float = 0.8,
) -> RAGGenerationRequest:
    """Build a small deterministic grounded request."""
    return RAGGenerationRequest(
        query="What should happen to a duplicate invoice?",
        context_documents=(
            RAGContextDocument(
                document_id="duplicate-policy",
                text="Duplicate invoices require review.",
                score=top_score,
                rank=1,
            ),
            RAGContextDocument(
                document_id="bank-policy",
                text="Bank changes require verification.",
                score=0.4,
                rank=2,
            ),
        ),
    )


def test_extractive_generator_returns_highest_ranked_evidence() -> None:
    """The baseline should return and cite the top evidence document."""
    generator = ExtractiveGroundedGenerator(
        minimum_score=0.5,
    )

    generation = generator.generate(
        build_request()
    )

    assert generation.answer == (
        "Duplicate invoices require review."
    )
    assert generation.cited_document_ids == (
        "duplicate-policy",
    )
    assert generation.abstained is False


def test_extractive_generator_abstains_below_threshold() -> None:
    """Weak retrieved evidence should produce explicit abstention."""
    generator = ExtractiveGroundedGenerator(
        minimum_score=0.75,
    )

    generation = generator.generate(
        build_request(
            top_score=0.6,
        )
    )

    assert generation.answer == "Insufficient evidence."
    assert generation.cited_document_ids == ()
    assert generation.abstained is True


def test_extractive_generator_abstains_without_context() -> None:
    """Missing evidence should never produce a grounded answer."""
    generator = ExtractiveGroundedGenerator()

    generation = generator.generate(
        RAGGenerationRequest(
            query="Unknown question",
            context_documents=(),
        )
    )

    assert generation.abstained is True
    assert generation.cited_document_ids == ()


def test_extractive_generator_exposes_threshold_in_identity() -> None:
    """Generator identity should preserve threshold configuration."""
    generator = ExtractiveGroundedGenerator(
        minimum_score=0.625,
    )

    assert generator.minimum_score == 0.625
    assert generator.implementation_id == (
        "extractive-grounded-generator@1.0.0:"
        "minimum-score=0.625000"
    )


def test_extractive_generator_rejects_invalid_threshold() -> None:
    """Cosine similarity thresholds must remain in valid range."""
    with pytest.raises(
        ValueError,
        match="between -1.0 and 1.0",
    ):
        ExtractiveGroundedGenerator(
            minimum_score=1.1,
        )
