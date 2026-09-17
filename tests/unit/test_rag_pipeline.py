"""Tests for grounded retrieval-augmented generation."""

from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from vait.rag.models import (
    RAGGeneration,
    RAGGenerationRequest,
)
from vait.rag.pipeline import RetrievalAugmentedGenerator
from vait.retrieval.in_memory import InMemoryCosineRetriever
from vait.retrieval.models import TextDocument


class StaticRAGEncoder:
    """Deterministic encoder for RAG tests."""

    @property
    def implementation_id(self) -> str:
        """Return the encoder identifier."""
        return "static-rag-encoder-v1"

    def encode(
        self,
        texts: Sequence[str],
    ) -> NDArray[np.float64]:
        """Return predefined semantic vectors."""
        vectors = {
            "Duplicate invoices require review.": (
                1.0,
                0.0,
            ),
            "Bank changes require verification.": (
                0.0,
                1.0,
            ),
            "duplicate invoice question": (
                0.9,
                0.1,
            ),
        }

        return np.asarray(
            [
                vectors[text]
                for text in texts
            ],
            dtype=np.float64,
        )


class CitingGenerator:
    """Deterministic grounded generator."""

    @property
    def implementation_id(self) -> str:
        """Return the generator identifier."""
        return "deterministic-citing-generator-v1"

    def generate(
        self,
        request: RAGGenerationRequest,
    ) -> RAGGeneration:
        """Cite the highest-ranked supplied evidence."""
        document = request.context_documents[0]

        return RAGGeneration(
            answer="The invoice requires review.",
            cited_document_ids=(
                document.document_id,
            ),
        )


class InvalidCitationGenerator:
    """Generator that invents an unavailable citation."""

    @property
    def implementation_id(self) -> str:
        """Return the generator identifier."""
        return "invalid-citation-generator-v1"

    def generate(
        self,
        request: RAGGenerationRequest,
    ) -> RAGGeneration:
        """Return deliberately invalid grounding metadata."""
        del request

        return RAGGeneration(
            answer="The invoice requires review.",
            cited_document_ids=(
                "invented-policy",
            ),
        )


class AbstainingGenerator:
    """Generator that explicitly declines to answer."""

    @property
    def implementation_id(self) -> str:
        """Return the generator identifier."""
        return "abstaining-generator-v1"

    def generate(
        self,
        request: RAGGenerationRequest,
    ) -> RAGGeneration:
        """Return an evidence-aware abstention."""
        del request

        return RAGGeneration(
            answer="Insufficient evidence.",
            cited_document_ids=(),
            abstained=True,
        )


def build_retriever() -> InMemoryCosineRetriever:
    """Build a deterministic semantic retriever."""
    return InMemoryCosineRetriever(
        encoder=StaticRAGEncoder(),
        documents=(
            TextDocument(
                document_id="duplicate-policy",
                text="Duplicate invoices require review.",
            ),
            TextDocument(
                document_id="bank-policy",
                text="Bank changes require verification.",
            ),
        ),
    )


def test_rag_returns_grounded_answer_with_context_citation() -> None:
    """Valid citations should survive grounding validation."""
    pipeline = RetrievalAugmentedGenerator(
        retriever=build_retriever(),
        generator=CitingGenerator(),
        retrieval_k=2,
        max_context_documents=2,
        max_context_chars=200,
    )

    result = pipeline.run(
        "duplicate invoice question"
    )

    assert result.answer == "The invoice requires review."
    assert result.cited_document_ids == (
        "duplicate-policy",
    )
    assert result.context_documents[0].document_id == (
        "duplicate-policy"
    )
    assert result.abstained is False


def test_rag_rejects_citation_not_present_in_context() -> None:
    """A generator must not cite evidence it did not receive."""
    pipeline = RetrievalAugmentedGenerator(
        retriever=build_retriever(),
        generator=InvalidCitationGenerator(),
        retrieval_k=2,
        max_context_documents=1,
        max_context_chars=200,
    )

    with pytest.raises(
        ValueError,
        match="not supplied in context",
    ):
        pipeline.run(
            "duplicate invoice question"
        )


def test_rag_supports_explicit_abstention() -> None:
    """A generator may abstain without claiming evidence."""
    pipeline = RetrievalAugmentedGenerator(
        retriever=build_retriever(),
        generator=AbstainingGenerator(),
        retrieval_k=2,
        max_context_documents=1,
        max_context_chars=200,
    )

    result = pipeline.run(
        "duplicate invoice question"
    )

    assert result.abstained is True
    assert result.cited_document_ids == ()
    assert result.answer == "Insufficient evidence."


def test_rag_rejects_context_budget_that_fits_no_document() -> None:
    """Context assembly must respect the declared character budget."""
    pipeline = RetrievalAugmentedGenerator(
        retriever=build_retriever(),
        generator=CitingGenerator(),
        retrieval_k=2,
        max_context_documents=2,
        max_context_chars=5,
    )

    with pytest.raises(
        ValueError,
        match="context budget",
    ):
        pipeline.run(
            "duplicate invoice question"
        )
