"""Tests for provider-neutral semantic retrieval."""

from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from vait.retrieval.in_memory import InMemoryCosineRetriever
from vait.retrieval.models import TextDocument


class StaticEmbeddingEncoder:
    """Deterministic encoder for retrieval contract tests."""

    def __init__(
        self,
        vectors: dict[str, tuple[float, ...]],
    ) -> None:
        """Store predefined vectors."""
        self._vectors = vectors

    @property
    def implementation_id(self) -> str:
        """Return the test encoder identifier."""
        return "static-test-encoder-v1"

    def encode(
        self,
        texts: Sequence[str],
    ) -> NDArray[np.float64]:
        """Return predefined vectors for supplied texts."""
        return np.asarray(
            [
                self._vectors[text]
                for text in texts
            ],
            dtype=np.float64,
        )


def test_retriever_ranks_semantically_closest_document() -> None:
    """Cosine retrieval should return the closest vector first."""
    encoder = StaticEmbeddingEncoder(
        {
            "Duplicate invoices require review.": (
                1.0,
                0.0,
                0.0,
            ),
            "Bank detail changes require escalation.": (
                0.0,
                1.0,
                0.0,
            ),
            "Purchase order mismatches require review.": (
                0.0,
                0.0,
                1.0,
            ),
            "duplicate payment policy": (
                0.9,
                0.1,
                0.0,
            ),
        }
    )

    retriever = InMemoryCosineRetriever(
        encoder=encoder,
        documents=(
            TextDocument(
                document_id="duplicate-policy",
                text="Duplicate invoices require review.",
            ),
            TextDocument(
                document_id="bank-policy",
                text="Bank detail changes require escalation.",
            ),
            TextDocument(
                document_id="po-policy",
                text="Purchase order mismatches require review.",
            ),
        ),
    )

    hits = retriever.retrieve(
        "duplicate payment policy",
        top_k=2,
    )

    assert len(hits) == 2
    assert hits[0].document.document_id == "duplicate-policy"
    assert hits[0].rank == 1
    assert hits[0].score > hits[1].score


def test_retriever_exposes_index_metadata() -> None:
    """Retriever should expose reproducibility-relevant metadata."""
    encoder = StaticEmbeddingEncoder(
        {
            "policy": (
                1.0,
                0.0,
            ),
        }
    )

    retriever = InMemoryCosineRetriever(
        encoder=encoder,
        documents=(
            TextDocument(
                document_id="policy-1",
                text="policy",
            ),
        ),
    )

    assert (
        retriever.encoder_implementation_id
        == "static-test-encoder-v1"
    )
    assert retriever.document_count == 1
    assert retriever.embedding_dimension == 2


def test_retriever_rejects_duplicate_document_ids() -> None:
    """Document identifiers must remain unambiguous."""
    encoder = StaticEmbeddingEncoder(
        {
            "first": (
                1.0,
                0.0,
            ),
            "second": (
                0.0,
                1.0,
            ),
        }
    )

    with pytest.raises(
        ValueError,
        match="identifiers must be unique",
    ):
        InMemoryCosineRetriever(
            encoder=encoder,
            documents=(
                TextDocument(
                    document_id="duplicate",
                    text="first",
                ),
                TextDocument(
                    document_id="duplicate",
                    text="second",
                ),
            ),
        )


def test_retriever_rejects_zero_vector() -> None:
    """Zero-magnitude vectors cannot define cosine similarity."""
    encoder = StaticEmbeddingEncoder(
        {
            "policy": (
                0.0,
                0.0,
            ),
        }
    )

    with pytest.raises(
        ValueError,
        match="non-zero magnitude",
    ):
        InMemoryCosineRetriever(
            encoder=encoder,
            documents=(
                TextDocument(
                    document_id="policy-1",
                    text="policy",
                ),
            ),
        )


def test_retriever_rejects_invalid_top_k() -> None:
    """Retrieval depth must be positive."""
    encoder = StaticEmbeddingEncoder(
        {
            "policy": (
                1.0,
                0.0,
            ),
            "query": (
                1.0,
                0.0,
            ),
        }
    )

    retriever = InMemoryCosineRetriever(
        encoder=encoder,
        documents=(
            TextDocument(
                document_id="policy-1",
                text="policy",
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="top_k",
    ):
        retriever.retrieve(
            "query",
            top_k=0,
        )
