"""Deterministic in-memory cosine retrieval."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from vait.retrieval.base import EmbeddingEncoder
from vait.retrieval.models import RetrievalHit, TextDocument


class InMemoryCosineRetriever:
    """Retrieve documents using normalized dense-vector similarity."""

    def __init__(
        self,
        encoder: EmbeddingEncoder,
        documents: Sequence[TextDocument],
    ) -> None:
        """Encode and index the supplied document collection."""
        if not documents:
            raise ValueError(
                "At least one document is required for retrieval."
            )

        document_ids = [
            document.document_id
            for document in documents
        ]

        if len(set(document_ids)) != len(document_ids):
            raise ValueError(
                "Document identifiers must be unique."
            )

        self._encoder = encoder
        self._documents = tuple(documents)

        embeddings = encoder.encode(
            [
                document.text
                for document in self._documents
            ]
        )

        _validate_embedding_matrix(
            embeddings,
            expected_rows=len(self._documents),
            label="document",
        )

        self._document_embeddings = _normalize_rows(
            embeddings
        )

    @property
    def encoder_implementation_id(self) -> str:
        """Return the embedding implementation identifier."""
        return self._encoder.implementation_id

    @property
    def document_count(self) -> int:
        """Return the number of indexed documents."""
        return len(self._documents)

    @property
    def embedding_dimension(self) -> int:
        """Return the indexed vector dimension."""
        return int(
            self._document_embeddings.shape[1]
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> tuple[RetrievalHit, ...]:
        """Return deterministic cosine-ranked documents."""
        if not query.strip():
            raise ValueError(
                "Retrieval query must not be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        query_embeddings = self._encoder.encode(
            [query]
        )

        _validate_embedding_matrix(
            query_embeddings,
            expected_rows=1,
            label="query",
        )

        if (
            query_embeddings.shape[1]
            != self.embedding_dimension
        ):
            raise ValueError(
                "Query and document embedding dimensions differ."
            )

        normalized_query = _normalize_rows(
            query_embeddings
        )[0]

        scores = (
            self._document_embeddings
            @ normalized_query
        )

        ordered_indices = sorted(
            range(len(self._documents)),
            key=lambda index: (
                -float(scores[index]),
                self._documents[index].document_id,
            ),
        )

        selected_indices = ordered_indices[
            : min(
                top_k,
                len(ordered_indices),
            )
        ]

        return tuple(
            RetrievalHit(
                document=self._documents[index],
                score=float(scores[index]),
                rank=rank,
            )
            for rank, index in enumerate(
                selected_indices,
                start=1,
            )
        )


def _validate_embedding_matrix(
    embeddings: NDArray[np.float64],
    *,
    expected_rows: int,
    label: str,
) -> None:
    """Validate encoder output before similarity operations."""
    if embeddings.ndim != 2:
        raise ValueError(
            f"{label} embeddings must be two-dimensional."
        )

    if embeddings.shape[0] != expected_rows:
        raise ValueError(
            f"{label} embedding row count does not match input count."
        )

    if embeddings.shape[1] == 0:
        raise ValueError(
            f"{label} embeddings must have at least one dimension."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            f"{label} embeddings must contain only finite values."
        )


def _normalize_rows(
    embeddings: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return row-normalized vectors for cosine similarity."""
    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    if np.any(norms == 0.0):
        raise ValueError(
            "Embedding vectors must have non-zero magnitude."
        )

    normalized: NDArray[np.float64] = (
        embeddings / norms
    )

    return normalized
