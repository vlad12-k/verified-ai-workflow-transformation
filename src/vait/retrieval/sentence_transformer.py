"""Local Sentence Transformer embedding adapter."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddingEncoder:
    """Encode text locally with a pinned Sentence Transformer model."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        device: str = "cpu",
        local_files_only: bool = False,
    ) -> None:
        """Load one explicitly versioned local embedding model."""
        if not model_id.strip():
            raise ValueError("model_id must not be empty.")

        if not revision.strip():
            raise ValueError("revision must not be empty.")

        if not device.strip():
            raise ValueError("device must not be empty.")

        self._model_id = model_id
        self._revision = revision
        self._device = device

        self._model = SentenceTransformer(
            model_id,
            revision=revision,
            device=device,
            local_files_only=local_files_only,
        )

        dimension = (
            self._model.get_embedding_dimension()
        )

        if dimension is None or dimension <= 0:
            raise ValueError(
                "Embedding model must expose a positive dimension."
            )

        self._embedding_dimension = int(dimension)

    @property
    def implementation_id(self) -> str:
        """Return a model-and-revision-specific implementation ID."""
        return (
            f"sentence-transformers:"
            f"{self._model_id}@{self._revision}"
        )

    @property
    def model_id(self) -> str:
        """Return the Hugging Face model identifier."""
        return self._model_id

    @property
    def revision(self) -> str:
        """Return the pinned model revision."""
        return self._revision

    @property
    def device(self) -> str:
        """Return the configured inference device."""
        return self._device

    @property
    def embedding_dimension(self) -> int:
        """Return the dense-vector dimension."""
        return self._embedding_dimension

    def encode(
        self,
        texts: Sequence[str],
    ) -> NDArray[np.float64]:
        """Encode supplied texts as a validated dense matrix."""
        if not texts:
            return np.empty(
                (
                    0,
                    self._embedding_dimension,
                ),
                dtype=np.float64,
            )

        raw_embeddings = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )

        embeddings: NDArray[np.float64] = np.asarray(
            raw_embeddings,
            dtype=np.float64,
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embedding model output must be two-dimensional."
            )

        if embeddings.shape[0] != len(texts):
            raise ValueError(
                "Embedding row count does not match input count."
            )

        if (
            embeddings.shape[1]
            != self._embedding_dimension
        ):
            raise ValueError(
                "Embedding dimension changed during inference."
            )

        if not np.isfinite(embeddings).all():
            raise ValueError(
                "Embedding output contains non-finite values."
            )

        return embeddings
