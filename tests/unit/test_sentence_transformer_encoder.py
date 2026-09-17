"""Tests for the local Sentence Transformer embedding adapter."""

from unittest.mock import patch

import numpy as np

from vait.retrieval.sentence_transformer import (
    SentenceTransformerEmbeddingEncoder,
)


class FakeSentenceTransformer:
    """Small deterministic stand-in for adapter tests."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: str,
        device: str,
        local_files_only: bool,
    ) -> None:
        """Capture model loading configuration."""
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.local_files_only = local_files_only

    def get_embedding_dimension(
        self,
    ) -> int:
        """Expose a fixed embedding dimension."""
        return 3

    def encode(
        self,
        texts: list[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> np.ndarray:
        """Return deterministic embeddings."""
        assert convert_to_numpy is True
        assert normalize_embeddings is False
        assert show_progress_bar is False

        return np.asarray(
            [
                (
                    float(index + 1),
                    1.0,
                    0.5,
                )
                for index, _ in enumerate(texts)
            ],
            dtype=np.float32,
        )


def test_encoder_exposes_reproducible_model_metadata() -> None:
    """Model ID and revision should form a stable implementation ID."""
    with patch(
        "vait.retrieval.sentence_transformer.SentenceTransformer",
        FakeSentenceTransformer,
    ):
        encoder = SentenceTransformerEmbeddingEncoder(
            model_id="example/model",
            revision="abc123",
            device="cpu",
            local_files_only=True,
        )

    assert encoder.model_id == "example/model"
    assert encoder.revision == "abc123"
    assert encoder.device == "cpu"
    assert encoder.embedding_dimension == 3
    assert (
        encoder.implementation_id
        == "sentence-transformers:example/model@abc123"
    )


def test_encoder_returns_float64_matrix() -> None:
    """Adapter output should satisfy the provider-neutral protocol."""
    with patch(
        "vait.retrieval.sentence_transformer.SentenceTransformer",
        FakeSentenceTransformer,
    ):
        encoder = SentenceTransformerEmbeddingEncoder(
            model_id="example/model",
            revision="abc123",
        )

        embeddings = encoder.encode(
            (
                "first text",
                "second text",
            )
        )

    assert embeddings.shape == (2, 3)
    assert embeddings.dtype == np.float64
    assert np.isfinite(embeddings).all()


def test_encoder_handles_empty_input() -> None:
    """Empty input should preserve the model dimension."""
    with patch(
        "vait.retrieval.sentence_transformer.SentenceTransformer",
        FakeSentenceTransformer,
    ):
        encoder = SentenceTransformerEmbeddingEncoder(
            model_id="example/model",
            revision="abc123",
        )

        embeddings = encoder.encode(())

    assert embeddings.shape == (0, 3)
    assert embeddings.dtype == np.float64
