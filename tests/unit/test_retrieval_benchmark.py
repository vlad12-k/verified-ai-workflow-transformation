"""Tests for semantic retrieval benchmark evaluation."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from vait.retrieval.benchmark import (
    RetrievalBenchmarkDataset,
    RetrievalBenchmarkDocument,
    RetrievalBenchmarkQuery,
    evaluate_retriever,
    load_retrieval_benchmark,
)
from vait.retrieval.in_memory import InMemoryCosineRetriever


class BenchmarkTestEncoder:
    """Deterministic embedding encoder for metric tests."""

    @property
    def implementation_id(self) -> str:
        """Return the deterministic test identifier."""
        return "benchmark-test-encoder-v1"

    def encode(
        self,
        texts: Sequence[str],
    ) -> NDArray[np.float64]:
        """Map known texts to deterministic vectors."""
        vectors = {
            "document-a": (1.0, 0.0, 0.0),
            "document-b": (0.0, 1.0, 0.0),
            "document-c": (0.0, 0.0, 1.0),
            "query-a": (1.0, 0.0, 0.0),
            "query-b": (0.1, 0.9, 0.0),
        }

        return np.asarray(
            [
                vectors[text]
                for text in texts
            ],
            dtype=np.float64,
        )


def test_repository_policy_benchmark_loads() -> None:
    """Versioned AP policy benchmark should be structurally valid."""
    dataset = load_retrieval_benchmark(
        "datasets/ap_policy_context/v0.1/benchmark.json"
    )

    assert dataset.benchmark_id == "ap-policy-context"
    assert dataset.version == "0.1"
    assert len(dataset.documents) == 12
    assert len(dataset.queries) == 15


def test_retrieval_metrics_measure_ranked_relevance() -> None:
    """Recall and MRR should reflect expected retrieval rankings."""
    dataset = RetrievalBenchmarkDataset(
        benchmark_id="test-retrieval",
        version="1",
        description="Deterministic metric test.",
        documents=(
            RetrievalBenchmarkDocument(
                document_id="a",
                text="document-a",
            ),
            RetrievalBenchmarkDocument(
                document_id="b",
                text="document-b",
            ),
            RetrievalBenchmarkDocument(
                document_id="c",
                text="document-c",
            ),
        ),
        queries=(
            RetrievalBenchmarkQuery(
                query_id="query-a",
                text="query-a",
                relevant_document_ids=("a",),
            ),
            RetrievalBenchmarkQuery(
                query_id="query-b",
                text="query-b",
                relevant_document_ids=("b",),
            ),
        ),
    )

    retriever = InMemoryCosineRetriever(
        encoder=BenchmarkTestEncoder(),
        documents=dataset.text_documents(),
    )

    report = evaluate_retriever(
        dataset,
        retriever,
    )

    assert report.query_count == 2
    assert report.recall_at_1 == 1.0
    assert report.recall_at_3 == 1.0
    assert report.mean_reciprocal_rank == 1.0
    assert all(
        result.first_relevant_rank == 1
        for result in report.results
    )
