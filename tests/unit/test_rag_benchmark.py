"""Tests for grounded RAG benchmark evaluation."""

from vait.rag.benchmark import (
    RAGBenchmarkCase,
    RAGBenchmarkDataset,
    evaluate_rag,
    load_rag_benchmark,
)
from vait.rag.models import (
    RAGContextDocument,
    RAGResult,
)


class DeterministicRAGRunner:
    """Small deterministic runner for metric tests."""

    def run(
        self,
        query: str,
    ) -> RAGResult:
        """Return one answerable or abstaining result."""
        if query == "answerable":
            context = (
                RAGContextDocument(
                    document_id="policy-a",
                    text="Policy A.",
                    score=0.9,
                    rank=1,
                ),
            )

            return RAGResult(
                query=query,
                answer="Policy A.",
                context_documents=context,
                cited_document_ids=(
                    "policy-a",
                ),
                generator_implementation_id=(
                    "deterministic-test-generator"
                ),
                abstained=False,
            )

        return RAGResult(
            query=query,
            answer="Insufficient evidence.",
            context_documents=(
                RAGContextDocument(
                    document_id="policy-a",
                    text="Policy A.",
                    score=0.1,
                    rank=1,
                ),
            ),
            cited_document_ids=(),
            generator_implementation_id=(
                "deterministic-test-generator"
            ),
            abstained=True,
        )


def test_repository_rag_benchmark_loads() -> None:
    """Repository RAG benchmark should remain structurally valid."""
    dataset = load_rag_benchmark(
        "datasets/ap_policy_rag/v0.1/benchmark.json"
    )

    assert dataset.benchmark_id == "ap-policy-rag"
    assert dataset.version == "0.1"
    assert len(dataset.cases) == 14
    assert sum(
        not case.expected_abstain
        for case in dataset.cases
    ) == 8
    assert sum(
        case.expected_abstain
        for case in dataset.cases
    ) == 6


def test_rag_metrics_measure_grounding_and_abstention() -> None:
    """Perfect deterministic behaviour should score one on all metrics."""
    dataset = RAGBenchmarkDataset(
        benchmark_id="test-rag",
        version="1",
        description="Deterministic metric test.",
        source_retrieval_benchmark_id="test",
        source_retrieval_benchmark_version="1",
        cases=(
            RAGBenchmarkCase(
                case_id="answerable",
                query="answerable",
                relevant_document_ids=(
                    "policy-a",
                ),
                expected_abstain=False,
            ),
            RAGBenchmarkCase(
                case_id="unanswerable",
                query="unanswerable",
                relevant_document_ids=(),
                expected_abstain=True,
            ),
        ),
    )

    report = evaluate_rag(
        dataset,
        DeterministicRAGRunner(),
    )

    assert report.case_count == 2
    assert report.evidence_coverage == 1.0
    assert report.citation_accuracy == 1.0
    assert report.abstention_accuracy == 1.0
    assert report.answerable_success_rate == 1.0
    assert report.overall_success_rate == 1.0
    assert all(
        result.success
        for result in report.results
    )
