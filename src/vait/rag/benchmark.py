"""Evaluation models and metrics for grounded RAG."""

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, model_validator

from vait.rag.models import RAGResult


class RAGRunner(Protocol):
    """Run one grounded RAG query."""

    def run(
        self,
        query: str,
    ) -> RAGResult:
        """Return one validated RAG result."""
        ...


class RAGBenchmarkCase(BaseModel):
    """One answerable or abstention benchmark case."""

    case_id: str
    query: str
    relevant_document_ids: tuple[str, ...]
    expected_abstain: bool

    @model_validator(mode="after")
    def validate_expectation(
        self,
    ) -> "RAGBenchmarkCase":
        """Require relevance labels only for answerable cases."""
        if self.expected_abstain:
            if self.relevant_document_ids:
                raise ValueError(
                    "Abstention cases must not declare relevant documents."
                )
        elif not self.relevant_document_ids:
            raise ValueError(
                "Answerable cases require relevant documents."
            )

        return self


class RAGBenchmarkDataset(BaseModel):
    """Versioned development benchmark for grounded RAG."""

    benchmark_id: str
    version: str
    description: str
    source_retrieval_benchmark_id: str
    source_retrieval_benchmark_version: str
    cases: tuple[RAGBenchmarkCase, ...]

    @model_validator(mode="after")
    def validate_dataset(
        self,
    ) -> "RAGBenchmarkDataset":
        """Require unique IDs and both benchmark case classes."""
        if not self.cases:
            raise ValueError(
                "RAG benchmark requires cases."
            )

        case_ids = [
            case.case_id
            for case in self.cases
        ]

        if len(set(case_ids)) != len(case_ids):
            raise ValueError(
                "RAG benchmark case IDs must be unique."
            )

        if not any(
            not case.expected_abstain
            for case in self.cases
        ):
            raise ValueError(
                "RAG benchmark requires answerable cases."
            )

        if not any(
            case.expected_abstain
            for case in self.cases
        ):
            raise ValueError(
                "RAG benchmark requires abstention cases."
            )

        return self


class RAGCaseEvaluation(BaseModel):
    """Observed result and correctness for one benchmark case."""

    case_id: str
    expected_abstain: bool
    observed_abstain: bool
    relevant_document_ids: tuple[str, ...]
    context_document_ids: tuple[str, ...]
    cited_document_ids: tuple[str, ...]
    evidence_available: bool
    citation_correct: bool
    success: bool


class RAGEvaluationReport(BaseModel):
    """Aggregate grounded RAG quality metrics."""

    benchmark_id: str
    benchmark_version: str
    generator_implementation_id: str
    case_count: int
    answerable_case_count: int
    abstention_case_count: int
    evidence_coverage: float
    citation_accuracy: float
    abstention_accuracy: float
    answerable_success_rate: float
    overall_success_rate: float
    results: tuple[RAGCaseEvaluation, ...]


def load_rag_benchmark(
    path: str | Path,
) -> RAGBenchmarkDataset:
    """Load and validate a RAG benchmark JSON file."""
    return RAGBenchmarkDataset.model_validate_json(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


def validate_rag_benchmark_documents(
    dataset: RAGBenchmarkDataset,
    *,
    known_document_ids: set[str],
) -> None:
    """Validate relevance references against the source corpus."""
    for case in dataset.cases:
        unknown_ids = (
            set(case.relevant_document_ids)
            - known_document_ids
        )

        if unknown_ids:
            raise ValueError(
                f"Case {case.case_id} references unknown "
                f"documents: {sorted(unknown_ids)}"
            )


def evaluate_rag(
    dataset: RAGBenchmarkDataset,
    pipeline: RAGRunner,
) -> RAGEvaluationReport:
    """Evaluate evidence, citation, and abstention behaviour."""
    results: list[RAGCaseEvaluation] = []

    answerable_count = sum(
        not case.expected_abstain
        for case in dataset.cases
    )
    abstention_count = (
        len(dataset.cases)
        - answerable_count
    )

    evidence_successes = 0
    citation_correct_outputs = 0
    non_abstained_outputs = 0
    abstention_matches = 0
    answerable_successes = 0
    total_successes = 0

    generator_id: str | None = None

    for case in dataset.cases:
        result = pipeline.run(
            case.query
        )

        if generator_id is None:
            generator_id = (
                result.generator_implementation_id
            )
        elif (
            result.generator_implementation_id
            != generator_id
        ):
            raise ValueError(
                "RAG generator implementation changed during evaluation."
            )

        relevant_ids = set(
            case.relevant_document_ids
        )
        context_ids = tuple(
            document.document_id
            for document in result.context_documents
        )
        cited_ids = result.cited_document_ids

        evidence_available = bool(
            relevant_ids
            & set(context_ids)
        )

        if not case.expected_abstain:
            evidence_successes += int(
                evidence_available
            )

        citation_correct = (
            not result.abstained
            and bool(cited_ids)
            and set(cited_ids).issubset(
                relevant_ids
            )
        )

        if not result.abstained:
            non_abstained_outputs += 1
            citation_correct_outputs += int(
                citation_correct
            )

        abstention_matches += int(
            result.abstained
            == case.expected_abstain
        )

        if case.expected_abstain:
            success = result.abstained
        else:
            success = (
                not result.abstained
                and evidence_available
                and citation_correct
            )
            answerable_successes += int(
                success
            )

        total_successes += int(
            success
        )

        results.append(
            RAGCaseEvaluation(
                case_id=case.case_id,
                expected_abstain=(
                    case.expected_abstain
                ),
                observed_abstain=(
                    result.abstained
                ),
                relevant_document_ids=(
                    case.relevant_document_ids
                ),
                context_document_ids=(
                    context_ids
                ),
                cited_document_ids=cited_ids,
                evidence_available=(
                    evidence_available
                ),
                citation_correct=(
                    citation_correct
                ),
                success=success,
            )
        )

    if generator_id is None:
        raise ValueError(
            "RAG evaluation produced no results."
        )

    total_count = len(dataset.cases)

    citation_accuracy = (
        citation_correct_outputs
        / non_abstained_outputs
        if non_abstained_outputs
        else 1.0
    )

    return RAGEvaluationReport(
        benchmark_id=dataset.benchmark_id,
        benchmark_version=dataset.version,
        generator_implementation_id=generator_id,
        case_count=total_count,
        answerable_case_count=answerable_count,
        abstention_case_count=abstention_count,
        evidence_coverage=(
            evidence_successes
            / answerable_count
        ),
        citation_accuracy=citation_accuracy,
        abstention_accuracy=(
            abstention_matches
            / total_count
        ),
        answerable_success_rate=(
            answerable_successes
            / answerable_count
        ),
        overall_success_rate=(
            total_successes
            / total_count
        ),
        results=tuple(results),
    )
