"""Evaluation models and metrics for semantic retrieval."""

from pathlib import Path

from pydantic import BaseModel, Field, JsonValue, model_validator

from vait.retrieval.in_memory import InMemoryCosineRetriever
from vait.retrieval.models import TextDocument


class RetrievalBenchmarkDocument(BaseModel):
    """One indexed benchmark document."""

    document_id: str
    text: str
    metadata: dict[str, JsonValue] = Field(
        default_factory=dict,
    )


class RetrievalBenchmarkQuery(BaseModel):
    """One retrieval query with declared relevant documents."""

    query_id: str
    text: str
    relevant_document_ids: tuple[str, ...]


class RetrievalBenchmarkDataset(BaseModel):
    """Versioned semantic retrieval benchmark."""

    benchmark_id: str
    version: str
    description: str
    documents: tuple[RetrievalBenchmarkDocument, ...]
    queries: tuple[RetrievalBenchmarkQuery, ...]

    @model_validator(mode="after")
    def validate_dataset(
        self,
    ) -> "RetrievalBenchmarkDataset":
        """Validate identifiers and relevance references."""
        document_ids = [
            document.document_id
            for document in self.documents
        ]
        query_ids = [
            query.query_id
            for query in self.queries
        ]

        if not self.documents:
            raise ValueError(
                "Retrieval benchmark requires documents."
            )

        if not self.queries:
            raise ValueError(
                "Retrieval benchmark requires queries."
            )

        if len(set(document_ids)) != len(document_ids):
            raise ValueError(
                "Retrieval benchmark document IDs must be unique."
            )

        if len(set(query_ids)) != len(query_ids):
            raise ValueError(
                "Retrieval benchmark query IDs must be unique."
            )

        known_document_ids = set(document_ids)

        for query in self.queries:
            if not query.relevant_document_ids:
                raise ValueError(
                    f"Query {query.query_id} has no relevant documents."
                )

            unknown_ids = (
                set(query.relevant_document_ids)
                - known_document_ids
            )

            if unknown_ids:
                raise ValueError(
                    f"Query {query.query_id} references unknown "
                    f"documents: {sorted(unknown_ids)}"
                )

        return self

    def text_documents(
        self,
    ) -> tuple[TextDocument, ...]:
        """Convert benchmark documents to retriever documents."""
        return tuple(
            TextDocument(
                document_id=document.document_id,
                text=document.text,
                metadata=document.metadata,
            )
            for document in self.documents
        )


class RetrievalQueryResult(BaseModel):
    """Observed ranking for one benchmark query."""

    query_id: str
    relevant_document_ids: tuple[str, ...]
    retrieved_document_ids: tuple[str, ...]
    first_relevant_rank: int | None


class RetrievalEvaluationReport(BaseModel):
    """Aggregate semantic retrieval quality metrics."""

    benchmark_id: str
    benchmark_version: str
    encoder_implementation_id: str
    query_count: int
    recall_at_1: float
    recall_at_3: float
    mean_reciprocal_rank: float
    results: tuple[RetrievalQueryResult, ...]


def load_retrieval_benchmark(
    path: str | Path,
) -> RetrievalBenchmarkDataset:
    """Load and validate a retrieval benchmark JSON file."""
    return RetrievalBenchmarkDataset.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def evaluate_retriever(
    dataset: RetrievalBenchmarkDataset,
    retriever: InMemoryCosineRetriever,
) -> RetrievalEvaluationReport:
    """Evaluate retrieval quality using recall and reciprocal rank."""
    results: list[RetrievalQueryResult] = []
    recall_at_1_total = 0.0
    recall_at_3_total = 0.0
    reciprocal_rank_total = 0.0

    for query in dataset.queries:
        hits = retriever.retrieve(
            query.text,
            top_k=min(
                3,
                retriever.document_count,
            ),
        )

        retrieved_ids = tuple(
            hit.document.document_id
            for hit in hits
        )
        relevant_ids = set(
            query.relevant_document_ids
        )

        recall_at_1_total += (
            len(
                relevant_ids
                & set(retrieved_ids[:1])
            )
            / len(relevant_ids)
        )

        recall_at_3_total += (
            len(
                relevant_ids
                & set(retrieved_ids[:3])
            )
            / len(relevant_ids)
        )

        first_relevant_rank: int | None = None

        for rank, document_id in enumerate(
            retrieved_ids,
            start=1,
        ):
            if document_id in relevant_ids:
                first_relevant_rank = rank
                reciprocal_rank_total += 1.0 / rank
                break

        results.append(
            RetrievalQueryResult(
                query_id=query.query_id,
                relevant_document_ids=(
                    query.relevant_document_ids
                ),
                retrieved_document_ids=retrieved_ids,
                first_relevant_rank=first_relevant_rank,
            )
        )

    query_count = len(dataset.queries)

    return RetrievalEvaluationReport(
        benchmark_id=dataset.benchmark_id,
        benchmark_version=dataset.version,
        encoder_implementation_id=(
            retriever.encoder_implementation_id
        ),
        query_count=query_count,
        recall_at_1=(
            recall_at_1_total / query_count
        ),
        recall_at_3=(
            recall_at_3_total / query_count
        ),
        mean_reciprocal_rank=(
            reciprocal_rank_total / query_count
        ),
        results=tuple(results),
    )
