"""Grounded retrieval-augmented generation pipeline."""

from collections.abc import Sequence

from vait.rag.base import GroundedGenerator
from vait.rag.models import (
    RAGContextDocument,
    RAGGeneration,
    RAGGenerationRequest,
    RAGResult,
)
from vait.retrieval.in_memory import InMemoryCosineRetriever
from vait.retrieval.models import RetrievalHit


class RetrievalAugmentedGenerator:
    """Combine semantic retrieval with grounded generation."""

    def __init__(
        self,
        *,
        retriever: InMemoryCosineRetriever,
        generator: GroundedGenerator,
        retrieval_k: int = 5,
        max_context_documents: int = 3,
        max_context_chars: int = 4000,
    ) -> None:
        """Configure retrieval and bounded context assembly."""
        if retrieval_k <= 0:
            raise ValueError(
                "retrieval_k must be greater than zero."
            )

        if max_context_documents <= 0:
            raise ValueError(
                "max_context_documents must be greater than zero."
            )

        if max_context_documents > retrieval_k:
            raise ValueError(
                "max_context_documents cannot exceed retrieval_k."
            )

        if max_context_chars <= 0:
            raise ValueError(
                "max_context_chars must be greater than zero."
            )

        self._retriever = retriever
        self._generator = generator
        self._retrieval_k = retrieval_k
        self._max_context_documents = max_context_documents
        self._max_context_chars = max_context_chars

    def run(
        self,
        query: str,
    ) -> RAGResult:
        """Retrieve evidence, generate, and validate grounding."""
        if not query.strip():
            raise ValueError(
                "RAG query must not be empty."
            )

        hits = self._retriever.retrieve(
            query,
            top_k=self._retrieval_k,
        )

        context_documents = _assemble_context(
            hits,
            max_documents=self._max_context_documents,
            max_chars=self._max_context_chars,
        )

        request = RAGGenerationRequest(
            query=query,
            context_documents=context_documents,
        )

        generation = self._generator.generate(
            request
        )

        _validate_generation(
            generation,
            context_documents=context_documents,
        )

        return RAGResult(
            query=query,
            answer=generation.answer.strip(),
            context_documents=context_documents,
            cited_document_ids=(
                generation.cited_document_ids
            ),
            generator_implementation_id=(
                self._generator.implementation_id
            ),
            abstained=generation.abstained,
        )


def _assemble_context(
    hits: Sequence[RetrievalHit],
    *,
    max_documents: int,
    max_chars: int,
) -> tuple[RAGContextDocument, ...]:
    """Build bounded context without silently truncating documents."""
    selected: list[RAGContextDocument] = []
    used_chars = 0

    for hit in hits:
        if len(selected) >= max_documents:
            break

        text = hit.document.text.strip()

        if not text:
            continue

        if used_chars + len(text) > max_chars:
            continue

        selected.append(
            RAGContextDocument(
                document_id=hit.document.document_id,
                text=text,
                score=hit.score,
                rank=hit.rank,
            )
        )
        used_chars += len(text)

    if not selected:
        raise ValueError(
            "No retrieved document fits the context budget."
        )

    return tuple(selected)


def _validate_generation(
    generation: RAGGeneration,
    *,
    context_documents: Sequence[RAGContextDocument],
) -> None:
    """Reject structurally ungrounded generator output."""
    if not generation.answer.strip():
        raise ValueError(
            "Generator answer must not be empty."
        )

    cited_ids = generation.cited_document_ids

    if len(set(cited_ids)) != len(cited_ids):
        raise ValueError(
            "Generator citations must not contain duplicates."
        )

    context_ids = {
        document.document_id
        for document in context_documents
    }

    unknown_ids = set(cited_ids) - context_ids

    if unknown_ids:
        raise ValueError(
            "Generator cited documents that were not supplied "
            f"in context: {sorted(unknown_ids)}"
        )

    if generation.abstained:
        if cited_ids:
            raise ValueError(
                "Abstained generations must not claim citations."
            )
        return

    if not cited_ids:
        raise ValueError(
            "Grounded generation requires at least one citation."
        )
