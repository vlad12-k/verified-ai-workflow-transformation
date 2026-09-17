"""Semantic-guarded generation with deterministic fallback."""

from vait.rag.base import GroundedGenerator
from vait.rag.models import (
    RAGContextDocument,
    RAGGeneration,
    RAGGenerationRequest,
)
from vait.rag.semantic_guard import assess_semantic_drift


class SemanticGuardedGenerator:
    """Suppress risky generated answers and use a grounded fallback."""

    def __init__(
        self,
        *,
        primary: GroundedGenerator,
        fallback: GroundedGenerator,
    ) -> None:
        """Configure primary generation and deterministic fallback."""
        self._primary = primary
        self._fallback = fallback

    @property
    def implementation_id(self) -> str:
        """Return a stable guarded-generator identifier."""
        return (
            "semantic-guarded-generator@1.0.0:"
            f"primary={self._primary.implementation_id}:"
            f"fallback={self._fallback.implementation_id}"
        )

    def generate(
        self,
        request: RAGGenerationRequest,
    ) -> RAGGeneration:
        """Return primary output only when bounded guard checks pass."""
        primary_generation = self._primary.generate(
            request
        )

        if primary_generation.abstained:
            return primary_generation

        cited_context = _resolve_cited_context(
            request=request,
            generation=primary_generation,
        )

        if cited_context is None:
            # Do not mask malformed grounding metadata.
            # The outer RAG pipeline remains responsible for
            # structural citation validation.
            return primary_generation

        if len(cited_context) != 1:
            return self._generate_fallback(
                request=request,
                context_documents=cited_context,
            )

        assessment = assess_semantic_drift(
            context_text=cited_context[0].text,
            answer_text=primary_generation.answer,
        )

        if assessment.passed:
            return primary_generation

        return self._generate_fallback(
            request=request,
            context_documents=cited_context,
        )

    def _generate_fallback(
        self,
        *,
        request: RAGGenerationRequest,
        context_documents: tuple[
            RAGContextDocument,
            ...,
        ],
    ) -> RAGGeneration:
        """Generate fallback output from cited evidence only."""
        fallback_request = RAGGenerationRequest(
            query=request.query,
            context_documents=context_documents,
        )

        return self._fallback.generate(
            fallback_request
        )


def _resolve_cited_context(
    *,
    request: RAGGenerationRequest,
    generation: RAGGeneration,
) -> tuple[RAGContextDocument, ...] | None:
    """Resolve valid cited documents without hiding citation errors."""
    cited_ids = generation.cited_document_ids

    if not cited_ids:
        return None

    if len(set(cited_ids)) != len(cited_ids):
        return None

    context_by_id = {
        document.document_id: document
        for document in request.context_documents
    }

    if any(
        document_id not in context_by_id
        for document_id in cited_ids
    ):
        return None

    return tuple(
        context_by_id[document_id]
        for document_id in cited_ids
    )
