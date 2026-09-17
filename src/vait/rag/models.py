"""Typed models for retrieval-augmented generation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RAGContextDocument:
    """One retrieved document supplied to a generator."""

    document_id: str
    text: str
    score: float
    rank: int


@dataclass(frozen=True)
class RAGGenerationRequest:
    """Structured generation request with bounded evidence."""

    query: str
    context_documents: tuple[RAGContextDocument, ...]


@dataclass(frozen=True)
class RAGGeneration:
    """Raw generator response before grounding validation."""

    answer: str
    cited_document_ids: tuple[str, ...]
    abstained: bool = False


@dataclass(frozen=True)
class RAGResult:
    """Validated grounded generation result."""

    query: str
    answer: str
    context_documents: tuple[RAGContextDocument, ...]
    cited_document_ids: tuple[str, ...]
    generator_implementation_id: str
    abstained: bool
