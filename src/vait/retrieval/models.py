"""Typed models for semantic retrieval."""

from dataclasses import dataclass, field

from pydantic import JsonValue


@dataclass(frozen=True)
class TextDocument:
    """One retrievable text document."""

    document_id: str
    text: str
    metadata: dict[str, JsonValue] = field(
        default_factory=dict,
    )


@dataclass(frozen=True)
class RetrievalHit:
    """One ranked semantic retrieval result."""

    document: TextDocument
    score: float
    rank: int
