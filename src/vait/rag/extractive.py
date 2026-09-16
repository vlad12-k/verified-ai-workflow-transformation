"""Deterministic extractive generator for grounded RAG baselines."""

from vait.rag.models import (
    RAGGeneration,
    RAGGenerationRequest,
)


class ExtractiveGroundedGenerator:
    """Return the highest-ranked evidence document as a grounded answer."""

    def __init__(
        self,
        *,
        minimum_score: float = 0.0,
        abstention_text: str = "Insufficient evidence.",
    ) -> None:
        """Configure deterministic evidence selection and abstention."""
        if not -1.0 <= minimum_score <= 1.0:
            raise ValueError(
                "minimum_score must be between -1.0 and 1.0."
            )

        if not abstention_text.strip():
            raise ValueError(
                "abstention_text must not be empty."
            )

        self._minimum_score = minimum_score
        self._abstention_text = abstention_text.strip()

    @property
    def implementation_id(self) -> str:
        """Return a stable implementation identifier."""
        return (
            "extractive-grounded-generator@1.0.0:"
            f"minimum-score={self._minimum_score:.6f}"
        )

    @property
    def minimum_score(self) -> float:
        """Return the configured evidence threshold."""
        return self._minimum_score

    def generate(
        self,
        request: RAGGenerationRequest,
    ) -> RAGGeneration:
        """Return the strongest supplied evidence or explicitly abstain."""
        if not request.context_documents:
            return self._abstain()

        document = min(
            request.context_documents,
            key=lambda context: context.rank,
        )

        if document.score < self._minimum_score:
            return self._abstain()

        return RAGGeneration(
            answer=document.text,
            cited_document_ids=(
                document.document_id,
            ),
            abstained=False,
        )

    def _abstain(
        self,
    ) -> RAGGeneration:
        """Return an explicit evidence-aware abstention."""
        return RAGGeneration(
            answer=self._abstention_text,
            cited_document_ids=(),
            abstained=True,
        )
