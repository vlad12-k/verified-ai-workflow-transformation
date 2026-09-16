"""Provider-neutral generation interfaces."""

from typing import Protocol

from vait.rag.models import (
    RAGGeneration,
    RAGGenerationRequest,
)


class GroundedGenerator(Protocol):
    """Generate an answer from explicitly supplied evidence."""

    @property
    def implementation_id(self) -> str:
        """Return a stable generator implementation identifier."""
        ...

    def generate(
        self,
        request: RAGGenerationRequest,
    ) -> RAGGeneration:
        """Generate an answer using only supplied context."""
        ...
