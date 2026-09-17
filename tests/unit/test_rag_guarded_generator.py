"""Tests for semantic-guarded grounded generation."""

from vait.rag.extractive import ExtractiveGroundedGenerator
from vait.rag.guarded import SemanticGuardedGenerator
from vait.rag.models import (
    RAGContextDocument,
    RAGGeneration,
    RAGGenerationRequest,
)


class StaticGenerator:
    """Return one predefined generation."""

    def __init__(
        self,
        generation: RAGGeneration,
        *,
        implementation_id: str = "static-primary-v1",
    ) -> None:
        self._generation = generation
        self._implementation_id = implementation_id

    @property
    def implementation_id(self) -> str:
        """Return deterministic implementation identity."""
        return self._implementation_id

    def generate(
        self,
        request: RAGGenerationRequest,
    ) -> RAGGeneration:
        """Return the configured generation."""
        del request
        return self._generation


class FailingGenerator:
    """Fail if an unexpected fallback call occurs."""

    @property
    def implementation_id(self) -> str:
        """Return deterministic implementation identity."""
        return "failing-generator-v1"

    def generate(
        self,
        request: RAGGenerationRequest,
    ) -> RAGGeneration:
        """Raise when fallback should not have been called."""
        del request
        raise AssertionError(
            "Fallback generator should not have been called."
        )


def build_request() -> RAGGenerationRequest:
    """Build one bounded policy-evidence request."""
    return RAGGenerationRequest(
        query="What happens to a duplicate invoice?",
        context_documents=(
            RAGContextDocument(
                document_id="duplicate-policy",
                text=(
                    "Duplicate invoices must be held "
                    "before approval."
                ),
                score=0.9,
                rank=1,
            ),
        ),
    )


def test_guarded_generator_preserves_safe_primary_answer() -> None:
    """A passing primary generation should remain unchanged."""
    primary = StaticGenerator(
        RAGGeneration(
            answer=(
                "Duplicate invoices must be held "
                "before approval."
            ),
            cited_document_ids=(
                "duplicate-policy",
            ),
        )
    )

    guarded = SemanticGuardedGenerator(
        primary=primary,
        fallback=FailingGenerator(),
    )

    result = guarded.generate(
        build_request()
    )

    assert result.answer == (
        "Duplicate invoices must be held before approval."
    )
    assert result.cited_document_ids == (
        "duplicate-policy",
    )
    assert result.abstained is False


def test_guarded_generator_replaces_flagged_answer() -> None:
    """A weakened obligation should trigger extractive fallback."""
    primary = StaticGenerator(
        RAGGeneration(
            answer=(
                "Duplicate invoices should be held "
                "before approval."
            ),
            cited_document_ids=(
                "duplicate-policy",
            ),
        )
    )

    guarded = SemanticGuardedGenerator(
        primary=primary,
        fallback=ExtractiveGroundedGenerator(),
    )

    result = guarded.generate(
        build_request()
    )

    assert result.answer == (
        "Duplicate invoices must be held before approval."
    )
    assert result.cited_document_ids == (
        "duplicate-policy",
    )
    assert result.abstained is False


def test_guarded_generator_preserves_primary_abstention() -> None:
    """Explicit primary abstention should not invoke fallback."""
    primary = StaticGenerator(
        RAGGeneration(
            answer="Insufficient evidence.",
            cited_document_ids=(),
            abstained=True,
        )
    )

    guarded = SemanticGuardedGenerator(
        primary=primary,
        fallback=FailingGenerator(),
    )

    result = guarded.generate(
        build_request()
    )

    assert result.abstained is True
    assert result.cited_document_ids == ()
    assert result.answer == "Insufficient evidence."


def test_guarded_generator_falls_back_for_multi_document_scope() -> None:
    """Unsupported multi-document guard scope should fail closed."""
    request = RAGGenerationRequest(
        query="What controls apply?",
        context_documents=(
            RAGContextDocument(
                document_id="duplicate-policy",
                text="Duplicate invoices must be reviewed.",
                score=0.9,
                rank=1,
            ),
            RAGContextDocument(
                document_id="bank-policy",
                text="Bank changes require verification.",
                score=0.8,
                rank=2,
            ),
        ),
    )

    primary = StaticGenerator(
        RAGGeneration(
            answer="Both controls apply.",
            cited_document_ids=(
                "duplicate-policy",
                "bank-policy",
            ),
        )
    )

    guarded = SemanticGuardedGenerator(
        primary=primary,
        fallback=ExtractiveGroundedGenerator(),
    )

    result = guarded.generate(
        request
    )

    assert result.answer == (
        "Duplicate invoices must be reviewed."
    )
    assert result.cited_document_ids == (
        "duplicate-policy",
    )


def test_guarded_generator_does_not_mask_invalid_citation() -> None:
    """Malformed primary grounding should remain visible downstream."""
    primary_generation = RAGGeneration(
        answer="Duplicate invoices must be reviewed.",
        cited_document_ids=(
            "invented-policy",
        ),
    )

    guarded = SemanticGuardedGenerator(
        primary=StaticGenerator(
            primary_generation
        ),
        fallback=FailingGenerator(),
    )

    result = guarded.generate(
        build_request()
    )

    assert result == primary_generation


def test_guarded_generator_exposes_stable_identity() -> None:
    """Implementation identity should include both generation paths."""
    guarded = SemanticGuardedGenerator(
        primary=StaticGenerator(
            RAGGeneration(
                answer="answer",
                cited_document_ids=(
                    "duplicate-policy",
                ),
            ),
            implementation_id="primary-v1",
        ),
        fallback=ExtractiveGroundedGenerator(),
    )

    assert guarded.implementation_id == (
        "semantic-guarded-generator@1.0.0:"
        "primary=primary-v1:"
        "fallback=extractive-grounded-generator@1.0.0:"
        "minimum-score=0.000000"
    )
