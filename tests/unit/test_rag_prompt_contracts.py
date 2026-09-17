"""Tests for versioned grounded RAG prompt contracts."""

from vait.rag.models import (
    RAGContextDocument,
    RAGGenerationRequest,
)
from vait.rag.prompt_contracts import (
    INSUFFICIENT_EVIDENCE,
    RAGPromptContract,
    build_rag_messages,
)


def build_request() -> RAGGenerationRequest:
    """Build one deterministic prompt-contract request."""
    return RAGGenerationRequest(
        query="How should this invoice be handled?",
        context_documents=(
            RAGContextDocument(
                document_id="policy-a",
                text=(
                    "Invoices require manual review unless "
                    "an approved exception applies."
                ),
                score=0.9,
                rank=1,
            ),
        ),
    )


def test_baseline_v1_preserves_original_contract() -> None:
    """Baseline prompt must remain reproducible."""
    request = build_request()

    messages = build_rag_messages(
        request,
        context_documents=request.context_documents,
        contract=RAGPromptContract.BASELINE_V1,
    )

    system_message = messages[0]["content"]

    assert "Do not add facts" in system_message
    assert INSUFFICIENT_EVIDENCE in system_message
    assert "policy-a" in messages[1]["content"]


def test_strict_v2_preserves_policy_qualifiers() -> None:
    """Strict prompt should explicitly prohibit semantic expansion."""
    request = build_request()

    messages = build_rag_messages(
        request,
        context_documents=request.context_documents,
        contract=RAGPromptContract.STRICT_V2,
    )

    system_message = messages[0]["content"]

    assert "Preserve every material condition" in system_message
    assert "Do not strengthen or weaken the policy" in system_message
    assert "at most two concise sentences" in system_message
    assert INSUFFICIENT_EVIDENCE in system_message
