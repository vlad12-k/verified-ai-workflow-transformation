"""Versioned prompt contracts for grounded RAG generation."""

from enum import StrEnum

from vait.rag.models import (
    RAGContextDocument,
    RAGGenerationRequest,
)

INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class RAGPromptContract(StrEnum):
    """Versioned generation contracts for controlled experiments."""

    BASELINE_V1 = "baseline-v1"
    STRICT_V2 = "strict-v2"


def build_rag_messages(
    request: RAGGenerationRequest,
    *,
    context_documents: tuple[RAGContextDocument, ...],
    contract: RAGPromptContract,
) -> list[dict[str, str]]:
    """Build messages under one explicit prompt contract."""
    context = "\n\n".join(
        (
            f"[{document.document_id}]\n"
            f"{document.text}"
        )
        for document in context_documents
    )

    if contract is RAGPromptContract.BASELINE_V1:
        system_message = (
            "Answer the question using only the supplied policy context. "
            "Do not add facts, reasons, consequences, procedures, or actions "
            "that are not explicitly stated in the context. "
            "If the evidence does not support an answer, respond exactly "
            f"{INSUFFICIENT_EVIDENCE}."
        )
    elif contract is RAGPromptContract.STRICT_V2:
        system_message = (
            "Answer only with policy statements explicitly supported by "
            "the supplied context. "
            "Restate only the minimum information needed to answer the "
            "question. "
            "Preserve every material condition, exception, qualifier, and "
            "modality, including words such as must, may, unless, before, "
            "after, any, and large. "
            "Do not infer or add unstated reasons, risks, motives, examples, "
            "actors, approvals, checks, documents, procedures, consequences, "
            "or recommendations. "
            "Do not explain why the policy exists. "
            "Do not strengthen or weaken the policy. "
            "Use at most two concise sentences. "
            "If the context does not explicitly answer the question, "
            f"respond exactly {INSUFFICIENT_EVIDENCE}."
        )
    else:
        raise ValueError(
            f"Unsupported RAG prompt contract: {contract}"
        )

    user_message = (
        "Policy context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{request.query}"
    )

    return [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]
