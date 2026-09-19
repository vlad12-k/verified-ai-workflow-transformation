"""Provider-backed grounded RAG generation."""

from dataclasses import dataclass
from hashlib import sha256

from vait.inference.providers.base import InferenceProvider
from vait.inference.providers.models import (
    InferenceMessage,
    InferenceRole,
    ProviderInferenceRequest,
    ProviderInferenceResponse,
)
from vait.rag.models import (
    RAGContextDocument,
    RAGGeneration,
    RAGGenerationRequest,
)
from vait.rag.prompt_contracts import (
    INSUFFICIENT_EVIDENCE,
    RAGPromptContract,
    build_rag_messages,
)


@dataclass(frozen=True)
class ProviderRAGGenerationEvidence:
    """One grounded generation plus provider-level inference evidence."""

    generation: RAGGeneration
    provider_response: ProviderInferenceResponse | None


class ProviderBackedRAGGenerator:
    """Generate grounded answers through a provider-neutral inference adapter."""

    def __init__(
        self,
        *,
        provider: InferenceProvider,
        model_id: str,
        model_revision: str,
        minimum_score: float = 0.4,
        max_new_tokens: int = 80,
        prompt_contract: RAGPromptContract = (
            RAGPromptContract.BASELINE_V1
        ),
    ) -> None:
        """Configure one pinned provider-backed RAG candidate."""
        if not model_id.strip():
            raise ValueError(
                "model_id must not be empty."
            )

        if not model_revision.strip():
            raise ValueError(
                "model_revision must not be empty."
            )

        if not -1.0 <= minimum_score <= 1.0:
            raise ValueError(
                "minimum_score must be between -1.0 and 1.0."
            )

        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than zero."
            )

        self._provider = provider
        self._model_id = model_id
        self._model_revision = model_revision
        self._minimum_score = minimum_score
        self._max_new_tokens = max_new_tokens
        self._prompt_contract = prompt_contract

    @property
    def implementation_id(self) -> str:
        """Return reproducible provider, model, and prompt identity."""
        return (
            "provider-rag:"
            f"{self._provider.provider_id}:"
            f"{self._provider.runtime_id}:"
            f"{self._model_id}@{self._model_revision}:"
            f"minimum-score={self._minimum_score:.6f}:"
            f"max-new-tokens={self._max_new_tokens}:"
            f"prompt-contract={self._prompt_contract.value}"
        )

    @property
    def provider_id(self) -> str:
        """Return provider identity."""
        return self._provider.provider_id

    @property
    def runtime_id(self) -> str:
        """Return provider runtime identity."""
        return self._provider.runtime_id

    @property
    def model_id(self) -> str:
        """Return pinned model identifier."""
        return self._model_id

    @property
    def model_revision(self) -> str:
        """Return pinned model revision."""
        return self._model_revision

    @property
    def minimum_score(self) -> float:
        """Return pre-generation retrieval threshold."""
        return self._minimum_score

    @property
    def prompt_contract(self) -> RAGPromptContract:
        """Return versioned grounding prompt contract."""
        return self._prompt_contract

    def generate(
        self,
        request: RAGGenerationRequest,
    ) -> RAGGeneration:
        """Generate one grounded answer."""
        return self.generate_with_evidence(
            request
        ).generation

    def generate_with_evidence(
        self,
        request: RAGGenerationRequest,
    ) -> ProviderRAGGenerationEvidence:
        """Generate while preserving provider-level inference evidence."""
        if not request.context_documents:
            return ProviderRAGGenerationEvidence(
                generation=self._abstain(),
                provider_response=None,
            )

        ordered_context = tuple(
            sorted(
                request.context_documents,
                key=lambda document: document.rank,
            )
        )

        strongest_document = ordered_context[0]

        if (
            strongest_document.score
            < self._minimum_score
        ):
            return ProviderRAGGenerationEvidence(
                generation=self._abstain(),
                provider_response=None,
            )

        rag_messages = build_rag_messages(
            request,
            context_documents=ordered_context,
            contract=self._prompt_contract,
        )

        messages = tuple(
            InferenceMessage(
                role=InferenceRole(
                    message["role"]
                ),
                content=message["content"],
            )
            for message in rag_messages
        )

        provider_request = (
            ProviderInferenceRequest(
                request_id=self._request_id(
                    request=request,
                    context_documents=(
                        ordered_context
                    ),
                ),
                model_id=self._model_id,
                messages=messages,
                max_output_tokens=(
                    self._max_new_tokens
                ),
                temperature=0.0,
                metadata={
                    "task_family": (
                        "retrieval-augmented-generation"
                    ),
                    "prompt_contract": (
                        self._prompt_contract.value
                    ),
                    "model_revision": (
                        self._model_revision
                    ),
                    "minimum_score": (
                        self._minimum_score
                    ),
                    "context_document_ids": [
                        document.document_id
                        for document
                        in ordered_context
                    ],
                },
            )
        )

        response = self._provider.generate(
            provider_request
        )

        if not response.succeeded:
            error = response.error

            if error is None:
                raise RuntimeError(
                    "Provider failed without "
                    "normalised error evidence."
                )

            raise RuntimeError(
                "Provider inference failed: "
                f"{error.error_type}: "
                f"{error.message}"
            )

        if response.model_id != self._model_id:
            raise RuntimeError(
                "Provider response model identity "
                "does not match the pinned candidate."
            )

        if (
            response.model_revision
            != self._model_revision
        ):
            raise RuntimeError(
                "Provider response model revision "
                "does not match the pinned candidate."
            )

        if response.output_text is None:
            raise RuntimeError(
                "Successful provider response "
                "must contain output text."
            )

        answer = response.output_text.strip()

        if (
            not answer
            or answer == INSUFFICIENT_EVIDENCE
        ):
            generation = self._abstain()
        else:
            generation = RAGGeneration(
                answer=answer,
                cited_document_ids=tuple(
                    document.document_id
                    for document
                    in ordered_context
                ),
                abstained=False,
            )

        return ProviderRAGGenerationEvidence(
            generation=generation,
            provider_response=response,
        )

    def _request_id(
        self,
        *,
        request: RAGGenerationRequest,
        context_documents: tuple[
            RAGContextDocument,
            ...,
        ],
    ) -> str:
        """Build deterministic identity for an exact RAG request."""
        components = [
            self._provider.provider_id,
            self._provider.runtime_id,
            self._model_id,
            self._model_revision,
            self._prompt_contract.value,
            str(self._max_new_tokens),
            f"{self._minimum_score:.17g}",
            request.query,
        ]

        for document in context_documents:
            components.extend(
                [
                    str(document.rank),
                    document.document_id,
                    f"{document.score:.17g}",
                    document.text,
                ]
            )

        digest = sha256(
            "\n".join(
                components
            ).encode(
                "utf-8"
            )
        ).hexdigest()

        return (
            "rag-"
            f"{digest[:24]}"
        )

    @staticmethod
    def _abstain() -> RAGGeneration:
        """Return deterministic no-evidence output."""
        return RAGGeneration(
            answer="Insufficient evidence.",
            cited_document_ids=(),
            abstained=True,
        )
