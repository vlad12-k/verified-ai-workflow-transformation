"""Tests for provider-backed grounded RAG generation."""

import pytest

from vait.inference.providers.models import (
    ProviderError,
    ProviderInferenceRequest,
    ProviderInferenceResponse,
    ProviderTiming,
    ProviderTokenUsage,
)
from vait.rag.models import (
    RAGContextDocument,
    RAGGenerationRequest,
)
from vait.rag.prompt_contracts import (
    RAGPromptContract,
)
from vait.rag.provider import (
    ProviderBackedRAGGenerator,
)


class RecordingProvider:
    """Deterministic provider double."""

    provider_id = "test-provider"
    runtime_id = "test-runtime"

    def __init__(
        self,
        *,
        output_text: str = (
            "Hold the invoice before payment."
        ),
        model_id: str = "example/model",
        revision: str = "abc123",
        failure: bool = False,
    ) -> None:
        """Configure deterministic response evidence."""
        self.output_text = output_text
        self.model_id = model_id
        self.revision = revision
        self.failure = failure
        self.requests: list[
            ProviderInferenceRequest
        ] = []

    def generate(
        self,
        request: ProviderInferenceRequest,
    ) -> ProviderInferenceResponse:
        """Record request and return deterministic response."""
        self.requests.append(
            request
        )

        if self.failure:
            return ProviderInferenceResponse(
                request_id=request.request_id,
                provider_id=self.provider_id,
                runtime_id=self.runtime_id,
                model_id=self.model_id,
                model_revision=self.revision,
                timing=ProviderTiming(
                    total_latency_ms=2.0,
                ),
                error=ProviderError(
                    error_type="test_failure",
                    message="Generation failed.",
                    retryable=False,
                ),
            )

        return ProviderInferenceResponse(
            request_id=request.request_id,
            provider_id=self.provider_id,
            runtime_id=self.runtime_id,
            model_id=self.model_id,
            model_revision=self.revision,
            output_text=self.output_text,
            usage=ProviderTokenUsage(
                input_tokens=20,
                output_tokens=7,
            ),
            timing=ProviderTiming(
                total_latency_ms=5.0,
                generation_latency_ms=4.0,
            ),
        )


def build_request(
    *,
    score: float = 0.9,
) -> RAGGenerationRequest:
    """Build deterministic grounded request."""
    return RAGGenerationRequest(
        query=(
            "What should happen to "
            "a duplicate invoice?"
        ),
        context_documents=(
            RAGContextDocument(
                document_id=(
                    "duplicate-policy"
                ),
                text=(
                    "Potential duplicate invoices "
                    "must be held before payment."
                ),
                score=score,
                rank=1,
            ),
        ),
    )


def build_generator(
    provider: RecordingProvider,
) -> ProviderBackedRAGGenerator:
    """Build pinned strict provider-backed generator."""
    return ProviderBackedRAGGenerator(
        provider=provider,
        model_id="example/model",
        model_revision="abc123",
        minimum_score=0.4,
        max_new_tokens=48,
        prompt_contract=(
            RAGPromptContract.STRICT_V2
        ),
    )


def test_provider_rag_maps_grounded_request(
) -> None:
    """Grounded request should map to provider-neutral inference."""
    provider = RecordingProvider()
    generator = build_generator(
        provider
    )

    evidence = (
        generator.generate_with_evidence(
            build_request()
        )
    )

    assert (
        evidence.generation.answer
        == "Hold the invoice before payment."
    )
    assert (
        evidence.generation
        .cited_document_ids
        == (
            "duplicate-policy",
        )
    )
    assert (
        evidence.generation.abstained
        is False
    )

    assert (
        evidence.provider_response
        is not None
    )

    assert len(
        provider.requests
    ) == 1

    request = provider.requests[0]

    assert request.model_id == (
        "example/model"
    )
    assert (
        request.max_output_tokens
        == 48
    )
    assert request.temperature == 0.0

    assert (
        request.metadata[
            "prompt_contract"
        ]
        == "strict-v2"
    )

    assert (
        request.metadata[
            "context_document_ids"
        ]
        == [
            "duplicate-policy",
        ]
    )

    assert (
        request.messages[0].role.value
        == "system"
    )
    assert (
        request.messages[1].role.value
        == "user"
    )

    assert (
        "duplicate-policy"
        in request.messages[1].content
    )


def test_provider_rag_abstains_before_inference_for_weak_evidence(
) -> None:
    """Weak retrieval evidence must skip provider inference."""
    provider = RecordingProvider()
    generator = build_generator(
        provider
    )

    evidence = (
        generator.generate_with_evidence(
            build_request(
                score=0.3,
            )
        )
    )

    assert (
        evidence.generation.abstained
        is True
    )
    assert (
        evidence.generation
        .cited_document_ids
        == ()
    )
    assert (
        evidence.provider_response
        is None
    )
    assert provider.requests == []


def test_provider_rag_maps_model_abstention(
) -> None:
    """Explicit model abstention must not claim citations."""
    provider = RecordingProvider(
        output_text=(
            "INSUFFICIENT_EVIDENCE"
        )
    )

    generator = build_generator(
        provider
    )

    evidence = (
        generator.generate_with_evidence(
            build_request()
        )
    )

    assert (
        evidence.generation.abstained
        is True
    )
    assert (
        evidence.generation.answer
        == "Insufficient evidence."
    )
    assert (
        evidence.generation
        .cited_document_ids
        == ()
    )
    assert (
        evidence.provider_response
        is not None
    )


def test_provider_rag_rejects_provider_failure(
) -> None:
    """Provider errors must not silently become grounded answers."""
    provider = RecordingProvider(
        failure=True
    )

    generator = build_generator(
        provider
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Provider inference failed"
        ),
    ):
        generator.generate(
            build_request()
        )


def test_provider_rag_rejects_revision_drift(
) -> None:
    """Observed model revision must match the pinned candidate."""
    provider = RecordingProvider(
        revision="different-revision"
    )

    generator = build_generator(
        provider
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "model revision"
        ),
    ):
        generator.generate(
            build_request()
        )


def test_provider_rag_identity_is_reproducible(
) -> None:
    """Implementation identity must encode the pinned execution contract."""
    provider = RecordingProvider()
    generator = build_generator(
        provider
    )

    identity = (
        generator.implementation_id
    )

    assert "test-provider" in identity
    assert "test-runtime" in identity
    assert (
        "example/model@abc123"
        in identity
    )
    assert (
        "prompt-contract=strict-v2"
        in identity
    )
    assert (
        "max-new-tokens=48"
        in identity
    )


def test_provider_rag_request_id_is_deterministic(
) -> None:
    """Equivalent grounded requests should retain stable identity."""
    provider = RecordingProvider()
    generator = build_generator(
        provider
    )

    generator.generate(
        build_request()
    )

    generator.generate(
        build_request()
    )

    assert len(
        provider.requests
    ) == 2

    assert (
        provider.requests[0].request_id
        == provider.requests[1].request_id
    )
