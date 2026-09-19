"""Tests for deterministic provider response caching."""

from pydantic import JsonValue

from vait.inference.providers.cache import (
    DeterministicCachingProvider,
)
from vait.inference.providers.models import (
    InferenceMessage,
    InferenceRole,
    ProviderError,
    ProviderInferenceRequest,
    ProviderInferenceResponse,
    ProviderTiming,
    ProviderTokenUsage,
)


def build_request(
    *,
    request_id: str = "request-1",
    content: str = "Return the verified result.",
    temperature: float = 0.0,
    metadata: dict[str, JsonValue] | None = None,
) -> ProviderInferenceRequest:
    """Build one deterministic provider request."""
    return ProviderInferenceRequest(
        request_id=request_id,
        model_id="test-model",
        messages=(
            InferenceMessage(
                role=InferenceRole.USER,
                content=content,
            ),
        ),
        max_output_tokens=16,
        temperature=temperature,
        metadata=metadata or {},
    )


class RecordingProvider:
    """Return successful responses while recording provider work."""

    provider_id = "test-provider"
    runtime_id = "test-runtime"

    def __init__(self) -> None:
        """Track exact provider invocations."""
        self.requests: list[
            ProviderInferenceRequest
        ] = []

    def generate(
        self,
        request: ProviderInferenceRequest,
    ) -> ProviderInferenceResponse:
        """Return one deterministic provider response."""
        self.requests.append(
            request
        )

        return ProviderInferenceResponse(
            request_id=request.request_id,
            provider_id=self.provider_id,
            runtime_id=self.runtime_id,
            model_id=request.model_id,
            model_revision="revision-1",
            output_text="verified answer",
            usage=ProviderTokenUsage(
                input_tokens=10,
                output_tokens=4,
            ),
            timing=ProviderTiming(
                total_latency_ms=5.0,
                generation_latency_ms=4.0,
            ),
            cost_usd=None,
            metadata={
                "backend": "fake",
            },
        )


def test_deterministic_cache_miss_then_hit() -> None:
    """Equivalent deterministic requests should require provider work once."""
    underlying = RecordingProvider()

    provider = (
        DeterministicCachingProvider(
            provider=underlying
        )
    )

    first = provider.generate(
        build_request(
            request_id="request-1"
        )
    )

    second = provider.generate(
        build_request(
            request_id="request-2"
        )
    )

    assert len(
        underlying.requests
    ) == 1

    assert first.succeeded
    assert second.succeeded

    assert (
        first.output_text
        == second.output_text
        == "verified answer"
    )

    assert (
        first.metadata[
            "cache_status"
        ]
        == "miss"
    )

    assert (
        second.metadata[
            "cache_status"
        ]
        == "hit"
    )

    assert (
        second.request_id
        == "request-2"
    )

    assert (
        second.runtime_id
        == (
            "test-runtime"
            "+deterministic-response-cache-v1"
        )
    )

    assert second.usage is not None

    assert (
        second.usage.input_tokens
        == 0
    )

    assert (
        second.usage.output_tokens
        == 0
    )

    assert (
        second.metadata[
            "cache_origin_input_tokens"
        ]
        == 10
    )

    assert (
        second.metadata[
            "cache_origin_output_tokens"
        ]
        == 4
    )

    assert (
        provider.cache_size
        == 1
    )

    assert (
        provider.cache_hits
        == 1
    )

    assert (
        provider.cache_misses
        == 1
    )

    assert (
        provider.cache_bypasses
        == 0
    )


def test_request_id_is_excluded_from_cache_key() -> None:
    """Correlation identity must not prevent semantic cache reuse."""
    underlying = RecordingProvider()

    provider = (
        DeterministicCachingProvider(
            provider=underlying
        )
    )

    first = provider.generate(
        build_request(
            request_id="first-id"
        )
    )

    second = provider.generate(
        build_request(
            request_id="second-id"
        )
    )

    assert len(
        underlying.requests
    ) == 1

    assert (
        first.metadata[
            "cache_key_sha256"
        ]
        == second.metadata[
            "cache_key_sha256"
        ]
    )


def test_cache_key_changes_with_semantic_request() -> None:
    """Different prompt semantics must not share cached output."""
    underlying = RecordingProvider()

    provider = (
        DeterministicCachingProvider(
            provider=underlying
        )
    )

    provider.generate(
        build_request(
            request_id="request-1",
            content="Question one",
        )
    )

    provider.generate(
        build_request(
            request_id="request-2",
            content="Question two",
        )
    )

    assert len(
        underlying.requests
    ) == 2

    assert (
        provider.cache_size
        == 2
    )

    assert (
        provider.cache_misses
        == 2
    )


def test_cache_key_canonicalises_metadata_order() -> None:
    """Equivalent JSON metadata must hash identically regardless of key order."""
    underlying = RecordingProvider()

    provider = (
        DeterministicCachingProvider(
            provider=underlying
        )
    )

    first = provider.generate(
        build_request(
            request_id="request-1",
            metadata={
                "b": 2,
                "a": 1,
            },
        )
    )

    second = provider.generate(
        build_request(
            request_id="request-2",
            metadata={
                "a": 1,
                "b": 2,
            },
        )
    )

    assert len(
        underlying.requests
    ) == 1

    assert (
        first.metadata[
            "cache_key_sha256"
        ]
        == second.metadata[
            "cache_key_sha256"
        ]
    )


def test_sampling_requests_bypass_deterministic_cache() -> None:
    """Positive-temperature requests must never enter deterministic cache."""
    underlying = RecordingProvider()

    provider = (
        DeterministicCachingProvider(
            provider=underlying
        )
    )

    first = provider.generate(
        build_request(
            request_id="request-1",
            temperature=0.7,
        )
    )

    second = provider.generate(
        build_request(
            request_id="request-2",
            temperature=0.7,
        )
    )

    assert len(
        underlying.requests
    ) == 2

    assert (
        first.metadata[
            "cache_status"
        ]
        == "bypass"
    )

    assert (
        second.metadata[
            "cache_status"
        ]
        == "bypass"
    )

    assert (
        provider.cache_size
        == 0
    )

    assert (
        provider.cache_hits
        == 0
    )

    assert (
        provider.cache_misses
        == 0
    )

    assert (
        provider.cache_bypasses
        == 2
    )


class FailThenSucceedProvider:
    """Fail once and succeed on the next identical request."""

    provider_id = "test-provider"
    runtime_id = "test-runtime"

    def __init__(self) -> None:
        """Track invocation count."""
        self.calls = 0

    def generate(
        self,
        request: ProviderInferenceRequest,
    ) -> ProviderInferenceResponse:
        """Return failure first, then success."""
        self.calls += 1

        if self.calls == 1:
            return ProviderInferenceResponse(
                request_id=request.request_id,
                provider_id=self.provider_id,
                runtime_id=self.runtime_id,
                model_id=request.model_id,
                model_revision="revision-1",
                timing=ProviderTiming(
                    total_latency_ms=1.0,
                ),
                error=ProviderError(
                    error_type="SyntheticFailure",
                    message="temporary failure",
                    retryable=True,
                ),
            )

        return ProviderInferenceResponse(
            request_id=request.request_id,
            provider_id=self.provider_id,
            runtime_id=self.runtime_id,
            model_id=request.model_id,
            model_revision="revision-1",
            output_text="recovered answer",
            usage=ProviderTokenUsage(
                input_tokens=5,
                output_tokens=2,
            ),
            timing=ProviderTiming(
                total_latency_ms=2.0,
                generation_latency_ms=1.0,
            ),
        )


def test_provider_failures_are_not_cached() -> None:
    """A failed provider response must never poison deterministic cache state."""
    underlying = (
        FailThenSucceedProvider()
    )

    provider = (
        DeterministicCachingProvider(
            provider=underlying
        )
    )

    first = provider.generate(
        build_request(
            request_id="request-1"
        )
    )

    second = provider.generate(
        build_request(
            request_id="request-2"
        )
    )

    third = provider.generate(
        build_request(
            request_id="request-3"
        )
    )

    assert (
        underlying.calls
        == 2
    )

    assert not first.succeeded
    assert second.succeeded
    assert third.succeeded

    assert (
        first.metadata[
            "cache_stored"
        ]
        is False
    )

    assert (
        second.metadata[
            "cache_status"
        ]
        == "miss"
    )

    assert (
        second.metadata[
            "cache_stored"
        ]
        is True
    )

    assert (
        third.metadata[
            "cache_status"
        ]
        == "hit"
    )

    assert (
        provider.cache_size
        == 1
    )


def test_clear_removes_cached_response() -> None:
    """Explicit invalidation must force the next provider execution."""
    underlying = RecordingProvider()

    provider = (
        DeterministicCachingProvider(
            provider=underlying
        )
    )

    provider.generate(
        build_request(
            request_id="request-1"
        )
    )

    assert (
        provider.cache_size
        == 1
    )

    provider.clear()

    assert (
        provider.cache_size
        == 0
    )

    provider.generate(
        build_request(
            request_id="request-2"
        )
    )

    assert len(
        underlying.requests
    ) == 2
