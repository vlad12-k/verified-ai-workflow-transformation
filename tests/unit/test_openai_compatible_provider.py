"""Tests for the generic OpenAI-compatible provider."""

from collections.abc import Mapping

from pydantic import JsonValue

from vait.inference.providers.models import (
    InferenceMessage,
    InferenceRole,
    ProviderInferenceRequest,
)
from vait.inference.providers.openai_compatible import (
    OpenAICompatibleHttpResponse,
    OpenAICompatibleProvider,
)


class FakeTransport:
    """Deterministic JSON transport used to test provider normalisation."""

    def __init__(
        self,
        response: OpenAICompatibleHttpResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        """Store one response or transport failure."""
        self.response = response
        self.error = error
        self.calls: list[
            tuple[
                str,
                dict[str, str],
                dict[str, JsonValue],
                float,
            ]
        ] = []

    def post_json(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: dict[str, JsonValue],
        timeout_seconds: float,
    ) -> OpenAICompatibleHttpResponse:
        """Record one request and return configured evidence."""
        self.calls.append(
            (
                url,
                dict(headers),
                payload,
                timeout_seconds,
            )
        )

        if self.error is not None:
            raise self.error

        if self.response is None:
            raise RuntimeError(
                "Fake transport has no configured response."
            )

        return self.response


def build_request(
    *,
    model_id: str = "portable-model",
) -> ProviderInferenceRequest:
    """Build one provider-neutral test request."""
    return ProviderInferenceRequest(
        request_id="request-1",
        model_id=model_id,
        messages=(
            InferenceMessage(
                role=InferenceRole.SYSTEM,
                content="Use evidence only.",
            ),
            InferenceMessage(
                role=InferenceRole.USER,
                content="What is the result?",
            ),
        ),
        max_output_tokens=32,
        temperature=0.0,
    )


def successful_http_response() -> OpenAICompatibleHttpResponse:
    """Build one canonical chat-completions response."""
    return OpenAICompatibleHttpResponse(
        status_code=200,
        body={
            "id": "chatcmpl-test",
            "model": "portable-model",
            "system_fingerprint": "fingerprint-1",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "verified response",
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 3,
                "total_tokens": 15,
            },
        },
    )


def test_openai_compatible_provider_normalises_success() -> None:
    """Chat completions should map into the portable response contract."""
    transport = FakeTransport(
        response=successful_http_response()
    )

    provider = OpenAICompatibleProvider(
        base_url="https://example.test",
        model_id="portable-model",
        transport=transport,
        api_key="secret-test-key",
        timeout_seconds=12.5,
    )

    response = provider.generate(
        build_request()
    )

    assert response.succeeded is True
    assert response.provider_id == "openai-compatible"
    assert response.runtime_id == "chat-completions-http"

    assert response.model_id == "portable-model"
    assert response.model_revision is None
    assert response.output_text == "verified response"

    assert response.usage is not None
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 3
    assert response.usage.total_tokens == 15

    assert response.timing.total_latency_ms >= 0.0
    assert response.timing.time_to_first_token_ms is None
    assert response.timing.generation_latency_ms is None

    assert response.cost_usd is None
    assert response.metadata["http_status"] == 200
    assert response.metadata["streaming"] is False
    assert response.metadata["ttft_available"] is False
    assert (
        response.metadata["provider_response_id"]
        == "chatcmpl-test"
    )
    assert (
        response.metadata["system_fingerprint"]
        == "fingerprint-1"
    )

    assert len(transport.calls) == 1

    url, headers, payload, timeout = transport.calls[0]

    assert (
        url
        == "https://example.test/v1/chat/completions"
    )
    assert (
        headers["Authorization"]
        == "Bearer secret-test-key"
    )
    assert headers["Content-Type"] == "application/json"

    assert payload["model"] == "portable-model"
    assert payload["max_tokens"] == 32
    assert payload["temperature"] == 0.0
    assert payload["stream"] is False
    assert timeout == 12.5


def test_openai_compatible_provider_accepts_v1_base_url() -> None:
    """A base URL already ending in v1 must not duplicate that path."""
    transport = FakeTransport(
        response=successful_http_response()
    )

    provider = OpenAICompatibleProvider(
        base_url="http://localhost:8000/v1/",
        model_id="portable-model",
        transport=transport,
    )

    provider.generate(
        build_request()
    )

    assert transport.calls[0][0] == (
        "http://localhost:8000/v1/chat/completions"
    )


def test_openai_compatible_provider_allows_missing_usage() -> None:
    """Compatible servers may omit token usage without invalidating output."""
    transport = FakeTransport(
        response=OpenAICompatibleHttpResponse(
            status_code=200,
            body={
                "choices": [
                    {
                        "message": {
                            "content": "response without usage",
                        }
                    }
                ]
            },
        )
    )

    provider = OpenAICompatibleProvider(
        base_url="https://example.test",
        model_id="portable-model",
        transport=transport,
    )

    response = provider.generate(
        build_request()
    )

    assert response.succeeded is True
    assert response.output_text == "response without usage"
    assert response.usage is None


def test_openai_compatible_provider_normalises_rate_limit() -> None:
    """HTTP 429 should become retryable provider-neutral error evidence."""
    transport = FakeTransport(
        response=OpenAICompatibleHttpResponse(
            status_code=429,
            body={
                "error": {
                    "type": "rate_limit_error",
                    "message": "Too many requests.",
                }
            },
        )
    )

    provider = OpenAICompatibleProvider(
        base_url="https://example.test",
        model_id="portable-model",
        transport=transport,
    )

    response = provider.generate(
        build_request()
    )

    assert response.succeeded is False
    assert response.error is not None
    assert response.error.error_type == "rate_limit_error"
    assert response.error.message == "Too many requests."
    assert response.error.retryable is True
    assert response.metadata["http_status"] == 429


def test_openai_compatible_provider_rejects_malformed_success() -> None:
    """A successful HTTP status without assistant content is invalid."""
    transport = FakeTransport(
        response=OpenAICompatibleHttpResponse(
            status_code=200,
            body={
                "choices": [],
            },
        )
    )

    provider = OpenAICompatibleProvider(
        base_url="https://example.test",
        model_id="portable-model",
        transport=transport,
    )

    response = provider.generate(
        build_request()
    )

    assert response.succeeded is False
    assert response.error is not None
    assert response.error.error_type == "invalid_response"
    assert response.error.retryable is False


def test_openai_compatible_provider_normalises_transport_failure() -> None:
    """Transport exceptions should not leak outside the provider boundary."""
    transport = FakeTransport(
        error=TimeoutError("endpoint timed out")
    )

    provider = OpenAICompatibleProvider(
        base_url="https://example.test",
        model_id="portable-model",
        transport=transport,
    )

    response = provider.generate(
        build_request()
    )

    assert response.succeeded is False
    assert response.error is not None
    assert response.error.error_type == "transport_error"
    assert response.error.message == "endpoint timed out"
    assert response.error.retryable is True


def test_openai_compatible_provider_rejects_model_mismatch() -> None:
    """Pinned provider configuration must not silently switch models."""
    transport = FakeTransport(
        response=successful_http_response()
    )

    provider = OpenAICompatibleProvider(
        base_url="https://example.test",
        model_id="portable-model",
        transport=transport,
    )

    response = provider.generate(
        build_request(
            model_id="different-model",
        )
    )

    assert response.succeeded is False
    assert response.error is not None
    assert response.error.error_type == "model_mismatch"
    assert transport.calls == []
