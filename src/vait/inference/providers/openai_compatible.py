"""Generic OpenAI-compatible inference provider."""

from collections.abc import Mapping
from time import perf_counter_ns
from typing import Protocol

from pydantic import BaseModel, Field, JsonValue

from vait.inference.providers.models import (
    ProviderError,
    ProviderInferenceRequest,
    ProviderInferenceResponse,
    ProviderTiming,
    ProviderTokenUsage,
)


class OpenAICompatibleHttpResponse(BaseModel):
    """Transport-neutral HTTP response from an OpenAI-compatible endpoint."""

    status_code: int = Field(
        ge=100,
        le=599,
    )
    body: dict[str, JsonValue] = Field(
        default_factory=dict,
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
    )


class OpenAICompatibleTransport(Protocol):
    """Minimal HTTP transport boundary required by the provider."""

    def post_json(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: dict[str, JsonValue],
        timeout_seconds: float,
    ) -> OpenAICompatibleHttpResponse:
        """POST one JSON request and return a normalised HTTP response."""
        ...


class OpenAICompatibleProvider:
    """Execute inference against a generic OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model_id: str,
        transport: OpenAICompatibleTransport,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        """Configure a provider without binding VAIT to a vendor SDK."""
        normalized_base_url = base_url.strip().rstrip("/")

        if not normalized_base_url:
            raise ValueError("base_url must not be empty.")

        if not model_id.strip():
            raise ValueError("model_id must not be empty.")

        if timeout_seconds <= 0.0:
            raise ValueError(
                "timeout_seconds must be greater than zero."
            )

        if api_key is not None and not api_key.strip():
            raise ValueError(
                "api_key must not be blank when supplied."
            )

        self._base_url = normalized_base_url
        self._model_id = model_id
        self._transport = transport
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._extra_headers = dict(
            extra_headers or {}
        )

    @property
    def provider_id(self) -> str:
        """Return stable provider identity."""
        return "openai-compatible"

    @property
    def runtime_id(self) -> str:
        """Return stable runtime identity."""
        return "chat-completions-http"

    @property
    def model_id(self) -> str:
        """Return the configured model identifier."""
        return self._model_id

    @property
    def chat_completions_url(self) -> str:
        """Return the normalised non-streaming chat-completions endpoint."""
        if self._base_url.endswith("/v1"):
            return (
                f"{self._base_url}/chat/completions"
            )

        return (
            f"{self._base_url}/v1/chat/completions"
        )

    def generate(
        self,
        request: ProviderInferenceRequest,
    ) -> ProviderInferenceResponse:
        """Execute one non-streaming OpenAI-compatible request."""
        started = perf_counter_ns()

        if request.model_id != self._model_id:
            return self._error_response(
                request=request,
                started=started,
                error_type="model_mismatch",
                message=(
                    f"Provider serves '{self._model_id}', "
                    f"not '{request.model_id}'."
                ),
                retryable=False,
            )

        payload: dict[str, JsonValue] = {
            "model": self._model_id,
            "messages": [
                {
                    "role": message.role.value,
                    "content": message.content,
                }
                for message in request.messages
            ],
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "stream": False,
        }

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        if self._api_key is not None:
            headers["Authorization"] = (
                f"Bearer {self._api_key}"
            )

        headers.update(
            self._extra_headers
        )

        try:
            http_response = self._transport.post_json(
                url=self.chat_completions_url,
                headers=headers,
                payload=payload,
                timeout_seconds=self._timeout_seconds,
            )
        except Exception as exc:
            return self._error_response(
                request=request,
                started=started,
                error_type="transport_error",
                message=(
                    str(exc).strip()
                    or type(exc).__name__
                ),
                retryable=True,
            )

        total_latency_ms = (
            perf_counter_ns() - started
        ) / 1_000_000

        if not 200 <= http_response.status_code < 300:
            error_type, message = (
                self._extract_http_error(
                    http_response
                )
            )

            return self._error_response(
                request=request,
                started=started,
                error_type=error_type,
                message=message,
                retryable=self._is_retryable_status(
                    http_response.status_code
                ),
                http_status=http_response.status_code,
                total_latency_ms=total_latency_ms,
            )

        output_text = self._extract_output_text(
            http_response.body
        )

        if output_text is None:
            return self._error_response(
                request=request,
                started=started,
                error_type="invalid_response",
                message=(
                    "OpenAI-compatible response did not "
                    "contain choices[0].message.content."
                ),
                retryable=False,
                http_status=http_response.status_code,
                total_latency_ms=total_latency_ms,
            )

        usage = self._extract_usage(
            http_response.body
        )

        response_model = http_response.body.get(
            "model"
        )

        model_id = (
            response_model
            if isinstance(response_model, str)
            and response_model.strip()
            else self._model_id
        )

        metadata: dict[str, JsonValue] = {
            "http_status": http_response.status_code,
            "streaming": False,
            "ttft_available": False,
        }

        response_id = http_response.body.get(
            "id"
        )

        if isinstance(response_id, str):
            metadata["provider_response_id"] = (
                response_id
            )

        system_fingerprint = (
            http_response.body.get(
                "system_fingerprint"
            )
        )

        if isinstance(system_fingerprint, str):
            metadata["system_fingerprint"] = (
                system_fingerprint
            )

        return ProviderInferenceResponse(
            request_id=request.request_id,
            provider_id=self.provider_id,
            runtime_id=self.runtime_id,
            model_id=model_id,
            model_revision=None,
            output_text=output_text,
            usage=usage,
            timing=ProviderTiming(
                total_latency_ms=total_latency_ms,
                time_to_first_token_ms=None,
                generation_latency_ms=None,
            ),
            cost_usd=None,
            metadata=metadata,
        )

    def _error_response(
        self,
        *,
        request: ProviderInferenceRequest,
        started: int,
        error_type: str,
        message: str,
        retryable: bool,
        http_status: int | None = None,
        total_latency_ms: float | None = None,
    ) -> ProviderInferenceResponse:
        """Return one normalised provider failure."""
        if total_latency_ms is None:
            total_latency_ms = (
                perf_counter_ns() - started
            ) / 1_000_000

        metadata: dict[str, JsonValue] = {
            "streaming": False,
            "ttft_available": False,
        }

        if http_status is not None:
            metadata["http_status"] = (
                http_status
            )

        return ProviderInferenceResponse(
            request_id=request.request_id,
            provider_id=self.provider_id,
            runtime_id=self.runtime_id,
            model_id=self._model_id,
            model_revision=None,
            timing=ProviderTiming(
                total_latency_ms=total_latency_ms,
            ),
            cost_usd=None,
            error=ProviderError(
                error_type=error_type,
                message=message,
                retryable=retryable,
            ),
            metadata=metadata,
        )

    @staticmethod
    def _extract_output_text(
        body: dict[str, JsonValue],
    ) -> str | None:
        """Extract choices[0].message.content when present."""
        choices = body.get("choices")

        if (
            not isinstance(choices, list)
            or not choices
        ):
            return None

        first_choice = choices[0]

        if not isinstance(first_choice, dict):
            return None

        message = first_choice.get(
            "message"
        )

        if not isinstance(message, dict):
            return None

        content = message.get(
            "content"
        )

        if not isinstance(content, str):
            return None

        return content.strip()

    @staticmethod
    def _extract_usage(
        body: dict[str, JsonValue],
    ) -> ProviderTokenUsage | None:
        """Extract standard OpenAI-compatible token usage when available."""
        usage = body.get("usage")

        if not isinstance(usage, dict):
            return None

        prompt_tokens = usage.get(
            "prompt_tokens"
        )
        completion_tokens = usage.get(
            "completion_tokens"
        )

        if (
            not isinstance(prompt_tokens, int)
            or isinstance(prompt_tokens, bool)
            or prompt_tokens < 0
            or not isinstance(completion_tokens, int)
            or isinstance(completion_tokens, bool)
            or completion_tokens < 0
        ):
            return None

        return ProviderTokenUsage(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
        )

    @staticmethod
    def _extract_http_error(
        response: OpenAICompatibleHttpResponse,
    ) -> tuple[str, str]:
        """Extract a portable error type and message from an HTTP failure."""
        error = response.body.get(
            "error"
        )

        error_type: str | None = None
        message: str | None = None

        if isinstance(error, dict):
            candidate_type = error.get(
                "type"
            )
            candidate_message = error.get(
                "message"
            )

            if (
                isinstance(candidate_type, str)
                and candidate_type.strip()
            ):
                error_type = candidate_type

            if (
                isinstance(candidate_message, str)
                and candidate_message.strip()
            ):
                message = candidate_message

        return (
            error_type
            or f"http_{response.status_code}",
            message
            or (
                "OpenAI-compatible endpoint returned "
                f"HTTP {response.status_code}."
            ),
        )

    @staticmethod
    def _is_retryable_status(
        status_code: int,
    ) -> bool:
        """Classify HTTP failures that may succeed when retried later."""
        return (
            status_code in {
                408,
                425,
                429,
            }
            or status_code >= 500
        )
