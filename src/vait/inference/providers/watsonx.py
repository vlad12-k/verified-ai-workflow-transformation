"""watsonx enterprise adapter behind the provider-neutral boundary."""

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


class WatsonxTransportResponse(BaseModel):
    """Normalised evidence returned by a watsonx-specific transport."""

    model_id: str = Field(min_length=1)
    output_text: str = Field(min_length=1)

    input_tokens: int | None = Field(
        default=None,
        ge=0,
    )
    output_tokens: int | None = Field(
        default=None,
        ge=0,
    )

    model_revision: str | None = None
    provider_request_id: str | None = None
    stop_reason: str | None = None

    metadata: dict[str, JsonValue] = Field(
        default_factory=dict,
    )


class WatsonxTransport(Protocol):
    """Boundary implemented by an IBM-specific watsonx integration."""

    def generate(
        self,
        *,
        model_id: str,
        messages: tuple[dict[str, str], ...],
        max_output_tokens: int,
        temperature: float,
        metadata: Mapping[str, JsonValue],
    ) -> WatsonxTransportResponse:
        """Execute one watsonx generation request."""
        ...


class WatsonxProvider:
    """Expose watsonx inference through the portable VAIT provider contract."""

    def __init__(
        self,
        *,
        model_id: str,
        transport: WatsonxTransport,
        deployment_id: str | None = None,
    ) -> None:
        """Configure one watsonx model or deployment adapter."""
        if not model_id.strip():
            raise ValueError("model_id must not be empty.")

        if (
            deployment_id is not None
            and not deployment_id.strip()
        ):
            raise ValueError(
                "deployment_id must not be blank when supplied."
            )

        self._model_id = model_id
        self._transport = transport
        self._deployment_id = deployment_id

    @property
    def provider_id(self) -> str:
        """Return stable provider identity."""
        return "watsonx"

    @property
    def runtime_id(self) -> str:
        """Return stable adapter runtime identity."""
        return "watsonx-enterprise-adapter"

    @property
    def model_id(self) -> str:
        """Return configured watsonx model identity."""
        return self._model_id

    @property
    def deployment_id(self) -> str | None:
        """Return optional watsonx deployment identity."""
        return self._deployment_id

    def generate(
        self,
        request: ProviderInferenceRequest,
    ) -> ProviderInferenceResponse:
        """Execute one portable request through the watsonx boundary."""
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

        messages = tuple(
            {
                "role": message.role.value,
                "content": message.content,
            }
            for message in request.messages
        )

        transport_metadata: dict[str, JsonValue] = dict(
            request.metadata
        )

        if self._deployment_id is not None:
            transport_metadata["deployment_id"] = (
                self._deployment_id
            )

        try:
            result = self._transport.generate(
                model_id=self._model_id,
                messages=messages,
                max_output_tokens=request.max_output_tokens,
                temperature=request.temperature,
                metadata=transport_metadata,
            )
        except Exception as exc:
            return self._error_response(
                request=request,
                started=started,
                error_type="watsonx_transport_error",
                message=(
                    str(exc).strip()
                    or type(exc).__name__
                ),
                retryable=True,
            )

        total_latency_ms = (
            perf_counter_ns() - started
        ) / 1_000_000

        usage = self._build_usage(
            result
        )

        metadata: dict[str, JsonValue] = dict(
            result.metadata
        )
        metadata.update(
            {
                "ttft_available": False,
                "generation_latency_available": False,
            }
        )

        if self._deployment_id is not None:
            metadata["deployment_id"] = (
                self._deployment_id
            )

        if result.provider_request_id is not None:
            metadata["provider_request_id"] = (
                result.provider_request_id
            )

        if result.stop_reason is not None:
            metadata["stop_reason"] = (
                result.stop_reason
            )

        return ProviderInferenceResponse(
            request_id=request.request_id,
            provider_id=self.provider_id,
            runtime_id=self.runtime_id,
            model_id=result.model_id,
            model_revision=result.model_revision,
            output_text=result.output_text,
            usage=usage,
            timing=ProviderTiming(
                total_latency_ms=total_latency_ms,
                time_to_first_token_ms=None,
                generation_latency_ms=None,
            ),
            cost_usd=None,
            metadata=metadata,
        )

    @staticmethod
    def _build_usage(
        result: WatsonxTransportResponse,
    ) -> ProviderTokenUsage | None:
        """Return portable token usage only when both counts are available."""
        if (
            result.input_tokens is None
            or result.output_tokens is None
        ):
            return None

        return ProviderTokenUsage(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

    def _error_response(
        self,
        *,
        request: ProviderInferenceRequest,
        started: int,
        error_type: str,
        message: str,
        retryable: bool,
    ) -> ProviderInferenceResponse:
        """Return one normalised watsonx failure."""
        total_latency_ms = (
            perf_counter_ns() - started
        ) / 1_000_000

        metadata: dict[str, JsonValue] = {
            "ttft_available": False,
            "generation_latency_available": False,
        }

        if self._deployment_id is not None:
            metadata["deployment_id"] = (
                self._deployment_id
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
