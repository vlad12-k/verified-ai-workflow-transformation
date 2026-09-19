"""Provider-neutral typed inference request and response models."""

from enum import StrEnum

from pydantic import BaseModel, Field, JsonValue


class InferenceRole(StrEnum):
    """Portable conversational roles supported by provider adapters."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class InferenceMessage(BaseModel):
    """One provider-neutral conversational message."""

    role: InferenceRole
    content: str = Field(min_length=1)


class ProviderInferenceRequest(BaseModel):
    """Portable generative inference request."""

    request_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)

    messages: tuple[InferenceMessage, ...] = Field(
        min_length=1,
    )

    max_output_tokens: int = Field(ge=1)
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        allow_inf_nan=False,
    )

    metadata: dict[str, JsonValue] = Field(
        default_factory=dict,
    )


class ProviderTokenUsage(BaseModel):
    """Normalised token usage returned by an inference provider."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)

    @property
    def total_tokens(self) -> int:
        """Return total input and output token usage."""
        return self.input_tokens + self.output_tokens


class ProviderTiming(BaseModel):
    """Normalised latency evidence returned by an inference provider."""

    total_latency_ms: float = Field(
        ge=0.0,
        allow_inf_nan=False,
    )

    time_to_first_token_ms: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
    )

    generation_latency_ms: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
    )


class ProviderError(BaseModel):
    """Normalised provider failure without leaking SDK-specific exceptions."""

    error_type: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool = False

    metadata: dict[str, JsonValue] = Field(
        default_factory=dict,
    )


class ProviderInferenceResponse(BaseModel):
    """Normalised result produced by any supported inference provider."""

    request_id: str = Field(min_length=1)

    provider_id: str = Field(min_length=1)
    runtime_id: str = Field(min_length=1)

    model_id: str = Field(min_length=1)
    model_revision: str | None = None

    output_text: str | None = None

    usage: ProviderTokenUsage | None = None
    timing: ProviderTiming

    cost_usd: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
    )

    error: ProviderError | None = None

    metadata: dict[str, JsonValue] = Field(
        default_factory=dict,
    )

    @property
    def succeeded(self) -> bool:
        """Return whether the provider completed without a normalised error."""
        return self.error is None
