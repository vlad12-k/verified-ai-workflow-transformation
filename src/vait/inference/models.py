"""Typed evidence models for controlled inference benchmarking."""

from datetime import datetime

from pydantic import BaseModel, Field, JsonValue


class InferenceConfiguration(BaseModel):
    """Execution configuration attached to one inference benchmark run."""

    provider: str = Field(min_length=1)
    runtime: str = Field(min_length=1)
    device: str = Field(min_length=1)
    dtype: str = Field(min_length=1)
    batch_size: int = Field(ge=1)

    model_id: str | None = None
    model_revision: str | None = None

    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class InferenceLatencySummary(BaseModel):
    """Latency distribution for measured inference observations."""

    count: int = Field(ge=1)

    mean_ms: float = Field(ge=0.0, allow_inf_nan=False)
    p50_ms: float = Field(ge=0.0, allow_inf_nan=False)
    p95_ms: float = Field(ge=0.0, allow_inf_nan=False)
    p99_ms: float = Field(ge=0.0, allow_inf_nan=False)
    max_ms: float = Field(ge=0.0, allow_inf_nan=False)


class InferenceThroughputSummary(BaseModel):
    """Normalised throughput measurements for one benchmark run."""

    cases_per_second: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
    )
    requests_per_second: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
    )
    tokens_per_second: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
    )


class GenerativeInferenceSample(BaseModel):
    """One generative inference observation."""

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

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class GenerativeInferenceSummary(BaseModel):
    """Aggregated generative inference evidence."""

    sample_count: int = Field(ge=1)

    time_to_first_token: InferenceLatencySummary | None = None
    generation_latency: InferenceLatencySummary | None = None

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class InferenceEnvironment(BaseModel):
    """Environment metadata required to interpret inference measurements."""

    python_version: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    system: str = Field(min_length=1)
    machine: str = Field(min_length=1)
    processor: str = Field(min_length=1)

    package_versions: dict[str, str] = Field(default_factory=dict)


class InferenceBenchmarkReport(BaseModel):
    """Versioned machine-readable controlled inference evidence."""

    schema_version: str = "0.1"

    run_id: str = Field(min_length=1)
    created_at: datetime

    candidate_implementation_id: str = Field(min_length=1)
    task_family: str = Field(min_length=1)

    warmup_iterations: int = Field(ge=0)
    measured_iterations: int = Field(ge=1)

    cold_start_latency_ms: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
    )

    configuration: InferenceConfiguration
    latency: InferenceLatencySummary
    throughput: InferenceThroughputSummary

    generative: GenerativeInferenceSummary | None = None
    environment: InferenceEnvironment

    evidence_metadata: dict[str, JsonValue] = Field(default_factory=dict)
