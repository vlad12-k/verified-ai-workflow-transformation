"""Bridge provider-neutral inference into controlled M4 benchmarking."""

from collections.abc import Mapping

from pydantic import JsonValue

from vait.inference.benchmark import (
    run_controlled_generative_benchmark,
)
from vait.inference.models import (
    GenerativeInferenceSample,
    InferenceBenchmarkReport,
    InferenceConfiguration,
    InferenceEnvironment,
    InferenceSetupEvidence,
    InferenceWorkload,
)
from vait.inference.providers.base import InferenceProvider
from vait.inference.providers.models import (
    ProviderInferenceRequest,
)


def run_provider_generative_benchmark(
    *,
    provider: InferenceProvider,
    request: ProviderInferenceRequest,
    configuration: InferenceConfiguration,
    warmup_iterations: int,
    measured_iterations: int,
    workload: InferenceWorkload | None = None,
    setup: InferenceSetupEvidence | None = None,
    environment: InferenceEnvironment | None = None,
    evidence_metadata: Mapping[str, JsonValue] | None = None,
) -> InferenceBenchmarkReport:
    """Benchmark any portable provider through the shared M4-A harness."""
    _validate_provider_configuration(
        provider=provider,
        request=request,
        configuration=configuration,
    )

    metadata: dict[str, JsonValue] = dict(
        evidence_metadata or {}
    )
    metadata.update(
        {
            "provider_id": provider.provider_id,
            "runtime_id": provider.runtime_id,
            "provider_request_id": request.request_id,
            "provider_model_id": request.model_id,
            "provider_bridge": "portable-generative-v1",
            "token_usage_required": True,
        }
    )

    def operation() -> GenerativeInferenceSample:
        response = provider.generate(
            request
        )

        if not response.succeeded:
            error = response.error

            if error is None:
                raise RuntimeError(
                    "Provider reported failure without error evidence."
                )

            raise RuntimeError(
                "Provider inference failed: "
                f"{error.error_type}: {error.message}"
            )

        if response.usage is None:
            raise RuntimeError(
                "Provider response did not include token usage; "
                "controlled generative token evidence cannot be "
                "constructed without observed usage."
            )

        return GenerativeInferenceSample(
            time_to_first_token_ms=(
                response.timing.time_to_first_token_ms
            ),
            generation_latency_ms=(
                response.timing.generation_latency_ms
            ),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    candidate_implementation_id = (
        f"provider:{provider.provider_id}:"
        f"runtime:{provider.runtime_id}:"
        f"model:{request.model_id}"
    )

    return run_controlled_generative_benchmark(
        candidate_implementation_id=(
            candidate_implementation_id
        ),
        operation=operation,
        task_family="provider-generative-inference",
        configuration=configuration,
        workload=workload,
        setup=setup,
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
        environment=environment,
        evidence_metadata=metadata,
    )


def _validate_provider_configuration(
    *,
    provider: InferenceProvider,
    request: ProviderInferenceRequest,
    configuration: InferenceConfiguration,
) -> None:
    """Prevent contradictory provider evidence from entering a report."""
    if configuration.provider != provider.provider_id:
        raise ValueError(
            "Inference configuration provider does not match "
            "the provider implementation."
        )

    if configuration.runtime != provider.runtime_id:
        raise ValueError(
            "Inference configuration runtime does not match "
            "the provider implementation."
        )

    if (
        configuration.model_id is not None
        and configuration.model_id != request.model_id
    ):
        raise ValueError(
            "Inference configuration model_id does not match "
            "the provider request."
        )
