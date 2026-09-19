"""Bridge provider-neutral inference into controlled M4 benchmarking."""

from collections.abc import Mapping
from threading import Lock

from pydantic import JsonValue

from vait.inference.batching import (
    run_controlled_native_generative_batch_benchmark,
)
from vait.inference.benchmark import (
    run_controlled_generative_benchmark,
)
from vait.inference.concurrency import (
    run_controlled_concurrent_generative_benchmark,
)
from vait.inference.models import (
    GenerativeInferenceSample,
    InferenceBenchmarkReport,
    InferenceConfiguration,
    InferenceEnvironment,
    InferenceSetupEvidence,
    InferenceWorkload,
)
from vait.inference.providers.base import (
    BatchInferenceProvider,
    InferenceProvider,
)
from vait.inference.providers.models import (
    ProviderInferenceRequest,
    ProviderInferenceResponse,
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
    evidence_metadata: Mapping[
        str,
        JsonValue,
    ]
    | None = None,
) -> InferenceBenchmarkReport:
    """Benchmark any portable provider through the shared M4-A harness."""
    _validate_provider_configuration(
        provider=provider,
        request=request,
        configuration=configuration,
    )

    metadata: dict[
        str,
        JsonValue,
    ] = dict(
        evidence_metadata or {}
    )

    metadata.update(
        {
            "provider_id": (
                provider.provider_id
            ),
            "runtime_id": (
                provider.runtime_id
            ),
            "provider_request_id": (
                request.request_id
            ),
            "provider_model_id": (
                request.model_id
            ),
            "provider_bridge": (
                "portable-generative-v1"
            ),
            "token_usage_required": True,
        }
    )

    def operation() -> GenerativeInferenceSample:
        response = provider.generate(
            request
        )

        return _provider_response_to_sample(
            response
        )

    return run_controlled_generative_benchmark(
        candidate_implementation_id=(
            _candidate_implementation_id(
                provider=provider,
                request=request,
            )
        ),
        operation=operation,
        task_family=(
            "provider-generative-inference"
        ),
        configuration=configuration,
        workload=workload,
        setup=setup,
        warmup_iterations=(
            warmup_iterations
        ),
        measured_iterations=(
            measured_iterations
        ),
        environment=environment,
        evidence_metadata=metadata,
    )


def run_provider_concurrent_generative_benchmark(
    *,
    provider: InferenceProvider,
    request: ProviderInferenceRequest,
    configuration: InferenceConfiguration,
    concurrency: int,
    warmup_rounds: int,
    measured_rounds: int,
    workload: InferenceWorkload | None = None,
    setup: InferenceSetupEvidence | None = None,
    environment: InferenceEnvironment | None = None,
    evidence_metadata: Mapping[
        str,
        JsonValue,
    ]
    | None = None,
) -> InferenceBenchmarkReport:
    """Benchmark concurrent independent requests through one provider."""
    _validate_provider_configuration(
        provider=provider,
        request=request,
        configuration=configuration,
    )

    metadata: dict[
        str,
        JsonValue,
    ] = dict(
        evidence_metadata or {}
    )

    metadata.update(
        {
            "provider_id": (
                provider.provider_id
            ),
            "runtime_id": (
                provider.runtime_id
            ),
            "provider_base_request_id": (
                request.request_id
            ),
            "provider_model_id": (
                request.model_id
            ),
            "provider_bridge": (
                "portable-concurrent-generative-v1"
            ),
            "token_usage_required": True,
            "request_id_strategy": (
                "base-plus-monotonic-invocation"
            ),
        }
    )

    invocation_lock = Lock()
    invocation_index = 0

    def operation(
        worker_index: int,
    ) -> GenerativeInferenceSample:
        nonlocal invocation_index

        with invocation_lock:
            current_invocation = (
                invocation_index
            )

            invocation_index += 1

        concurrent_request = (
            request.model_copy(
                update={
                    "request_id": (
                        f"{request.request_id}-"
                        f"concurrent-"
                        f"{current_invocation:06d}"
                    ),
                    "metadata": {
                        **request.metadata,
                        "concurrency_worker_index": (
                            worker_index
                        ),
                        "concurrency_invocation_index": (
                            current_invocation
                        ),
                    },
                }
            )
        )

        response = provider.generate(
            concurrent_request
        )

        if (
            response.request_id
            != concurrent_request.request_id
        ):
            raise RuntimeError(
                "Provider response request_id does not "
                "match the concurrent request."
            )

        return _provider_response_to_sample(
            response
        )

    return (
        run_controlled_concurrent_generative_benchmark(
            candidate_implementation_id=(
                _candidate_implementation_id(
                    provider=provider,
                    request=request,
                )
            ),
            operation=operation,
            task_family=(
                "provider-generative-inference"
            ),
            configuration=configuration,
            concurrency=concurrency,
            warmup_rounds=(
                warmup_rounds
            ),
            measured_rounds=(
                measured_rounds
            ),
            workload=workload,
            setup=setup,
            environment=environment,
            evidence_metadata=metadata,
        )
    )




def run_provider_native_batch_generative_benchmark(
    *,
    provider: BatchInferenceProvider,
    request: ProviderInferenceRequest,
    configuration: InferenceConfiguration,
    warmup_rounds: int,
    measured_rounds: int,
    workload: InferenceWorkload | None = None,
    setup: InferenceSetupEvidence | None = None,
    environment: InferenceEnvironment | None = None,
    evidence_metadata: Mapping[
        str,
        JsonValue,
    ]
    | None = None,
) -> InferenceBenchmarkReport:
    """Benchmark genuine native batching through one portable provider."""
    _validate_provider_configuration(
        provider=provider,
        request=request,
        configuration=configuration,
    )

    batch_size = (
        configuration.batch_size
    )

    metadata: dict[
        str,
        JsonValue,
    ] = dict(
        evidence_metadata or {}
    )

    metadata.update(
        {
            "provider_id": (
                provider.provider_id
            ),
            "runtime_id": (
                provider.runtime_id
            ),
            "provider_base_request_id": (
                request.request_id
            ),
            "provider_model_id": (
                request.model_id
            ),
            "provider_bridge": (
                "portable-native-batch-generative-v1"
            ),
            "token_usage_required": True,
            "request_id_strategy": (
                "base-plus-batch-invocation-and-item"
            ),
            "native_batch_required": True,
        }
    )

    batch_invocation_index = 0

    def operation() -> tuple[
        GenerativeInferenceSample,
        ...,
    ]:
        nonlocal batch_invocation_index

        current_batch_index = (
            batch_invocation_index
        )

        batch_invocation_index += 1

        requests = tuple(
            request.model_copy(
                update={
                    "request_id": (
                        f"{request.request_id}-"
                        f"batch-"
                        f"{current_batch_index:06d}-"
                        f"item-{item_index:03d}"
                    ),
                    "metadata": {
                        **request.metadata,
                        "native_batch_invocation_index": (
                            current_batch_index
                        ),
                        "native_batch_item_index": (
                            item_index
                        ),
                        "native_batch_size": (
                            batch_size
                        ),
                    },
                }
            )
            for item_index
            in range(
                batch_size
            )
        )

        responses = (
            provider.generate_batch(
                requests
            )
        )

        if len(
            responses
        ) != batch_size:
            raise RuntimeError(
                "Provider native batch returned "
                f"{len(responses)} responses "
                f"for batch_size={batch_size}."
            )

        samples: list[
            GenerativeInferenceSample
        ] = []

        for (
            batch_request,
            response,
        ) in zip(
            requests,
            responses,
            strict=True,
        ):
            if (
                response.request_id
                != batch_request.request_id
            ):
                raise RuntimeError(
                    "Provider native-batch response "
                    "request_id does not match "
                    "its exact request."
                )

            samples.append(
                _provider_response_to_sample(
                    response
                )
            )

        return tuple(
            samples
        )

    return (
        run_controlled_native_generative_batch_benchmark(
            candidate_implementation_id=(
                _candidate_implementation_id(
                    provider=provider,
                    request=request,
                )
            ),
            operation=operation,
            task_family=(
                "provider-generative-inference"
            ),
            configuration=configuration,
            warmup_rounds=(
                warmup_rounds
            ),
            measured_rounds=(
                measured_rounds
            ),
            workload=workload,
            setup=setup,
            environment=environment,
            evidence_metadata=metadata,
        )
    )


def _provider_response_to_sample(
    response: ProviderInferenceResponse,
) -> GenerativeInferenceSample:
    """Convert one normalised provider response into benchmark evidence."""
    if not response.succeeded:
        error = response.error

        if error is None:
            raise RuntimeError(
                "Provider reported failure without "
                "error evidence."
            )

        raise RuntimeError(
            "Provider inference failed: "
            f"{error.error_type}: "
            f"{error.message}"
        )

    if response.usage is None:
        raise RuntimeError(
            "Provider response did not include token usage; "
            "controlled generative token evidence cannot be "
            "constructed without observed usage."
        )

    return GenerativeInferenceSample(
        time_to_first_token_ms=(
            response
            .timing
            .time_to_first_token_ms
        ),
        generation_latency_ms=(
            response
            .timing
            .generation_latency_ms
        ),
        input_tokens=(
            response.usage.input_tokens
        ),
        output_tokens=(
            response.usage.output_tokens
        ),
    )


def _candidate_implementation_id(
    *,
    provider: InferenceProvider,
    request: ProviderInferenceRequest,
) -> str:
    """Return the stable provider benchmark candidate identity."""
    return (
        f"provider:{provider.provider_id}:"
        f"runtime:{provider.runtime_id}:"
        f"model:{request.model_id}"
    )


def _validate_provider_configuration(
    *,
    provider: InferenceProvider,
    request: ProviderInferenceRequest,
    configuration: InferenceConfiguration,
) -> None:
    """Prevent contradictory provider evidence from entering a report."""
    if (
        configuration.provider
        != provider.provider_id
    ):
        raise ValueError(
            "Inference configuration provider does not "
            "match the provider implementation."
        )

    if (
        configuration.runtime
        != provider.runtime_id
    ):
        raise ValueError(
            "Inference configuration runtime does not "
            "match the provider implementation."
        )

    if (
        configuration.model_id
        is not None
        and configuration.model_id
        != request.model_id
    ):
        raise ValueError(
            "Inference configuration model_id does not "
            "match the provider request."
        )
