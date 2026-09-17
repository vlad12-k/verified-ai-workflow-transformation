"""Benchmark pinned local Qwen through the provider-neutral M4 bridge."""

from hashlib import sha256
from time import perf_counter_ns

from vait.inference.models import (
    InferenceConfiguration,
    InferenceSetupEvidence,
    InferenceWorkload,
)
from vait.inference.providers.benchmark import (
    run_provider_generative_benchmark,
)
from vait.inference.providers.huggingface import (
    LocalHuggingFaceProvider,
)
from vait.inference.providers.models import (
    InferenceMessage,
    InferenceRole,
    ProviderInferenceRequest,
)
from vait.inference.reporting import (
    write_inference_report,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = (
    "7ae557604adf67be50417f59c2c2f167def9a775"
)
DEVICE = "cpu"

MAX_OUTPUT_TOKENS = 32
WARMUP_ITERATIONS = 1
MEASURED_ITERATIONS = 5

ARTIFACT_PATH = (
    "artifacts/benchmarks/"
    "m4-provider-qwen-inference-v0.1.json"
)

SYSTEM_MESSAGE = (
    "Follow the user instruction exactly and respond concisely."
)
USER_MESSAGE = (
    "State that this is a controlled provider-neutral "
    "inference benchmark."
)


def main() -> None:
    """Run one real provider-neutral local Qwen benchmark."""
    request = ProviderInferenceRequest(
        request_id="m4-b6-local-qwen-v0.1",
        model_id=MODEL_ID,
        messages=(
            InferenceMessage(
                role=InferenceRole.SYSTEM,
                content=SYSTEM_MESSAGE,
            ),
            InferenceMessage(
                role=InferenceRole.USER,
                content=USER_MESSAGE,
            ),
        ),
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.0,
        metadata={
            "benchmark_scope": (
                "provider-runtime-integration"
            ),
            "quality_claim": False,
        },
    )

    fingerprint_source = (
        f"{MODEL_ID}\n"
        f"{MODEL_REVISION}\n"
        f"{SYSTEM_MESSAGE}\n"
        f"{USER_MESSAGE}\n"
        f"{MAX_OUTPUT_TOKENS}\n"
        "temperature=0.0"
    )

    workload = InferenceWorkload(
        workload_id="m4-b6-provider-qwen",
        version="0.1",
        item_count=1,
        fingerprint_sha256=sha256(
            fingerprint_source.encode("utf-8")
        ).hexdigest(),
        metadata={
            "scope": "provider-runtime-integration",
            "quality_evaluation": False,
        },
    )

    setup_started = perf_counter_ns()

    provider = LocalHuggingFaceProvider(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        device=DEVICE,
        local_files_only=True,
    )

    setup_duration_ms = (
        perf_counter_ns() - setup_started
    ) / 1_000_000

    setup = InferenceSetupEvidence(
        duration_ms=setup_duration_ms,
        scope="local-huggingface-provider-preparation",
        included_operations=[
            "tokenizer-load",
            "model-load",
            "device-placement",
            "evaluation-mode",
        ],
        metadata={
            "inference_excluded": True,
            "local_files_only": True,
        },
    )

    configuration = InferenceConfiguration(
        provider=provider.provider_id,
        runtime=provider.runtime_id,
        device=DEVICE,
        dtype="framework-default",
        batch_size=1,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        metadata={
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0.0,
            "local_files_only": True,
            "streaming": False,
        },
    )

    report = run_provider_generative_benchmark(
        provider=provider,
        request=request,
        configuration=configuration,
        workload=workload,
        setup=setup,
        warmup_iterations=WARMUP_ITERATIONS,
        measured_iterations=MEASURED_ITERATIONS,
        evidence_metadata={
            "experiment": "m4-b6",
            "provider_path": (
                "local-huggingface-provider"
            ),
            "quality_claim": False,
            "measurement_interpretation": (
                "runtime-performance-evidence-only"
            ),
        },
    )

    artifact_path = write_inference_report(
        report,
        ARTIFACT_PATH,
    )

    print("M4-B6 provider-neutral local Qwen benchmark")
    print()
    print(f"Provider: {provider.provider_id}")
    print(f"Runtime: {provider.runtime_id}")
    print(f"Model: {MODEL_ID}")
    print(f"Revision: {MODEL_REVISION}")
    print(f"Device: {DEVICE}")
    print()
    print(
        f"Setup duration: "
        f"{setup_duration_ms:.2f} ms"
    )
    print(
        f"Warm-up iterations: "
        f"{report.warmup_iterations}"
    )
    print(
        f"Measured iterations: "
        f"{report.measured_iterations}"
    )
    print()
    print(
        f"Request p50: "
        f"{report.latency.p50_ms:.2f} ms"
    )
    print(
        f"Request p95: "
        f"{report.latency.p95_ms:.2f} ms"
    )
    print(
        f"Request p99: "
        f"{report.latency.p99_ms:.2f} ms"
    )

    if (
        report.generative is not None
        and report.generative.generation_latency
        is not None
    ):
        print(
            f"Generation p50: "
            f"{report.generative.generation_latency.p50_ms:.2f} ms"
        )
        print(
            f"Generation p95: "
            f"{report.generative.generation_latency.p95_ms:.2f} ms"
        )

    if report.generative is not None:
        print(
            f"Input tokens: "
            f"{report.generative.input_tokens}"
        )
        print(
            f"Output tokens: "
            f"{report.generative.output_tokens}"
        )

    print(
        f"Requests/sec: "
        f"{report.throughput.requests_per_second}"
    )
    print(
        f"Output tokens/sec: "
        f"{report.throughput.tokens_per_second}"
    )

    print(
        "TTFT: unavailable for blocking local "
        "generate() backend"
    )
    print()
    print(f"Run ID: {report.run_id}")
    print(f"Evidence report: {artifact_path}")


if __name__ == "__main__":
    main()
