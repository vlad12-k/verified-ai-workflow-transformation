"""Build controlled local Qwen generative inference evidence for M4."""

import hashlib
import json
from pathlib import Path
from time import perf_counter_ns

from vait.inference.benchmark import (
    run_controlled_generative_benchmark,
)
from vait.inference.models import (
    GenerativeInferenceSample,
    InferenceConfiguration,
    InferenceSetupEvidence,
    InferenceWorkload,
)
from vait.inference.reporting import write_inference_report
from vait.rag.huggingface import HuggingFaceCausalGenerator
from vait.rag.models import (
    RAGContextDocument,
    RAGGenerationRequest,
)
from vait.rag.prompt_contracts import RAGPromptContract
from vait.retrieval.benchmark import load_retrieval_benchmark

DATASET_PATH = Path(
    "datasets/ap_policy_context/v0.1/benchmark.json"
)

ARTIFACT_PATH = Path(
    "artifacts/benchmarks/"
    "m4-ap-qwen-generative-inference-v0.1.json"
)

GENERATOR_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

GENERATOR_REVISION = (
    "7ae557604adf67be50417f59c2c2f167def9a775"
)

DEVICE = "cpu"
MAX_NEW_TOKENS = 32
WARMUP_ITERATIONS = 1
MEASURED_ITERATIONS = 5

QUERY_ID = "duplicate-01"
CONTEXT_DOCUMENT_ID = "duplicate-invoice-policy"


def workload_fingerprint(
    *,
    query: str,
    context_text: str,
) -> str:
    """Return deterministic identity for the controlled generation workload."""
    payload = {
        "query": query,
        "context_document_id": CONTEXT_DOCUMENT_ID,
        "context_text": context_text,
        "prompt_contract": RAGPromptContract.STRICT_V2.value,
    }

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    """Benchmark pinned local Qwen generation with exact token evidence."""
    dataset = load_retrieval_benchmark(
        DATASET_PATH
    )

    query = next(
        item
        for item in dataset.queries
        if item.query_id == QUERY_ID
    )

    context_document = next(
        item
        for item in dataset.documents
        if item.document_id == CONTEXT_DOCUMENT_ID
    )

    if CONTEXT_DOCUMENT_ID not in query.relevant_document_ids:
        raise RuntimeError(
            "Selected context document is not declared relevant "
            "for the benchmark query."
        )

    request = RAGGenerationRequest(
        query=query.text,
        context_documents=(
            RAGContextDocument(
                document_id=context_document.document_id,
                text=context_document.text,
                score=1.0,
                rank=1,
            ),
        ),
    )

    setup_started = perf_counter_ns()

    generator = HuggingFaceCausalGenerator(
        model_id=GENERATOR_MODEL_ID,
        revision=GENERATOR_REVISION,
        device=DEVICE,
        minimum_score=0.4,
        max_new_tokens=MAX_NEW_TOKENS,
        local_files_only=True,
        prompt_contract=RAGPromptContract.STRICT_V2,
    )

    setup_duration_ms = (
        perf_counter_ns() - setup_started
    ) / 1_000_000

    observed_answers: list[str] = []
    observed_abstentions: list[bool] = []

    def operation() -> GenerativeInferenceSample:
        evidence = generator.generate_with_evidence(
            request
        )

        observed_answers.append(
            evidence.generation.answer
        )
        observed_abstentions.append(
            evidence.generation.abstained
        )

        return GenerativeInferenceSample(
            time_to_first_token_ms=None,
            generation_latency_ms=(
                evidence.model_generation_latency_ms
            ),
            input_tokens=evidence.input_tokens,
            output_tokens=evidence.output_tokens,
        )

    report = run_controlled_generative_benchmark(
        candidate_implementation_id=generator.implementation_id,
        operation=operation,
        task_family="grounded-generation",
        configuration=InferenceConfiguration(
            provider="local",
            runtime="huggingface-transformers",
            device=DEVICE,
            dtype="framework-default",
            batch_size=1,
            model_id=GENERATOR_MODEL_ID,
            model_revision=GENERATOR_REVISION,
            metadata={
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": False,
                "local_files_only": True,
                "prompt_contract": (
                    RAGPromptContract.STRICT_V2.value
                ),
                "ttft_available": False,
            },
        ),
        workload=InferenceWorkload(
            workload_id="ap-policy-generation",
            version="0.1",
            item_count=1,
            fingerprint_sha256=workload_fingerprint(
                query=query.text,
                context_text=context_document.text,
            ),
            metadata={
                "dataset_id": dataset.benchmark_id,
                "dataset_version": dataset.version,
                "query_id": QUERY_ID,
                "context_document_id": (
                    CONTEXT_DOCUMENT_ID
                ),
                "retrieval_included": False,
            },
        ),
        setup=InferenceSetupEvidence(
            duration_ms=setup_duration_ms,
            scope="local-huggingface-model-preparation",
            included_operations=[
                "tokenizer-load",
                "model-load",
                "device-placement",
                "evaluation-mode",
            ],
            metadata={
                "included_in_inference_latency": False,
                "local_files_only": True,
            },
        ),
        warmup_iterations=WARMUP_ITERATIONS,
        measured_iterations=MEASURED_ITERATIONS,
        evidence_metadata={
            "benchmark_scope": (
                "controlled-local-generation-only"
            ),
            "retrieval_latency_included": False,
            "embedding_latency_included": False,
            "ttft_status": (
                "unavailable-blocking-generation-backend"
            ),
        },
    )

    unique_answers = sorted(
        set(observed_answers)
    )
    unique_abstention_states = sorted(
        set(observed_abstentions)
    )

    report.evidence_metadata[
        "observed_unique_answer_count"
    ] = len(unique_answers)

    report.evidence_metadata[
        "observed_abstention_states"
    ] = unique_abstention_states

    report_path = write_inference_report(
        report=report,
        path=ARTIFACT_PATH,
    )

    generative = report.generative

    if generative is None:
        raise RuntimeError(
            "Expected generative inference evidence."
        )

    if generative.generation_latency is None:
        raise RuntimeError(
            "Expected generation latency evidence."
        )

    tokens_per_second = (
        report.throughput.tokens_per_second
    )
    requests_per_second = (
        report.throughput.requests_per_second
    )

    if tokens_per_second is None:
        raise RuntimeError(
            "Expected output tokens/sec evidence."
        )

    if requests_per_second is None:
        raise RuntimeError(
            "Expected requests/sec evidence."
        )

    print("M4-A local Qwen generative inference benchmark")
    print()
    print(f"Model: {GENERATOR_MODEL_ID}")
    print(f"Revision: {GENERATOR_REVISION}")
    print(f"Device: {DEVICE}")
    print(
        "Prompt contract: "
        f"{RAGPromptContract.STRICT_V2.value}"
    )
    print()
    print(
        f"Setup duration: {setup_duration_ms:.2f} ms"
    )
    print(
        f"Warm-up iterations: {report.warmup_iterations}"
    )
    print(
        f"Measured iterations: {report.measured_iterations}"
    )
    print()
    print(
        f"Request p50: {report.latency.p50_ms:.2f} ms"
    )
    print(
        f"Request p95: {report.latency.p95_ms:.2f} ms"
    )
    print(
        f"Request p99: {report.latency.p99_ms:.2f} ms"
    )
    print(
        "Generation p50: "
        f"{generative.generation_latency.p50_ms:.2f} ms"
    )
    print(
        "Generation p95: "
        f"{generative.generation_latency.p95_ms:.2f} ms"
    )
    print()
    print(
        f"Input tokens: {generative.input_tokens}"
    )
    print(
        f"Output tokens: {generative.output_tokens}"
    )
    print(
        f"Output tokens/sec: {tokens_per_second:.2f}"
    )
    print(
        f"Requests/sec: {requests_per_second:.4f}"
    )
    print("TTFT: unavailable for blocking generate() backend")
    print()
    print(
        "Observed unique answers: "
        f"{len(unique_answers)}"
    )
    print(f"Run ID: {report.run_id}")
    print(f"Evidence report: {report_path}")


if __name__ == "__main__":
    main()
