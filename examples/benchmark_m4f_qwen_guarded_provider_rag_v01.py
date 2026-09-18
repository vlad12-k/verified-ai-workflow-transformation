"""Benchmark semantic-guarded provider-neutral Qwen RAG."""

import json
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from vait.inference.benchmark import (
    capture_inference_environment,
)
from vait.inference.providers.base import (
    InferenceProvider,
)
from vait.inference.providers.huggingface import (
    LocalHuggingFaceProvider,
)
from vait.inference.providers.models import (
    ProviderInferenceRequest,
    ProviderInferenceResponse,
)
from vait.rag.benchmark import (
    evaluate_rag,
    load_rag_benchmark,
    validate_rag_benchmark_documents,
)
from vait.rag.extractive import (
    ExtractiveGroundedGenerator,
)
from vait.rag.guarded import (
    SemanticGuardedGenerator,
)
from vait.rag.models import (
    RAGGeneration,
    RAGGenerationRequest,
    RAGResult,
)
from vait.rag.pipeline import (
    RetrievalAugmentedGenerator,
)
from vait.rag.prompt_contracts import (
    INSUFFICIENT_EVIDENCE,
    RAGPromptContract,
)
from vait.rag.provider import (
    ProviderBackedRAGGenerator,
)
from vait.rag.semantic_guard import (
    assess_semantic_drift,
)
from vait.retrieval.benchmark import (
    load_retrieval_benchmark,
)
from vait.retrieval.in_memory import (
    InMemoryCosineRetriever,
)
from vait.retrieval.sentence_transformer import (
    SentenceTransformerEmbeddingEncoder,
)

EMBEDDING_MODEL_ID = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

EMBEDDING_REVISION = (
    "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)

GENERATOR_MODEL_ID = (
    "Qwen/"
    "Qwen2.5-0.5B-Instruct"
)

GENERATOR_REVISION = (
    "7ae557604adf67be50417f59c2c2f167def9a775"
)

RETRIEVAL_PATH = Path(
    "datasets/ap_policy_context/"
    "v0.1/benchmark.json"
)

RAG_PATH = Path(
    "datasets/ap_policy_rag/"
    "v0.1/benchmark.json"
)

ARTIFACT_PATH = Path(
    "artifacts/benchmarks/"
    "m4f-ap-policy-qwen2.5-0.5b-"
    "guarded-provider-rag-v0.1.json"
)

MINIMUM_SCORE = 0.4
MAX_NEW_TOKENS = 80

RETRIEVAL_K = 1
MAX_CONTEXT_DOCUMENTS = 1
MAX_CONTEXT_CHARS = 2000


class RecordingProvider:
    """Record exact provider responses."""

    def __init__(
        self,
        provider: InferenceProvider,
    ) -> None:
        """Wrap one inference provider."""
        self._provider = provider
        self.responses: list[
            ProviderInferenceResponse
        ] = []

    @property
    def provider_id(self) -> str:
        """Return provider identity."""
        return self._provider.provider_id

    @property
    def runtime_id(self) -> str:
        """Return runtime identity."""
        return self._provider.runtime_id

    def generate(
        self,
        request: ProviderInferenceRequest,
    ) -> ProviderInferenceResponse:
        """Generate and retain provider evidence."""
        response = self._provider.generate(
            request
        )

        self.responses.append(
            response
        )

        return response


class RecordingFallbackGenerator:
    """Record deterministic fallback routing."""

    def __init__(
        self,
        delegate: ExtractiveGroundedGenerator,
    ) -> None:
        """Store deterministic fallback."""
        self._delegate = delegate
        self.call_count = 0

    @property
    def implementation_id(self) -> str:
        """Return fallback implementation identity."""
        return self._delegate.implementation_id

    def generate(
        self,
        request: RAGGenerationRequest,
    ) -> RAGGeneration:
        """Record and execute one fallback."""
        self.call_count += 1

        return self._delegate.generate(
            request
        )


def _assessment_payload(
    *,
    context_text: str,
    answer_text: str,
) -> dict[str, Any]:
    """Return serialisable semantic guard evidence."""
    assessment = assess_semantic_drift(
        context_text=context_text,
        answer_text=answer_text,
    )

    return {
        "passed": assessment.passed,
        "issues": [
            {
                "risk": issue.risk.value,
                "marker": issue.marker,
                "detail": issue.detail,
            }
            for issue
            in assessment.issues
        ],
    }


class RecordingRAGRunner:
    """Record raw primary and surfaced guarded evidence."""

    def __init__(
        self,
        *,
        pipeline: RetrievalAugmentedGenerator,
        provider: RecordingProvider,
        fallback: RecordingFallbackGenerator,
    ) -> None:
        """Store evaluated components."""
        self._pipeline = pipeline
        self._provider = provider
        self._fallback = fallback

        self.traces: list[
            dict[str, Any]
        ] = []

    def run(
        self,
        query: str,
    ) -> RAGResult:
        """Run one guarded RAG query."""
        provider_before = len(
            self._provider.responses
        )

        fallback_before = (
            self._fallback.call_count
        )

        started = perf_counter_ns()

        result = self._pipeline.run(
            query
        )

        end_to_end_latency_ms = (
            perf_counter_ns()
            - started
        ) / 1_000_000

        new_responses = (
            self._provider.responses[
                provider_before:
            ]
        )

        if len(new_responses) > 1:
            raise RuntimeError(
                "One RAG query produced more "
                "than one provider response."
            )

        provider_response = (
            new_responses[0]
            if new_responses
            else None
        )

        fallback_used = (
            self._fallback.call_count
            > fallback_before
        )

        context_text = "\n\n".join(
            document.text
            for document
            in result.context_documents
        )

        raw_primary_semantic = None

        if (
            provider_response
            is not None
            and provider_response.succeeded
            and provider_response.output_text
            is not None
        ):
            raw_text = (
                provider_response
                .output_text
                .strip()
            )

            if (
                raw_text
                and raw_text
                != INSUFFICIENT_EVIDENCE
            ):
                raw_primary_semantic = (
                    _assessment_payload(
                        context_text=(
                            context_text
                        ),
                        answer_text=raw_text,
                    )
                )

        surfaced_semantic = None

        if not result.abstained:
            surfaced_semantic = (
                _assessment_payload(
                    context_text=(
                        context_text
                    ),
                    answer_text=(
                        result.answer
                    ),
                )
            )

        self.traces.append(
            {
                "query": query,
                "answer": result.answer,
                "abstained": (
                    result.abstained
                ),
                "provider_called": (
                    provider_response
                    is not None
                ),
                "fallback_used": (
                    fallback_used
                ),
                "cited_document_ids": list(
                    result.cited_document_ids
                ),
                "context": [
                    {
                        "document_id": (
                            document.document_id
                        ),
                        "score": (
                            document.score
                        ),
                        "rank": (
                            document.rank
                        ),
                        "text": (
                            document.text
                        ),
                    }
                    for document
                    in result.context_documents
                ],
                "end_to_end_latency_ms": (
                    end_to_end_latency_ms
                ),
                "raw_primary_semantic_guard": (
                    raw_primary_semantic
                ),
                "surfaced_semantic_guard": (
                    surfaced_semantic
                ),
                "provider_response": (
                    provider_response
                    .model_dump(
                        mode="json"
                    )
                    if provider_response
                    is not None
                    else None
                ),
            }
        )

        return result


def main() -> None:
    """Execute guarded provider-neutral Qwen RAG."""
    retrieval_dataset = (
        load_retrieval_benchmark(
            RETRIEVAL_PATH
        )
    )

    rag_dataset = (
        load_rag_benchmark(
            RAG_PATH
        )
    )

    validate_rag_benchmark_documents(
        rag_dataset,
        known_document_ids={
            document.document_id
            for document
            in retrieval_dataset.documents
        },
    )

    print(
        "=== M4-F GUARDED SLM/RAG ==="
    )
    print(
        "generator:",
        GENERATOR_MODEL_ID,
    )
    print(
        "revision:",
        GENERATOR_REVISION,
    )
    print(
        "prompt_contract:",
        RAGPromptContract.STRICT_V2.value,
    )

    print()
    print(
        "=== SETUP ==="
    )

    retrieval_started = (
        perf_counter_ns()
    )

    encoder = (
        SentenceTransformerEmbeddingEncoder(
            model_id=(
                EMBEDDING_MODEL_ID
            ),
            revision=(
                EMBEDDING_REVISION
            ),
            device="cpu",
            local_files_only=True,
        )
    )

    retriever = (
        InMemoryCosineRetriever(
            encoder=encoder,
            documents=(
                retrieval_dataset
                .text_documents()
            ),
        )
    )

    retrieval_setup_ms = (
        perf_counter_ns()
        - retrieval_started
    ) / 1_000_000

    provider_started = (
        perf_counter_ns()
    )

    local_provider = (
        LocalHuggingFaceProvider(
            model_id=(
                GENERATOR_MODEL_ID
            ),
            revision=(
                GENERATOR_REVISION
            ),
            device="cpu",
            local_files_only=True,
        )
    )

    provider_setup_ms = (
        perf_counter_ns()
        - provider_started
    ) / 1_000_000

    provider = RecordingProvider(
        local_provider
    )

    primary = (
        ProviderBackedRAGGenerator(
            provider=provider,
            model_id=(
                GENERATOR_MODEL_ID
            ),
            model_revision=(
                GENERATOR_REVISION
            ),
            minimum_score=(
                MINIMUM_SCORE
            ),
            max_new_tokens=(
                MAX_NEW_TOKENS
            ),
            prompt_contract=(
                RAGPromptContract
                .STRICT_V2
            ),
        )
    )

    fallback = (
        RecordingFallbackGenerator(
            ExtractiveGroundedGenerator(
                minimum_score=(
                    MINIMUM_SCORE
                ),
            )
        )
    )

    guarded = (
        SemanticGuardedGenerator(
            primary=primary,
            fallback=fallback,
        )
    )

    pipeline = (
        RetrievalAugmentedGenerator(
            retriever=retriever,
            generator=guarded,
            retrieval_k=(
                RETRIEVAL_K
            ),
            max_context_documents=(
                MAX_CONTEXT_DOCUMENTS
            ),
            max_context_chars=(
                MAX_CONTEXT_CHARS
            ),
        )
    )

    runner = RecordingRAGRunner(
        pipeline=pipeline,
        provider=provider,
        fallback=fallback,
    )

    print(
        "retrieval_setup_ms:",
        f"{retrieval_setup_ms:.3f}",
    )
    print(
        "provider_setup_ms:",
        f"{provider_setup_ms:.3f}",
    )

    print()
    print(
        "=== VERIFIED GUARDED PATH ==="
    )

    report = evaluate_rag(
        rag_dataset,
        runner,
    )

    provider_failures = [
        response
        for response
        in provider.responses
        if not response.succeeded
    ]

    successful_responses = [
        response
        for response
        in provider.responses
        if response.succeeded
    ]

    input_tokens = sum(
        response.usage.input_tokens
        for response
        in successful_responses
        if response.usage is not None
    )

    output_tokens = sum(
        response.usage.output_tokens
        for response
        in successful_responses
        if response.usage is not None
    )

    generation_latencies = [
        response
        .timing
        .generation_latency_ms
        for response
        in successful_responses
        if (
            response
            .timing
            .generation_latency_ms
            is not None
        )
    ]

    generation_seconds = (
        sum(
            generation_latencies
        )
        / 1000.0
    )

    tokens_per_second = (
        output_tokens
        / generation_seconds
        if generation_seconds > 0.0
        else None
    )

    raw_semantic = [
        trace[
            "raw_primary_semantic_guard"
        ]
        for trace
        in runner.traces
        if trace[
            "raw_primary_semantic_guard"
        ]
        is not None
    ]

    raw_semantic_passes = sum(
        1
        for result
        in raw_semantic
        if (
            isinstance(
                result,
                dict,
            )
            and result.get(
                "passed"
            )
            is True
        )
    )

    raw_semantic_pass_rate = (
        raw_semantic_passes
        / len(
            raw_semantic
        )
        if raw_semantic
        else 1.0
    )

    surfaced_semantic = [
        trace[
            "surfaced_semantic_guard"
        ]
        for trace
        in runner.traces
        if trace[
            "surfaced_semantic_guard"
        ]
        is not None
    ]

    surfaced_semantic_passes = sum(
        1
        for result
        in surfaced_semantic
        if (
            isinstance(
                result,
                dict,
            )
            and result.get(
                "passed"
            )
            is True
        )
    )

    surfaced_semantic_pass_rate = (
        surfaced_semantic_passes
        / len(
            surfaced_semantic
        )
        if surfaced_semantic
        else 1.0
    )

    fallback_count = sum(
        bool(
            trace[
                "fallback_used"
            ]
        )
        for trace
        in runner.traces
    )

    primary_surfaced_count = sum(
        (
            not bool(
                trace[
                    "abstained"
                ]
            )
            and not bool(
                trace[
                    "fallback_used"
                ]
            )
        )
        for trace
        in runner.traces
    )

    provider_skipped_count = sum(
        not bool(
            trace[
                "provider_called"
            ]
        )
        for trace
        in runner.traces
    )

    mean_latency_ms = (
        sum(
            float(
                trace[
                    "end_to_end_latency_ms"
                ]
            )
            for trace
            in runner.traces
        )
        / len(
            runner.traces
        )
    )

    print(
        "cases:",
        report.case_count,
    )
    print(
        "evidence_coverage:",
        f"{report.evidence_coverage:.3f}",
    )
    print(
        "citation_accuracy:",
        f"{report.citation_accuracy:.3f}",
    )
    print(
        "abstention_accuracy:",
        f"{report.abstention_accuracy:.3f}",
    )
    print(
        "answerable_success_rate:",
        f"{report.answerable_success_rate:.3f}",
    )
    print(
        "overall_success_rate:",
        f"{report.overall_success_rate:.3f}",
    )

    print()
    print(
        "=== GUARD ROUTING ==="
    )
    print(
        "provider_calls:",
        len(
            provider.responses
        ),
    )
    print(
        "provider_skipped_cases:",
        provider_skipped_count,
    )
    print(
        "provider_failures:",
        len(
            provider_failures
        ),
    )
    print(
        "primary_surfaced:",
        primary_surfaced_count,
    )
    print(
        "fallback_surfaced:",
        fallback_count,
    )
    print(
        "raw_primary_semantic_pass_rate:",
        f"{raw_semantic_pass_rate:.3f}",
    )
    print(
        "surfaced_semantic_pass_rate:",
        f"{surfaced_semantic_pass_rate:.3f}",
    )

    print()
    print(
        "=== INFERENCE ==="
    )
    print(
        "input_tokens:",
        input_tokens,
    )
    print(
        "output_tokens:",
        output_tokens,
    )

    if tokens_per_second is not None:
        print(
            "generation_tokens_per_second:",
            f"{tokens_per_second:.3f}",
        )

    print(
        "mean_end_to_end_latency_ms:",
        f"{mean_latency_ms:.3f}",
    )

    print()
    print(
        "=== PER CASE ==="
    )

    for (
        case,
        evaluation,
        trace,
    ) in zip(
        rag_dataset.cases,
        report.results,
        runner.traces,
        strict=True,
    ):
        print()
        print(
            f"[{case.case_id}]"
        )
        print(
            " expected_abstain=",
            case.expected_abstain,
            " observed_abstain=",
            evaluation.observed_abstain,
            " success=",
            evaluation.success,
            sep="",
        )
        print(
            " provider_called=",
            trace[
                "provider_called"
            ],
            " fallback_used=",
            trace[
                "fallback_used"
            ],
            sep="",
        )

        raw_guard = trace[
            "raw_primary_semantic_guard"
        ]

        if isinstance(
            raw_guard,
            dict,
        ):
            print(
                " raw_primary_guard=",
                raw_guard[
                    "passed"
                ],
                sep="",
            )

        surfaced_guard = trace[
            "surfaced_semantic_guard"
        ]

        if isinstance(
            surfaced_guard,
            dict,
        ):
            print(
                " surfaced_guard=",
                surfaced_guard[
                    "passed"
                ],
                sep="",
            )

        print(
            " answer=",
            trace[
                "answer"
            ],
            sep="",
        )

    payload: dict[
        str,
        Any,
    ] = {
        "schema_version": "0.1",
        "milestone": "M4-F",
        "experiment": (
            "qwen2.5-0.5b-provider-rag-"
            "semantic-guarded"
        ),
        "verify_first_optimise_second": True,
        "configuration": {
            "embedding_model_id": (
                EMBEDDING_MODEL_ID
            ),
            "embedding_revision": (
                EMBEDDING_REVISION
            ),
            "generator_model_id": (
                GENERATOR_MODEL_ID
            ),
            "generator_revision": (
                GENERATOR_REVISION
            ),
            "provider_id": (
                provider.provider_id
            ),
            "runtime_id": (
                provider.runtime_id
            ),
            "device": "cpu",
            "minimum_score": (
                MINIMUM_SCORE
            ),
            "max_new_tokens": (
                MAX_NEW_TOKENS
            ),
            "temperature": 0.0,
            "prompt_contract": (
                RAGPromptContract
                .STRICT_V2
                .value
            ),
            "retrieval_k": (
                RETRIEVAL_K
            ),
            "max_context_documents": (
                MAX_CONTEXT_DOCUMENTS
            ),
            "max_context_chars": (
                MAX_CONTEXT_CHARS
            ),
            "semantic_guard": True,
            "fallback": (
                fallback
                .implementation_id
            ),
        },
        "setup": {
            "retrieval_setup_ms": (
                retrieval_setup_ms
            ),
            "provider_setup_ms": (
                provider_setup_ms
            ),
        },
        "quality": (
            report.model_dump(
                mode="json"
            )
        ),
        "guard": {
            "raw_primary_evaluated": (
                len(
                    raw_semantic
                )
            ),
            "raw_primary_passes": (
                raw_semantic_passes
            ),
            "raw_primary_pass_rate": (
                raw_semantic_pass_rate
            ),
            "surfaced_evaluated": (
                len(
                    surfaced_semantic
                )
            ),
            "surfaced_passes": (
                surfaced_semantic_passes
            ),
            "surfaced_pass_rate": (
                surfaced_semantic_pass_rate
            ),
            "primary_surfaced": (
                primary_surfaced_count
            ),
            "fallback_surfaced": (
                fallback_count
            ),
        },
        "inference": {
            "provider_calls": len(
                provider.responses
            ),
            "provider_skipped_cases": (
                provider_skipped_count
            ),
            "provider_failures": len(
                provider_failures
            ),
            "input_tokens": (
                input_tokens
            ),
            "output_tokens": (
                output_tokens
            ),
            "generation_tokens_per_second": (
                tokens_per_second
            ),
            "mean_end_to_end_latency_ms": (
                mean_latency_ms
            ),
        },
        "environment": (
            capture_inference_environment()
            .model_dump(
                mode="json"
            )
        ),
        "traces": (
            runner.traces
        ),
        "observations": [
            (
                "Raw primary SLM outputs are assessed "
                "independently from surfaced guarded outputs."
            ),
            (
                "Semantic guard failures route to a "
                "deterministic extractive fallback."
            ),
            (
                "Fallback routing does not erase the "
                "recorded raw SLM inference evidence."
            ),
            (
                "Out-of-domain cases may abstain before "
                "provider inference when retrieval evidence "
                "is below the configured threshold."
            ),
        ],
    }

    ARTIFACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        f"Wrote {ARTIFACT_PATH}"
    )


if __name__ == "__main__":
    main()
