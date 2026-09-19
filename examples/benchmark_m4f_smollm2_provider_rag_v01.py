"""Run the first verified M4-F SLM/RAG inference experiment."""

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
from vait.rag.models import RAGResult
from vait.rag.pipeline import (
    RetrievalAugmentedGenerator,
)
from vait.rag.prompt_contracts import (
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
    "HuggingFaceTB/"
    "SmolLM2-135M-Instruct"
)

GENERATOR_REVISION = (
    "12fd25f77366fa6b3b4b768ec3050bf629380bac"
)

RETRIEVAL_DATASET_PATH = Path(
    "datasets/ap_policy_context/"
    "v0.1/benchmark.json"
)

RAG_DATASET_PATH = Path(
    "datasets/ap_policy_rag/"
    "v0.1/benchmark.json"
)

ARTIFACT_PATH = Path(
    "artifacts/benchmarks/"
    "m4f-ap-policy-smollm2-provider-rag-v0.1.json"
)

MINIMUM_SCORE = 0.4
MAX_NEW_TOKENS = 80

RETRIEVAL_K = 1
MAX_CONTEXT_DOCUMENTS = 1
MAX_CONTEXT_CHARS = 2000


class RecordingProvider:
    """Record provider-neutral responses without changing execution."""

    def __init__(
        self,
        provider: InferenceProvider,
    ) -> None:
        """Wrap one real provider."""
        self._provider = provider
        self.responses: list[
            ProviderInferenceResponse
        ] = []

    @property
    def provider_id(self) -> str:
        """Return wrapped provider identity."""
        return self._provider.provider_id

    @property
    def runtime_id(self) -> str:
        """Return wrapped runtime identity."""
        return self._provider.runtime_id

    def generate(
        self,
        request: ProviderInferenceRequest,
    ) -> ProviderInferenceResponse:
        """Generate and retain exact provider evidence."""
        response = self._provider.generate(
            request
        )

        self.responses.append(
            response
        )

        return response


class RecordingRAGRunner:
    """Record one end-to-end RAG trace per benchmark case."""

    def __init__(
        self,
        *,
        pipeline: RetrievalAugmentedGenerator,
        provider: RecordingProvider,
    ) -> None:
        """Store pipeline and provider recorder."""
        self._pipeline = pipeline
        self._provider = provider
        self.traces: list[
            dict[str, Any]
        ] = []

    def run(
        self,
        query: str,
    ) -> RAGResult:
        """Execute one RAG case and retain inference evidence."""
        response_count_before = len(
            self._provider.responses
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
                response_count_before:
            ]
        )

        if len(new_responses) > 1:
            raise RuntimeError(
                "One RAG query produced more than "
                "one provider inference response."
            )

        provider_response = (
            new_responses[0]
            if new_responses
            else None
        )

        context_text = "\n\n".join(
            document.text
            for document
            in result.context_documents
        )

        semantic_guard = None

        if not result.abstained:
            semantic_guard = (
                assess_semantic_drift(
                    context_text=context_text,
                    answer_text=result.answer,
                )
            )

        trace: dict[str, Any] = {
            "query": query,
            "answer": result.answer,
            "abstained": (
                result.abstained
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
            "provider_called": (
                provider_response
                is not None
            ),
            "semantic_guard": (
                {
                    "passed": (
                        semantic_guard.passed
                    ),
                    "issues": [
                        {
                            "risk": (
                                issue.risk.value
                            ),
                            "marker": (
                                issue.marker
                            ),
                            "detail": (
                                issue.detail
                            ),
                        }
                        for issue
                        in semantic_guard.issues
                    ],
                }
                if semantic_guard
                is not None
                else None
            ),
        }

        if provider_response is not None:
            trace[
                "provider_response"
            ] = (
                provider_response.model_dump(
                    mode="json"
                )
            )
        else:
            trace[
                "provider_response"
            ] = None

        self.traces.append(
            trace
        )

        return result


def main() -> None:
    """Run pinned SmolLM2 through provider-neutral verified RAG."""
    retrieval_dataset = (
        load_retrieval_benchmark(
            RETRIEVAL_DATASET_PATH
        )
    )

    rag_dataset = (
        load_rag_benchmark(
            RAG_DATASET_PATH
        )
    )

    known_document_ids = {
        document.document_id
        for document
        in retrieval_dataset.documents
    }

    validate_rag_benchmark_documents(
        rag_dataset,
        known_document_ids=(
            known_document_ids
        ),
    )

    print(
        "=== M4-F VERIFIED SLM/RAG BASELINE ==="
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

    retrieval_setup_started = (
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
        - retrieval_setup_started
    ) / 1_000_000

    provider_setup_started = (
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
        - provider_setup_started
    ) / 1_000_000

    recording_provider = (
        RecordingProvider(
            local_provider
        )
    )

    generator = (
        ProviderBackedRAGGenerator(
            provider=recording_provider,
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

    pipeline = (
        RetrievalAugmentedGenerator(
            retriever=retriever,
            generator=generator,
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
        provider=(
            recording_provider
        ),
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
        "=== RAG VERIFICATION ==="
    )

    report = evaluate_rag(
        rag_dataset,
        runner,
    )

    successful_provider_responses = [
        response
        for response
        in recording_provider.responses
        if response.succeeded
    ]

    failed_provider_responses = [
        response
        for response
        in recording_provider.responses
        if not response.succeeded
    ]

    total_input_tokens = sum(
        response.usage.input_tokens
        for response
        in successful_provider_responses
        if response.usage
        is not None
    )

    total_output_tokens = sum(
        response.usage.output_tokens
        for response
        in successful_provider_responses
        if response.usage
        is not None
    )

    generation_latencies_ms = [
        response
        .timing
        .generation_latency_ms
        for response
        in successful_provider_responses
        if (
            response
            .timing
            .generation_latency_ms
            is not None
        )
    ]

    total_generation_seconds = (
        sum(
            generation_latencies_ms
        )
        / 1000.0
    )

    generation_tokens_per_second = (
        (
            total_output_tokens
            / total_generation_seconds
        )
        if (
            total_generation_seconds
            > 0.0
        )
        else None
    )

    end_to_end_latencies_ms = [
        float(
            trace[
                "end_to_end_latency_ms"
            ]
        )
        for trace
        in runner.traces
    ]

    semantic_guard_results = [
        trace[
            "semantic_guard"
        ]
        for trace
        in runner.traces
        if trace[
            "semantic_guard"
        ]
        is not None
    ]

    semantic_guard_passes = sum(
        1
        for result
        in semantic_guard_results
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

    semantic_guard_pass_rate = (
        (
            semantic_guard_passes
            / len(
                semantic_guard_results
            )
        )
        if semantic_guard_results
        else 1.0
    )

    provider_skipped_cases = sum(
        1
        for trace
        in runner.traces
        if not trace[
            "provider_called"
        ]
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
        "=== INFERENCE EVIDENCE ==="
    )
    print(
        "provider_calls:",
        len(
            recording_provider.responses
        ),
    )
    print(
        "provider_skipped_cases:",
        provider_skipped_cases,
    )
    print(
        "provider_failures:",
        len(
            failed_provider_responses
        ),
    )
    print(
        "input_tokens:",
        total_input_tokens,
    )
    print(
        "output_tokens:",
        total_output_tokens,
    )

    if (
        generation_tokens_per_second
        is not None
    ):
        print(
            "generation_tokens_per_second:",
            f"{generation_tokens_per_second:.3f}",
        )

    if end_to_end_latencies_ms:
        print(
            "mean_end_to_end_latency_ms:",
            f"{sum(end_to_end_latencies_ms) / len(end_to_end_latencies_ms):.3f}",
        )

    print(
        "semantic_guard_pass_rate:",
        f"{semantic_guard_pass_rate:.3f}",
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
            sep="",
        )

        semantic = trace[
            "semantic_guard"
        ]

        if isinstance(
            semantic,
            dict,
        ):
            print(
                " semantic_guard_passed=",
                semantic[
                    "passed"
                ],
                sep="",
            )

        provider_response = trace[
            "provider_response"
        ]

        if isinstance(
            provider_response,
            dict,
        ):
            usage = (
                provider_response.get(
                    "usage"
                )
            )

            timing = (
                provider_response.get(
                    "timing"
                )
            )

            if isinstance(
                usage,
                dict,
            ):
                print(
                    " input_tokens=",
                    usage[
                        "input_tokens"
                    ],
                    " output_tokens=",
                    usage[
                        "output_tokens"
                    ],
                    sep="",
                )

            if isinstance(
                timing,
                dict,
            ):
                print(
                    " generation_latency_ms=",
                    timing.get(
                        "generation_latency_ms"
                    ),
                    sep="",
                )

        print(
            " answer=",
            trace[
                "answer"
            ],
            sep="",
        )

    environment = (
        capture_inference_environment()
    )

    payload: dict[
        str,
        Any,
    ] = {
        "schema_version": "0.1",
        "milestone": "M4-F",
        "experiment": (
            "verified-provider-neutral-"
            "smollm2-rag-baseline"
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
                recording_provider
                .provider_id
            ),
            "runtime_id": (
                recording_provider
                .runtime_id
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
            "local_files_only": True,
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
        "inference": {
            "provider_calls": len(
                recording_provider.responses
            ),
            "provider_skipped_cases": (
                provider_skipped_cases
            ),
            "provider_failures": len(
                failed_provider_responses
            ),
            "input_tokens": (
                total_input_tokens
            ),
            "output_tokens": (
                total_output_tokens
            ),
            "generation_tokens_per_second": (
                generation_tokens_per_second
            ),
            "mean_end_to_end_latency_ms": (
                (
                    sum(
                        end_to_end_latencies_ms
                    )
                    / len(
                        end_to_end_latencies_ms
                    )
                )
                if end_to_end_latencies_ms
                else None
            ),
            "semantic_guard_evaluated_outputs": (
                len(
                    semantic_guard_results
                )
            ),
            "semantic_guard_passes": (
                semantic_guard_passes
            ),
            "semantic_guard_pass_rate": (
                semantic_guard_pass_rate
            ),
        },
        "environment": (
            environment.model_dump(
                mode="json"
            )
        ),
        "traces": (
            runner.traces
        ),
        "limitations": [
            (
                "This is a local CPU development benchmark, "
                "not a production deployment claim."
            ),
            (
                "The deterministic semantic guard covers "
                "bounded known drift classes and is not a "
                "complete factuality judge."
            ),
            (
                "Citation correctness verifies document "
                "identity, while answer faithfulness requires "
                "separate evidence."
            ),
            (
                "Retrieval and generation latency are included "
                "in end-to-end RAG latency."
            ),
            (
                "KV-cache optimisation is not varied in M4-F "
                "and remains outside this experiment."
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
