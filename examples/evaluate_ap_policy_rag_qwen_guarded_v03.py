"""Evaluate semantic-guarded strict-v2 Qwen RAG on AP policy benchmark."""

import json
from pathlib import Path
from time import perf_counter

from vait.rag.benchmark import (
    evaluate_rag,
    load_rag_benchmark,
    validate_rag_benchmark_documents,
)
from vait.rag.extractive import ExtractiveGroundedGenerator
from vait.rag.guarded import SemanticGuardedGenerator
from vait.rag.huggingface import HuggingFaceCausalGenerator
from vait.rag.models import (
    RAGGeneration,
    RAGGenerationRequest,
    RAGResult,
)
from vait.rag.pipeline import RetrievalAugmentedGenerator
from vait.rag.prompt_contracts import RAGPromptContract
from vait.retrieval.benchmark import load_retrieval_benchmark
from vait.retrieval.in_memory import InMemoryCosineRetriever
from vait.retrieval.sentence_transformer import (
    SentenceTransformerEmbeddingEncoder,
)

EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = (
    "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)

GENERATOR_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
GENERATOR_REVISION = (
    "7ae557604adf67be50417f59c2c2f167def9a775"
)

MINIMUM_SCORE = 0.4
MAX_NEW_TOKENS = 80


class RecordingFallbackGenerator:
    """Record deterministic fallback invocations."""

    def __init__(
        self,
        delegate: ExtractiveGroundedGenerator,
    ) -> None:
        """Store fallback implementation and invocation count."""
        self._delegate = delegate
        self.call_count = 0

    @property
    def implementation_id(self) -> str:
        """Expose the underlying fallback identity."""
        return self._delegate.implementation_id

    def generate(
        self,
        request: RAGGenerationRequest,
    ) -> RAGGeneration:
        """Record and delegate one fallback generation."""
        self.call_count += 1
        return self._delegate.generate(request)


class RecordingRAGRunner:
    """Record guarded answers, routing decisions, and latency."""

    def __init__(
        self,
        *,
        pipeline: RetrievalAugmentedGenerator,
        fallback: RecordingFallbackGenerator,
    ) -> None:
        """Store the evaluated guarded pipeline."""
        self._pipeline = pipeline
        self._fallback = fallback
        self.traces: list[dict[str, object]] = []

    def run(
        self,
        query: str,
    ) -> RAGResult:
        """Run one query and record surfaced output."""
        fallback_calls_before = self._fallback.call_count

        started = perf_counter()

        result = self._pipeline.run(query)

        latency_ms = (
            perf_counter() - started
        ) * 1000.0

        fallback_used = (
            self._fallback.call_count
            > fallback_calls_before
        )

        self.traces.append(
            {
                "query": query,
                "answer": result.answer,
                "abstained": result.abstained,
                "fallback_used": fallback_used,
                "cited_document_ids": list(
                    result.cited_document_ids
                ),
                "context": [
                    {
                        "document_id": document.document_id,
                        "score": document.score,
                        "rank": document.rank,
                        "text": document.text,
                    }
                    for document in result.context_documents
                ],
                "latency_ms": latency_ms,
            }
        )

        return result


retrieval_dataset = load_retrieval_benchmark(
    "datasets/ap_policy_context/v0.1/benchmark.json"
)

rag_dataset = load_rag_benchmark(
    "datasets/ap_policy_rag/v0.1/benchmark.json"
)

known_document_ids = {
    document.document_id
    for document in retrieval_dataset.documents
}

validate_rag_benchmark_documents(
    rag_dataset,
    known_document_ids=known_document_ids,
)

encoder = SentenceTransformerEmbeddingEncoder(
    model_id=EMBEDDING_MODEL_ID,
    revision=EMBEDDING_REVISION,
    device="cpu",
    local_files_only=True,
)

retriever = InMemoryCosineRetriever(
    encoder=encoder,
    documents=retrieval_dataset.text_documents(),
)

primary_generator = HuggingFaceCausalGenerator(
    model_id=GENERATOR_MODEL_ID,
    revision=GENERATOR_REVISION,
    device="cpu",
    minimum_score=MINIMUM_SCORE,
    max_new_tokens=MAX_NEW_TOKENS,
    local_files_only=True,
    prompt_contract=RAGPromptContract.STRICT_V2,
)

fallback_generator = RecordingFallbackGenerator(
    ExtractiveGroundedGenerator(
        minimum_score=MINIMUM_SCORE,
    )
)

guarded_generator = SemanticGuardedGenerator(
    primary=primary_generator,
    fallback=fallback_generator,
)

pipeline = RetrievalAugmentedGenerator(
    retriever=retriever,
    generator=guarded_generator,
    retrieval_k=1,
    max_context_documents=1,
    max_context_chars=2000,
)

runner = RecordingRAGRunner(
    pipeline=pipeline,
    fallback=fallback_generator,
)

report = evaluate_rag(
    rag_dataset,
    runner,
)

print(
    f"Benchmark: {report.benchmark_id}@"
    f"{report.benchmark_version}"
)
print(f"Encoder: {encoder.implementation_id}")
print(
    "Primary generator: "
    f"{primary_generator.implementation_id}"
)
print(
    "Guarded generator: "
    f"{guarded_generator.implementation_id}"
)
print(f"Cases: {report.case_count}")
print(
    f"Evidence coverage: "
    f"{report.evidence_coverage:.3f}"
)
print(
    f"Citation accuracy: "
    f"{report.citation_accuracy:.3f}"
)
print(
    f"Abstention accuracy: "
    f"{report.abstention_accuracy:.3f}"
)
print(
    f"Answerable success: "
    f"{report.answerable_success_rate:.3f}"
)
print(
    f"Overall success: "
    f"{report.overall_success_rate:.3f}"
)

print()
print("Per-case surfaced outputs:")

for case, evaluation, trace in zip(
    rag_dataset.cases,
    report.results,
    runner.traces,
    strict=True,
):
    context = trace["context"]

    top_score = (
        context[0]["score"]
        if isinstance(context, list)
        and context
        and isinstance(context[0], dict)
        else None
    )

    print()
    print(f"[{case.case_id}]")
    print(
        f"expected_abstain={case.expected_abstain} "
        f"observed_abstain={evaluation.observed_abstain} "
        f"success={evaluation.success}"
    )
    print(
        f"fallback_used={trace['fallback_used']}"
    )
    print(f"top_score={top_score}")
    print(
        f"latency_ms="
        f"{float(trace['latency_ms']):.2f}"
    )
    print(f"answer={trace['answer']}")

answerable_traces = [
    trace
    for trace in runner.traces
    if not trace["abstained"]
]

fallback_count = sum(
    bool(trace["fallback_used"])
    for trace in answerable_traces
)

primary_count = (
    len(answerable_traces)
    - fallback_count
)

print()
print("Guard routing summary")
print(
    f"Answerable surfaced outputs: "
    f"{len(answerable_traces)}"
)
print(f"Primary surfaced: {primary_count}")
print(f"Fallback surfaced: {fallback_count}")

output_path = Path(
    "artifacts/benchmarks/"
    "ap-policy-rag-qwen-strict-guarded-v0.3.json"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

output_path.write_text(
    json.dumps(
        {
            "experiment_id": (
                "ap-policy-rag-qwen-strict-guarded-v0.3"
            ),
            "embedding_model_id": EMBEDDING_MODEL_ID,
            "embedding_revision": EMBEDDING_REVISION,
            "generator_model_id": GENERATOR_MODEL_ID,
            "generator_revision": GENERATOR_REVISION,
            "configuration": {
                "device": "cpu",
                "minimum_score": MINIMUM_SCORE,
                "retrieval_k": 1,
                "max_context_documents": 1,
                "max_context_chars": 2000,
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": False,
                "local_files_only": True,
                "prompt_contract": (
                    RAGPromptContract.STRICT_V2.value
                ),
                "semantic_guard": True,
                "fallback": (
                    fallback_generator.implementation_id
                ),
            },
            "routing": {
                "answerable_surfaced_outputs": (
                    len(answerable_traces)
                ),
                "primary_surfaced": primary_count,
                "fallback_surfaced": fallback_count,
            },
            "report": report.model_dump(
                mode="json"
            ),
            "traces": runner.traces,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print()
print(f"Evidence report: {output_path}")
