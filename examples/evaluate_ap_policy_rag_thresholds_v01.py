"""Sweep abstention thresholds for grounded AP policy RAG v0.1."""

import json
from pathlib import Path

from vait.rag.benchmark import (
    evaluate_rag,
    load_rag_benchmark,
    validate_rag_benchmark_documents,
)
from vait.rag.extractive import ExtractiveGroundedGenerator
from vait.rag.pipeline import RetrievalAugmentedGenerator
from vait.retrieval.benchmark import load_retrieval_benchmark
from vait.retrieval.in_memory import InMemoryCosineRetriever
from vait.retrieval.sentence_transformer import (
    SentenceTransformerEmbeddingEncoder,
)

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

THRESHOLDS = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
)

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
    model_id=MODEL_ID,
    revision=MODEL_REVISION,
    device="cpu",
    local_files_only=True,
)

retriever = InMemoryCosineRetriever(
    encoder=encoder,
    documents=retrieval_dataset.text_documents(),
)

print(
    f"RAG benchmark: {rag_dataset.benchmark_id}@"
    f"{rag_dataset.version}"
)
print(f"Encoder: {encoder.implementation_id}")
print()
print("Observed top-1 retrieval scores:")

for case in rag_dataset.cases:
    hit = retriever.retrieve(
        case.query,
        top_k=1,
    )[0]

    label = (
        "ABSTAIN"
        if case.expected_abstain
        else "ANSWER"
    )

    print(
        f"{case.case_id:20} "
        f"{label:7} "
        f"score={hit.score:.4f} "
        f"doc={hit.document.document_id}"
    )

reports = []

print()
print(
    "threshold  evidence  citation  abstention  "
    "answerable  overall"
)

for threshold in THRESHOLDS:
    generator = ExtractiveGroundedGenerator(
        minimum_score=threshold,
    )

    pipeline = RetrievalAugmentedGenerator(
        retriever=retriever,
        generator=generator,
        retrieval_k=3,
        max_context_documents=3,
        max_context_chars=2000,
    )

    report = evaluate_rag(
        rag_dataset,
        pipeline,
    )

    reports.append(
        {
            "threshold": threshold,
            "report": report.model_dump(
                mode="json"
            ),
        }
    )

    print(
        f"{threshold:9.2f}  "
        f"{report.evidence_coverage:8.3f}  "
        f"{report.citation_accuracy:8.3f}  "
        f"{report.abstention_accuracy:10.3f}  "
        f"{report.answerable_success_rate:10.3f}  "
        f"{report.overall_success_rate:7.3f}"
    )

output_path = Path(
    "artifacts/benchmarks/"
    "ap-policy-rag-v0.1-threshold-sweep.json"
)
output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

output_path.write_text(
    json.dumps(
        {
            "benchmark_id": rag_dataset.benchmark_id,
            "benchmark_version": rag_dataset.version,
            "encoder_implementation_id": (
                encoder.implementation_id
            ),
            "thresholds": reports,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print()
print(f"Evidence report: {output_path}")
