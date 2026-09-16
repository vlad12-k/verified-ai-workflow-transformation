"""Evaluate local semantic retrieval on AP policy benchmark v0.1."""

from pathlib import Path

from vait.retrieval.benchmark import (
    evaluate_retriever,
    load_retrieval_benchmark,
)
from vait.retrieval.in_memory import InMemoryCosineRetriever
from vait.retrieval.sentence_transformer import (
    SentenceTransformerEmbeddingEncoder,
)

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = (
    "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)

dataset = load_retrieval_benchmark(
    "datasets/ap_policy_context/v0.1/benchmark.json"
)

encoder = SentenceTransformerEmbeddingEncoder(
    model_id=MODEL_ID,
    revision=MODEL_REVISION,
    device="cpu",
    local_files_only=True,
)

retriever = InMemoryCosineRetriever(
    encoder=encoder,
    documents=dataset.text_documents(),
)

report = evaluate_retriever(
    dataset,
    retriever,
)

output_path = Path(
    "artifacts/benchmarks/"
    "ap-policy-context-v0.1-retrieval.json"
)
output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)
output_path.write_text(
    report.model_dump_json(indent=2) + "\n",
    encoding="utf-8",
)

print(
    f"Benchmark: {report.benchmark_id}@"
    f"{report.benchmark_version}"
)
print(
    f"Encoder: {report.encoder_implementation_id}"
)
print(f"Queries: {report.query_count}")
print(f"Recall@1: {report.recall_at_1:.3f}")
print(f"Recall@3: {report.recall_at_3:.3f}")
print(
    "MRR: "
    f"{report.mean_reciprocal_rank:.3f}"
)

failures = [
    result
    for result in report.results
    if result.first_relevant_rank != 1
]

print(f"Non-rank-1 queries: {len(failures)}")

for result in failures:
    print(
        f"  {result.query_id}: "
        f"relevant={result.relevant_document_ids}, "
        f"retrieved={result.retrieved_document_ids}"
    )

print(f"Evidence report: {output_path}")
