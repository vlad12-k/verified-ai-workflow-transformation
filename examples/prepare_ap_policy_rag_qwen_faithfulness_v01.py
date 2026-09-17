"""Prepare exact Qwen outputs for independent faithfulness review."""

import json
from pathlib import Path

from vait.rag.faithfulness import answer_sha256

input_path = Path(
    "artifacts/benchmarks/"
    "ap-policy-rag-qwen-v0.1.json"
)

payload = json.loads(
    input_path.read_text(
        encoding="utf-8"
    )
)

print("Faithfulness review candidates:")

for trace in payload["traces"]:
    if trace["abstained"]:
        continue

    answer = trace["answer"]

    print()
    print(f"Query: {trace['query']}")
    print(
        "SHA-256: "
        f"{answer_sha256(answer)}"
    )
    print("Context:")

    for context in trace["context"]:
        print(
            f"  [{context['document_id']}] "
            f"{context['text']}"
        )

    print("Generated answer:")
    print(answer)
