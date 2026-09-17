"""Validate AP specialist SFT formatting with the pinned Qwen tokenizer."""

import json
from pathlib import Path

from transformers import AutoTokenizer

from vait.rag.specialist_sft import (
    IGNORE_INDEX,
    format_specialist_sft_sample,
)
from vait.rag.specialist_training import (
    build_ap_policy_specialist_corpus,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = (
    "7ae557604adf67be50417f59c2c2f167def9a775"
)
MAX_LENGTH = 512

OUTPUT_PATH = Path(
    "artifacts/benchmarks/"
    "ap-policy-specialist-tokenization-v0.1.json"
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    local_files_only=True,
)

corpus = build_ap_policy_specialist_corpus()

samples = [
    format_specialist_sft_sample(
        example,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )
    for example in corpus.examples
]

if len(samples) != len(corpus.examples):
    raise RuntimeError(
        "Tokenized sample count does not match corpus size."
    )

if any(
    sample.supervised_token_count <= 0
    for sample in samples
):
    raise RuntimeError(
        "Every specialist sample must supervise assistant tokens."
    )

if any(
    label != IGNORE_INDEX
    for sample in samples
    for label in sample.labels[
        : sample.prompt_token_count
    ]
):
    raise RuntimeError(
        "Prompt tokens must remain masked from the loss."
    )

if any(
    sample.token_count > MAX_LENGTH
    for sample in samples
):
    raise RuntimeError(
        "At least one tokenized sample exceeds max length."
    )

rows = []

for example, sample in zip(
    corpus.examples,
    samples,
    strict=True,
):
    rows.append(
        {
            "example_id": example.example_id,
            "split": example.split,
            "token_count": sample.token_count,
            "prompt_token_count": (
                sample.prompt_token_count
            ),
            "supervised_token_count": (
                sample.supervised_token_count
            ),
        }
    )

token_counts = [
    sample.token_count
    for sample in samples
]

supervised_counts = [
    sample.supervised_token_count
    for sample in samples
]

report = {
    "artifact_id": (
        "ap-policy-specialist-tokenization-v0.1"
    ),
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "local_files_only": True,
    "max_length": MAX_LENGTH,
    "corpus_id": corpus.corpus_id,
    "corpus_version": corpus.version,
    "corpus_sha256": corpus.sha256,
    "example_count": len(samples),
    "train_count": len(corpus.train_examples),
    "dev_count": len(corpus.dev_examples),
    "min_token_count": min(token_counts),
    "max_token_count": max(token_counts),
    "min_supervised_token_count": min(
        supervised_counts
    ),
    "max_supervised_token_count": max(
        supervised_counts
    ),
    "prompt_masking_valid": True,
    "assistant_supervision_valid": True,
    "max_length_valid": True,
    "samples": rows,
}

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

print("AP specialist tokenizer validation")
print()
print(f"Model: {MODEL_ID}@{MODEL_REVISION}")
print(f"Corpus: {corpus.corpus_id}@{corpus.version}")
print(f"Corpus SHA256: {corpus.sha256}")
print(f"Examples: {len(samples)}")
print(f"Train: {len(corpus.train_examples)}")
print(f"Dev: {len(corpus.dev_examples)}")
print(f"Min tokens: {min(token_counts)}")
print(f"Max tokens: {max(token_counts)}")
print(
    "Min supervised tokens:",
    min(supervised_counts),
)
print(
    "Max supervised tokens:",
    max(supervised_counts),
)
print("Prompt masking: PASS")
print("Assistant supervision: PASS")
print("Max length: PASS")
print()
print("Per-example token counts:")

for row in rows:
    print(
        f"{row['example_id']}: "
        f"tokens={row['token_count']} "
        f"prompt={row['prompt_token_count']} "
        f"supervised={row['supervised_token_count']}"
    )

print()
print(f"Evidence report: {OUTPUT_PATH}")
