"""Smoke-test one real QLoRA backward pass on pinned local Qwen."""

import json
from pathlib import Path

import bitsandbytes as bnb
import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from vait.rag.specialist_sft import (
    format_specialist_sft_sample,
)
from vait.rag.specialist_training import (
    build_ap_policy_specialist_corpus,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

MODEL_REVISION = (
    "7ae557604adf67be50417f59c2c2f167def9a775"
)

DEVICE = "mps"

TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
)

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

MAX_LENGTH = 512

OUTPUT_PATH = Path(
    "artifacts/benchmarks/"
    "ap-policy-specialist-qlora-feasibility-v0.1.json"
)


if not torch.backends.mps.is_available():
    raise RuntimeError(
        "MPS is required for this QLoRA smoke test."
    )


quantization_config = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    local_files_only=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    local_files_only=True,
    quantization_config=quantization_config,
    device_map={"": DEVICE},
)

linear4bit_modules = [
    name
    for name, module in base_model.named_modules()
    if isinstance(
        module,
        bnb.nn.Linear4bit,  # type: ignore[attr-defined]
    )
]

if not linear4bit_modules:
    raise RuntimeError(
        "QLoRA base model contains no Linear4bit modules."
    )

prepared_model = prepare_model_for_kbit_training(  # type: ignore[no-untyped-call]
    base_model,
    use_gradient_checkpointing=False,
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=list(TARGET_MODULES),
    bias="none",
)

model = get_peft_model(
    prepared_model,
    lora_config,
)

model.train()

corpus = build_ap_policy_specialist_corpus()

example = corpus.train_examples[0]

sample = format_specialist_sft_sample(
    example,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH,
)

input_ids = torch.tensor(
    [sample.input_ids],
    dtype=torch.long,
    device=DEVICE,
)

attention_mask = torch.tensor(
    [sample.attention_mask],
    dtype=torch.long,
    device=DEVICE,
)

labels = torch.tensor(
    [sample.labels],
    dtype=torch.long,
    device=DEVICE,
)

trainable_parameters = {
    name: parameter
    for name, parameter in model.named_parameters()
    if parameter.requires_grad
}

frozen_parameters = {
    name: parameter
    for name, parameter in model.named_parameters()
    if not parameter.requires_grad
}

trainable_count = sum(
    parameter.numel()
    for parameter in trainable_parameters.values()
)

if trainable_count <= 0:
    raise RuntimeError(
        "QLoRA model exposes no trainable parameters."
    )

outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
)

loss = outputs.loss

if loss is None:
    raise RuntimeError(
        "QLoRA forward pass returned no loss."
    )

if not torch.isfinite(loss):
    raise RuntimeError(
        f"QLoRA loss is not finite: {loss.item()}"
    )

loss.backward()

trainable_with_gradient = [
    name
    for name, parameter
    in trainable_parameters.items()
    if parameter.grad is not None
]

trainable_with_nonzero_gradient = [
    name
    for name, parameter
    in trainable_parameters.items()
    if (
        parameter.grad is not None
        and torch.count_nonzero(
            parameter.grad
        ).item()
        > 0
    )
]

frozen_with_gradient = [
    name
    for name, parameter
    in frozen_parameters.items()
    if parameter.grad is not None
]

if not trainable_with_gradient:
    raise RuntimeError(
        "No QLoRA parameter received a gradient."
    )

if not trainable_with_nonzero_gradient:
    raise RuntimeError(
        "No QLoRA parameter received a non-zero gradient."
    )

if frozen_with_gradient:
    raise RuntimeError(
        "Frozen quantized base-model parameters "
        "unexpectedly received gradients."
    )

quantized_memory_bytes = int(
    model.get_memory_footprint()
)

quantized_memory_mib = (
    quantized_memory_bytes / 1024**2
)

report = {
    "artifact_id": (
        "ap-policy-specialist-qlora-feasibility-v0.1"
    ),
    "development_evidence": True,
    "model": {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "device": DEVICE,
    },
    "quantization": {
        "method": "nf4",
        "load_in_4bit": True,
        "double_quantization": True,
        "compute_dtype": "float16",
        "linear4bit_module_count": len(
            linear4bit_modules
        ),
        "memory_footprint_bytes": (
            quantized_memory_bytes
        ),
        "memory_footprint_mib": (
            quantized_memory_mib
        ),
    },
    "lora": {
        "rank": LORA_R,
        "alpha": LORA_ALPHA,
        "dropout": LORA_DROPOUT,
        "target_modules": list(
            TARGET_MODULES
        ),
        "trainable_parameters": (
            trainable_count
        ),
    },
    "sample": {
        "example_id": example.example_id,
        "sequence_tokens": (
            sample.token_count
        ),
        "supervised_tokens": (
            sample.supervised_token_count
        ),
        "max_length": MAX_LENGTH,
    },
    "gradient_evidence": {
        "loss": float(
            loss.item()
        ),
        "trainable_tensors_with_gradients": (
            len(
                trainable_with_gradient
            )
        ),
        "trainable_tensors_with_nonzero_gradients": (
            len(
                trainable_with_nonzero_gradient
            )
        ),
        "frozen_tensors_with_gradients": (
            len(
                frozen_with_gradient
            )
        ),
    },
    "checks": {
        "nf4_load": True,
        "kbit_preparation": True,
        "forward_pass": True,
        "backward_pass": True,
        "qlora_gradients": True,
        "frozen_quantized_base": True,
    },
    "software": {
        "torch": torch.__version__,
        "bitsandbytes": bnb.__version__,
    },
    "scope": (
        "Development feasibility evidence from one local "
        "QLoRA forward/backward pass. Memory footprint is "
        "the model-reported footprint, not peak end-to-end "
        "training memory. This does not establish model "
        "quality, unseen generalisation, or production safety."
    ),
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

print("AP specialist QLoRA gradient smoke")
print()
print(f"Model: {MODEL_ID}@{MODEL_REVISION}")
print(f"Device: {DEVICE}")
print(f"bitsandbytes: {bnb.__version__}")
print(f"4-bit modules: {len(linear4bit_modules)}")
print(
    "Quantized memory footprint:",
    f"{quantized_memory_mib:.2f} MiB",
)
print(f"Example: {example.example_id}")
print(f"Sequence tokens: {sample.token_count}")
print(
    "Supervised tokens:",
    sample.supervised_token_count,
)
print(
    "Target modules:",
    ", ".join(TARGET_MODULES),
)
print(f"LoRA rank: {LORA_R}")
print(f"LoRA alpha: {LORA_ALPHA}")
print(f"LoRA dropout: {LORA_DROPOUT}")
print(
    "Trainable parameters:",
    f"{trainable_count:,}",
)
print(f"Loss: {loss.item():.6f}")
print(
    "Trainable tensors with gradients:",
    len(trainable_with_gradient),
)
print(
    "Trainable tensors with non-zero gradients:",
    len(trainable_with_nonzero_gradient),
)
print(
    "Frozen tensors with gradients:",
    len(frozen_with_gradient),
)
print()
print("NF4 load: PASS")
print("K-bit preparation: PASS")
print("Forward pass: PASS")
print("Backward pass: PASS")
print("QLoRA gradients: PASS")
print("Frozen quantized base: PASS")
print()
print(
    "Evidence report:",
    OUTPUT_PATH,
)
