"""Smoke-test one real LoRA training step on the pinned Qwen specialist."""

import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
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


if not torch.backends.mps.is_available():
    raise RuntimeError(
        "MPS is required for this local LoRA smoke test."
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
)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=list(TARGET_MODULES),
    bias="none",
)

model = get_peft_model(
    base_model,
    config,
)

model.to(DEVICE)
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

total_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

if trainable_count != 1_081_344:
    raise RuntimeError(
        "Unexpected LoRA trainable parameter count: "
        f"{trainable_count:,}"
    )

outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
)

loss = outputs.loss

if loss is None:
    raise RuntimeError(
        "Causal LM forward pass did not return a loss."
    )

if not torch.isfinite(loss):
    raise RuntimeError(
        f"Training loss is not finite: {loss.item()}"
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
    if parameter.grad is not None
    and torch.count_nonzero(
        parameter.grad
    ).item()
    > 0
]

frozen_with_gradient = [
    name
    for name, parameter
    in frozen_parameters.items()
    if parameter.grad is not None
]

if not trainable_with_gradient:
    raise RuntimeError(
        "No LoRA parameter received a gradient."
    )

if not trainable_with_nonzero_gradient:
    raise RuntimeError(
        "No LoRA parameter received a non-zero gradient."
    )

if frozen_with_gradient:
    raise RuntimeError(
        "Frozen base-model parameters unexpectedly "
        "received gradients."
    )

print("AP specialist LoRA gradient smoke")
print()
print(f"Model: {MODEL_ID}@{MODEL_REVISION}")
print(f"Device: {DEVICE}")
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
    "Total parameters:",
    f"{total_count:,}",
)
print(
    "Trainable parameters:",
    f"{trainable_count:,}",
)
print(
    "Trainable percentage:",
    f"{100 * trainable_count / total_count:.4f}%",
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
print("Forward pass: PASS")
print("Backward pass: PASS")
print("LoRA gradients: PASS")
print("Frozen base model: PASS")
