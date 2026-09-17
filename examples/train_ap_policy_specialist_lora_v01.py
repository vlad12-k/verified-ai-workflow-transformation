"""Train the bounded AP policy LoRA specialist on pinned local Qwen."""

import json
import random
import shutil
from pathlib import Path

import peft
import torch
import transformers
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
    SpecialistTrainingExample,
    build_ap_policy_specialist_corpus,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = (
    "7ae557604adf67be50417f59c2c2f167def9a775"
)

DEVICE = "mps"
SEED = 20260917

TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
)

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

LEARNING_RATE = 2e-4
EPOCHS = 6
MAX_LENGTH = 512

ADAPTER_PATH = Path(
    "artifacts/models/"
    "ap-policy-specialist-lora-v0.1"
)

REPORT_PATH = Path(
    "artifacts/benchmarks/"
    "ap-policy-specialist-lora-training-v0.1.json"
)


if not torch.backends.mps.is_available():
    raise RuntimeError(
        "MPS is required for this local training run."
    )


random.seed(SEED)
torch.manual_seed(SEED)


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

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=list(TARGET_MODULES),
    bias="none",
)

model = get_peft_model(
    base_model,
    lora_config,
)

model.to(DEVICE)

corpus = build_ap_policy_specialist_corpus()


def tensors_for_example(
    example: SpecialistTrainingExample,
) -> dict[str, torch.Tensor]:
    """Create one device-local causal-LM training batch."""
    sample = format_specialist_sft_sample(
        example,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    return {
        "input_ids": torch.tensor(
            [sample.input_ids],
            dtype=torch.long,
            device=DEVICE,
        ),
        "attention_mask": torch.tensor(
            [sample.attention_mask],
            dtype=torch.long,
            device=DEVICE,
        ),
        "labels": torch.tensor(
            [sample.labels],
            dtype=torch.long,
            device=DEVICE,
        ),
    }


train_examples = list(
    corpus.train_examples
)

dev_examples = list(
    corpus.dev_examples
)

train_batches = {
    example.example_id: tensors_for_example(
        example
    )
    for example in train_examples
}

dev_batches = {
    example.example_id: tensors_for_example(
        example
    )
    for example in dev_examples
}

optimizer = torch.optim.AdamW(
    (
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ),
    lr=LEARNING_RATE,
)

trainable_count = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

total_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)


def evaluate_dev_loss() -> float:
    """Return mean dev loss without updating parameters."""
    model.eval()

    losses = []

    with torch.inference_mode():
        for example in dev_examples:
            outputs = model(
                **dev_batches[
                    example.example_id
                ]
            )

            if outputs.loss is None:
                raise RuntimeError(
                    "Dev forward pass returned no loss."
                )

            losses.append(
                float(
                    outputs.loss.detach().cpu().item()
                )
            )

    model.train()

    return sum(losses) / len(losses)


initial_dev_loss = evaluate_dev_loss()

print("AP policy specialist LoRA training")
print()
print(f"Model: {MODEL_ID}@{MODEL_REVISION}")
print(f"Device: {DEVICE}")
print(f"Seed: {SEED}")
print(f"Corpus: {corpus.corpus_id}@{corpus.version}")
print(f"Corpus SHA256: {corpus.sha256}")
print(f"Train examples: {len(train_examples)}")
print(f"Dev examples: {len(dev_examples)}")
print(
    "Trainable parameters:",
    f"{trainable_count:,}",
)
print(
    "Trainable percentage:",
    f"{100 * trainable_count / total_count:.4f}%",
)
print(f"Learning rate: {LEARNING_RATE}")
print(f"Epochs: {EPOCHS}")
print(f"Initial dev loss: {initial_dev_loss:.6f}")
print()

if ADAPTER_PATH.exists():
    shutil.rmtree(
        ADAPTER_PATH
    )

ADAPTER_PATH.mkdir(
    parents=True,
    exist_ok=True,
)

history = []

best_dev_loss = float("inf")
best_epoch = 0

for epoch in range(
    1,
    EPOCHS + 1,
):
    model.train()

    epoch_examples = list(
        train_examples
    )

    random.Random(
        SEED + epoch
    ).shuffle(
        epoch_examples
    )

    training_losses = []

    for example in epoch_examples:
        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(
            **train_batches[
                example.example_id
            ]
        )

        loss = outputs.loss

        if loss is None:
            raise RuntimeError(
                "Training forward pass returned no loss."
            )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Training produced non-finite loss."
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            (
                parameter
                for parameter
                in model.parameters()
                if parameter.requires_grad
            ),
            max_norm=1.0,
        )

        optimizer.step()

        training_losses.append(
            float(
                loss.detach().cpu().item()
            )
        )

    mean_train_loss = (
        sum(training_losses)
        / len(training_losses)
    )

    dev_loss = evaluate_dev_loss()

    history.append(
        {
            "epoch": epoch,
            "train_loss": mean_train_loss,
            "dev_loss": dev_loss,
        }
    )

    print(
        f"Epoch {epoch}: "
        f"train_loss={mean_train_loss:.6f} "
        f"dev_loss={dev_loss:.6f}"
    )

    if dev_loss < best_dev_loss:
        best_dev_loss = dev_loss
        best_epoch = epoch

        model.save_pretrained(
            ADAPTER_PATH
        )

if best_epoch == 0:
    raise RuntimeError(
        "No valid LoRA checkpoint was selected."
    )

report = {
    "experiment_id": (
        "ap-policy-specialist-lora-training-v0.1"
    ),
    "development_evidence": True,
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "device": DEVICE,
    "seed": SEED,
    "software": {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
    },
    "corpus": {
        "corpus_id": corpus.corpus_id,
        "version": corpus.version,
        "sha256": corpus.sha256,
        "train_examples": len(
            train_examples
        ),
        "dev_examples": len(
            dev_examples
        ),
    },
    "lora": {
        "target_modules": list(
            TARGET_MODULES
        ),
        "rank": LORA_R,
        "alpha": LORA_ALPHA,
        "dropout": LORA_DROPOUT,
        "trainable_parameters": (
            trainable_count
        ),
        "total_parameters": total_count,
        "trainable_percentage": (
            100
            * trainable_count
            / total_count
        ),
    },
    "training": {
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "max_length": MAX_LENGTH,
        "gradient_clip_norm": 1.0,
        "initial_dev_loss": (
            initial_dev_loss
        ),
        "best_epoch": best_epoch,
        "best_dev_loss": best_dev_loss,
        "history": history,
    },
    "adapter_path": str(
        ADAPTER_PATH
    ),
}

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_PATH.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

print()
print(f"Best epoch: {best_epoch}")
print(
    f"Best dev loss: "
    f"{best_dev_loss:.6f}"
)
print(
    "Dev-loss change:",
    f"{best_dev_loss - initial_dev_loss:+.6f}",
)
print(
    f"Saved adapter: {ADAPTER_PATH}"
)
print(
    f"Evidence report: {REPORT_PATH}"
)
