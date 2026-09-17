# ADR-002: Keep QLoRA as an Optional Specialist Training Path

## Status

Accepted.

## Context

VAIT includes a bounded local fine-tuning experiment for the AP policy
specialist using the pinned `Qwen/Qwen2.5-0.5B-Instruct` model.

The reference PEFT experiment uses standard LoRA with:

- rank: 8
- alpha: 16
- dropout: 0.05
- target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- trainable parameters: 1,081,344

The project also evaluated whether QLoRA is technically justified for the
same bounded experiment.

The tested local environment was:

- Apple Silicon arm64
- PyTorch 2.14.0
- MPS available
- PEFT 0.21.0
- bitsandbytes 0.50.2
- Qwen2.5-0.5B-Instruct pinned to revision
  `7ae557604adf67be50417f59c2c2f167def9a775`

A real NF4 capability probe created 168 `Linear4bit` modules and completed a
finite MPS forward pass.

A subsequent QLoRA training smoke test completed:

- k-bit training preparation
- causal-LM forward pass
- backward pass
- non-zero LoRA gradients
- zero gradients on the frozen quantized base model

The QLoRA path therefore demonstrated real local training feasibility rather
than import-only or configuration-only compatibility.

## Measured footprint

Using the same LoRA adapter configuration:

| Candidate | Reported model footprint | Trainable parameters |
| --- | ---: | ---: |
| Standard LoRA | 946.42 MiB | 1,081,344 |
| QLoRA NF4 | 694.34 MiB | 1,081,344 |

The reported model-footprint reduction was:

- 252.08 MiB
- 26.64%

These numbers describe the tested local model footprint and must not be
interpreted as universal peak-training-memory savings.

## Decision

Standard LoRA remains the reference training path for the current
0.5B AP policy specialist.

QLoRA remains an optional constrained-memory training path.

The reason is not lack of QLoRA support. QLoRA was successfully verified on
the tested Apple Silicon/MPS stack.

Instead, the current 0.5B model already fits comfortably under standard LoRA,
while QLoRA introduces additional quantization and backend complexity.

For this bounded experiment, the additional complexity is not required to
make training feasible.

QLoRA becomes more relevant when a future candidate is large enough for
memory pressure to materially constrain standard LoRA training.

## Consequences

VAIT may claim that:

- QLoRA feasibility was verified on the tested local Apple Silicon/MPS stack.
- NF4 quantization reduced the reported model footprint by 26.64% in the
  measured comparison.
- the same number of LoRA adapter parameters remained trainable.
- standard LoRA was intentionally retained as the reference experiment.

VAIT must not claim that:

- QLoRA is universally faster or more memory-efficient across hardware;
- the measured footprint represents total or peak training memory;
- quantized training improves model quality;
- QLoRA is required for the current 0.5B experiment.

A future larger-model experiment may revisit this decision.
