# Verified AI Workflow Transformation (VAIT)

VAIT is a provider-neutral verification and evaluation layer for assessing
supplied AI workflow transformation candidates under explicit behavioural,
risk, statistical, latency, and economic evidence.

The project asks a narrower question than a general AI optimiser:

> Given an already-specified workflow and a supplied replacement candidate,
> what evidence can be established before accepting the transformation?

VAIT does not synthesize arbitrary replacements, act as a runtime
orchestrator, or automatically rank heterogeneous AI systems.

## Verification outcomes

VAIT distinguishes between:

- **EXACT** — exact obligations have been established for a supported,
  explicitly bounded deterministic scope;
- **BOUNDED** — the candidate satisfies a declared statistical verification
  policy on the evaluated benchmark and risk strata;
- **REJECT** — verification constraints were not satisfied;
- **NOT_APPLICABLE** — the candidate does not satisfy the declared workflow
  applicability requirements and is therefore not executed.

`BOUNDED` is not a claim of universal semantic equivalence.

## Implemented AI breadth

The current M3 development evidence includes:

### Structured decision systems

- deterministic Python transformations;
- scikit-learn Random Forest;
- XGBoost;
- PyTorch neural inference;
- hard-label and distilled compact students;
- TensorFlow/Keras comparative inference.

### Retrieval and generative AI

- sentence-transformer embeddings;
- retrieval evaluation;
- local Hugging Face / Transformers inference;
- Qwen2.5-0.5B-Instruct;
- RAG prompt-contract experiments;
- semantic guards, abstention, and fallback routing.

### Model adaptation

- specialist supervised corpus construction;
- PEFT / LoRA;
- QLoRA with NF4 4-bit loading;
- real adapter-gradient backward-pass evidence.

### Data engineering for evaluation

- deterministic synthetic training data;
- class-balance and integrity validation;
- exact evaluation-input exclusion;
- provenance-oriented evidence artifacts.

## A useful verification result

A controlled PyTorch-versus-Keras experiment demonstrates why VAIT separates
training quality from verification.

Both implementations used the same 11-feature AP representation, the same
32/16 hidden-layer architecture, 963 parameters, and approximately 99.67%
development training accuracy.

On AP benchmark v0.2:

    PyTorch: BOUNDED — 0/20 observed disagreements
    Keras:   REJECT  — 1/20 observed disagreements

The Keras disagreement occurred on a reproducible high-risk amount-mismatch
boundary case. The result is development evidence and is not a general
framework comparison.

## Unified heterogeneous evidence

After generating the individual development artifacts, run:

    python examples/build_m3_heterogeneous_evidence_v01.py

This creates:

    artifacts/benchmarks/m3-heterogeneous-evidence-v0.1.json

The report groups evidence by compatible task family and explicitly disables
cross-task ranking and automatic candidate selection.

## Cost and performance evidence

VAIT can attach provider-neutral economic evidence to a benchmark verification
report without allowing cost to override behavioural or risk failures.

The controlled M2 experiment can be run with:

    python examples/evaluate_ap_cost_performance_v01.py

It writes:

    artifacts/benchmarks/ap-v0.2-cost-performance-evidence-v0.1.json

The current scenario uses declared development-only cost assumptions to
exercise the evidence pipeline. These values are not provider prices,
production cost estimates, or ROI claims.

Candidate latency is measured in the current execution environment. The AP
benchmark gold-label runner is not treated as a timed reference
implementation, so comparative reference latency and latency deltas are not
reported unless a genuinely measured reference is available.

## Engineering quality

The repository currently uses:

- Python 3.12;
- Pydantic and Typer;
- pytest and Hypothesis;
- Ruff;
- strict mypy;
- scikit-learn and XGBoost;
- PyTorch;
- TensorFlow/Keras;
- Hugging Face Transformers;
- PEFT and bitsandbytes.

Generated benchmark evidence under `artifacts/` is intentionally excluded from
version control.

## Scope

The Accounts Payable invoice-exception domain is a synthetic development
benchmark used to exercise VAIT's verification and AI-engineering paths.

It is not a production financial decision system.

Current evidence does not establish unseen generalisation, universal model
superiority, hallucination-free generation, or production safety.

See:

- `docs/PROJECT-SPEC.md`
- `docs/ARCHITECTURE.md`
- `docs/BENCHMARK.md`
- `docs/APPLIED-TRANSFORMATIONS.md`
- `docs/LIMITATIONS.md`
- `docs/RESEARCH-BACKGROUND.md`
