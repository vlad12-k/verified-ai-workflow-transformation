# Verified AI Workflow Transformation (VAIT)

VAIT is a provider-neutral verification and evaluation layer for assessing supplied AI workflow transformation candidates under explicit behavioural, risk, statistical, latency, resource, and economic evidence.

The project asks a narrower question than a general AI optimiser:

> Given an already-specified workflow and a supplied replacement candidate, what evidence can be established before accepting the transformation?

VAIT does not synthesize arbitrary replacements or act as a runtime orchestrator.

Its optimisation layer operates only on already verified and admissible candidate evidence within an explicitly comparable search scope. Recommendation planning does not automatically execute a transformation.

The core engineering rule is:

```text
VERIFY FIRST -> OPTIMISE SECOND
```

## Verification outcomes

VAIT distinguishes between:

- **EXACT** — exact obligations have been established for a supported, explicitly bounded deterministic scope;
- **BOUNDED** — the candidate satisfies a declared statistical verification policy on the evaluated benchmark and risk strata;
- **REJECT** — verification constraints were not satisfied;
- **NOT_APPLICABLE** — the candidate does not satisfy the declared workflow applicability requirements and is therefore not executed.

`BOUNDED` is not a claim of universal semantic equivalence.

A `REJECT` result cannot later become admissible or recommendable because a candidate is faster, cheaper, smaller, or otherwise attractive on an optimisation metric.

## Implemented AI breadth

M3 established heterogeneous development evidence across several AI implementation families.

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

A controlled PyTorch-versus-Keras experiment demonstrates why VAIT separates training quality from verification.

Both implementations used the same 11-feature AP representation, the same 32/16 hidden-layer architecture, 963 parameters, and approximately 99.67% development training accuracy.

On AP benchmark v0.2:

```text
PyTorch: BOUNDED — 0/20 observed disagreements
Keras:   REJECT  — 1/20 observed disagreements
```

The Keras disagreement occurred on a reproducible high-risk amount-mismatch boundary case.

The result is development evidence and is not a general framework comparison.

## Unified heterogeneous evidence

After generating the individual development artifacts, run:

```bash
python examples/build_m3_heterogeneous_evidence_v01.py
```

This creates:

```text
artifacts/benchmarks/m3-heterogeneous-evidence-v0.1.json
```

The report groups evidence by compatible task family and explicitly disables cross-task ranking and automatic candidate selection.

## Cost and performance evidence

VAIT can attach provider-neutral economic evidence to a benchmark verification report without allowing cost to override behavioural or risk failures.

The controlled M2 experiment can be run with:

```bash
python examples/evaluate_ap_cost_performance_v01.py
```

It writes:

```text
artifacts/benchmarks/ap-v0.2-cost-performance-evidence-v0.1.json
```

The current scenario uses declared development-only cost assumptions to exercise the evidence pipeline.

These values are not provider prices, production cost estimates, or ROI claims.

Candidate latency is measured in the current execution environment.

The AP benchmark gold-label runner is not treated as a timed reference implementation, so comparative reference latency and latency deltas are not reported unless genuinely measured reference evidence is available.

## Verify-first inference optimisation

M4 extends VAIT from transformation verification into bounded inference engineering and multi-objective optimisation while preserving the original verification boundary.

The high-level flow is:

```text
declared candidate / search point
        |
        v
compatibility
        |
        v
verification
        |
        v
admissibility
        |
        v
controlled benchmark evidence
        |
        v
resource evidence
        |
        v
comparability checks
        |
        v
hard optimisation constraints
        |
        v
Pareto frontier
        |
        v
explicit preference policy, if declared
        |
        v
non-executing recommendation plan
```

Optimisation therefore happens only after the candidate has passed the relevant compatibility, verification, and admissibility gates.

## M4 structured inference engineering

M4 adds provider-neutral inference configurations and controlled benchmark evidence for structured and generative workloads.

Implemented evidence includes:

- scalar inference benchmarking;
- repeated benchmark execution;
- native batch execution;
- latency distributions;
- throughput measurements;
- workload identity and fingerprinting;
- environment and source provenance;
- model/configuration identity;
- warm-up and measured execution regions.

The provider-neutral core is separated from provider adapters.

IBM watsonx is treated as a reference integration rather than the project abstraction.

## M4 precision, compression, and compilation

M4-D and M4-E evaluate structured inference alternatives only after verification.

The development evidence includes:

- eager FP32 PyTorch inference;
- FP16;
- BF16;
- dynamic INT8;
- `torch.compile` / Inductor;
- hard-label compact students;
- distilled compact students;
- model-size evidence;
- scalar and batch performance comparison.

Example builders:

```text
examples/build_m4d_structured_comparison_v01.py
examples/build_m4e_precision_compression_comparison_v01.py
examples/build_m4e_final_comparison_v01.py
```

These experiments are hardware- and workload-specific.

They do not establish universal superiority of one precision, compiler, or model representation.

## M4 SLM and guarded RAG evidence

M4-F extends the inference path to small language models and provider-backed RAG.

The development comparison includes:

- SmolLM2;
- Qwen2.5-0.5B;
- raw provider generation;
- guarded provider generation;
- semantic admissibility;
- fallback routing;
- surfaced semantic pass evidence;
- token throughput;
- end-to-end latency.

The comparison builder is:

```text
examples/build_m4f_slm_rag_comparison_v01.py
```

Raw semantic generation and surfaced guarded behaviour are reported separately.

A guarded result must not be interpreted as evidence that the underlying model itself became semantically reliable.

## M4 batching, concurrency, and deterministic caching

M4-G evaluates execution strategies while keeping verification gates intact.

Implemented evidence includes:

- request concurrency;
- native batching;
- same-width concurrency-versus-batching comparisons;
- deterministic response caching;
- cold and warm cache evidence;
- provider-generation avoidance evidence.

The summary builder is:

```text
examples/build_m4g_execution_optimisation_summary_v01.py
```

Performance observations are descriptive for the measured workload and environment.

They are not production capacity guarantees.

## M4 resource profiling

M4-H adds process-level resource evidence.

The profiling layer currently records:

- measured-region wall time;
- process CPU time;
- CPU-time-to-wall-time ratio;
- process peak RSS baseline;
- process peak RSS;
- process peak RSS growth;
- model parameter count;
- model parameter storage.

The profiling example is:

```text
examples/profile_ap_pytorch_resources_v01.py
```

Peak RSS is explicitly treated as a **process-lifetime high-water mark**.

It is not candidate-exclusive resident memory, GPU memory, accelerator memory, or formal memory-complexity evidence.

A zero RSS-growth observation means only that the measured region did not establish a new process high-water mark.

## M4 multi-objective optimisation

M4-I adds a pure decision layer over already admissible and comparable evidence.

The optimisation order is:

```text
admissibility
    ->
workload / protocol comparability
    ->
resource-evidence coverage
    ->
hard constraints
    ->
Pareto frontier
    ->
explicit preference policy
```

Supported optimisation metrics include:

- mean latency;
- mean cases per second;
- mean process CPU time;
- maximum process peak RSS;
- model size.

Hard constraints are evaluated before Pareto analysis.

A constraint-rejected point cannot re-enter the Pareto frontier.

A dominated point cannot become preferred through a later tie-break.

The optimiser intentionally does not create an opaque global "winner".

The example analysis is:

```bash
python examples/analyse_m4i_multi_objective_v01.py
```

It writes:

```text
artifacts/benchmarks/m4i-multi-objective-analysis-v0.1.json
```

## M4 recommendation planning

M4-J converts the result of multi-objective analysis into an explicit transformation recommendation plan.

Three states are represented:

- `READY`;
- `UNRESOLVED_PREFERENCE_TIE`;
- `NO_FEASIBLE_CANDIDATE`.

A `READY` plan is produced only when the preferred Pareto frontier resolves to exactly one admissible candidate.

An unresolved tie is preserved rather than silently broken.

A no-feasible-candidate state remains an explicit hold state.

Recommendation actions include:

```text
apply_selected_search_point
hold_unresolved_preference_tie
hold_no_feasible_candidate
```

Required next gates include:

```text
reverify_after_transformation
resolve_preference_tie
revise_feasibility
```

Automatic execution is explicitly disabled:

```text
automatic_execution_allowed = false
```

A selected transformation must therefore be reverified after it is applied.

The M4-J example is:

```bash
python examples/build_m4j_recommendation_plan_v01.py
```

It writes:

```text
artifacts/benchmarks/m4j-recommendation-plan-v0.1.json
```

## M4 release evidence

M4-K adds a canonical release-evidence manifest that binds milestone evidence to one clean source revision.

The manifest records, for each included evidence artifact:

- milestone identity;
- repository-relative path;
- SHA-256 digest;
- artifact size.

The manifest builder rejects:

- dirty source revisions;
- missing evidence artifacts;
- invalid JSON evidence;
- paths that escape the repository;
- duplicate evidence paths.

The builder is:

```bash
python examples/build_m4k_release_evidence_manifest_v01.py
```

The resulting artifact is:

```text
artifacts/benchmarks/m4k-release-evidence-manifest-v0.1.json
```

The manifest is evidence provenance, not a performance or production-safety claim.

## Engineering quality

The repository uses:

- Python 3.12;
- Pydantic and Typer;
- pytest and Hypothesis;
- Ruff;
- strict mypy;
- branch-aware test coverage;
- scikit-learn and XGBoost;
- PyTorch;
- TensorFlow/Keras;
- Hugging Face Transformers;
- PEFT and bitsandbytes;
- GitHub Actions CI.

The current quality gate includes:

```text
python -m compileall -q src tests examples
ruff check .
mypy src tests examples
pytest --cov=vait --cov-branch --cov-fail-under=85
```

M4-I and M4-J evidence builders are also executed as CI smoke checks.

Generated benchmark evidence under `artifacts/` is intentionally excluded from version control.

## Scope

The Accounts Payable invoice-exception domain is a synthetic development benchmark used to exercise VAIT's verification and AI-engineering paths.

It is not a production financial decision system.

The current evidence does not establish:

- unseen generalisation;
- universal model superiority;
- universal semantic equivalence;
- hallucination-free generation;
- production safety;
- production capacity;
- real financial ROI;
- compliance certification.

Multi-objective results are meaningful only within their declared comparable workload and measurement protocol.

Recommendation planning does not imply autonomous execution.

See:

- `docs/PROJECT-SPEC.md`
- `docs/ARCHITECTURE.md`
- `docs/BENCHMARK.md`
- `docs/APPLIED-TRANSFORMATIONS.md`
- `docs/LIMITATIONS.md`
- `docs/RESEARCH-BACKGROUND.md`
