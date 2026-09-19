# Limitations and Claim Boundaries

## 1. Purpose

This document defines what VAIT v0.1 does **not** prove.

These limitations are part of the engineering contract of the project, not optional disclaimers.

VAIT is designed around the principle:

```text
VERIFY FIRST -> OPTIMISE SECOND
```

Optimisation evidence must not weaken, bypass, or reinterpret a failed verification decision.

## 2. No universal semantic equivalence

VAIT v0.1 does not prove semantic equivalence for arbitrary LLMs, neural models, open-ended workflows, or unrestricted probabilistic systems.

`EXACT` is restricted to explicitly supported deterministic obligations and bounded transformation classes.

A successful deterministic check must not be presented as a proof of arbitrary model equivalence.

## 3. Statistical evidence is conditional

`BOUNDED` decisions are conditional on:

- the declared evaluation distribution;
- the versioned dataset;
- the chosen metrics;
- the stated confidence level;
- the declared risk strata;
- the verification contract;
- the evaluated model and configuration.

A `BOUNDED` result is not a guarantee about all future inputs.

It does not establish universal safety, unseen generalisation, or production reliability.

## 4. Synthetic benchmark limitations

The Accounts Payable invoice-exception benchmark is a controlled development benchmark.

It may include synthetic or deliberately constructed cases.

Therefore VAIT v0.1 must not claim:

- real operational ROI;
- real fraud reduction;
- real financial-loss reduction;
- real human-productivity savings;
- production-grade financial safety;
- compliance certification;
- real-world deployment readiness.

The benchmark exists to exercise the verification and optimisation architecture.

VAIT is not an Accounts Payable automation product.

## 5. No automatic candidate synthesis

VAIT evaluates supplied transformation candidates and explicitly declared search points.

It does not autonomously invent arbitrary replacement implementations.

It does not perform unrestricted model architecture search.

It does not autonomously generate production transformation code.

## 6. Bounded optimisation is not general workflow optimisation

M4 includes multi-objective optimisation over explicitly declared, already verified, admissible, and comparable candidate configurations.

This does not make VAIT a general:

- enterprise workflow optimiser;
- AI workload compiler;
- autonomous model router;
- orchestration platform;
- architecture-search platform;
- agentic optimisation system.

Optimisation remains bounded by:

- supplied candidate configurations;
- explicit verification evidence;
- admissibility rules;
- comparability requirements;
- declared objectives;
- hard optimisation constraints;
- optional explicit preference policy.

VAIT does not optimise an unrestricted enterprise workflow.

## 7. Verification failure cannot be rescued by performance

A candidate classified as `REJECT` cannot later become acceptable because it is:

- faster;
- cheaper;
- smaller;
- more CPU-efficient;
- more memory-efficient;
- higher-throughput.

Verification and admissibility precede optimisation.

Performance evidence is downstream evidence only.

## 8. NOT_APPLICABLE is separate from REJECT

`NOT_APPLICABLE` means that a candidate does not satisfy the declared applicability requirements for the workflow context and therefore is not executed as a valid candidate.

It must not be interpreted as:

- a successful verification;
- a failed statistical verification;
- a performance ranking;
- an optimisation result.

`NOT_APPLICABLE` and `REJECT` represent different conditions.

## 9. No enterprise workflow execution

VAIT does not own runtime orchestration.

It does not provide:

- durable workflow execution;
- distributed job queues;
- retry orchestration;
- production scheduling;
- workflow state management;
- deployment control;
- automatic rollback;
- autonomous transformation application.

Recommendation planning produces evidence, not execution commands.

## 10. No human-capacity optimisation

VAIT v0.1 does not model or optimise:

- workforce queues;
- staffing levels;
- shift schedules;
- organisational capacity;
- employee productivity;
- human resource allocation.

Human review may appear as a policy or fallback outcome, but it is not a workforce-optimisation model.

## 11. No privacy proof

VAIT may preserve declared privacy-related metadata where relevant, but v0.1 does not provide formal privacy verification.

It does not establish:

- differential privacy;
- formal information-flow guarantees;
- anonymisation guarantees;
- data-protection compliance;
- protection against all data leakage.

Such claims would require an explicit threat model and enforceable privacy constraints.

## 12. No compliance certification

Passing a VAIT verification contract does not imply compliance with:

- GDPR;
- financial regulation;
- audit requirements;
- internal governance policies;
- safety standards;
- sector-specific regulation.

VAIT evidence may support engineering review, but it is not a compliance certificate.

## 13. Cost evidence is conditional

Cost evidence depends on the declared scenario and measurement inputs.

Development-only cost assumptions must not be presented as:

- current provider pricing;
- production operating cost;
- realised savings;
- business ROI;
- contractual pricing evidence.

Cost evidence cannot override verification or risk failures.

## 14. Latency and throughput evidence is conditional

Latency and throughput depend on factors including:

- hardware;
- operating system;
- model version;
- implementation;
- runtime;
- thread configuration;
- provider;
- geography;
- network conditions;
- workload shape;
- batch size;
- concurrency;
- warm-up;
- measurement protocol.

Measured development performance must not be presented as a universal production benchmark.

A frozen benchmark gold-label runner is not automatically a valid timed reference implementation.

Reference-versus-candidate latency comparisons require genuinely comparable measured evidence.

## 15. Precision, compression, and compilation evidence is workload-specific

Observed behaviour for:

- FP16;
- BF16;
- dynamic INT8;
- `torch.compile`;
- compact student models;
- distilled models;

is specific to the evaluated implementation, workload, hardware, and environment.

An observed speedup, slowdown, or model-size reduction does not establish universal superiority of that technique.

## 16. Batching and concurrency evidence is scoped

Native batching and request concurrency represent different execution strategies.

Their results depend on:

- workload width;
- batch size;
- concurrency level;
- scheduling overhead;
- model implementation;
- execution environment.

A development throughput measurement is not a production capacity guarantee.

## 17. Cache evidence is conditional

Deterministic response-cache evidence applies only where the cache key and execution semantics are valid for the evaluated workload.

A cache hit does not establish that arbitrary generative workloads are safely cacheable.

Cache measurements must distinguish cold and warm behaviour.

Avoided provider generation is evidence about the evaluated deterministic cache path only.

## 18. Guarded generation is not raw-model reliability

Semantic guards, abstention, and fallback routing may improve surfaced system behaviour.

This does not mean the underlying language model itself has become semantically reliable.

VAIT therefore distinguishes between:

- raw generation behaviour;
- guarded surfaced behaviour;
- fallback behaviour.

A successful guard must not be presented as proof of hallucination-free generation.

## 19. Retrieval and RAG evidence is benchmark-specific

Retrieval and RAG measurements are conditional on:

- the indexed corpus;
- corpus version;
- chunking;
- embedding model;
- retrieval configuration;
- query set;
- ranking procedure;
- prompt contract;
- evaluation policy.

Passing a controlled retrieval or faithfulness evaluation does not establish universal grounding.

## 20. Resource evidence is scoped

M4-H process-resource profiling is descriptive evidence for the measured execution environment.

Current process-level evidence includes:

- wall time;
- process CPU time;
- CPU-time-to-wall-time ratio;
- process peak RSS baseline;
- process peak RSS;
- peak-RSS growth;
- parameter count;
- parameter storage.

Process peak RSS is a **process-lifetime high-water mark**.

It is not:

- candidate-exclusive resident memory;
- GPU memory;
- accelerator memory;
- formal memory complexity;
- a complete allocation trace.

A zero peak-RSS-growth observation means only that the measured region did not establish a new process high-water mark.

## 21. Multi-objective optimisation requires comparability

M4-I does not compare arbitrary heterogeneous evidence.

Candidates must share an explicit compatible workload and measurement protocol before entering the same multi-objective analysis.

Cross-task optimisation is not supported.

Results from different task families, incompatible benchmark protocols, or incomparable environments must not be combined into a single optimisation frontier.

## 22. Hard constraints precede Pareto analysis

Hard optimisation constraints are applied before Pareto-frontier construction.

A constraint-rejected point cannot re-enter the frontier because it performs well on another objective.

Similarly, a verification `REJECT` cannot enter optimisation at all.

This precedence is intentional.

## 23. Pareto membership is not universal superiority

A Pareto-frontier candidate is not automatically the "best" system.

Pareto membership only means that, under the declared objectives and comparable evidence, the point is not dominated by another feasible point.

Different objectives or constraints may produce a different frontier.

VAIT therefore does not claim a universal global winner.

## 24. Preference policy must be explicit

A preference policy may narrow a Pareto frontier only when explicitly declared.

It must not introduce hidden tie-breaking.

If several preferred frontier candidates remain, the result is an unresolved preference tie.

VAIT must preserve that uncertainty rather than silently select one candidate.

## 25. Recommendation does not mean automatic execution

M4-J recommendation planning has three explicit states:

```text
READY
UNRESOLVED_PREFERENCE_TIE
NO_FEASIBLE_CANDIDATE
```

A `READY` result means that one uniquely preferred admissible frontier point has been identified under the declared evidence and preference policy.

It does **not** mean that VAIT automatically changes the deployed system.

Automatic execution is explicitly disabled:

```text
automatic_execution_allowed = false
```

If a selected transformation is applied, the required next gate is reverification.

## 26. Recommendation is conditional on its source evidence

A recommendation is only as valid as the evidence from which it was constructed.

Changes to any of the following may invalidate the recommendation:

- model version;
- implementation;
- hardware;
- workload;
- benchmark;
- provider;
- inference configuration;
- resource measurements;
- constraints;
- preference policy.

Recommendation artifacts must therefore preserve relevant provenance.

## 27. Release-evidence manifest is provenance, not certification

M4-K binds selected generated evidence artifacts to a clean Git source revision using:

- repository-relative artifact paths;
- SHA-256 digests;
- artifact sizes;
- source revision identity.

This provides artifact provenance.

It does not prove:

- correctness of every underlying benchmark assumption;
- production readiness;
- external reproducibility;
- safety;
- compliance;
- performance superiority.

A checksum proves artifact identity, not scientific validity.

## 28. Generated artifacts are not source code

Generated benchmark evidence under `artifacts/` is intentionally excluded from version control.

Evidence should be reproducible through the corresponding builders where the required dependencies, model assets, environment, and inputs are available.

Some model- or provider-backed experiments may require external resources that are not available in every environment.

## 29. Provider neutrality does not mean identical provider behaviour

VAIT's core contracts are provider-neutral.

This does not imply that:

- different providers behave identically;
- model outputs are interchangeable;
- latency is directly comparable across providers without a controlled protocol;
- pricing is comparable without explicit cost evidence.

Provider-specific behaviour must remain visible in the evidence.

## 30. Development evidence is not production safety

The current project demonstrates engineering mechanisms for:

- verification;
- admissibility;
- benchmarking;
- resource profiling;
- multi-objective analysis;
- recommendation planning;
- evidence provenance.

It does not establish that any evaluated candidate is safe for unrestricted production use.

## 31. Benchmark claims must remain scoped

Public claims should use wording such as:

> On benchmark X, under contract Y and measurement protocol Z, candidate C satisfied the declared verification and optimisation constraints.

Avoid wording such as:

> Candidate C is universally safe.

Avoid:

> Candidate C is always faster.

Avoid:

> Candidate C is the best model.

Avoid:

> VAIT proves production readiness.

Claims should remain tied to the actual evaluated evidence.
