# Architecture — Verified AI Workflow Transformation v0.1

**Status:** implemented v0.1 architecture
**Scope:** bounded verification, inference optimisation evidence, and non-executing transformation recommendation planning

## 1. Architectural principle

VAIT is a **verification and evidence-driven decision layer**, not a workflow runtime.

Its core engineering rule is:

```text
VERIFY FIRST -> OPTIMISE SECOND
```

VAIT evaluates supplied transformation candidates under explicit contracts and evidence.

A candidate must pass compatibility, verification, and admissibility requirements before optimisation evidence can influence any downstream recommendation.

High-level architecture:

```text
Reference implementation -----\
                               \
Candidate implementation ------> Candidate Runner ----\
                                                       \
Transformation contract -------------------------------> Verification Engine
                                                        /        |
Evaluation dataset ------------------------------------/         |
                                                                  v
                                                        EXACT / BOUNDED / REJECT
                                                                  |
                                                                  v
                                                         Admissibility Gate
                                                                  |
                                                                  v
                                                  Controlled Inference Evidence
                                                                  |
                                                                  v
                                                     Comparability Validation
                                                                  |
                                                                  v
                                                       Hard Constraints
                                                                  |
                                                                  v
                                                        Pareto Analysis
                                                                  |
                                                                  v
                                                    Preference Policy, if any
                                                                  |
                                                                  v
                                               Recommendation Planning
                                                                  |
                                                                  v
                                              Reverification Required
```

A faster, cheaper, smaller, or lower-resource candidate cannot bypass a failed verification decision.

## 2. Core components

### 2.1 Contract layer

Responsibilities:

- validate transformation-contract syntax;
- define input/output schemas;
- define invariants;
- define risk strata;
- define statistical thresholds;
- define hard rejection conditions;
- optionally define cost and latency constraints.

Implemented surface:

```text
src/vait/contracts/
```

Contracts establish obligations before benchmark results are interpreted.

### 2.2 Runner layer

Responsibilities:

- execute supplied implementations;
- normalise execution results;
- preserve implementation identity;
- capture outputs, latency, errors, and optional execution metadata.

Implemented surface:

```text
src/vait/runners/
```

The runner layer does not decide whether a transformation is acceptable.

### 2.3 Verification layer

Responsibilities:

- deterministic equivalence checks;
- behavioural comparison;
- risk-stratified comparison;
- bounded statistical verification;
- statistical uncertainty calculations;
- hard rejection evaluation.

Implemented surface:

```text
src/vait/verification/
```

Verification outcomes are:

```text
EXACT
BOUNDED
REJECT
```

`EXACT` is intentionally narrow.

`BOUNDED` is conditional on the declared benchmark, risk strata, thresholds, and statistical evidence.

`REJECT` is terminal for the evaluated candidate evidence.

### 2.4 Decision layer

Responsibilities:

- combine verification evidence;
- apply precedence rules;
- emit explicit verification classifications;
- preserve machine-readable reason semantics.

Implemented surface:

```text
src/vait/decision/
```

Hard failures always take precedence over average performance.

Example:

```text
high overall agreement
BUT
high-risk constraint fails

=> REJECT
```

### 2.5 Transformation layer

Responsibilities:

- describe supplied transformation candidates;
- validate transformation applicability;
- preserve transformation identity and version;
- prepare supported candidate implementations for evaluation.

Implemented surface:

```text
src/vait/transformations/
```

VAIT does not synthesize arbitrary transformation candidates.

### 2.6 Economics layer

Responsibilities:

- represent controlled or declared cost evidence;
- compare cost evidence where meaningful;
- compare measured latency evidence;
- calculate simple deltas;
- preserve economic evidence without overriding verification failures.

Implemented surface:

```text
src/vait/economics/
```

Economics is evidence, not acceptance authority.

### 2.7 Retrieval and RAG layer

Responsibilities:

- encode and retrieve versioned evidence;
- evaluate ranked retrieval;
- construct grounded RAG contexts;
- preserve abstention behaviour;
- evaluate faithfulness;
- apply semantic guards and fallback behaviour;
- support provider-backed generation through provider-neutral interfaces.

Implemented surfaces:

```text
src/vait/retrieval/
src/vait/rag/
```

Raw model generation and guarded surfaced behaviour remain distinct evidence.

### 2.8 Inference layer

Responsibilities:

- define inference workloads;
- define controlled benchmark configurations;
- capture latency and throughput;
- benchmark generative inference;
- support native batching;
- support controlled concurrency;
- support deterministic response caching;
- preserve environment and configuration provenance;
- capture process-level resource evidence.

Implemented surface:

```text
src/vait/inference/
```

Provider adapters are separated from the provider-neutral inference contracts.

Current provider-related implementations include local Hugging Face, OpenAI-compatible transport, and IBM watsonx reference integration.

The provider abstraction remains independent of any single vendor.

### 2.9 Search-space and admissibility layer

Responsibilities:

- describe explicit candidate/configuration search points;
- check configuration compatibility;
- preserve verification requirements;
- reject incompatible points before execution;
- evaluate verification-backed admissibility;
- prevent `REJECT` evidence from entering optimisation.

Implemented surface:

```text
src/vait/optimisation/search_space.py
src/vait/optimisation/compatibility.py
src/vait/optimisation/admissibility.py
src/vait/optimisation/search_plan.py
```

The search space is supplied and explicitly bounded.

VAIT does not perform unrestricted architecture search or autonomous candidate discovery.

### 2.10 Structured inference benchmarking

Responsibilities:

- benchmark only admissible search points;
- preserve candidate and configuration identity;
- run repeated controlled measurements;
- aggregate repeatability evidence;
- benchmark native batch scaling.

Implemented surface:

```text
src/vait/optimisation/structured_benchmark.py
src/vait/optimisation/structured_batch_benchmark.py
```

Benchmark evidence remains scoped to the declared workload and environment.

### 2.11 Resource profiling

Responsibilities:

- record measured-region wall time;
- record process CPU time;
- calculate CPU-time-to-wall-time ratio;
- record process peak RSS baseline;
- record process peak RSS;
- record peak-RSS growth.

Implemented surface:

```text
src/vait/inference/profiling.py
```

Peak RSS is a process-lifetime high-water mark.

It is not candidate-exclusive memory and must not be presented as GPU or accelerator-memory evidence.

### 2.12 Multi-objective optimisation layer

M4-I introduces a pure decision layer over already admissible evidence.

The required order is:

```text
admissibility
        |
        v
workload / protocol comparability
        |
        v
resource-evidence coverage
        |
        v
hard optimisation constraints
        |
        v
Pareto frontier
        |
        v
explicit preference policy, if declared
```

Implemented surface:

```text
src/vait/optimisation/multi_objective.py
```

Supported objective evidence includes:

- mean latency;
- mean throughput;
- mean process CPU time;
- maximum process peak RSS;
- model size.

Hard constraints are applied before Pareto analysis.

A constraint-rejected point cannot enter the Pareto frontier.

A dominated point cannot later become preferred.

The optimiser does not invent an opaque global winner.

### 2.13 Recommendation-planning layer

M4-J consumes an existing multi-objective analysis.

Responsibilities:

- preserve candidate identity;
- preserve verification/admissibility basis;
- preserve comparability scope;
- preserve objectives and constraints;
- preserve Pareto-frontier evidence;
- preserve preference results;
- emit an explicit recommendation state.

Implemented surface:

```text
src/vait/optimisation/recommendation.py
```

Recommendation states are:

```text
READY
UNRESOLVED_PREFERENCE_TIE
NO_FEASIBLE_CANDIDATE
```

A `READY` plan requires exactly one preferred admissible frontier candidate.

An unresolved tie remains unresolved.

A no-feasible-candidate result remains an explicit hold state.

Automatic execution is prohibited:

```text
automatic_execution_allowed = false
```

A recommendation plan is evidence, not an execution command.

If a selected transformation is applied, the required next gate is reverification.

### 2.14 Release-evidence layer

M4-K introduces a versioned release-evidence manifest.

Responsibilities:

- bind milestone artifacts to one source revision;
- require a clean source tree;
- require repository-relative evidence paths;
- require valid JSON evidence;
- record SHA-256 fingerprints;
- record artifact sizes;
- reject duplicate evidence paths.

Implemented surface:

```text
src/vait/evidence/release_manifest.py
```

The manifest provides provenance over generated evidence.

It does not create a performance, safety, or production-readiness claim.

## 3. Core conceptual data model

Important conceptual types include:

```text
TransformationContract
EvaluationCase
ExecutionResult
InvariantResult
RiskStratumResult
StatisticalEvidence
EconomicEvidence
VerificationResult
TransformationDecision
EvidenceReport

InferenceWorkload
InferenceConfiguration
InferenceBenchmarkReport
InferenceResourceEvidence

InferenceSearchPoint
SearchPointCompatibility
SearchPointVerificationEvidence
SearchPointAdmissibility
ExecutableSearchPlan

MultiObjectiveCandidateEvidence
CandidateObjectiveVector
ObjectiveConstraint
ConstraintRejectedCandidate
MultiObjectiveAnalysis

TransformationRecommendationPlan

ReleaseEvidenceArtifact
ReleaseEvidenceManifest
```

Externally persisted evidence structures use explicit versioned schemas.

## 4. Verification precedence

The verification path follows this conceptual order:

```text
1. Contract validity
2. Execution validity
3. Deterministic / hard invariants
4. Risk-specific constraints
5. Exact-verification checks, when supported
6. Bounded statistical checks
7. Verification classification
8. Economic / performance evidence
```

Performance evidence cannot override verification failure.

## 5. Optimisation precedence

For M4 optimisation, the downstream order is:

```text
1. Candidate compatibility
2. Verification evidence
3. Admissibility
4. Controlled benchmark evidence
5. Workload / protocol comparability
6. Required resource-evidence coverage
7. Hard optimisation constraints
8. Pareto frontier
9. Explicit preference policy, if declared
10. Recommendation planning
```

No later stage may resurrect:

- a verification `REJECT`;
- an incompatible search point;
- a non-admissible point;
- a constraint-rejected point;
- a dominated point.

## 6. EXACT path

The exact path is intentionally narrow.

It is appropriate only where VAIT can establish the declared deterministic obligations for a supported scope, including cases such as:

- output equality;
- invariant preservation;
- deterministic equality over an explicitly bounded domain;
- deterministic rule or relational checks implemented by the verifier.

`EXACT` must not imply universal semantic equivalence for arbitrary probabilistic systems.

## 7. BOUNDED path

The bounded path is the primary statistical verification path.

For a fixed versioned evaluation set it can evaluate:

- overall disagreement;
- risk-stratified disagreement;
- high-risk failures;
- statistical confidence bounds;
- task-specific evidence;
- latency evidence;
- economic evidence.

A `BOUNDED` result is conditional on the declared evaluation setup.

It is not a guarantee for unseen future inputs.

## 8. REJECT path

`REJECT` is returned when declared verification obligations fail or sufficient evidence is not established.

Examples include:

```text
SCHEMA_MISMATCH
EXECUTION_FAILURE
HARD_INVARIANT_VIOLATION
RISK_THRESHOLD_EXCEEDED
QUALITY_THRESHOLD_EXCEEDED
INSUFFICIENT_EVIDENCE
UNSUPPORTED_TRANSFORMATION
```

A `REJECT` result cannot enter downstream optimisation regardless of performance.

## 9. NOT_APPLICABLE path

`NOT_APPLICABLE` is separate from verification failure.

It means a transformation does not meet declared applicability requirements and therefore is not executed as a valid candidate for that workflow context.

It must not be conflated with `REJECT`.

## 10. Repository structure

```text
verified-ai-workflow-transformation/
├── src/
│   └── vait/
│       ├── benchmark/
│       ├── contracts/
│       ├── decision/
│       ├── economics/
│       ├── evidence/
│       ├── inference/
│       ├── optimisation/
│       ├── rag/
│       ├── retrieval/
│       ├── runners/
│       ├── transformations/
│       └── verification/
├── benchmarks/
├── tests/
│   ├── unit/
│   ├── property/
│   └── integration/
├── docs/
├── examples/
├── artifacts/
├── pyproject.toml
├── README.md
└── LICENSE
```

Generated evidence under `artifacts/` is intentionally excluded from version control.

## 11. Dependency direction

Core verification logic must not depend on cloud/provider SDK abstractions.

Conceptually:

```text
contracts
    |
    v
runners / transformations
    |
    v
verification
    |
    v
decision
```

Inference engineering extends that flow as:

```text
provider-neutral inference contracts
        |
        v
provider adapters
        |
        v
controlled benchmark evidence
        |
        v
admissibility-preserving optimisation
        |
        v
non-executing recommendation planning
```

Evidence flow:

```text
verification evidence
        |
        v
admissibility
        |
        v
comparable performance/resource evidence
        |
        v
multi-objective analysis
        |
        v
recommendation plan
        |
        v
reverification after transformation
```

The optimisation and recommendation layers consume evidence.

They do not weaken or replace verification.

## 12. Provider neutrality

The project core must remain provider-neutral.

Provider-specific adapters may implement shared contracts, but provider identity must not determine verification or optimisation semantics.

IBM watsonx is a reference integration only.

The same core evidence and decision contracts must remain usable with local or other compatible providers.

## 13. Evidence provenance

Evidence artifacts preserve relevant provenance where available, including:

- source revision;
- workload identity;
- workload fingerprint;
- model identity;
- inference configuration;
- environment;
- benchmark configuration;
- measurement scope.

M4-K adds artifact-level SHA-256 binding through the release-evidence manifest.

## 14. Deferred architecture

The following remain outside the v0.1 architecture unless a measured requirement justifies them:

- distributed workflow orchestration;
- durable job queues;
- Redis-based orchestration state;
- Kubernetes/OpenShift deployment control;
- React application UI;
- Neo4j or graph-database workflow modelling;
- multi-agent frameworks;
- automatic candidate synthesis;
- autonomous transformation execution;
- unrestricted architecture search;
- unrestricted enterprise-workflow optimisation;
- human-capacity optimisation;
- compliance certification.

VAIT remains a bounded verification and evidence-driven optimisation system rather than an autonomous workflow transformation platform.
