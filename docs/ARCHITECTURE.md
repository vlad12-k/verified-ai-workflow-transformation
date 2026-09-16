# Architecture — Verified AI Workflow Transformation v0.1

**Status:** v0.1 build architecture
**Scope:** bounded, read-only decision transformations

## 1. Architectural principle

VAIT is a **verification and decision layer**, not a workflow runtime.

It compares a reference implementation with a supplied candidate under an explicit contract, evaluates evidence, and emits a transformation decision.

```text
Reference implementation -----\
                               \
Candidate implementation ------> Candidate Runner ----\
                                                       \
Transformation contract -------------------------------> Verification Engine
                                                        /        |
Evaluation dataset ------------------------------------/         |
                                                                  v
                                                          Decision Engine
                                                                  |
                                                                  v
                                                       Evidence Report
                                                       EXACT / BOUNDED / REJECT
```

## 2. Core components

### 2.1 Contract layer

Responsibilities:

- validate transformation-contract syntax;
- define input/output schemas;
- define invariants;
- define risk strata;
- define statistical thresholds;
- define hard rejection conditions;
- optionally define cost/latency thresholds.

Proposed module:

```text
src/vait/contracts/
    models.py
    invariants.py
    risk.py
```

### 2.2 Runner layer

Responsibilities:

- execute the reference implementation;
- execute the candidate implementation;
- normalise execution results;
- capture outputs, latency, errors, and optional usage metadata.

Proposed module:

```text
src/vait/runners/
    base.py
    python_runner.py
    provider_runner.py
```

`provider_runner.py` is an interface. An IBM watsonx.ai adapter can be added after the provider-neutral interface is stable.

### 2.3 Verification layer

Responsibilities:

- schema compatibility checks;
- deterministic invariant checks;
- behavioural comparison;
- risk-stratified comparison;
- statistical uncertainty calculations;
- threshold evaluation.

Proposed module:

```text
src/vait/verification/
    schema.py
    behaviour.py
    statistics.py
    risk.py
```

### 2.4 Decision layer

Responsibilities:

- combine verification evidence;
- apply contract precedence rules;
- emit `EXACT`, `BOUNDED`, or `REJECT`;
- emit machine-readable reason codes.

Proposed module:

```text
src/vait/decision/
    engine.py
    models.py
```

Hard failures always take precedence over average performance.

Example:

```text
99.5% overall agreement
BUT
high-risk invariant violation > 0
=> REJECT
```

### 2.5 Economics layer

Responsibilities:

- compare measured/declared execution cost;
- compare latency;
- calculate simple per-case deltas;
- report economic evidence without overriding safety/risk decisions.

Proposed module:

```text
src/vait/economics/
    metrics.py
```

Economics is evidence, not the authority for accepting an unsafe transformation.

### 2.6 Reporting layer

Responsibilities:

- generate structured JSON evidence;
- render a readable CLI summary;
- preserve failed examples and decision reasons.

Proposed module:

```text
src/vait/reporting/
    report.py
```

## 3. Data model

Minimum conceptual types:

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
```

All externally persisted or exchanged structures should use explicit versioned schemas.

## 4. Decision precedence

Recommended v0.1 order:

```text
1. Contract/schema validity
2. Execution validity
3. Hard invariants
4. Risk-specific hard thresholds
5. Exact-verification checks (when supported)
6. Bounded statistical checks
7. Cost/latency evidence
8. Final classification
```

This prevents a cheaper or faster candidate from masking a hard safety failure.

## 5. EXACT path

The exact path is intentionally narrow.

It can be used for supported deterministic fragments where the project can compare declared deterministic obligations, such as:

- output-schema equality;
- invariant preservation;
- deterministic output equality across an exhaustive or contract-defined finite domain;
- deterministic relational/rule checks implemented explicitly by the verifier.

It must not imply universal semantic equivalence for arbitrary probabilistic models.

## 6. BOUNDED path

The bounded path is the main AI-specific path.

For a fixed versioned evaluation set, it can evaluate:

- overall agreement;
- task metric delta;
- disagreement by risk class;
- unsafe disagreement count/rate;
- confidence interval or upper bound;
- latency delta;
- cost delta.

The contract defines acceptable thresholds before benchmark results are inspected.

## 7. REJECT path

`REJECT` is returned when any hard condition fails or evidence is insufficient.

Suggested reason-code classes:

```text
SCHEMA_MISMATCH
EXECUTION_FAILURE
HARD_INVARIANT_VIOLATION
RISK_THRESHOLD_EXCEEDED
QUALITY_THRESHOLD_EXCEEDED
INSUFFICIENT_EVIDENCE
UNSUPPORTED_TRANSFORMATION
```

## 8. Repository structure

```text
verified-ai-workflow-transformation/
├── src/
│   └── vait/
│       ├── contracts/
│       ├── runners/
│       ├── verification/
│       ├── decision/
│       ├── economics/
│       └── reporting/
├── benchmarks/
│   └── ap_exception/
├── tests/
│   ├── unit/
│   ├── property/
│   └── integration/
├── docs/
│   ├── PROJECT-SPEC.md
│   ├── ARCHITECTURE.md
│   ├── BENCHMARK.md
│   ├── LIMITATIONS.md
│   └── RESEARCH-BACKGROUND.md
├── examples/
├── pyproject.toml
├── README.md
└── LICENSE
```

## 9. Dependency direction

Core verification code must not depend on cloud/provider SDKs.

```text
contracts
    ↓
runners abstraction
    ↓
verification
    ↓
decision
    ↓
reporting

provider adapters -> runner abstraction
CLI/API -> public application/service layer
```

## 10. Deferred architecture

Do not add in v0.1 unless a measured requirement appears:

- distributed job queues;
- Redis;
- Kubernetes/OpenShift;
- React UI;
- Neo4j/graph database;
- multi-agent frameworks;
- automatic candidate synthesis;
- optimiser/search engine;
- human-capacity optimiser.
