# Project Specification — Verified AI Workflow Transformation

**Status:** Build-ready v0.1
**Language:** English only
**Project type:** AI engineering / verification R&D system
**Working title:** Verified AI Workflow Transformation (VAIT)

## 1. Project purpose

VAIT evaluates whether a supplied replacement for an existing AI workflow step can be accepted without violating declared behavioural, risk, policy, cost, or latency constraints.

The project does **not** attempt to discover or orchestrate an entire enterprise workflow. It operates on a bounded, already-specified decision step and compares a reference implementation with a candidate replacement.

The system classifies each proposed transformation as:

- **EXACT** — supported deterministic checks show that the candidate preserves the declared contract within the supported exact-verification scope.
- **BOUNDED** — exact equivalence is not available, but the candidate satisfies pre-declared statistical degradation and risk constraints on the evaluation set.
- **REJECT** — one or more hard constraints fail, the evidence is insufficient, or the candidate falls outside the supported verification scope.

## 2. Problem statement

Teams replacing expensive or probabilistic AI components with cheaper deterministic rules, retrieval pipelines, or smaller models need more than a cost comparison. A replacement can appear successful on average while failing on the highest-risk cases.

VAIT addresses one narrow problem:

> Given a reference implementation, a candidate replacement, a transformation contract, and an evaluation set, determine whether the candidate can be accepted under explicit behavioural and risk constraints, and produce evidence supporting the decision.

## 3. Target user

Primary user hypothesis:

- AI Platform Engineer
- ML Platform Engineer
- AI/ML Systems Engineer
- Enterprise AI Architect

The v0.1 project does not claim validated commercial demand. It is an engineering and evaluation system intended to demonstrate a technically defensible workflow-transformation verification capability.

## 4. v0.1 scope

The first release supports a **bounded, read-only decision pipeline**.

Primary transformation family:

```text
probabilistic/reference decision
        ->
deterministic Python/rule candidate
```

Optional second transformation family, only after the first vertical slice works:

```text
large/probabilistic model
        ->
smaller supplied model
```

The candidate implementation is supplied to VAIT. VAIT does not synthesize arbitrary replacements in v0.1.

## 5. Inputs

A verification run requires:

1. **Reference implementation**
   - callable or provider-backed implementation used as the comparison reference.
2. **Candidate implementation**
   - supplied replacement implementation.
3. **Transformation contract**
   - typed input/output expectations;
   - invariants;
   - risk strata;
   - allowed degradation thresholds;
   - hard rejection conditions;
   - optional cost and latency limits.
4. **Evaluation dataset**
   - fixed, versioned examples with labels/metadata required by the contract.
5. **Execution observations**
   - outputs;
   - latency;
   - error status;
   - optional provider usage/cost metadata.

## 6. Outputs

Each run produces:

- decision: `EXACT`, `BOUNDED`, or `REJECT`;
- decision reason codes;
- behavioural agreement metrics;
- risk-stratified disagreement metrics;
- hard-invariant results;
- confidence intervals or upper bounds where applicable;
- latency delta;
- cost delta where measurable;
- failure examples;
- machine-readable evidence report.

## 7. Core technical contribution

The v0.1 contribution is **not** general workflow optimisation, model routing, agent orchestration, or candidate synthesis.

The project contribution is the transformation-verification layer:

1. a typed transformation contract;
2. deterministic/schema/invariant checks;
3. statistical bounded-degradation checks;
4. risk-stratified evaluation;
5. decision rules that classify a supplied rewrite as `EXACT`, `BOUNDED`, or `REJECT`;
6. an evidence report explaining the decision.

## 8. Decision semantics

### EXACT

Use only when the supported deterministic checks can establish the declared equivalence obligations for the tested transformation class.

`EXACT` must never be used to claim universal semantic equivalence for arbitrary LLM behaviour.

### BOUNDED

Use when exact equivalence is not available but all declared statistical and risk constraints pass.

Example contract conditions may include:

- maximum overall disagreement;
- maximum disagreement for high-risk cases;
- minimum task metric;
- no violation of specified hard invariants;
- confidence level for the reported bound.

### REJECT

Use when:

- a hard invariant fails;
- a risk threshold fails;
- a statistical threshold fails;
- schema/type compatibility fails;
- execution evidence is insufficient;
- the candidate falls outside the supported verification scope.

## 9. Benchmark domain

Accounts Payable invoice-exception decisions are used as the **first benchmark domain only**.

The benchmark is useful because it naturally includes:

- structured financial fields;
- deterministic matching/rules;
- policy constraints;
- risk classes;
- clear rejection conditions.

VAIT is not an AP automation product.

## 10. Success criteria for v0.1

A v0.1 release is successful when all of the following are true:

1. A versioned transformation contract can be parsed and validated.
2. Reference and candidate implementations can be executed on the same evaluation set.
3. The evaluator computes overall and risk-stratified behavioural differences.
4. Hard invariants can force `REJECT` even when average agreement is high.
5. Statistical uncertainty is reported rather than hidden behind a single accuracy number.
6. The decision engine produces reproducible `EXACT`, `BOUNDED`, or `REJECT` outcomes.
7. An end-to-end CLI run produces a machine-readable evidence report.
8. Unit, property, and integration tests pass in CI.
9. The repository clearly states limitations and does not claim universal formal verification or production financial safety.

## 11. Non-goals for v0.1

VAIT v0.1 will not:

- compile or optimise an entire enterprise workflow;
- discover candidate architectures automatically;
- synthesize arbitrary production code;
- own runtime orchestration, retries, queues, or durable state;
- optimise human workforce capacity;
- provide compliance certification;
- prove arbitrary LLM semantic equivalence;
- claim real-world FTE savings from synthetic data;
- provide a production AP automation product;
- require React, Kubernetes, Neo4j, or a multi-agent framework.

## 12. Technology boundary

Core implementation:

- Python 3.12+
- Pydantic v2
- pytest
- Hypothesis
- NumPy
- pandas
- SciPy
- PyYAML
- Typer (CLI)

Optional after the core vertical slice works:

- FastAPI
- Docker
- GitHub Actions
- OpenTelemetry
- IBM watsonx.ai adapter

The core must remain provider-neutral. IBM may be used as a reference AI provider, not as the project abstraction.

## 13. Release target

The one-week target is a credible `v0.1.0` engineering release containing:

- working verification engine;
- one AP benchmark transformation;
- deterministic and statistical checks;
- evidence report;
- tests and CI;
- reproducible CLI command;
- architecture and benchmark documentation;
- explicit limitations.
