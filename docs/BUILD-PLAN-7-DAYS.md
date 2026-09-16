# Seven-Day Build Plan — VAIT v0.1

## Goal

Ship a credible, reproducible engineering release in one week that verifies one real transformation family end-to-end.

The priority is a working core, not feature count.

## Day 1 — Freeze contracts and bootstrap repository

Deliverables:

- finalise `PROJECT-SPEC.md`;
- finalise `ARCHITECTURE.md`;
- finalise `BENCHMARK.md`;
- finalise `LIMITATIONS.md`;
- create repository;
- create `pyproject.toml`;
- create Python package skeleton;
- configure pytest;
- configure Ruff/type checking if used;
- configure GitHub Actions;
- add first contract-model unit tests.

Definition of done:

```text
package installs
pytest runs
CI runs
TransformationDecision enum exists
TransformationContract parses a valid example
invalid contract is rejected
```

## Day 2 — Contract and deterministic verification core

Implement:

- `TransformationContract`;
- risk strata;
- invariant definitions;
- reference/candidate runner abstraction;
- Python runner;
- deterministic schema checks;
- deterministic invariant checks;
- decision reason codes.

Tests:

- unit tests;
- Hypothesis property tests for contract validation;
- deterministic exact/reject examples.

## Day 3 — Statistical bounded-degradation engine

Implement:

- behavioural agreement;
- disagreement table;
- risk-stratified metrics;
- confidence interval/upper-bound utilities;
- bounded-threshold evaluation;
- `BOUNDED` decision path.

Definition of done:

A high-average-agreement candidate with a high-risk failure must return `REJECT`.

## Day 4 — AP benchmark

Build controlled/versioned benchmark cases covering:

- normal invoice;
- price variance;
- quantity mismatch;
- missing PO;
- missing goods receipt;
- duplicate invoice;
- blocked vendor;
- policy threshold boundary;
- unsupported/ambiguous case.

Create:

- reference behaviour;
- deterministic candidate;
- deliberately unsafe candidate;
- benchmark contract.

Run end-to-end integration tests.

## Day 5 — Probabilistic/provider reference

Add a provider-neutral runner interface.

If IBM watsonx.ai credentials/configuration are available, add one watsonx adapter as the first live reference provider.

If provider access blocks progress, use frozen versioned reference outputs and keep live-provider support optional.

Do not block v0.1 on cloud access.

## Day 6 — CLI, packaging, Docker, observability basics

Implement:

- Typer CLI;
- JSON evidence report;
- readable terminal summary;
- reproducible benchmark command;
- Dockerfile;
- structured logging;
- minimal OpenTelemetry instrumentation only if it does not delay core stability.

FastAPI is optional and should be added only if the core is already complete.

## Day 7 — Release and employer-facing evidence

Complete:

- clean `README.md`;
- architecture diagram;
- benchmark results table;
- limitations;
- reproducibility commands;
- tagged `v0.1.0` release;
- one concise project description for CV/LinkedIn;
- screenshots/output evidence if useful.

Do not delay release for React, Kubernetes, or a polished dashboard.

## Week-one stop conditions

Do not add scope if any of these threaten the release date:

- candidate synthesis;
- SLM fine-tuning;
- multiple cloud providers;
- workflow optimisation/search;
- agent frameworks;
- UI;
- Kubernetes;
- enterprise database layer.

## v0.1 release statement

A defensible release claim is:

> VAIT v0.1 evaluates supplied AI workflow replacements under explicit behavioural and risk contracts and classifies them as EXACT, BOUNDED, or REJECT. The first benchmark demonstrates probabilistic/reference-to-deterministic transformation verification on controlled AP exception cases.
