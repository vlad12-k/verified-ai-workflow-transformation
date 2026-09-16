# Research Background

## 1. Why the project was narrowed

The project began with a broader idea: automatically compile an enterprise AI workflow into a cheaper combination of frontier models, smaller models, retrieval, deterministic code, rules, and human review.

Prior-art analysis showed that this broad formulation overlaps substantially with existing research and commercial systems in workflow optimisation, AI workload planning, routing, deterministic compilation, agent orchestration, and process optimisation.

The broad compiler thesis was therefore rejected.

## 2. Important prior-art categories identified

The research review identified significant overlap with systems and research including:

- Murakkab — workflow/resource/model optimisation for agentic cloud workloads;
- AWO — trace-derived deterministic meta-tools;
- Compiled AI — compile-time generation and validation of deterministic artifacts;
- DSPy — metric-driven compilation/optimisation of LM programs;
- Palimpzest — declarative optimisation of AI workloads with alternative physical implementations;
- FlowCompile — compile-time optimisation of structured LLM workflows;
- DocETL/MOAR — pipeline rewriting and multi-objective optimisation;
- LOTUS — semantic operators and execution optimisation with statistical accuracy guarantees;
- Semantic Integrity Constraints — declarative reliability constraints for AI-augmented data processing;
- commercial orchestration/process systems including UiPath, Microsoft, IBM, and Celonis.

The research conclusion was that multi-paradigm workflow optimisation is not, by itself, a defensible novelty claim.

## 3. Surviving engineering question

A narrower question remained useful:

> Can a system evaluate a supplied cross-implementation transformation under explicit typed, behavioural, risk, and statistical contracts, and classify the transformation as exact, acceptably bounded, or inadmissible?

The first build therefore focuses on the verification layer rather than workflow search or orchestration.

## 4. v0.1 research boundary

The v0.1 project intentionally restricts itself to:

- bounded, read-only decision pipelines;
- supplied candidate implementations;
- explicit risk/invariant contracts;
- deterministic checks where exact verification is supported;
- statistical bounded-degradation checks for probabilistic behaviour;
- refusal (`REJECT`) when constraints fail or evidence is insufficient.

## 5. Benchmark role

Accounts Payable invoice exceptions are retained only as a first benchmark because the domain provides clear deterministic rules, policy constraints, risk classes, and rejection conditions.

The project is not positioned as an AP automation product.

## 6. Research claim discipline

The project should not claim general novelty solely from combining existing components.

The engineering value of v0.1 is demonstrated through:

- an explicit transformation-contract abstraction;
- reproducible exact/bounded/reject decision logic;
- risk-stratified evaluation;
- transparent statistical evidence;
- reproducible benchmark results;
- clear failure and rejection behaviour.
