# Limitations and Claim Boundaries

## 1. Purpose

This document defines what VAIT v0.1 does **not** prove. It is part of the engineering contract of the project, not an optional disclaimer.

## 2. No universal semantic equivalence

VAIT v0.1 does not prove semantic equivalence for arbitrary LLMs, neural models, or open-ended workflows.

`EXACT` is restricted to explicitly supported deterministic obligations and transformation classes.

## 3. Statistical evidence is conditional

`BOUNDED` decisions are conditional on:

- the declared evaluation distribution;
- the versioned dataset;
- the chosen metrics;
- the stated confidence level;
- the risk strata encoded in the contract.

A bounded result is not a guarantee about all future inputs.

## 4. Synthetic benchmark limitations

The initial AP benchmark may use controlled or synthetic cases.

Therefore v0.1 must not claim:

- real operational ROI;
- real fraud reduction;
- real human-productivity savings;
- production-grade financial safety;
- compliance certification.

## 5. No automatic candidate synthesis

VAIT v0.1 verifies supplied candidates. It does not automatically generate or discover the optimal replacement implementation.

## 6. No enterprise workflow compilation

The project is not a general AI workload compiler, workflow optimiser, model router, or orchestration platform.

It operates on bounded transformations inside an already-specified workflow step.

## 7. No human-capacity optimisation

Human review may be represented as a policy outcome in later work, but v0.1 does not model workforce queues, staffing, shift schedules, or organisational capacity.

## 8. No privacy proof

VAIT may report declared privacy-related metadata in future versions, but v0.1 does not claim formal privacy verification unless a specific threat model and enforceable constraints are implemented.

## 9. No compliance certification

Passing a VAIT contract does not imply compliance with GDPR, financial regulation, internal audit requirements, or any other external standard.

## 10. Cost and latency evidence is conditional

Latency and cost measurements depend on implementation, provider, model
version, hardware, geography, pricing, load, and measurement time. Reports
must preserve relevant provenance when available.

Declared or controlled cost assumptions are evidence-pipeline inputs, not
claims about current provider pricing, production operating cost, or ROI.

A frozen benchmark gold-label runner is not a timed reference implementation.
Its placeholder latency must not be used to calculate or claim comparative
performance. Reference-versus-candidate latency deltas require genuinely
measured and comparable execution evidence on both sides.

## 11. Benchmark claims must remain scoped

Public claims should use wording such as:

> On benchmark X and contract Y, candidate Z satisfied the declared bounded-degradation and risk constraints.

Avoid wording such as:

> Candidate Z is safe in production.
