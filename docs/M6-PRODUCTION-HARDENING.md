# M6 — Production Hardening

**Status:** Active  
**Branch:** `feat/m6-production-hardening`

## Goal

Harden the production-shaped VAIT platform established in M5 without
weakening the verification-first architecture.

M6 is not a rewrite of the VAIT core.

Its role is to attack, validate, stress, recover, observe, and secure the
existing platform boundaries before external validation and the v1 release
phase.

The core invariant remains:

    VERIFY FIRST -> OPTIMISE SECOND

A faster, cheaper, smaller, externally hosted, or operationally convenient
candidate must never bypass verification, admissibility, or hard constraints.

## Architectural law

Dependency direction remains:

    external providers / HTTP / PostgreSQL / workers / telemetry
                               |
                               v
                        vait.platform
                               |
                               v
                           VAIT core

Core verification and optimisation packages must not depend upward on:

- FastAPI;
- PostgreSQL;
- IBM watsonx;
- cloud SDKs;
- deployment systems;
- observability backends.

IBM watsonx remains a reference provider integration, not the project
abstraction.

---

## M6-0 — Rebaseline and M5 residual closure plan

### Purpose

M6-0 establishes the authoritative M6 roadmap and explicitly records the
operational work carried forward from M5.

It must:

- mark M5 as completed and merged;
- preserve M5 implementation history;
- avoid pretending deferred work was completed inside M5;
- map all remaining M5 operational gaps to explicit M6 slices;
- introduce no application behaviour changes.

### M5 residual mapping

    M5 documentation state
        -> M6-0

    deployable worker runtime
        -> M6-D

    OpenTelemetry metrics
        -> M6-H

    full HTTP-to-evidence platform E2E
        -> M6-K

### Carried-forward M5 gaps

1. **OpenTelemetry metrics**
   - M5 established structured logging and OpenTelemetry tracing.
   - Metrics and SLO-oriented instrumentation are completed in M6-H.

2. **Deployable worker process**
   - M5 established the durable job model, state machine, PostgreSQL-backed
     claiming, leases, retries, heartbeat semantics, cancellation semantics,
     and worker service abstraction.
   - A separately deployable worker runtime and Compose worker service are
     completed in M6-D.

3. **Full platform workflow E2E**
   - M5 CI proves clean container bootstrap, migration completion, non-root
     runtime, liveness, readiness, migration-head agreement, and cleanup.
   - M6-K completes the full path:

         HTTP request
             -> experiment
             -> durable job
             -> worker
             -> VAIT verification / evaluation
             -> evidence persistence
             -> result retrieval

4. **Documentation and release-state alignment**
   - M5 implementation and CI evidence are complete.
   - M6-0 aligns documentation with the merged state and records deferred
     operational work explicitly.

These items do not weaken or redefine M5 semantics.

---

## M6-A — Threat and trust model

### Goal

Create an explicit security model for the platform before adding production
authentication, external-provider credentials, or live cloud execution.

### Deliverables

- formal asset inventory;
- trust-boundary map;
- attack-surface inventory;
- protected-data classification;
- API trust assumptions;
- PostgreSQL trust assumptions;
- ArtifactStore trust assumptions;
- worker trust assumptions;
- job-queue trust assumptions;
- external-provider trust assumptions;
- provider-egress threat model;
- explicit security non-goals.

### Threat classes

The threat model must include, where applicable:

- credential compromise;
- privilege escalation;
- malformed or hostile API input;
- SSRF-style outbound abuse;
- replay;
- duplicate execution;
- provider-response manipulation;
- telemetry leakage;
- artifact tampering;
- database compromise;
- denial of service;
- dependency compromise;
- container supply-chain risk.

---

## M6-B — Authentication, secrets, PII, and provider security

### Goal

Establish the security boundary required before any live external provider is
allowed.

### Deliverables

- authentication boundary;
- authorisation policy;
- RBAC only where justified by actual platform roles;
- secure secret-loading policy;
- protected-environment configuration;
- telemetry redaction;
- error-message redaction;
- PII classification;
- PII handling policy;
- provider-egress policy;
- SSRF controls;
- hostile-input tests;
- explicit IBM credential boundary.

Credentials must never appear in:

- source control;
- logs;
- traces;
- metrics;
- API responses;
- persisted evidence payloads unless explicitly required and safely
  transformed.

IBM credentials must not be required for:

- ordinary development;
- unit tests;
- integration tests;
- pull-request CI;
- push CI.

---

## M6-C — Rate, quota, and resource-budget controls

### Goal

Prevent uncontrolled platform or provider usage.

### Deliverables

- API request limits;
- job submission limits;
- worker execution limits where justified;
- provider request budgets;
- provider token budgets;
- retry budgets;
- live-test request ceilings;
- optional monetary or spend ceilings;
- deterministic handling when a budget is exceeded.

### Safety rule

A provider budget violation must block the external call before execution.

It must not merely report excessive usage after the request has already been
sent.

---

## M6-D — Worker operational hardening

### Goal

Turn the M5 worker abstraction into a separately deployable and recoverable
runtime.

### Deliverables

- worker runtime entry point;
- separately deployable worker process;
- Docker worker runtime;
- Docker Compose worker service;
- safe startup;
- graceful shutdown;
- concurrent worker claiming;
- heartbeat lifecycle;
- lease expiration;
- expired-lease recovery;
- bounded retries;
- idempotency behaviour;
- duplicate-delivery tests;
- poison-job handling;
- crash/restart tests;
- PostgreSQL interruption tests;
- recovery after database availability returns.

### Persistence rule

PostgreSQL remains the durable source of truth.

Redis remains unnecessary unless a new measured requirement overturns the
existing ADR decision.

---

## M6-E — Persistence recovery and migration safety

### Goal

Establish operational recovery evidence for PostgreSQL and schema changes.

### Deliverables

- PostgreSQL backup procedure;
- tested restore procedure;
- migration rollback strategy;
- downgrade validation where supported;
- failure-safe migration guidance;
- recovery-point evidence;
- recovery-time evidence where meaningfully measurable;
- operator recovery runbook.

### Claim boundary

VAIT must not claim zero-downtime migration support without explicit measured
evidence.

---

## M6-F — Supply-chain and CI security

### Goal

Add automated security evidence to the build and delivery pipeline.

### Deliverables

- dependency vulnerability scanning;
- static analysis / SAST;
- container image scanning;
- Software Bill of Materials generation;
- least-privilege CI permissions review;
- workflow dependency review;
- CI security gates;
- documented false-positive handling;
- documented upstream vulnerability handling.

Security scanners are evidence sources.

They are not proof of complete application security.

---

## M6-G — Load, soak, and failure injection

### Goal

Test platform behaviour under controlled stress and failure.

### Deliverables

- concurrent API load;
- concurrent job execution;
- queue saturation tests;
- worker termination during execution;
- PostgreSQL interruption;
- service restart;
- recovery validation;
- timeout behaviour;
- bounded resource observations;
- soak-test evidence;
- failure-injection evidence.

### Claim boundary

Load-test results are environment-specific.

They must not be presented as universal production-capacity guarantees.

---

## M6-H — Observability maturity

### Goal

Complete the observability model started in M5.

The required observability triad is:

    structured logging
    +
    OpenTelemetry tracing
    +
    OpenTelemetry metrics

### Deliverables

- OpenTelemetry metrics foundation;
- HTTP request metrics;
- API error metrics;
- job metrics;
- worker metrics;
- persistence health metrics;
- provider-call metrics;
- retry metrics;
- timeout metrics;
- request-to-job correlation;
- job-to-worker correlation;
- provider correlation;
- experiment/evidence correlation;
- secret redaction;
- sensitive-payload redaction;
- SLO-oriented measurements;
- initial SLO definitions;
- machine-readable operational evidence;
- dashboards where justified;
- incident runbook;
- recovery runbook.

### SLO rule

An SLO must not be described as satisfied without measured evidence.

---

## M6-I — Bounded workflow composition and risk budget

### Goal

Close the deferred workflow-level verification gap without turning VAIT into
an unrestricted workflow compiler or autonomous agent platform.

### Deliverables

- bounded workflow-node identity;
- explicit composition contracts;
- node-level verification-state propagation;
- workflow-level hard-failure precedence;
- bounded workflow risk budget;
- unsupported-composition handling;
- composition evidence;
- workflow-level admissibility tests.

### Required invariants

A node-level `REJECT` must not become workflow-admissible.

Optimisation must not override workflow risk constraints.

The following semantic states remain distinct:

- `EXACT`;
- `BOUNDED`;
- `REJECT`;
- `NOT_APPLICABLE`.

Automatic transformation execution remains prohibited.

---

## M6-J — IBM watsonx live integration

### Goal

Connect the existing provider-neutral watsonx boundary to a real IBM service
without coupling the VAIT core to IBM-specific SDK or runtime semantics.

M6-J is the first milestone slice that may require access to the IBM account.

No IBM API key, billable request, live cloud configuration, or IBM deployment
is required before M6-J.

### Existing foundation

M4 already established:

- `WatsonxProvider`;
- `WatsonxTransport`;
- provider-neutral request models;
- provider-neutral response models;
- normalised provider errors;
- token-usage evidence;
- model identity;
- provider identity;
- timing evidence;
- deterministic unit tests using a fake transport.

M6-J adds the concrete live transport and external-service evidence.

### M6-J sequence

    M6-A -> M6-I local hardening
            |
            v
    M6-J0 IBM account inventory
            |
            v
    M6-J1 credentials and configuration
            |
            v
    M6-J2 concrete WatsonxTransport
            |
            v
    M6-J3 mocked failure and security tests
            |
            v
    M6-J4 live canary: 1-3 requests
            |
            v
    M6-J5 small frozen provider benchmark
            |
            v
    M6-J6 comparative evidence analysis

### M6-J0 — IBM account inventory

Only at this point do we enter the IBM account and inspect the actual live
environment.

We record:

- available watsonx service;
- account / plan state;
- region;
- endpoint;
- project identity;
- project ID or space ID;
- available model identities;
- request limits;
- token limits;
- applicable pricing;
- deployment requirements if any.

No credential is committed to the repository.

### M6-J1 — Credentials and configuration

Create the minimum configuration required for controlled live integration.

Requirements:

- live IBM mode disabled by default;
- IBM credentials stored only through approved secret mechanisms;
- project / space identity pinned;
- model identity pinned;
- endpoint pinned;
- live-request budget configured;
- token budget configured;
- timeout configured.

### M6-J2 — Concrete WatsonxTransport

Implement a concrete IBM-specific transport behind the existing
`WatsonxTransport` protocol.

IBM-specific implementation details must remain outside the provider-neutral
core.

The public VAIT boundary remains `WatsonxProvider`.

### M6-J3 — Mocked failure and security tests

Before any live request, test:

- authentication failure;
- authorisation failure;
- timeout;
- rate limit;
- retryable server error;
- non-retryable server error;
- malformed provider response;
- empty provider response;
- model mismatch;
- project/deployment mismatch;
- request budget exhaustion;
- token budget exhaustion;
- secret redaction;
- telemetry redaction.

### M6-J4 — Live canary

The first real IBM execution is intentionally small:

    1-3 live requests

The purpose is integration validation, not model benchmarking.

### M6-J5 — Frozen provider benchmark

After the canary passes:

    approximately 5 integration cases
        ->
    small frozen provider benchmark

The frozen provider benchmark must not be used for tuning after its outputs
are observed.

### M6-J6 — Comparative evidence analysis

IBM evidence must pass through the same VAIT decision architecture:

    IBM provider execution
        ->
    verification
        ->
    EXACT / BOUNDED / REJECT
        ->
    admissibility
        ->
    performance / resource / economic evidence
        ->
    comparable analysis

IBM must not receive separate acceptance semantics.

### IBM live gate

| Gate | Requirement |
| --- | --- |
| IBM-01 | Live IBM execution is disabled by default |
| IBM-02 | Missing credentials fail closed before network access |
| IBM-03 | API keys are absent from logs, traces, metrics, and errors |
| IBM-04 | Provider/model/project or deployment identity is pinned |
| IBM-05 | Network timeout is bounded |
| IBM-06 | Authentication failures are explicit and non-successful |
| IBM-07 | Rate-limit handling is controlled and bounded |
| IBM-08 | Retryable server failures use a bounded retry policy |
| IBM-09 | Malformed provider responses fail safely |
| IBM-10 | Token budget is enforced before external execution |
| IBM-11 | Live-request budget is enforced before external execution |
| IBM-12 | 1-3 real canary calls normalise successfully |
| IBM-13 | Provider provenance is persisted |
| IBM-14 | Frozen benchmark runs without tuning on its outputs |
| IBM-15 | Comparison uses only genuinely comparable evidence |
| IBM-16 | No credential or sensitive artifact remains after cleanup |

### CI rule

Live IBM tests must not run in ordinary pull-request or push CI.

Ordinary CI may run:

- provider-contract tests;
- fake-transport tests;
- security tests;
- malformed-response tests;
- retry tests;
- budget tests;
- redaction tests.

Live execution must be explicitly opt-in through a protected manual workflow
or equivalent mechanism using separately managed secrets.

---

## M6-K — Full platform E2E and M6 release evidence

### Goal

Prove the complete production-shaped execution path across platform
boundaries.

### Required path

    HTTP request
        ->
    experiment
        ->
    durable job
        ->
    worker
        ->
    VAIT verification / evaluation
        ->
    evidence persistence
        ->
    result retrieval

### Deliverables

- containerised API;
- PostgreSQL;
- deployable worker;
- migration completion;
- experiment creation;
- durable job creation;
- worker claim;
- worker execution;
- request/job/experiment correlation;
- verification execution;
- evidence persistence;
- result retrieval;
- restart/recovery evidence;
- observability evidence;
- security gates;
- M6 release-evidence manifest;
- final full regression;
- CI confirmation on the final source revision.

---

## Deferred to M7

The following remain M7 concerns:

- external public or donor dataset validation;
- frozen external hold-out evaluation;
- final comparative research study;
- IBM reference deployment;
- cloud reference-environment documentation;
- final technical report;
- final reproduction package;
- release-candidate hardening;
- versioned v1 release;
- v1 release tag.

M6 may prepare interfaces and evidence required by M7.

It must not claim M7 external validation prematurely.

---

## M6 completion criteria

M6 is complete only when:

1. M6-0 through M6-K have explicit completion evidence.
2. Security and trust boundaries are documented and tested.
3. Authentication and authorisation boundaries are explicit.
4. Secret handling fails closed.
5. Provider egress is controlled.
6. Rate, quota, retry, and provider budgets are bounded.
7. A separately deployable worker runtime exists and is tested.
8. Worker crash and lease recovery have evidence.
9. PostgreSQL backup and restore have execution evidence.
10. Migration recovery has execution evidence.
11. Supply-chain security gates run in CI.
12. Load and soak tests have reproducible evidence.
13. Selected failure-injection scenarios have recovery evidence.
14. Structured logging remains operational.
15. OpenTelemetry tracing remains operational.
16. OpenTelemetry metrics are implemented.
17. Workflow composition preserves verification precedence.
18. IBM live integration passes its explicit manual live gate.
19. Full HTTP-to-evidence E2E passes.
20. Full repository quality gates pass.
21. M6 release evidence is bound to a clean source revision.
22. No M6 capability weakens `VERIFY FIRST -> OPTIMISE SECOND`.

---

## M6 non-goals

M6 does not claim:

- universal production security;
- penetration-test certification;
- compliance certification;
- universal cloud portability;
- unrestricted distributed orchestration;
- autonomous agent orchestration;
- automatic candidate synthesis;
- automatic workflow transformation;
- Kubernetes necessity;
- Redis necessity;
- universal performance superiority;
- external research validity.

Those claims require separate evidence or remain outside project scope.
