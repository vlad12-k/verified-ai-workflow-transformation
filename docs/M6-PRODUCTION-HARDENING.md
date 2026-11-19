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

### Provider-dispatch boundary (M6-C3.5)

Platform-controlled provider work must enter through
`vait.platform.provider_execution.BudgetedJobProviderInvocation` with a
`RUNNING` job record returned by a committed durable claim. Before planning,
the adapter re-reads that ID and requires a live `RUNNING` row whose `claimed_by`
matches the adapter's trusted worker ID and whose lease has not expired. The
resolver receives this database-backed record, not caller-supplied job fields.
The invocation identity is the persisted `job_id`; request IDs and caller
idempotency keys are not execution authority. Its injected
`ProviderBudgetPlanResolver` must obtain the request policy and estimate from
trusted server-side context, never from caller-controlled payload or inference
metadata.

The provider wrapper rechecks the current durable owner, status, attempt, and
lease at the `generate` boundary after plan resolution and immediately before
delegating to the underlying provider. If lease recovery wins during planning,
provider I/O is blocked and the already-dispatched reservation is settled
conservatively with unknown consumption. This fence is point-in-time rather
than an atomic lease lock: a process paused immediately after the check could
resume after expiry and start provider I/O. M6-D must define a durable dispatch
fence or equivalent stale-worker protection, along with bounded cancellation
and a shared authoritative or sufficiently synchronized UTC time source for
claim, recovery, and lease validation. The lease can also expire during an
outbound provider request; those worker lifecycle controls remain M6-D
responsibilities.

The adapter delegates to the durable budget dispatcher, which commits
admission and a strict dispatch claim before provider I/O and records unknown
consumption after an indeterminate provider failure. It supports one
`generate` call on attempt one only. Retry attempts and native batch dispatch
are unsupported and must fail closed. Any deterministic cache decorator must
sit behind this boundary so cache hits, misses, and bypasses all require
admission. Research benchmarks and provider-neutral RAG/cache utilities are
not platform-controlled execution paths; they may still incur external usage
when a caller supplies a remote provider and must not be mistaken for guarded
platform work.

The current `WorkerService` has only a generic executor protocol; no concrete
provider job executor, trusted policy resolver, or provider composition root
is wired into the platform API. M6-D must connect those components and verify
lease recovery and retry behavior before the adapter is considered reachable
for operational platform execution. Until then, the adapter and its
PostgreSQL-backed tests are a ready-to-wire boundary, not evidence that a live
platform provider path is active or production-ready. Provider integrations
with opaque internal retries remain unsupported until their actual attempt
semantics are verified and represented in budget accounting.

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

### M6-D3B1 — Trusted local worker composition

**Checkpoint: composition boundary implemented; executable worker remains
NO-GO.**

`TrustedLocalExecutorRegistry` is a frozen, slotted, routing-only allowlist.
It defensively copies an explicit operation mapping, exposes only immutable
operation values, and accepts no executor instances or callable references.
Its code-owned registration contains only `verification`; provider-inference is
excluded. Routing reads `JobRecord.operation` only and never interprets payload
fields as implementation names, imports, or policy.

Executable local worker startup remains disabled at both entry points. The CLI
`--run` returns refusal before runtime creation. Programmatic
`run_configured_worker()` validates the routing registry and then raises before
loading worker settings, creating a database runtime, or starting a process.
Compose continues to expose configuration preflight, not an executable worker
service.

This does not change the pre-existing M5 `WorkerService.run_once()` contract.
Direct callers can still supply a generic `JobExecutor`; for non-provider
operations, a normal return is followed by `mark_succeeded()` without a generic
verification-evidence check (`src/vait/platform/jobs/worker.py`). That direct
path remains unsupported for verification work and is not made safe by the
D3B1 bootstrap refusal. Do not wire it into production verification startup;
D2F3/D3B2 must supply a trusted implementation identity and durable atomic
verification-evidence completion contract before any such path is enabled.

The deterministic verification primitive accepts typed
`TransformationContract`, `VerificationCase`, and `ImplementationRunner`
inputs, but the platform does not yet persist or resolve a trusted executable
work specification. `JobRecord.payload` is generic JSON; experiment persistence
stores contract identifiers/version and dataset fingerprint/partition, but no
typed contract contents, finite case snapshot, or trusted implementation
reference is available through a platform repository. `PythonImplementationRunner`
requires a supplied callable, and the platform has no code-owned implementation
registry bound to an experiment revision. The generic evidence envelope does
not define a verification-result schema or an atomic evidence-and-completion
writer for local verification under the active job claim.

No job schema, dummy success executor, dynamic import, `eval`, or `exec` path
was introduced. Before D3B2 enables an executable Docker worker, define and
persist the trusted work reference, resolve only allowlisted implementation
code, and add a verification evidence contract whose persistence and job
completion preserve transactional ownership and lease checks.

#### D3B1R conditional acceptance — routing-only scope

The owner-approved conditional acceptance applies only to this non-executable,
routing-only security checkpoint. It does not enable worker execution or
override the existing D2F3 security dispositions.

The verified full suite reported 1,026 passed, with no failures, errors, or
skips. Combined coverage was 90.1335%, meeting the existing 85% GitHub CI
threshold. Independent project-wide branch coverage was 80.9735%, compared
with the clean pre-D3B1 baseline of 80.8807%. This is not a regression, but it
remains below the separate 85% quality objective, which remains OPEN. Branch
coverage on changed/new production code was 93.75%.

The local Docker build remains BLOCKED because the selected `desktop-linux`
daemon socket was unavailable; this is not a passing build. Final checkpoint
acceptance remains conditional on both `Quality gates` and `Container E2E`
succeeding for the new commit. Do not report final acceptance before both
exact-head GitHub Actions jobs succeed.


### M6-D2F3C2 — Confirmed post-validation dispatch race

**Security disposition: OPEN / NO-GO for strict no-late-I/O guarantees.**

Evidence: `test_pause_after_final_claim_check_exposes_late_provider_io`
in `tests/integration/test_platform_postgresql_job_lifecycle.py`.

The deterministic PostgreSQL test demonstrated that a worker can:

1. Commit one durable budget dispatch.
2. Pass the final database-backed lease check.
3. Pause before calling the underlying provider.
4. Lose its job lease.
5. Resume and invoke the fake provider after lease expiry.

The test passed because it successfully reproduced the unsafe timing
window. It must not be interpreted as proof that the race is fixed.

Existing protections:

- One durable budget dispatch is claimed per execution identity.
- Provider-inference retry attempts are rejected at the dispatch boundary.
- Invalid ownership before dispatch blocks provider I/O.
- Known pre-dispatch rejection releases reserved budget capacity.
- Unknown post-dispatch consumption is not refunded as known zero.

Remaining limitations:

- PostgreSQL ownership checks cannot atomically control external HTTP I/O.
- Provider request IDs are not proof of remote idempotency.
- A receiver-enforced fencing mechanism has not been demonstrated.
- Provider-specific implicit retry behaviour requires verification.

Required before claiming strict prevention of late external requests:

- Define the desired delivery guarantee and failure-state contract.
- Evaluate provider-supported idempotency and fencing capabilities.
- Establish receiver-side enforcement where strict fencing is required.
- Test paused workers, crashes, retry ambiguity and stale ownership.
- Record an explicit residual-risk decision for unsupported providers.

Do not enable an unrestricted live provider worker on the assumption
that successful CI proves this security guarantee.

### M6-D2F3C6A — Provider delivery guarantees and risk decision

**Status: OPEN / NO-GO for strict no-late-external-I/O claims.**

This acceptance contract applies to platform-controlled inference through
`BudgetedJobProviderInvocation`. It does not establish guarantees for
research benchmarks, direct provider calls, or third-party services.

#### Verified guarantees

- A durable budget dispatch is claimed once per execution identity.
- Unsupported retry attempts are rejected at the platform boundary.
- Known invalid job claims are rejected before provider dispatch.
- Built-in adapter identity is checked against dispatch policy.
- Remote providers require the audited built-in HTTPS transport configuration.
- Automatic inference POST redirects are denied by `StdlibJsonTransport`.
- Controlled timeout tests show no automatic POST replay.
- Unknown post-dispatch consumption is not recorded as known zero.

These are bounded, tested properties. They do not constitute a
remote exactly-once execution guarantee.

#### Dispatch mode contract

| Mode | Current disposition |
| --- | --- |
| `LOCAL_ONLY` | Allows a recognised offline local adapter; rejects remote adapter identity |
| `REMOTE_RISK_ACCEPTED` | Requires recognised remote adapter, approved transport configuration and explicit residual-risk acknowledgement |
| `STRICT_NO_LATE_IO` | Rejected until receiver-side fencing is independently verified |

The flags `residual_risk_accepted` and
`internal_retries_verified_disabled` express trusted configuration intent.
They are not, on their own, independent proof of an external provider's
retry behaviour, idempotency or fencing capabilities.

The current trusted remote transport gate rejects the unaudited Watsonx
transport. No exception is authorised by this document.

#### Confirmed open risks

1. A worker can lose its lease after the last database validation and
   still invoke the provider. The PostgreSQL TOCTOU test reproduces this.
2. PostgreSQL transactions cannot atomically fence an unrelated HTTP
   receiver from accepting an already-issued or late request.
3. A request identifier does not prove remote idempotency.
4. A timeout after POST delivery leaves remote execution and consumption
   uncertain even when the local client does not retry.
5. Provider-side behaviour and end-to-end production composition require
   independent evidence before operational approval.

#### Required receiver-side evidence for strict guarantees

- Define a durable, monotonic fencing token and its ownership lifecycle.
- Demonstrate that the receiving service rejects stale fencing tokens.
- Verify enforcement atomically with the receiver's side effect.
- Test lease expiration between validation and network transmission.
- Test response loss, worker crash, recovery and duplicate delivery.
- Verify the external provider's documented idempotency semantics.
- Confirm that retry behaviour is bounded and represented in budget evidence.

Idempotency by itself may suppress duplicate processing but does not prove
that a stale lease holder cannot execute a first, late request.

#### Security decision

**NO-GO** for claiming strict no-late-I/O or exactly-once remote effects.

A separately scoped remote-risk-accepted mode is not equivalent to strict
fencing. Its operational use requires a documented risk owner, verified
transport semantics, provider-specific evidence and an explicit deployment
decision.

Do not merge the M6 PR or enable unrestricted remote worker execution
based solely on green CI.

#### Evidence

- `tests/integration/test_platform_postgresql_job_lifecycle.py`
  - `test_pause_after_final_claim_check_exposes_late_provider_io`
- `tests/integration/test_platform_postgresql_worker.py`
  - `test_provider_transport_error_is_terminal_without_evidence`
- `tests/unit/test_stdlib_http_egress.py`
  - `test_post_received_before_response_timeout_is_not_replayed`
  - `test_307_308_do_not_replay_inference_post`
- `tests/unit/test_budget_guard.py`
  - `test_post_dispatch_timeout_preserves_unknown_consumption`
- `tests/unit/test_provider_dispatch_policy.py`
- `tests/unit/test_provider_transport_integrity.py`

Checkpoint: `58fca70ddd775f347152a72c5bcf643f37b03083`.
GitHub CI: run `43928087983`, successful.

**Next: M6-D2F3C6B — receiver-capability assessment and technical
fencing feasibility decision.**


### M6-D2F3C6B — Receiver fencing feasibility

**Decision: STRICT NO-GO. Receiver-side fencing is not verified.**

Assessment basis: repository checkpoint `58fca70`, 16 November 2026.
This is a source-code feasibility assessment, not a live provider test.

#### Receiver capability matrix

| Adapter | Verified platform boundary | Receiver evidence | Strict mode |
| --- | --- | --- | --- |
| OpenAI-compatible | Recognised adapter, approved HTTPS transport, egress control and denied POST redirects | No specific external receiver has demonstrated atomic fencing or idempotency | NO-GO |
| Watsonx | Adapter exists, but trusted remote transport gate rejects it | Transport retry behaviour and receiver fencing not verified | NO-GO |
| Offline Hugging Face | Local-only configuration requires offline model loading | External HTTP receiver fencing is not applicable to this adapter | LOCAL_ONLY |

An OpenAI-compatible API contract does not establish identical
idempotency or fencing semantics across implementations.

A configured request identifier is not proof of receiver-side
idempotency. Neither a database transaction nor a client-side
lease check can force an unrelated HTTP receiver to reject
a request from an expired lease holder.

#### Technical options

**Option A — Receiver-controlled fencing**

Applicable only when the receiving service is under sufficient
control to enforce a monotonic fencing token atomically with
its protected side effect.

Required evidence:

- Receiver rejects stale or superseded tokens.
- Token ownership and advancement are durable.
- Duplicate request handling is explicitly defined.
- Paused-worker and expired-lease tests prove rejection.
- Crash, timeout and recovery tests preserve the contract.
- Any downstream service effects are included in the guarantee.

A locally controlled gateway does not automatically provide
fencing for a third-party service behind that gateway.

**Option B — External receiver without verified fencing**

Strict no-late-I/O is not supportable on the available evidence.

The existing REMOTE_RISK_ACCEPTED mode represents a narrower
execution contract. It must not be labelled exactly-once,
at-most-once remote effect, or strict no-late-I/O.

Any operational use requires a scoped approval, identified
risk owner, provider-specific retry and idempotency evidence,
bounded exposure, and an explicit deployment decision.

#### Acceptance decision

- Keep STRICT_NO_LATE_IO fail-closed.
- Do not broaden the trusted transport allowlist.
- Do not claim remote idempotency based on request IDs alone.
- Do not treat green CI as receiver-side fencing evidence.
- Preserve the documented PostgreSQL TOCTOU reproduction.
- Do not enable unrestricted live external provider dispatch.
- Do not merge Draft PR #9 based on this assessment.

#### Next implementation gate — C6C

Before changing dispatch semantics, identify a concrete
receiver endpoint and API version, verify its documented
idempotency/fencing capabilities, and choose the supported
delivery guarantee.

If the receiver cannot enforce fencing, retain strict
NO-GO and design only a separately approved,
risk-bounded remote execution mode.


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

The M6-B4 address-pinned `StdlibJsonTransport` is a tested security primitive,
not evidence that production provider traffic already uses it. M6-J2 must wire
the concrete provider transport through this primitive and retain its egress,
TLS, redirect, and request-budget tests before any live canary.

The transport's monotonic request budget covers policy checks, approved-address
connection attempts, TCP, TLS, request writes, and HTTP response status, headers,
and body reads. Operating-system DNS resolution remains synchronous and cannot
be interrupted by `socket.getaddrinfo`; time spent resolving consumes the budget
and is rejected immediately after the lookup returns if expired, but a blocked
resolver can still make elapsed time exceed the configured budget. A strict
DNS-inclusive deadline requires a resolver with native bounded/cancellable
queries; cancelling a Python future around `getaddrinfo` is not sufficient.

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
