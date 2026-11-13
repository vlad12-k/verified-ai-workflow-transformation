# M6-A — Threat and Trust Model

**Status:** Completed
**Milestone:** M6-A
**Parent plan:** `docs/M6-PRODUCTION-HARDENING.md`

## 1. Purpose

This document defines the security and trust model for the production-shaped
VAIT platform before authentication, live provider credentials, external
provider execution, rate limiting, and deeper operational hardening are added.

The model is derived from the implemented M5/M6 platform architecture.

It does not assume controls that do not yet exist.

The system invariant remains:

    VERIFY FIRST -> OPTIMISE SECOND

Security and infrastructure controls must not weaken verification,
admissibility, optimisation, or recommendation semantics.

## 2. Scope

M6-A covers the trust and security boundaries around:

- the FastAPI service boundary;
- platform configuration;
- PostgreSQL persistence;
- experiment and evidence registry state;
- durable jobs;
- worker execution;
- artifact storage;
- structured logging;
- OpenTelemetry tracing;
- provider-neutral inference interfaces;
- future external-provider egress;
- future IBM watsonx live integration;
- container and CI execution boundaries.

M6-A documents threats and required controls.

Implementation of the controls is distributed across later M6 slices.

## 3. Current architecture security inventory

### A-01 — External HTTP client

**Type:** external actor

**Trust level:** untrusted

**Current interaction:**

    external client
        ->
    FastAPI / ASGI boundary

Current externally reachable API routes are limited to platform metadata,
liveness, readiness, and generated API documentation surfaces.

Future platform workflow endpoints will increase this attack surface.

**Primary concerns:**

- hostile input;
- oversized input;
- header manipulation;
- request flooding;
- authentication bypass once protected endpoints exist;
- replay;
- correlation-identifier abuse.

---

### A-02 — FastAPI platform boundary

**Type:** application boundary

**Trust level:** boundary between untrusted external input and trusted
platform orchestration

**Implemented controls:**

- versioned API prefix;
- typed FastAPI/Pydantic request handling;
- stable public error envelopes;
- sanitised validation-error responses;
- generated request identifiers;
- bounded and character-restricted external correlation identifiers;
- protected-environment debug validation.

The request middleware accepts an external correlation identifier only when
it satisfies the configured length and character restrictions.

Internal request identifiers are generated independently.

**Current gaps carried into M6:**

- authentication;
- authorisation;
- request-rate controls;
- request-size policy;
- endpoint-specific quotas;
- hostile-input testing.

---

### A-03 — Platform configuration

**Type:** configuration boundary

**Trust level:** trusted only when supplied by an authorised deployment
environment

**Implemented controls:**

- `VAIT_` environment-variable namespace;
- typed settings;
- `SecretStr` representation for the database URL;
- staging/production debug rejection;
- validated API prefix.

**Current gaps carried into M6:**

- explicit external-provider secret settings;
- secret-source policy;
- credential rotation policy;
- fail-closed live-provider enablement;
- provider endpoint allow-listing;
- request/token/spend budgets.

---

### A-04 — PostgreSQL

**Type:** durable state boundary

**Trust level:** privileged internal infrastructure

**Role:**

PostgreSQL is the canonical durable source of truth for platform state.

It stores, among other platform data:

- experiment lineage;
- candidate identities;
- evidence envelopes;
- artifact metadata;
- durable jobs;
- job lifecycle state;
- retry state;
- lease ownership;
- provenance metadata.

**Implemented controls:**

- SQLAlchemy persistence boundary;
- explicit transactional sessions;
- database constraints;
- foreign keys;
- indexes;
- Alembic migrations;
- database readiness checks;
- PostgreSQL-backed integration tests;
- row-locking job claims;
- `FOR UPDATE SKIP LOCKED` for competing workers.

Database readiness errors are not exposed through the public health response.

**Current gaps carried into M6:**

- operational least-privilege role validation;
- backup evidence;
- restore evidence;
- migration rollback evidence;
- database outage recovery exercises;
- credential-rotation evidence.

---

### A-05 — Experiment and Evidence Registry

**Type:** provenance and evidence boundary

**Trust level:** trusted durable platform state after validation

**Stored identity includes:**

- source revision;
- dataset fingerprint;
- dataset partition;
- transformation contract identity and version;
- candidate identity and version;
- provider identity;
- model identity;
- runtime identity;
- benchmark configuration;
- evidence kind;
- timestamps.

The registry accepts structured JSON evidence payloads.

**Primary concerns:**

- sensitive data accidentally entering evidence payloads;
- provenance spoofing;
- malicious or oversized metadata;
- artifact/evidence identity mismatch;
- evidence tampering;
- unsafe retention of provider output.

M6-B must define what categories of data may and may not enter durable
evidence payloads.

---

### A-06 — ArtifactStore boundary

**Type:** external byte-storage abstraction

**Trust level:** implementation-dependent

The current platform defines a provider-neutral `ArtifactStore` protocol with:

- `put`;
- `get`;
- `delete`.

A successful write returns:

- durable location;
- SHA-256 digest;
- byte size.

The current architecture intentionally keeps large artifact bytes outside
PostgreSQL.

**Current security state:**

The boundary contract exists.

A production artifact-store implementation is not yet established as part of
the current platform runtime.

**Primary concerns:**

- path or object-key traversal;
- untrusted storage locations;
- tampered artifact bytes;
- digest mismatch;
- unauthorised deletion;
- sensitive artifact disclosure;
- malicious artifact content.

Future implementations must verify integrity at trust-boundary crossings.

---

### A-07 — Durable job queue

**Type:** asynchronous execution boundary

**Trust level:** trusted platform state, potentially influenced by externally
originating requests

**Implemented properties:**

- explicit durable job identity;
- explicit job state machine;
- attempt count;
- maximum attempts;
- idempotency key;
- worker ownership;
- heartbeat state;
- lease expiration;
- retry state;
- terminal state;
- PostgreSQL-backed atomic claiming.

**Primary concerns:**

- duplicate submission;
- replay;
- poisoned payloads;
- unbounded retries;
- stolen/stale worker ownership;
- expired leases;
- malformed operation payloads;
- resource exhaustion;
- unauthorised operation selection.

M6-C and M6-D provide the main controls for this boundary.

---

### A-08 — Worker service

**Type:** privileged execution boundary

**Trust level:** trusted platform component processing durable job input

The current `WorkerService`:

- claims a durable job inside a transaction;
- commits ownership before executing the job;
- executes the job outside the persistence transaction;
- records success or retry/failure in a later transaction;
- binds experiment, job, and worker identities into observability context;
- can create a worker tracing span.

**Important current limitation:**

M5 provides the worker service abstraction but not a separately deployable
worker process.

M6-D owns that operational gap.

**Primary concerns:**

- malicious job payload execution;
- process crash after claim;
- duplicate side effects;
- lease expiration during long execution;
- retry storms;
- poison jobs;
- compromised worker identity;
- privilege inherited from the runtime environment.

---

### A-09 — VAIT verification and optimisation core

**Type:** trusted semantic authority

**Trust level:** trusted code operating on validated inputs and evidence

The core controls the semantics of:

- verification;
- `EXACT`;
- `BOUNDED`;
- `REJECT`;
- `NOT_APPLICABLE`;
- admissibility;
- optimisation;
- recommendation planning.

Infrastructure must not alter these semantics.

The following precedence remains security-relevant:

    verification failure
        ->
    cannot be overridden by performance,
    cost, provider identity, or operational convenience

A compromised or incorrect platform layer must not be allowed to convert a
semantic `REJECT` into an admissible or recommended result.

---

### A-10 — Structured logging

**Type:** observability boundary

**Trust level:** output boundary that may leave the application process

Current structured logging uses an allow-listed JSON representation.

It records:

- timestamp;
- level;
- logger;
- event;
- bounded contextual identifiers;
- exception type when available.

The formatter does not automatically serialise arbitrary exception bodies,
request bodies, provider payloads, or application objects.

**Primary concerns:**

- secrets passed directly as event text;
- sensitive identifiers added to context in future code;
- provider output leakage;
- PII leakage;
- log injection through future uncontrolled fields.

M6-B and M6-H must provide explicit redaction tests and policy.

---

### A-11 — OpenTelemetry tracing

**Type:** observability boundary

**Trust level:** telemetry output boundary

The current tracing runtime:

- creates an isolated `TracerProvider`;
- creates VAIT-specific spans;
- binds trace and span identifiers into structured log context;
- supports an optional span exporter;
- shuts down through application lifecycle handling.

No exporter is required by default.

**Current gaps carried into M6:**

- OpenTelemetry metrics;
- telemetry field policy;
- secret-redaction tests;
- sensitive-payload policy;
- cross-process propagation;
- production telemetry backend decisions.

---

### A-12 — Provider-neutral inference boundary

**Type:** external computation boundary

**Trust level:** provider responses are untrusted external evidence until
validated

Portable provider requests include:

- request identity;
- model identity;
- messages;
- maximum output tokens;
- temperature;
- metadata.

Portable responses may include:

- provider identity;
- runtime identity;
- model identity;
- model revision;
- output text;
- token usage;
- timing evidence;
- cost evidence;
- provider error;
- metadata.

Provider output must not be treated as trusted merely because the provider
request succeeded.

The VAIT verification layer remains authoritative.

---

### A-13 — IBM watsonx adapter boundary

**Type:** future external-provider egress boundary

**Trust level:** external provider

The current implementation provides:

- `WatsonxProvider`;
- `WatsonxTransport` protocol;
- `WatsonxTransportResponse`;
- normalised success handling;
- normalised failure handling;
- provider/model identity;
- optional deployment identity;
- token usage where available;
- timing evidence;
- provider request metadata.

A concrete live IBM transport is not yet present.

No live IBM credential is required before M6-J.

**Primary concerns for M6-J:**

- credential leakage;
- endpoint spoofing;
- incorrect project/space identity;
- model substitution;
- unbounded network waits;
- excessive retry;
- excessive token usage;
- excessive request usage;
- rate limiting;
- provider 5xx failures;
- malformed provider responses;
- provider-output injection;
- sensitive prompt or evidence disclosure.

---

### A-14 — Container runtime

**Type:** process isolation and deployment boundary

**Trust level:** privileged deployment infrastructure

Current properties include:

- containerised application;
- non-root application user;
- Docker Compose orchestration;
- PostgreSQL health checks;
- migration-before-API dependency;
- API readiness checks.

**Current gaps carried into M6:**

- separately deployable worker container;
- image vulnerability scanning;
- SBOM;
- stronger image reproducibility;
- runtime resource boundaries;
- failure and restart exercises.

---

### A-15 — CI pipeline

**Type:** build and verification boundary

**Trust level:** privileged repository automation

Current CI includes:

- read-only repository contents permission;
- Python compilation;
- Ruff;
- strict mypy;
- pytest and branch coverage;
- PostgreSQL migrations;
- PostgreSQL integration tests;
- benchmark smoke tests;
- optimisation smoke tests;
- Docker/Compose E2E;
- non-root runtime validation;
- migration-head validation.

**Current gaps carried into M6:**

- dependency vulnerability scanning;
- SAST;
- container scanning;
- SBOM generation;
- explicit security gate policy;
- protected manual IBM live workflow.

---

## 4. Initial asset classification

| Asset class | Examples | Minimum handling expectation |
| --- | --- | --- |
| Public | API metadata, public documentation | May be exposed intentionally |
| Internal | request IDs, job IDs, experiment IDs, runtime metadata | Do not expose unnecessarily |
| Evidence-sensitive | datasets, evidence payloads, provider outputs, artifacts | Validate, minimise, preserve provenance |
| Confidential | PII, customer/donor data, private benchmark inputs | Redact from telemetry; restrict persistence and access |
| Secret | API keys, database credentials, provider credentials | Never log, trace, persist as evidence, or commit |
| Integrity-critical | contracts, verification result, source revision, dataset fingerprint, artifact digest | Tampering must be detectable or rejected |

Classification is based on data sensitivity and integrity requirements, not
only confidentiality.

## 5. Initial trust-boundary inventory

| Boundary | From | To | Current validation/control | M6 owner |
| --- | --- | --- | --- | --- |
| TB-01 | External client | FastAPI | Pydantic validation, safe request/correlation IDs | M6-B/C |
| TB-02 | FastAPI | Platform orchestration | typed settings and API contracts | M6-B/C |
| TB-03 | Platform | PostgreSQL | SQLAlchemy repositories and DB constraints | M6-E |
| TB-04 | Durable job state | Worker | atomic claim, ownership, lease state | M6-D |
| TB-05 | Worker/platform | VAIT core | typed internal contracts | M6-D/I/K |
| TB-06 | Platform | ArtifactStore | protocol + SHA-256 metadata | M6-B/K |
| TB-07 | Platform | Logs/traces/metrics | allow-listed logs and isolated tracing | M6-B/H |
| TB-08 | Provider adapter | External provider | provider-neutral contract; live transport not yet enabled | M6-B/C/J |
| TB-09 | CI | Build/runtime artifacts | least-privilege contents permission and quality gates | M6-F |
| TB-10 | Containers | Host/network | non-root application runtime; localhost development bindings | M6-F/G |

## 6. Current security strengths

The current platform already establishes useful defensive properties:

- provider-neutral core boundaries;
- separation of semantic authority from infrastructure;
- typed configuration;
- protected-environment debug rejection;
- sanitised request/correlation identifiers;
- sanitised public API errors;
- database errors hidden from readiness responses;
- transactional persistence boundaries;
- database constraints;
- durable job ownership and leases;
- structured allow-listed logging;
- isolated tracing runtime;
- non-root application container;
- least-privilege CI contents permission;
- provider-neutral external inference contracts;
- explicit evidence provenance.

These controls reduce risk but do not constitute production-security
certification.

## 7. Security gaps entering M6

The current architecture does not yet establish:

- production authentication;
- production authorisation;
- formal RBAC;
- PII governance;
- secret rotation;
- provider egress controls;
- SSRF hardening;
- API rate limiting;
- provider request/token/spend budgets;
- separately deployable worker runtime;
- backup/restore proof;
- migration rollback proof;
- dependency scanning;
- SAST;
- container scanning;
- SBOM;
- load/soak evidence;
- failure-injection evidence;
- OpenTelemetry metrics;
- formal SLOs;
- operational incident runbooks;
- live IBM transport;
- full HTTP-to-evidence E2E.

These gaps are intentionally mapped to M6-B through M6-K.

## 8. M6-A next steps

M6-A continues with:

    M6-A1
    current architecture and asset inventory
        ->
    M6-A2
    explicit data-flow and trust-boundary model
        ->
    M6-A3
    threat register and risk classification
        ->
    M6-A4
    acceptance criteria and threat-model closure

No security control should be implemented before its boundary and threat are
identified unless the control is required to safely perform the threat-model
work itself.

---

## 9. M6-A2 — Data-flow and trust-boundary model

### 9.1 Purpose

M6-A2 makes the M6-A trust boundaries operational by identifying:

- what data crosses each boundary;
- where that data originates;
- whether the origin is trusted;
- what validation exists today;
- what validation is still required;
- what failure behaviour is required;
- which later M6 slice owns the control.

Crossing a validation boundary does not permanently convert data into
"trusted data".

Persisted, queued, retrieved, provider-generated, and previously validated
data may still be malformed, stale, tampered with, unauthorised, or unsafe
for a different use.

### 9.2 End-to-end platform data flow

The target production-shaped path is:

    external client
        |
        | TB-01
        v
    FastAPI boundary
        |
        | TB-02
        v
    platform orchestration
        |
        +----------------------+
        |                      |
        | TB-03                | TB-07
        v                      v
    PostgreSQL            observability
        |
        | durable experiment /
        | candidate / evidence /
        | job state
        |
        v
    durable job
        |
        | TB-04
        v
    worker
        |
        | TB-05
        v
    VAIT core
        |
        +-----------------------------+
        |                             |
        | TB-06                       | TB-08
        v                             v
    ArtifactStore                provider adapter
                                      |
                                      | external egress
                                      v
                                external provider

Additional operational boundaries are:

    CI / repository automation
        |
        | TB-09
        v
    build and container artifacts

    container runtime
        |
        | TB-10
        v
    host / network / external services

Not every target edge above is fully deployable today.

In particular:

- the separately deployable worker runtime belongs to M6-D;
- the production ArtifactStore implementation is not yet established;
- OpenTelemetry metrics belong to M6-H;
- live IBM network execution belongs to M6-J;
- the complete HTTP-to-evidence E2E path belongs to M6-K.

### 9.3 DF-01 — External client to FastAPI

**Boundary:** TB-01

**Data crossing the boundary may include:**

- HTTP method;
- URL path;
- query parameters;
- request headers;
- correlation identifier;
- future request body;
- future authentication material.

**Origin trust:** untrusted.

An Internet or network client must not become trusted because a request is
syntactically valid.

**Controls implemented today:**

- generated internal request ID;
- bounded external correlation-ID length;
- restricted external correlation-ID character set;
- typed framework validation where request models are used;
- sanitised public validation errors.

**Controls required later:**

- authentication;
- authorisation;
- request-size limits;
- request-rate limits;
- endpoint quotas;
- hostile-input tests.

**Failure rule:**

Invalid external input must fail before it can become an executable platform
operation.

**M6 owners:** M6-B, M6-C.

### 9.4 DF-02 — FastAPI to platform orchestration

**Boundary:** TB-02

**Data crossing the boundary may include:**

- validated API parameters;
- request identity;
- correlation identity;
- authenticated principal in the future;
- requested platform operation;
- experiment/job submission data in the future.

**Origin trust:** partially validated but not inherently trusted.

Syntactic API validation does not establish authorisation, business validity,
semantic safety, or resource safety.

**Required properties:**

- operation must be explicitly supported;
- authorisation must be checked before protected actions;
- resource budgets must be evaluated before expensive work;
- external correlation identifiers must never become authority tokens;
- API metadata must not silently control semantic verification outcomes.

**Failure rule:**

Unsupported or unauthorised operations must fail closed.

**M6 owners:** M6-B, M6-C, M6-K.

### 9.5 DF-03 — Platform orchestration to PostgreSQL

**Boundary:** TB-03

**Data crossing the boundary includes:**

- experiment records;
- candidate records;
- evidence envelopes;
- artifact metadata;
- durable jobs;
- retry state;
- lease state;
- provenance metadata.

**Origin trust:** trusted application code may still supply invalid,
oversized, sensitive, or inconsistent data.

**Controls implemented today:**

- typed registry contracts;
- SQLAlchemy models;
- transactions;
- foreign keys;
- check constraints;
- uniqueness constraints;
- indexes;
- Alembic-managed schema.

**Security rule:**

Database persistence is not a sanitisation mechanism.

A value being present in PostgreSQL does not make that value safe for:

- telemetry;
- shell execution;
- file-system access;
- provider requests;
- HTML rendering;
- authorisation decisions.

**Failure rule:**

Persistence failures must not be converted into successful evidence or job
completion.

**M6 owners:** M6-B, M6-D, M6-E.

### 9.6 DF-04 — Durable job state to worker

**Boundary:** TB-04

**Data crossing the boundary includes:**

- job ID;
- experiment ID;
- operation;
- payload;
- attempt count;
- retry limit;
- idempotency key;
- claimed worker identity;
- lease timestamps.

**Origin trust:** platform-generated durable state, but job operation and
payload must still be treated as execution input.

**Controls implemented today:**

- explicit job states;
- attempt limits;
- atomic PostgreSQL claim;
- worker ownership;
- heartbeat fields;
- lease expiration;
- `FOR UPDATE SKIP LOCKED`.

**Required additional controls:**

- operation allow-listing;
- payload validation per operation;
- deployable worker identity;
- heartbeat lifecycle;
- duplicate-delivery tests;
- poison-job handling;
- bounded retry scheduling;
- crash/restart recovery.

**Failure rule:**

A malformed, expired, unauthorised, or unsupported job must not produce
uncontrolled execution.

**M6 owner:** M6-D.

### 9.7 DF-05 — Worker to VAIT core

**Boundary:** TB-05

**Data crossing the boundary may include:**

- transformation contract;
- candidate identity;
- benchmark identity;
- evaluation data;
- provider/model identity;
- verification request;
- inference/evaluation configuration.

**Origin trust:** validated internal orchestration input, not semantic truth.

**Security invariant:**

The worker orchestrates the VAIT core.

The worker does not have authority to redefine core semantics.

In particular:

    REJECT
        cannot become
    admissible / recommendable

because of:

- provider identity;
- latency;
- cost;
- retry outcome;
- infrastructure state;
- worker identity.

**Failure rule:**

Platform orchestration failure must never be interpreted as successful
verification.

**M6 owners:** M6-D, M6-I, M6-K.

### 9.8 DF-06 — Platform/core to ArtifactStore

**Boundary:** TB-06

**Data crossing the boundary may include:**

- artifact key;
- artifact bytes;
- storage location;
- SHA-256 digest;
- artifact size;
- media type;
- experiment/evidence lineage.

**Origin trust:** artifact bytes may contain generated, externally sourced,
or evidence-sensitive content.

The current implementation defines the `ArtifactStore` protocol but does not
yet establish a production storage implementation.

**Required controls for a concrete implementation:**

- opaque and validated object identity;
- path/key traversal prevention;
- storage-location validation;
- integrity verification;
- access control;
- safe deletion;
- sensitive-data classification;
- storage retention policy.

**Integrity rule:**

Artifact metadata must not be considered sufficient evidence that retrieved
bytes are intact.

Where integrity matters, retrieved bytes must be checked against the
recorded digest.

**M6 owners:** M6-B, M6-K.

### 9.9 DF-07 — Platform to observability

**Boundary:** TB-07

**Data crossing the boundary may include:**

- event names;
- request ID;
- correlation ID;
- experiment ID;
- job ID;
- worker ID;
- trace ID;
- span ID;
- exception type;
- future metric labels.

**Origin trust:** internal but potentially sensitive.

Telemetry is an egress boundary even when the telemetry backend is operated
by the same organisation.

**Prohibited telemetry content includes:**

- API keys;
- database passwords;
- provider credentials;
- authentication tokens;
- raw secret values;
- unapproved PII;
- unapproved request bodies;
- unapproved provider prompt/output content.

**Current controls:**

- allow-listed JSON logging fields;
- exception type rather than automatic exception-body serialisation;
- bounded contextual identifiers;
- isolated OpenTelemetry tracing runtime.

**Required controls:**

- explicit redaction tests;
- telemetry field policy;
- OpenTelemetry metrics;
- bounded metric cardinality;
- cross-process correlation policy.

**M6 owners:** M6-B, M6-H.

### 9.10 DF-08 — Provider adapter to external provider

**Boundary:** TB-08

**Data crossing the boundary may include:**

Outbound:

- model identity;
- messages or prompt-equivalent content;
- token limit;
- generation configuration;
- approved metadata;
- provider credential at the transport layer.

Inbound:

- generated output;
- model identity;
- model revision;
- token usage;
- provider request identity;
- stop reason;
- error response;
- provider metadata.

**Origin trust:**

Outbound data is locally produced but may contain sensitive content.

Inbound provider data is external and untrusted.

A successful TLS/HTTP request does not establish semantic correctness.

**Current state:**

Provider-neutral contracts exist.

The watsonx adapter boundary exists.

A concrete live IBM transport is not yet enabled.

**Required controls before IBM live execution:**

- fail-closed live-mode switch;
- secret isolation;
- approved endpoint configuration;
- project/space identity pinning;
- model identity pinning;
- bounded timeout;
- bounded retries;
- request budget;
- token budget;
- response validation;
- telemetry redaction.

**Semantic rule:**

External-provider success is evidence of execution only.

It is not evidence of verification success.

**M6 owners:** M6-B, M6-C, M6-J.

### 9.11 DF-09 — CI to build and runtime artifacts

**Boundary:** TB-09

**Data crossing the boundary includes:**

- source revision;
- dependency definitions;
- downloaded dependencies;
- generated packages;
- container layers;
- test artifacts;
- future SBOM;
- future security-scan results.

**Origin trust:** mixed.

Repository source is controlled, while external dependencies, package
registries, container bases, and reusable CI actions are external supply-chain
inputs.

**Current controls:**

- least-privilege `contents: read`;
- compilation;
- Ruff;
- mypy;
- tests;
- branch coverage gate;
- PostgreSQL integration;
- migration validation;
- Compose E2E.

**Required controls:**

- dependency scanning;
- SAST;
- container scanning;
- SBOM generation;
- workflow dependency review;
- explicit handling of security findings.

**M6 owner:** M6-F.

### 9.12 DF-10 — Container runtime to host and network

**Boundary:** TB-10

**Data crossing the boundary may include:**

- database network traffic;
- API network traffic;
- future provider network traffic;
- environment configuration;
- container filesystem access;
- runtime process resources.

**Origin trust:** mixed.

Containers reduce process isolation risk but are not a complete security
boundary.

**Current controls:**

- non-root application user;
- localhost-bound development ports;
- explicit Compose dependencies;
- health/readiness checks.

**Required controls:**

- worker container;
- image scanning;
- runtime resource limits where justified;
- provider egress policy;
- restart/failure testing;
- stronger image reproducibility evidence.

**M6 owners:** M6-D, M6-F, M6-G, M6-J.

### 9.13 Boundary validation matrix

| Boundary | Data authority after crossing | Must still be treated as untrusted for |
| --- | --- | --- |
| TB-01 | Syntactically accepted request | Auth, authorisation, semantics, resource use |
| TB-02 | Accepted platform command | Persistence safety, job safety, semantic verification |
| TB-03 | Persisted durable state | Telemetry, execution, provider egress, authorisation |
| TB-04 | Claimed job | Operation safety, payload safety, idempotency |
| TB-05 | Core input | Verification result until core evaluates it |
| TB-06 | Stored artifact | Integrity until digest/provenance checks pass |
| TB-07 | Telemetry event | External disclosure and cardinality safety |
| TB-08 | Provider response | Semantic correctness and safety |
| TB-09 | Built artifact | Supply-chain integrity and runtime safety |
| TB-10 | Running container | Host/network trust and external-service trust |

### 9.14 Security invariants across all boundaries

The following invariants apply globally:

1. Validation is contextual, not permanent.
2. Persistence does not make input trustworthy.
3. Authentication does not imply authorisation.
4. Authorisation does not imply semantic correctness.
5. Provider success does not imply verification success.
6. Telemetry is treated as an external disclosure boundary.
7. Secrets must never become ordinary evidence or telemetry.
8. Correlation identifiers are observability identifiers, not credentials.
9. Retry must never become unbounded execution.
10. A stale or expired worker lease must not confer valid ownership.
11. Artifact location metadata does not prove artifact integrity.
12. Infrastructure evidence must not override a core `REJECT`.
13. Optimisation must not override verification or workflow risk constraints.
14. Live external-provider execution must be explicitly enabled.
15. External-provider budgets must be checked before the external request.

### 9.15 M6-A2 outcome

M6-A2 establishes the concrete data-flow model required for the threat
register.

The next step is M6-A3:

    assets
        +
    trust boundaries
        +
    data flows
        ->
    threat scenarios
        ->
    likelihood / impact
        ->
    control owner
        ->
    residual risk

No threat may be marked mitigated merely because a future M6 slice intends to
implement a control.

---

## 10. M6-A3 — Threat register and risk classification

### 10.1 Purpose

M6-A3 converts the architecture, assets, trust boundaries, and data flows
identified in M6-A1 and M6-A2 into explicit threat scenarios.

Each threat records:

- affected assets;
- affected trust boundaries;
- attack or failure scenario;
- current controls;
- missing controls;
- likelihood;
- impact;
- current risk;
- owning M6 slice;
- closure condition.

A threat is not considered mitigated merely because a future M6 slice plans
to implement a control.

### 10.2 Risk scale

Likelihood is classified as:

- `LOW` — requires unusual access, conditions, or multiple failures;
- `MEDIUM` — plausible under realistic misuse or operational failure;
- `HIGH` — likely under direct exposure, routine failure, or intentional abuse.

Impact is classified as:

- `LOW` — limited operational inconvenience with no meaningful integrity,
  confidentiality, or semantic impact;
- `MEDIUM` — bounded service disruption or limited sensitive-data exposure;
- `HIGH` — major service disruption, material integrity failure, credential
  exposure, or incorrect platform operation;
- `CRITICAL` — compromise of semantic authority, secrets, protected data,
  evidence integrity, or uncontrolled external execution.

Current risk is classified as:

- `LOW`;
- `MEDIUM`;
- `HIGH`;
- `CRITICAL`.

Risk classification reflects the architecture entering M6.

It is not a certification or quantitative probability estimate.

### 10.3 Threat register

| ID | Threat | Boundaries | Likelihood | Impact | Current risk | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| T-01 | Unauthenticated access to future protected operations | TB-01, TB-02 | HIGH | HIGH | CRITICAL | M6-B |
| T-02 | Authorised identity performs an unauthorised operation | TB-02 | MEDIUM | HIGH | HIGH | M6-B |
| T-03 | Hostile or oversized API input causes unsafe processing or resource exhaustion | TB-01, TB-02 | MEDIUM | HIGH | HIGH | M6-B, M6-C, M6-G |
| T-04 | SSRF or uncontrolled provider egress reaches an unintended endpoint | TB-02, TB-08, TB-10 | MEDIUM | CRITICAL | CRITICAL | M6-B, M6-J |
| T-05 | Secret or credential is emitted into logs, traces, metrics, errors, evidence, or artifacts | TB-07, TB-08 | MEDIUM | CRITICAL | CRITICAL | M6-B, M6-H, M6-J |
| T-06 | PII or sensitive benchmark/provider content is disclosed through persistence or external-provider execution | TB-03, TB-06, TB-08 | MEDIUM | CRITICAL | CRITICAL | M6-B, M6-J |
| T-07 | Duplicate or replayed job causes repeated execution or side effects | TB-03, TB-04, TB-05 | MEDIUM | HIGH | HIGH | M6-C, M6-D |
| T-08 | Stale lease or worker-ownership error allows concurrent execution of one job | TB-04 | MEDIUM | HIGH | HIGH | M6-D |
| T-09 | Poison job or repeated failure creates an unbounded retry storm | TB-04, TB-05 | MEDIUM | HIGH | HIGH | M6-C, M6-D |
| T-10 | Database tampering or inconsistent durable state corrupts provenance or decision evidence | TB-03 | LOW | CRITICAL | HIGH | M6-E, M6-K |
| T-11 | Backup or restore failure prevents recovery of canonical durable state | TB-03 | MEDIUM | HIGH | HIGH | M6-E |
| T-12 | Artifact path/key abuse, deletion, or tampering breaks evidence integrity or confidentiality | TB-06 | MEDIUM | HIGH | HIGH | M6-B, M6-K |
| T-13 | External provider returns substituted identity, malformed output, or manipulated metadata | TB-08 | MEDIUM | HIGH | HIGH | M6-J |
| T-14 | Provider timeout, rate limit, or server failure causes uncontrolled retries, spend, or stalled work | TB-08, TB-10 | HIGH | HIGH | CRITICAL | M6-C, M6-J |
| T-15 | Telemetry leaks sensitive content or creates unbounded-cardinality resource pressure | TB-07 | MEDIUM | HIGH | HIGH | M6-B, M6-H |
| T-16 | Compromised dependency, CI action, package, or container base enters the build | TB-09, TB-10 | MEDIUM | CRITICAL | CRITICAL | M6-F |
| T-17 | Container runtime or worker receives excessive privilege or consumes uncontrolled host resources | TB-10 | MEDIUM | HIGH | HIGH | M6-D, M6-F, M6-G |
| T-18 | Platform or infrastructure logic bypasses VAIT semantic authority and converts a failed candidate into an admissible/recommended result | TB-02, TB-04, TB-05 | LOW | CRITICAL | HIGH | M6-I, M6-K |
| T-19 | Failed or partially applied database migration leaves schema and application state incompatible | TB-03, TB-09 | MEDIUM | HIGH | HIGH | M6-E |
| T-20 | Live external-provider execution is enabled accidentally or without explicit budget/credential controls | TB-08, TB-09, TB-10 | MEDIUM | CRITICAL | CRITICAL | M6-B, M6-C, M6-J |

### 10.4 T-01 — Unauthenticated protected operation

**Scenario:**

A future endpoint capable of creating experiments, jobs, provider calls, or
evidence operations becomes reachable without authentication.

**Affected assets:**

- platform operations;
- job queue;
- provider budget;
- evidence state.

**Existing controls:**

- typed API boundary;
- request identifiers;
- safe validation errors.

**Missing controls:**

- authentication;
- protected-route policy;
- default-deny behaviour.

**Closure condition:**

Protected operations require an authenticated principal and fail closed when
authentication is absent or invalid.

**Owner:** M6-B.

### 10.5 T-02 — Authorisation or privilege escalation failure

**Scenario:**

An authenticated identity accesses an operation or resource outside its
permitted role or scope.

**Existing controls:**

No production authorisation model exists yet.

**Missing controls:**

- explicit authorisation policy;
- role/resource mapping;
- default-deny checks;
- negative authorisation tests.

**Closure condition:**

Protected actions are authorised explicitly and denied by default.

**Owner:** M6-B.

### 10.6 T-03 — Hostile input and resource exhaustion

**Scenario:**

A client submits malformed, oversized, deeply nested, or high-volume input
that causes excessive CPU, memory, database, job, or provider use.

**Existing controls:**

- typed framework validation;
- bounded correlation identifiers;
- job attempt limits.

**Missing controls:**

- request-size policy;
- rate limiting;
- job quotas;
- hostile-input tests;
- load/failure evidence.

**Closure condition:**

Inputs and request rates are bounded before expensive platform work begins.

**Owners:** M6-B, M6-C, M6-G.

### 10.7 T-04 — SSRF and uncontrolled provider egress

**Scenario:**

Input or configuration causes the platform to make a network request to an
unapproved endpoint or internal network resource.

**Affected boundaries:**

- TB-02;
- TB-08;
- TB-10.

**Existing controls:**

Provider-neutral adapters use configured transport boundaries.

There is no live IBM transport yet.

**Missing controls:**

- approved endpoint policy;
- endpoint validation;
- redirect policy;
- provider allow-listing;
- hostile endpoint tests.

**Closure condition:**

Live provider traffic can reach only explicitly approved provider endpoints.

**Owners:** M6-B, M6-J.

### 10.8 T-05 — Secret leakage

**Scenario:**

A database password, IBM API key, authentication token, or other credential
enters:

- structured logs;
- trace attributes;
- metric labels;
- public error responses;
- evidence payloads;
- artifacts.

**Existing controls:**

- database URL represented by `SecretStr`;
- allow-listed JSON logging;
- public API error sanitisation;
- exception type logged without automatic exception-body serialisation.

**Missing controls:**

- explicit secret-classification policy;
- redaction tests;
- provider-secret settings;
- metrics redaction policy;
- live-provider secret tests.

**Closure condition:**

Known secret values are absent from every supported observability and
evidence output under positive and failure paths.

**Owners:** M6-B, M6-H, M6-J.

### 10.9 T-06 — Sensitive-data or PII disclosure

**Scenario:**

PII, donor/customer data, benchmark records, prompt content, or provider
output is persisted or transmitted beyond its approved scope.

**Existing controls:**

- evidence registry has explicit schemas;
- ArtifactStore is abstracted;
- telemetry is currently narrow.

**Missing controls:**

- data classification policy;
- PII governance;
- provider-egress data policy;
- persistence minimisation;
- retention guidance.

**Closure condition:**

Every sensitive-data class has an explicit policy for persistence,
telemetry, artifact storage, and external-provider transmission.

**Owners:** M6-B, M6-J.

### 10.10 T-07 — Duplicate or replayed durable job

**Scenario:**

The same logical operation is submitted or delivered more than once and
produces duplicate execution or side effects.

**Existing controls:**

- durable job identity;
- idempotency-key field;
- atomic claim;
- explicit job state.

**Missing controls:**

- enforced idempotency semantics;
- duplicate-submission tests;
- duplicate-delivery tests;
- side-effect policy.

**Closure condition:**

Duplicate delivery has deterministic, tested behaviour and cannot silently
create uncontrolled duplicate effects.

**Owners:** M6-C, M6-D.

### 10.11 T-08 — Stale lease and concurrent worker ownership

**Scenario:**

A worker continues processing after its lease is no longer valid while
another worker recovers and executes the same job.

**Existing controls:**

- claimed worker identity;
- heartbeat fields;
- lease expiration;
- ownership checks when persisting success/failure;
- expired-lease recovery.

**Missing controls:**

- operational heartbeat lifecycle;
- long-running-job lease renewal;
- concurrent recovery tests;
- stale-owner tests.

**Closure condition:**

Lease expiry and recovery behaviour is explicitly tested under concurrent
worker failure scenarios.

**Owner:** M6-D.

### 10.12 T-09 — Poison job and retry storm

**Scenario:**

A permanently failing payload repeatedly consumes worker and database
resources.

**Existing controls:**

- maximum attempt count;
- `RETRY_PENDING`;
- terminal `FAILED` state.

**Missing controls:**

- operational requeue policy;
- bounded retry timing;
- poison-job classification;
- observability and alert evidence.

**Closure condition:**

A permanently failing job reaches a deterministic terminal state without
unbounded execution.

**Owners:** M6-C, M6-D, M6-H.

### 10.13 T-10 — Durable-state or evidence tampering

**Scenario:**

Experiment lineage, candidate identity, evidence payload, recommendation
evidence, or artifact metadata is altered inconsistently.

**Existing controls:**

- typed models;
- relational constraints;
- source revision;
- dataset fingerprint;
- artifact SHA-256 metadata;
- M4 release-evidence manifest.

**Missing controls:**

- recovery validation;
- end-to-end persisted-evidence checks;
- artifact retrieval integrity checks;
- production access-control evidence.

**Closure condition:**

M6-K demonstrates durable evidence lineage across the complete platform path,
and integrity-critical artifacts are verifiable.

**Owners:** M6-E, M6-K.

### 10.14 T-11 — Backup and restore failure

**Scenario:**

Canonical PostgreSQL state is lost or corrupted and cannot be restored
correctly.

**Existing controls:**

- persistent PostgreSQL volume;
- migrations;
- reproducible clean bootstrap.

**Missing controls:**

- backup procedure;
- restore procedure;
- restored-state validation;
- recovery exercise.

**Closure condition:**

A tested backup can restore a known platform state with validated schema and
record integrity.

**Owner:** M6-E.

### 10.15 T-12 — Artifact storage abuse or tampering

**Scenario:**

An artifact key or location performs traversal, points to an unauthorised
object, is deleted without authority, or returns bytes that do not match the
recorded digest.

**Existing controls:**

- provider-neutral ArtifactStore contract;
- SHA-256 metadata;
- recorded byte size.

**Missing controls:**

- concrete storage implementation;
- key/location policy;
- retrieval-time integrity verification;
- access-control behaviour.

**Closure condition:**

Concrete artifact storage used by the platform enforces object identity,
access policy, and integrity checks.

**Owners:** M6-B, M6-K.

### 10.16 T-13 — External-provider identity or response manipulation

**Scenario:**

The configured external provider returns:

- a different model identity;
- malformed response fields;
- invalid usage metadata;
- unexpected deployment identity;
- attacker-controlled metadata.

**Existing controls:**

- provider-neutral response model;
- watsonx model mismatch handling;
- normalised provider errors.

**Missing controls:**

- concrete transport validation;
- pinned live project/model/deployment identity;
- hostile-response tests.

**Closure condition:**

Unexpected provider identity or malformed response fails safely and cannot
silently become valid VAIT evidence.

**Owner:** M6-J.

### 10.17 T-14 — Unbounded provider retry, latency, or spend

**Scenario:**

Timeouts, 429 responses, 5xx failures, or retry logic cause excessive
provider requests, token usage, latency, or cost.

**Existing controls:**

Provider-neutral responses support normalised retryability information.

**Missing controls:**

- bounded timeout;
- retry budget;
- request budget;
- token budget;
- optional spend budget;
- live execution ceiling.

**Closure condition:**

External execution is bounded before and during provider use, and exhaustion
fails deterministically.

**Owners:** M6-C, M6-J.

### 10.18 T-15 — Telemetry leakage or cardinality explosion

**Scenario:**

Sensitive or uncontrolled values enter logs, traces, or future metric labels,
causing disclosure or excessive telemetry resource usage.

**Existing controls:**

- allow-listed log payload;
- bounded context identifiers;
- isolated tracer.

**Missing controls:**

- formal telemetry field policy;
- metric label policy;
- redaction tests;
- cardinality tests.

**Closure condition:**

Telemetry fields are explicitly controlled, sensitive values are rejected or
redacted, and metric cardinality is bounded.

**Owners:** M6-B, M6-H.

### 10.19 T-16 — Software supply-chain compromise

**Scenario:**

A vulnerable or malicious dependency, package, reusable workflow, or
container base enters the build and executes with CI or runtime authority.

**Existing controls:**

- quality gates;
- read-only repository contents permission;
- reproducible source revision.

**Missing controls:**

- dependency vulnerability scanning;
- SAST;
- container scanning;
- SBOM;
- security finding policy.

**Closure condition:**

M6-F produces machine-readable supply-chain evidence and blocks explicitly
defined unacceptable findings.

**Owner:** M6-F.

### 10.20 T-17 — Runtime privilege or resource abuse

**Scenario:**

A compromised or malfunctioning API/worker process consumes uncontrolled
host resources or receives unnecessary privilege.

**Existing controls:**

- non-root API image;
- Compose isolation;
- localhost-bound development service ports.

**Missing controls:**

- worker runtime hardening;
- resource limits where justified;
- container scanning;
- load and failure testing.

**Closure condition:**

Runtime privilege and resource behaviour are explicitly validated under the
M6 deployment model.

**Owners:** M6-D, M6-F, M6-G.

### 10.21 T-18 — Semantic-authority bypass

**Scenario:**

Platform state, worker logic, provider identity, performance evidence, or
optimisation logic causes a candidate with failed verification to become
admissible or recommended.

**Impact:**

This threatens the central VAIT safety invariant.

**Existing controls:**

- `EXACT / BOUNDED / REJECT` semantics;
- admissibility gate;
- optimisation precedence;
- recommendation non-execution boundary.

**Required additional evidence:**

- workflow-composition precedence;
- full platform E2E tests;
- persisted-result precedence tests.

**Closure condition:**

A `REJECT` remains terminal across workflow composition, persistence,
background execution, optimisation, recommendation, and result retrieval.

**Owners:** M6-I, M6-K.

### 10.22 T-19 — Migration failure or partial schema transition

**Scenario:**

A migration fails or application/schema versions become incompatible,
leaving durable state unavailable or inconsistent.

**Existing controls:**

- Alembic migration chain;
- clean upgrade validation;
- CI migration gate.

**Missing controls:**

- rollback strategy;
- downgrade evidence where supported;
- failure recovery procedure;
- restore interaction with migration state.

**Closure condition:**

A documented and tested recovery path exists for the supported migration
failure scenarios.

**Owner:** M6-E.

### 10.23 T-20 — Accidental live external-provider activation

**Scenario:**

Development, CI, or an operator accidentally enables IBM or another live
external provider, causing unintended data egress or billable requests.

**Existing controls:**

A concrete IBM live transport is not yet present.

**Missing controls:**

- fail-closed live switch;
- explicit protected environment;
- credential presence check;
- request/token budget;
- manual live workflow;
- ordinary-CI exclusion.

**Closure condition:**

Live IBM execution requires explicit enablement plus valid protected
configuration and cannot occur in ordinary development or CI by accident.

**Owners:** M6-B, M6-C, M6-J.

### 10.24 Threat-to-M6 ownership matrix

| M6 slice | Threats primarily addressed |
| --- | --- |
| M6-B | T-01, T-02, T-03, T-04, T-05, T-06, T-12, T-15, T-20 |
| M6-C | T-03, T-07, T-09, T-14, T-20 |
| M6-D | T-07, T-08, T-09, T-17 |
| M6-E | T-10, T-11, T-19 |
| M6-F | T-16, T-17 |
| M6-G | T-03, T-17 |
| M6-H | T-05, T-09, T-15 |
| M6-I | T-18 |
| M6-J | T-04, T-05, T-06, T-13, T-14, T-20 |
| M6-K | T-10, T-12, T-18 |

### 10.25 Critical-risk gate

The following threats enter M6 with `CRITICAL` current risk:

- T-01 — unauthenticated protected operation;
- T-04 — SSRF or uncontrolled provider egress;
- T-05 — secret leakage;
- T-06 — sensitive-data or PII disclosure;
- T-14 — unbounded provider retry, latency, or spend;
- T-16 — software supply-chain compromise;
- T-20 — accidental live external-provider activation.

This classification does not mean the current local development system is
actively exposed to every scenario.

It means these scenarios have critical impact under the target
production-shaped architecture and require explicit controls before the
relevant capability is exposed.

### 10.26 Risk acceptance rule

A threat may move from `OPEN` to `CONTROLLED` only when:

1. the control exists in code, deployment configuration, or an explicit
   operational procedure;
2. the relevant positive and negative tests exist;
3. the tests pass;
4. evidence is reproducible;
5. remaining limitations are documented.

A planned control is not an implemented control.

A passing scanner is not proof that a threat is eliminated.

A low observed failure count is not proof of zero risk.

### 10.27 M6-A3 outcome

M6-A3 establishes the threat register that drives implementation work in
M6-B through M6-K.

The next step is M6-A4:

    architecture inventory
        +
    trust boundaries
        +
    data flows
        +
    threat register
        ->
    M6-A acceptance criteria
        ->
    closure evidence
        ->
    M6-A complete

M6-A4 must verify that every critical and high threat has:

- an explicit owner;
- a closure condition;
- a mapped later M6 slice;
- no false claim of mitigation before implementation.

---

## 11. M6-A4 — Acceptance and closure

### 11.1 Purpose

M6-A4 closes the threat-modelling slice by verifying traceability between:

- implemented architecture;
- protected assets;
- trust boundaries;
- data flows;
- threat scenarios;
- risk classification;
- closure conditions;
- later M6 implementation owners.

M6-A is an analysis and architecture-security milestone.

It identifies and classifies security requirements.

It does not claim that risks assigned to M6-B through M6-K are already
mitigated.

### 11.2 M6-A acceptance criteria

M6-A is complete only when all of the following are true:

1. The security model is based on implemented VAIT architecture rather than
   a hypothetical generic platform.
2. Assets A-01 through A-15 are explicitly identified.
3. Trust boundaries TB-01 through TB-10 are explicitly identified.
4. Data flows DF-01 through DF-10 are explicitly identified.
5. Threats T-01 through T-20 are explicitly identified.
6. Every registered threat has an M6 implementation owner.
7. Every registered threat has a closure condition.
8. Critical-risk threats are explicitly identified.
9. Existing controls are distinguished from missing controls.
10. Planned controls are not described as implemented controls.
11. External-provider responses remain untrusted until validated.
12. Provider success is not treated as verification success.
13. Infrastructure state cannot override a VAIT semantic `REJECT`.
14. Secrets are classified separately from ordinary evidence.
15. Telemetry is treated as a disclosure boundary.
16. Durable persistence is not treated as automatic sanitisation.
17. Worker/job ownership and retry risks are represented.
18. PostgreSQL recovery and migration risks are represented.
19. Artifact integrity and storage risks are represented.
20. Supply-chain and container risks are represented.
21. IBM live-provider risks are represented before any live IBM execution.
22. No IBM account, credential, or billable request is required to complete
    M6-A.
23. All critical and high risks remain explicitly open until their owning M6
    slices produce implementation and test evidence.
24. The model preserves `VERIFY FIRST -> OPTIMISE SECOND`.

### 11.3 Architecture-to-threat traceability

| Architecture area | Assets | Primary boundaries | Primary flows | Principal threats |
| --- | --- | --- | --- | --- |
| External API | A-01, A-02 | TB-01, TB-02 | DF-01, DF-02 | T-01, T-02, T-03 |
| Configuration and secrets | A-03 | TB-02, TB-07, TB-08 | DF-02, DF-07, DF-08 | T-04, T-05, T-20 |
| PostgreSQL | A-04, A-05 | TB-03 | DF-03 | T-06, T-10, T-11, T-19 |
| Durable jobs | A-07 | TB-03, TB-04 | DF-03, DF-04 | T-07, T-08, T-09 |
| Worker | A-08 | TB-04, TB-05 | DF-04, DF-05 | T-07, T-08, T-09, T-17 |
| VAIT semantic core | A-09 | TB-05 | DF-05 | T-18 |
| Artifact storage | A-06 | TB-06 | DF-06 | T-06, T-12 |
| Logging and tracing | A-10, A-11 | TB-07 | DF-07 | T-05, T-15 |
| Provider abstraction | A-12 | TB-08 | DF-08 | T-13, T-14 |
| IBM watsonx boundary | A-13 | TB-08, TB-10 | DF-08, DF-10 | T-04, T-05, T-06, T-13, T-14, T-20 |
| Containers | A-14 | TB-10 | DF-10 | T-16, T-17 |
| CI / supply chain | A-15 | TB-09 | DF-09 | T-16, T-19, T-20 |

### 11.4 Critical-risk implementation gates

The M6-A threat register identifies these current critical-risk scenarios:

| Threat | Capability blocked until controls exist | Primary closure slice |
| --- | --- | --- |
| T-01 | Protected production operations | M6-B |
| T-04 | Live external-provider egress | M6-B / M6-J |
| T-05 | Live secrets and production telemetry | M6-B / M6-H / M6-J |
| T-06 | Sensitive-data external-provider use | M6-B / M6-J |
| T-14 | Unbounded live provider execution | M6-C / M6-J |
| T-16 | Release-grade build supply chain | M6-F |
| T-20 | IBM live execution | M6-B / M6-C / M6-J |

These gates intentionally block exposure of the relevant capability.

They do not block unrelated local development or mock-based testing.

### 11.5 IBM pre-live security gate

M6-A establishes that IBM live integration must not begin until the
prerequisite controls have been implemented and tested.

The required sequence remains:

    M6-A
        ->
    threat model established
        ->
    M6-B
        ->
    secrets / auth / PII / provider-egress controls
        ->
    M6-C
        ->
    request / token / retry / spend budgets
        ->
    operational hardening
        ->
    M6-J
        ->
    IBM account inventory
        ->
    concrete transport
        ->
    hostile-response tests
        ->
    manual live canary

Entering an IBM account is not required during M6-A.

Creating an IBM API key is not required during M6-A.

Performing a live IBM request is prohibited as part of M6-A.

### 11.6 Semantic-authority security gate

Threat T-18 is treated separately from ordinary infrastructure risk because
it protects the defining VAIT invariant.

Across every later M6 slice:

    REJECT
        ->
    remains terminal

and cannot be converted into:

    EXACT
    BOUNDED
    admissible
    Pareto-feasible
    preferred
    READY

because of:

- API state;
- database state;
- worker state;
- retry state;
- provider identity;
- latency;
- throughput;
- resource evidence;
- cost evidence;
- operational convenience.

M6-I must prove this rule for workflow composition.

M6-K must prove this rule across the persisted end-to-end platform path.

### 11.7 Risk-state convention

Threat status is represented conceptually as:

    OPEN
        ->
    CONTROL IMPLEMENTED
        ->
    CONTROL TESTED
        ->
    CONTROLLED

A threat must not move directly from:

    OPEN
        ->
    CONTROLLED

because a control was merely planned or documented.

`CONTROLLED` does not mean zero risk.

It means the defined control and its evidence satisfy the declared closure
condition for the current project scope.

Residual limitations must remain documented.

### 11.8 Deferred control ownership

M6-A intentionally does not implement the controls identified by the threat
register.

Implementation ownership is:

| Slice | Security responsibility |
| --- | --- |
| M6-B | Authentication, authorisation, secrets, PII, telemetry redaction, provider egress |
| M6-C | Rate limits, quotas, request/token/retry/spend budgets |
| M6-D | Worker lifecycle, idempotency, duplicate delivery, poison jobs, lease recovery |
| M6-E | Backup, restore, migration recovery |
| M6-F | Dependency, SAST, container, SBOM and CI security |
| M6-G | Load, soak and controlled failure injection |
| M6-H | Metrics, telemetry policy, SLO evidence and operational observability |
| M6-I | Workflow composition and semantic risk precedence |
| M6-J | Concrete IBM transport, hostile-provider tests and live canary |
| M6-K | Full persisted platform E2E and final M6 evidence |

### 11.9 M6-A closure evidence

The M6-A documentation must contain and validate:

    A-01..A-15
        +
    TB-01..TB-10
        +
    DF-01..DF-10
        +
    T-01..T-20
        +
    ownership
        +
    closure conditions
        +
    critical-risk gates

M6-A completion establishes the security design input for M6-B through M6-K.

It does not itself reduce every registered risk.

### 11.10 M6-A closure decision

When the structural and repository-quality checks for this document pass:

    M6-A — Threat and Trust Model
        =
    COMPLETE

The next implementation slice is:

    M6-B — Authentication, secrets, PII, and provider security

M6-B must consume this threat model rather than creating an unrelated
security design.

Any material architecture change introduced later in M6 that creates a new
asset, trust boundary, data flow, or threat must update this model.
