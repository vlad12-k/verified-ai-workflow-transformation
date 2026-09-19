# M5 — AI Platform Engineering

**Status:** In progress

## Goal

Turn the existing VAIT verification and optimisation engine into a
production-shaped platform while preserving the provider-neutral core.

M5 does not change the semantic authority of verification.

The system invariant remains:

```text
VERIFY FIRST -> OPTIMISE SECOND
```

Platform infrastructure may expose, persist, schedule, and observe VAIT
operations.

It must not weaken verification, admissibility, optimisation, or
recommendation semantics.

## Architectural law

Dependency direction is:

```text
FastAPI / PostgreSQL / workers / Redis / OpenTelemetry
                         |
                         v
                  vait.platform
                         |
                         v
                     VAIT core
```

Core packages must never depend upward on the platform stack.

## M5-A — Platform boundary and configuration

Deliverables:

- isolated `vait.platform` package;
- typed platform configuration;
- `VAIT_` environment-variable namespace;
- safe protected-environment defaults;
- architectural dependency guard;
- trust-boundary documentation;
- M5/M6 security responsibility boundary.

## M5-B — FastAPI service contract

Deliverables:

- FastAPI application factory;
- versioned `/api/v1` API;
- liveness endpoint;
- readiness endpoint;
- platform/version metadata;
- typed API errors;
- request/correlation identifiers;
- OpenAPI contract;
- API tests.

## M5-C — PostgreSQL persistence

Deliverables:

- SQLAlchemy 2 persistence layer;
- psycopg 3 PostgreSQL driver;
- repository boundary;
- connection lifecycle management;
- explicit transactions;
- database constraints and indexes;
- integration tests against real PostgreSQL.

PostgreSQL is the durable source of truth for platform state.

## M5-D — Schema migrations

Deliverables:

- Alembic migration framework;
- versioned migrations;
- upgrade-from-empty validation;
- migration integration tests;
- schema compatibility rules.

Zero-downtime migration and rollback hardening remain M6 concerns.

## M5-E — Experiment and Evidence Registry

Persist the lineage:

```text
ExperimentRun
    |
    v
Candidate / SearchPoint
    |
    v
Verification Evidence
    |
    v
Inference / Resource / Economic Evidence
    |
    v
Optimisation Analysis
    |
    v
Recommendation Plan
```

Preserve where applicable:

- source revision;
- dataset fingerprint;
- dataset partition identity;
- transformation contract identity/version;
- candidate identity/version;
- model/provider/runtime identity;
- benchmark configuration;
- artifact SHA-256;
- timestamps;
- reproduction lineage.

Large binary artifacts must not be stored blindly inside PostgreSQL.

PostgreSQL stores durable artifact metadata, integrity information, and
artifact location.

Artifact bytes are accessed through an `ArtifactStore` boundary.

## M5-F — Background execution

Long-running model training, evaluation, benchmark, inference, and
verification operations must execute outside long-lived HTTP requests.

Deliverables:

- durable job model;
- explicit job state machine;
- worker abstraction;
- safe job claiming;
- bounded retries;
- lease/heartbeat semantics;
- cancellation semantics;
- idempotency baseline;
- crash/restart tests.

Initial job execution may use PostgreSQL-backed claiming.

PostgreSQL remains the durable source of truth.

## M5-G — Redis decision gate

Redis is optional.

It is introduced only if measured requirements justify one or more of:

- multiple distributed workers;
- shared queueing;
- delayed execution;
- back-pressure;
- distributed coordination;
- short-lived shared caching.

Valid outcomes are:

```text
REDIS_REQUIRED
REDIS_NOT_JUSTIFIED
```

Redis must never become the sole durable store for experiment or evidence
state.

## M5-H — Observability

Observability consists of three distinct concerns:

```text
structured JSON logging
+
OpenTelemetry tracing
+
OpenTelemetry metrics
```

Correlation must work across:

```text
HTTP request
    -> experiment
    -> job
    -> worker
    -> VAIT core
    -> provider/model
    -> persistence
```

Secrets, credentials, authentication material, and sensitive payloads
must not be emitted into telemetry.

## M5-I — Containers and environments

Deliverables:

- hardened Dockerfile;
- non-root runtime;
- Docker Compose development stack;
- PostgreSQL service;
- worker service;
- optional Redis service if justified;
- OpenTelemetry Collector where justified;
- separate development, test, and staging configuration.

## M5-J — CI/CD and platform evidence

Deliverables:

- platform dependency installation in CI;
- PostgreSQL integration tests;
- migration validation;
- container build;
- container smoke tests;
- end-to-end API/job/evidence validation;
- deployment environment configuration;
- M5 release evidence.

## Security baseline

Security is cross-cutting throughout M5.

M5 must establish:

- no secrets committed to source control;
- no credentials emitted into logs or telemetry;
- validated external input boundaries;
- safe configuration defaults;
- least-privilege CI permissions;
- least-privilege database design;
- non-root container execution;
- protected environment separation;
- telemetry redaction;
- durable evidence provenance;
- architectural dependency enforcement.

M6 will perform deeper attack-oriented and operational hardening.

## M5 to M6 security relationship

```text
M5
build explicit secure platform boundaries
             |
             v
M6
attack, validate, stress, and harden those boundaries
```

M6 includes later work such as:

- authentication and authorisation;
- RBAC where justified;
- PII governance;
- secrets rotation;
- rate limiting and quotas;
- SSRF and hostile-input testing;
- retry/idempotency failure testing;
- dependency and container scanning;
- SAST;
- SBOM;
- backup/restore;
- migration rollback;
- load and soak tests;
- failure injection;
- chaos testing;
- SLOs;
- runbooks;
- formal threat modelling;
- recovery exercises.

## Data and model validation continuity

The platform must preserve future validation using:

```text
controlled synthetic benchmark
+
public or donor data
+
adversarial data
+
frozen external hold-out data
```

Training/development data and frozen validation data must have distinct
provenance.

Frozen external hold-out data must not be silently reused for tuning.

Training, evaluation, and external validation runs must retain enough
metadata to identify the exact dataset version and partition used.

## M5-A acceptance criteria

M5-A is complete only when:

1. `vait.platform` exists as a separate package.
2. Platform settings are typed.
3. Environment variables use the `VAIT_` prefix.
4. Debug mode is rejected in staging and production.
5. API-prefix configuration is validated.
6. Core packages are guarded from platform-framework imports.
7. CI installs the platform dependency group.
8. ADR-003 documents dependency and trust boundaries.
9. M5 and M6 security responsibilities are explicitly separated.
10. Focused tests, Ruff, mypy, and `git diff --check` pass.
11. Full repository regression passes before M5-A is committed.

## M5 non-goals

M5 does not claim:

- production IAM maturity;
- complete RBAC;
- complete PII governance;
- universal distributed execution;
- Redis necessity;
- zero-downtime migrations;
- disaster-recovery maturity;
- production SLO compliance;
- penetration-test certification;
- production security certification.

Those require later M6/M7 evidence.
