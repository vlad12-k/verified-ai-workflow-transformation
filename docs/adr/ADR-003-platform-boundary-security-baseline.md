# ADR-003: Platform Boundary and Security Baseline

## Status

Accepted.

## Context

VAIT already has a provider-neutral verification, evidence, inference,
optimisation, and recommendation core.

M5 introduces the platform layer required to expose and operate that core
through APIs, durable persistence, background execution, observability,
and deployment environments.

Platform infrastructure must not become part of VAIT's verification or
optimisation semantics.

Security must also be designed into these boundaries during M5 rather than
retrofitted after the platform is built.

M5 establishes secure architectural boundaries.

M6 performs deeper security, reliability, failure, load, and adversarial
hardening against those boundaries.

## Decision

VAIT uses the following dependency direction:

```text
FastAPI / PostgreSQL / workers / Redis / OpenTelemetry
                         |
                         v
                  platform layer
                         |
                         v
                     VAIT core
```

The VAIT core must never depend upward on the platform layer.

Core packages must not directly depend on:

- FastAPI;
- SQLAlchemy;
- Alembic;
- Redis;
- OpenTelemetry;
- `vait.platform`.

Platform code may depend on and orchestrate the core through stable
interfaces.

## Trust boundaries

M5 recognises these initial trust boundaries:

```text
external client
    |
    v
FastAPI boundary
    |
    +------> PostgreSQL
    |
    +------> job dispatch
                 |
                 v
              worker
                 |
                 +------> VAIT core
                 |
                 +------> model/provider adapters
                 |
                 +------> evidence/artifact storage

platform
    |
    +------> logs / traces / metrics
```

Data crossing a trust boundary must be validated, minimised, and handled
according to its sensitivity.

## PostgreSQL responsibility

PostgreSQL is the durable source of truth for platform state.

Durable experiment, job, evidence, and recommendation state must not exist
only inside an ephemeral queue or cache.

Persistence will be exposed behind explicit platform repository
interfaces.

Database schema evolution will use migrations from the beginning of
PostgreSQL integration.

## Background execution responsibility

Long-running model training, evaluation, benchmarking, inference, and
verification must not depend on a long-lived HTTP request.

Background jobs will have explicit durable lifecycle state.

The initial implementation may use PostgreSQL-backed job claiming.

A queue implementation must not become the canonical source of experiment
or evidence state.

## Redis decision

Redis is optional.

Redis must not be introduced solely to add another technology to the
project.

M5 will evaluate whether measured requirements justify Redis for one or
more of:

- multiple distributed workers;
- shared queueing;
- delayed execution;
- back-pressure;
- distributed coordination;
- short-lived shared caching.

The decision may legitimately be either:

```text
REDIS_REQUIRED
```

or:

```text
REDIS_NOT_JUSTIFIED
```

PostgreSQL remains the durable source of truth in either case.

## Observability boundary

Observability is separated into:

- structured application logging;
- OpenTelemetry tracing;
- OpenTelemetry metrics.

Correlation identifiers must allow an operation to be followed across:

```text
HTTP request
    -> experiment
    -> job
    -> worker
    -> VAIT core
    -> provider/model
    -> persistence
```

Secrets, credentials, authentication material, and sensitive payloads must
not be emitted into logs, traces, metrics, evidence, or artifacts.

## M5 security baseline

M5 must establish at least these properties:

- secrets are never committed to source control;
- unsafe debug configuration fails closed in protected environments;
- external inputs are validated before entering the core;
- service configuration is explicit and environment-scoped;
- database access is designed for least privilege;
- durable evidence preserves provenance and integrity metadata;
- telemetry is designed for redaction;
- containers will run without unnecessary privilege;
- CI permissions remain least privilege;
- platform dependencies cannot leak into provider-neutral core packages.

## M5 and M6 boundary

M5 is responsible for:

```text
secure-by-design architecture
typed configuration
validated service boundaries
durable state
basic idempotency contracts
safe telemetry boundaries
least-privilege defaults
environment separation
```

M6 is responsible for deeper hardening, including:

```text
authentication and authorisation
PII governance
stronger secrets management and rotation
rate limiting and quotas
SSRF and hostile-input testing
retry and idempotency failure testing
dependency scanning
container scanning
SAST
SBOM
backup and restore
migration rollback strategy
load and soak testing
failure injection
chaos testing
SLOs
operational runbooks
formal threat modelling
incident and recovery exercises
```

The intended relationship is:

```text
M5 builds secure platform boundaries.
M6 attacks, validates, and hardens those boundaries.
```

## Consequences

Benefits:

- verification semantics remain independent of infrastructure frameworks;
- platform infrastructure can evolve without contaminating the core;
- PostgreSQL remains the canonical durable state;
- Redis remains optional until justified by evidence;
- security decisions are made before public service interfaces are fixed;
- M6 receives explicit trust boundaries to attack and harden.

Costs:

- additional repository and adapter boundaries are required;
- background execution requires explicit lifecycle semantics;
- idempotency must be designed rather than assumed;
- security validation becomes cross-cutting throughout M5.

## Rejected alternatives

### Platform framework imports inside the core

Rejected because infrastructure concerns would become coupled to
verification and optimisation semantics.

### Redis as the initial source of truth

Rejected because ephemeral queue or cache infrastructure must not own
canonical experiment or evidence state.

### Security deferred entirely to M6

Rejected because API, database, worker, telemetry, and deployment
boundaries would already have been structurally committed.

### Long-running work inside HTTP requests

Rejected because training, benchmarks, inference, and verification may
exceed safe request lifetimes and require durable asynchronous execution.
