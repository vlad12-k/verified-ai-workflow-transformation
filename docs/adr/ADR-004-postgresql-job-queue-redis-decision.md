# ADR-004: PostgreSQL Job Queue and Redis Decision

## Status

Accepted

## Date

2026-09-20

## Context

M5-F introduced durable background execution for VAIT using PostgreSQL as
the canonical source of truth.

The platform now provides:

- durable job state in PostgreSQL;
- explicit job lifecycle states;
- atomic queue claiming with `SELECT ... FOR UPDATE SKIP LOCKED`;
- worker ownership;
- bounded attempts;
- heartbeat and lease expiry;
- retry and terminal failure handling;
- expired-lease recovery;
- thin worker orchestration that commits `RUNNING` before executing work.

M5-G evaluates whether Redis should be added as a second queue or
coordination dependency.

Adding Redis without a demonstrated platform requirement would introduce:

- another runtime service;
- another operational failure mode;
- additional deployment and monitoring requirements;
- potential consistency boundaries between PostgreSQL durable state and
  Redis queue state;
- additional recovery semantics;
- additional development and CI complexity.

The decision therefore must be based on measured evidence rather than
technology-stack convention.

## Decision

Redis will not be introduced at M5.

PostgreSQL remains the canonical durable queue and job-state store.

Workers continue to claim eligible jobs through PostgreSQL using
`FOR UPDATE SKIP LOCKED`.

This is a NO-GO decision for Redis at the current platform scale, not a
permanent prohibition on Redis.

## Evidence

A local M5-G queue baseline was executed against PostgreSQL 17 using the
real M5-F worker path:

`PENDING -> RUNNING -> executor -> SUCCEEDED`

The benchmark used:

- 500 durable jobs;
- 8 concurrent workers;
- the production `WorkerService`;
- the production PostgreSQL `JobRepository`;
- `FOR UPDATE SKIP LOCKED`;
- a no-op executor so that the measurement primarily represented platform
  queue and persistence overhead.

Observed result:

| Metric | Result |
| --- | ---: |
| Configured jobs | 500 |
| Claims | 500 |
| Unique claims | 500 |
| Duplicate claims | 0 |
| Successful jobs | 500 |
| Elapsed time | 0.979 s |
| Throughput | 510.52 jobs/s |
| Worker-cycle p50 | 15.12 ms |
| Worker-cycle p95 | 19.19 ms |
| Worker-cycle maximum | 34.59 ms |

The benchmark completed successfully with no missing or duplicate claims.

These measurements are a local development baseline, not a production SLO,
capacity guarantee, or proof of PostgreSQL's maximum queue throughput.
They are sufficient for the M5-G architectural question because no current
Redis-specific bottleneck has been demonstrated.

## Rationale

PostgreSQL already owns the durable experiment, evidence, artifact metadata,
and job state required by the platform.

Keeping queue coordination in the same durable system currently provides:

- one authoritative source of job truth;
- transactional state transitions;
- deterministic recovery after worker failure;
- no dual-write requirement between queue and database;
- fewer runtime dependencies;
- simpler local development and deployment;
- simpler operational recovery.

The measured baseline also demonstrates substantial headroom relative to the
current VAIT workload while preserving correctness under concurrent workers.

Introducing Redis now would therefore increase system complexity without
solving an observed problem.

## Redis Reconsideration Triggers

The decision should be revisited if measured evidence demonstrates one or
more of the following:

1. Sustained PostgreSQL queue-lock contention materially increases job
   acquisition latency.
2. Queue backlog grows despite available worker capacity.
3. Queue workload materially degrades the primary PostgreSQL persistence
   workload.
4. Required worker throughput or latency cannot be achieved economically
   through PostgreSQL queue tuning or worker scaling.
5. The platform requires queue semantics that are not cleanly provided by
   the current PostgreSQL design, such as high-volume pub/sub, fan-out,
   distributed rate coordination, or ephemeral event distribution.
6. Production measurements demonstrate that separating queue coordination
   from durable relational state provides a clear reliability or performance
   benefit.

Any future Redis adoption must be justified by a new benchmark and an
explicit consistency model between Redis and PostgreSQL.

## Consequences

### Positive

- No new runtime dependency is introduced in M5.
- PostgreSQL remains the single durable source of truth.
- Existing M5-F correctness and recovery semantics remain unchanged.
- Deployment and operational complexity remain lower.
- Future Redis adoption remains possible when supported by evidence.

### Negative

- PostgreSQL continues to handle both durable application data and queue
  coordination.
- Very large future queue workloads may require database tuning, partitioning,
  or a dedicated queue technology.
- Pub/sub and other Redis-native coordination patterns are not introduced.

## Follow-up

M5-H may add observability around queue acquisition latency, execution
duration, retries, lease expiry, and backlog depth.

Those measurements can provide production evidence for revisiting this ADR
if the current assumptions stop holding.
