# M6-B — Sensitive Data and Redaction Policy

**Status:** Active
**Milestone:** M6-B3
**Threat owners:** T-05, T-06, T-15
**Parent plan:** `docs/M6-PRODUCTION-HARDENING.md`

## 1. Purpose

This document defines how VAIT classifies and handles data that may cross
platform observability, persistence, artifact, API, and external-provider
boundaries.

The policy applies before live external-provider execution is enabled.

The primary rule is:

    collect only what is required
        ->
    classify before crossing a boundary
        ->
    minimise
        ->
    redact or reject sensitive content
        ->
    preserve only authorised evidence

Redaction is a defensive control.

It is not a substitute for avoiding unnecessary collection or persistence.

## 2. Data classes

VAIT uses the following data classes.

### PUBLIC

Information intentionally safe for public exposure.

Examples:

- service name;
- API version;
- public health state;
- public documentation metadata.

May appear in:

- API responses;
- logs;
- traces;
- metrics;
- persisted evidence where relevant.

### INTERNAL

Operational identifiers that are not secrets but should not be exposed
without need.

Examples:

- request ID;
- correlation ID;
- experiment ID;
- job ID;
- worker ID;
- trace ID;
- span ID;
- runtime identity;
- source revision.

May appear in bounded observability context and durable provenance.

Must not be treated as authentication credentials.

### EVIDENCE_SENSITIVE

Inputs or outputs required to establish verification or benchmark evidence,
but which may contain business-sensitive content.

Examples:

- benchmark examples;
- provider outputs;
- model outputs;
- transformation evidence;
- evaluation payloads;
- detailed failure evidence.

Must be minimised before persistence.

Must not automatically enter logs, traces, or metrics.

### CONFIDENTIAL

Personal, customer, donor, private dataset, or otherwise protected data.

Examples may include:

- names;
- email addresses;
- telephone numbers;
- postal addresses;
- account identifiers;
- customer records;
- private benchmark content;
- donor or organisational records.

CONFIDENTIAL data must not enter telemetry.

Persistence requires an explicit schema or policy justification.

External-provider transmission requires a separately authorised provider
boundary.

### SECRET

Credential material or values that grant authority.

Examples:

- API keys;
- service tokens;
- bearer tokens;
- passwords;
- database passwords;
- connection strings containing credentials;
- provider credentials;
- private signing material;
- access tokens;
- refresh tokens.

SECRET values must never appear in:

- source control;
- logs;
- traces;
- metrics;
- public API responses;
- evidence payloads;
- artifacts intended as ordinary evidence;
- exception messages exposed outside the trusted process.

## 3. Boundary policy

| Boundary | PUBLIC | INTERNAL | EVIDENCE_SENSITIVE | CONFIDENTIAL | SECRET |
| --- | --- | --- | --- | --- | --- |
| Public API response | allowed | minimise | deny by default | deny | deny |
| Structured logs | allowed | allow-listed only | deny by default | deny | deny |
| Traces | allowed | allow-listed only | deny by default | deny | deny |
| Metrics | allowed | low-cardinality only | deny | deny | deny |
| PostgreSQL evidence | allowed | allowed where provenance requires | minimise and validate | explicit justification only | deny |
| Artifact storage | allowed | allowed | explicit evidence purpose | explicit justification only | deny |
| External provider | allowed | minimise | explicit request purpose | explicit authorisation required | credential only in transport boundary |

## 4. Telemetry rule

Telemetry is not an evidence store.

Logs, traces, and metrics must contain operational metadata only.

They must not contain:

- request bodies;
- provider prompts;
- provider responses;
- benchmark records;
- evidence payloads;
- authentication headers;
- service tokens;
- API keys;
- database URLs containing credentials;
- arbitrary exception messages.

Structured logs must continue to use an allow-listed envelope.

Redaction must also protect the textual event field because application code
may accidentally pass sensitive values through formatted log messages.

## 5. Logging policy

The structured logging boundary currently allows these base fields:

- timestamp;
- level;
- logger;
- event;
- allow-listed correlation context;
- exception type.

Arbitrary `logging.extra` values are not serialised.

Exception messages and tracebacks are not serialised.

M6-B3 must additionally ensure that sensitive values embedded in the textual
event cannot leave the process unredacted.

The required output for detected sensitive content is a stable redaction
marker rather than the original value.

## 6. Tracing policy

Current VAIT tracing records:

- service identity;
- deployment environment;
- controlled span names;
- trace ID;
- span ID.

Provider payloads, request bodies, evidence payloads, credentials, and
exception messages must not be attached as span attributes.

Future trace attributes require explicit allow-listing.

## 7. Metrics policy

Metrics are introduced in M6-H.

Metrics must use bounded-cardinality dimensions.

The following must never become metric labels:

- secrets;
- email addresses;
- request bodies;
- provider responses;
- evidence payloads;
- arbitrary user identifiers;
- arbitrary exception text.

M6-B3 establishes this policy before metrics are implemented.

## 8. API error policy

Public API errors must remain sanitised.

Authentication errors expose only:

    authentication_required

Authorisation errors expose only:

    permission_denied

Validation errors may expose field location and validation category but must
not echo secret field values.

Internal exception messages must not be reflected into public responses.

## 9. Evidence persistence policy

`EvidenceRecord.payload` is a durable evidence boundary.

It is not a general-purpose arbitrary-data store.

Before persistence, evidence must satisfy all of the following:

- no SECRET values;
- no credential-bearing headers;
- no database connection strings containing credentials;
- no private authentication material;
- only evidence required for verification, benchmark, provenance, or
  recommendation claims;
- CONFIDENTIAL content only when explicitly justified by the evidence schema
  and deployment policy.

M6-B3 will add a reusable validation boundary before sensitive evidence is
accepted for persistence.

## 10. Artifact policy

Artifacts may contain larger evidence bytes that do not belong in
PostgreSQL.

Artifact metadata must not contain secrets.

Artifact bytes containing CONFIDENTIAL information require an explicitly
approved artifact-store policy.

A digest proves integrity of bytes.

It does not make sensitive content safe to store.

## 11. External-provider policy

Provider credentials belong only to the provider transport/security boundary.

Credential material must not propagate into:

- provider-neutral request models;
- provider-neutral response models;
- verification evidence;
- logs;
- traces;
- metrics.

Provider input containing CONFIDENTIAL data must not be sent merely because
credentials are available.

Live-provider enablement remains fail-closed and is completed in M6-J.

## 12. Detection versus classification

Automated redaction can detect known sensitive patterns and known sensitive
field names.

It cannot prove that arbitrary text contains no confidential information.

Therefore VAIT uses two complementary controls:

    prevention / allow-listing
        +
    defensive redaction

The architecture must not claim that regex-based redaction alone provides
complete PII detection.

## 13. Required sensitive field names

At minimum, M6-B3 defensive controls must recognise credential-bearing names
such as:

- authorization;
- proxy-authorization;
- api_key;
- api-key;
- apikey;
- token;
- access_token;
- refresh_token;
- auth_token;
- password;
- passwd;
- secret;
- client_secret;
- database_url;
- connection_string.

Matching must be case-insensitive.

Nested structures must not bypass the guard.

## 14. Required redaction behaviour

For telemetry, recognised sensitive values must be replaced with a stable
non-secret marker:

    [REDACTED]

Redaction must not preserve:

- prefixes sufficient to recover the original secret;
- secret length where unnecessary;
- encoded credential content.

The original value must not appear anywhere in the serialised telemetry
output.

## 15. Fail-closed persistence behaviour

Durable evidence handling differs from telemetry.

Telemetry may redact defensively.

Evidence persistence must reject prohibited SECRET-bearing structures rather
than silently persisting modified evidence.

The caller must receive a deterministic validation failure before the
prohibited payload reaches PostgreSQL.

This distinction preserves evidence integrity:

    telemetry
        -> redact

    prohibited durable evidence
        -> reject

VAIT must not silently mutate scientific or verification evidence in order
to make it persistable.

## 16. Threat mapping

### T-05 — Secret leakage

Controls required:

- secret-aware telemetry redaction;
- secret-bearing evidence rejection;
- sanitised API errors;
- provider credential isolation;
- hostile leakage tests.

### T-06 — PII / sensitive-data disclosure

Controls required:

- explicit data classification;
- telemetry minimisation;
- evidence handling policy;
- provider-boundary review;
- no claim of universal automated PII detection.

### T-15 — Telemetry leakage / cardinality

Controls required:

- allow-listed log context;
- controlled span attributes;
- bounded metric dimensions;
- no arbitrary sensitive payloads in telemetry.

## 17. Acceptance criteria

M6-B3 is not complete until tests prove at least:

- a secret placed in `logging.extra` is absent from output;
- a secret formatted into a log event is absent from output;
- exception messages containing secrets are absent from output;
- bearer/API-key style values are redacted from log events;
- sensitive nested evidence fields are rejected;
- benign evidence remains unchanged;
- API authentication/authorisation errors do not echo credentials;
- existing request/job/trace correlation remains functional;
- public health/meta behaviour is unchanged.

## 18. Non-goals

M6-B3 does not claim:

- perfect PII detection;
- DLP certification;
- automated legal classification;
- GDPR compliance by redaction alone;
- protection against a fully compromised application process.

Those require broader organisational and deployment controls.
