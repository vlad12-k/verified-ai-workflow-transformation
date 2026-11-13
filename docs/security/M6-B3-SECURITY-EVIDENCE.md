# M6-B3 — Sensitive Data and Telemetry Security Evidence

**Status:** Completed
**Milestone:** M6-B3
**Threats addressed:** T-05, T-06, T-15
**Parent plan:** `docs/M6-PRODUCTION-HARDENING.md`
**Policy:** `docs/security/M6-B-DATA-HANDLING-POLICY.md`

## 1. Scope

M6-B3 establishes and tests defensive handling for sensitive material across:

- structured logging;
- OpenTelemetry tracing;
- public API error handling;
- authentication failures;
- authorisation failures;
- durable evidence payloads;
- validation-error representations.

M6-B3 does not claim complete closure of T-05, T-06, or T-15.

Residual ownership remains in later milestones where the relevant capability
does not yet exist.

## 2. Implemented controls

### 2.1 Telemetry redaction

`vait.platform.security.redaction` recognises credential-bearing telemetry
patterns including:

- authorization credentials;
- API keys;
- access, refresh, auth, and generic tokens;
- passwords;
- secrets;
- client secrets;
- database URLs;
- connection strings;
- Bearer and Basic credentials;
- credential-bearing URI userinfo.

Recognised values are replaced with the fixed marker:

    [REDACTED]

Safe operational fields such as `token_count`, `token_budget`,
`password_policy`, and `secret_rotation` are preserved.

### 2.2 Structured logging

Structured platform logging:

- serialises an allow-listed envelope;
- does not serialise arbitrary `extra` fields;
- redacts credential material from formatted log messages;
- records exception type without exception message or traceback;
- preserves request, correlation, experiment, job, worker, trace, and span
  identifiers where authorised.

### 2.3 OpenTelemetry tracing

Hostile regression testing discovered that the default OpenTelemetry
`start_as_current_span()` behaviour recorded exception content in:

- `exception.message`;
- `exception.stacktrace`;
- exception-derived span status description.

A deliberately injected provider secret was observable in exported span
telemetry.

The tracing boundary was hardened so that platform spans:

- disable automatic exception recording;
- disable automatic exception-derived status descriptions;
- retain `ERROR` status for failed operations;
- use the constant status description `operation failed`;
- continue propagating the original application exception;
- continue resetting log context correctly.

The regression test proves that the injected secret and traceback are absent
from exported span telemetry.

## 3. Durable evidence protection

Durable evidence is not silently redacted.

`EvidenceRecord` validates payloads before persistence and rejects recognised
secret-bearing structures.

The guard recursively covers nested mappings and lists and recognises both:

- credential-bearing field names;
- unambiguous credential material embedded inside string values.

Valid evidence is returned unchanged.

This preserves the distinction between:

    telemetry
        -> redact defensively

and:

    durable evidence
        -> reject prohibited secret material

Pydantic registry models use `hide_input_in_errors=True` so rejected secret
values are not reproduced by the normal text representation of validation
errors.

## 4. Public error boundary

Security regression tests confirm that:

- invalid service tokens are not reflected in HTTP responses;
- configured service tokens are not reflected in HTTP responses;
- HTTP exception details are replaced by stable public error contracts;
- validation errors omit rejected raw input;
- authentication failures remain distinct from authorisation failures;
- 401, 403, and 422 responses do not expose tested sensitive values.

## 5. Hostile regression evidence

The M6-B3.4 hostile leakage regression covered:

- telemetry redaction;
- structured logging;
- exception logging;
- OpenTelemetry tracing;
- API tracing;
- request observability;
- API error sanitisation;
- authentication;
- authorisation;
- durable evidence validation.

Result:

    109 passed

Full repository regression after the hostile tracing fix:

    tests:    719
    passed:   719
    failures: 0
    errors:   0
    skipped:  0

Repository-wide branch-aware coverage:

    89.99%

Security-critical modules measured at:

    100.00%  src/vait/platform/observability/tracing.py
    100.00%  src/vait/platform/security/redaction.py
    100.00%  src/vait/platform/registry/sensitive_data.py
    100.00%  src/vait/platform/registry/contracts.py
    100.00%  src/vait/platform/security/authentication.py
    100.00%  src/vait/platform/security/authorization.py
    100.00%  src/vait/platform/api/errors.py
    100.00%  src/vait/platform/settings.py

The four Alembic migration files reported as zero coverage by the repo-wide
`--cov=vait` measurement remain a known coverage-accounting issue associated
with their dynamic migration loading. Their focused migration tests are
separate from this M6-B3 security closure.

## 6. Threat-state evidence

### T-05 — Secret leakage

M6-B3 status:

    CONTROL TESTED for the implemented local boundaries

Tested controls now cover:

- logs;
- traces;
- API errors;
- authentication failures;
- durable evidence rejection;
- validation-error text.

T-05 is not declared fully `CONTROLLED`.

Residual ownership remains in:

- M6-H for metrics and mature production observability;
- M6-J for live provider credentials and live external-provider paths.

### T-06 — Sensitive-data or PII disclosure

M6-B3 status:

    CONTROL TESTED for the implemented persistence and telemetry boundaries

Implemented evidence includes:

- explicit data classification policy;
- persistence minimisation rules;
- fail-closed secret-bearing evidence validation;
- telemetry exclusion/redaction rules.

T-06 is not declared fully `CONTROLLED`.

Residual ownership remains in M6-J for external-provider data transmission
and live-provider handling.

M6-B3 does not claim complete automated PII detection.

### T-15 — Telemetry leakage or cardinality explosion

M6-B3 status:

    CONTROL TESTED for logging and tracing leakage

Implemented evidence includes:

- allow-listed structured logging;
- credential redaction;
- exception-message suppression;
- OpenTelemetry exception-content suppression;
- safe error status preservation;
- hostile leakage regression.

T-15 is not declared fully `CONTROLLED`.

Residual ownership remains in M6-H for:

- OpenTelemetry metrics;
- metric label policy;
- bounded-cardinality tests;
- production telemetry maturity.

## 7. M6-B3 closure decision

The following M6-B3 slices are complete:

    M6-B3.1  data classification and handling policy
    M6-B3.2  central telemetry redaction
    M6-B3.3  durable evidence sensitive-data guard
    M6-B3.4  hostile leakage regression and evidence

Therefore:

    M6-B3 — COMPLETE

This closure does not change the remaining M6-B roadmap.

The next security implementation slice is:

    M6-B4 — provider egress and SSRF controls
