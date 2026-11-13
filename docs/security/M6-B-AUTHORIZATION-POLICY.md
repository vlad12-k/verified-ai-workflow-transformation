# M6-B — Platform Authorisation Policy

**Status:** Implemented
**Milestone:** M6-B2
**Threat owner:** T-02
**Parent plan:** `docs/M6-PRODUCTION-HARDENING.md`

## 1. Purpose

This document defines the authorisation policy applied after the M6-B
authentication boundary.

Authentication establishes identity.

Authorisation determines whether that authenticated identity may perform a
specific protected platform operation.

The security rule is:

    authenticate
        ->
    establish trusted principal
        ->
    evaluate explicit permission
        ->
    ALLOW or DENY

Authentication alone never grants unrestricted platform authority.

The default outcome for an unknown role, unknown permission relationship, or
missing permission is denial.

## 2. Architectural boundary

The implemented request boundary is:

    HTTP request
        ->
    service-token authentication
        ->
    AuthenticatedPrincipal
        ->
    explicit permission dependency
        ->
    protected operation

The authenticated principal contains:

- trusted subject;
- authentication method;
- trusted platform role.

The role is loaded from trusted platform configuration.

It is not accepted from:

- request headers;
- query parameters;
- request bodies;
- cookies;
- provider responses.

Client-controlled role escalation is therefore outside the trusted input
boundary.

## 3. Roles

The current role model is deliberately small.

### viewer

Read-only platform consumer.

Granted permissions:

- `experiment:read`
- `job:read`
- `evidence:read`

### operator

Operational service capable of initiating and managing ordinary platform
work.

Granted permissions:

- `experiment:read`
- `experiment:run`
- `job:read`
- `job:cancel`
- `evidence:read`

### admin

Privileged operational identity.

Granted permissions:

- every declared `PlatformPermission`

Administrative authority remains explicit and is not implied by successful
authentication.

## 4. Permission model

The implemented permission vocabulary is:

| Permission | Intended protected capability |
| --- | --- |
| `experiment:read` | Read experiment identity and non-secret experiment state |
| `experiment:run` | Initiate an experiment/workflow execution path |
| `job:read` | Read durable job state |
| `job:cancel` | Request cancellation of a cancellable durable job |
| `evidence:read` | Read authorised evidence and artifact metadata |
| `platform:admin` | Perform explicitly administrative platform operations |

Permission names describe authority.

They do not imply that a corresponding public HTTP endpoint already exists.

## 5. Current API exposure

The current production-shaped HTTP API exposes only public platform surfaces:

- liveness;
- readiness;
- non-sensitive service metadata;
- generated API/OpenAPI documentation.

No production experiment, job, evidence, or administrative HTTP workflow
route is currently exposed.

The M6-B authorisation implementation therefore establishes the reusable
security boundary before those protected workflow routes are added later.

Test-only routes are used solely to prove enforcement semantics.

They are not production API routes.

## 6. Future protected operation policy

When workflow HTTP routes are introduced, the following policy applies.

| Operation class | Required permission |
| --- | --- |
| Read experiment | `experiment:read` |
| Submit/run experiment | `experiment:run` |
| Read job status | `job:read` |
| Cancel job | `job:cancel` |
| Read verification/evaluation evidence | `evidence:read` |
| Read artifact metadata associated with authorised evidence | `evidence:read` |
| Administrative platform operation | `platform:admin` |

Every protected route must declare its required permission explicitly.

A route must not rely merely on the presence of an authenticated principal.

## 7. Internal worker operations

Worker-internal lifecycle operations are not external RBAC capabilities.

Examples include:

- claim job;
- heartbeat;
- persist success;
- persist failure/retry;
- requeue retry;
- recover expired leases.

These operations belong to the trusted worker/persistence boundary.

They must not become externally callable merely because an external role has
a broad permission.

Future worker-service hardening remains owned by M6-D.

## 8. Registry boundary

The platform registry currently persists:

- experiments;
- candidates;
- evidence;
- artifact metadata.

Repository methods are internal persistence primitives.

Repository `add()` or `get()` methods are not themselves an external
authorisation boundary.

Authorisation must occur at the platform orchestration/API boundary before a
request reaches privileged persistence operations.

This preserves dependency direction:

    external request
        ->
    vait.platform security/orchestration
        ->
    persistence repository

Persistence repositories must not import FastAPI authentication or
authorisation concerns.

## 9. Default-deny rules

The following conditions deny access:

- authentication is absent;
- authentication fails;
- authentication mode is disabled for a protected dependency;
- role is unknown;
- required permission is absent;
- client attempts to supply an elevated role;
- a future operation is not mapped to an explicit permission.

Authentication failure returns:

    HTTP 401
    authentication_required

Authorisation failure returns:

    HTTP 403
    permission_denied

The two states remain semantically distinct.

## 10. Public-route rule

Authorisation is dependency-based rather than blanket middleware.

The following remain intentionally public:

- health probes;
- non-sensitive service metadata.

Public infrastructure surfaces must not accidentally inherit protected-route
dependencies.

Future public endpoints require explicit security review before exposure.

## 11. Security properties established

M6-B2 establishes the following controls:

- authentication and authorisation are separated;
- roles originate only from trusted configuration;
- permissions are explicit;
- viewer is least-privilege;
- unknown roles receive no permissions;
- protected operations fail closed;
- `401` and `403` have distinct typed API contracts;
- request headers cannot escalate platform role;
- administrative permission is explicit;
- health/meta remain outside blanket RBAC;
- external RBAC does not grant worker-internal lifecycle authority.

## 12. Threat T-02 closure

Threat:

    T-02 — Authorised identity performs an unauthorised operation

Required controls from M6-A:

- explicit authorisation policy;
- role/resource mapping;
- default-deny checks;
- negative authorisation tests.

Implemented evidence:

- `PlatformRole`;
- `PlatformPermission`;
- explicit role-to-permission mapping;
- `require_permission(...)`;
- fail-closed unknown-role handling;
- typed `403 permission_denied`;
- privilege-escalation negative tests;
- public-route isolation tests;
- 100% focused branch coverage across authentication, authorisation,
  settings, and typed API error handling.

Full regression after implementation:

    653 passed
    0 failures
    0 errors
    0 skipped

T-02 may therefore move from:

    OPEN
        ->
    CONTROL IMPLEMENTED
        ->
    CONTROL TESTED

Final milestone-level security closure remains subject to the combined M6-B
security regression in M6-B5.

## 13. Non-goals

M6-B2 does not introduce:

- OAuth;
- JWT;
- user-account management;
- tenant isolation;
- arbitrary policy languages;
- attribute-based access control;
- external identity-provider integration.

Those mechanisms are not currently justified by the platform architecture.

The service-token boundary remains the deliberately minimal production-shaped
foundation for M6.
