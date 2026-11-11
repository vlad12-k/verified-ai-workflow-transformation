"""Provider-usage budget policies and platform-neutral contracts."""

import re
from decimal import Decimal
from enum import StrEnum
from typing import Protocol, Self, runtime_checkable
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)


class BudgetContractModel(BaseModel):
    """Immutable, strict base for budget policy contracts."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        hide_input_in_errors=True,
    )


class BudgetScopeKind(StrEnum):
    """Trusted identity or durable execution context owning a budget."""

    PRINCIPAL = "principal"
    EXPERIMENT = "experiment"
    JOB = "job"


class BudgetScope(BudgetContractModel):
    """One explicit budget owner, built from trusted platform context."""

    kind: BudgetScopeKind
    scope_id: str = Field(min_length=1, max_length=255)

    @model_validator(mode="after")
    def validate_scope_identifier(self) -> Self:
        """Require configured principal IDs or canonical durable UUIDs."""
        if self.kind is BudgetScopeKind.PRINCIPAL:
            if (
                len(self.scope_id) > 128
                or re.fullmatch(r"[A-Za-z0-9._:-]+", self.scope_id) is None
            ):
                raise ValueError("principal scope_id is invalid")
            return self

        try:
            parsed_id = UUID(self.scope_id)
        except ValueError as exc:
            raise ValueError("experiment and job scope IDs must be UUIDs") from exc

        if str(parsed_id) != self.scope_id:
            raise ValueError("experiment and job scope IDs must be canonical UUIDs")

        return self


class BudgetWindowKind(StrEnum):
    """Reset semantics for request, token, and spend limits."""

    PER_EXECUTION = "per_execution"
    LIFETIME = "lifetime"
    FIXED_WINDOW = "fixed_window"


class BudgetWindow(BudgetContractModel):
    """Explicit accounting period; fixed windows are UTC-aligned in C2."""

    kind: BudgetWindowKind
    interval_seconds: int | None = Field(default=None, gt=0, strict=True)

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        """Require an interval only for fixed-window policies."""
        if self.kind is BudgetWindowKind.FIXED_WINDOW:
            if self.interval_seconds is None:
                raise ValueError("fixed-window policies require interval_seconds")
        elif self.interval_seconds is not None:
            raise ValueError("only fixed-window policies may set interval_seconds")

        return self


class ProviderRequestLimit(BudgetContractModel):
    """Maximum logical provider operations in the policy window."""

    max_requests: int = Field(gt=0, strict=True)


class ProviderTokenLimit(BudgetContractModel):
    """Maximum provider tokens in the policy window."""

    max_tokens: int = Field(gt=0, strict=True)


class RetryBudget(BudgetContractModel):
    """Maximum additional actual provider calls; zero means no retries."""

    max_retries: int = Field(ge=0, strict=True)


class ProviderAttemptBudget(BudgetContractModel):
    """Maximum actual provider calls, including initial attempts."""

    max_attempts: int = Field(gt=0, strict=True)


def _validate_attempt_counts(
    provider_requests: int,
    provider_attempts: int,
    retries: int,
) -> None:
    """Require one initial call per logical operation plus every retry."""
    if provider_attempts != provider_requests + retries:
        raise ValueError(
            "provider_attempts must equal provider_requests plus retries"
        )


class PriceSourceReference(BudgetContractModel):
    """Versioned pricing-source reference for trusted approval resolution.

    The reference alone does not establish approval. A trusted policy resolver
    must confirm its approval before the policy is used.
    """

    source_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
    )
    source_version: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
    )
    approval_reference: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
    )


class MonetarySpendLimit(BudgetContractModel):
    """Optional USD ceiling tied to a versioned price-source reference."""

    max_cost_usd: Decimal = Field(gt=0)
    price_source: PriceSourceReference

    @field_validator("max_cost_usd", mode="before")
    @classmethod
    def require_exact_decimal_input(cls, value: object) -> object:
        """Reject binary floats and booleans at the monetary boundary."""
        if isinstance(value, (bool, float)):
            raise ValueError("max_cost_usd must be a Decimal, integer, or decimal string")
        return value

    @field_validator("max_cost_usd")
    @classmethod
    def require_finite_amount(cls, value: Decimal) -> Decimal:
        """Reject NaN and infinite monetary limits."""
        if not value.is_finite():
            raise ValueError("max_cost_usd must be finite")
        return value


class BudgetPolicy(BudgetContractModel):
    """Explicit versioned provider budget for one trusted scope."""

    policy_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
    )
    policy_version: int = Field(gt=0, strict=True)
    scope: BudgetScope
    window: BudgetWindow
    request_limit: ProviderRequestLimit
    token_limit: ProviderTokenLimit
    retry_budget: RetryBudget
    provider_attempt_budget: ProviderAttemptBudget
    monetary_limit: MonetarySpendLimit | None = None

    @model_validator(mode="after")
    def validate_attempt_budgets(self) -> Self:
        """Ensure retry allowance fits within the total attempt ceiling."""
        if (
            self.retry_budget.max_retries
            >= self.provider_attempt_budget.max_attempts
        ):
            raise ValueError(
                "max_retries must be less than max_provider_attempts"
            )

        return self


class BudgetEstimate(BudgetContractModel):
    """Conservative pre-dispatch upper bound for one provider operation.

    ``provider_requests`` counts logical provider operations; a retry does not
    increment request units. ``provider_attempts`` counts actual provider
    calls, including initial attempts and retries.

    Token and spend estimates must include provenance. An explicit zero is a
    known estimate; ``None`` is unknown and is rejected by enforced limits.
    """

    provider_requests: int = Field(
        gt=0,
        strict=True,
        description=(
            "Logical provider operations; retries do not increment request units."
        ),
    )
    provider_attempts: int = Field(
        gt=0,
        strict=True,
        description=(
            "Actual provider calls, including initial attempts and retries."
        ),
    )
    retries: int = Field(ge=0, strict=True)
    provider_tokens: int | None = Field(default=None, ge=0, strict=True)
    token_estimator_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
    )
    token_estimator_version: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
    )
    estimated_spend_usd: Decimal | None = Field(default=None, ge=0)
    price_source: PriceSourceReference | None = None
    spend_estimator_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
    )
    spend_estimator_version: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
    )

    @field_validator("estimated_spend_usd", mode="before")
    @classmethod
    def require_exact_estimate_input(cls, value: object) -> object:
        """Reject floats and booleans for estimated monetary amounts."""
        if isinstance(value, (bool, float)):
            raise ValueError(
                "estimated_spend_usd must be a Decimal, integer, or decimal string"
            )
        return value

    @field_validator("estimated_spend_usd")
    @classmethod
    def require_finite_estimate(cls, value: Decimal | None) -> Decimal | None:
        """Reject non-finite spend estimates."""
        if value is not None and not value.is_finite():
            raise ValueError("estimated_spend_usd must be finite")
        return value

    @model_validator(mode="after")
    def validate_estimate_provenance(self) -> Self:
        """Require provenance for measured upper bounds, never zero-fill."""
        _validate_attempt_counts(
            self.provider_requests,
            self.provider_attempts,
            self.retries,
        )

        token_provenance = (
            self.token_estimator_id,
            self.token_estimator_version,
        )
        if self.provider_tokens is None and any(
            value is not None for value in token_provenance
        ):
            raise ValueError(
                "token estimator provenance requires a token estimate"
            )
        if self.provider_tokens is not None and any(
            value is None for value in token_provenance
        ):
            raise ValueError(
                "token estimates require estimator identity and version"
            )

        spend_provenance = (
            self.price_source,
            self.spend_estimator_id,
            self.spend_estimator_version,
        )
        if self.estimated_spend_usd is None and any(
            value is not None for value in spend_provenance
        ):
            raise ValueError("spend provenance requires a spend estimate")
        if self.estimated_spend_usd is not None and any(
            value is None for value in spend_provenance
        ):
            raise ValueError(
                "spend estimates require price-source and estimator provenance"
            )

        return self


class BudgetConsumption(BudgetContractModel):
    """Actual usage; requests are logical operations, attempts are actual calls.

    ``None`` means unknown, never zero.
    """

    provider_requests: int = Field(
        ge=0,
        strict=True,
        description=(
            "Logical provider operations; retries do not increment request units."
        ),
    )
    provider_attempts: int = Field(
        ge=0,
        strict=True,
        description=(
            "Actual provider calls, including initial attempts and retries."
        ),
    )
    retries: int = Field(ge=0, strict=True)
    provider_tokens: int | None = Field(default=None, ge=0, strict=True)
    spend_usd: Decimal | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_attempt_counts(self) -> Self:
        """Keep request, retry and actual-call totals consistent."""
        _validate_attempt_counts(
            self.provider_requests,
            self.provider_attempts,
            self.retries,
        )
        return self

    @field_validator("spend_usd", mode="before")
    @classmethod
    def require_exact_consumption_input(cls, value: object) -> object:
        """Reject binary floats and booleans for actual spend."""
        if isinstance(value, (bool, float)):
            raise ValueError("spend_usd must be a Decimal, integer, or decimal string")
        return value

    @field_validator("spend_usd")
    @classmethod
    def require_finite_consumption(cls, value: Decimal | None) -> Decimal | None:
        """Reject non-finite actual spend."""
        if value is not None and not value.is_finite():
            raise ValueError("spend_usd must be finite")
        return value


class BudgetReservationAmounts(BudgetContractModel):
    """Amounts held by active reservations for one budget scope.

    Requests count logical provider operations; attempts count actual calls.

    ``spend_usd=None`` means reserved spend is unknown or unpriced, not zero.
    """

    provider_requests: int = Field(
        ge=0,
        strict=True,
        description=(
            "Logical provider operations; retries do not increment request units."
        ),
    )
    provider_attempts: int = Field(
        ge=0,
        strict=True,
        description=(
            "Actual provider calls, including initial attempts and retries."
        ),
    )
    retries: int = Field(ge=0, strict=True)
    provider_tokens: int = Field(ge=0, strict=True)
    spend_usd: Decimal | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_attempt_counts(self) -> Self:
        """Keep request, retry and actual-call totals consistent."""
        _validate_attempt_counts(
            self.provider_requests,
            self.provider_attempts,
            self.retries,
        )
        return self

    @field_validator("spend_usd", mode="before")
    @classmethod
    def require_exact_reserved_spend_input(cls, value: object) -> object:
        """Reject binary floats and booleans for reserved spend."""
        if isinstance(value, (bool, float)):
            raise ValueError("reserved spend must be a Decimal, integer, or decimal string")
        return value

    @field_validator("spend_usd")
    @classmethod
    def require_finite_reserved_spend(cls, value: Decimal | None) -> Decimal | None:
        """Reject non-finite reserved spend."""
        if value is not None and not value.is_finite():
            raise ValueError("reserved spend must be finite")
        return value


class BudgetUsageSnapshot(BudgetContractModel):
    """Consumed and reserved totals read for one exact budget scope."""

    consumed: BudgetConsumption
    reserved: BudgetReservationAmounts


class BudgetDenialCode(StrEnum):
    """Stable machine-readable budget-admission denial reasons."""

    POLICY_REQUIRED = "policy_required"
    SCOPE_MISMATCH = "scope_mismatch"
    REQUEST_LIMIT_EXCEEDED = "request_limit_exceeded"
    TOKEN_LIMIT_EXCEEDED = "token_limit_exceeded"
    TOKEN_ESTIMATE_UNAVAILABLE = "token_estimate_unavailable"
    TOKEN_CONSUMPTION_UNKNOWN = "token_consumption_unknown"
    SPEND_LIMIT_EXCEEDED = "spend_limit_exceeded"
    SPEND_ESTIMATE_UNAVAILABLE = "spend_estimate_unavailable"
    SPEND_CONSUMPTION_UNKNOWN = "spend_consumption_unknown"
    SPEND_RESERVATION_UNKNOWN = "spend_reservation_unknown"
    PRICE_SOURCE_MISMATCH = "price_source_mismatch"
    RETRY_BUDGET_EXCEEDED = "retry_budget_exceeded"
    PROVIDER_ATTEMPT_LIMIT_EXCEEDED = "provider_attempt_limit_exceeded"


class BudgetDecisionStatus(StrEnum):
    """Whether the requested provider operation fits its policy."""

    ALLOWED = "allowed"
    DENIED = "denied"


class BudgetDenial(BudgetContractModel):
    """One safe, typed reason a provider operation cannot proceed."""

    code: BudgetDenialCode


_BUDGET_EVALUATOR_ORIGIN = object()


class BudgetDecision(BudgetContractModel):
    """Pure evaluation result; denial is not a verification result."""

    status: BudgetDecisionStatus
    scope: BudgetScope
    policy_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
    )
    policy_version: int | None = Field(default=None, gt=0, strict=True)
    denials: tuple[BudgetDenial, ...] = ()
    _evaluation_origin: object | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_decision_consistency(self) -> Self:
        """Prevent contradictory allow and deny evidence."""
        if self.status is BudgetDecisionStatus.ALLOWED:
            if self.denials:
                raise ValueError("an allowed budget decision cannot contain denials")
            if self.policy_id is None:
                raise ValueError("an allowed budget decision requires a policy")
        if self.status is BudgetDecisionStatus.DENIED and not self.denials:
            raise ValueError("a denied budget decision requires a denial reason")
        if (self.policy_id is None) != (self.policy_version is None):
            raise ValueError("policy ID and version must be present together")
        policy_required = any(
            denial.code is BudgetDenialCode.POLICY_REQUIRED
            for denial in self.denials
        )
        if policy_required:
            if len(self.denials) != 1 or self.policy_id is not None:
                raise ValueError("a missing-policy denial cannot identify a policy")
        elif self.status is BudgetDecisionStatus.DENIED and self.policy_id is None:
            raise ValueError("a policy denial requires a policy identity")
        return self

    @property
    def is_allowed(self) -> bool:
        """Return whether budget policy admits the operation."""
        status = getattr(self, "status", None)
        return (
            status is BudgetDecisionStatus.ALLOWED
            and self._evaluation_origin
            == (
                _BUDGET_EVALUATOR_ORIGIN,
                status,
                getattr(self, "scope", None),
                getattr(self, "policy_id", None),
                getattr(self, "policy_version", None),
                getattr(self, "denials", ()),
            )
        )

    def require_allowed(self) -> None:
        """Raise a safe typed exception when admission was denied."""
        if not self.is_allowed:
            raise BudgetDeniedError(self)


class BudgetPolicyErrorCode(StrEnum):
    """Stable machine-readable budget-policy validation failures."""

    INVALID_POLICY = "invalid_policy"


class InvalidBudgetPolicyError(ValueError):
    """Raised when untrusted configuration cannot form a valid policy."""

    code = BudgetPolicyErrorCode.INVALID_POLICY

    def __init__(self) -> None:
        super().__init__("Budget policy is invalid.")


class BudgetDeniedError(RuntimeError):
    """Raised when a deterministic budget decision denies execution."""

    def __init__(self, decision: BudgetDecision) -> None:
        denials = getattr(decision, "denials", ())
        if (
            getattr(decision, "status", None) is not BudgetDecisionStatus.DENIED
            or not denials
        ):
            raise ValueError("BudgetDeniedError requires a denied decision with a denial reason")

        self.decision = decision
        self.code = denials[0].code
        super().__init__("Provider operation denied by budget policy.")


def load_budget_policy(value: object) -> BudgetPolicy:
    """Validate policy input without exposing rejected configuration values."""
    try:
        return BudgetPolicy.model_validate(value)
    except ValidationError:
        raise InvalidBudgetPolicyError() from None


def evaluate_budget(
    *,
    policy: BudgetPolicy | None,
    scope: BudgetScope,
    usage: BudgetUsageSnapshot,
    estimate: BudgetEstimate,
) -> BudgetDecision:
    """Evaluate a proposed reservation without I/O or mutable state.

    The caller must obtain ``usage`` and commit the resulting reservation
    atomically in the persistence adapter. This function alone is not
    concurrency-safe.
    """

    def mark_evaluated(decision: BudgetDecision) -> BudgetDecision:
        """Attach private provenance to decisions produced by this evaluator."""
        decision._evaluation_origin = (
            _BUDGET_EVALUATOR_ORIGIN,
            decision.status,
            decision.scope,
            decision.policy_id,
            decision.policy_version,
            decision.denials,
        )
        return decision

    if policy is None:
        return mark_evaluated(
            BudgetDecision(
                status=BudgetDecisionStatus.DENIED,
                scope=scope,
                denials=(BudgetDenial(code=BudgetDenialCode.POLICY_REQUIRED),),
            )
        )

    if policy.scope != scope:
        return mark_evaluated(
            BudgetDecision(
                status=BudgetDecisionStatus.DENIED,
                scope=scope,
                policy_id=policy.policy_id,
                policy_version=policy.policy_version,
                denials=(BudgetDenial(code=BudgetDenialCode.SCOPE_MISMATCH),),
            )
        )

    denials: list[BudgetDenial] = []

    if (
        usage.consumed.provider_requests
        + usage.reserved.provider_requests
        + estimate.provider_requests
        > policy.request_limit.max_requests
    ):
        denials.append(
            BudgetDenial(code=BudgetDenialCode.REQUEST_LIMIT_EXCEEDED)
        )

    if estimate.provider_tokens is None:
        denials.append(
            BudgetDenial(code=BudgetDenialCode.TOKEN_ESTIMATE_UNAVAILABLE)
        )
    elif usage.consumed.provider_tokens is None:
        denials.append(
            BudgetDenial(code=BudgetDenialCode.TOKEN_CONSUMPTION_UNKNOWN)
        )
    elif (
        usage.consumed.provider_tokens
        + usage.reserved.provider_tokens
        + estimate.provider_tokens
        > policy.token_limit.max_tokens
    ):
        denials.append(
            BudgetDenial(code=BudgetDenialCode.TOKEN_LIMIT_EXCEEDED)
        )

    if policy.monetary_limit is not None:
        if estimate.estimated_spend_usd is None:
            denials.append(
                BudgetDenial(code=BudgetDenialCode.SPEND_ESTIMATE_UNAVAILABLE)
            )
        elif estimate.price_source != policy.monetary_limit.price_source:
            denials.append(
                BudgetDenial(code=BudgetDenialCode.PRICE_SOURCE_MISMATCH)
            )

        if usage.consumed.spend_usd is None:
            denials.append(
                BudgetDenial(code=BudgetDenialCode.SPEND_CONSUMPTION_UNKNOWN)
            )
        if usage.reserved.spend_usd is None:
            denials.append(
                BudgetDenial(code=BudgetDenialCode.SPEND_RESERVATION_UNKNOWN)
            )
        elif (
            usage.consumed.spend_usd is not None
            and estimate.estimated_spend_usd is not None
            and usage.consumed.spend_usd
            + usage.reserved.spend_usd
            + estimate.estimated_spend_usd
            > policy.monetary_limit.max_cost_usd
        ):
            denials.append(
                BudgetDenial(code=BudgetDenialCode.SPEND_LIMIT_EXCEEDED)
            )

    if (
        usage.consumed.provider_attempts
        + usage.reserved.provider_attempts
        + estimate.provider_attempts
        > policy.provider_attempt_budget.max_attempts
    ):
        denials.append(
            BudgetDenial(
                code=BudgetDenialCode.PROVIDER_ATTEMPT_LIMIT_EXCEEDED
            )
        )

    if (
        usage.consumed.retries
        + usage.reserved.retries
        + estimate.retries
        > policy.retry_budget.max_retries
    ):
        denials.append(
            BudgetDenial(code=BudgetDenialCode.RETRY_BUDGET_EXCEEDED)
        )

    return mark_evaluated(
        BudgetDecision(
            status=(BudgetDecisionStatus.DENIED if denials else BudgetDecisionStatus.ALLOWED),
            scope=scope,
            policy_id=policy.policy_id,
            policy_version=policy.policy_version,
            denials=tuple(denials),
        )
    )


class BudgetReservationRequest(BudgetContractModel):
    """Idempotent request for a future atomic budget reservation."""

    reservation_id: UUID
    policies: tuple[BudgetPolicy, ...] = Field(min_length=1)
    estimate: BudgetEstimate

    @model_validator(mode="after")
    def validate_policy_scopes(self) -> Self:
        """Reject duplicate budget scopes in one atomic reservation."""
        scope_keys = {
            (policy.scope.kind, policy.scope.scope_id)
            for policy in self.policies
        }
        if len(scope_keys) != len(self.policies):
            raise ValueError("reservation policies must have unique scopes")
        return self


class BudgetReservationState(StrEnum):
    """Lifecycle states required by the next persistence slice."""

    RESERVED = "reserved"
    DISPATCHED = "dispatched"
    SETTLED = "settled"
    RELEASED = "released"


class BudgetPolicyReference(BudgetContractModel):
    """Immutable identity of one policy applied to a reservation."""

    policy_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
    )
    policy_version: int = Field(gt=0, strict=True)
    scope: BudgetScope


class BudgetReservation(BudgetContractModel):
    """Immutable handle returned for a persisted reservation."""

    reservation_id: UUID
    policies: tuple[BudgetPolicyReference, ...] = Field(min_length=1)
    amounts: BudgetReservationAmounts
    state: BudgetReservationState

    @model_validator(mode="after")
    def validate_policy_scopes(self) -> Self:
        """Keep the applied scope set explicit and duplicate-free."""
        scope_keys = {
            (policy.scope.kind, policy.scope.scope_id)
            for policy in self.policies
        }
        if len(scope_keys) != len(self.policies):
            raise ValueError("reservation policy scopes must be unique")
        return self


class BudgetSettlement(BudgetContractModel):
    """Actual or explicitly unknown consumption for one reservation."""

    reservation_id: UUID
    consumption: BudgetConsumption


@runtime_checkable
class BudgetReservationLedger(Protocol):
    """Persistence boundary for atomic reserve, dispatch and settlement.

    Implementations must evaluate and reserve all request policies atomically
    within one database transaction. Reusing a reservation ID with identical
    input must be idempotent; reusing it with different input must fail without
    changing any reservation.
    ``release`` is valid only before ``mark_dispatched``; an ambiguous outcome
    after dispatch must be settled with unknown consumption, not refunded.
    """

    def reserve(
        self,
        request: BudgetReservationRequest,
    ) -> BudgetReservation:
        """Atomically reserve all scopes or raise a typed budget denial."""
        ...

    def mark_dispatched(
        self,
        reservation_id: UUID,
    ) -> BudgetReservation:
        """Persist that external dispatch is about to begin."""
        ...

    def settle(
        self,
        settlement: BudgetSettlement,
    ) -> BudgetReservation:
        """Reconcile actual usage or explicit unknown consumption once."""
        ...

    def release(
        self,
        reservation_id: UUID,
    ) -> BudgetReservation:
        """Release a reservation known not to have been dispatched."""
        ...
