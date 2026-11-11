"""Tests for deterministic provider-budget contracts."""

from decimal import Decimal
from uuid import UUID

import pytest
from pydantic import ValidationError

from vait.contracts.budgeting import (
    BudgetConsumption,
    BudgetDecision,
    BudgetDecisionStatus,
    BudgetDenial,
    BudgetDenialCode,
    BudgetDeniedError,
    BudgetEstimate,
    BudgetPolicy,
    BudgetPolicyReference,
    BudgetReservation,
    BudgetReservationAmounts,
    BudgetReservationRequest,
    BudgetReservationState,
    BudgetScope,
    BudgetScopeKind,
    BudgetUsageSnapshot,
    InvalidBudgetPolicyError,
    MonetarySpendLimit,
    PriceSourceReference,
    evaluate_budget,
    load_budget_policy,
)

_RESERVATION_ID = UUID("00000000-0000-0000-0000-000000000010")


def _policy_payload() -> dict[str, object]:
    """Return a complete, explicitly scoped policy fixture."""
    return {
        "policy_id": "service-provider-budget",
        "policy_version": 1,
        "scope": {
            "kind": "principal",
            "scope_id": "vait-service",
        },
        "window": {
            "kind": "fixed_window",
            "interval_seconds": 60,
        },
        "request_limit": {
            "max_requests": 10,
        },
        "token_limit": {
            "max_tokens": 1000,
        },
        "retry_budget": {
            "max_retries": 0,
        },
        "provider_attempt_budget": {
            "max_attempts": 100,
        },
    }


def test_policy_requires_explicit_scoped_limits() -> None:
    """A valid policy identifies its owner, version, window and limits."""
    policy = BudgetPolicy.model_validate(_policy_payload())

    assert policy.scope.kind is BudgetScopeKind.PRINCIPAL
    assert policy.policy_version == 1
    assert policy.request_limit.max_requests == 10
    assert policy.token_limit.max_tokens == 1000
    assert policy.retry_budget.max_retries == 0
    assert policy.provider_attempt_budget.max_attempts == 100


def test_policy_rejects_missing_required_limits() -> None:
    """Missing request or token limits must not mean unlimited usage."""
    payload = _policy_payload()
    del payload["request_limit"]

    try:
        BudgetPolicy.model_validate(payload)
    except ValidationError as exc:
        assert any(error["loc"] == ("request_limit",) for error in exc.errors())
    else:
        raise AssertionError("A policy without request_limit was accepted.")


def _price_source() -> PriceSourceReference:
    """Return a versioned, reviewable pricing-source reference."""
    return PriceSourceReference(
        source_id="approved-catalogue",
        source_version="2026-11",
        approval_reference="owner-approval-1",
    )


def _budget_estimate(
    *,
    retries: int = 0,
    provider_tokens: int | None = 50,
    estimated_spend_usd: Decimal | None = None,
    price_source: PriceSourceReference | None = None,
) -> BudgetEstimate:
    """Return a bounded single-logical-operation estimate."""
    return BudgetEstimate(
        provider_requests=1,
        provider_attempts=1 + retries,
        retries=retries,
        provider_tokens=provider_tokens,
        token_estimator_id=("tokenizer" if provider_tokens is not None else None),
        token_estimator_version=("1" if provider_tokens is not None else None),
        estimated_spend_usd=estimated_spend_usd,
        price_source=price_source,
        spend_estimator_id=(
            "cost-estimator" if estimated_spend_usd is not None else None
        ),
        spend_estimator_version=(
            "1" if estimated_spend_usd is not None else None
        ),
    )


def _usage_snapshot(
    *,
    requests: int,
    tokens: int | None = 900,
    attempts: int | None = None,
    retries: int = 0,
    spend: Decimal | None = None,
    reserved_requests: int = 0,
    reserved_tokens: int = 0,
    reserved_attempts: int | None = None,
    reserved_retries: int = 0,
    reserved_spend: Decimal | None = None,
) -> BudgetUsageSnapshot:
    """Build explicit consumed and reserved totals for one scope."""
    return BudgetUsageSnapshot(
        consumed=BudgetConsumption(
            provider_requests=requests,
            provider_attempts=(
                requests + retries if attempts is None else attempts
            ),
            retries=retries,
            provider_tokens=tokens,
            spend_usd=spend,
        ),
        reserved=BudgetReservationAmounts(
            provider_requests=reserved_requests,
            provider_attempts=(
                reserved_requests + reserved_retries
                if reserved_attempts is None
                else reserved_attempts
            ),
            retries=reserved_retries,
            provider_tokens=reserved_tokens,
            spend_usd=reserved_spend,
        ),
    )


def test_evaluation_allows_usage_within_explicit_limits() -> None:
    """A request fitting every budget dimension is admitted."""
    policy = BudgetPolicy.model_validate(_policy_payload())

    decision = evaluate_budget(
        policy=policy,
        scope=policy.scope,
        usage=_usage_snapshot(requests=9),
        estimate=_budget_estimate(),
    )

    assert decision.status is BudgetDecisionStatus.ALLOWED
    assert decision.denials == ()


def test_evaluation_denies_request_that_exceeds_limit() -> None:
    """A request over the provider-request ceiling is denied."""
    policy = BudgetPolicy.model_validate(_policy_payload())

    decision = evaluate_budget(
        policy=policy,
        scope=policy.scope,
        usage=_usage_snapshot(requests=10, tokens=900),
        estimate=_budget_estimate(),
    )

    assert decision.status is BudgetDecisionStatus.DENIED
    assert tuple(denial.code for denial in decision.denials) == (
        BudgetDenialCode.REQUEST_LIMIT_EXCEEDED,
    )


def test_missing_policy_is_denied_instead_of_unlimited() -> None:
    """A missing required policy produces a deterministic denial."""
    decision = evaluate_budget(
        policy=None,
        scope=BudgetPolicy.model_validate(_policy_payload()).scope,
        usage=_usage_snapshot(requests=0, tokens=0),
        estimate=_budget_estimate(),
    )

    assert decision.status is BudgetDecisionStatus.DENIED
    assert tuple(denial.code for denial in decision.denials) == (
        BudgetDenialCode.POLICY_REQUIRED,
    )


def test_allowed_decision_requires_policy_identity() -> None:
    """A bare allowed decision cannot bypass required policy evaluation."""
    scope = BudgetScope(
        kind=BudgetScopeKind.PRINCIPAL,
        scope_id="vait-service",
    )

    with pytest.raises(ValidationError):
        BudgetDecision(
            status=BudgetDecisionStatus.ALLOWED,
            scope=scope,
        ).require_allowed()


def test_directly_constructed_allowed_decision_is_not_evaluator_approved() -> None:
    """A valid policy reference alone is not evidence of budget evaluation."""
    scope = BudgetScope(
        kind=BudgetScopeKind.PRINCIPAL,
        scope_id="vait-service",
    )
    decision = BudgetDecision(
        status=BudgetDecisionStatus.ALLOWED,
        scope=scope,
        policy_id="service-provider-budget",
        policy_version=1,
    )

    assert decision.is_allowed is False
    with pytest.raises(ValueError, match="denied decision"):
        decision.require_allowed()


def test_model_validated_allowed_decision_is_not_evaluator_approved() -> None:
    """An ALLOWED payload cannot manufacture evaluator provenance."""
    scope = BudgetScope(
        kind=BudgetScopeKind.PRINCIPAL,
        scope_id="vait-service",
    )
    decision = BudgetDecision.model_validate(
        {
            "status": BudgetDecisionStatus.ALLOWED,
            "scope": scope,
            "policy_id": "service-provider-budget",
            "policy_version": 1,
        }
    )

    assert decision.is_allowed is False
    with pytest.raises(ValueError, match="denied decision"):
        decision.require_allowed()


def test_model_constructed_allowed_decision_fails_closed_without_denials() -> None:
    """Unvalidated construction cannot allow or crash while reporting denial."""
    scope = BudgetScope(
        kind=BudgetScopeKind.PRINCIPAL,
        scope_id="vait-service",
    )
    decision = BudgetDecision.model_construct(
        status=BudgetDecisionStatus.ALLOWED,
        scope=scope,
        policy_id="service-provider-budget",
        policy_version=1,
    )

    assert decision.is_allowed is False
    with pytest.raises(ValueError, match="denied decision"):
        decision.require_allowed()


def test_evaluator_approved_allowed_decision_can_be_required() -> None:
    """A successful policy evaluation marks its ALLOWED result as admissible."""
    policy = BudgetPolicy.model_validate(_policy_payload())
    decision = evaluate_budget(
        policy=policy,
        scope=policy.scope,
        usage=_usage_snapshot(requests=9),
        estimate=_budget_estimate(),
    )

    assert decision.is_allowed is True
    decision.require_allowed()


def test_copy_with_unvalidated_policy_mutation_loses_evaluator_provenance() -> None:
    """Pydantic's unchecked model_copy update cannot rebind an evaluation."""
    policy = BudgetPolicy.model_validate(_policy_payload())
    decision = evaluate_budget(
        policy=policy,
        scope=policy.scope,
        usage=_usage_snapshot(requests=9),
        estimate=_budget_estimate(),
    )
    copied_decision = decision.model_copy(update={"policy_id": "caller-selected"})

    assert copied_decision.is_allowed is False


def test_denied_error_rejects_invalid_decision_with_no_denial_reason() -> None:
    """A malformed ALLOWED model cannot trigger empty-denial indexing."""
    decision = BudgetDecision.model_construct(
        status=BudgetDecisionStatus.ALLOWED,
    )

    with pytest.raises(ValueError, match="denied decision"):
        BudgetDeniedError(decision)


@pytest.mark.parametrize(
    ("policy_id", "policy_version"),
    [
        ("", 1),
        ("invalid policy id", 1),
        ("service-provider-budget", 0),
        ("service-provider-budget", True),
    ],
)
def test_allowed_decision_requires_a_valid_policy_reference(
    policy_id: str,
    policy_version: int,
) -> None:
    """Decision policy identity uses the policy reference constraints."""
    scope = BudgetScope(
        kind=BudgetScopeKind.PRINCIPAL,
        scope_id="vait-service",
    )

    with pytest.raises(ValidationError):
        BudgetDecision(
            status=BudgetDecisionStatus.ALLOWED,
            scope=scope,
            policy_id=policy_id,
            policy_version=policy_version,
        )


def test_policy_required_denial_cannot_claim_a_policy() -> None:
    """A policy-required denial remains distinct from a policy denial."""
    scope = BudgetScope(
        kind=BudgetScopeKind.PRINCIPAL,
        scope_id="vait-service",
    )

    with pytest.raises(ValidationError):
        BudgetDecision(
            status=BudgetDecisionStatus.DENIED,
            scope=scope,
            policy_id="service-provider-budget",
            policy_version=1,
            denials=(
                BudgetDenial(code=BudgetDenialCode.POLICY_REQUIRED),
            ),
        )


def test_unknown_consumed_tokens_are_not_treated_as_zero() -> None:
    """Unknown token usage denies admission under an enforced token cap."""
    policy = BudgetPolicy.model_validate(_policy_payload())

    decision = evaluate_budget(
        policy=policy,
        scope=policy.scope,
        usage=_usage_snapshot(requests=0, tokens=None),
        estimate=_budget_estimate(),
    )

    assert decision.status is BudgetDecisionStatus.DENIED
    assert tuple(denial.code for denial in decision.denials) == (
        BudgetDenialCode.TOKEN_CONSUMPTION_UNKNOWN,
    )


def test_missing_token_estimate_is_denied_before_execution() -> None:
    """A required pre-call token estimate cannot silently default to zero."""
    policy = BudgetPolicy.model_validate(_policy_payload())
    estimate = BudgetEstimate(
        provider_requests=1,
        provider_attempts=1,
        retries=0,
    )

    decision = evaluate_budget(
        policy=policy,
        scope=policy.scope,
        usage=_usage_snapshot(requests=0, tokens=0),
        estimate=estimate,
    )

    assert decision.status is BudgetDecisionStatus.DENIED
    assert tuple(denial.code for denial in decision.denials) == (
        BudgetDenialCode.TOKEN_ESTIMATE_UNAVAILABLE,
    )


def test_explicit_zero_token_estimate_is_distinct_from_unknown() -> None:
    """An explicit zero is valid when estimator provenance exists."""
    policy = BudgetPolicy.model_validate(_policy_payload())
    estimate = BudgetEstimate(
        provider_requests=1,
        provider_attempts=1,
        retries=0,
        provider_tokens=0,
        token_estimator_id="tokenizer",
        token_estimator_version="1",
    )

    decision = evaluate_budget(
        policy=policy,
        scope=policy.scope,
        usage=_usage_snapshot(requests=0, tokens=0),
        estimate=estimate,
    )

    assert decision.status is BudgetDecisionStatus.ALLOWED


def test_scope_mismatch_is_denied() -> None:
    """A policy cannot be evaluated against another principal's scope."""
    policy = BudgetPolicy.model_validate(_policy_payload())

    decision = evaluate_budget(
        policy=policy,
        scope=BudgetScope(
            kind=BudgetScopeKind.PRINCIPAL,
            scope_id="other-service",
        ),
        usage=_usage_snapshot(requests=0, tokens=0),
        estimate=_budget_estimate(),
    )

    assert decision.status is BudgetDecisionStatus.DENIED
    assert tuple(denial.code for denial in decision.denials) == (
        BudgetDenialCode.SCOPE_MISMATCH,
    )


def test_denied_decision_raises_a_safe_typed_error() -> None:
    """A caller can raise a typed denial without exposing usage payloads."""
    policy = BudgetPolicy.model_validate(_policy_payload())
    decision = evaluate_budget(
        policy=policy,
        scope=policy.scope,
        usage=_usage_snapshot(requests=10, tokens=0),
        estimate=_budget_estimate(),
    )

    with pytest.raises(BudgetDeniedError) as captured:
        decision.require_allowed()

    assert str(captured.value) == "Provider operation denied by budget policy."
    assert captured.value.code is BudgetDenialCode.REQUEST_LIMIT_EXCEEDED


def test_reserved_requests_and_tokens_count_toward_limits() -> None:
    """Unsettled reservations consume capacity before another dispatch."""
    policy = BudgetPolicy.model_validate(_policy_payload())

    decision = evaluate_budget(
        policy=policy,
        scope=policy.scope,
        usage=_usage_snapshot(
            requests=9,
            tokens=901,
            reserved_requests=1,
            reserved_tokens=50,
        ),
        estimate=_budget_estimate(),
    )

    assert decision.status is BudgetDecisionStatus.DENIED
    assert tuple(denial.code for denial in decision.denials) == (
        BudgetDenialCode.REQUEST_LIMIT_EXCEEDED,
        BudgetDenialCode.TOKEN_LIMIT_EXCEEDED,
    )


def test_exact_request_and_token_limits_are_admitted() -> None:
    """A reservation exactly at both ceilings remains admissible."""
    policy = BudgetPolicy.model_validate(_policy_payload())

    decision = evaluate_budget(
        policy=policy,
        scope=policy.scope,
        usage=_usage_snapshot(requests=9, tokens=950),
        estimate=_budget_estimate(),
    )

    assert decision.status is BudgetDecisionStatus.ALLOWED


def test_retry_and_provider_attempt_budgets_are_independent() -> None:
    """A retry is denied when its retry allowance is spent."""
    policy_payload = _policy_payload()
    policy_payload["retry_budget"] = {"max_retries": 1}
    policy_payload["provider_attempt_budget"] = {"max_attempts": 4}
    policy = BudgetPolicy.model_validate(policy_payload)

    decision = evaluate_budget(
        policy=policy,
        scope=policy.scope,
        usage=_usage_snapshot(
            requests=1,
            tokens=0,
            attempts=2,
            retries=1,
        ),
        estimate=_budget_estimate(retries=1),
    )

    assert decision.status is BudgetDecisionStatus.DENIED
    assert tuple(denial.code for denial in decision.denials) == (
        BudgetDenialCode.RETRY_BUDGET_EXCEEDED,
    )


def test_attempt_counts_match_logical_requests_and_retries_across_contracts() -> None:
    """One request with one retry records two actual provider attempts."""
    estimate = _budget_estimate(retries=1)
    consumption = BudgetConsumption(
        provider_requests=1,
        provider_attempts=2,
        retries=1,
        provider_tokens=50,
    )
    reservation = BudgetReservationAmounts(
        provider_requests=1,
        provider_attempts=2,
        retries=1,
        provider_tokens=50,
    )

    assert estimate.provider_requests == consumption.provider_requests == 1
    assert estimate.provider_attempts == consumption.provider_attempts == 2
    assert estimate.retries == consumption.retries == 1
    assert reservation.provider_requests == 1
    assert reservation.provider_attempts == 2
    assert reservation.retries == 1


def test_attempt_counts_must_equal_requests_plus_retries() -> None:
    """Estimates, actual usage and reservations share one attempt invariant."""
    with pytest.raises(ValidationError):
        BudgetEstimate(provider_requests=1, provider_attempts=1, retries=1)

    with pytest.raises(ValidationError):
        BudgetConsumption(provider_requests=1, provider_attempts=1, retries=1)

    with pytest.raises(ValidationError):
        BudgetReservationAmounts(
            provider_requests=1,
            provider_attempts=1,
            retries=1,
            provider_tokens=50,
        )


def test_provider_request_and_attempt_units_are_documented() -> None:
    """Request units stay logical while attempt units count actual calls."""
    for contract in (BudgetEstimate, BudgetConsumption, BudgetReservationAmounts):
        request_description = contract.model_fields["provider_requests"].description
        attempt_description = contract.model_fields["provider_attempts"].description

        assert request_description is not None
        assert "logical provider operations" in request_description.lower()
        assert "retries do not increment" in request_description.lower()
        assert attempt_description is not None
        assert "actual provider calls" in attempt_description.lower()


def test_monetary_limit_requires_known_cost_and_matching_price_source() -> None:
    """A spend ceiling requires pre-call price provenance and known totals."""
    payload = _policy_payload()
    payload["monetary_limit"] = {
        "max_cost_usd": "1.00",
        "price_source": _price_source().model_dump(),
    }
    policy = BudgetPolicy.model_validate(payload)

    decision = evaluate_budget(
        policy=policy,
        scope=policy.scope,
        usage=_usage_snapshot(
            requests=0,
            tokens=0,
            spend=Decimal("0.50"),
            reserved_spend=Decimal("0.20"),
        ),
        estimate=_budget_estimate(
            estimated_spend_usd=Decimal("0.30"),
            price_source=_price_source(),
        ),
    )

    assert decision.status is BudgetDecisionStatus.ALLOWED


def test_monetary_limit_rejects_unknown_cost() -> None:
    """Unknown cost is not treated as zero under an enabled spend ceiling."""
    payload = _policy_payload()
    payload["monetary_limit"] = {
        "max_cost_usd": "1.00",
        "price_source": _price_source().model_dump(),
    }
    policy = BudgetPolicy.model_validate(payload)

    decision = evaluate_budget(
        policy=policy,
        scope=policy.scope,
        usage=_usage_snapshot(
            requests=0,
            tokens=0,
            reserved_spend=Decimal("0.00"),
        ),
        estimate=_budget_estimate(
            estimated_spend_usd=Decimal("0.10"),
            price_source=_price_source(),
        ),
    )

    assert decision.status is BudgetDecisionStatus.DENIED
    assert tuple(denial.code for denial in decision.denials) == (
        BudgetDenialCode.SPEND_CONSUMPTION_UNKNOWN,
    )


def test_monetary_limit_rejects_unknown_reserved_spend() -> None:
    """An unresolved active reservation is not assumed to cost zero."""
    payload = _policy_payload()
    payload["monetary_limit"] = {
        "max_cost_usd": "1.00",
        "price_source": _price_source().model_dump(),
    }
    policy = BudgetPolicy.model_validate(payload)

    decision = evaluate_budget(
        policy=policy,
        scope=policy.scope,
        usage=_usage_snapshot(
            requests=0,
            tokens=0,
            spend=Decimal("0.00"),
            reserved_spend=None,
        ),
        estimate=_budget_estimate(
            estimated_spend_usd=Decimal("0.10"),
            price_source=_price_source(),
        ),
    )

    assert decision.status is BudgetDecisionStatus.DENIED
    assert tuple(denial.code for denial in decision.denials) == (
        BudgetDenialCode.SPEND_RESERVATION_UNKNOWN,
    )


def test_monetary_limit_rejects_a_different_price_source() -> None:
    """A cost estimate from a different tariff version is inadmissible."""
    payload = _policy_payload()
    payload["monetary_limit"] = {
        "max_cost_usd": "1.00",
        "price_source": _price_source().model_dump(),
    }
    policy = BudgetPolicy.model_validate(payload)
    different_source = PriceSourceReference(
        source_id="approved-catalogue",
        source_version="2026-10",
        approval_reference="owner-approval-1",
    )

    decision = evaluate_budget(
        policy=policy,
        scope=policy.scope,
        usage=_usage_snapshot(
            requests=0,
            tokens=0,
            spend=Decimal("0.00"),
            reserved_spend=Decimal("0.00"),
        ),
        estimate=_budget_estimate(
            estimated_spend_usd=Decimal("0.10"),
            price_source=different_source,
        ),
    )

    assert decision.status is BudgetDecisionStatus.DENIED
    assert tuple(denial.code for denial in decision.denials) == (
        BudgetDenialCode.PRICE_SOURCE_MISMATCH,
    )


def test_monetary_limit_rejects_conservative_estimate_over_ceiling() -> None:
    """Consumed, reserved and estimated spend are compared before dispatch."""
    payload = _policy_payload()
    payload["monetary_limit"] = {
        "max_cost_usd": "1.00",
        "price_source": _price_source().model_dump(),
    }
    policy = BudgetPolicy.model_validate(payload)

    decision = evaluate_budget(
        policy=policy,
        scope=policy.scope,
        usage=_usage_snapshot(
            requests=0,
            tokens=0,
            spend=Decimal("0.70"),
            reserved_spend=Decimal("0.20"),
        ),
        estimate=_budget_estimate(
            estimated_spend_usd=Decimal("0.20"),
            price_source=_price_source(),
        ),
    )

    assert decision.status is BudgetDecisionStatus.DENIED
    assert tuple(denial.code for denial in decision.denials) == (
        BudgetDenialCode.SPEND_LIMIT_EXCEEDED,
    )


def test_enabled_monetary_limit_rejects_missing_pre_call_estimate() -> None:
    """A spend ceiling cannot run without a priced upper-bound estimate."""
    payload = _policy_payload()
    payload["monetary_limit"] = {
        "max_cost_usd": "1.00",
        "price_source": _price_source().model_dump(),
    }
    policy = BudgetPolicy.model_validate(payload)

    decision = evaluate_budget(
        policy=policy,
        scope=policy.scope,
        usage=_usage_snapshot(
            requests=0,
            tokens=0,
            spend=Decimal("0.00"),
            reserved_spend=Decimal("0.00"),
        ),
        estimate=_budget_estimate(),
    )

    assert decision.status is BudgetDecisionStatus.DENIED
    assert tuple(denial.code for denial in decision.denials) == (
        BudgetDenialCode.SPEND_ESTIMATE_UNAVAILABLE,
    )


def test_provider_attempt_limit_is_enforced_independently() -> None:
    """A remaining request quota cannot bypass the attempt ceiling."""
    payload = _policy_payload()
    payload["provider_attempt_budget"] = {"max_attempts": 1}
    policy = BudgetPolicy.model_validate(payload)

    decision = evaluate_budget(
        policy=policy,
        scope=policy.scope,
        usage=_usage_snapshot(requests=1, tokens=0, attempts=1),
        estimate=_budget_estimate(),
    )

    assert decision.status is BudgetDecisionStatus.DENIED
    assert tuple(denial.code for denial in decision.denials) == (
        BudgetDenialCode.PROVIDER_ATTEMPT_LIMIT_EXCEEDED,
    )


def test_retry_limit_cannot_equal_total_attempt_limit() -> None:
    """Retry count excludes the initial provider attempt."""
    payload = _policy_payload()
    payload["retry_budget"] = {"max_retries": 2}
    payload["provider_attempt_budget"] = {"max_attempts": 2}

    with pytest.raises(ValidationError):
        BudgetPolicy.model_validate(payload)


@pytest.mark.parametrize(
    ("section_name", "field_name", "value"),
    [
        ("request_limit", "max_requests", 0),
        ("request_limit", "max_requests", -1),
        ("token_limit", "max_tokens", 0),
        ("token_limit", "max_tokens", -1),
        ("retry_budget", "max_retries", -1),
        ("provider_attempt_budget", "max_attempts", 0),
        ("provider_attempt_budget", "max_attempts", -1),
    ],
)
def test_policy_rejects_zero_or_negative_required_limits(
    section_name: str,
    field_name: str,
    value: object,
) -> None:
    """Required limits reject accidental zero and negative values."""
    payload = _policy_payload()
    section = payload[section_name]
    assert isinstance(section, dict)
    section[field_name] = value

    with pytest.raises(ValidationError):
        BudgetPolicy.model_validate(payload)


def test_policy_accepts_zero_retries_as_an_explicit_no_retry_policy() -> None:
    """Zero is valid only for the retry count, not as an unlimited sentinel."""
    policy = BudgetPolicy.model_validate(_policy_payload())

    assert policy.retry_budget.max_retries == 0


def test_policy_rejects_missing_token_limit() -> None:
    """A missing token ceiling cannot silently disable token enforcement."""
    payload = _policy_payload()
    del payload["token_limit"]

    with pytest.raises(ValidationError):
        BudgetPolicy.model_validate(payload)


def test_policy_rejects_boolean_numeric_limit() -> None:
    """Booleans cannot be coerced into positive integer limits."""
    payload = _policy_payload()
    payload["request_limit"] = {"max_requests": True}

    with pytest.raises(ValidationError):
        BudgetPolicy.model_validate(payload)


def test_fixed_window_requires_a_positive_interval() -> None:
    """A fixed-window policy cannot omit or zero its interval."""
    payload = _policy_payload()
    payload["window"] = {"kind": "fixed_window"}

    with pytest.raises(ValidationError):
        BudgetPolicy.model_validate(payload)

    payload["window"] = {
        "kind": "fixed_window",
        "interval_seconds": 0,
    }
    with pytest.raises(ValidationError):
        BudgetPolicy.model_validate(payload)


def test_scope_identifiers_are_type_checked() -> None:
    """Experiment and job scope identifiers must be canonical UUIDs."""
    with pytest.raises(ValidationError):
        BudgetScope(kind=BudgetScopeKind.EXPERIMENT, scope_id="client-value")

    with pytest.raises(ValidationError):
        BudgetScope(kind=BudgetScopeKind.PRINCIPAL, scope_id="token value")

    policy = BudgetPolicy.model_validate(_policy_payload())
    with pytest.raises(ValidationError):
        BudgetPolicyReference(
            policy_id="invalid policy id",
            policy_version=1,
            scope=policy.scope,
        )


def test_policy_version_must_be_positive() -> None:
    """Policy revisions start at one and cannot use zero as a sentinel."""
    payload = _policy_payload()
    payload["policy_version"] = 0

    with pytest.raises(ValidationError):
        BudgetPolicy.model_validate(payload)


@pytest.mark.parametrize(
    "amount",
    [Decimal("0"), Decimal("-0.01"), float("0.1"), Decimal("NaN")],
)
def test_monetary_limit_rejects_zero_negative_float_and_non_finite(
    amount: object,
) -> None:
    """Spend ceilings require a positive finite exact decimal amount."""
    with pytest.raises(ValidationError):
        MonetarySpendLimit.model_validate(
            {
                "max_cost_usd": amount,
                "price_source": _price_source().model_dump(),
            }
        )


def test_monetary_limit_requires_price_source_and_approval_reference() -> None:
    """A hard spend ceiling cannot omit price provenance."""
    with pytest.raises(ValidationError):
        MonetarySpendLimit.model_validate({"max_cost_usd": "1.00"})

    source = _price_source().model_dump()
    del source["approval_reference"]
    with pytest.raises(ValidationError):
        MonetarySpendLimit.model_validate(
            {
                "max_cost_usd": "1.00",
                "price_source": source,
            }
        )


def test_spend_estimate_requires_versioned_estimator_provenance() -> None:
    """A spend upper bound identifies the estimator that produced it."""
    with pytest.raises(ValidationError):
        BudgetEstimate(
            provider_requests=1,
            provider_attempts=1,
            retries=0,
            provider_tokens=0,
            token_estimator_id="tokenizer",
            token_estimator_version="1",
            estimated_spend_usd=Decimal("0.10"),
            price_source=_price_source(),
        )


def test_unknown_consumption_is_distinct_from_known_zero() -> None:
    """Unknown token/cost values remain distinct from measured zero."""
    unknown = BudgetConsumption(
        provider_requests=1,
        provider_attempts=1,
        retries=0,
        provider_tokens=None,
        spend_usd=None,
    )
    known_zero = BudgetConsumption(
        provider_requests=1,
        provider_attempts=1,
        retries=0,
        provider_tokens=0,
        spend_usd=Decimal("0.00"),
    )

    assert unknown.provider_tokens is None
    assert unknown.spend_usd is None
    assert known_zero.provider_tokens == 0
    assert known_zero.spend_usd == Decimal("0.00")


def test_invalid_policy_loader_returns_typed_non_sensitive_error() -> None:
    """Invalid configuration is mapped to a stable error without its input."""
    payload = _policy_payload()
    payload["request_limit"] = {"max_requests": 0}

    with pytest.raises(InvalidBudgetPolicyError) as captured:
        load_budget_policy(payload)

    assert captured.value.code.value == "invalid_policy"
    assert str(captured.value) == "Budget policy is invalid."
    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__ is True


def test_reservation_contract_records_all_applicable_scopes() -> None:
    """One reservation can bind principal and job policies atomically."""
    principal_policy = BudgetPolicy.model_validate(_policy_payload())
    job_payload = _policy_payload()
    job_payload["scope"] = {
        "kind": "job",
        "scope_id": "00000000-0000-0000-0000-000000000001",
    }
    job_payload["window"] = {"kind": "per_execution"}
    job_policy = BudgetPolicy.model_validate(job_payload)

    request = BudgetReservationRequest(
        reservation_id=_RESERVATION_ID,
        policies=(principal_policy, job_policy),
        estimate=_budget_estimate(),
    )
    reservation = BudgetReservation(
        reservation_id=request.reservation_id,
        policies=(
            BudgetPolicyReference(
                policy_id=principal_policy.policy_id,
                policy_version=principal_policy.policy_version,
                scope=principal_policy.scope,
            ),
            BudgetPolicyReference(
                policy_id=job_policy.policy_id,
                policy_version=job_policy.policy_version,
                scope=job_policy.scope,
            ),
        ),
        amounts=BudgetReservationAmounts(
            provider_requests=1,
            provider_attempts=1,
            retries=0,
            provider_tokens=50,
            spend_usd=None,
        ),
        state=BudgetReservationState.RESERVED,
    )

    assert len(request.policies) == 2
    assert len(reservation.policies) == 2


def test_reservation_rejects_duplicate_scope_policies() -> None:
    """One operation cannot double-count the same scope ambiguously."""
    policy = BudgetPolicy.model_validate(_policy_payload())

    with pytest.raises(ValidationError):
        BudgetReservationRequest(
            reservation_id=_RESERVATION_ID,
            policies=(
                policy,
                policy.model_copy(update={"policy_version": 2}),
            ),
            estimate=_budget_estimate(),
        )
