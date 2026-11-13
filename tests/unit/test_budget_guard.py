"""Isolated tests for the provider budget dispatch boundary."""

from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest

import vait.platform.budget_guard as guard
from vait.contracts.budgeting import (
    BudgetEstimate,
    BudgetPolicy,
    BudgetReservation,
    BudgetReservationRequest,
    BudgetReservationState,
    BudgetSettlement,
)
from vait.inference.providers.models import (
    InferenceMessage,
    InferenceRole,
    ProviderInferenceRequest,
    ProviderInferenceResponse,
    ProviderTiming,
    ProviderTokenUsage,
)
from vait.platform.persistence.engine import DatabaseRuntime


class FakeProvider:
    provider_id = "fake"
    runtime_id = "fake-runtime"

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.fail = False
        self.wrong_identity = False

    def generate(
        self, request: ProviderInferenceRequest
    ) -> ProviderInferenceResponse:
        self.events.append("generate")
        if self.fail:
            raise ConnectionError("simulated transport failure")
        return ProviderInferenceResponse(
            request_id=(
                "wrong-request" if self.wrong_identity
                else request.request_id
            ),
            provider_id=self.provider_id,
            runtime_id=self.runtime_id,
            model_id=request.model_id,
            output_text="test response",
            usage=ProviderTokenUsage(input_tokens=2, output_tokens=3),
            timing=ProviderTiming(total_latency_ms=1.0),
            cost_usd=0.25,
        )


class FakeLedger:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.fail_at: str | None = None
        self.settlements: list[BudgetSettlement] = []

    def reserve(
        self, request: BudgetReservationRequest
    ) -> BudgetReservation:
        self.events.append("reserve")
        if self.fail_at == "reserve":
            raise RuntimeError("reserve denied")
        return cast(
            BudgetReservation,
            SimpleNamespace(state=BudgetReservationState.RESERVED),
        )

    def mark_dispatched(
        self,
        reservation_id: object,
        *,
        require_new_transition: bool = False,
    ) -> BudgetReservation:
        if not require_new_transition:
            raise AssertionError("Strict dispatch claim required")
        self.events.append("dispatch")
        if self.fail_at == "dispatch":
            raise RuntimeError("dispatch failed")
        return cast(
            BudgetReservation,
            SimpleNamespace(state=BudgetReservationState.DISPATCHED),
        )

    def settle(
        self, settlement: BudgetSettlement
    ) -> BudgetReservation:
        self.events.append("settle")
        if self.fail_at == "settle":
            raise RuntimeError("settlement failed")
        self.settlements.append(settlement)
        return cast(
            BudgetReservation,
            SimpleNamespace(state=BudgetReservationState.SETTLED),
        )


def make_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    guard.BudgetedProviderDispatcher,
    FakeProvider,
    FakeLedger,
    list[str],
]:
    events: list[str] = []
    provider = FakeProvider(events)
    ledger = FakeLedger(events)

    @contextmanager
    def fake_transaction(runtime: DatabaseRuntime) -> Iterator[object]:
        del runtime
        try:
            yield object()
            events.append("commit")
        except Exception:
            events.append("rollback")
            raise

    monkeypatch.setattr(
        guard, "transactional_session", fake_transaction
    )
    monkeypatch.setattr(
        guard,
        "PostgreSQLBudgetReservationLedger",
        lambda session: ledger,
    )

    runtime = cast(DatabaseRuntime, object())
    dispatcher = guard.BudgetedProviderDispatcher(
        provider=provider,
        runtime=runtime,
    )
    return dispatcher, provider, ledger, events


def input_data() -> tuple[
    ProviderInferenceRequest,
    tuple[BudgetPolicy, ...],
    BudgetEstimate,
]:
    request = ProviderInferenceRequest(
        request_id="test-request",
        model_id="test-model",
        messages=(
            InferenceMessage(role=InferenceRole.USER, content="Hello"),
        ),
        max_output_tokens=10,
    )
    policy = BudgetPolicy.model_validate({
        "policy_id": "test-policy",
        "policy_version": 1,
        "scope": {"kind": "principal", "scope_id": "test-user"},
        "window": {"kind": "lifetime"},
        "request_limit": {"max_requests": 10},
        "token_limit": {"max_tokens": 100},
        "retry_budget": {"max_retries": 0},
        "provider_attempt_budget": {"max_attempts": 10},
    })
    estimate = BudgetEstimate(
        provider_requests=1,
        provider_attempts=1,
        retries=0,
        provider_tokens=20,
        token_estimator_id="test-estimator",
        token_estimator_version="1",
    )
    return request, (policy,), estimate


def test_successful_dispatch_commits_each_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher, _, ledger, events = make_harness(monkeypatch)
    request, policies, estimate = input_data()

    response = dispatcher.execute(
        request=request, policies=policies, estimate=estimate
    , execution_id=uuid4())

    assert response.succeeded
    assert events == [
        "reserve", "commit",
        "dispatch", "commit",
        "generate",
        "settle", "commit",
    ]
    assert len(ledger.settlements) == 1
    consumption = ledger.settlements[0].consumption
    assert consumption.provider_tokens == 5
    assert consumption.spend_usd == Decimal("0.25")


@pytest.mark.parametrize("failure", ["reserve", "dispatch"])
def test_no_provider_call_without_committed_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    dispatcher, _, ledger, events = make_harness(monkeypatch)
    ledger.fail_at = failure
    request, policies, estimate = input_data()

    with pytest.raises(RuntimeError):
        dispatcher.execute(
            request=request, policies=policies, estimate=estimate
        , execution_id=uuid4())

    assert "generate" not in events
    assert "settle" not in events


def test_provider_exception_records_unknown_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher, provider, ledger, events = make_harness(monkeypatch)
    provider.fail = True
    request, policies, estimate = input_data()

    with pytest.raises(ConnectionError):
        dispatcher.execute(
            request=request, policies=policies, estimate=estimate
        , execution_id=uuid4())

    assert events[-2:] == ["settle", "commit"]
    consumption = ledger.settlements[0].consumption
    assert consumption.provider_attempts == 1
    assert consumption.provider_tokens is None
    assert consumption.spend_usd is None


def test_response_identity_mismatch_is_not_trusted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher, provider, ledger, _ = make_harness(monkeypatch)
    provider.wrong_identity = True
    request, policies, estimate = input_data()

    with pytest.raises(ValueError, match="identity mismatch"):
        dispatcher.execute(
            request=request, policies=policies, estimate=estimate
        , execution_id=uuid4())

    assert ledger.settlements[0].consumption.provider_tokens is None


def test_settlement_failure_is_not_silenced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher, _, ledger, events = make_harness(monkeypatch)
    ledger.fail_at = "settle"
    request, policies, estimate = input_data()

    with pytest.raises(RuntimeError, match="settlement failed"):
        dispatcher.execute(
            request=request, policies=policies, estimate=estimate
        , execution_id=uuid4())

    assert events[-2:] == ["settle", "rollback"]


def test_multi_attempt_estimate_rejected_before_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher, _, _, events = make_harness(monkeypatch)
    request, policies, _ = input_data()
    estimate = BudgetEstimate(
        provider_requests=1,
        provider_attempts=2,
        retries=1,
        provider_tokens=20,
        token_estimator_id="test-estimator",
        token_estimator_version="1",
    )

    with pytest.raises(ValueError, match="one-attempt"):
        dispatcher.execute(
            request=request, policies=policies, estimate=estimate
        , execution_id=uuid4())

    assert events == []


def test_dispatch_commit_failure_blocks_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider must not run if dispatch commit fails."""
    dispatcher, _, _, events = make_harness(monkeypatch)
    request, policies, estimate = input_data()

    @contextmanager
    def failing_transaction(runtime: DatabaseRuntime) -> Iterator[object]:
        del runtime
        try:
            yield object()
            events.append("commit")
            if events.count("commit") == 2:
                raise RuntimeError("dispatch commit failed")
        except Exception:
            events.append("rollback")
            raise

    monkeypatch.setattr(
        guard,
        "transactional_session",
        failing_transaction,
    )

    with pytest.raises(RuntimeError, match="dispatch commit failed"):
        dispatcher.execute(
            request=request,
            policies=policies,
            estimate=estimate,
         execution_id=uuid4(),)

    assert "generate" not in events
    assert "settle" not in events
    assert events == [
        "reserve",
        "commit",
        "dispatch",
        "commit",
        "rollback",
    ]


def test_stable_execution_identity_is_reserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher, _, ledger, events = make_harness(monkeypatch)
    request, policies, estimate = input_data()
    execution_id = uuid4()
    observed: list[object] = []
    original_reserve = ledger.reserve

    def capture_reserve(
        reservation: BudgetReservationRequest,
    ) -> BudgetReservation:
        observed.append(reservation.reservation_id)
        return original_reserve(reservation)

    monkeypatch.setattr(ledger, "reserve", capture_reserve)

    dispatcher.execute(
        request=request,
        policies=policies,
        estimate=estimate,
        execution_id=execution_id,
    )

    assert observed == [execution_id]
    assert events.count("generate") == 1
