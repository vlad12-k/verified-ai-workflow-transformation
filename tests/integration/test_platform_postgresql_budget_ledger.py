"""PostgreSQL integration tests for the durable budget ledger."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from ipaddress import ip_address
from pathlib import Path
from threading import Barrier, Event, Lock, local
from uuid import UUID, uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from vait.contracts.budgeting import (
    BudgetConsumption,
    BudgetDenialCode,
    BudgetDeniedError,
    BudgetEstimate,
    BudgetPolicy,
    BudgetPolicyReference,
    BudgetReservationRequest,
    BudgetReservationState,
    BudgetSettlement,
)
from vait.platform.persistence.repositories.budgets import (
    BudgetReservationConflictError,
    BudgetReservationStateError,
    PostgreSQLBudgetReservationLedger,
)

_DATABASE_ENV = "VAIT_TEST_DATABASE_URL"
_ROOT = Path(__file__).resolve().parents[2]
_ALEMBIC_INI = _ROOT / "alembic.ini"
_EXPECTED_REVISION = "0004_m6c2_budget_ledger"


@pytest.fixture
def migrated_engine(monkeypatch: pytest.MonkeyPatch) -> Iterator[Engine]:
    """Apply the repository migration history to an explicitly named test DB."""
    database_url = os.getenv(_DATABASE_ENV)
    if database_url is None:
        pytest.skip(f"{_DATABASE_ENV} is not configured")

    probe_engine = create_engine(database_url)
    try:
        with probe_engine.connect() as connection:
            connection.exec_driver_sql("SET TRANSACTION READ ONLY")
            identity = connection.execute(
                text(
                    "SELECT current_database(), current_setting('server_encoding'), "
                    "current_setting('client_encoding'), inet_server_addr()::text"
                )
            ).one()
            connection.rollback()
    finally:
        probe_engine.dispose()

    database_name, server_encoding, client_encoding, server_address = identity
    if not str(database_name).endswith(("_test", "_ci")):
        pytest.fail("Budget-ledger tests require an explicitly named test database")
    if server_encoding != "UTF8" or client_encoding != "UTF8":
        pytest.fail("Budget-ledger tests require UTF-8 server and client encodings")
    if server_address is not None:
        try:
            is_loopback = ip_address(str(server_address)).is_loopback
        except ValueError:
            pytest.fail("Budget-ledger tests require a local PostgreSQL server")
        if not is_loopback:
            pytest.fail("Budget-ledger tests require a local PostgreSQL server")

    monkeypatch.setenv("VAIT_DATABASE_URL", database_url)
    command.upgrade(Config(str(_ALEMBIC_INI)), "head")
    engine = create_engine(database_url)
    try:
        yield engine
    finally:
        engine.dispose()


def test_migration_installs_the_budget_ledger_schema(
    migrated_engine: Engine,
) -> None:
    """The real Alembic history creates both ledger tables at the current head."""
    with migrated_engine.connect() as connection:
        revision = connection.scalar(text("SELECT version_num FROM alembic_version"))
        reservations = connection.scalar(
            text("SELECT to_regclass('budget_reservations') IS NOT NULL")
        )
        scopes = connection.scalar(
            text("SELECT to_regclass('budget_reservation_scopes') IS NOT NULL")
        )

    assert revision == _EXPECTED_REVISION
    assert reservations is True
    assert scopes is True


def test_budget_ledger_migration_downgrade_upgrade_round_trip(
    migrated_engine: Engine,
) -> None:
    """The M6-C2 revision removes and restores only its ledger schema."""
    config = Config(str(_ALEMBIC_INI))

    try:
        command.downgrade(config, "0003_m5f_jobs")
        with migrated_engine.connect() as connection:
            revision = connection.scalar(text("SELECT version_num FROM alembic_version"))
            reservations = connection.scalar(
                text("SELECT to_regclass('budget_reservations') IS NOT NULL")
            )
            scopes = connection.scalar(
                text("SELECT to_regclass('budget_reservation_scopes') IS NOT NULL")
            )

        assert revision == "0003_m5f_jobs"
        assert reservations is False
        assert scopes is False

        command.upgrade(config, "head")
        command.check(config)
        with migrated_engine.connect() as connection:
            revisions: Sequence[str] = connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalars().all()
            ledger_schema = connection.execute(
                text(
                    "SELECT to_regclass('budget_reservations') IS NOT NULL, "
                    "to_regclass('budget_reservation_scopes') IS NOT NULL, "
                    "to_regclass('ix_budget_reservation_scopes_scope_reservation') "
                    "IS NOT NULL"
                )
            ).one()

        assert revisions == [_EXPECTED_REVISION]
        assert ledger_schema == (True, True, True)
    finally:
        command.upgrade(config, "head")


def test_postgresql_ledger_is_exported_from_repository_package() -> None:
    """Callers can obtain the adapter through the persistence API package."""
    repository_package = import_module("vait.platform.persistence.repositories")

    assert hasattr(repository_package, "PostgreSQLBudgetReservationLedger")
    assert "PostgreSQLBudgetReservationLedger" in repository_package.__all__


def _policy(
    *,
    scope_id: str | None = None,
    max_requests: int = 10,
    max_tokens: int = 1000,
) -> BudgetPolicy:
    """Build one complete test policy with no production defaults."""
    return BudgetPolicy.model_validate(
        {
            "policy_id": "postgres-test-budget",
            "policy_version": 7,
            "scope": {
                "kind": "principal",
                "scope_id": scope_id or str(uuid4()),
            },
            "window": {"kind": "lifetime"},
            "request_limit": {"max_requests": max_requests},
            "token_limit": {"max_tokens": max_tokens},
            "retry_budget": {"max_retries": 3},
            "provider_attempt_budget": {"max_attempts": 20},
        }
    )


def _request(
    reservation_id: UUID | None = None,
    *,
    policy: BudgetPolicy | None = None,
    provider_tokens: int = 25,
) -> BudgetReservationRequest:
    """Build a single provider-operation reservation request."""
    return BudgetReservationRequest(
        reservation_id=reservation_id or uuid4(),
        policies=(policy or _policy(),),
        estimate=BudgetEstimate(
            provider_requests=1,
            provider_attempts=1,
            retries=0,
            provider_tokens=provider_tokens,
            token_estimator_id="postgres-test-estimator",
            token_estimator_version="1",
        ),
    )


def test_successful_reservation_persists_the_policy_version(
    migrated_engine: Engine,
) -> None:
    """A within-limit reservation persists its immutable policy snapshot."""
    request = _request()
    with Session(migrated_engine) as session:
        reservation = PostgreSQLBudgetReservationLedger(session).reserve(request)
        session.commit()

    assert reservation.reservation_id == request.reservation_id
    assert reservation.state is BudgetReservationState.RESERVED
    assert reservation.policies == (
        BudgetPolicyReference(
            policy_id="postgres-test-budget",
            policy_version=7,
            scope=request.policies[0].scope,
        ),
    )
    assert reservation.amounts.provider_tokens == 25

    with migrated_engine.connect() as connection:
        scope_version = connection.scalar(
            text(
                "SELECT policy_version FROM budget_reservation_scopes "
                "WHERE reservation_id = :reservation_id"
            ),
            {"reservation_id": request.reservation_id},
        )
        payload = connection.scalar(
            text(
                "SELECT request_payload FROM budget_reservations "
                "WHERE reservation_id = :reservation_id"
            ),
            {"reservation_id": request.reservation_id},
        )

    assert scope_version == 7
    assert isinstance(payload, dict)
    assert payload["estimate"]["provider_tokens"] == 25


def test_request_ceiling_rejects_a_second_active_reservation(
    migrated_engine: Engine,
) -> None:
    """An active reservation consumes its request capacity immediately."""
    scope_id = str(uuid4())
    policy = _policy(scope_id=scope_id, max_requests=1)
    with Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).reserve(_request(policy=policy))
        session.commit()

    with pytest.raises(BudgetDeniedError) as captured, Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).reserve(
            _request(policy=policy)
        )

    assert captured.value.code is BudgetDenialCode.REQUEST_LIMIT_EXCEEDED
    with migrated_engine.connect() as connection:
        scope_count = connection.scalar(
            text(
                "SELECT COUNT(*) FROM budget_reservation_scopes "
                "WHERE scope_kind = 'principal' AND scope_id = :scope_id"
            ),
            {"scope_id": scope_id},
        )
    assert scope_count == 1


def test_token_ceiling_rejects_estimates_that_exceed_capacity(
    migrated_engine: Engine,
) -> None:
    """Active token estimates are reserved before another admission."""
    policy = _policy(max_tokens=40)
    with Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).reserve(
            _request(policy=policy, provider_tokens=25)
        )
        session.commit()

    with pytest.raises(BudgetDeniedError) as captured, Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).reserve(
            _request(policy=policy, provider_tokens=25)
        )

    assert captured.value.code is BudgetDenialCode.TOKEN_LIMIT_EXCEEDED


def test_denied_scope_leaves_no_partial_reservation_in_other_scopes(
    migrated_engine: Engine,
) -> None:
    """Admission across multiple scopes succeeds or writes nothing."""
    available_policy = _policy(max_requests=5)
    full_policy = _policy(max_requests=1)
    with Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).reserve(
            _request(policy=full_policy)
        )
        session.commit()

    reservation_id = uuid4()
    base_request = _request(policy=available_policy)
    multi_scope_request = BudgetReservationRequest(
        reservation_id=reservation_id,
        policies=(available_policy, full_policy),
        estimate=base_request.estimate,
    )
    with pytest.raises(BudgetDeniedError) as captured, Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).reserve(multi_scope_request)

    assert captured.value.code is BudgetDenialCode.REQUEST_LIMIT_EXCEEDED
    with migrated_engine.connect() as connection:
        reservation_count = connection.scalar(
            text(
                "SELECT COUNT(*) FROM budget_reservations "
                "WHERE reservation_id = :reservation_id"
            ),
            {"reservation_id": reservation_id},
        )
        scope_count = connection.scalar(
            text(
                "SELECT COUNT(*) FROM budget_reservation_scopes "
                "WHERE reservation_id = :reservation_id"
            ),
            {"reservation_id": reservation_id},
        )

    assert reservation_count == 0
    assert scope_count == 0


def test_concurrent_sessions_cannot_reserve_the_last_request_slot(
    migrated_engine: Engine,
) -> None:
    """Two independent PostgreSQL transactions admit only one final request."""
    scope_id = str(uuid4())
    policy = _policy(scope_id=scope_id, max_requests=2)
    with Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).reserve(_request(policy=policy))
        session.commit()

    barrier = Barrier(2)

    def try_reservation(request: BudgetReservationRequest) -> tuple[str, int]:
        with Session(migrated_engine) as session:
            backend_pid = session.scalar(text("SELECT pg_backend_pid()"))
            assert isinstance(backend_pid, int)
            barrier.wait(timeout=10)
            try:
                PostgreSQLBudgetReservationLedger(session).reserve(request)
                session.commit()
                return "reserved", backend_pid
            except BudgetDeniedError:
                session.rollback()
                return "denied", backend_pid

    requests = (_request(policy=policy), _request(policy=policy))
    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(try_reservation, requests))

    assert [outcome for outcome, _pid in outcomes].count("reserved") == 1
    assert [outcome for outcome, _pid in outcomes].count("denied") == 1
    assert len({pid for _outcome, pid in outcomes}) == 2

    with migrated_engine.connect() as connection:
        scope_count = connection.scalar(
            text(
                "SELECT COUNT(*) FROM budget_reservation_scopes "
                "WHERE scope_kind = 'principal' AND scope_id = :scope_id"
            ),
            {"scope_id": scope_id},
        )
    assert scope_count == 2


def test_overlapping_multi_scope_reservations_lock_in_stable_order(
    migrated_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reversed policy order cannot deadlock or over-admit shared scopes."""
    first_policy = _policy(scope_id=str(uuid4()), max_requests=2)
    second_policy = _policy(scope_id=str(uuid4()), max_requests=2)
    estimate = _request(policy=first_policy).estimate

    def request_for(policies: tuple[BudgetPolicy, ...]) -> BudgetReservationRequest:
        return BudgetReservationRequest(
            reservation_id=uuid4(),
            policies=policies,
            estimate=estimate,
        )

    seed_request = request_for((first_policy, second_policy))
    with Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).reserve(seed_request)
        session.commit()

    requests = {
        "forward": request_for((first_policy, second_policy)),
        "reversed": request_for((second_policy, first_policy)),
    }
    start_barrier = Barrier(2)
    operation = local()
    recorded_scope_locks: dict[str, list[str]] = {}
    recorded_lock_guard = Lock()
    acquire_lock = PostgreSQLBudgetReservationLedger._acquire_advisory_lock

    def record_lock_order(
        ledger: PostgreSQLBudgetReservationLedger,
        domain: str,
        key: str,
    ) -> None:
        request_name = getattr(operation, "request_name", None)
        if domain == "scope" and isinstance(request_name, str):
            with recorded_lock_guard:
                recorded_scope_locks.setdefault(request_name, []).append(key)
        acquire_lock(ledger, domain, key)

    monkeypatch.setattr(
        PostgreSQLBudgetReservationLedger,
        "_acquire_advisory_lock",
        record_lock_order,
    )

    def reserve(request_name: str) -> tuple[str, BudgetDenialCode | None, int]:
        operation.request_name = request_name
        with Session(migrated_engine) as session:
            session.execute(text("SET LOCAL statement_timeout = '5s'"))
            backend_pid = session.scalar(text("SELECT pg_backend_pid()"))
            assert isinstance(backend_pid, int)
            start_barrier.wait(timeout=10)
            try:
                PostgreSQLBudgetReservationLedger(session).reserve(
                    requests[request_name]
                )
                session.commit()
                return "reserved", None, backend_pid
            except BudgetDeniedError as error:
                session.rollback()
                if not isinstance(error.code, BudgetDenialCode):
                    raise AssertionError(
                        "The denial code must use its typed enum."
                    ) from error
                return "denied", error.code, backend_pid

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(reserve, request_name)
            for request_name in requests
        ]
        outcomes = {
            request_name: future.result(timeout=10)
            for request_name, future in zip(requests, futures, strict=True)
        }

    assert len({outcome[2] for outcome in outcomes.values()}) == 2
    assert [outcome[0] for outcome in outcomes.values()].count("reserved") == 1
    assert [outcome[0] for outcome in outcomes.values()].count("denied") == 1
    assert all(
        outcome[1] in (None, BudgetDenialCode.REQUEST_LIMIT_EXCEEDED)
        for outcome in outcomes.values()
    )

    expected_lock_order = sorted(
        f"{policy.scope.kind.value}\0{policy.scope.scope_id}"
        for policy in (first_policy, second_policy)
    )
    assert recorded_scope_locks == {
        request_name: expected_lock_order for request_name in requests
    }

    with migrated_engine.connect() as connection:
        scope_counts = connection.execute(
            text(
                "SELECT scope_id, COUNT(*) FROM budget_reservation_scopes "
                "WHERE scope_kind = 'principal' AND scope_id IN "
                "(:first_scope, :second_scope) GROUP BY scope_id"
            ),
            {
                "first_scope": first_policy.scope.scope_id,
                "second_scope": second_policy.scope.scope_id,
            },
        ).all()
        request_scope_counts = connection.execute(
            text(
                "SELECT reservation_id, COUNT(*) FROM budget_reservation_scopes "
                "WHERE reservation_id IN (:forward_id, :reversed_id) "
                "GROUP BY reservation_id"
            ),
            {
                "forward_id": requests["forward"].reservation_id,
                "reversed_id": requests["reversed"].reservation_id,
            },
        ).all()

    assert {scope_id: count for scope_id, count in scope_counts} == {
        first_policy.scope.scope_id: 2,
        second_policy.scope.scope_id: 2,
    }
    admitted_id = next(
        requests[name].reservation_id
        for name, outcome in outcomes.items()
        if outcome[0] == "reserved"
    )
    assert len(request_scope_counts) == 1
    assert request_scope_counts[0][0] == admitted_id
    assert request_scope_counts[0][1] == 2


def _assert_transition_serializes_with_admission(
    *,
    migrated_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
    transition: Callable[[PostgreSQLBudgetReservationLedger], object],
    admission_request: BudgetReservationRequest,
    expected_denial: BudgetDenialCode,
) -> None:
    """Hold a lifecycle scope lock while an independent session seeks admission."""
    transition_scope_locked = Event()
    admission_scope_attempted = Event()
    release_transition = Event()
    operation = local()
    acquire_lock = PostgreSQLBudgetReservationLedger._acquire_advisory_lock

    def observe_lock(
        ledger: PostgreSQLBudgetReservationLedger,
        domain: str,
        key: str,
    ) -> None:
        if domain == "scope" and getattr(operation, "name", None) == "admission":
            admission_scope_attempted.set()
        acquire_lock(ledger, domain, key)
        if domain == "scope" and getattr(operation, "name", None) == "transition":
            transition_scope_locked.set()
            if not release_transition.wait(timeout=10):
                raise TimeoutError("The test did not release the lifecycle lock.")

    monkeypatch.setattr(
        PostgreSQLBudgetReservationLedger,
        "_acquire_advisory_lock",
        observe_lock,
    )

    def run_transition() -> None:
        operation.name = "transition"
        with Session(migrated_engine) as session:
            transition(PostgreSQLBudgetReservationLedger(session))
            session.commit()

    def run_admission() -> BudgetDenialCode | None:
        operation.name = "admission"
        with Session(migrated_engine) as session:
            try:
                PostgreSQLBudgetReservationLedger(session).reserve(admission_request)
                session.commit()
                return None
            except BudgetDeniedError as error:
                session.rollback()
                if not isinstance(error.code, BudgetDenialCode):
                    raise AssertionError(
                        "The denial code must use its typed enum."
                    ) from error
                return error.code

    with ThreadPoolExecutor(max_workers=2) as executor:
        transition_future = executor.submit(run_transition)
        if not transition_scope_locked.wait(timeout=5):
            release_transition.set()
            transition_future.result(timeout=5)
            pytest.fail("The lifecycle transition did not acquire its scope lock.")

        try:
            admission_future = executor.submit(run_admission)
            assert admission_scope_attempted.wait(timeout=5)
        finally:
            release_transition.set()

        transition_future.result(timeout=10)
        assert admission_future.result(timeout=10) is expected_denial


def test_dispatch_scope_lock_serializes_with_reservation_admission(
    migrated_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dispatch transition and a competing admission share the scope lock."""
    policy = _policy(max_requests=5, max_tokens=1000)
    dispatched_request = _request(policy=policy)
    with Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).reserve(dispatched_request)
        session.commit()

    _assert_transition_serializes_with_admission(
        migrated_engine=migrated_engine,
        monkeypatch=monkeypatch,
        transition=lambda ledger: ledger.mark_dispatched(
            dispatched_request.reservation_id
        ),
        admission_request=_request(policy=policy),
        expected_denial=BudgetDenialCode.TOKEN_CONSUMPTION_UNKNOWN,
    )


def test_settlement_scope_lock_serializes_with_reservation_admission(
    migrated_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settled actual attempts are visible before another scope admission."""
    base_policy = _policy(max_requests=5)
    policy = BudgetPolicy.model_validate(
        {
            **base_policy.model_dump(mode="json"),
            "retry_budget": {"max_retries": 1},
            "provider_attempt_budget": {"max_attempts": 2},
        }
    )
    dispatched_request = _request(policy=policy)
    with Session(migrated_engine) as session:
        ledger = PostgreSQLBudgetReservationLedger(session)
        ledger.reserve(dispatched_request)
        session.commit()

    with Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).mark_dispatched(
            dispatched_request.reservation_id
        )
        session.commit()

    settlement = BudgetSettlement(
        reservation_id=dispatched_request.reservation_id,
        consumption=BudgetConsumption(
            provider_requests=1,
            provider_attempts=2,
            retries=1,
            provider_tokens=25,
            spend_usd=None,
        ),
    )
    _assert_transition_serializes_with_admission(
        migrated_engine=migrated_engine,
        monkeypatch=monkeypatch,
        transition=lambda ledger: ledger.settle(settlement),
        admission_request=_request(policy=policy),
        expected_denial=BudgetDenialCode.PROVIDER_ATTEMPT_LIMIT_EXCEEDED,
    )


def test_identical_reservation_request_returns_the_existing_reservation(
    migrated_engine: Engine,
) -> None:
    """An identical retry returns one durable reservation instead of duplicating it."""
    request = _request()
    with Session(migrated_engine) as session:
        first = PostgreSQLBudgetReservationLedger(session).reserve(request)
        session.commit()

    try:
        with Session(migrated_engine) as session:
            retry = PostgreSQLBudgetReservationLedger(session).reserve(request)
            session.commit()
    except IntegrityError:
        pytest.fail("An identical reservation retry must not hit the primary key.")

    assert retry == first
    with migrated_engine.connect() as connection:
        reservation_count = connection.scalar(
            text(
                "SELECT COUNT(*) FROM budget_reservations "
                "WHERE reservation_id = :reservation_id"
            ),
            {"reservation_id": request.reservation_id},
        )
    assert reservation_count == 1


def test_conflicting_reservation_id_reuse_is_rejected_without_mutation(
    migrated_engine: Engine,
) -> None:
    """A reservation ID cannot be reassigned to a different estimate."""
    reservation_id = uuid4()
    policy = _policy()
    original = _request(reservation_id, policy=policy)
    with Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).reserve(original)
        session.commit()

    conflicting = _request(
        reservation_id,
        policy=policy,
        provider_tokens=26,
    )
    with pytest.raises(BudgetReservationConflictError), Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).reserve(conflicting)

    with migrated_engine.connect() as connection:
        stored_tokens = connection.scalar(
            text(
                "SELECT estimated_provider_tokens FROM budget_reservations "
                "WHERE reservation_id = :reservation_id"
            ),
            {"reservation_id": reservation_id},
        )
    assert stored_tokens == 25


def test_dispatch_transition_is_durable(
    migrated_engine: Engine,
) -> None:
    """Marking dispatch persists the transition before external execution."""
    request = _request()
    with Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).reserve(request)
        session.commit()

    with Session(migrated_engine) as session:
        dispatched = PostgreSQLBudgetReservationLedger(session).mark_dispatched(
            request.reservation_id
        )
        session.commit()

    assert dispatched.state is BudgetReservationState.DISPATCHED
    with migrated_engine.connect() as connection:
        persisted_state = connection.execute(
            text(
                "SELECT state, dispatched_at FROM budget_reservations "
                "WHERE reservation_id = :reservation_id"
            ),
            {"reservation_id": request.reservation_id},
        ).one()
    assert persisted_state.state == BudgetReservationState.DISPATCHED.value
    assert persisted_state.dispatched_at is not None


def test_release_before_dispatch_persists_terminal_state(
    migrated_engine: Engine,
) -> None:
    """A pre-dispatch release is durable and records its transition time."""
    request = _request()
    with Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).reserve(request)
        session.commit()

    with Session(migrated_engine) as session:
        released = PostgreSQLBudgetReservationLedger(session).release(
            request.reservation_id
        )
        session.commit()

    assert released.state is BudgetReservationState.RELEASED
    with migrated_engine.connect() as connection:
        persisted_state = connection.execute(
            text(
                "SELECT state, released_at FROM budget_reservations "
                "WHERE reservation_id = :reservation_id"
            ),
            {"reservation_id": request.reservation_id},
        ).one()
    assert persisted_state.state == BudgetReservationState.RELEASED.value
    assert persisted_state.released_at is not None


def test_unknown_consumption_after_dispatch_is_not_refunded(
    migrated_engine: Engine,
) -> None:
    """Unknown dispatched token usage blocks later admissions in the scope."""
    policy = _policy(max_requests=5, max_tokens=1000)
    request = _request(policy=policy)
    with Session(migrated_engine) as session:
        ledger = PostgreSQLBudgetReservationLedger(session)
        ledger.reserve(request)
        session.commit()

    with Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).mark_dispatched(
            request.reservation_id
        )
        session.commit()

    unknown_settlement = BudgetSettlement(
        reservation_id=request.reservation_id,
        consumption=BudgetConsumption(
            provider_requests=1,
            provider_attempts=1,
            retries=0,
            provider_tokens=None,
            spend_usd=None,
        ),
    )
    with Session(migrated_engine) as session:
        settled = PostgreSQLBudgetReservationLedger(session).settle(
            unknown_settlement
        )
        session.commit()
    assert settled.state is BudgetReservationState.SETTLED

    with pytest.raises(BudgetDeniedError) as captured, Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).reserve(
            _request(policy=policy)
        )
    assert captured.value.code is BudgetDenialCode.TOKEN_CONSUMPTION_UNKNOWN


def test_fixed_window_reservations_share_the_database_aligned_window(
    migrated_engine: Engine,
) -> None:
    """An active reservation consumes capacity in its persisted fixed window."""
    scope_id = str(uuid4())
    base_policy = _policy(scope_id=scope_id, max_requests=1)
    policy = BudgetPolicy.model_validate(
        {
            **base_policy.model_dump(mode="json"),
            "window": {"kind": "fixed_window", "interval_seconds": 3600},
        }
    )
    with Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).reserve(_request(policy=policy))
        session.commit()

    with pytest.raises(BudgetDeniedError) as captured, Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).reserve(
            _request(policy=policy)
        )
    assert captured.value.code is BudgetDenialCode.REQUEST_LIMIT_EXCEEDED

    with migrated_engine.connect() as connection:
        stored_window = connection.scalar(
            text(
                "SELECT window_start FROM budget_reservation_scopes "
                "WHERE scope_kind = 'principal' AND scope_id = :scope_id"
            ),
            {"scope_id": scope_id},
        )
    assert stored_window is not None
    assert int(stored_window.timestamp()) % 3600 == 0


def test_identical_settlement_is_idempotent(
    migrated_engine: Engine,
) -> None:
    """Repeating the same settlement returns the durable terminal result."""
    request = _request()
    settlement = BudgetSettlement(
        reservation_id=request.reservation_id,
        consumption=BudgetConsumption(
            provider_requests=1,
            provider_attempts=1,
            retries=0,
            provider_tokens=18,
            spend_usd=None,
        ),
    )
    with Session(migrated_engine) as session:
        ledger = PostgreSQLBudgetReservationLedger(session)
        ledger.reserve(request)
        session.commit()

    with Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).mark_dispatched(
            request.reservation_id
        )
        session.commit()

    with Session(migrated_engine) as session:
        first = PostgreSQLBudgetReservationLedger(session).settle(settlement)
        session.commit()
    with migrated_engine.connect() as connection:
        first_settled_at = connection.scalar(
            text(
                "SELECT settled_at FROM budget_reservations "
                "WHERE reservation_id = :reservation_id"
            ),
            {"reservation_id": request.reservation_id},
        )

    with Session(migrated_engine) as session:
        retry = PostgreSQLBudgetReservationLedger(session).settle(settlement)
        session.commit()

    with migrated_engine.connect() as connection:
        retry_settled_at = connection.scalar(
            text(
                "SELECT settled_at FROM budget_reservations "
                "WHERE reservation_id = :reservation_id"
            ),
            {"reservation_id": request.reservation_id},
        )
    assert retry == first
    assert retry_settled_at == first_settled_at


def test_dispatch_transition_is_idempotent(
    migrated_engine: Engine,
) -> None:
    """Repeating dispatch does not change the durable transition timestamp."""
    request = _request()
    with Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).reserve(request)
        session.commit()

    with Session(migrated_engine) as session:
        first = PostgreSQLBudgetReservationLedger(session).mark_dispatched(
            request.reservation_id
        )
        session.commit()
    with migrated_engine.connect() as connection:
        first_dispatched_at = connection.scalar(
            text(
                "SELECT dispatched_at FROM budget_reservations "
                "WHERE reservation_id = :reservation_id"
            ),
            {"reservation_id": request.reservation_id},
        )

    with Session(migrated_engine) as session:
        retry = PostgreSQLBudgetReservationLedger(session).mark_dispatched(
            request.reservation_id
        )
        session.commit()
    with migrated_engine.connect() as connection:
        retry_dispatched_at = connection.scalar(
            text(
                "SELECT dispatched_at FROM budget_reservations "
                "WHERE reservation_id = :reservation_id"
            ),
            {"reservation_id": request.reservation_id},
        )

    assert first.state is BudgetReservationState.DISPATCHED
    assert retry == first
    assert retry_dispatched_at == first_dispatched_at


def test_releasing_a_reservation_returns_its_capacity(
    migrated_engine: Engine,
) -> None:
    """A reservation released before dispatch no longer consumes capacity."""
    policy = _policy(max_requests=1)
    first_request = _request(policy=policy)
    with Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).reserve(first_request)
        session.commit()

    with Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).release(
            first_request.reservation_id
        )
        session.commit()

    with Session(migrated_engine) as session:
        replacement = PostgreSQLBudgetReservationLedger(session).reserve(
            _request(policy=policy)
        )
        session.commit()
    assert replacement.state is BudgetReservationState.RESERVED


def test_release_after_dispatch_is_rejected_without_refund(
    migrated_engine: Engine,
) -> None:
    """A dispatched reservation cannot be released or returned to capacity."""
    policy = _policy(max_requests=5)
    request = _request(policy=policy)
    with Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).reserve(request)
        session.commit()
    with Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).mark_dispatched(
            request.reservation_id
        )
        session.commit()

    with pytest.raises(BudgetReservationStateError), Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).release(
            request.reservation_id
        )

    with migrated_engine.connect() as connection:
        state = connection.scalar(
            text(
                "SELECT state FROM budget_reservations "
                "WHERE reservation_id = :reservation_id"
            ),
            {"reservation_id": request.reservation_id},
        )
    assert state == BudgetReservationState.DISPATCHED.value


def test_rolled_back_reservation_does_not_leak_capacity(
    migrated_engine: Engine,
) -> None:
    """Rolling back an uncommitted reservation leaves no durable usage."""
    policy = _policy(max_requests=1)
    request = _request(policy=policy)
    with Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).reserve(request)
        session.rollback()

    with Session(migrated_engine) as session:
        reservation = PostgreSQLBudgetReservationLedger(session).reserve(request)
        session.commit()
    assert reservation.state is BudgetReservationState.RESERVED

    with migrated_engine.connect() as connection:
        row_count = connection.scalar(
            text(
                "SELECT COUNT(*) FROM budget_reservations "
                "WHERE reservation_id = :reservation_id"
            ),
            {"reservation_id": request.reservation_id},
        )
    assert row_count == 1


@pytest.mark.parametrize(
    ("transition_name", "initial_state", "expected_state"),
    [
        (
            "dispatch",
            BudgetReservationState.RESERVED,
            BudgetReservationState.DISPATCHED,
        ),
        (
            "settlement",
            BudgetReservationState.DISPATCHED,
            BudgetReservationState.SETTLED,
        ),
        (
            "release",
            BudgetReservationState.RESERVED,
            BudgetReservationState.RELEASED,
        ),
    ],
)
def test_lifecycle_transition_rollback_is_invisible_to_independent_sessions(
    migrated_engine: Engine,
    transition_name: str,
    initial_state: BudgetReservationState,
    expected_state: BudgetReservationState,
) -> None:
    """Uncommitted lifecycle changes disappear for independent readers."""
    request = _request()
    with Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).reserve(request)
        session.commit()

    if transition_name == "settlement":
        with Session(migrated_engine) as session:
            PostgreSQLBudgetReservationLedger(session).mark_dispatched(
                request.reservation_id
            )
            session.commit()

    settlement = BudgetSettlement(
        reservation_id=request.reservation_id,
        consumption=BudgetConsumption(
            provider_requests=1,
            provider_attempts=1,
            retries=0,
            provider_tokens=18,
            spend_usd=None,
        ),
    )
    lifecycle_statement = text(
        "SELECT pg_backend_pid() AS backend_pid, state, dispatched_at, released_at, "
        "settled_at, consumption_fingerprint, consumption_payload, "
        "consumed_provider_requests, consumed_provider_attempts, consumed_retries, "
        "consumed_provider_tokens, consumed_spend_usd "
        "FROM budget_reservations WHERE reservation_id = :reservation_id"
    )

    with Session(migrated_engine) as transition_session:
        transition_pid = transition_session.scalar(text("SELECT pg_backend_pid()"))
        assert isinstance(transition_pid, int)

        ledger = PostgreSQLBudgetReservationLedger(transition_session)
        if transition_name == "dispatch":
            transitioned = ledger.mark_dispatched(request.reservation_id)
        elif transition_name == "settlement":
            transitioned = ledger.settle(settlement)
        else:
            transitioned = ledger.release(request.reservation_id)
        assert transitioned.state is expected_state

        with Session(migrated_engine) as observer:
            before_rollback = observer.execute(
                lifecycle_statement,
                {"reservation_id": request.reservation_id},
            ).one()

        assert before_rollback.backend_pid != transition_pid
        assert before_rollback.state == initial_state.value
        transition_session.rollback()

    with Session(migrated_engine) as observer:
        after_rollback = observer.execute(
            lifecycle_statement,
            {"reservation_id": request.reservation_id},
        ).one()

    assert after_rollback.backend_pid != transition_pid
    assert after_rollback.state == initial_state.value
    assert (
        before_rollback.dispatched_at,
        before_rollback.released_at,
        before_rollback.settled_at,
        before_rollback.consumption_fingerprint,
        before_rollback.consumption_payload,
        before_rollback.consumed_provider_requests,
        before_rollback.consumed_provider_attempts,
        before_rollback.consumed_retries,
        before_rollback.consumed_provider_tokens,
        before_rollback.consumed_spend_usd,
    ) == (
        after_rollback.dispatched_at,
        after_rollback.released_at,
        after_rollback.settled_at,
        after_rollback.consumption_fingerprint,
        after_rollback.consumption_payload,
        after_rollback.consumed_provider_requests,
        after_rollback.consumed_provider_attempts,
        after_rollback.consumed_retries,
        after_rollback.consumed_provider_tokens,
        after_rollback.consumed_spend_usd,
    )


def test_dispatched_reservation_survives_engine_recreation_conservatively(
    migrated_engine: Engine,
) -> None:
    """A fresh engine recovers dispatched state without refunding unknown tokens."""
    policy = _policy(max_requests=5)
    request = _request(policy=policy)
    with Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).reserve(request)
        session.commit()
    with Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).mark_dispatched(
            request.reservation_id
        )
        session.commit()

    restarted_engine = create_engine(migrated_engine.url)
    try:
        with Session(restarted_engine) as session:
            recovered = PostgreSQLBudgetReservationLedger(session).reserve(request)
            session.commit()
        assert recovered.state is BudgetReservationState.DISPATCHED

        with pytest.raises(BudgetDeniedError) as captured, Session(restarted_engine) as session:
            PostgreSQLBudgetReservationLedger(session).reserve(
                _request(policy=policy)
            )
        assert captured.value.code is BudgetDenialCode.TOKEN_CONSUMPTION_UNKNOWN
    finally:
        restarted_engine.dispose()


def test_settlement_replaces_the_estimate_with_actual_consumption(
    migrated_engine: Engine,
) -> None:
    """Settled actual usage replaces, rather than duplicates, its reservation."""
    policy = _policy(max_tokens=40)
    request = _request(policy=policy, provider_tokens=25)
    with Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).reserve(request)
        session.commit()
    with Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).mark_dispatched(
            request.reservation_id
        )
        session.commit()

    settlement = BudgetSettlement(
        reservation_id=request.reservation_id,
        consumption=BudgetConsumption(
            provider_requests=1,
            provider_attempts=1,
            retries=0,
            provider_tokens=18,
            spend_usd=None,
        ),
    )
    with Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).settle(settlement)
        session.commit()

    with Session(migrated_engine) as session:
        at_limit = PostgreSQLBudgetReservationLedger(session).reserve(
            _request(policy=policy, provider_tokens=22)
        )
        session.commit()
    assert at_limit.state is BudgetReservationState.RESERVED

    with pytest.raises(BudgetDeniedError) as captured, Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).reserve(
            _request(policy=policy, provider_tokens=1)
        )
    assert captured.value.code is BudgetDenialCode.TOKEN_LIMIT_EXCEEDED
