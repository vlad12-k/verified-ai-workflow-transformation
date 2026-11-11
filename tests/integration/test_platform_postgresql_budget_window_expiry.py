"""PostgreSQL integration coverage for fixed-window dispatch expiry."""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import timedelta
from decimal import Decimal
from ipaddress import ip_address
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from test_platform_postgresql_budget_ledger import _policy, _request

from vait.contracts.budgeting import (
    BudgetConsumption,
    BudgetDenialCode,
    BudgetDeniedError,
    BudgetEstimate,
    BudgetPolicy,
    BudgetReservationRequest,
    BudgetReservationState,
    BudgetSettlement,
    PriceSourceReference,
)
from vait.platform.persistence.repositories.budgets import (
    BudgetReservationStateError,
    PostgreSQLBudgetReservationLedger,
)


@pytest.fixture
def migrated_engine(monkeypatch: pytest.MonkeyPatch) -> Iterator[Engine]:
    """Apply migrations only to an explicitly named local test database."""
    database_url = os.getenv("VAIT_TEST_DATABASE_URL")
    if database_url is None:
        pytest.skip("VAIT_TEST_DATABASE_URL is not configured")

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
    repository_root = Path(__file__).resolve().parents[2]
    command.upgrade(Config(str(repository_root / "alembic.ini")), "head")
    engine = create_engine(database_url)
    try:
        yield engine
    finally:
        engine.dispose()


def test_fixed_window_reservation_cannot_dispatch_after_expiry(
    migrated_engine: Engine,
) -> None:
    """A fixed-window reservation cannot dispatch against an expired window."""
    base_policy = _policy(scope_id=str(uuid4()))
    policy = BudgetPolicy.model_validate(
        {
            **base_policy.model_dump(mode="json"),
            "window": {"kind": "fixed_window", "interval_seconds": 3600},
        }
    )
    request = _request(policy=policy)
    with Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).reserve(request)
        session.commit()

    with migrated_engine.begin() as connection:
        interval_seconds = connection.scalar(
            text(
                "SELECT interval_seconds FROM budget_reservation_scopes "
                "WHERE reservation_id = :reservation_id"
            ),
            {"reservation_id": request.reservation_id},
        )
        assert interval_seconds == 3600
        connection.execute(
            text(
                "UPDATE budget_reservation_scopes "
                "SET window_start = window_start - :interval_seconds * INTERVAL '1 second' "
                "WHERE reservation_id = :reservation_id"
            ),
            {
                "interval_seconds": interval_seconds,
                "reservation_id": request.reservation_id,
            },
        )
        connection.execute(
            text(
                "UPDATE budget_reservations "
                "SET created_at = created_at - :interval_seconds * INTERVAL '1 second' "
                "WHERE reservation_id = :reservation_id"
            ),
            {
                "interval_seconds": interval_seconds,
                "reservation_id": request.reservation_id,
            },
        )

    with pytest.raises(BudgetReservationStateError), Session(
        migrated_engine
    ) as session:
        PostgreSQLBudgetReservationLedger(session).mark_dispatched(
            request.reservation_id
        )


def test_fixed_window_includes_start_and_excludes_end_boundaries(
    migrated_engine: Engine,
) -> None:
    """A reservation at the start counts; one at the end belongs to the next window."""
    base_policy = _policy(scope_id=str(uuid4()), max_requests=1)
    policy = BudgetPolicy.model_validate(
        {
            **base_policy.model_dump(mode="json"),
            "window": {"kind": "fixed_window", "interval_seconds": 3600},
        }
    )
    first_request = _request(policy=policy)
    with Session(migrated_engine) as session:
        PostgreSQLBudgetReservationLedger(session).reserve(first_request)
        session.commit()

    with migrated_engine.begin() as connection:
        window_start = connection.scalar(
            text(
                "SELECT window_start FROM budget_reservation_scopes "
                "WHERE reservation_id = :reservation_id"
            ),
            {"reservation_id": first_request.reservation_id},
        )
        assert window_start is not None
        window_end = window_start + timedelta(seconds=3600)
        connection.execute(
            text(
                "UPDATE budget_reservations SET created_at = :created_at "
                "WHERE reservation_id = :reservation_id"
            ),
            {
                "created_at": window_start,
                "reservation_id": first_request.reservation_id,
            },
        )

    with pytest.raises(BudgetDeniedError) as captured, Session(
        migrated_engine
    ) as session:
        PostgreSQLBudgetReservationLedger(session).reserve(_request(policy=policy))
    assert captured.value.code is BudgetDenialCode.REQUEST_LIMIT_EXCEEDED

    with migrated_engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE budget_reservations SET created_at = :created_at "
                "WHERE reservation_id = :reservation_id"
            ),
            {
                "created_at": window_end,
                "reservation_id": first_request.reservation_id,
            },
        )

    with Session(migrated_engine) as session:
        reservation = PostgreSQLBudgetReservationLedger(session).reserve(
            _request(policy=policy)
        )
        session.commit()

    assert reservation.state is BudgetReservationState.RESERVED


def test_unknown_spend_after_dispatch_is_not_treated_as_zero(
    migrated_engine: Engine,
) -> None:
    """Unknown settled cost blocks later admissions under a monetary limit."""
    price_source = PriceSourceReference(
        source_id="postgres-test-pricing",
        source_version="1",
        approval_reference="test-only",
    )
    base_policy = _policy(scope_id=str(uuid4()))
    policy = BudgetPolicy.model_validate(
        {
            **base_policy.model_dump(mode="json"),
            "monetary_limit": {
                "max_cost_usd": "2.00",
                "price_source": price_source.model_dump(mode="json"),
            },
        }
    )

    def monetary_request() -> BudgetReservationRequest:
        return BudgetReservationRequest(
            reservation_id=uuid4(),
            policies=(policy,),
            estimate=BudgetEstimate(
                provider_requests=1,
                provider_attempts=1,
                retries=0,
                provider_tokens=25,
                token_estimator_id="postgres-test-estimator",
                token_estimator_version="1",
                estimated_spend_usd=Decimal("0.25"),
                price_source=price_source,
                spend_estimator_id="postgres-test-cost-estimator",
                spend_estimator_version="1",
            ),
        )

    request = monetary_request()
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

    with pytest.raises(BudgetDeniedError) as captured, Session(
        migrated_engine
    ) as session:
        PostgreSQLBudgetReservationLedger(session).reserve(monetary_request())
    assert captured.value.code is BudgetDenialCode.SPEND_CONSUMPTION_UNKNOWN
