"""PostgreSQL implementation of the platform budget reservation ledger."""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import cast
from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from vait.contracts.budgeting import (
    BudgetConsumption,
    BudgetDeniedError,
    BudgetPolicy,
    BudgetPolicyReference,
    BudgetReservation,
    BudgetReservationAmounts,
    BudgetReservationRequest,
    BudgetReservationState,
    BudgetSettlement,
    BudgetUsageSnapshot,
    BudgetWindowKind,
    evaluate_budget,
)
from vait.platform.persistence.budget_models import (
    BudgetReservationRow,
    BudgetReservationScopeRow,
)


class BudgetReservationStateError(RuntimeError):
    """Raised when the ledger cannot safely persist a reservation."""


class BudgetReservationConflictError(BudgetReservationStateError):
    """Raised when a reservation ID is reused with different input."""


class BudgetReservationNotFoundError(BudgetReservationStateError):
    """Raised when a lifecycle operation references an unknown reservation."""


class BudgetReservationWindowExpiredError(BudgetReservationStateError):
    """Raised when fixed-window admission expires before dispatch."""


class PostgreSQLBudgetReservationLedger:
    """Persist budget reservations in caller-owned SQLAlchemy transactions.

    Callers must commit or roll back each operation promptly. Commit the
    reservation and dispatch transition before external provider work; never
    hold the database transaction open while that work runs.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def reserve(self, request: BudgetReservationRequest) -> BudgetReservation:
        """Atomically evaluate and stage all requested scopes for commit."""
        if self._session.get_bind().dialect.name != "postgresql":
            raise BudgetReservationStateError(
                "The budget ledger requires a PostgreSQL session."
            )

        validated_request = BudgetReservationRequest.model_validate(
            request.model_dump(mode="json")
        )
        request_payload = cast(
            dict[str, object],
            validated_request.model_dump(mode="json"),
        )
        canonical_request = json.dumps(
            request_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        request_fingerprint = hashlib.sha256(
            canonical_request.encode("utf-8")
        ).hexdigest()

        isolation = self._session.scalar(text("SHOW transaction_isolation"))
        if isolation != "read committed":
            raise BudgetReservationStateError(
                "The budget ledger requires READ COMMITTED transactions."
            )

        self._acquire_advisory_lock(
            "reservation",
            str(validated_request.reservation_id),
        )
        scope_keys = sorted(
            {
                (policy.scope.kind.value, policy.scope.scope_id)
                for policy in validated_request.policies
            }
        )
        for scope_kind, scope_id in scope_keys:
            self._acquire_advisory_lock(
                "scope",
                f"{scope_kind}\0{scope_id}",
            )

        existing = self._session.scalar(
            select(BudgetReservationRow)
            .where(
                BudgetReservationRow.reservation_id
                == validated_request.reservation_id
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if existing is not None:
            if (
                existing.request_fingerprint != request_fingerprint
                or _canonical_json(existing.request_payload) != canonical_request
            ):
                raise BudgetReservationConflictError(
                    "A reservation ID is already used for different input."
                )
            return _reservation_from_row(existing)

        reservation_time = _database_timestamp(self._session)
        fixed_window_policies = [
            policy
            for policy in validated_request.policies
            if policy.window.kind is BudgetWindowKind.FIXED_WINDOW
        ]
        window_starts: dict[tuple[str, str], datetime] = {}
        for policy in fixed_window_policies:
            interval_seconds = policy.window.interval_seconds
            if interval_seconds is None:
                raise BudgetReservationStateError(
                    "A fixed-window policy has no interval."
                )
            window_starts[
                (policy.scope.kind.value, policy.scope.scope_id)
            ] = _fixed_window_start(reservation_time, interval_seconds)

        estimate = validated_request.estimate
        for policy in validated_request.policies:
            decision = evaluate_budget(
                policy=policy,
                scope=policy.scope,
                usage=self._usage_snapshot(
                    policy,
                    window_start=window_starts.get(
                        (policy.scope.kind.value, policy.scope.scope_id)
                    ),
                ),
                estimate=estimate,
            )
            if not decision.is_allowed:
                raise BudgetDeniedError(decision)

        if estimate.provider_tokens is None:
            raise BudgetReservationStateError(
                "An admitted reservation requires a known token estimate."
            )

        reservation_row = BudgetReservationRow(
            reservation_id=validated_request.reservation_id,
            request_fingerprint=request_fingerprint,
            request_payload=request_payload,
            state=BudgetReservationState.RESERVED.value,
            created_at=reservation_time,
            estimated_provider_requests=estimate.provider_requests,
            estimated_provider_attempts=estimate.provider_attempts,
            estimated_retries=estimate.retries,
            estimated_provider_tokens=estimate.provider_tokens,
            estimated_spend_usd=estimate.estimated_spend_usd,
        )
        self._session.add(reservation_row)
        self._session.add_all(
            [
                BudgetReservationScopeRow(
                    reservation_id=validated_request.reservation_id,
                    scope_kind=policy.scope.kind.value,
                    scope_id=policy.scope.scope_id,
                    policy_id=policy.policy_id,
                    policy_version=policy.policy_version,
                    window_kind=policy.window.kind.value,
                    interval_seconds=policy.window.interval_seconds,
                    window_start=window_starts.get(
                        (policy.scope.kind.value, policy.scope.scope_id)
                    ),
                )
                for policy in validated_request.policies
            ]
        )
        self._session.flush()

        return BudgetReservation(
            reservation_id=validated_request.reservation_id,
            policies=tuple(
                BudgetPolicyReference(
                    policy_id=policy.policy_id,
                    policy_version=policy.policy_version,
                    scope=policy.scope,
                )
                for policy in validated_request.policies
            ),
            amounts=BudgetReservationAmounts(
                provider_requests=estimate.provider_requests,
                provider_attempts=estimate.provider_attempts,
                retries=estimate.retries,
                provider_tokens=estimate.provider_tokens,
                spend_usd=estimate.estimated_spend_usd,
            ),
            state=BudgetReservationState.RESERVED,
        )

    def _acquire_advisory_lock(self, domain: str, key: str) -> None:
        """Acquire a transaction-scoped lock with stable key ordering."""
        digest = hashlib.sha256(
            f"vait-budget-ledger\0{domain}\0{key}".encode()
        ).digest()[:8]
        lock_key = int.from_bytes(digest, byteorder="big", signed=True)
        self._session.execute(
            text("SELECT pg_advisory_xact_lock(CAST(:lock_key AS bigint))"),
            {"lock_key": lock_key},
        )

    def _lock_reservation_scopes(
        self,
        reservation_id: UUID,
    ) -> list[BudgetReservationScopeRow]:
        """Lock every persisted scope for a lifecycle transition in stable order."""
        scope_rows = list(
            self._session.scalars(
                select(BudgetReservationScopeRow)
                .where(
                    BudgetReservationScopeRow.reservation_id == reservation_id
                )
                .order_by(
                    BudgetReservationScopeRow.scope_kind,
                    BudgetReservationScopeRow.scope_id,
                )
            ).all()
        )
        if not scope_rows:
            raise BudgetReservationStateError(
                "A persisted reservation has no budget scopes."
            )

        scope_keys = sorted(
            {(scope_row.scope_kind, scope_row.scope_id) for scope_row in scope_rows}
        )
        for scope_kind, scope_id in scope_keys:
            self._acquire_advisory_lock(
                "scope",
                f"{scope_kind}\0{scope_id}",
            )
        return scope_rows

    def _usage_snapshot(
        self,
        policy: BudgetPolicy,
        *,
        window_start: datetime | None = None,
    ) -> BudgetUsageSnapshot:
        """Aggregate usage and reservations for the exact scope/window."""
        if policy.window.kind is BudgetWindowKind.PER_EXECUTION:
            return _empty_usage_snapshot()
        if (
            policy.window.kind is BudgetWindowKind.FIXED_WINDOW
            and window_start is None
        ):
            raise BudgetReservationStateError(
                "A fixed-window usage snapshot requires its database window."
            )

        statement = (
            select(BudgetReservationRow)
            .join(
                BudgetReservationScopeRow,
                BudgetReservationScopeRow.reservation_id
                == BudgetReservationRow.reservation_id,
            )
            .where(
                BudgetReservationScopeRow.scope_kind == policy.scope.kind.value,
                BudgetReservationScopeRow.scope_id == policy.scope.scope_id,
                BudgetReservationRow.state.in_(
                    (
                        BudgetReservationState.RESERVED.value,
                        BudgetReservationState.DISPATCHED.value,
                        BudgetReservationState.SETTLED.value,
                    )
                ),
            )
        )
        if policy.window.kind is BudgetWindowKind.FIXED_WINDOW:
            if window_start is None:
                raise BudgetReservationStateError(
                    "A fixed-window usage snapshot requires its database window."
                )
            interval_seconds = policy.window.interval_seconds
            if interval_seconds is None:
                raise BudgetReservationStateError(
                    "A fixed-window policy has no interval."
                )
            window_end = window_start + timedelta(seconds=interval_seconds)
            statement = statement.where(
                BudgetReservationRow.created_at >= window_start,
                BudgetReservationRow.created_at < window_end,
            )
        rows = list(self._session.scalars(statement).all())
        consumed_requests = 0
        consumed_attempts = 0
        consumed_retries = 0
        consumed_tokens = 0
        consumed_spend = Decimal("0")
        tokens_unknown = False
        spend_unknown = False
        reserved_requests = 0
        reserved_attempts = 0
        reserved_retries = 0
        reserved_tokens = 0
        reserved_spend = Decimal("0")
        reserved_spend_unknown = False

        for row in rows:
            if row.state == BudgetReservationState.RESERVED.value:
                reserved_requests += row.estimated_provider_requests
                reserved_attempts += row.estimated_provider_attempts
                reserved_retries += row.estimated_retries
                reserved_tokens += row.estimated_provider_tokens
                if row.estimated_spend_usd is None:
                    reserved_spend_unknown = True
                else:
                    reserved_spend += row.estimated_spend_usd
            elif row.state == BudgetReservationState.DISPATCHED.value:
                consumed_requests += row.estimated_provider_requests
                consumed_attempts += row.estimated_provider_attempts
                consumed_retries += row.estimated_retries
                tokens_unknown = True
                spend_unknown = True
            elif row.state == BudgetReservationState.SETTLED.value:
                if (
                    row.consumed_provider_requests is None
                    or row.consumed_provider_attempts is None
                    or row.consumed_retries is None
                ):
                    raise BudgetReservationStateError(
                        "A settled reservation has incomplete usage counters."
                    )
                consumed_requests += row.consumed_provider_requests
                consumed_attempts += row.consumed_provider_attempts
                consumed_retries += row.consumed_retries
                if row.consumed_provider_tokens is None:
                    tokens_unknown = True
                else:
                    consumed_tokens += row.consumed_provider_tokens
                if row.consumed_spend_usd is None:
                    spend_unknown = True
                else:
                    consumed_spend += row.consumed_spend_usd

        return BudgetUsageSnapshot(
            consumed=BudgetConsumption(
                provider_requests=consumed_requests,
                provider_attempts=consumed_attempts,
                retries=consumed_retries,
                provider_tokens=None if tokens_unknown else consumed_tokens,
                spend_usd=None if spend_unknown else consumed_spend,
            ),
            reserved=BudgetReservationAmounts(
                provider_requests=reserved_requests,
                provider_attempts=reserved_attempts,
                retries=reserved_retries,
                provider_tokens=reserved_tokens,
                spend_usd=(
                    None if reserved_spend_unknown else reserved_spend
                ),
            ),
        )

    def mark_dispatched(
        self,
        reservation_id: UUID,
        *,
        require_new_transition: bool = False,
    ) -> BudgetReservation:
        """Stage dispatch; the caller must commit before provider execution."""
        _assert_postgresql_read_committed(self._session)
        self._acquire_advisory_lock("reservation", str(reservation_id))
        row = self._session.scalar(
            select(BudgetReservationRow)
            .where(BudgetReservationRow.reservation_id == reservation_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if row is None:
            raise BudgetReservationNotFoundError(
                "The budget reservation does not exist."
            )

        state = _reservation_state(row.state)
        if state in {
            BudgetReservationState.DISPATCHED,
            BudgetReservationState.SETTLED,
        }:
            if require_new_transition:
                raise BudgetReservationStateError(
                    "Provider dispatch already claimed."
                )
            return _reservation_from_row(row)
        if state is BudgetReservationState.RELEASED:
            raise BudgetReservationStateError(
                "A released reservation cannot be dispatched."
            )

        scope_rows = self._lock_reservation_scopes(reservation_id)
        database_now = _database_timestamp(self._session)
        for scope_row in scope_rows:
            if scope_row.window_kind != BudgetWindowKind.FIXED_WINDOW.value:
                continue
            if scope_row.interval_seconds is None or scope_row.window_start is None:
                raise BudgetReservationStateError(
                    "A fixed-window reservation has incomplete window metadata."
                )
            current_window = _fixed_window_start(
                database_now,
                scope_row.interval_seconds,
            )
            window_end = current_window + timedelta(
                seconds=scope_row.interval_seconds
            )
            if (
                scope_row.window_start != current_window
                or row.created_at < current_window
                or row.created_at >= window_end
            ):
                raise BudgetReservationWindowExpiredError(
                    "The reservation's fixed budget window has expired."
                )

        row.state = BudgetReservationState.DISPATCHED.value
        row.dispatched_at = database_now.astimezone(UTC)
        self._session.flush()
        return _reservation_from_row(row)

    def settle(self, settlement: BudgetSettlement) -> BudgetReservation:
        """Record actual or explicitly unknown consumption."""
        validated_settlement = BudgetSettlement.model_validate(
            settlement.model_dump(mode="json")
        )
        consumption_payload = cast(
            dict[str, object],
            validated_settlement.consumption.model_dump(mode="json"),
        )
        canonical_consumption = _canonical_json(consumption_payload)
        consumption_fingerprint = hashlib.sha256(
            canonical_consumption.encode("utf-8")
        ).hexdigest()

        _assert_postgresql_read_committed(self._session)
        self._acquire_advisory_lock(
            "reservation",
            str(validated_settlement.reservation_id),
        )
        row = self._session.scalar(
            select(BudgetReservationRow)
            .where(
                BudgetReservationRow.reservation_id
                == validated_settlement.reservation_id
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if row is None:
            raise BudgetReservationNotFoundError(
                "The budget reservation does not exist."
            )

        state = _reservation_state(row.state)
        if state is BudgetReservationState.SETTLED:
            if (
                row.consumption_fingerprint != consumption_fingerprint
                or row.consumption_payload is None
                or _canonical_json(row.consumption_payload)
                != canonical_consumption
            ):
                raise BudgetReservationConflictError(
                    "A settled reservation cannot be changed to different consumption."
                )
            return _reservation_from_row(row)
        if state is not BudgetReservationState.DISPATCHED:
            raise BudgetReservationStateError(
                "Only a dispatched reservation may be settled."
            )

        self._lock_reservation_scopes(validated_settlement.reservation_id)
        consumption = validated_settlement.consumption

        # Read database time before changing ORM state to avoid
        # autoflushing an incomplete settled reservation.
        settled_at = _database_timestamp(self._session)

        row.state = BudgetReservationState.SETTLED.value
        row.settled_at = settled_at
        row.consumption_fingerprint = consumption_fingerprint
        row.consumption_payload = consumption_payload
        row.consumed_provider_requests = consumption.provider_requests
        row.consumed_provider_attempts = consumption.provider_attempts
        row.consumed_retries = consumption.retries
        row.consumed_provider_tokens = consumption.provider_tokens
        row.consumed_spend_usd = consumption.spend_usd
        self._session.flush()
        return _reservation_from_row(row)

    def release(self, reservation_id: UUID) -> BudgetReservation:
        """Release capacity only before dispatch."""
        _assert_postgresql_read_committed(self._session)
        self._acquire_advisory_lock("reservation", str(reservation_id))
        row = self._session.scalar(
            select(BudgetReservationRow)
            .where(BudgetReservationRow.reservation_id == reservation_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if row is None:
            raise BudgetReservationNotFoundError(
                "The budget reservation does not exist."
            )

        state = _reservation_state(row.state)
        if state is BudgetReservationState.RELEASED:
            return _reservation_from_row(row)
        if state is not BudgetReservationState.RESERVED:
            raise BudgetReservationStateError(
                "Only a reservation not yet dispatched may be released."
            )

        self._lock_reservation_scopes(reservation_id)
        database_now = self._session.scalar(select(func.statement_timestamp()))
        if not isinstance(database_now, datetime):
            raise BudgetReservationStateError(
                "PostgreSQL did not return a usable database timestamp."
            )
        if database_now.tzinfo is None or database_now.utcoffset() is None:
            raise BudgetReservationStateError(
                "PostgreSQL returned a timestamp without a timezone."
            )

        row.state = BudgetReservationState.RELEASED.value
        row.released_at = database_now.astimezone(UTC)
        self._session.flush()
        return _reservation_from_row(row)


def _empty_usage_snapshot() -> BudgetUsageSnapshot:
    """Return known-zero usage for a scope with no persisted history."""
    return BudgetUsageSnapshot(
        consumed=BudgetConsumption(
            provider_requests=0,
            provider_attempts=0,
            retries=0,
            provider_tokens=0,
            spend_usd=Decimal("0"),
        ),
        reserved=BudgetReservationAmounts(
            provider_requests=0,
            provider_attempts=0,
            retries=0,
            provider_tokens=0,
            spend_usd=Decimal("0"),
        ),
    )


def _canonical_json(payload: dict[str, object]) -> str:
    """Serialise a validated JSON object deterministically."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _assert_postgresql_read_committed(session: Session) -> None:
    """Require the isolation level used by the advisory-lock protocol."""
    if session.get_bind().dialect.name != "postgresql":
        raise BudgetReservationStateError(
            "The budget ledger requires a PostgreSQL session."
        )
    isolation = session.scalar(text("SHOW transaction_isolation"))
    if isolation != "read committed":
        raise BudgetReservationStateError(
            "The budget ledger requires READ COMMITTED transactions."
        )


def _reservation_state(value: str) -> BudgetReservationState:
    """Map a persisted lifecycle state without accepting corrupt values."""
    try:
        return BudgetReservationState(value)
    except ValueError:
        raise BudgetReservationStateError(
            "A reservation has an unsupported persisted lifecycle state."
        ) from None


def _database_timestamp(session: Session) -> datetime:
    """Return an aware UTC timestamp from the active PostgreSQL transaction."""
    database_now = session.scalar(select(func.statement_timestamp()))
    if not isinstance(database_now, datetime):
        raise BudgetReservationStateError(
            "PostgreSQL did not return a usable database timestamp."
        )
    if database_now.tzinfo is None or database_now.utcoffset() is None:
        raise BudgetReservationStateError(
            "PostgreSQL returned a timestamp without a timezone."
        )
    return database_now.astimezone(UTC)


def _fixed_window_start(database_now: datetime, interval_seconds: int) -> datetime:
    """Align a database-provided timestamp to a UTC epoch-based window."""
    if interval_seconds <= 0:
        raise BudgetReservationStateError(
            "A fixed-window interval must be positive."
        )
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    elapsed = database_now.astimezone(UTC) - epoch
    whole_seconds = elapsed.days * 86_400 + elapsed.seconds
    window_start_seconds = (whole_seconds // interval_seconds) * interval_seconds
    return epoch + timedelta(seconds=window_start_seconds)


def _reservation_from_row(row: BudgetReservationRow) -> BudgetReservation:
    """Reconstruct a public reservation using its persisted request snapshot."""
    request = BudgetReservationRequest.model_validate(row.request_payload)
    estimate = request.estimate
    if estimate.provider_tokens is None:
        raise BudgetReservationStateError(
            "A persisted reservation has no known token estimate."
        )
    state = _reservation_state(row.state)

    return BudgetReservation(
        reservation_id=request.reservation_id,
        policies=tuple(
            BudgetPolicyReference(
                policy_id=policy.policy_id,
                policy_version=policy.policy_version,
                scope=policy.scope,
            )
            for policy in request.policies
        ),
        amounts=BudgetReservationAmounts(
            provider_requests=estimate.provider_requests,
            provider_attempts=estimate.provider_attempts,
            retries=estimate.retries,
            provider_tokens=estimate.provider_tokens,
            spend_usd=estimate.estimated_spend_usd,
        ),
        state=state,
    )
