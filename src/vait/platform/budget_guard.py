"""Budget-controlled execution of one provider inference attempt."""

from collections.abc import Sequence
from decimal import Decimal
from uuid import uuid4

from vait.contracts.budgeting import (
    BudgetConsumption,
    BudgetEstimate,
    BudgetPolicy,
    BudgetReservationRequest,
    BudgetReservationState,
    BudgetSettlement,
)
from vait.inference.providers.base import InferenceProvider
from vait.inference.providers.models import (
    ProviderInferenceRequest,
    ProviderInferenceResponse,
)
from vait.platform.persistence.engine import DatabaseRuntime
from vait.platform.persistence.repositories.budgets import (
    PostgreSQLBudgetReservationLedger,
)
from vait.platform.persistence.session import transactional_session


class BudgetedProviderDispatcher:
    """Enforce durable budget admission before one provider call.

    Policies and estimates must come from trusted platform context.

    This component performs exactly one provider attempt.
    It does not implement retries or native batch operations.
    """

    def __init__(
        self,
        *,
        provider: InferenceProvider,
        runtime: DatabaseRuntime,
    ) -> None:
        self._provider = provider
        self._runtime = runtime

    def execute(
        self,
        *,
        request: ProviderInferenceRequest,
        policies: Sequence[BudgetPolicy],
        estimate: BudgetEstimate,
    ) -> ProviderInferenceResponse:
        """Reserve, dispatch, execute and settle separately."""

        if (
            estimate.provider_requests != 1
            or estimate.provider_attempts != 1
            or estimate.retries != 0
        ):
            raise ValueError(
                "Single-attempt dispatch requires a one-attempt estimate."
            )

        reservation_request = BudgetReservationRequest(
            reservation_id=uuid4(),
            policies=tuple(policies),
            estimate=estimate,
        )

        reservation_id = reservation_request.reservation_id

        # Transaction 1: durable budget admission.
        with transactional_session(self._runtime) as session:
            ledger = PostgreSQLBudgetReservationLedger(session)
            reserved = ledger.reserve(reservation_request)

            if reserved.state is not BudgetReservationState.RESERVED:
                raise RuntimeError(
                    "Budget admission did not produce a reserved state."
                )

        # Transaction 2: persist dispatch BEFORE provider I/O.
        with transactional_session(self._runtime) as session:
            ledger = PostgreSQLBudgetReservationLedger(session)
            dispatched = ledger.mark_dispatched(reservation_id)

            if dispatched.state is not BudgetReservationState.DISPATCHED:
                raise RuntimeError(
                    "Budget dispatch did not produce a dispatched state."
                )

        response: ProviderInferenceResponse | None = None

        try:
            result = self._provider.generate(request)

            if (
                result.request_id != request.request_id
                or result.model_id != request.model_id
                or result.provider_id != self._provider.provider_id
                or result.runtime_id != self._provider.runtime_id
            ):
                raise ValueError(
                    "Provider response identity mismatch."
                )

            response = result

        finally:
            # Unknown consumption must never be treated as zero.
            consumption = BudgetConsumption(
                provider_requests=1,
                provider_attempts=1,
                retries=0,
                provider_tokens=(
                    response.usage.total_tokens
                    if response is not None
                    and response.usage is not None
                    else None
                ),
                spend_usd=(
                    Decimal(str(response.cost_usd))
                    if response is not None
                    and response.cost_usd is not None
                    else None
                ),
            )

            # Transaction 3: reconcile actual or unknown usage.
            with transactional_session(self._runtime) as session:
                ledger = PostgreSQLBudgetReservationLedger(session)
                ledger.settle(
                    BudgetSettlement(
                        reservation_id=reservation_id,
                        consumption=consumption,
                    )
                )

        return response
