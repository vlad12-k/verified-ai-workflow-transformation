"""Tests for fail-closed future trusted verification resolution."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import JsonValue

from vait.contracts.models import (
    RiskConstraint,
    RiskLevel,
    TransformationContract,
    VerificationCase,
)
from vait.platform.jobs.models import JobRecord
from vait.platform.jobs.verification_resolution import (
    resolve_trusted_verification_job,
)
from vait.platform.registry.contracts import (
    CandidateRecord,
    ExperimentRunRecord,
)
from vait.runners.base import ImplementationRunner
from vait.runners.python_runner import PythonImplementationRunner
from vait.transformations.library.synthetic_ap import synthetic_ap_policy
from vait.verification.trusted_bundle import TrustedVerificationBundle
from vait.verification.trusted_implementations import (
    TrustedImplementationRegistration,
    TrustedImplementationRegistry,
)

_SOURCE_REVISION = "a" * 40
_EXPERIMENT_ID = uuid4()
_REFERENCE_ID = "synthetic-ap-policy-v0.2"
_CANDIDATE_ID = "candidate-policy"


def _experiment(
    *,
    experiment_id: UUID = _EXPERIMENT_ID,
    contract_id: str = "trusted-contract",
    contract_version: str = "1.0.0",
    source_revision: str = _SOURCE_REVISION,
) -> ExperimentRunRecord:
    return ExperimentRunRecord(
        experiment_id=experiment_id,
        source_revision=source_revision,
        dataset_fingerprint="b" * 64,
        dataset_partition="validation-v1",
        transformation_contract_id=contract_id,
        transformation_contract_version=contract_version,
        created_at=datetime.now(UTC),
    )


def _bundle(
    *,
    experiment_id: UUID = _EXPERIMENT_ID,
    source_revision: str = _SOURCE_REVISION,
    contract_id: str = "trusted-contract",
    contract_version: str = "1.0.0",
) -> TrustedVerificationBundle:
    contract = TransformationContract(
        id=contract_id,
        reference_implementation_id=_REFERENCE_ID,
        candidate_implementation_id=_CANDIDATE_ID,
        risk=RiskConstraint(level=RiskLevel.LOW),
    )
    return TrustedVerificationBundle(
        experiment_id=experiment_id,
        contract=contract,
        cases=(
            VerificationCase(
                id="case-1",
                input_data={"invoice_amount": 1},
            ),
        ),
        contract_version=contract_version,
        source_revision=source_revision,
    )


class _Experiments:
    def __init__(self, record: ExperimentRunRecord | None) -> None:
        self.record = record
        self.calls = 0

    def get(self, entity_id: UUID) -> ExperimentRunRecord | None:
        self.calls += 1
        return self.record


class _Bundles:
    def __init__(self, bundle: TrustedVerificationBundle | None) -> None:
        self.bundle = bundle
        self.calls = 0

    def get(self, entity_id: UUID) -> TrustedVerificationBundle | None:
        self.calls += 1
        return self.bundle


class _Candidates:
    def __init__(self, records: tuple[CandidateRecord, ...]) -> None:
        self.records = records
        self.calls = 0

    def list_for_experiment(self, experiment_id: UUID) -> tuple[CandidateRecord, ...]:
        self.calls += 1
        return self.records


def _candidate() -> CandidateRecord:
    return CandidateRecord(
        candidate_id=uuid4(),
        experiment_id=_EXPERIMENT_ID,
        candidate_identity=_CANDIDATE_ID,
        candidate_version="2.1.0",
        created_at=datetime.now(UTC),
    )


def _registry(factory_calls: list[str] | None = None) -> TrustedImplementationRegistry:
    def reference_factory() -> ImplementationRunner:
        if factory_calls is not None:
            factory_calls.append("reference")
        return PythonImplementationRunner(
            implementation_id=_REFERENCE_ID,
            function=synthetic_ap_policy,
        )

    def candidate_factory() -> ImplementationRunner:
        if factory_calls is not None:
            factory_calls.append("candidate")
        def unused(_data: dict[str, JsonValue]) -> JsonValue:
            raise AssertionError("Resolution must not execute candidate code")

        return PythonImplementationRunner(
            implementation_id=_CANDIDATE_ID,
            function=unused,
        )

    return TrustedImplementationRegistry(
        (
            TrustedImplementationRegistration(
                implementation_id=_REFERENCE_ID,
                version="1.0.0",
                factory=reference_factory,
            ),
            TrustedImplementationRegistration(
                implementation_id=_CANDIDATE_ID,
                version="2.1.0",
                factory=candidate_factory,
            ),
        )
    )


def _job(
    *,
    experiment_id: UUID = _EXPERIMENT_ID,
    operation: str = "verification",
    payload: dict[str, JsonValue] | None = None,
) -> JobRecord:
    now = datetime.now(UTC)
    return JobRecord(
        job_id=uuid4(),
        experiment_id=experiment_id,
        operation=operation,
        payload=payload or {},
        status="PENDING",
        idempotency_key=str(uuid4()),
        created_at=now,
    )


def test_future_resolution_reads_trusted_lineage_then_resolves_without_execution() -> None:
    experiments = _Experiments(_experiment())
    bundles = _Bundles(_bundle())
    candidates = _Candidates((_candidate(),))
    factory_calls: list[str] = []
    registry = _registry(factory_calls)

    result = resolve_trusted_verification_job(
        _job(),
        experiment_repository=experiments,
        bundle_repository=bundles,
        candidate_repository=candidates,
        implementation_registry=registry,
    )

    assert result.experiment.experiment_id == _EXPERIMENT_ID
    assert result.bundle.experiment_id == _EXPERIMENT_ID
    assert result.reference.implementation_id == _REFERENCE_ID
    assert result.candidate.implementation_id == _CANDIDATE_ID
    assert experiments.calls == 1
    assert bundles.calls == 1
    assert candidates.calls == 1
    assert factory_calls == ["reference", "candidate"]


def test_payload_controlled_import_or_callable_instructions_are_rejected() -> None:
    experiments = _Experiments(_experiment())
    bundles = _Bundles(_bundle())
    candidates = _Candidates((_candidate(),))
    factory_calls: list[str] = []

    with pytest.raises(ValueError, match="empty payload"):
        resolve_trusted_verification_job(
            _job(
                payload={
                    "module": "os",
                    "callable": "system",
                    "arguments": ["not-executed"],
                }
            ),
            experiment_repository=experiments,
            bundle_repository=bundles,
            candidate_repository=candidates,
            implementation_registry=_registry(factory_calls),
        )

    assert experiments.calls == 0
    assert bundles.calls == 0
    assert candidates.calls == 0
    assert factory_calls == []


def test_unrecognized_job_operation_is_rejected_before_repository_reads() -> None:
    experiments = _Experiments(_experiment())
    bundles = _Bundles(_bundle())
    candidates = _Candidates((_candidate(),))

    with pytest.raises(ValueError, match="verification operation"):
        resolve_trusted_verification_job(
            _job(operation="provider-inference"),
            experiment_repository=experiments,
            bundle_repository=bundles,
            candidate_repository=candidates,
            implementation_registry=_registry(),
        )

    assert experiments.calls == 0


def test_persisted_experiment_identity_mismatch_fails_before_bundle_or_resolution() -> None:
    experiments = _Experiments(_experiment(experiment_id=uuid4()))
    bundles = _Bundles(_bundle())
    candidates = _Candidates((_candidate(),))
    factory_calls: list[str] = []

    with pytest.raises(ValueError, match="identity"):
        resolve_trusted_verification_job(
            _job(),
            experiment_repository=experiments,
            bundle_repository=bundles,
            candidate_repository=candidates,
            implementation_registry=_registry(factory_calls),
        )

    assert bundles.calls == 0
    assert candidates.calls == 0
    assert factory_calls == []


def test_missing_experiment_or_bundle_fails_before_resolution() -> None:
    for experiments, bundles in (
        (_Experiments(None), _Bundles(_bundle())),
        (_Experiments(_experiment()), _Bundles(None)),
    ):
        factory_calls: list[str] = []
        with pytest.raises(ValueError):
            resolve_trusted_verification_job(
                _job(),
                experiment_repository=experiments,
                bundle_repository=bundles,
                candidate_repository=_Candidates((_candidate(),)),
                implementation_registry=_registry(factory_calls),
            )
        assert factory_calls == []


@pytest.mark.parametrize(
    "lineage_mismatch",
    [
        "experiment_id",
        "source_revision",
        "contract_id",
        "contract_version",
    ],
)
def test_wrong_bundle_lineage_fails_before_implementation_factories(
    lineage_mismatch: str,
) -> None:
    experiment = _experiment()
    bundle = _bundle()
    if lineage_mismatch == "experiment_id":
        bundle = _bundle(experiment_id=uuid4())
    elif lineage_mismatch == "source_revision":
        bundle = _bundle(source_revision="c" * 40)
    elif lineage_mismatch == "contract_id":
        bundle = _bundle(contract_id="other-contract")
    else:
        bundle = _bundle(contract_version="2.0.0")
    experiments = _Experiments(experiment)
    bundles = _Bundles(bundle)
    candidates = _Candidates((_candidate(),))
    factory_calls: list[str] = []

    with pytest.raises(ValueError, match="lineage"):
        resolve_trusted_verification_job(
            _job(),
            experiment_repository=experiments,
            bundle_repository=bundles,
            candidate_repository=candidates,
            implementation_registry=_registry(factory_calls),
        )

    assert candidates.calls == 0
    assert factory_calls == []


def test_missing_or_ambiguous_persisted_candidate_version_fails_closed() -> None:
    for candidate_rows in (
        (),
        (
            _candidate(),
            _candidate(),
        ),
    ):
        candidates = _Candidates(candidate_rows)
        factory_calls: list[str] = []
        with pytest.raises(ValueError, match="candidate"):
            resolve_trusted_verification_job(
                _job(),
                experiment_repository=_Experiments(_experiment()),
                bundle_repository=_Bundles(_bundle()),
                candidate_repository=candidates,
                implementation_registry=_registry(factory_calls),
            )
        assert factory_calls == []


def test_candidate_from_another_experiment_is_rejected() -> None:
    candidate = _candidate().model_copy(update={"experiment_id": uuid4()})
    factory_calls: list[str] = []

    with pytest.raises(ValueError, match="another experiment"):
        resolve_trusted_verification_job(
            _job(),
            experiment_repository=_Experiments(_experiment()),
            bundle_repository=_Bundles(_bundle()),
            candidate_repository=_Candidates((candidate,)),
            implementation_registry=_registry(factory_calls),
        )

    assert factory_calls == []
