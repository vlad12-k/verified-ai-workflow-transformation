"""PostgreSQL integration tests for immutable trusted verification bundles."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from ipaddress import ip_address
from pathlib import Path
from threading import Barrier
from typing import Protocol, cast
from uuid import UUID, uuid4

import pytest
from alembic import command
from alembic.config import Config
from pydantic import SecretStr
from sqlalchemy import create_engine, text, update
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError, SQLAlchemyError
from sqlalchemy.orm import Session

from vait.contracts.models import (
    Effect,
    ExactVerificationScope,
    RiskConstraint,
    RiskLevel,
    TransformationContract,
    VerificationCase,
)
from vait.platform.persistence import (
    DatabaseRuntime,
    create_database_runtime,
    transactional_session,
)
from vait.platform.persistence import repositories as persistence_repositories
from vait.platform.persistence.registry_models import (
    ExperimentRunRow,
    TrustedVerificationBundleRow,
)
from vait.platform.persistence.repositories import (
    ExperimentRunRepository,
)
from vait.platform.registry.contracts import ExperimentRunRecord
from vait.platform.settings import PlatformSettings
from vait.verification.trusted_bundle import TrustedVerificationBundle

_DATABASE_ENV = "VAIT_TEST_REGISTRY_DATABASE_URL"
_MARKER_ENV = "VAIT_TEST_REGISTRY_DATABASE_MARKER"
_ROOT = Path(__file__).resolve().parents[2]
_ALEMBIC_INI = _ROOT / "alembic.ini"
_EXPECTED_HEAD = "0005_m6d3b_trusted_bundle"
_FORBIDDEN_LIBPQ_ROUTING = {"hostaddr", "service", "servicefile"}
_FORBIDDEN_ROUTING_ENV = (
    "PGHOST",
    "PGHOSTADDR",
    "PGPORT",
    "PGSERVICE",
    "PGSERVICEFILE",
)


class _BundleRepository(Protocol):
    """Narrow test contract for the public bundle repository API."""

    def add(self, bundle: TrustedVerificationBundle) -> None: ...

    def get(self, experiment_id: UUID) -> TrustedVerificationBundle | None: ...


def _is_allowed_test_server(
    database_url: str,
    server_address: str | None,
) -> bool:
    """Allow a local socket/loopback DB or the existing isolated CI service."""
    if server_address is None:
        return True
    try:
        address = ip_address(server_address)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    if os.getenv("GITHUB_ACTIONS") != "true" or not address.is_private:
        return False
    try:
        client_address = ip_address(make_url(database_url).host or "")
    except ValueError:
        return False
    return client_address.is_loopback


def _preflight_test_database(database_url: str) -> str:
    """Check a dedicated test target before any migration or write."""
    try:
        url = make_url(database_url)
    except (ArgumentError, TypeError, ValueError):
        pytest.fail("Bundle integration tests require a valid PostgreSQL test URL")
    if url.drivername != "postgresql+psycopg" or not url.database:
        pytest.fail("Bundle integration tests require a PostgreSQL test database")
    if not url.username:
        pytest.fail("Bundle integration tests require an explicit database user")
    if _FORBIDDEN_LIBPQ_ROUTING.intersection(url.query):
        pytest.fail("Bundle integration tests reject libpq service/address routing")
    if any(os.getenv(name) for name in _FORBIDDEN_ROUTING_ENV):
        pytest.fail("Bundle integration tests reject inherited libpq routing")
    if not url.database.endswith(("_test", "_ci")):
        pytest.fail("Bundle integration tests require an explicitly named test database")

    host = url.host
    socket_hosts = url.query.get("host")
    socket_host: str | None
    if isinstance(socket_hosts, tuple):
        if len(socket_hosts) != 1:
            pytest.fail("Bundle integration tests reject multiple socket hosts")
        socket_host = socket_hosts[0]
    else:
        socket_host = socket_hosts
    socket_path = (
        socket_host
        if isinstance(socket_host, str) and socket_host.startswith("/")
        else None
    )

    expected_marker: str | None = None
    if socket_path is not None:
        expected_marker = os.getenv(_MARKER_ENV, "")
        if re.fullmatch(r"vait-d3b2b-[0-9a-f]{12}", expected_marker) is None:
            pytest.fail("Bundle integration tests require a valid task database marker")
        supplied_path = Path(socket_path)
        scratch_value = os.getenv("TMPDIR")
        if not scratch_value:
            pytest.fail("Bundle integration tests require the active profile scratch path")
        scratch = Path(scratch_value).resolve(strict=True)
        scratch_stat = scratch.stat()
        if scratch_stat.st_uid != os.geteuid() or scratch_stat.st_mode & 0o077:
            pytest.fail("Bundle integration scratch directory must be owner-only")
        for parent in scratch.parents:
            if parent.stat().st_mode & 0o022:
                pytest.fail("Bundle integration scratch path has a writable parent")
        try:
            resolved_path = supplied_path.resolve(strict=True)
        except OSError:
            pytest.fail("Bundle integration tests require an existing private socket directory")
        if not supplied_path.is_absolute() or supplied_path != resolved_path:
            pytest.fail("Bundle integration tests reject socket path aliases")
        if not resolved_path.is_relative_to(scratch):
            pytest.fail("Bundle integration socket must be inside the active profile scratch area")
        socket_stat = resolved_path.stat()
        if socket_stat.st_uid != os.geteuid() or socket_stat.st_mode & 0o077:
            pytest.fail("Bundle integration socket directory must be owner-only")
    elif host is None or host not in {"localhost", "127.0.0.1", "::1"}:
        pytest.fail("Bundle integration tests require a local socket or loopback URL")

    marker_value: object | None = None
    probe_engine = create_engine(url)
    try:
        with probe_engine.connect() as connection:
            connection.exec_driver_sql("SET TRANSACTION READ ONLY")
            identity = connection.execute(
                text(
                    "SELECT current_database(), current_user, "
                    "current_setting('server_encoding'), "
                    "current_setting('client_encoding'), host(inet_server_addr()), "
                    "current_setting('listen_addresses')"
                )
            ).one()
            if socket_path is not None:
                marker_value = connection.scalar(
                    text(
                        "SELECT marker_value "
                        "FROM public.vait_test_database_marker "
                        "WHERE singleton IS TRUE"
                    )
                )
            connection.rollback()
    except SQLAlchemyError:
        pytest.fail(
            "Bundle integration database identity or marker preflight failed",
            pytrace=False,
        )
    finally:
        probe_engine.dispose()

    (
        database_name,
        database_user,
        server_encoding,
        client_encoding,
        server_address,
        listen_addresses,
    ) = identity
    if database_name != url.database:
        pytest.fail("Bundle integration database identity does not match its URL")
    if database_user != url.username:
        pytest.fail("Bundle integration user identity does not match its URL")
    if server_encoding != "UTF8" or client_encoding != "UTF8":
        pytest.fail("Bundle integration tests require UTF-8 server and client encodings")
    normalized_address = str(server_address) if server_address is not None else None
    if not _is_allowed_test_server(database_url, normalized_address):
        pytest.fail("Bundle integration tests require a local or isolated CI PostgreSQL server")
    if socket_path is not None and (normalized_address is not None or listen_addresses != ""):
        pytest.fail("Bundle integration socket target must have no TCP listener")
    if socket_path is not None and marker_value != expected_marker:
        pytest.fail("Bundle integration database task marker does not match")
    return url.render_as_string(hide_password=False)


def test_local_database_preflight_rejects_wrong_task_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local destructive tests require the pre-existing task-owned marker."""
    database_url = os.getenv(_DATABASE_ENV)
    if database_url is None:
        pytest.skip(f"{_DATABASE_ENV} is not configured")
    url = make_url(database_url)
    socket_path = url.query.get("host")
    if not isinstance(socket_path, str) or not socket_path.startswith("/"):
        pytest.skip("Task-marker preflight applies to local socket databases")

    monkeypatch.setenv(
        _MARKER_ENV,
        "vait-d3b2b-000000000000",
    )
    with pytest.raises(pytest.fail.Exception, match="marker"):
        _preflight_test_database(database_url)


def _repository(session: Session) -> _BundleRepository:
    """Resolve the repository through its public persistence package export."""
    repository_type = getattr(
        persistence_repositories,
        "TrustedVerificationBundleRepository",
        None,
    )
    assert callable(repository_type), "TrustedVerificationBundleRepository is not exported"
    return cast(_BundleRepository, repository_type(session))


def _repository_error(name: str) -> type[ValueError]:
    """Resolve a typed public repository rejection error."""
    error_type = getattr(persistence_repositories, name, None)
    assert isinstance(error_type, type) and issubclass(error_type, ValueError)
    return error_type


@pytest.fixture

def bundle_runtime(monkeypatch: pytest.MonkeyPatch) -> Iterator[DatabaseRuntime]:
    """Use a preflighted, explicitly named PostgreSQL test database."""
    database_url = os.getenv(_DATABASE_ENV)
    if database_url is None:
        pytest.skip(f"{_DATABASE_ENV} is not configured")

    database_url = _preflight_test_database(database_url)
    monkeypatch.setenv("VAIT_DATABASE_URL", database_url)
    command.upgrade(Config(str(_ALEMBIC_INI)), "head")

    runtime = create_database_runtime(
        PlatformSettings(
            environment="test",
            database_url=SecretStr(database_url),
        )
    )
    try:
        with runtime.engine.connect() as connection:
            revision = connection.scalar(text("SELECT version_num FROM alembic_version"))
        assert revision == _EXPECTED_HEAD
        yield runtime
    finally:
        runtime.dispose()


def _contract() -> TransformationContract:
    """Build one contract whose declared cases are complete and finite."""
    return TransformationContract(
        id="invoice-normalisation-v1",
        reference_implementation_id="reference-v1",
        candidate_implementation_id="candidate-v1",
        risk=RiskConstraint(level=RiskLevel.HIGH),
        exact_verification_scope=ExactVerificationScope(
            exhaustive_case_ids=frozenset({"case-1", "case-2"}),
        ),
        allowed_effects={Effect.NONE, Effect.READ},
    )


def _cases() -> list[VerificationCase]:
    """Build a small finite input set without external data or providers."""
    return [
        VerificationCase(
            id="case-1",
            input_data={"invoice": {"amount": 12}},
            risk_level=RiskLevel.LOW,
        ),
        VerificationCase(
            id="case-2",
            input_data={"invoice": {"amount": 34}},
            risk_level=RiskLevel.HIGH,
        ),
    ]


def _experiment(experiment_id: UUID) -> ExperimentRunRecord:
    """Build persisted lineage matching the bundle fixtures."""
    return ExperimentRunRecord(
        experiment_id=experiment_id,
        source_revision="a" * 40,
        dataset_fingerprint="b" * 64,
        dataset_partition="verification-fixture-v1",
        transformation_contract_id="invoice-normalisation-v1",
        transformation_contract_version="1.0",
        created_at=datetime.now(UTC),
    )


def _bundle(experiment_id: UUID) -> TrustedVerificationBundle:
    """Build a canonical trusted snapshot bound to the experiment fixture."""
    return TrustedVerificationBundle(
        experiment_id=experiment_id,
        contract=_contract(),
        cases=_cases(),
        contract_version="1.0",
        source_revision="a" * 40,
    )


def test_bundle_insert_and_read_preserve_exact_canonical_content(
    bundle_runtime: DatabaseRuntime,
) -> None:
    """A persisted bundle reconstructs exactly from canonical serialized text."""
    experiment_id = uuid4()
    experiment = _experiment(experiment_id)
    bundle = _bundle(experiment_id)

    with transactional_session(bundle_runtime) as session:
        ExperimentRunRepository(session).add(experiment)
        _repository(session).add(bundle)

    with transactional_session(bundle_runtime) as session:
        reconstructed = _repository(session).get(experiment_id)
        row = session.get(TrustedVerificationBundleRow, experiment_id)

    assert reconstructed is not None
    assert reconstructed.to_json() == bundle.to_json()
    assert reconstructed.fingerprint_sha256 == bundle.fingerprint_sha256
    assert row is not None
    assert row.canonical_json == bundle.to_json()
    assert row.contract_id == _contract().id
    assert row.contract_version == experiment.transformation_contract_version
    assert row.source_revision == experiment.source_revision


def test_duplicate_registration_is_rejected_without_changing_committed_bundle(
    bundle_runtime: DatabaseRuntime,
) -> None:
    """A repeated insert is explicit and cannot overwrite the first snapshot."""
    experiment_id = uuid4()
    experiment = _experiment(experiment_id)
    bundle = _bundle(experiment_id)

    with transactional_session(bundle_runtime) as session:
        ExperimentRunRepository(session).add(experiment)
        _repository(session).add(bundle)

    with (
        pytest.raises(_repository_error("DuplicateTrustedVerificationBundleError")),
        transactional_session(bundle_runtime) as session,
    ):
        _repository(session).add(bundle)

    with transactional_session(bundle_runtime) as session:
        reconstructed = _repository(session).get(experiment_id)
        row = session.get(TrustedVerificationBundleRow, experiment_id)

    assert reconstructed is not None
    assert reconstructed.to_json() == bundle.to_json()
    assert row is not None
    assert row.canonical_json == bundle.to_json()


def test_conflicting_registration_is_rejected_without_replacing_bundle(
    bundle_runtime: DatabaseRuntime,
) -> None:
    """Different canonical content cannot replace the committed bundle."""
    experiment_id = uuid4()
    experiment = _experiment(experiment_id)
    original = _bundle(experiment_id)
    altered_cases = _cases()
    altered_cases[0] = VerificationCase(
        id="case-1",
        input_data={"invoice": {"amount": 999}},
        risk_level=RiskLevel.LOW,
    )
    conflicting = TrustedVerificationBundle(
        experiment_id=experiment_id,
        contract=_contract(),
        cases=altered_cases,
        contract_version=experiment.transformation_contract_version,
        source_revision=experiment.source_revision,
    )

    with transactional_session(bundle_runtime) as session:
        ExperimentRunRepository(session).add(experiment)
        _repository(session).add(original)

    with (
        pytest.raises(_repository_error("ConflictingTrustedVerificationBundleError")),
        transactional_session(bundle_runtime) as session,
    ):
        _repository(session).add(conflicting)

    with transactional_session(bundle_runtime) as session:
        reconstructed = _repository(session).get(experiment_id)
        row = session.get(TrustedVerificationBundleRow, experiment_id)

    assert reconstructed is not None
    assert reconstructed.to_json() == original.to_json()
    assert row is not None
    assert row.canonical_json == original.to_json()


def test_bundle_registration_rejects_unknown_experiment(
    bundle_runtime: DatabaseRuntime,
) -> None:
    """A bundle cannot introduce an experiment absent from durable lineage."""
    bundle = _bundle(uuid4())

    with (
        pytest.raises(_repository_error("UnknownExperimentError")),
        transactional_session(bundle_runtime) as session,
    ):
        _repository(session).add(bundle)

    with transactional_session(bundle_runtime) as session:
        assert _repository(session).get(bundle.experiment_id) is None
        assert session.get(
            TrustedVerificationBundleRow,
            bundle.experiment_id,
        ) is None


@pytest.mark.parametrize(
    "mismatch",
    ("contract_id", "contract_version", "source_revision"),
)
def test_bundle_registration_rejects_experiment_lineage_mismatch(
    bundle_runtime: DatabaseRuntime,
    mismatch: str,
) -> None:
    """Persisted experiment identity controls the bundle's lineage."""
    experiment_id = uuid4()
    experiment = _experiment(experiment_id)
    contract_payload = _contract().model_dump(mode="python")
    contract_version = experiment.transformation_contract_version
    source_revision = experiment.source_revision
    if mismatch == "contract_id":
        contract_payload["id"] = "unrelated-contract"
    elif mismatch == "contract_version":
        contract_version = "2.0"
    else:
        source_revision = "c" * 40

    bundle = TrustedVerificationBundle(
        experiment_id=experiment_id,
        contract=TransformationContract.model_validate(contract_payload),
        cases=_cases(),
        contract_version=contract_version,
        source_revision=source_revision,
    )

    with transactional_session(bundle_runtime) as session:
        ExperimentRunRepository(session).add(experiment)

    with (
        pytest.raises(_repository_error("TrustedVerificationBundleLineageError")),
        transactional_session(bundle_runtime) as session,
    ):
        _repository(session).add(bundle)

    with transactional_session(bundle_runtime) as session:
        assert session.get(
            TrustedVerificationBundleRow,
            experiment_id,
        ) is None


def test_bundle_read_rejects_fingerprint_column_tampering(
    bundle_runtime: DatabaseRuntime,
) -> None:
    """A matching JSON digest cannot mask a modified persisted fingerprint."""
    experiment_id = uuid4()
    experiment = _experiment(experiment_id)
    bundle = _bundle(experiment_id)
    altered_fingerprint = (
        "0" * 64
        if bundle.fingerprint_sha256 != "0" * 64
        else "1" * 64
    )

    with transactional_session(bundle_runtime) as session:
        ExperimentRunRepository(session).add(experiment)
        _repository(session).add(bundle)

    with transactional_session(bundle_runtime) as session:
        session.execute(
            update(TrustedVerificationBundleRow)
            .where(TrustedVerificationBundleRow.experiment_id == experiment_id)
            .values(fingerprint_sha256=altered_fingerprint)
        )

    with pytest.raises(
        _repository_error("PersistedTrustedVerificationBundleIntegrityError"),
    ), transactional_session(bundle_runtime) as session:
        _repository(session).get(experiment_id)


def test_bundle_read_rejects_canonical_content_tampering(
    bundle_runtime: DatabaseRuntime,
) -> None:
    """A valid-looking row cannot hide content changed without its fingerprint."""
    experiment_id = uuid4()
    experiment = _experiment(experiment_id)
    bundle = _bundle(experiment_id)
    tampered_json = bundle.to_json().replace('"amount":12', '"amount":13')
    assert tampered_json != bundle.to_json()

    with transactional_session(bundle_runtime) as session:
        ExperimentRunRepository(session).add(experiment)
        _repository(session).add(bundle)

    with transactional_session(bundle_runtime) as session:
        session.execute(
            update(TrustedVerificationBundleRow)
            .where(TrustedVerificationBundleRow.experiment_id == experiment_id)
            .values(canonical_json=tampered_json)
        )

    with pytest.raises(
        _repository_error("PersistedTrustedVerificationBundleIntegrityError"),
    ), transactional_session(bundle_runtime) as session:
        _repository(session).get(experiment_id)


def test_bundle_read_rejects_drifted_persisted_experiment_lineage(
    bundle_runtime: DatabaseRuntime,
) -> None:
    """Read-back revalidates bundle lineage against current persisted experiment."""
    experiment_id = uuid4()
    experiment = _experiment(experiment_id)
    bundle = _bundle(experiment_id)

    with transactional_session(bundle_runtime) as session:
        ExperimentRunRepository(session).add(experiment)
        _repository(session).add(bundle)

    with transactional_session(bundle_runtime) as session:
        session.execute(
            update(ExperimentRunRow)
            .where(ExperimentRunRow.experiment_id == experiment_id)
            .values(source_revision="c" * 40)
        )

    with (
        pytest.raises(_repository_error("TrustedVerificationBundleLineageError")),
        transactional_session(bundle_runtime) as session,
    ):
        _repository(session).get(experiment_id)


def test_bundle_get_refreshes_experiment_lineage_from_database(
    bundle_runtime: DatabaseRuntime,
) -> None:
    """An identity-map copy cannot hide newer experiment lineage in the DB."""
    experiment_id = uuid4()
    experiment = _experiment(experiment_id)
    bundle = _bundle(experiment_id)

    with transactional_session(bundle_runtime) as session:
        ExperimentRunRepository(session).add(experiment)
        _repository(session).add(bundle)

    with (
        pytest.raises(_repository_error("TrustedVerificationBundleLineageError")),
        transactional_session(bundle_runtime) as session,
    ):
        cached_experiment = session.get(ExperimentRunRow, experiment_id)
        assert cached_experiment is not None
        assert cached_experiment.source_revision == "a" * 40
        with bundle_runtime.engine.begin() as connection:
            connection.execute(
                update(ExperimentRunRow)
                .where(ExperimentRunRow.experiment_id == experiment_id)
                .values(source_revision="c" * 40)
            )
        with bundle_runtime.engine.connect() as connection:
            persisted_revision = connection.scalar(
                text(
                    "SELECT source_revision FROM experiment_runs "
                    "WHERE experiment_id = :experiment_id"
                ),
                {"experiment_id": experiment_id},
            )
        assert persisted_revision == "c" * 40
        assert cached_experiment.source_revision == "a" * 40
        _repository(session).get(experiment_id)


@pytest.mark.parametrize("tampering", ("fingerprint", "bundle_version"))
def test_bundle_registration_rejects_invalid_integrity_metadata(
    bundle_runtime: DatabaseRuntime,
    tampering: str,
) -> None:
    """Registration refuses unsupported versions and tampered fingerprints."""
    experiment_id = uuid4()
    experiment = _experiment(experiment_id)
    bundle = _bundle(experiment_id)
    if tampering == "fingerprint":
        altered_fingerprint = (
            "0" * 64
            if bundle.fingerprint_sha256 != "0" * 64
            else "1" * 64
        )
        object.__setattr__(bundle, "fingerprint_sha256", altered_fingerprint)
    else:
        object.__setattr__(bundle, "bundle_version", "unsupported")

    with transactional_session(bundle_runtime) as session:
        ExperimentRunRepository(session).add(experiment)

    with (
        pytest.raises(_repository_error("TrustedVerificationBundleIntegrityError")),
        transactional_session(bundle_runtime) as session,
    ):
        _repository(session).add(bundle)

    with transactional_session(bundle_runtime) as session:
        assert session.get(
            TrustedVerificationBundleRow,
            experiment_id,
        ) is None


def test_bundle_insert_obeys_caller_transaction_rollback(
    bundle_runtime: DatabaseRuntime,
) -> None:
    """A rolled-back caller transaction leaves no durable bundle row."""
    experiment_id = uuid4()
    experiment = _experiment(experiment_id)
    bundle = _bundle(experiment_id)

    with transactional_session(bundle_runtime) as session:
        ExperimentRunRepository(session).add(experiment)

    with (
        pytest.raises(RuntimeError, match="forced rollback"),
        transactional_session(bundle_runtime) as session,
    ):
        _repository(session).add(bundle)
        raise RuntimeError("forced rollback")

    with transactional_session(bundle_runtime) as session:
        assert _repository(session).get(experiment_id) is None
        assert session.get(
            TrustedVerificationBundleRow,
            experiment_id,
        ) is None


def test_concurrent_registration_keeps_exactly_one_immutable_bundle(
    bundle_runtime: DatabaseRuntime,
) -> None:
    """Concurrent registrations produce one insert and explicit duplicates."""
    experiment_id = uuid4()
    experiment = _experiment(experiment_id)
    bundle = _bundle(experiment_id)
    workers = 4
    start = Barrier(workers)

    with transactional_session(bundle_runtime) as session:
        ExperimentRunRepository(session).add(experiment)

    def register() -> str:
        start.wait(timeout=10)
        try:
            with transactional_session(bundle_runtime) as session:
                _repository(session).add(bundle)
        except _repository_error("DuplicateTrustedVerificationBundleError"):
            return "duplicate"
        return "inserted"

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(register) for _ in range(workers)]
        outcomes = tuple(future.result(timeout=20) for future in futures)

    assert outcomes.count("inserted") == 1
    assert outcomes.count("duplicate") == workers - 1
    with transactional_session(bundle_runtime) as session:
        reconstructed = _repository(session).get(experiment_id)
    assert reconstructed is not None
    assert reconstructed.to_json() == bundle.to_json()


def test_bundle_read_rejects_mismatched_persisted_experiment_identity(
    bundle_runtime: DatabaseRuntime,
) -> None:
    """A valid canonical digest cannot change the row's experiment identity."""
    experiment_id = uuid4()
    experiment = _experiment(experiment_id)
    bundle = _bundle(experiment_id)
    unrelated_bundle = _bundle(uuid4())

    with transactional_session(bundle_runtime) as session:
        ExperimentRunRepository(session).add(experiment)
        _repository(session).add(bundle)

    with transactional_session(bundle_runtime) as session:
        session.execute(
            update(TrustedVerificationBundleRow)
            .where(TrustedVerificationBundleRow.experiment_id == experiment_id)
            .values(
                canonical_json=unrelated_bundle.to_json(),
                fingerprint_sha256=unrelated_bundle.fingerprint_sha256,
            )
        )

    with pytest.raises(
        _repository_error("PersistedTrustedVerificationBundleIntegrityError"),
    ), transactional_session(bundle_runtime) as session:
        _repository(session).get(experiment_id)
