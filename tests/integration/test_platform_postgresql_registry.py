"""Real PostgreSQL tests for the M5-E durable registry."""

import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from vait.platform.persistence import (
    DatabaseRuntime,
    create_database_runtime,
    transactional_session,
)
from vait.platform.persistence.repositories import (
    ArtifactMetadataRepository,
    CandidateRepository,
    EvidenceRepository,
    ExperimentRunRepository,
)
from vait.platform.registry import (
    ArtifactRecord,
    CandidateRecord,
    EvidenceRecord,
    ExperimentRunRecord,
)
from vait.platform.settings import PlatformSettings

_DATABASE_ENV = "VAIT_TEST_REGISTRY_DATABASE_URL"
_ROOT = Path(__file__).resolve().parents[2]
_ALEMBIC_INI = _ROOT / "alembic.ini"


@pytest.fixture
def registry_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[DatabaseRuntime]:
    """Provide a migrated and isolated real PostgreSQL registry."""
    database_url = os.getenv(_DATABASE_ENV)

    if database_url is None:
        pytest.skip(
            f"{_DATABASE_ENV} is not configured"
        )

    monkeypatch.setenv(
        "VAIT_DATABASE_URL",
        database_url,
    )

    command.upgrade(
        Config(str(_ALEMBIC_INI)),
        "head",
    )

    runtime = create_database_runtime(
        PlatformSettings()
    )

    def clean() -> None:
        with runtime.engine.begin() as connection:
            connection.execute(
                text(
                    "TRUNCATE TABLE "
                    "artifacts, "
                    "evidence_records, "
                    "candidates, "
                    "experiment_runs "
                    "CASCADE"
                )
            )

    clean()

    try:
        yield runtime
    finally:
        clean()
        runtime.dispose()


def test_registry_persists_and_reads_full_lineage(
    registry_runtime: DatabaseRuntime,
) -> None:
    """Experiment -> candidate -> evidence -> artifact lineage must round-trip."""
    created_at = datetime.now(UTC)

    experiment = ExperimentRunRecord(
        experiment_id=uuid4(),
        source_revision="a" * 40,
        dataset_fingerprint="b" * 64,
        dataset_partition="validation-v1",
        transformation_contract_id="ap-invoice-rule",
        transformation_contract_version="1.0",
        benchmark_configuration={
            "batch_size": 8,
            "deterministic": True,
        },
        created_at=created_at,
    )

    candidate = CandidateRecord(
        candidate_id=uuid4(),
        experiment_id=experiment.experiment_id,
        candidate_identity="candidate-001",
        candidate_version="1",
        provider_identity="watsonx",
        model_identity="model-a",
        runtime_identity="runtime-1",
        created_at=created_at,
    )

    evidence = EvidenceRecord(
        evidence_id=uuid4(),
        experiment_id=experiment.experiment_id,
        candidate_id=candidate.candidate_id,
        kind="verification",
        payload={
            "decision": "EXACT",
        },
        created_at=created_at,
    )

    artifact = ArtifactRecord(
        artifact_id=uuid4(),
        experiment_id=experiment.experiment_id,
        evidence_id=evidence.evidence_id,
        storage_location="file:///evidence/result.json",
        sha256="c" * 64,
        size_bytes=128,
        media_type="application/json",
        created_at=created_at,
    )

    with transactional_session(registry_runtime) as session:
        ExperimentRunRepository(session).add(experiment)
        CandidateRepository(session).add(candidate)
        EvidenceRepository(session).add(evidence)
        ArtifactMetadataRepository(session).add(artifact)

    with transactional_session(registry_runtime) as session:
        experiments = ExperimentRunRepository(session)
        candidates = CandidateRepository(session)
        evidence_records = EvidenceRepository(session)
        artifacts = ArtifactMetadataRepository(session)

        assert experiments.get(experiment.experiment_id) == experiment
        assert candidates.get(candidate.candidate_id) == candidate
        assert evidence_records.get(evidence.evidence_id) == evidence
        assert artifacts.get(artifact.artifact_id) == artifact

        assert candidates.list_for_experiment(
            experiment.experiment_id
        ) == (candidate,)

        assert evidence_records.list_for_experiment(
            experiment.experiment_id
        ) == (evidence,)

        assert artifacts.list_for_experiment(
            experiment.experiment_id
        ) == (artifact,)


def test_registry_enforces_foreign_key_lineage(
    registry_runtime: DatabaseRuntime,
) -> None:
    """A candidate cannot reference an experiment that does not exist."""
    orphan = CandidateRecord(
        candidate_id=uuid4(),
        experiment_id=uuid4(),
        candidate_identity="orphan-candidate",
        candidate_version="1",
        created_at=datetime.now(UTC),
    )

    with (
        pytest.raises(IntegrityError),
        transactional_session(registry_runtime) as session,
    ):
        CandidateRepository(session).add(orphan)

    with transactional_session(registry_runtime) as session:
        assert CandidateRepository(session).get(
            orphan.candidate_id
        ) is None


def test_registry_transaction_rolls_back_partial_lineage(
    registry_runtime: DatabaseRuntime,
) -> None:
    """Failure inside a transaction must not persist partial lineage."""
    experiment = ExperimentRunRecord(
        experiment_id=uuid4(),
        source_revision="d" * 40,
        dataset_fingerprint="e" * 64,
        dataset_partition="rollback-test",
        transformation_contract_id="rollback-contract",
        transformation_contract_version="1",
        created_at=datetime.now(UTC),
    )

    with (
        pytest.raises(RuntimeError, match="forced rollback"),
        transactional_session(registry_runtime) as session,
    ):
        ExperimentRunRepository(session).add(experiment)
        raise RuntimeError("forced rollback")

    with transactional_session(registry_runtime) as session:
        assert ExperimentRunRepository(session).get(
            experiment.experiment_id
        ) is None
