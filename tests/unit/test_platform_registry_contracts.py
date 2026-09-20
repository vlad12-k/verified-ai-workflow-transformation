"""Tests for M5-E registry and ArtifactStore contracts."""

import hashlib
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from vait.platform.registry import (
    ArtifactRecord,
    ArtifactStore,
    ArtifactWriteResult,
    CandidateRecord,
    EvidenceRecord,
    ExperimentRunRecord,
)


class InMemoryArtifactStore:
    """Minimal structural implementation used to verify the boundary."""

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}

    def put(
        self,
        *,
        key: str,
        data: bytes,
    ) -> ArtifactWriteResult:
        self._objects[key] = data

        return ArtifactWriteResult(
            location=f"memory://{key}",
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
        )

    def get(
        self,
        *,
        location: str,
    ) -> bytes:
        key = location.removeprefix("memory://")
        return self._objects[key]

    def delete(
        self,
        *,
        location: str,
    ) -> None:
        key = location.removeprefix("memory://")
        del self._objects[key]


def test_experiment_run_preserves_reproducibility_lineage() -> None:
    """Experiment records must retain exact source and dataset identity."""
    experiment_id = uuid4()

    record = ExperimentRunRecord(
        experiment_id=experiment_id,
        source_revision="a" * 40,
        dataset_fingerprint="b" * 64,
        dataset_partition="train-v1",
        transformation_contract_id="ap-invoice-rule",
        transformation_contract_version="1.0",
        benchmark_configuration={
            "batch_size": 8,
            "deterministic": True,
        },
        created_at=datetime.now(UTC),
    )

    assert record.experiment_id == experiment_id
    assert record.source_revision == "a" * 40
    assert record.dataset_fingerprint == "b" * 64
    assert record.reproduction_of_experiment_id is None


def test_experiment_run_rejects_invalid_source_revision() -> None:
    """Source lineage must use a full lowercase Git SHA-1 identity."""
    with pytest.raises(ValidationError):
        ExperimentRunRecord(
            experiment_id=uuid4(),
            source_revision="not-a-revision",
            dataset_fingerprint="b" * 64,
            dataset_partition="train-v1",
            transformation_contract_id="contract",
            transformation_contract_version="1",
            created_at=datetime.now(UTC),
        )


def test_candidate_can_preserve_model_runtime_identity() -> None:
    """Candidate lineage may include provider/model/runtime identity."""
    experiment_id = uuid4()

    candidate = CandidateRecord(
        candidate_id=uuid4(),
        experiment_id=experiment_id,
        candidate_identity="candidate-001",
        candidate_version="1",
        provider_identity="watsonx",
        model_identity="model-a",
        runtime_identity="runtime-1",
        created_at=datetime.now(UTC),
    )

    assert candidate.experiment_id == experiment_id
    assert candidate.provider_identity == "watsonx"


def test_evidence_links_experiment_and_candidate() -> None:
    """Evidence must preserve the lineage of the evaluated candidate."""
    experiment_id = uuid4()
    candidate_id = uuid4()

    evidence = EvidenceRecord(
        evidence_id=uuid4(),
        experiment_id=experiment_id,
        candidate_id=candidate_id,
        kind="verification",
        payload={
            "decision": "EXACT",
        },
        created_at=datetime.now(UTC),
    )

    assert evidence.experiment_id == experiment_id
    assert evidence.candidate_id == candidate_id
    assert evidence.kind == "verification"


def test_artifact_record_contains_metadata_not_artifact_bytes() -> None:
    """PostgreSQL registry records must not accept raw artifact content."""
    payload = {
        "artifact_id": uuid4(),
        "experiment_id": uuid4(),
        "storage_location": "file:///evidence/result.json",
        "sha256": "c" * 64,
        "size_bytes": 128,
        "media_type": "application/json",
        "created_at": datetime.now(UTC),
        "content": b"must-not-be-stored-here",
    }

    with pytest.raises(ValidationError):
        ArtifactRecord.model_validate(payload)


def test_artifact_store_is_provider_neutral_structural_boundary() -> None:
    """Artifact storage implementations must satisfy the same protocol."""
    store = InMemoryArtifactStore()
    data = b'{"decision":"EXACT"}'

    assert isinstance(store, ArtifactStore)

    result = store.put(
        key="experiment/evidence.json",
        data=data,
    )

    assert result.sha256 == hashlib.sha256(data).hexdigest()
    assert result.size_bytes == len(data)
    assert store.get(location=result.location) == data

    store.delete(location=result.location)

    with pytest.raises(KeyError):
        store.get(location=result.location)
