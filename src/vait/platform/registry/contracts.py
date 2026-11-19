"""Typed contracts for the M5 experiment and evidence registry."""

from typing import Literal
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
)

from vait.platform.registry.sensitive_data import (
    validate_evidence_payload,
)

EvidenceKind = Literal[
    "verification",
    "inference",
    "resource",
    "economic",
    "optimisation",
    "recommendation",
]


class RegistryRecord(BaseModel):
    """Immutable validated base for persisted registry records."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        hide_input_in_errors=True,
    )


class ExperimentRunRecord(RegistryRecord):
    """Durable identity and reproducibility lineage for one experiment."""

    experiment_id: UUID
    source_revision: str = Field(
        pattern=r"^[0-9a-f]{40}$",
    )
    dataset_fingerprint: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )
    dataset_partition: str = Field(
        min_length=1,
    )
    transformation_contract_id: str = Field(
        min_length=1,
    )
    transformation_contract_version: str = Field(
        min_length=1,
    )
    benchmark_configuration: dict[str, JsonValue] = Field(
        default_factory=dict,
    )
    reproduction_of_experiment_id: UUID | None = None
    created_at: AwareDatetime


class CandidateRecord(RegistryRecord):
    """Candidate or search-point identity evaluated by an experiment."""

    candidate_id: UUID
    experiment_id: UUID
    candidate_identity: str = Field(
        min_length=1,
    )
    candidate_version: str = Field(
        min_length=1,
    )
    provider_identity: str | None = Field(
        default=None,
        min_length=1,
    )
    model_identity: str | None = Field(
        default=None,
        min_length=1,
    )
    runtime_identity: str | None = Field(
        default=None,
        min_length=1,
    )
    created_at: AwareDatetime


class EvidenceRecord(RegistryRecord):
    """Typed registry envelope for one evidence payload."""

    evidence_id: UUID
    experiment_id: UUID
    candidate_id: UUID | None = None
    kind: EvidenceKind
    payload: dict[str, JsonValue]
    created_at: AwareDatetime

    @field_validator(
        "payload",
    )
    @classmethod
    def reject_prohibited_secret_material(
        cls,
        value: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        """Reject credentials before evidence reaches persistence."""
        return validate_evidence_payload(
            value
        )


class ArtifactRecord(RegistryRecord):
    """Durable metadata for artifact bytes stored outside PostgreSQL."""

    artifact_id: UUID
    experiment_id: UUID
    evidence_id: UUID | None = None
    storage_location: str = Field(
        min_length=1,
    )
    sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )
    size_bytes: int = Field(
        ge=1,
    )
    media_type: str | None = Field(
        default=None,
        min_length=1,
    )
    created_at: AwareDatetime
