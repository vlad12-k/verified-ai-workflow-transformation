"""Artifact storage boundary for evidence bytes."""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class ArtifactWriteResult(BaseModel):
    """Integrity metadata returned after persisting artifact bytes."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    location: str = Field(
        min_length=1,
    )
    sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )
    size_bytes: int = Field(
        ge=1,
    )


@runtime_checkable
class ArtifactStore(Protocol):
    """Provider-neutral boundary for durable artifact byte storage."""

    def put(
        self,
        *,
        key: str,
        data: bytes,
    ) -> ArtifactWriteResult:
        """Persist bytes and return their durable integrity metadata."""
        ...

    def get(
        self,
        *,
        location: str,
    ) -> bytes:
        """Load artifact bytes from an opaque storage location."""
        ...

    def delete(
        self,
        *,
        location: str,
    ) -> None:
        """Delete artifact bytes from storage."""
        ...
