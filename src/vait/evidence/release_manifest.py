"""Versioned release-evidence manifest for VAIT milestone artifacts."""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator


class ReleaseEvidenceArtifact(BaseModel):
    """Immutable identity for one generated evidence artifact."""

    milestone: str = Field(
        min_length=1,
    )

    path: str = Field(
        min_length=1,
    )

    sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    size_bytes: int = Field(
        ge=1,
    )


class ReleaseEvidenceManifest(BaseModel):
    """Canonical evidence inventory for one clean repository revision."""

    schema_version: str = "0.1"

    manifest_id: str = Field(
        min_length=1,
    )

    evidence_scope: str = (
        "M0-M4-J-development-evidence"
    )

    performance_claim: Literal[False] = False

    source_revision: str = Field(
        pattern=r"^[0-9a-f]{40}$",
    )

    source_dirty: Literal[False] = False

    artifacts: tuple[
        ReleaseEvidenceArtifact,
        ...,
    ] = Field(
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_unique_artifacts(
        self,
    ) -> Self:
        """Require every evidence path to appear exactly once."""
        paths = tuple(
            artifact.path
            for artifact in self.artifacts
        )

        if len(paths) != len(
            set(paths)
        ):
            raise ValueError(
                "Release evidence artifact paths "
                "must be unique."
            )

        return self


def build_release_evidence_manifest(
    *,
    entries: Sequence[
        tuple[str, str]
    ],
    repo_root: Path,
    source_revision: str,
    source_dirty: bool,
) -> ReleaseEvidenceManifest:
    """Bind versioned milestone evidence to one clean source revision."""
    if source_dirty:
        raise ValueError(
            "Release evidence manifest requires "
            "a clean source tree."
        )

    if not entries:
        raise ValueError(
            "Release evidence manifest requires "
            "at least one artifact."
        )

    resolved_root = (
        repo_root.resolve()
    )

    artifacts: list[
        ReleaseEvidenceArtifact
    ] = []

    for milestone, path_text in entries:
        relative_path = Path(
            path_text
        )

        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            raise ValueError(
                "Evidence artifact paths must be "
                "repository-relative and cannot traverse "
                "outside the repository."
            )

        artifact_path = (
            resolved_root
            / relative_path
        ).resolve()

        try:
            artifact_path.relative_to(
                resolved_root
            )
        except ValueError as exc:
            raise ValueError(
                "Evidence artifact path resolves "
                "outside the repository."
            ) from exc

        if not artifact_path.is_file():
            raise FileNotFoundError(
                "Required release evidence artifact "
                f"is missing: {path_text}"
            )

        raw = artifact_path.read_bytes()

        try:
            payload: object = json.loads(
                raw.decode(
                    "utf-8"
                )
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise ValueError(
                "Evidence artifact must contain "
                f"valid UTF-8 JSON: {path_text}"
            ) from exc

        if not isinstance(
            payload,
            dict,
        ):
            raise ValueError(
                "Evidence artifact JSON must "
                f"be an object: {path_text}"
            )

        artifacts.append(
            ReleaseEvidenceArtifact(
                milestone=milestone,
                path=relative_path.as_posix(),
                sha256=hashlib.sha256(
                    raw
                ).hexdigest(),
                size_bytes=len(
                    raw
                ),
            )
        )

    return ReleaseEvidenceManifest(
        manifest_id=(
            "m4k-release-evidence-manifest-v0.1"
        ),
        source_revision=(
            source_revision
        ),
        source_dirty=False,
        artifacts=tuple(
            artifacts
        ),
    )
