"""Tests for M4-K unified release-evidence manifests."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from vait.evidence.release_manifest import (
    ReleaseEvidenceArtifact,
    ReleaseEvidenceManifest,
    build_release_evidence_manifest,
)

REVISION = "a" * 40


def _write_json(
    root: Path,
    name: str,
) -> str:
    """Write one synthetic evidence artifact."""
    path = (
        root
        / name
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            {
                "schema_version": "0.1",
                "evidence": "synthetic",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    return name


def test_manifest_binds_evidence_to_clean_revision(
    tmp_path: Path,
) -> None:
    """A clean revision can bind valid JSON evidence."""
    first = _write_json(
        tmp_path,
        "artifacts/first.json",
    )

    second = _write_json(
        tmp_path,
        "artifacts/second.json",
    )

    manifest = (
        build_release_evidence_manifest(
            entries=(
                ("M0", first),
                ("M4-J", second),
            ),
            repo_root=tmp_path,
            source_revision=REVISION,
            source_dirty=False,
        )
    )

    assert (
        manifest.source_revision
        == REVISION
    )

    assert (
        manifest.source_dirty
        is False
    )

    assert (
        manifest.performance_claim
        is False
    )

    assert (
        tuple(
            artifact.milestone
            for artifact
            in manifest.artifacts
        )
        == (
            "M0",
            "M4-J",
        )
    )

    for artifact in manifest.artifacts:
        assert len(
            artifact.sha256
        ) == 64

        assert (
            artifact.size_bytes
            > 0
        )


def test_dirty_tree_cannot_build_release_manifest(
    tmp_path: Path,
) -> None:
    """Release evidence must not claim a dirty source revision."""
    path = _write_json(
        tmp_path,
        "artifact.json",
    )

    with pytest.raises(
        ValueError,
        match="clean source tree",
    ):
        build_release_evidence_manifest(
            entries=(
                ("M4-J", path),
            ),
            repo_root=tmp_path,
            source_revision=REVISION,
            source_dirty=True,
        )


def test_missing_artifact_is_rejected(
    tmp_path: Path,
) -> None:
    """Every declared milestone artifact must exist."""
    with pytest.raises(
        FileNotFoundError,
        match="is missing",
    ):
        build_release_evidence_manifest(
            entries=(
                (
                    "M4-J",
                    "missing.json",
                ),
            ),
            repo_root=tmp_path,
            source_revision=REVISION,
            source_dirty=False,
        )


def test_invalid_json_artifact_is_rejected(
    tmp_path: Path,
) -> None:
    """Release evidence must remain machine-readable JSON."""
    path = (
        tmp_path
        / "broken.json"
    )

    path.write_text(
        "{broken",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="valid UTF-8 JSON",
    ):
        build_release_evidence_manifest(
            entries=(
                (
                    "M3",
                    "broken.json",
                ),
            ),
            repo_root=tmp_path,
            source_revision=REVISION,
            source_dirty=False,
        )


def test_repository_traversal_is_rejected(
    tmp_path: Path,
) -> None:
    """Manifest paths cannot escape the repository root."""
    with pytest.raises(
        ValueError,
        match="cannot traverse",
    ):
        build_release_evidence_manifest(
            entries=(
                (
                    "M4-I",
                    "../outside.json",
                ),
            ),
            repo_root=tmp_path,
            source_revision=REVISION,
            source_dirty=False,
        )


def test_duplicate_manifest_paths_are_rejected() -> None:
    """One evidence artifact cannot appear twice in a manifest."""
    artifact = (
        ReleaseEvidenceArtifact(
            milestone="M4-J",
            path="artifact.json",
            sha256="b" * 64,
            size_bytes=100,
        )
    )

    with pytest.raises(
        ValidationError,
        match="must be unique",
    ):
        ReleaseEvidenceManifest(
            manifest_id="duplicate-test",
            source_revision=REVISION,
            source_dirty=False,
            artifacts=(
                artifact,
                artifact,
            ),
        )
