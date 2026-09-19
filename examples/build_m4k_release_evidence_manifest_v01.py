"""Build the canonical M0-M4-J release-evidence manifest."""

import subprocess
from pathlib import Path

from vait.evidence.release_manifest import (
    build_release_evidence_manifest,
)

ARTIFACT_PATH = Path(
    "artifacts/benchmarks/"
    "m4k-release-evidence-manifest-v0.1.json"
)


EVIDENCE_ENTRIES: tuple[
    tuple[str, str],
    ...,
] = (
    (
        "M0",
        "artifacts/benchmarks/"
        "ap-v0.1-quality.json",
    ),
    (
        "M0",
        "artifacts/benchmarks/"
        "ap-v0.1-verification.json",
    ),
    (
        "M1",
        "artifacts/benchmarks/"
        "ap-v0.2-transformation-assessment.json",
    ),
    (
        "M2",
        "artifacts/benchmarks/"
        "ap-v0.2-cost-performance-evidence-v0.1.json",
    ),
    (
        "M3",
        "artifacts/benchmarks/"
        "m3-heterogeneous-evidence-v0.1.json",
    ),
    (
        "M4-A/B/C",
        "artifacts/benchmarks/"
        "m4-structured-optimisation-evidence-v0.1.json",
    ),
    (
        "M4-A/B/C",
        "artifacts/benchmarks/"
        "m4-provider-qwen-inference-v0.1.json",
    ),
    (
        "M4-D",
        "artifacts/benchmarks/"
        "m4d-structured-comparison-v0.1.json",
    ),
    (
        "M4-E",
        "artifacts/benchmarks/"
        "m4e-final-precision-compression-compilation-v0.1.json",
    ),
    (
        "M4-F",
        "artifacts/benchmarks/"
        "m4f-slm-rag-comparison-v0.1.json",
    ),
    (
        "M4-G",
        "artifacts/benchmarks/"
        "m4g-execution-optimisation-summary-v0.1.json",
    ),
    (
        "M4-H",
        "artifacts/benchmarks/"
        "m4h-ap-pytorch-resource-profile-v0.1.json",
    ),
    (
        "M4-I",
        "artifacts/benchmarks/"
        "m4i-multi-objective-analysis-v0.1.json",
    ),
    (
        "M4-J",
        "artifacts/benchmarks/"
        "m4j-recommendation-plan-v0.1.json",
    ),
)


def git_output(
    *args: str,
) -> str:
    """Return stripped stdout for one local Git command."""
    completed = subprocess.run(
        (
            "git",
            *args,
        ),
        check=True,
        capture_output=True,
        text=True,
    )

    return (
        completed.stdout.strip()
    )


def main() -> None:
    """Build a checksum-bound manifest from a clean repository."""
    source_revision = git_output(
        "rev-parse",
        "HEAD",
    )

    source_dirty = bool(
        git_output(
            "status",
            "--porcelain",
        )
    )

    manifest = (
        build_release_evidence_manifest(
            entries=EVIDENCE_ENTRIES,
            repo_root=Path("."),
            source_revision=(
                source_revision
            ),
            source_dirty=(
                source_dirty
            ),
        )
    )

    ARTIFACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT_PATH.write_text(
        manifest.model_dump_json(
            indent=2
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "M4-K unified release evidence"
    )
    print()

    print(
        "Manifest:",
        manifest.manifest_id,
    )

    print(
        "Evidence scope:",
        manifest.evidence_scope,
    )

    print(
        "Artifacts:",
        len(
            manifest.artifacts
        ),
    )

    milestone_counts: dict[
        str,
        int,
    ] = {}

    for artifact in manifest.artifacts:
        milestone_counts[
            artifact.milestone
        ] = (
            milestone_counts.get(
                artifact.milestone,
                0,
            )
            + 1
        )

    for milestone, count in (
        milestone_counts.items()
    ):
        print(
            f"{milestone}: {count}"
        )

    print()
    print(
        "Performance claim:",
        manifest.performance_claim,
    )

    print(
        "Source revision:",
        manifest.source_revision,
    )

    print(
        "Source dirty:",
        manifest.source_dirty,
    )

    print(
        "Evidence artifact:",
        ARTIFACT_PATH,
    )


if __name__ == "__main__":
    main()
