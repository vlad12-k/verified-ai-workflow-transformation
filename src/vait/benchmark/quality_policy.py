"""Versioned benchmark quality policies and quality artifacts."""

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from vait.benchmark.models import BenchmarkDataset
from vait.benchmark.provenance import benchmark_fingerprint
from vait.benchmark.quality import (
    BenchmarkQualityPolicy,
    BenchmarkQualityReport,
    evaluate_benchmark_quality,
)


class BenchmarkQualityPolicyDocument(BaseModel):
    """Versioned quality requirements for one benchmark."""

    benchmark_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    policy: BenchmarkQualityPolicy


class BenchmarkQualityArtifact(BaseModel):
    """Reproducible machine-readable benchmark quality evidence."""

    benchmark_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)

    dataset_fingerprint_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    quality_policy_fingerprint_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    policy: BenchmarkQualityPolicy
    quality: BenchmarkQualityReport


def load_quality_policy(
    path: str | Path,
) -> BenchmarkQualityPolicyDocument:
    """Load and validate a versioned benchmark quality policy."""
    policy_path = Path(path)

    payload = yaml.safe_load(
        policy_path.read_text(encoding="utf-8")
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "Benchmark quality policy must contain a mapping."
        )

    return BenchmarkQualityPolicyDocument.model_validate(payload)


def quality_policy_fingerprint(
    document: BenchmarkQualityPolicyDocument,
) -> str:
    """Return a deterministic SHA-256 fingerprint for a quality policy."""
    payload = document.model_dump(mode="json")

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def evaluate_versioned_benchmark_quality(
    dataset: BenchmarkDataset,
    document: BenchmarkQualityPolicyDocument,
) -> BenchmarkQualityArtifact:
    """Evaluate a dataset against the policy declared for its version."""
    if document.benchmark_id != dataset.benchmark_id:
        raise ValueError(
            "Quality policy benchmark ID does not match the dataset."
        )

    if document.benchmark_version != dataset.version:
        raise ValueError(
            "Quality policy benchmark version does not match the dataset."
        )

    quality = evaluate_benchmark_quality(
        dataset=dataset,
        policy=document.policy,
    )

    return BenchmarkQualityArtifact(
        benchmark_id=dataset.benchmark_id,
        benchmark_version=dataset.version,
        dataset_fingerprint_sha256=benchmark_fingerprint(dataset),
        quality_policy_fingerprint_sha256=(
            quality_policy_fingerprint(document)
        ),
        policy=document.policy,
        quality=quality,
    )
