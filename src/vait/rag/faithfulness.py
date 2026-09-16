"""Human-adjudicated faithfulness evidence for grounded RAG."""

import hashlib
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, model_validator


class FaithfulnessVerdict(StrEnum):
    """Human adjudication of generated-answer support."""

    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    UNSUPPORTED = "UNSUPPORTED"


class FaithfulnessAnnotation(BaseModel):
    """One human-reviewed generated answer."""

    case_id: str
    answer_sha256: str
    verdict: FaithfulnessVerdict
    rationale: str

    @model_validator(mode="after")
    def validate_annotation(
        self,
    ) -> "FaithfulnessAnnotation":
        """Validate deterministic review metadata."""
        if len(self.answer_sha256) != 64:
            raise ValueError(
                "answer_sha256 must contain a SHA-256 digest."
            )

        if not self.rationale.strip():
            raise ValueError(
                "Faithfulness rationale must not be empty."
            )

        return self


class FaithfulnessAnnotationSet(BaseModel):
    """Versioned human faithfulness review."""

    benchmark_id: str
    benchmark_version: str
    experiment_id: str
    annotations: tuple[FaithfulnessAnnotation, ...]

    @model_validator(mode="after")
    def validate_annotations(
        self,
    ) -> "FaithfulnessAnnotationSet":
        """Require unique reviewed case identifiers."""
        case_ids = [
            annotation.case_id
            for annotation in self.annotations
        ]

        if len(set(case_ids)) != len(case_ids):
            raise ValueError(
                "Faithfulness case IDs must be unique."
            )

        return self


def answer_sha256(
    answer: str,
) -> str:
    """Return the SHA-256 digest of exact generated text."""
    return hashlib.sha256(
        answer.encode("utf-8")
    ).hexdigest()


def load_faithfulness_annotations(
    path: str | Path,
) -> FaithfulnessAnnotationSet:
    """Load a versioned faithfulness annotation set."""
    return FaithfulnessAnnotationSet.model_validate_json(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


class FaithfulnessEvaluationReport(BaseModel):
    """Aggregate manually approved faithfulness metrics."""

    experiment_id: str
    reviewed_count: int
    supported_count: int
    partial_count: int
    unsupported_count: int
    supported_rate: float
    non_unsupported_rate: float


def evaluate_faithfulness_annotations(
    annotations: FaithfulnessAnnotationSet,
    *,
    answers_by_case_id: dict[str, str],
) -> FaithfulnessEvaluationReport:
    """Validate exact answer hashes and summarize reviewed verdicts."""
    if not annotations.annotations:
        raise ValueError(
            "Faithfulness evaluation requires annotations."
        )

    for annotation in annotations.annotations:
        if annotation.case_id not in answers_by_case_id:
            raise ValueError(
                f"Missing generated answer for case "
                f"{annotation.case_id}."
            )

        observed_hash = answer_sha256(
            answers_by_case_id[
                annotation.case_id
            ]
        )

        if observed_hash != annotation.answer_sha256:
            raise ValueError(
                f"Generated answer hash changed for case "
                f"{annotation.case_id}."
            )

    supported_count = sum(
        annotation.verdict
        == FaithfulnessVerdict.SUPPORTED
        for annotation in annotations.annotations
    )
    partial_count = sum(
        annotation.verdict
        == FaithfulnessVerdict.PARTIAL
        for annotation in annotations.annotations
    )
    unsupported_count = sum(
        annotation.verdict
        == FaithfulnessVerdict.UNSUPPORTED
        for annotation in annotations.annotations
    )

    reviewed_count = len(
        annotations.annotations
    )

    return FaithfulnessEvaluationReport(
        experiment_id=annotations.experiment_id,
        reviewed_count=reviewed_count,
        supported_count=supported_count,
        partial_count=partial_count,
        unsupported_count=unsupported_count,
        supported_rate=(
            supported_count / reviewed_count
        ),
        non_unsupported_rate=(
            (supported_count + partial_count)
            / reviewed_count
        ),
    )
