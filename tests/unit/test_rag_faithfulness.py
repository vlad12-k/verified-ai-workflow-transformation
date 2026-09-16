"""Tests for human-adjudicated RAG faithfulness evidence."""

import pytest
from pydantic import ValidationError

from vait.rag.faithfulness import (
    FaithfulnessAnnotation,
    FaithfulnessAnnotationSet,
    FaithfulnessVerdict,
    answer_sha256,
)


def test_answer_hash_is_deterministic() -> None:
    """Identical generated text should have an identical digest."""
    answer = "Manual review is required."

    assert answer_sha256(answer) == answer_sha256(answer)
    assert len(answer_sha256(answer)) == 64


def test_annotation_set_accepts_unique_cases() -> None:
    """Distinct reviewed cases should validate."""
    annotation_set = FaithfulnessAnnotationSet(
        benchmark_id="rag-test",
        benchmark_version="1",
        experiment_id="experiment-1",
        annotations=(
            FaithfulnessAnnotation(
                case_id="case-1",
                answer_sha256="a" * 64,
                verdict=FaithfulnessVerdict.SUPPORTED,
                rationale="All claims are supported.",
            ),
        ),
    )

    assert len(annotation_set.annotations) == 1


def test_annotation_set_rejects_duplicate_cases() -> None:
    """One generated case must not receive conflicting reviews."""
    annotation = FaithfulnessAnnotation(
        case_id="case-1",
        answer_sha256="a" * 64,
        verdict=FaithfulnessVerdict.UNSUPPORTED,
        rationale="Contains an unsupported claim.",
    )

    with pytest.raises(
        ValidationError,
        match="case IDs must be unique",
    ):
        FaithfulnessAnnotationSet(
            benchmark_id="rag-test",
            benchmark_version="1",
            experiment_id="experiment-1",
            annotations=(
                annotation,
                annotation,
            ),
        )


def test_annotation_rejects_invalid_digest() -> None:
    """Reviews must identify the exact generated answer."""
    with pytest.raises(
        ValidationError,
        match="SHA-256",
    ):
        FaithfulnessAnnotation(
            case_id="case-1",
            answer_sha256="not-a-digest",
            verdict=FaithfulnessVerdict.PARTIAL,
            rationale="Incomplete but supported.",
        )


def test_faithfulness_evaluation_validates_hashes() -> None:
    """Reviewed verdicts must refer to the exact generated text."""
    from vait.rag.faithfulness import (
        evaluate_faithfulness_annotations,
    )

    answer = "Supported answer."

    annotation_set = FaithfulnessAnnotationSet(
        benchmark_id="rag-test",
        benchmark_version="1",
        experiment_id="experiment-1",
        annotations=(
            FaithfulnessAnnotation(
                case_id="case-1",
                answer_sha256=answer_sha256(
                    answer
                ),
                verdict=FaithfulnessVerdict.SUPPORTED,
                rationale="Fully supported.",
            ),
        ),
    )

    report = evaluate_faithfulness_annotations(
        annotation_set,
        answers_by_case_id={
            "case-1": answer,
        },
    )

    assert report.reviewed_count == 1
    assert report.supported_count == 1
    assert report.partial_count == 0
    assert report.unsupported_count == 0
    assert report.supported_rate == 1.0
    assert report.non_unsupported_rate == 1.0
