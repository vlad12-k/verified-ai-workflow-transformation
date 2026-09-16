"""Verification-aware assessment of supplied workflow transformations."""

from collections.abc import Iterable
from typing import Self

from pydantic import BaseModel, model_validator

from vait.benchmark.models import BenchmarkDataset
from vait.benchmark.verification import (
    BenchmarkVerificationReport,
    verify_benchmark_bounded,
)
from vait.contracts.models import (
    BoundedVerificationPolicy,
    Effect,
)
from vait.decision.models import Decision
from vait.transformations.applicability import (
    ApplicabilityContext,
    ApplicabilityResult,
)
from vait.transformations.python_callable import PythonCallableTransformation


class TransformationAssessment(BaseModel):
    """Applicability and verification evidence for one transformation."""

    canonical_id: str
    candidate_implementation_id: str
    applicability: ApplicabilityResult
    verification: BenchmarkVerificationReport | None = None

    @model_validator(mode="after")
    def validate_evidence_consistency(self) -> Self:
        """Ensure applicability and verification evidence are consistent."""
        if self.applicability.is_applicable and self.verification is None:
            raise ValueError(
                "An applicable candidate must include verification evidence."
            )

        if (
            not self.applicability.is_applicable
            and self.verification is not None
        ):
            raise ValueError(
                "A not-applicable candidate cannot include "
                "verification evidence."
            )

        if (
            self.verification is not None
            and self.verification.candidate_implementation_id
            != self.candidate_implementation_id
        ):
            raise ValueError(
                "Verification evidence must match the candidate "
                "implementation ID."
            )

        return self

    @property
    def decision(self) -> Decision | None:
        """Return the verification decision when verification was performed."""
        if self.verification is None:
            return None

        return self.verification.decision


class TransformationAssessmentReport(BaseModel):
    """Assessment results for supplied transformations on one benchmark."""

    benchmark_id: str
    benchmark_version: str
    assessments: tuple[TransformationAssessment, ...] = ()


def assess_python_transformations(
    *,
    dataset: BenchmarkDataset,
    transformations: Iterable[PythonCallableTransformation],
    context: ApplicabilityContext,
    policy: BoundedVerificationPolicy,
    allowed_effects: Iterable[Effect] = (Effect.NONE,),
) -> TransformationAssessmentReport:
    """Assess supplied Python-backed transformations independently."""
    assessments: list[TransformationAssessment] = []

    ordered_transformations = sorted(
        transformations,
        key=lambda transformation: transformation.descriptor.canonical_id,
    )

    for transformation in ordered_transformations:
        preparation = transformation.prepare_candidate(context)

        if preparation.runner is None:
            assessments.append(
                TransformationAssessment(
                    canonical_id=transformation.descriptor.canonical_id,
                    candidate_implementation_id=transformation.implementation_id,
                    applicability=preparation.applicability,
                )
            )
            continue

        verification = verify_benchmark_bounded(
            dataset=dataset,
            candidate=preparation.runner,
            policy=policy,
            allowed_effects=allowed_effects,
            candidate_configuration={
                "transformation_id": (
                    transformation.descriptor.transformation_id
                ),
                "transformation_version": (
                    transformation.descriptor.version
                ),
            },
        )

        assessments.append(
            TransformationAssessment(
                canonical_id=transformation.descriptor.canonical_id,
                candidate_implementation_id=transformation.implementation_id,
                applicability=preparation.applicability,
                verification=verification,
            )
        )

    return TransformationAssessmentReport(
        benchmark_id=dataset.benchmark_id,
        benchmark_version=dataset.version,
        assessments=tuple(assessments),
    )

def assess_python_rule_transformations(
    *,
    dataset: BenchmarkDataset,
    transformations: Iterable[PythonCallableTransformation],
    context: ApplicabilityContext,
    policy: BoundedVerificationPolicy,
    allowed_effects: Iterable[Effect] = (Effect.NONE,),
) -> TransformationAssessmentReport:
    """Assess Python transformations through the legacy public entry point."""
    return assess_python_transformations(
        dataset=dataset,
        transformations=transformations,
        context=context,
        policy=policy,
        allowed_effects=allowed_effects,
    )
