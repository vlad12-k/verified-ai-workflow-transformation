"""Transformation support for supplied deterministic Python rule candidates."""

from dataclasses import dataclass

from vait.runners.python_runner import (
    ImplementationFunction,
    PythonImplementationRunner,
)
from vait.transformations.applicability import (
    ApplicabilityContext,
    ApplicabilityResult,
    evaluate_applicability,
)
from vait.transformations.models import TransformationDescriptor


@dataclass(frozen=True, slots=True)
class CandidatePreparation:
    """Result of preparing a supplied candidate for verification."""

    descriptor: TransformationDescriptor
    applicability: ApplicabilityResult
    runner: PythonImplementationRunner | None

    @property
    def ready(self) -> bool:
        """Return whether the candidate is ready for verification."""
        return self.runner is not None


@dataclass(frozen=True, slots=True)
class PythonRuleTransformation:
    """A supplied deterministic Python rule candidate transformation."""

    descriptor: TransformationDescriptor
    implementation_id: str
    function: ImplementationFunction

    def __post_init__(self) -> None:
        """Validate local transformation configuration."""
        if not self.implementation_id.strip():
            raise ValueError("implementation_id must not be empty.")

    def assess_applicability(
        self,
        context: ApplicabilityContext,
    ) -> ApplicabilityResult:
        """Evaluate whether this transformation may be attempted."""
        return evaluate_applicability(
            self.descriptor,
            context,
        )

    def prepare_candidate(
        self,
        context: ApplicabilityContext,
    ) -> CandidatePreparation:
        """Prepare the supplied candidate when applicability requirements pass."""
        applicability = self.assess_applicability(context)

        if not applicability.is_applicable:
            return CandidatePreparation(
                descriptor=self.descriptor,
                applicability=applicability,
                runner=None,
            )

        runner = PythonImplementationRunner(
            implementation_id=self.implementation_id,
            function=self.function,
            declared_effects=self.descriptor.declared_effects,
        )

        return CandidatePreparation(
            descriptor=self.descriptor,
            applicability=applicability,
            runner=runner,
        )
