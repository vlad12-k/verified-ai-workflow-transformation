"""Compatibility filtering for bounded inference search points."""

from collections.abc import Sequence
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, model_validator

from vait.optimisation.search_space import (
    InferenceSearchPoint,
)


class CompatibilityStatus(StrEnum):
    """Outcome of search-point compatibility evaluation."""

    COMPATIBLE = "compatible"
    NOT_COMPATIBLE = "not_compatible"


class CompatibilityIssueCode(StrEnum):
    """Machine-readable reasons a search point cannot execute."""

    PROVIDER_UNAVAILABLE = "provider_unavailable"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    DEVICE_UNAVAILABLE = "device_unavailable"
    DTYPE_UNSUPPORTED = "dtype_unsupported"
    BATCH_SIZE_EXCEEDED = "batch_size_exceeded"
    MODEL_UNAVAILABLE = "model_unavailable"
    MISSING_CAPABILITY = "missing_capability"


class InferenceCompatibilityContext(BaseModel):
    """Declared runtime resources available to optimisation."""

    available_providers: frozenset[str] = Field(
        min_length=1,
    )
    available_runtimes: frozenset[str] = Field(
        min_length=1,
    )
    available_devices: frozenset[str] = Field(
        min_length=1,
    )
    available_dtypes: frozenset[str] = Field(
        min_length=1,
    )

    max_batch_size: int = Field(
        ge=1,
    )

    available_model_ids: frozenset[str] | None = None

    available_capabilities: frozenset[str] = Field(
        default_factory=frozenset,
    )


class CompatibilityIssue(BaseModel):
    """One explicit compatibility rejection reason."""

    code: CompatibilityIssueCode
    subject: str = Field(min_length=1)
    message: str = Field(min_length=1)


class SearchPointCompatibility(BaseModel):
    """Auditable compatibility decision for one search point."""

    search_point: InferenceSearchPoint

    status: CompatibilityStatus

    issues: tuple[
        CompatibilityIssue,
        ...,
    ] = ()

    @model_validator(mode="after")
    def validate_status_consistency(
        self,
    ) -> Self:
        """Prevent contradictory compatibility evidence."""
        if (
            self.status
            is CompatibilityStatus.COMPATIBLE
            and self.issues
        ):
            raise ValueError(
                "A compatible search point cannot contain issues."
            )

        if (
            self.status
            is CompatibilityStatus.NOT_COMPATIBLE
            and not self.issues
        ):
            raise ValueError(
                "A not-compatible search point must contain "
                "at least one issue."
            )

        return self

    @property
    def is_compatible(self) -> bool:
        """Return whether this point may proceed."""
        return (
            self.status
            is CompatibilityStatus.COMPATIBLE
        )


class CompatibilityFilterResult(BaseModel):
    """Partition compatible search points from explicit rejections."""

    compatible_points: tuple[
        InferenceSearchPoint,
        ...,
    ]

    rejected: tuple[
        SearchPointCompatibility,
        ...,
    ]


def evaluate_search_point_compatibility(
    search_point: InferenceSearchPoint,
    context: InferenceCompatibilityContext,
) -> SearchPointCompatibility:
    """Evaluate one registered search point against runtime resources."""
    configuration = search_point.configuration

    issues: list[CompatibilityIssue] = []

    if (
        configuration.provider
        not in context.available_providers
    ):
        issues.append(
            CompatibilityIssue(
                code=(
                    CompatibilityIssueCode.PROVIDER_UNAVAILABLE
                ),
                subject=configuration.provider,
                message=(
                    f"Provider '{configuration.provider}' "
                    "is not available."
                ),
            )
        )

    if (
        configuration.runtime
        not in context.available_runtimes
    ):
        issues.append(
            CompatibilityIssue(
                code=(
                    CompatibilityIssueCode.RUNTIME_UNAVAILABLE
                ),
                subject=configuration.runtime,
                message=(
                    f"Runtime '{configuration.runtime}' "
                    "is not available."
                ),
            )
        )

    if (
        configuration.device
        not in context.available_devices
    ):
        issues.append(
            CompatibilityIssue(
                code=(
                    CompatibilityIssueCode.DEVICE_UNAVAILABLE
                ),
                subject=configuration.device,
                message=(
                    f"Device '{configuration.device}' "
                    "is not available."
                ),
            )
        )

    if (
        configuration.dtype
        not in context.available_dtypes
    ):
        issues.append(
            CompatibilityIssue(
                code=(
                    CompatibilityIssueCode.DTYPE_UNSUPPORTED
                ),
                subject=configuration.dtype,
                message=(
                    f"Dtype '{configuration.dtype}' "
                    "is not supported by the declared context."
                ),
            )
        )

    if (
        configuration.batch_size
        > context.max_batch_size
    ):
        issues.append(
            CompatibilityIssue(
                code=(
                    CompatibilityIssueCode.BATCH_SIZE_EXCEEDED
                ),
                subject=str(
                    configuration.batch_size
                ),
                message=(
                    f"Batch size {configuration.batch_size} "
                    "exceeds the declared maximum "
                    f"{context.max_batch_size}."
                ),
            )
        )

    if (
        configuration.model_id is not None
        and context.available_model_ids is not None
        and configuration.model_id
        not in context.available_model_ids
    ):
        issues.append(
            CompatibilityIssue(
                code=(
                    CompatibilityIssueCode.MODEL_UNAVAILABLE
                ),
                subject=configuration.model_id,
                message=(
                    f"Model '{configuration.model_id}' "
                    "is not available."
                ),
            )
        )

    missing_capabilities = sorted(
        search_point.required_capabilities
        - context.available_capabilities
    )

    for capability in missing_capabilities:
        issues.append(
            CompatibilityIssue(
                code=(
                    CompatibilityIssueCode.MISSING_CAPABILITY
                ),
                subject=capability,
                message=(
                    f"Required capability '{capability}' "
                    "is not available."
                ),
            )
        )

    if issues:
        return SearchPointCompatibility(
            search_point=search_point,
            status=(
                CompatibilityStatus.NOT_COMPATIBLE
            ),
            issues=tuple(issues),
        )

    return SearchPointCompatibility(
        search_point=search_point,
        status=CompatibilityStatus.COMPATIBLE,
    )


def filter_compatible_search_points(
    search_points: Sequence[
        InferenceSearchPoint
    ],
    context: InferenceCompatibilityContext,
) -> CompatibilityFilterResult:
    """Keep only compatible points while preserving rejection evidence."""
    compatible: list[
        InferenceSearchPoint
    ] = []

    rejected: list[
        SearchPointCompatibility
    ] = []

    for search_point in sorted(
        search_points,
        key=lambda point: point.search_point_id,
    ):
        result = (
            evaluate_search_point_compatibility(
                search_point,
                context,
            )
        )

        if result.is_compatible:
            compatible.append(
                search_point
            )
        else:
            rejected.append(
                result
            )

    return CompatibilityFilterResult(
        compatible_points=tuple(
            compatible
        ),
        rejected=tuple(
            rejected
        ),
    )
