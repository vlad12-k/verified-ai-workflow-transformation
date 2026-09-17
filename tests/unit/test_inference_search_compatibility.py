"""Tests for bounded inference search-point compatibility."""

from vait.inference.models import (
    InferenceConfiguration,
)
from vait.optimisation.compatibility import (
    CompatibilityIssueCode,
    CompatibilityStatus,
    InferenceCompatibilityContext,
    evaluate_search_point_compatibility,
    filter_compatible_search_points,
)
from vait.optimisation.search_space import (
    CandidateConfiguration,
    InferenceCandidate,
    InferenceSearchPoint,
    InferenceSearchSpace,
    enumerate_search_points,
)


def build_point(
    *,
    search_point_id: str = "candidate::default",
    provider: str = "local",
    runtime: str = "python",
    device: str = "cpu",
    dtype: str = "native",
    batch_size: int = 1,
    model_id: str | None = None,
    required_capabilities: frozenset[str] = frozenset(),
) -> InferenceSearchPoint:
    """Build one explicit search point."""
    return InferenceSearchPoint(
        search_point_id=search_point_id,
        search_space_id="test-space",
        task_family="structured-decision",
        candidate_id="candidate",
        implementation_id="candidate-v1",
        configuration_id="default",
        configuration=InferenceConfiguration(
            provider=provider,
            runtime=runtime,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            model_id=model_id,
            model_revision=None,
        ),
        requires_verification=True,
        required_capabilities=required_capabilities,
    )


def build_context(
    *,
    max_batch_size: int = 8,
    available_model_ids: (
        frozenset[str] | None
    ) = None,
    available_capabilities: (
        frozenset[str]
    ) = frozenset(),
) -> InferenceCompatibilityContext:
    """Build one CPU-only compatibility context."""
    return InferenceCompatibilityContext(
        available_providers=frozenset(
            {"local"}
        ),
        available_runtimes=frozenset(
            {"python"}
        ),
        available_devices=frozenset(
            {"cpu"}
        ),
        available_dtypes=frozenset(
            {"native"}
        ),
        max_batch_size=max_batch_size,
        available_model_ids=available_model_ids,
        available_capabilities=(
            available_capabilities
        ),
    )


def test_compatible_search_point_passes() -> None:
    """A fully supported point may proceed."""
    result = (
        evaluate_search_point_compatibility(
            build_point(),
            build_context(),
        )
    )

    assert (
        result.status
        is CompatibilityStatus.COMPATIBLE
    )
    assert result.is_compatible is True
    assert result.issues == ()


def test_reports_unavailable_runtime_resources() -> None:
    """Unavailable runtime dimensions must be explicit."""
    point = build_point(
        provider="remote",
        runtime="unknown-runtime",
        device="cuda",
        dtype="float16",
    )

    result = (
        evaluate_search_point_compatibility(
            point,
            build_context(),
        )
    )

    assert result.is_compatible is False

    assert [
        issue.code
        for issue in result.issues
    ] == [
        CompatibilityIssueCode.PROVIDER_UNAVAILABLE,
        CompatibilityIssueCode.RUNTIME_UNAVAILABLE,
        CompatibilityIssueCode.DEVICE_UNAVAILABLE,
        CompatibilityIssueCode.DTYPE_UNSUPPORTED,
    ]


def test_rejects_batch_size_above_limit() -> None:
    """Declared resource limits must gate benchmarking."""
    result = (
        evaluate_search_point_compatibility(
            build_point(
                batch_size=16,
            ),
            build_context(
                max_batch_size=8,
            ),
        )
    )

    assert result.is_compatible is False
    assert len(result.issues) == 1
    assert (
        result.issues[0].code
        is CompatibilityIssueCode.BATCH_SIZE_EXCEEDED
    )


def test_enforces_declared_model_inventory() -> None:
    """A model inventory, when declared, is a hard compatibility bound."""
    result = (
        evaluate_search_point_compatibility(
            build_point(
                model_id="model-b",
            ),
            build_context(
                available_model_ids=frozenset(
                    {"model-a"}
                ),
            ),
        )
    )

    assert result.is_compatible is False
    assert (
        result.issues[0].code
        is CompatibilityIssueCode.MODEL_UNAVAILABLE
    )


def test_undeclared_model_inventory_does_not_invent_rejection() -> None:
    """Unknown model inventory must remain unknown rather than false."""
    result = (
        evaluate_search_point_compatibility(
            build_point(
                model_id="model-a",
            ),
            build_context(
                available_model_ids=None,
            ),
        )
    )

    assert result.is_compatible is True


def test_rejects_missing_declared_capabilities() -> None:
    """Candidate requirements must be satisfied before execution."""
    result = (
        evaluate_search_point_compatibility(
            build_point(
                required_capabilities=frozenset(
                    {
                        "verified-runtime",
                        "batch-inference",
                    }
                ),
            ),
            build_context(
                available_capabilities=frozenset(
                    {"verified-runtime"}
                ),
            ),
        )
    )

    assert result.is_compatible is False
    assert len(result.issues) == 1
    assert (
        result.issues[0].code
        is CompatibilityIssueCode.MISSING_CAPABILITY
    )
    assert (
        result.issues[0].subject
        == "batch-inference"
    )


def test_filter_keeps_only_compatible_points() -> None:
    """Rejected points must not reach downstream verification/benchmarking."""
    compatible = build_point(
        search_point_id="candidate::cpu",
    )

    incompatible = build_point(
        search_point_id="candidate::cuda",
        device="cuda",
    )

    result = filter_compatible_search_points(
        (
            incompatible,
            compatible,
        ),
        build_context(),
    )

    assert [
        point.search_point_id
        for point in result.compatible_points
    ] == [
        "candidate::cpu",
    ]

    assert len(result.rejected) == 1

    rejection = result.rejected[0]

    assert (
        rejection.search_point.search_point_id
        == "candidate::cuda"
    )
    assert (
        rejection.status
        is CompatibilityStatus.NOT_COMPATIBLE
    )
    assert (
        rejection.issues[0].code
        is CompatibilityIssueCode.DEVICE_UNAVAILABLE
    )


def test_search_space_propagates_declared_capabilities() -> None:
    """Candidate and configuration requirements must both reach the point."""
    candidate = InferenceCandidate(
        candidate_id="candidate",
        implementation_id="candidate-v1",
        required_capabilities=frozenset(
            {"verified-runtime"}
        ),
        allowed_configurations=(
            CandidateConfiguration(
                configuration_id="cpu-b8",
                configuration=InferenceConfiguration(
                    provider="local",
                    runtime="python",
                    device="cpu",
                    dtype="native",
                    batch_size=8,
                    model_id=None,
                    model_revision=None,
                ),
                required_capabilities=frozenset(
                    {"batch-inference"}
                ),
            ),
        ),
    )

    search_space = InferenceSearchSpace(
        search_space_id="capability-space",
        version="0.1.0",
        task_family="structured-decision",
        candidates=(candidate,),
    )

    point = enumerate_search_points(
        search_space
    )[0]

    assert point.required_capabilities == frozenset(
        {
            "verified-runtime",
            "batch-inference",
        }
    )
