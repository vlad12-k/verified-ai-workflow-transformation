"""Tests for typed VAIT transformation declarations."""

import pytest
from pydantic import ValidationError

from vait.contracts.models import Effect, RiskLevel
from vait.transformations.models import (
    TransformationCategory,
    TransformationDescriptor,
)


def test_descriptor_has_versioned_canonical_id() -> None:
    """Transformation identities should include their version."""
    descriptor = TransformationDescriptor(
        transformation_id="remove-redundant-step",
        version="1.0.0",
        name="Remove redundant step",
        description="Remove a declared redundant workflow step.",
        category=TransformationCategory.STRUCTURAL,
    )

    assert (
        descriptor.canonical_id
        == "remove-redundant-step@1.0.0"
    )


def test_descriptor_defaults_to_no_effects() -> None:
    """A transformation should default to a pure declaration."""
    descriptor = TransformationDescriptor(
        transformation_id="pure-transform",
        version="1.0.0",
        name="Pure transformation",
        description="A pure transformation fixture.",
        category=TransformationCategory.OPTIMIZATION,
    )

    assert descriptor.declared_effects == frozenset(
        {Effect.NONE}
    )
    assert descriptor.supported_risk_levels == frozenset(
        RiskLevel
    )


def test_none_effect_cannot_be_combined_with_other_effects() -> None:
    """Contradictory effect declarations should be rejected."""
    with pytest.raises(
        ValidationError,
        match="Effect.NONE cannot be combined",
    ):
        TransformationDescriptor(
            transformation_id="invalid-effects",
            version="1.0.0",
            name="Invalid effects",
            description="Invalid transformation fixture.",
            category=TransformationCategory.DATA,
            declared_effects=frozenset(
                {
                    Effect.NONE,
                    Effect.READ,
                }
            ),
        )


def test_invalid_transformation_id_is_rejected() -> None:
    """Transformation IDs should remain machine-safe."""
    with pytest.raises(ValidationError):
        TransformationDescriptor(
            transformation_id="Invalid Transformation",
            version="1.0.0",
            name="Invalid transformation",
            description="Invalid identifier fixture.",
            category=TransformationCategory.STRUCTURAL,
        )


def test_invalid_version_is_rejected() -> None:
    """Transformation versions should use semantic version triples."""
    with pytest.raises(ValidationError):
        TransformationDescriptor(
            transformation_id="bad-version",
            version="v1",
            name="Bad version",
            description="Invalid version fixture.",
            category=TransformationCategory.STRUCTURAL,
        )
