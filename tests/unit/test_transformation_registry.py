"""Tests for the VAIT transformation registry."""

from dataclasses import dataclass

import pytest

from vait.transformations.models import (
    TransformationCategory,
    TransformationDescriptor,
)
from vait.transformations.registry import TransformationRegistry


@dataclass(frozen=True)
class ExampleTransformation:
    """Minimal registered transformation fixture."""

    descriptor: TransformationDescriptor


def build_transformation(
    transformation_id: str,
    version: str = "1.0.0",
) -> ExampleTransformation:
    """Create a registry test transformation."""
    return ExampleTransformation(
        descriptor=TransformationDescriptor(
            transformation_id=transformation_id,
            version=version,
            name=transformation_id,
            description="Registry test transformation.",
            category=TransformationCategory.STRUCTURAL,
        )
    )


def test_registry_registers_and_retrieves_transformation() -> None:
    """Registered transformations should be addressable by ID and version."""
    registry = TransformationRegistry()
    transformation = build_transformation("remove-step")

    registry.register(transformation)

    assert registry.get(
        "remove-step",
        "1.0.0",
    ) is transformation
    assert len(registry) == 1


def test_registry_supports_multiple_versions() -> None:
    """Different versions of one transformation should coexist."""
    registry = TransformationRegistry()

    version_one = build_transformation(
        "remove-step",
        "1.0.0",
    )
    version_two = build_transformation(
        "remove-step",
        "2.0.0",
    )

    registry.register(version_one)
    registry.register(version_two)

    assert registry.get("remove-step", "1.0.0") is version_one
    assert registry.get("remove-step", "2.0.0") is version_two
    assert len(registry) == 2


def test_duplicate_canonical_id_is_rejected() -> None:
    """One canonical transformation identity must remain unique."""
    registry = TransformationRegistry()

    registry.register(build_transformation("duplicate"))

    with pytest.raises(
        ValueError,
        match="already registered",
    ):
        registry.register(
            build_transformation("duplicate")
        )


def test_unknown_transformation_is_rejected() -> None:
    """Missing transformations should fail explicitly."""
    registry = TransformationRegistry()

    with pytest.raises(
        KeyError,
        match="not registered",
    ):
        registry.get(
            "missing",
            "1.0.0",
        )


def test_registry_listing_is_deterministic() -> None:
    """Registry discovery should not depend on insertion order."""
    registry = TransformationRegistry()

    registry.register(build_transformation("z-transform"))
    registry.register(build_transformation("a-transform"))
    registry.register(build_transformation("m-transform"))

    canonical_ids = [
        descriptor.canonical_id
        for descriptor in registry.list_descriptors()
    ]

    assert canonical_ids == [
        "a-transform@1.0.0",
        "m-transform@1.0.0",
        "z-transform@1.0.0",
    ]
