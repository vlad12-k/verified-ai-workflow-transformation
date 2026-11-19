"""Tests for the trusted code-owned implementation registry."""

from typing import cast

import pytest
from pydantic import JsonValue

from vait.contracts.models import Effect
from vait.runners.base import ImplementationRunner
from vait.runners.python_runner import PythonImplementationRunner
from vait.transformations.library.synthetic_ap import synthetic_ap_policy
from vait.verification.trusted_implementations import (
    AmbiguousTrustedImplementationVersionError,
    ConflictingTrustedImplementationRegistrationError,
    InvalidTrustedImplementationRunnerError,
    TrustedImplementationRegistration,
    TrustedImplementationRegistry,
    TrustedImplementationSourceRevisionError,
    UnknownTrustedImplementationError,
    UnsupportedTrustedImplementationVersionError,
    create_trusted_implementation_registry,
)


def test_existing_reviewed_implementation_resolves_without_execution() -> None:
    """Resolve the existing synthetic AP policy without running its function."""
    registry = create_trusted_implementation_registry()

    runner = registry.resolve(
        implementation_id="synthetic-ap-policy-v0.2",
        version="1.0.0",
    )

    assert isinstance(runner, PythonImplementationRunner)
    assert runner.implementation_id == "synthetic-ap-policy-v0.2"
    assert runner.function is synthetic_ap_policy


def test_unknown_implementation_id_is_rejected() -> None:
    registry = create_trusted_implementation_registry()

    with pytest.raises(UnknownTrustedImplementationError):
        registry.resolve(implementation_id="unregistered-policy", version="1.0.0")


def test_unsupported_implementation_version_is_rejected() -> None:
    registry = create_trusted_implementation_registry()

    with pytest.raises(UnsupportedTrustedImplementationVersionError):
        registry.resolve(
            implementation_id="synthetic-ap-policy-v0.2",
            version="9.9.9",
        )


def test_conflicting_code_owned_registration_is_rejected() -> None:
    registration = TrustedImplementationRegistration(
        implementation_id="synthetic-ap-policy-v0.2",
        version="1.0.0",
        factory=lambda: PythonImplementationRunner(
            implementation_id="synthetic-ap-policy-v0.2",
            function=synthetic_ap_policy,
        ),
    )

    with pytest.raises(ConflictingTrustedImplementationRegistrationError):
        TrustedImplementationRegistry((registration, registration))


def test_id_only_resolution_rejects_multiple_approved_versions() -> None:
    def factory() -> ImplementationRunner:
        return PythonImplementationRunner(
            implementation_id="test-policy",
            function=synthetic_ap_policy,
        )

    registry = TrustedImplementationRegistry(
        (
            TrustedImplementationRegistration(
                implementation_id="test-policy",
                version="1.0.0",
                factory=factory,
            ),
            TrustedImplementationRegistration(
                implementation_id="test-policy",
                version="2.0.0",
                factory=factory,
            ),
        )
    )

    with pytest.raises(AmbiguousTrustedImplementationVersionError):
        registry.resolve_unique_id(implementation_id="test-policy")


def test_pinned_source_revision_mismatch_is_rejected_before_factory_call() -> None:
    calls = 0

    def factory() -> ImplementationRunner:
        nonlocal calls
        calls += 1
        return PythonImplementationRunner(
            implementation_id="synthetic-ap-policy-v0.2",
            function=synthetic_ap_policy,
        )

    registry = TrustedImplementationRegistry(
        (
            TrustedImplementationRegistration(
                implementation_id="synthetic-ap-policy-v0.2",
                version="1.0.0",
                factory=factory,
                source_revision="a" * 40,
            ),
        )
    )

    with pytest.raises(TrustedImplementationSourceRevisionError):
        registry.resolve(
            implementation_id="synthetic-ap-policy-v0.2",
            version="1.0.0",
            source_revision="b" * 40,
        )

    assert calls == 0


def test_registry_snapshots_registrations_and_exposes_no_mutable_mapping() -> None:
    registration = TrustedImplementationRegistration(
        implementation_id="synthetic-ap-policy-v0.2",
        version="1.0.0",
        factory=lambda: PythonImplementationRunner(
            implementation_id="synthetic-ap-policy-v0.2",
            function=synthetic_ap_policy,
        ),
    )
    registrations = [registration]
    registry = TrustedImplementationRegistry(registrations)

    registrations.clear()

    assert registry.resolve(
        implementation_id="synthetic-ap-policy-v0.2",
        version="1.0.0",
    ).implementation_id == "synthetic-ap-policy-v0.2"
    with pytest.raises(TypeError):
        cast(
            dict[tuple[str, str], TrustedImplementationRegistration],
            registry._registrations,
        )[("another-implementation", "1.0.0")] = registration
    with pytest.raises((AttributeError, TypeError)):
        registry.__setattr__("_registrations", {})


def test_resolving_runner_does_not_execute_its_implementation() -> None:
    calls = 0

    def implementation(_data: dict[str, JsonValue]) -> JsonValue:
        nonlocal calls
        calls += 1
        raise AssertionError("Runner implementation must not execute during resolution")

    registry = TrustedImplementationRegistry(
        (
            TrustedImplementationRegistration(
                implementation_id="test-policy",
                version="1.0.0",
                factory=lambda: PythonImplementationRunner(
                    implementation_id="test-policy",
                    function=implementation,
                    declared_effects=frozenset({Effect.NONE}),
                ),
            ),
        )
    )

    runner = registry.resolve(implementation_id="test-policy", version="1.0.0")

    assert runner.implementation_id == "test-policy"
    assert calls == 0


def test_factory_cannot_return_runner_for_a_different_identity() -> None:
    registry = TrustedImplementationRegistry(
        (
            TrustedImplementationRegistration(
                implementation_id="expected-policy",
                version="1.0.0",
                factory=lambda: PythonImplementationRunner(
                    implementation_id="other-policy",
                    function=synthetic_ap_policy,
                ),
            ),
        )
    )

    with pytest.raises(InvalidTrustedImplementationRunnerError):
        registry.resolve(
            implementation_id="expected-policy",
            version="1.0.0",
        )
