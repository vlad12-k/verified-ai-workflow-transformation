"""Tests for M5-A platform configuration."""

import pytest
from pydantic import ValidationError

from vait.platform import PlatformSettings


def test_platform_settings_have_safe_defaults() -> None:
    """Default platform configuration must be development-safe."""
    settings = PlatformSettings()

    assert settings.service_name == "vait-platform"
    assert settings.environment == "development"
    assert settings.api_prefix == "/api/v1"
    assert settings.debug is False


def test_platform_settings_read_prefixed_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VAIT-prefixed environment variables must configure the platform."""
    monkeypatch.setenv(
        "VAIT_SERVICE_NAME",
        "vait-test-platform",
    )
    monkeypatch.setenv(
        "VAIT_ENVIRONMENT",
        "test",
    )

    settings = PlatformSettings()

    assert settings.service_name == "vait-test-platform"
    assert settings.environment == "test"


@pytest.mark.parametrize(
    "environment",
    [
        "staging",
        "production",
    ],
)
def test_debug_is_rejected_in_protected_environments(
    environment: str,
) -> None:
    """Protected environments must fail closed when debug is enabled."""
    with pytest.raises(
        ValidationError,
        match="debug must be disabled",
    ):
        PlatformSettings(
            environment=environment,  # type: ignore[arg-type]
            debug=True,
        )


def test_api_prefix_requires_absolute_path() -> None:
    """API prefix must be an absolute URL path."""
    with pytest.raises(
        ValidationError,
        match="api_prefix must start",
    ):
        PlatformSettings(
            api_prefix="api/v1",
        )
