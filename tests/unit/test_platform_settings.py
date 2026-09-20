"""Tests for M5-A platform configuration."""

import pytest
from pydantic import SecretStr, ValidationError

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


def test_authentication_is_disabled_by_default() -> None:
    """Local default configuration must not invent an authentication secret."""
    settings = PlatformSettings()

    assert settings.authentication_mode == "disabled"
    assert settings.auth_token is None
    assert settings.auth_principal == "vait-service"


def test_service_token_authentication_requires_token() -> None:
    """Service-token mode must fail closed without a configured token."""
    with pytest.raises(
        ValidationError,
        match="auth_token is required",
    ):
        PlatformSettings(
            authentication_mode="service_token",
        )


def test_service_auth_token_requires_minimum_length() -> None:
    """Inbound service tokens must have a non-trivial minimum length."""
    with pytest.raises(
        ValidationError,
        match="at least 32 characters",
    ):
        PlatformSettings(
            authentication_mode="service_token",
            auth_token=SecretStr("too-short"),
        )


def test_disabled_authentication_rejects_dormant_token() -> None:
    """A secret must not remain configured behind disabled authentication."""
    with pytest.raises(
        ValidationError,
        match="must not be configured",
    ):
        PlatformSettings(
            authentication_mode="disabled",
            auth_token=SecretStr("a" * 32),
        )


@pytest.mark.parametrize(
    "environment",
    [
        "staging",
        "production",
    ],
)
def test_protected_environment_requires_authentication(
    environment: str,
) -> None:
    """Protected environments must not run with authentication disabled."""
    with pytest.raises(
        ValidationError,
        match="authentication must use service_token",
    ):
        PlatformSettings(
            environment=environment,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "environment",
    [
        "staging",
        "production",
    ],
)
def test_protected_environment_accepts_service_token_authentication(
    environment: str,
) -> None:
    """Protected environments may start with valid service-token auth."""
    settings = PlatformSettings(
        environment=environment,  # type: ignore[arg-type]
        authentication_mode="service_token",
        auth_token=SecretStr("a" * 32),
    )

    assert settings.authentication_mode == "service_token"
    assert settings.auth_token is not None


def test_auth_token_remains_secret_in_settings_representation() -> None:
    """Configured authentication tokens must not leak through repr."""
    secret = "super-secret-platform-token-123456789"

    settings = PlatformSettings(
        authentication_mode="service_token",
        auth_token=SecretStr(secret),
    )

    assert secret not in repr(settings)


def test_authentication_settings_read_prefixed_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VAIT-prefixed environment variables must configure authentication."""
    monkeypatch.setenv(
        "VAIT_AUTHENTICATION_MODE",
        "service_token",
    )
    monkeypatch.setenv(
        "VAIT_AUTH_TOKEN",
        "environment-service-token-123456789",
    )
    monkeypatch.setenv(
        "VAIT_AUTH_PRINCIPAL",
        "ci-service",
    )

    settings = PlatformSettings()

    assert settings.authentication_mode == "service_token"
    assert settings.auth_principal == "ci-service"
    assert settings.auth_token is not None


def test_service_auth_token_rejects_surrounding_whitespace() -> None:
    """Service tokens must not accept ambiguous surrounding whitespace."""
    with pytest.raises(
        ValidationError,
        match="leading or trailing whitespace",
    ):
        PlatformSettings(
            authentication_mode="service_token",
            auth_token=SecretStr(
                " token-with-enough-characters-123456 "
            ),
        )


def test_auth_principal_rejects_unsafe_characters() -> None:
    """Configured principals must remain bounded and log-safe."""
    with pytest.raises(
        ValidationError,
    ):
        PlatformSettings(
            authentication_mode="service_token",
            auth_token=SecretStr("a" * 32),
            auth_principal="unsafe principal",
        )


def test_authorization_role_defaults_to_viewer() -> None:
    """Authentication must default to the least-privileged platform role."""
    settings = PlatformSettings()

    assert settings.auth_role == "viewer"


@pytest.mark.parametrize(
    "role",
    [
        "viewer",
        "operator",
        "admin",
    ],
)
def test_service_token_accepts_supported_authorization_roles(
    role: str,
) -> None:
    """Authenticated services may receive only declared platform roles."""
    settings = PlatformSettings(
        authentication_mode="service_token",
        auth_token=SecretStr("a" * 32),
        auth_role=role,  # type: ignore[arg-type]
    )

    assert settings.auth_role == role


def test_disabled_authentication_rejects_elevated_role() -> None:
    """Dormant elevated authority must not exist behind disabled auth."""
    with pytest.raises(
        ValidationError,
        match="elevated auth_role must not be configured",
    ):
        PlatformSettings(
            authentication_mode="disabled",
            auth_role="admin",
        )


def test_auth_role_reads_prefixed_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VAIT_AUTH_ROLE must pass through typed settings validation."""
    monkeypatch.setenv(
        "VAIT_AUTHENTICATION_MODE",
        "service_token",
    )
    monkeypatch.setenv(
        "VAIT_AUTH_TOKEN",
        "environment-service-token-123456789",
    )
    monkeypatch.setenv(
        "VAIT_AUTH_ROLE",
        "operator",
    )

    settings = PlatformSettings()

    assert settings.auth_role == "operator"


def test_unknown_authorization_role_is_rejected() -> None:
    """Configuration must reject undeclared platform roles."""
    with pytest.raises(
        ValidationError,
    ):
        PlatformSettings(
            authentication_mode="service_token",
            auth_token=SecretStr("a" * 32),
            auth_role="superuser",  # type: ignore[arg-type]
        )
