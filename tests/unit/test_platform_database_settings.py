"""Tests for M5-C database configuration."""

from vait.platform import PlatformSettings


def test_database_url_is_optional_by_default() -> None:
    """Local platform configuration must not invent database credentials."""
    settings = PlatformSettings()

    assert settings.database_url is None


def test_database_url_is_loaded_from_vait_environment(
    monkeypatch: object,
) -> None:
    """Database URL must use the existing VAIT environment namespace."""
    from pytest import MonkeyPatch

    assert isinstance(monkeypatch, MonkeyPatch)

    database_url = (
        "postgresql+psycopg://vait_user:"
        "secret-password@localhost:5432/vait"
    )

    monkeypatch.setenv(
        "VAIT_DATABASE_URL",
        database_url,
    )

    settings = PlatformSettings()

    assert settings.database_url is not None
    assert settings.database_url.get_secret_value() == database_url


def test_database_url_is_redacted_from_settings_representation(
    monkeypatch: object,
) -> None:
    """Database credentials must not leak through settings repr."""
    from pytest import MonkeyPatch

    assert isinstance(monkeypatch, MonkeyPatch)

    secret_password = "do-not-leak-this-password"

    monkeypatch.setenv(
        "VAIT_DATABASE_URL",
        (
            "postgresql+psycopg://vait_user:"
            f"{secret_password}@localhost:5432/vait"
        ),
    )

    settings = PlatformSettings()

    representation = repr(settings)

    assert secret_password not in representation
    assert "**********" in representation
