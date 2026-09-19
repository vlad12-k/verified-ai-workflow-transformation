"""Typed configuration for the VAIT platform layer."""

from typing import Literal, Self

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DeploymentEnvironment = Literal[
    "development",
    "test",
    "staging",
    "production",
]


class PlatformSettings(BaseSettings):
    """Platform configuration loaded from VAIT-prefixed environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="VAIT_",
        case_sensitive=False,
        extra="ignore",
    )

    service_name: str = "vait-platform"
    environment: DeploymentEnvironment = "development"
    api_prefix: str = "/api/v1"
    debug: bool = False

    @model_validator(mode="after")
    def validate_secure_defaults(self) -> Self:
        """Reject unsafe platform configuration combinations."""
        if (
            self.environment in {"staging", "production"}
            and self.debug
        ):
            raise ValueError(
                "debug must be disabled in staging and production"
            )

        if not self.api_prefix.startswith("/"):
            raise ValueError(
                "api_prefix must start with '/'"
            )

        return self
