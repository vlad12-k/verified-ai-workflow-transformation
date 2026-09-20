"""Typed configuration for the VAIT platform layer."""

from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DeploymentEnvironment = Literal[
    "development",
    "test",
    "staging",
    "production",
]

AuthenticationMode = Literal[
    "disabled",
    "service_token",
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
    database_url: SecretStr | None = None

    authentication_mode: AuthenticationMode = "disabled"
    auth_token: SecretStr | None = None
    auth_principal: str = Field(
        default="vait-service",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
    )

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

        if self.authentication_mode == "service_token":
            if self.auth_token is None:
                raise ValueError(
                    "auth_token is required when authentication_mode "
                    "is service_token"
                )

            token = self.auth_token.get_secret_value()

            if token != token.strip():
                raise ValueError(
                    "auth_token must not contain leading or trailing "
                    "whitespace"
                )

            if len(token) < 32:
                raise ValueError(
                    "auth_token must contain at least 32 characters"
                )
        elif self.auth_token is not None:
            raise ValueError(
                "auth_token must not be configured when "
                "authentication_mode is disabled"
            )

        if (
            self.environment in {"staging", "production"}
            and self.authentication_mode != "service_token"
        ):
            raise ValueError(
                "authentication must use service_token in staging "
                "and production"
            )

        return self
