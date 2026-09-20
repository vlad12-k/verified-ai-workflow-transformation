"""Inbound authentication boundary for protected platform operations."""

from secrets import compare_digest
from typing import Annotated, Literal, cast

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from vait.platform.settings import PlatformSettings

_BEARER_SCHEME = HTTPBearer(
    auto_error=False,
    scheme_name="VAITServiceToken",
    description=(
        "Bearer service token for protected VAIT platform operations."
    ),
)


class AuthenticatedPrincipal(BaseModel):
    """Identity established by the platform authentication boundary."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    subject: str = Field(
        min_length=1,
        max_length=128,
    )
    authentication_method: Literal[
        "service_token"
    ] = "service_token"


def _authentication_required() -> HTTPException:
    """Return a non-sensitive authentication failure."""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required.",
        headers={
            "WWW-Authenticate": "Bearer",
        },
    )


def require_authenticated_principal(
    request: Request,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(_BEARER_SCHEME),
    ],
) -> AuthenticatedPrincipal:
    """Authenticate one protected request using the configured service token."""
    settings = cast(
        PlatformSettings,
        request.app.state.settings,
    )

    if (
        settings.authentication_mode != "service_token"
        or settings.auth_token is None
    ):
        raise _authentication_required()

    if credentials is None:
        raise _authentication_required()

    if credentials.scheme.lower() != "bearer":
        raise _authentication_required()

    supplied_token = credentials.credentials
    expected_token = settings.auth_token.get_secret_value()

    if not supplied_token:
        raise _authentication_required()

    if not compare_digest(
        supplied_token,
        expected_token,
    ):
        raise _authentication_required()

    return AuthenticatedPrincipal(
        subject=settings.auth_principal,
    )
