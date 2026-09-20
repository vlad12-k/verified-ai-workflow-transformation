"""Security boundaries for the VAIT platform."""

from vait.platform.security.authentication import (
    AuthenticatedPrincipal,
    require_authenticated_principal,
)

__all__ = [
    "AuthenticatedPrincipal",
    "require_authenticated_principal",
]
