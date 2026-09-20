"""Security boundaries for the VAIT platform."""

from vait.platform.security.authentication import (
    AuthenticatedPrincipal,
    require_authenticated_principal,
)
from vait.platform.security.authorization import (
    PlatformPermission,
    PlatformRole,
    permissions_for_role,
    require_permission,
)

__all__ = [
    "AuthenticatedPrincipal",
    "PlatformPermission",
    "PlatformRole",
    "permissions_for_role",
    "require_authenticated_principal",
    "require_permission",
]
