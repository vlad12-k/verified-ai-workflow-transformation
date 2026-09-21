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
from vait.platform.security.redaction import (
    REDACTION_MARKER,
    redact_sensitive_text,
)

__all__ = [
    "AuthenticatedPrincipal",
    "PlatformPermission",
    "PlatformRole",
    "REDACTION_MARKER",
    "permissions_for_role",
    "redact_sensitive_text",
    "require_authenticated_principal",
    "require_permission",
]
