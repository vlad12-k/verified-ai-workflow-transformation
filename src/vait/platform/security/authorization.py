"""Default-deny authorisation policy for protected platform operations."""

from collections.abc import Callable
from enum import StrEnum
from typing import Annotated

from fastapi import Depends, HTTPException, status

from vait.platform.security.authentication import (
    AuthenticatedPrincipal,
    require_authenticated_principal,
)


class PlatformRole(StrEnum):
    """Trusted platform roles assigned after authentication."""

    VIEWER = "viewer"
    OPERATOR = "operator"
    ADMIN = "admin"


class PlatformPermission(StrEnum):
    """Explicit permissions for protected platform operations."""

    EXPERIMENT_READ = "experiment:read"
    EXPERIMENT_RUN = "experiment:run"

    JOB_READ = "job:read"
    JOB_CANCEL = "job:cancel"

    EVIDENCE_READ = "evidence:read"

    PLATFORM_ADMIN = "platform:admin"


_ROLE_PERMISSIONS: dict[
    PlatformRole,
    frozenset[PlatformPermission],
] = {
    PlatformRole.VIEWER: frozenset(
        {
            PlatformPermission.EXPERIMENT_READ,
            PlatformPermission.JOB_READ,
            PlatformPermission.EVIDENCE_READ,
        }
    ),
    PlatformRole.OPERATOR: frozenset(
        {
            PlatformPermission.EXPERIMENT_READ,
            PlatformPermission.EXPERIMENT_RUN,
            PlatformPermission.JOB_READ,
            PlatformPermission.JOB_CANCEL,
            PlatformPermission.EVIDENCE_READ,
        }
    ),
    PlatformRole.ADMIN: frozenset(
        PlatformPermission
    ),
}


def permissions_for_role(
    role: str,
) -> frozenset[PlatformPermission]:
    """Return permissions for a trusted role, defaulting to deny-all."""
    try:
        platform_role = PlatformRole(role)
    except ValueError:
        return frozenset()

    return _ROLE_PERMISSIONS[
        platform_role
    ]


def _permission_denied() -> HTTPException:
    """Return a non-sensitive authorisation failure."""
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Permission denied.",
    )


def require_permission(
    permission: PlatformPermission,
) -> Callable[..., AuthenticatedPrincipal]:
    """Build a dependency requiring one explicit platform permission."""

    def dependency(
        principal: Annotated[
            AuthenticatedPrincipal,
            Depends(require_authenticated_principal),
        ],
    ) -> AuthenticatedPrincipal:
        granted = permissions_for_role(
            principal.role
        )

        if permission not in granted:
            raise _permission_denied()

        return principal

    return dependency
