"""Tests for the M6-B platform authorisation boundary."""

from typing import Annotated

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from vait.platform import PlatformSettings
from vait.platform.api import create_app
from vait.platform.security import (
    AuthenticatedPrincipal,
    PlatformPermission,
    PlatformRole,
    permissions_for_role,
    require_permission,
)

_VALID_TOKEN = "test-platform-service-token-123456789"


def _authorization_app(
    *,
    role: str,
) -> FastAPI:
    """Create test-only routes protected by explicit permissions."""
    settings = PlatformSettings(
        environment="test",
        authentication_mode="service_token",
        auth_token=SecretStr(_VALID_TOKEN),
        auth_principal="m6-authorization-service",
        auth_role=role,  # type: ignore[arg-type]
    )

    app = create_app(settings)

    def read_experiment(
        principal: Annotated[
            AuthenticatedPrincipal,
            Depends(
                require_permission(
                    PlatformPermission.EXPERIMENT_READ
                )
            ),
        ],
    ) -> dict[str, str]:
        return {
            "subject": principal.subject,
            "role": principal.role,
        }

    def run_experiment(
        principal: Annotated[
            AuthenticatedPrincipal,
            Depends(
                require_permission(
                    PlatformPermission.EXPERIMENT_RUN
                )
            ),
        ],
    ) -> dict[str, str]:
        return {
            "subject": principal.subject,
            "role": principal.role,
        }

    def platform_admin(
        principal: Annotated[
            AuthenticatedPrincipal,
            Depends(
                require_permission(
                    PlatformPermission.PLATFORM_ADMIN
                )
            ),
        ],
    ) -> dict[str, str]:
        return {
            "subject": principal.subject,
            "role": principal.role,
        }

    app.add_api_route(
        "/api/v1/_test/experiments/read",
        read_experiment,
        methods=["GET"],
    )

    app.add_api_route(
        "/api/v1/_test/experiments/run",
        run_experiment,
        methods=["POST"],
    )

    app.add_api_route(
        "/api/v1/_test/admin",
        platform_admin,
        methods=["POST"],
    )

    return app


def _authorization_header() -> dict[str, str]:
    """Return one valid authentication header."""
    return {
        "Authorization": (
            f"Bearer {_VALID_TOKEN}"
        ),
    }


def test_role_permission_mapping_is_explicit() -> None:
    """Each supported role must receive only its declared permissions."""
    viewer = permissions_for_role(
        PlatformRole.VIEWER
    )

    operator = permissions_for_role(
        PlatformRole.OPERATOR
    )

    admin = permissions_for_role(
        PlatformRole.ADMIN
    )

    assert viewer == frozenset(
        {
            PlatformPermission.EXPERIMENT_READ,
            PlatformPermission.JOB_READ,
            PlatformPermission.EVIDENCE_READ,
        }
    )

    assert operator == frozenset(
        {
            PlatformPermission.EXPERIMENT_READ,
            PlatformPermission.EXPERIMENT_RUN,
            PlatformPermission.JOB_READ,
            PlatformPermission.JOB_CANCEL,
            PlatformPermission.EVIDENCE_READ,
        }
    )

    assert admin == frozenset(
        PlatformPermission
    )


def test_unknown_role_is_default_deny() -> None:
    """Unknown roles must never inherit platform permissions."""
    assert permissions_for_role(
        "unknown-role"
    ) == frozenset()


def test_viewer_can_read_but_cannot_run() -> None:
    """Viewer role must remain read-only."""
    app = _authorization_app(
        role="viewer",
    )

    with TestClient(app) as client:
        read_response = client.get(
            "/api/v1/_test/experiments/read",
            headers=_authorization_header(),
        )

        run_response = client.post(
            "/api/v1/_test/experiments/run",
            headers=_authorization_header(),
        )

    assert read_response.status_code == 200
    assert read_response.json() == {
        "subject": "m6-authorization-service",
        "role": "viewer",
    }

    assert run_response.status_code == 403

    error = run_response.json()["error"]

    assert error["code"] == "permission_denied"
    assert error["message"] == "Permission denied."
    assert error["details"] == []


def test_operator_can_run_but_cannot_administer_platform() -> None:
    """Operator role must not receive platform-administration authority."""
    app = _authorization_app(
        role="operator",
    )

    with TestClient(app) as client:
        run_response = client.post(
            "/api/v1/_test/experiments/run",
            headers=_authorization_header(),
        )

        admin_response = client.post(
            "/api/v1/_test/admin",
            headers=_authorization_header(),
        )

    assert run_response.status_code == 200
    assert run_response.json()["role"] == "operator"

    assert admin_response.status_code == 403
    assert (
        admin_response.json()["error"]["code"]
        == "permission_denied"
    )


def test_admin_receives_platform_admin_permission() -> None:
    """Admin role may cross the explicit platform-admin gate."""
    app = _authorization_app(
        role="admin",
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/_test/admin",
            headers=_authorization_header(),
        )

    assert response.status_code == 200
    assert response.json()["role"] == "admin"


def test_missing_authentication_is_401_not_403() -> None:
    """Authentication failure must remain distinct from authorisation."""
    app = _authorization_app(
        role="operator",
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/_test/experiments/run"
        )

    assert response.status_code == 401
    assert (
        response.json()["error"]["code"]
        == "authentication_required"
    )


def test_request_header_cannot_escalate_viewer_role() -> None:
    """Untrusted headers must not override the trusted configured role."""
    app = _authorization_app(
        role="viewer",
    )

    headers = _authorization_header()
    headers["X-VAIT-Role"] = "admin"

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/_test/admin",
            headers=headers,
        )

    assert response.status_code == 403
    assert (
        response.json()["error"]["code"]
        == "permission_denied"
    )


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/health/live",
        "/api/v1/meta",
    ],
)
def test_public_routes_remain_public(
    path: str,
) -> None:
    """Authorisation must not become blanket middleware."""
    app = _authorization_app(
        role="viewer",
    )

    with TestClient(app) as client:
        response = client.get(path)

    assert response.status_code == 200


def test_authorized_route_keeps_bearer_security_contract() -> None:
    """Permission dependencies must retain authentication in OpenAPI."""
    app = _authorization_app(
        role="operator",
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/openapi.json"
        )

    assert response.status_code == 200

    operation = response.json()[
        "paths"
    ][
        "/api/v1/_test/experiments/run"
    ][
        "post"
    ]

    assert operation["security"] == [
        {
            "VAITServiceToken": [],
        }
    ]
