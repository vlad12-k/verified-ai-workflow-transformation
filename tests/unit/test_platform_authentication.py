"""Tests for the M6-B inbound platform authentication boundary."""

import json
from typing import Annotated, cast

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient
from pydantic import SecretStr
from starlette.requests import Request
from starlette.types import Scope

from vait.platform import PlatformSettings
from vait.platform.api import create_app
from vait.platform.security import (
    AuthenticatedPrincipal,
    require_authenticated_principal,
)

_VALID_TOKEN = "test-platform-service-token-123456789"


def _protected_app(
    settings: PlatformSettings,
) -> FastAPI:
    """Create an app with one test-only protected operation."""
    app = create_app(settings)

    def protected_operation(
        principal: Annotated[
            AuthenticatedPrincipal,
            Depends(require_authenticated_principal),
        ],
    ) -> dict[str, str]:
        return {
            "subject": principal.subject,
            "authentication_method": (
                principal.authentication_method
            ),
            "role": principal.role,
        }

    app.add_api_route(
        "/api/v1/_test/protected",
        protected_operation,
        methods=["GET"],
    )

    return app


def test_public_health_remains_available_when_authentication_enabled() -> None:
    """Health probes must remain usable without platform credentials."""
    app = _protected_app(
        PlatformSettings(
            environment="test",
            authentication_mode="service_token",
            auth_token=SecretStr(_VALID_TOKEN),
        )
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/health/live"
        )

    assert response.status_code == 200


def test_protected_operation_fails_closed_when_authentication_disabled() -> None:
    """A protected dependency must not silently bypass disabled auth."""
    app = _protected_app(
        PlatformSettings(
            environment="test",
        )
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/_test/protected"
        )

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert (
        response.json()["error"]["code"]
        == "authentication_required"
    )


def test_protected_operation_requires_bearer_credentials() -> None:
    """Missing credentials must return the typed authentication failure."""
    app = _protected_app(
        PlatformSettings(
            environment="test",
            authentication_mode="service_token",
            auth_token=SecretStr(_VALID_TOKEN),
        )
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/_test/protected"
        )

    assert response.status_code == 401

    body = response.json()

    assert body["error"]["code"] == "authentication_required"
    assert body["error"]["message"] == "Authentication required."
    assert body["error"]["details"] == []

    assert (
        body["error"]["request_id"]
        == response.headers["X-Request-ID"]
    )
    assert (
        body["error"]["correlation_id"]
        == response.headers["X-Correlation-ID"]
    )

    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_wrong_service_token_is_rejected_without_secret_leakage() -> None:
    """Invalid credentials must not appear in the public failure body."""
    wrong_token = "wrong-platform-service-token-123456789"

    app = _protected_app(
        PlatformSettings(
            environment="test",
            authentication_mode="service_token",
            auth_token=SecretStr(_VALID_TOKEN),
        )
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/_test/protected",
            headers={
                "Authorization": f"Bearer {wrong_token}",
            },
        )

    assert response.status_code == 401

    serialised = json.dumps(
        response.json()
    )

    assert wrong_token not in serialised
    assert _VALID_TOKEN not in serialised


def test_non_bearer_authentication_is_rejected() -> None:
    """Protected operations must not accept an alternative auth scheme."""
    app = _protected_app(
        PlatformSettings(
            environment="test",
            authentication_mode="service_token",
            auth_token=SecretStr(_VALID_TOKEN),
        )
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/_test/protected",
            headers={
                "Authorization": (
                    "Basic dXNlcjpwYXNzd29yZA=="
                ),
            },
        )

    assert response.status_code == 401
    assert (
        response.json()["error"]["code"]
        == "authentication_required"
    )


def test_valid_service_token_establishes_principal() -> None:
    """A correct service token must establish the configured principal."""
    app = _protected_app(
        PlatformSettings(
            environment="test",
            authentication_mode="service_token",
            auth_token=SecretStr(_VALID_TOKEN),
            auth_principal="m6-test-service",
        )
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/_test/protected",
            headers={
                "Authorization": (
                    f"Bearer {_VALID_TOKEN}"
                ),
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "subject": "m6-test-service",
        "authentication_method": "service_token",
        "role": "viewer",
    }


def test_direct_authentication_dependency_rejects_malformed_credentials(
    ) -> None:
    """Authentication dependency must fail closed for malformed credentials."""
    app = create_app(
        PlatformSettings(
            environment="test",
            authentication_mode="service_token",
            auth_token=SecretStr(_VALID_TOKEN),
        )
    )

    request = Request(
        cast(
            Scope,
            {
                "type": "http",
                "app": app,
            },
        )
    )

    malformed_credentials = [
        HTTPAuthorizationCredentials(
            scheme="Basic",
            credentials=_VALID_TOKEN,
        ),
        HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="",
        ),
    ]

    for credentials in malformed_credentials:
        with pytest.raises(
            HTTPException,
        ) as exc_info:
            require_authenticated_principal(
                request,
                credentials,
            )

        assert exc_info.value.status_code == 401
        assert exc_info.value.headers == {
            "WWW-Authenticate": "Bearer",
        }


def test_openapi_documents_service_token_security_scheme() -> None:
    """Protected operations must expose their authentication requirement."""
    app = _protected_app(
        PlatformSettings(
            environment="test",
            authentication_mode="service_token",
            auth_token=SecretStr(_VALID_TOKEN),
        )
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/openapi.json"
        )

    assert response.status_code == 200

    schema = response.json()

    security_schemes = schema[
        "components"
    ]["securitySchemes"]

    assert security_schemes[
        "VAITServiceToken"
    ]["type"] == "http"

    assert security_schemes[
        "VAITServiceToken"
    ]["scheme"] == "bearer"

    protected_operation = schema[
        "paths"
    ]["/api/v1/_test/protected"]["get"]

    assert protected_operation["security"] == [
        {
            "VAITServiceToken": [],
        }
    ]

    public_health = schema[
        "paths"
    ]["/api/v1/health/live"]["get"]

    assert "security" not in public_health


def test_service_token_requires_exact_match() -> None:
    """Valid tokens must not authenticate by prefix or suffix matching."""
    app = _protected_app(
        PlatformSettings(
            environment="test",
            authentication_mode="service_token",
            auth_token=SecretStr(_VALID_TOKEN),
        )
    )

    candidates = [
        _VALID_TOKEN[:-1],
        _VALID_TOKEN + "x",
        "x" + _VALID_TOKEN,
    ]

    with TestClient(app) as client:
        for candidate in candidates:
            response = client.get(
                "/api/v1/_test/protected",
                headers={
                    "Authorization": f"Bearer {candidate}",
                },
            )

            assert response.status_code == 401
            assert (
                response.json()["error"]["code"]
                == "authentication_required"
            )


def test_authentication_failure_does_not_echo_authorization_header() -> None:
    """Authentication failures must not reflect credentials into responses."""
    supplied_secret = (
        "attacker-controlled-secret-token-123456789"
    )

    app = _protected_app(
        PlatformSettings(
            environment="test",
            authentication_mode="service_token",
            auth_token=SecretStr(_VALID_TOKEN),
        )
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/_test/protected",
            headers={
                "Authorization": (
                    f"Bearer {supplied_secret}"
                ),
            },
        )

    assert response.status_code == 401

    serialised = json.dumps(response.json())

    assert supplied_secret not in serialised
    assert _VALID_TOKEN not in serialised


def test_public_metadata_remains_available_when_authentication_enabled() -> None:
    """Public service metadata must not require the service credential."""
    app = _protected_app(
        PlatformSettings(
            environment="test",
            authentication_mode="service_token",
            auth_token=SecretStr(_VALID_TOKEN),
        )
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/meta"
        )

    assert response.status_code == 200
