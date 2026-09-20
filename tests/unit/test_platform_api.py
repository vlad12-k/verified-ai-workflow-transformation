"""Tests for the M5-B FastAPI application contract."""

import json
from typing import Annotated

from fastapi import Query
from fastapi.testclient import TestClient

from vait.platform import PlatformSettings
from vait.platform.api import create_app
from vait.platform.api.readiness import ReadinessCheck


def test_liveness_endpoint_reports_platform_identity() -> None:
    """Liveness must expose only basic process identity."""
    app = create_app(
        PlatformSettings(
            environment="test",
            service_name="vait-test-platform",
        )
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/health/live")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "vait-test-platform",
        "environment": "test",
    }


def test_api_prefix_controls_liveness_route() -> None:
    """Configured API prefix must control route placement."""
    app = create_app(
        PlatformSettings(
            environment="test",
            api_prefix="/custom/v2",
        )
    )

    with TestClient(app) as client:
        default_response = client.get("/api/v1/health/live")
        custom_response = client.get("/custom/v2/health/live")

    assert default_response.status_code == 404
    assert custom_response.status_code == 200


def test_openapi_is_exposed_under_versioned_prefix() -> None:
    """OpenAPI must be exposed under the configured API prefix."""
    app = create_app(
        PlatformSettings(
            environment="test",
        )
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/openapi.json")

    assert response.status_code == 200

    schema = response.json()

    assert schema["info"]["title"] == "VAIT Platform API"
    assert "/api/v1/health/live" in schema["paths"]

def test_readiness_is_ready_without_required_dependencies() -> None:
    """Readiness succeeds before external dependencies are introduced."""
    app = create_app(
        PlatformSettings(
            environment="test",
        )
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": [],
    }


def test_failed_dependency_makes_readiness_unavailable() -> None:
    """A failed required dependency must return HTTP 503."""

    def readiness_probe() -> tuple[ReadinessCheck, ...]:
        return (
            ReadinessCheck(
                name="database",
                ready=False,
            ),
        )

    app = create_app(
        PlatformSettings(
            environment="test",
        ),
        readiness_probe=readiness_probe,
    )

    with TestClient(app) as client:
        readiness_response = client.get("/api/v1/health/ready")
        liveness_response = client.get("/api/v1/health/live")

    assert readiness_response.status_code == 503
    assert readiness_response.json() == {
        "status": "not_ready",
        "checks": [
            {
                "name": "database",
                "ready": False,
            }
        ],
    }

    assert liveness_response.status_code == 200


def test_ready_dependency_is_reported_without_internal_details() -> None:
    """Readiness exposes dependency state without diagnostic internals."""

    def readiness_probe() -> tuple[ReadinessCheck, ...]:
        return (
            ReadinessCheck(
                name="database",
                ready=True,
            ),
        )

    app = create_app(
        PlatformSettings(
            environment="test",
        ),
        readiness_probe=readiness_probe,
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": [
            {
                "name": "database",
                "ready": True,
            }
        ],
    }

def test_service_metadata_exposes_non_sensitive_identity() -> None:
    """Metadata must expose stable service identity without debug details."""
    app = create_app(
        PlatformSettings(
            environment="test",
            service_name="vait-test-platform",
        )
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/meta")

    assert response.status_code == 200

    body = response.json()

    assert body["service"] == "vait-test-platform"
    assert body["environment"] == "test"
    assert body["api_version"] == "v1"
    assert isinstance(body["package_version"], str)
    assert body["package_version"]

    assert "debug" not in body


def test_request_context_headers_are_generated() -> None:
    """Every HTTP response must carry request and correlation identifiers."""
    from uuid import UUID

    app = create_app(
        PlatformSettings(
            environment="test",
        )
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/health/live")

    request_id = response.headers["X-Request-ID"]
    correlation_id = response.headers["X-Correlation-ID"]

    assert str(UUID(request_id)) == request_id
    assert correlation_id == request_id


def test_safe_inbound_correlation_id_is_propagated() -> None:
    """A bounded log-safe correlation identifier may cross the API boundary."""
    app = create_app(
        PlatformSettings(
            environment="test",
        )
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/health/live",
            headers={
                "X-Correlation-ID": "experiment-123:job-456",
            },
        )

    assert response.status_code == 200
    assert (
        response.headers["X-Correlation-ID"]
        == "experiment-123:job-456"
    )
    assert (
        response.headers["X-Request-ID"]
        != response.headers["X-Correlation-ID"]
    )


def test_unsafe_inbound_correlation_id_is_not_propagated() -> None:
    """Unsafe correlation identifiers must be replaced at the trust boundary."""
    unsafe_id = "unsafe correlation id with spaces"

    app = create_app(
        PlatformSettings(
            environment="test",
        )
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/health/live",
            headers={
                "X-Correlation-ID": unsafe_id,
            },
        )

    request_id = response.headers["X-Request-ID"]

    assert response.status_code == 200
    assert response.headers["X-Correlation-ID"] == request_id
    assert response.headers["X-Correlation-ID"] != unsafe_id


def test_overlong_correlation_id_is_not_propagated() -> None:
    """Correlation identifiers must have a bounded external representation."""
    overlong_id = "a" * 129

    app = create_app(
        PlatformSettings(
            environment="test",
        )
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/health/live",
            headers={
                "X-Correlation-ID": overlong_id,
            },
        )

    assert (
        response.headers["X-Correlation-ID"]
        == response.headers["X-Request-ID"]
    )

def test_not_found_uses_typed_error_contract() -> None:
    """Unknown routes must use the public VAIT error envelope."""
    app = create_app(
        PlatformSettings(
            environment="test",
        )
    )

    correlation_id = "request-chain-404"

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/does-not-exist",
            headers={
                "X-Correlation-ID": correlation_id,
            },
        )

    assert response.status_code == 404

    body = response.json()
    error = body["error"]

    assert error["code"] == "not_found"
    assert error["message"] == "Resource not found."
    assert error["details"] == []

    assert (
        error["request_id"]
        == response.headers["X-Request-ID"]
    )
    assert error["correlation_id"] == correlation_id
    assert (
        response.headers["X-Correlation-ID"]
        == correlation_id
    )

    assert "detail" not in body


def test_validation_error_uses_sanitised_typed_contract() -> None:
    """Validation failures must not echo raw invalid input."""

    app = create_app(
        PlatformSettings(
            environment="test",
        )
    )

    def validated_probe(
        limit: Annotated[int, Query(ge=1, le=10)],
    ) -> dict[str, int]:
        return {"limit": limit}

    app.add_api_route(
        "/api/v1/_test/validated",
        validated_probe,
        methods=["GET"],
    )

    invalid_value = "super-secret-invalid-value"

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/_test/validated",
            params={"limit": invalid_value},
        )

    assert response.status_code == 422

    body = response.json()
    error = body["error"]

    assert error["code"] == "validation_error"
    assert error["message"] == "Request validation failed."

    assert (
        error["request_id"]
        == response.headers["X-Request-ID"]
    )
    assert (
        error["correlation_id"]
        == response.headers["X-Correlation-ID"]
    )

    assert len(error["details"]) == 1

    issue = error["details"][0]

    assert issue["location"] == ["query", "limit"]
    assert isinstance(issue["message"], str)
    assert issue["message"]
    assert isinstance(issue["type"], str)
    assert issue["type"]

    serialised = json.dumps(body)

    assert invalid_value not in serialised
    assert "input" not in issue
    assert "ctx" not in issue


def test_openapi_documents_platform_error_contract() -> None:
    """OpenAPI must expose the same error model returned at runtime."""
    app = create_app(
        PlatformSettings(
            environment="test",
        )
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/openapi.json")

    assert response.status_code == 200

    schema = response.json()

    assert "ErrorResponse" in schema["components"]["schemas"]
    assert "ErrorBody" in schema["components"]["schemas"]
    assert "ValidationIssue" in schema["components"]["schemas"]

    live_responses = schema["paths"][
        "/api/v1/health/live"
    ]["get"]["responses"]

    assert (
        live_responses["404"]["content"][
            "application/json"
        ]["schema"]["$ref"]
        == "#/components/schemas/ErrorResponse"
    )

    assert (
        live_responses["422"]["content"][
            "application/json"
        ]["schema"]["$ref"]
        == "#/components/schemas/ErrorResponse"
    )
