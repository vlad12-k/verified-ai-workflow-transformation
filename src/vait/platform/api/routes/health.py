"""Platform health endpoints."""

from typing import Literal, cast

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel

from vait.platform.api.readiness import ReadinessProbe
from vait.platform.settings import (
    DeploymentEnvironment,
    PlatformSettings,
)

router = APIRouter(
    prefix="/health",
    tags=["health"],
)


class LivenessResponse(BaseModel):
    """Response returned by the liveness endpoint."""

    status: Literal["ok"] = "ok"
    service: str
    environment: DeploymentEnvironment


class ReadinessCheckResponse(BaseModel):
    """Public readiness result for one required dependency."""

    name: str
    ready: bool


class ReadinessResponse(BaseModel):
    """Response returned by the readiness endpoint."""

    status: Literal["ready", "not_ready"]
    checks: list[ReadinessCheckResponse]


@router.get(
    "/live",
    response_model=LivenessResponse,
)
def liveness(request: Request) -> LivenessResponse:
    """Report whether the API process is alive."""
    settings = cast(
        PlatformSettings,
        request.app.state.settings,
    )

    return LivenessResponse(
        service=settings.service_name,
        environment=settings.environment,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
)
def readiness(
    request: Request,
    response: Response,
) -> ReadinessResponse:
    """Report whether required platform dependencies are ready."""
    probe = cast(
        ReadinessProbe,
        request.app.state.readiness_probe,
    )

    checks = tuple(probe())
    is_ready = all(check.ready for check in checks)

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status="ready" if is_ready else "not_ready",
        checks=[
            ReadinessCheckResponse(
                name=check.name,
                ready=check.ready,
            )
            for check in checks
        ],
    )
