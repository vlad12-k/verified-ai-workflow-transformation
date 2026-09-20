"""Public service metadata endpoint."""

from importlib.metadata import PackageNotFoundError, version
from typing import Literal, cast

from fastapi import APIRouter, Request
from pydantic import BaseModel

from vait.platform.settings import (
    DeploymentEnvironment,
    PlatformSettings,
)

_PACKAGE_NAME = "verified-ai-workflow-transformation"

router = APIRouter(tags=["platform"])


class ServiceMetadataResponse(BaseModel):
    """Non-sensitive public metadata for the running API."""

    service: str
    environment: DeploymentEnvironment
    api_version: Literal["v1"] = "v1"
    package_version: str


def _package_version() -> str:
    """Return installed package version without failing source execution."""
    try:
        return version(_PACKAGE_NAME)
    except PackageNotFoundError:
        return "unknown"


@router.get(
    "/meta",
    response_model=ServiceMetadataResponse,
)
def service_metadata(request: Request) -> ServiceMetadataResponse:
    """Return stable non-sensitive service identity metadata."""
    settings = cast(
        PlatformSettings,
        request.app.state.settings,
    )

    return ServiceMetadataResponse(
        service=settings.service_name,
        environment=settings.environment,
        package_version=_package_version(),
    )
