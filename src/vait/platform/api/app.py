"""FastAPI application factory for the VAIT platform."""

from fastapi import FastAPI

from vait.platform.api.errors import (
    COMMON_ERROR_RESPONSES,
    register_error_handlers,
)
from vait.platform.api.middleware import RequestContextMiddleware
from vait.platform.api.readiness import (
    ReadinessProbe,
    ready_without_dependencies,
)
from vait.platform.api.routes.health import router as health_router
from vait.platform.api.routes.meta import router as meta_router
from vait.platform.settings import PlatformSettings


def create_app(
    settings: PlatformSettings | None = None,
    readiness_probe: ReadinessProbe | None = None,
) -> FastAPI:
    """Create a configured VAIT FastAPI application."""
    resolved_settings = settings or PlatformSettings()

    app = FastAPI(
        title="VAIT Platform API",
        version="0.1.0",
        debug=resolved_settings.debug,
        docs_url=f"{resolved_settings.api_prefix}/docs",
        openapi_url=f"{resolved_settings.api_prefix}/openapi.json",
        redoc_url=None,
        responses=COMMON_ERROR_RESPONSES,
    )

    app.add_middleware(RequestContextMiddleware)
    register_error_handlers(app)

    app.state.settings = resolved_settings
    app.state.readiness_probe = (
        readiness_probe or ready_without_dependencies
    )

    app.include_router(
        health_router,
        prefix=resolved_settings.api_prefix,
    )
    app.include_router(
        meta_router,
        prefix=resolved_settings.api_prefix,
    )

    return app
