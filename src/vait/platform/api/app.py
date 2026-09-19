"""FastAPI application factory for the VAIT platform."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from vait.platform.api.database import create_database_readiness_probe
from vait.platform.api.errors import (
    COMMON_ERROR_RESPONSES,
    register_error_handlers,
)
from vait.platform.api.middleware import RequestContextMiddleware
from vait.platform.api.readiness import (
    ReadinessProbe,
    combine_readiness_probes,
    ready_without_dependencies,
)
from vait.platform.api.routes.health import router as health_router
from vait.platform.api.routes.meta import router as meta_router
from vait.platform.persistence import (
    DatabaseRuntime,
    create_database_runtime,
)
from vait.platform.settings import PlatformSettings


def create_app(
    settings: PlatformSettings | None = None,
    readiness_probe: ReadinessProbe | None = None,
    database_runtime_factory: Callable[
        [PlatformSettings],
        DatabaseRuntime,
    ] = create_database_runtime,
) -> FastAPI:
    """Create a configured VAIT FastAPI application."""
    resolved_settings = settings or PlatformSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime: DatabaseRuntime | None = None

        active_probe = (
            readiness_probe
            or ready_without_dependencies
        )

        try:
            if resolved_settings.database_url is not None:
                runtime = database_runtime_factory(
                    resolved_settings
                )

                app.state.database_runtime = runtime

                active_probe = combine_readiness_probes(
                    active_probe,
                    create_database_readiness_probe(runtime),
                )

            app.state.readiness_probe = active_probe

            yield
        finally:
            if runtime is not None:
                runtime.dispose()

            app.state.database_runtime = None

    app = FastAPI(
        title="VAIT Platform API",
        version="0.1.0",
        debug=resolved_settings.debug,
        docs_url=f"{resolved_settings.api_prefix}/docs",
        openapi_url=f"{resolved_settings.api_prefix}/openapi.json",
        redoc_url=None,
        responses=COMMON_ERROR_RESPONSES,
        lifespan=lifespan,
    )

    app.add_middleware(RequestContextMiddleware)
    register_error_handlers(app)

    app.state.settings = resolved_settings
    app.state.database_runtime = None
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
