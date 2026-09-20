"""Typed HTTP error contracts for the VAIT platform."""

from collections.abc import Mapping
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

ErrorCode = Literal[
    "not_found",
    "http_error",
    "validation_error",
]


class ValidationIssue(BaseModel):
    """Sanitised description of one request validation failure."""

    location: list[str]
    message: str
    type: str


class ErrorBody(BaseModel):
    """Stable public error representation."""

    code: ErrorCode
    message: str
    request_id: str
    correlation_id: str
    details: list[ValidationIssue] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Top-level API error envelope."""

    error: ErrorBody


COMMON_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    404: {
        "model": ErrorResponse,
        "description": "Resource not found.",
    },
    422: {
        "model": ErrorResponse,
        "description": "Request validation failed.",
    },
}


def _request_context(request: Request) -> tuple[str, str]:
    """Return request identifiers established by request middleware."""
    request_id_value = getattr(request.state, "request_id", None)
    correlation_id_value = getattr(
        request.state,
        "correlation_id",
        None,
    )

    request_id = (
        request_id_value
        if isinstance(request_id_value, str)
        else "unknown"
    )
    correlation_id = (
        correlation_id_value
        if isinstance(correlation_id_value, str)
        else request_id
    )

    return request_id, correlation_id


def _error_response(
    *,
    request: Request,
    status_code: int,
    code: ErrorCode,
    message: str,
    details: list[ValidationIssue] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    """Build the stable public error envelope."""
    request_id, correlation_id = _request_context(request)

    body = ErrorResponse(
        error=ErrorBody(
            code=code,
            message=message,
            request_id=request_id,
            correlation_id=correlation_id,
            details=details or [],
        )
    )

    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
        headers=headers,
    )


async def http_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Convert framework HTTP errors to the VAIT error contract."""
    if not isinstance(exc, StarletteHTTPException):
        raise TypeError(
            "http_exception_handler received a non-HTTP exception"
        )

    if exc.status_code == 404:
        code: ErrorCode = "not_found"
        message = "Resource not found."
    else:
        code = "http_error"
        message = "HTTP request failed."

    return _error_response(
        request=request,
        status_code=exc.status_code,
        code=code,
        message=message,
        headers=exc.headers,
    )


async def validation_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Return sanitised request-validation failures."""
    if not isinstance(exc, RequestValidationError):
        raise TypeError(
            "validation_exception_handler received "
            "a non-validation exception"
        )

    details = [
        ValidationIssue(
            location=[str(part) for part in error["loc"]],
            message=error["msg"],
            type=error["type"],
        )
        for error in exc.errors()
    ]

    return _error_response(
        request=request,
        status_code=422,
        code="validation_error",
        message="Request validation failed.",
        details=details,
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register the platform-wide typed error handlers."""
    app.add_exception_handler(
        StarletteHTTPException,
        http_exception_handler,
    )
    app.add_exception_handler(
        RequestValidationError,
        validation_exception_handler,
    )
