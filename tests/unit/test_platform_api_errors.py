"""Security-focused tests for the VAIT public API error boundary."""

import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
from starlette.types import Scope

from vait.platform.api.errors import (
    http_exception_handler,
    register_error_handlers,
    validation_exception_handler,
)


def _request(
    *,
    request_id: object | None = None,
    correlation_id: object | None = None,
) -> Request:
    """Create a minimal HTTP request with optional platform context."""
    scope: Scope = {
        "type": "http",
    }

    request = Request(scope)

    if request_id is not None:
        request.state.request_id = request_id

    if correlation_id is not None:
        request.state.correlation_id = correlation_id

    return request


@pytest.mark.parametrize(
    (
        "status_code",
        "expected_code",
        "expected_message",
    ),
    [
        (
            401,
            "authentication_required",
            "Authentication required.",
        ),
        (
            404,
            "not_found",
            "Resource not found.",
        ),
        (
            418,
            "http_error",
            "HTTP request failed.",
        ),
    ],
)
def test_http_errors_use_sanitised_public_contract(
    status_code: int,
    expected_code: str,
    expected_message: str,
) -> None:
    """HTTP failures must expose only the stable public error contract."""
    if status_code == 418:
        request = _request()
    else:
        request = _request(
            request_id="request-123",
            correlation_id="correlation-456",
        )

    secret_detail = "internal-secret-http-detail"

    exception = StarletteHTTPException(
        status_code=status_code,
        detail=secret_detail,
        headers={
            "X-Test-Error-Header": "present",
        },
    )

    response = asyncio.run(
        http_exception_handler(
            request,
            exception,
        )
    )

    assert response.status_code == status_code
    assert (
        response.headers["X-Test-Error-Header"]
        == "present"
    )

    body = json.loads(
        bytes(response.body).decode("utf-8")
    )

    assert body["error"]["code"] == expected_code
    assert (
        body["error"]["message"]
        == expected_message
    )
    assert body["error"]["details"] == []

    if status_code == 418:
        assert body["error"]["request_id"] == "unknown"
        assert body["error"]["correlation_id"] == "unknown"
    else:
        assert body["error"]["request_id"] == "request-123"
        assert (
            body["error"]["correlation_id"]
            == "correlation-456"
        )

    assert secret_detail not in json.dumps(body)


def test_http_handler_rejects_non_http_exception() -> None:
    """Wrong exception types must fail closed at the handler boundary."""
    request = _request()

    with pytest.raises(
        TypeError,
        match=(
            "http_exception_handler received "
            "a non-HTTP exception"
        ),
    ):
        asyncio.run(
            http_exception_handler(
                request,
                RuntimeError(
                    "must not be treated as HTTP"
                ),
            )
        )


def test_validation_handler_sanitises_raw_input() -> None:
    """Validation output must omit the rejected raw input value."""
    request = _request(
        request_id="validation-request",
        correlation_id="validation-correlation",
    )

    raw_secret = "secret-invalid-value"

    exception = RequestValidationError(
        [
            {
                "type": "int_parsing",
                "loc": (
                    "query",
                    "limit",
                ),
                "msg": (
                    "Input should be a valid integer"
                ),
                "input": raw_secret,
            }
        ]
    )

    response = asyncio.run(
        validation_exception_handler(
            request,
            exception,
        )
    )

    assert response.status_code == 422

    body = json.loads(
        bytes(response.body).decode("utf-8")
    )

    assert body == {
        "error": {
            "code": "validation_error",
            "message": "Request validation failed.",
            "request_id": "validation-request",
            "correlation_id": "validation-correlation",
            "details": [
                {
                    "location": [
                        "query",
                        "limit",
                    ],
                    "message": (
                        "Input should be a valid integer"
                    ),
                    "type": "int_parsing",
                }
            ],
        }
    }

    assert raw_secret not in json.dumps(body)


def test_validation_handler_rejects_wrong_exception_type() -> None:
    """Non-validation exceptions must not enter validation handling."""
    request = _request()

    with pytest.raises(
        TypeError,
        match=(
            "validation_exception_handler received "
            "a non-validation exception"
        ),
    ):
        asyncio.run(
            validation_exception_handler(
                request,
                RuntimeError(
                    "wrong exception class"
                ),
            )
        )


def test_error_handlers_register_expected_exception_boundaries() -> None:
    """FastAPI must register the exact VAIT public error handlers."""
    app = FastAPI()

    register_error_handlers(app)

    assert (
        app.exception_handlers[
            StarletteHTTPException
        ]
        is http_exception_handler
    )

    assert (
        app.exception_handlers[
            RequestValidationError
        ]
        is validation_exception_handler
    )
