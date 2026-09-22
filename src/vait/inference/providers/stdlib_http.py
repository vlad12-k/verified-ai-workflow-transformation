"""Standard-library JSON HTTP transport for inference providers."""

import json
from collections.abc import Mapping
from http.client import HTTPMessage
from typing import IO, Protocol
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)

from pydantic import JsonValue, TypeAdapter

from vait.inference.providers.openai_compatible import (
    OpenAICompatibleHttpResponse,
)

_JSON_OBJECT_ADAPTER = TypeAdapter(
    dict[str, JsonValue]
)


class OutboundUrlValidator(Protocol):
    """Validate one outbound provider URL before network access."""

    def validate_url(
        self,
        url: str,
    ) -> None:
        """Reject an outbound URL that violates provider egress policy."""
        ...


class ProviderRedirectDenied(ValueError):
    """Raised when a provider redirect crosses the original origin."""

    def __init__(
        self,
    ) -> None:
        super().__init__(
            "cross-origin provider redirect denied"
        )


class _ValidatedRedirectHandler(
    HTTPRedirectHandler
):
    """Validate redirect targets before urllib follows them."""

    def __init__(
        self,
        *,
        url_validator: OutboundUrlValidator,
    ) -> None:
        self._url_validator = url_validator

    def redirect_request(
        self,
        req: Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> Request | None:
        """Permit only validated same-origin provider redirects."""
        try:
            self._url_validator.validate_url(
                newurl
            )

            if (
                _origin_identity(
                    req.full_url
                )
                != _origin_identity(
                    newurl
                )
            ):
                raise ProviderRedirectDenied()
        except Exception:
            fp.close()
            raise

        return super().redirect_request(
            req,
            fp,
            code,
            msg,
            headers,
            newurl,
        )


class StdlibJsonTransport:
    """Execute synchronous JSON POST requests through controlled egress."""

    def __init__(
        self,
        *,
        url_validator: OutboundUrlValidator,
    ) -> None:
        """Configure transport with a mandatory outbound URL validator."""
        self._url_validator = url_validator

        self._opener = build_opener(
            ProxyHandler(
                {}
            ),
            _ValidatedRedirectHandler(
                url_validator=url_validator,
            ),
        )

    def post_json(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: dict[str, JsonValue],
        timeout_seconds: float,
    ) -> OpenAICompatibleHttpResponse:
        """POST JSON only after the complete outbound URL passes policy."""
        self._url_validator.validate_url(
            url
        )

        encoded_payload = json.dumps(
            payload,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

        request = Request(
            url=url,
            data=encoded_payload,
            headers=dict(headers),
            method="POST",
        )

        try:
            with self._opener.open(
                request,
                timeout=timeout_seconds,
            ) as response:
                response_body = response.read()

                return OpenAICompatibleHttpResponse(
                    status_code=response.status,
                    body=_decode_json_object(
                        response_body
                    ),
                    headers={
                        name: value
                        for name, value
                        in response.headers.items()
                    },
                )

        except HTTPError as exc:
            response_body = exc.read()

            return OpenAICompatibleHttpResponse(
                status_code=exc.code,
                body=_decode_json_object(
                    response_body
                ),
                headers={
                    name: value
                    for name, value
                    in exc.headers.items()
                },
            )


def _origin_identity(
    url: str,
) -> tuple[str, str]:
    """Return strict scheme-and-authority identity for redirect comparison."""
    parts = urlsplit(
        url
    )

    return (
        parts.scheme.casefold(),
        parts.netloc.casefold(),
    )


def _decode_json_object(
    payload: bytes,
) -> dict[str, JsonValue]:
    """Decode one JSON object returned by a compatible endpoint."""
    if not payload:
        return {}

    decoded: object = json.loads(
        payload.decode("utf-8")
    )

    return _JSON_OBJECT_ADAPTER.validate_python(
        decoded
    )
