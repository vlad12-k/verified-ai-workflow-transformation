"""Standard-library JSON HTTP transport for inference providers."""

import json
from collections.abc import Mapping
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from pydantic import JsonValue, TypeAdapter

from vait.inference.providers.openai_compatible import (
    OpenAICompatibleHttpResponse,
)

_JSON_OBJECT_ADAPTER = TypeAdapter(
    dict[str, JsonValue]
)


class StdlibJsonTransport:
    """Execute synchronous JSON POST requests using Python's standard library."""

    def post_json(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: dict[str, JsonValue],
        timeout_seconds: float,
    ) -> OpenAICompatibleHttpResponse:
        """POST JSON and preserve HTTP response evidence."""
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
            with urlopen(
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
