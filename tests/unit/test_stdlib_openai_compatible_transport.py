"""Tests for the stdlib OpenAI-compatible HTTP transport."""

import json
from dataclasses import dataclass, field
from http.server import (
    BaseHTTPRequestHandler,
    ThreadingHTTPServer,
)
from threading import Thread
from typing import Any

from vait.inference.providers.models import (
    InferenceMessage,
    InferenceRole,
    ProviderInferenceRequest,
)
from vait.inference.providers.openai_compatible import (
    OpenAICompatibleProvider,
)
from vait.inference.providers.stdlib_http import (
    StdlibJsonTransport,
)


@dataclass
class RequestRecord:
    """Evidence captured from one localhost HTTP request."""

    path: str | None = None
    headers: dict[str, str] = field(
        default_factory=dict
    )
    body: dict[str, Any] = field(
        default_factory=dict
    )


def build_handler(
    *,
    record: RequestRecord,
    status_code: int,
    response_body: dict[str, Any],
) -> type[BaseHTTPRequestHandler]:
    """Build one deterministic localhost request handler."""

    class Handler(BaseHTTPRequestHandler):
        """Serve one deterministic OpenAI-compatible response."""

        def do_POST(self) -> None:
            """Record POST evidence and return configured JSON."""
            content_length = int(
                self.headers.get(
                    "Content-Length",
                    "0",
                )
            )

            raw_body = self.rfile.read(
                content_length
            )

            decoded_body = json.loads(
                raw_body.decode("utf-8")
            )

            if not isinstance(decoded_body, dict):
                raise TypeError(
                    "Expected JSON object request body."
                )

            record.path = self.path
            record.headers = {
                name: value
                for name, value
                in self.headers.items()
            }
            record.body = decoded_body

            encoded_response = json.dumps(
                response_body
            ).encode("utf-8")

            self.send_response(status_code)
            self.send_header(
                "Content-Type",
                "application/json",
            )
            self.send_header(
                "Content-Length",
                str(len(encoded_response)),
            )
            self.end_headers()

            self.wfile.write(
                encoded_response
            )

        def log_message(
            self,
            format: str,
            *args: object,
        ) -> None:
            """Suppress localhost server logging during tests."""

    return Handler


def start_server(
    *,
    record: RequestRecord,
    status_code: int,
    response_body: dict[str, Any],
) -> tuple[
    ThreadingHTTPServer,
    Thread,
]:
    """Start one ephemeral localhost JSON server."""
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        build_handler(
            record=record,
            status_code=status_code,
            response_body=response_body,
        ),
    )

    thread = Thread(
        target=server.serve_forever,
        daemon=True,
    )
    thread.start()

    return server, thread


def build_request() -> ProviderInferenceRequest:
    """Build one portable request for HTTP integration tests."""
    return ProviderInferenceRequest(
        request_id="localhost-request",
        model_id="portable-model",
        messages=(
            InferenceMessage(
                role=InferenceRole.SYSTEM,
                content="Use supplied evidence only.",
            ),
            InferenceMessage(
                role=InferenceRole.USER,
                content="Return the verified result.",
            ),
        ),
        max_output_tokens=24,
        temperature=0.0,
    )


def test_stdlib_transport_executes_real_local_http_round_trip() -> None:
    """Provider should complete through a real localhost HTTP POST."""
    record = RequestRecord()

    server, thread = start_server(
        record=record,
        status_code=200,
        response_body={
            "id": "chatcmpl-local-test",
            "model": "portable-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "verified localhost response",
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 4,
                "total_tokens": 15,
            },
        },
    )

    try:
        host, port = server.server_address

        provider = OpenAICompatibleProvider(
            base_url=f"http://{host}:{port}",
            model_id="portable-model",
            transport=StdlibJsonTransport(),
            api_key="local-test-key",
            timeout_seconds=5.0,
        )

        response = provider.generate(
            build_request()
        )

    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)

    assert response.succeeded is True
    assert response.output_text == (
        "verified localhost response"
    )

    assert response.usage is not None
    assert response.usage.input_tokens == 11
    assert response.usage.output_tokens == 4

    assert response.timing.total_latency_ms > 0.0
    assert response.timing.time_to_first_token_ms is None
    assert response.timing.generation_latency_ms is None

    assert record.path == (
        "/v1/chat/completions"
    )

    assert (
        record.headers["Authorization"]
        == "Bearer local-test-key"
    )
    assert record.body["model"] == "portable-model"
    assert record.body["max_tokens"] == 24
    assert record.body["stream"] is False

    messages = record.body["messages"]

    assert isinstance(messages, list)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_stdlib_transport_preserves_http_error_for_provider_normalisation() -> None:
    """Real HTTP 429 evidence should remain available to the provider."""
    record = RequestRecord()

    server, thread = start_server(
        record=record,
        status_code=429,
        response_body={
            "error": {
                "type": "rate_limit_error",
                "message": "Local test rate limit.",
            }
        },
    )

    try:
        host, port = server.server_address

        provider = OpenAICompatibleProvider(
            base_url=f"http://{host}:{port}/v1",
            model_id="portable-model",
            transport=StdlibJsonTransport(),
            timeout_seconds=5.0,
        )

        response = provider.generate(
            build_request()
        )

    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)

    assert response.succeeded is False

    assert response.error is not None
    assert response.error.error_type == (
        "rate_limit_error"
    )
    assert response.error.message == (
        "Local test rate limit."
    )
    assert response.error.retryable is True

    assert response.metadata["http_status"] == 429

    assert record.path == (
        "/v1/chat/completions"
    )
