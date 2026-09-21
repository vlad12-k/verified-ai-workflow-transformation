"""Structured JSON logging for the VAIT platform."""

import json
import logging
from datetime import UTC, datetime
from typing import TextIO

from vait.platform.observability.context import (
    current_log_context,
)
from vait.platform.security.redaction import (
    redact_sensitive_text,
)

_LOGGER_NAME = "vait"


class JsonLogFormatter(logging.Formatter):
    """Serialize allow-listed platform log data as one JSON object."""

    def format(
        self,
        record: logging.LogRecord,
    ) -> str:
        """Return a deterministic structured log representation."""
        timestamp = datetime.fromtimestamp(
            record.created,
            tz=UTC,
        ).isoformat(
            timespec="milliseconds",
        ).replace(
            "+00:00",
            "Z",
        )

        payload: dict[str, str] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "event": redact_sensitive_text(
                record.getMessage()
            ),
        }

        payload.update(
            current_log_context()
        )

        if record.exc_info is not None:
            exception_type = record.exc_info[0]

            if exception_type is not None:
                payload["exception_type"] = (
                    exception_type.__name__
                )

        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def create_json_log_handler(
    *,
    stream: TextIO | None = None,
) -> logging.Handler:
    """Create one structured stream handler."""
    handler = logging.StreamHandler(
        stream
    )
    handler.setFormatter(
        JsonLogFormatter()
    )

    return handler


def configure_platform_logging(
    *,
    level: int = logging.INFO,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure the dedicated VAIT logger with one JSON handler."""
    logger = logging.getLogger(
        _LOGGER_NAME
    )

    logger.handlers.clear()
    logger.addHandler(
        create_json_log_handler(
            stream=stream,
        )
    )
    logger.setLevel(
        level
    )
    logger.propagate = False

    return logger
