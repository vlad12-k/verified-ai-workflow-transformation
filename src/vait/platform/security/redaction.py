"""Defensive redaction primitives for platform telemetry."""

import re
from typing import Final

REDACTION_MARKER: Final[str] = "[REDACTED]"

_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)"
    r"(?P<prefix>\b(?:"
    r"authorization|proxy-authorization|"
    r"api[_-]?key|apikey|"
    r"access[_-]?token|refresh[_-]?token|auth[_-]?token|token|"
    r"password|passwd|secret|client[_-]?secret|"
    r"database[_-]?url|connection[_-]?string"
    r")\b[\"']?\s*[:=]\s*)"
    r"(?P<value>"
    r"(?:Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+|"
    r"\"[^\"]*\"|'[^']*'|"
    r"[^\s,;}\]&]+"
    r")"
)

_AUTH_SCHEME_CREDENTIAL = re.compile(
    r"(?i)"
    r"\b(?P<scheme>Bearer|Basic)\s+"
    r"[A-Za-z0-9._~+/=-]+"
)

_URI_USERINFO = re.compile(
    r"(?i)"
    r"(?P<scheme>[a-z][a-z0-9+.-]*://)"
    r"[^/\s:@]+:[^@\s/]+@"
)


def redact_sensitive_text(
    value: str,
) -> str:
    """Redact recognised credential material from telemetry text."""
    redacted = _SENSITIVE_ASSIGNMENT.sub(
        rf"\g<prefix>{REDACTION_MARKER}",
        value,
    )

    redacted = _AUTH_SCHEME_CREDENTIAL.sub(
        rf"\g<scheme> {REDACTION_MARKER}",
        redacted,
    )

    return _URI_USERINFO.sub(
        rf"\g<scheme>{REDACTION_MARKER}@",
        redacted,
    )
