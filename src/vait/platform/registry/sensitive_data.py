"""Fail-closed sensitive-data validation for durable evidence payloads."""

import re
from collections.abc import Mapping
from typing import Final

from pydantic import JsonValue

_PROHIBITED_FIELD_NAMES: Final[frozenset[str]] = frozenset(
    {
        "authorization",
        "proxyauthorization",
        "apikey",
        "token",
        "accesstoken",
        "refreshtoken",
        "authtoken",
        "password",
        "passwd",
        "secret",
        "clientsecret",
        "databaseurl",
        "connectionstring",
    }
)

_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)"
    r"\b(?:"
    r"authorization|proxy[-_. ]?authorization|"
    r"api[-_. ]?key|apikey|"
    r"access[-_. ]?token|refresh[-_. ]?token|"
    r"auth[-_. ]?token|token|"
    r"password|passwd|secret|client[-_. ]?secret|"
    r"database[-_. ]?url|connection[-_. ]?string"
    r")\b"
    r"\s*[:=]"
)

_AUTH_SCHEME_CREDENTIAL = re.compile(
    r"(?i)"
    r"(?:^|\s)"
    r"(?:Bearer|Basic)\s+"
    r"[A-Za-z0-9._~+/=-]+"
)

_URI_USERINFO = re.compile(
    r"(?i)"
    r"[a-z][a-z0-9+.-]*://"
    r"[^/\s:@]+:[^@\s/]+@"
)


def _normalise_field_name(
    value: str,
) -> str:
    """Return a separator-insensitive field identity."""
    return "".join(
        character
        for character in value.casefold()
        if character.isalnum()
    )


def _is_prohibited_field_name(
    value: str,
) -> bool:
    """Return whether a field name denotes credential material."""
    return (
        _normalise_field_name(value)
        in _PROHIBITED_FIELD_NAMES
    )


def _string_contains_prohibited_secret(
    value: str,
) -> bool:
    """Detect unambiguous credential structures inside free text."""
    return any(
        pattern.search(value) is not None
        for pattern in (
            _CREDENTIAL_ASSIGNMENT,
            _AUTH_SCHEME_CREDENTIAL,
            _URI_USERINFO,
        )
    )


def contains_prohibited_evidence_secret(
    value: JsonValue,
) -> bool:
    """Recursively detect prohibited credential material in JSON evidence."""
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            if _is_prohibited_field_name(
                key
            ):
                return True

            if contains_prohibited_evidence_secret(
                nested_value
            ):
                return True

        return False

    if isinstance(value, list):
        return any(
            contains_prohibited_evidence_secret(
                item
            )
            for item in value
        )

    if isinstance(value, str):
        return _string_contains_prohibited_secret(
            value
        )

    return False


def validate_evidence_payload(
    payload: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Reject secret-bearing evidence without mutating valid evidence."""
    if contains_prohibited_evidence_secret(
        payload
    ):
        raise ValueError(
            "evidence payload contains prohibited secret material"
        )

    return payload
