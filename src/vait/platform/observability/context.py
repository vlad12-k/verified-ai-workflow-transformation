"""Safe correlation context for platform observability."""

from contextvars import ContextVar, Token
from typing import Final

_ALLOWED_CONTEXT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "request_id",
        "correlation_id",
        "experiment_id",
        "job_id",
        "worker_id",
        "trace_id",
        "span_id",
    }
)

_LOG_CONTEXT: ContextVar[dict[str, str] | None] = ContextVar(
    "vait_log_context",
    default=None,
)


def current_log_context() -> dict[str, str]:
    """Return an isolated snapshot of the current observability context."""
    context = _LOG_CONTEXT.get()

    if context is None:
        return {}

    return dict(context)


def bind_log_context(
    **fields: str | None,
) -> Token[dict[str, str] | None]:
    """Bind allow-listed correlation fields to the current context."""
    unknown_fields = (
        set(fields)
        - _ALLOWED_CONTEXT_FIELDS
    )

    if unknown_fields:
        names = ", ".join(
            sorted(unknown_fields)
        )
        raise ValueError(
            f"Unsupported observability context fields: {names}"
        )

    context = current_log_context()

    for name, value in fields.items():
        if value is None:
            context.pop(
                name,
                None,
            )
            continue

        if not value:
            raise ValueError(
                f"{name} must not be empty"
            )

        context[name] = value

    return _LOG_CONTEXT.set(
        context
    )


def reset_log_context(
    token: Token[dict[str, str] | None],
) -> None:
    """Restore the observability context associated with a previous token."""
    _LOG_CONTEXT.reset(
        token
    )
