"""Tests for the M5-H observability correlation context."""

import pytest

from vait.platform.observability import (
    bind_log_context,
    current_log_context,
    reset_log_context,
)


def test_context_is_empty_by_default() -> None:
    """A new execution context must not invent correlation metadata."""
    assert current_log_context() == {}


def test_context_fields_can_be_bound_and_restored() -> None:
    """Correlation metadata must be scoped and reversible."""
    token = bind_log_context(
        request_id="request-001",
        correlation_id="correlation-001",
        experiment_id="experiment-001",
    )

    try:
        assert current_log_context() == {
            "request_id": "request-001",
            "correlation_id": "correlation-001",
            "experiment_id": "experiment-001",
        }
    finally:
        reset_log_context(
            token
        )

    assert current_log_context() == {}


def test_nested_context_restores_parent_values() -> None:
    """Nested worker or request scopes must not leak context."""
    outer = bind_log_context(
        correlation_id="correlation-outer",
        experiment_id="experiment-001",
    )

    try:
        inner = bind_log_context(
            correlation_id="correlation-inner",
            job_id="job-001",
            worker_id="worker-001",
        )

        try:
            assert current_log_context() == {
                "correlation_id": "correlation-inner",
                "experiment_id": "experiment-001",
                "job_id": "job-001",
                "worker_id": "worker-001",
            }
        finally:
            reset_log_context(
                inner
            )

        assert current_log_context() == {
            "correlation_id": "correlation-outer",
            "experiment_id": "experiment-001",
        }
    finally:
        reset_log_context(
            outer
        )

    assert current_log_context() == {}


def test_none_removes_field_only_within_nested_scope() -> None:
    """A child scope may intentionally remove inherited metadata."""
    outer = bind_log_context(
        request_id="request-001",
        correlation_id="correlation-001",
    )

    try:
        inner = bind_log_context(
            request_id=None,
        )

        try:
            assert current_log_context() == {
                "correlation_id": "correlation-001",
            }
        finally:
            reset_log_context(
                inner
            )

        assert current_log_context() == {
            "request_id": "request-001",
            "correlation_id": "correlation-001",
        }
    finally:
        reset_log_context(
            outer
        )


def test_unsupported_context_field_is_rejected() -> None:
    """Arbitrary application data must not enter log context."""
    with pytest.raises(
        ValueError,
        match="Unsupported observability context fields",
    ):
        bind_log_context(
            customer_email="secret@example.test",
        )


def test_empty_context_value_is_rejected() -> None:
    """Bound correlation identifiers must never be empty strings."""
    with pytest.raises(
        ValueError,
        match="job_id must not be empty",
    ):
        bind_log_context(
            job_id="",
        )


def test_context_snapshot_cannot_mutate_active_context() -> None:
    """Consumers must not be able to mutate ContextVar state by reference."""
    token = bind_log_context(
        job_id="job-001",
    )

    try:
        snapshot = current_log_context()
        snapshot["job_id"] = "mutated"

        assert current_log_context() == {
            "job_id": "job-001",
        }
    finally:
        reset_log_context(
            token
        )
