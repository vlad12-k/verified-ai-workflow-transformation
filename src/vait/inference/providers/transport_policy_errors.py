"""Shared fail-closed transport security denial type."""


class ProviderTransportPolicyDenied(ValueError):
    """A transport security policy rejection, not a retryable failure."""
