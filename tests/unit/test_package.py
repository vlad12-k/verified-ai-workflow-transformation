"""Basic package tests."""

import vait


def test_package_imports() -> None:
    """The package should be importable."""
    assert vait is not None
