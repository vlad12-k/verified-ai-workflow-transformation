"""Tests for the VAIT command-line interface."""

from typer.testing import CliRunner

from vait.cli import app

runner = CliRunner()


def test_version_command() -> None:
    """The version command should report the installed VAIT version."""
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert "VAIT 0.1.0" in result.stdout
