"""Tests for the M5-D Alembic migration framework."""

from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

from vait.platform.persistence.models import (
    NAMING_CONVENTION,
    Base,
)

_ROOT = Path(__file__).resolve().parents[2]
_ALEMBIC_INI = _ROOT / "alembic.ini"


def test_alembic_preserves_baseline_and_has_single_registry_head() -> None:
    """Migration history must preserve the baseline and one current head."""
    config = Config(str(_ALEMBIC_INI))
    scripts = ScriptDirectory.from_config(config)

    assert scripts.get_current_head() == "0002_m5e_registry"

    baseline = scripts.get_revision("0001_m5d_baseline")
    registry = scripts.get_revision("0002_m5e_registry")

    assert baseline is not None
    assert baseline.down_revision is None
    assert registry is not None
    assert registry.down_revision == "0001_m5d_baseline"


def test_alembic_configuration_contains_no_database_credentials() -> None:
    """Repository migration config must not persist connection secrets."""
    content = _ALEMBIC_INI.read_text(encoding="utf-8").lower()

    assert "sqlalchemy.url" not in content
    assert "postgresql://" not in content
    assert "postgresql+psycopg://" not in content
    assert "password" not in content


def test_database_metadata_uses_deterministic_naming_convention() -> None:
    """Future generated constraints must receive stable names."""
    assert Base.metadata.naming_convention is not None

    for key in ("ix", "uq", "ck", "fk", "pk"):
        assert key in NAMING_CONVENTION
        assert key in Base.metadata.naming_convention
