"""Architecture guard for the M5 platform dependency boundary."""

from pathlib import Path

CORE_PACKAGES = (
    "contracts",
    "decision",
    "economics",
    "evidence",
    "inference",
    "optimisation",
    "rag",
    "retrieval",
    "runners",
    "transformations",
    "verification",
)

FORBIDDEN_IMPORT_MARKERS = (
    "from fastapi",
    "import fastapi",
    "from sqlalchemy",
    "import sqlalchemy",
    "from redis",
    "import redis",
    "from alembic",
    "import alembic",
    "from opentelemetry",
    "import opentelemetry",
    "from vait.platform",
    "import vait.platform",
)

SOURCE_ROOT = Path("src/vait")


def test_core_does_not_depend_on_platform_layer() -> None:
    """VAIT core packages must not import platform frameworks."""
    violations: list[str] = []

    for package_name in CORE_PACKAGES:
        package_root = SOURCE_ROOT / package_name

        if not package_root.exists():
            continue

        for path in sorted(package_root.rglob("*.py")):
            source = path.read_text(encoding="utf-8")

            for marker in FORBIDDEN_IMPORT_MARKERS:
                if marker in source:
                    violations.append(
                        f"{path}: forbidden import marker {marker!r}"
                    )

    assert not violations, "\n".join(violations)
