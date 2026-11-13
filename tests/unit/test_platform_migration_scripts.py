"""Direct tests for executable Alembic migration scripts."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]

_MIGRATIONS = (
    _ROOT
    / "src"
    / "vait"
    / "platform"
    / "persistence"
    / "migrations"
    / "versions"
)


class RecordingOperations:
    """Record structural Alembic operations without requiring PostgreSQL."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def create_table(
        self,
        name: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        del args, kwargs
        self.events.append(
            ("create_table", name)
        )

    def create_index(
        self,
        name: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        del args, kwargs
        self.events.append(
            ("create_index", name)
        )

    def drop_index(
        self,
        name: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        del args, kwargs
        self.events.append(
            ("drop_index", name)
        )

    def drop_table(
        self,
        name: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        del args, kwargs
        self.events.append(
            ("drop_table", name)
        )


def _load_migration(filename: str) -> Any:
    """Load one migration directly from its repository source path."""
    path = _MIGRATIONS / filename

    module_name = (
        "_vait_test_migration_"
        + filename.removesuffix(".py")
    )

    spec = spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise AssertionError(
            f"Unable to load migration: {path}"
        )

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def test_baseline_migration_is_explicit_noop() -> None:
    """The migration baseline must remain an executable no-op."""
    migration = _load_migration(
        "0001_m5d_baseline.py"
    )

    assert migration.revision == "0001_m5d_baseline"
    assert migration.down_revision is None

    migration.upgrade()
    migration.downgrade()


def test_registry_migration_declares_expected_operations() -> None:
    """Registry upgrade/downgrade must preserve its structural contract."""
    migration = _load_migration(
        "0002_m5e_registry.py"
    )

    assert migration.revision == "0002_m5e_registry"
    assert migration.down_revision == "0001_m5d_baseline"

    operations = RecordingOperations()
    migration.op = operations

    migration.upgrade()

    assert operations.events == [
        ("create_table", "experiment_runs"),
        (
            "create_index",
            "ix_experiment_runs_reproduction_of_experiment_id",
        ),
        ("create_table", "candidates"),
        (
            "create_index",
            "ix_candidates_experiment_id",
        ),
        ("create_table", "evidence_records"),
        (
            "create_index",
            "ix_evidence_records_candidate_id",
        ),
        (
            "create_index",
            "ix_evidence_records_experiment_id",
        ),
        (
            "create_index",
            "ix_evidence_records_kind",
        ),
        ("create_table", "artifacts"),
        (
            "create_index",
            "ix_artifacts_evidence_id",
        ),
        (
            "create_index",
            "ix_artifacts_experiment_id",
        ),
    ]

    operations.events.clear()

    migration.downgrade()

    assert operations.events == [
        (
            "drop_index",
            "ix_artifacts_experiment_id",
        ),
        (
            "drop_index",
            "ix_artifacts_evidence_id",
        ),
        ("drop_table", "artifacts"),
        (
            "drop_index",
            "ix_evidence_records_kind",
        ),
        (
            "drop_index",
            "ix_evidence_records_experiment_id",
        ),
        (
            "drop_index",
            "ix_evidence_records_candidate_id",
        ),
        ("drop_table", "evidence_records"),
        (
            "drop_index",
            "ix_candidates_experiment_id",
        ),
        ("drop_table", "candidates"),
        (
            "drop_index",
            "ix_experiment_runs_reproduction_of_experiment_id",
        ),
        ("drop_table", "experiment_runs"),
    ]


def test_jobs_migration_declares_expected_operations() -> None:
    """Jobs migration must preserve its upgrade/downgrade structure."""
    migration = _load_migration(
        "0003_m5f_jobs.py"
    )

    assert migration.revision == "0003_m5f_jobs"
    assert migration.down_revision == "0002_m5e_registry"

    operations = RecordingOperations()
    migration.op = operations

    migration.upgrade()

    assert operations.events == [
        ("create_table", "jobs"),
        (
            "create_index",
            "ix_jobs_experiment_id",
        ),
        (
            "create_index",
            "ix_jobs_status_created_at",
        ),
        (
            "create_index",
            "ix_jobs_lease_expires_at",
        ),
    ]

    operations.events.clear()

    migration.downgrade()

    assert operations.events == [
        (
            "drop_index",
            "ix_jobs_lease_expires_at",
        ),
        (
            "drop_index",
            "ix_jobs_status_created_at",
        ),
        (
            "drop_index",
            "ix_jobs_experiment_id",
        ),
        ("drop_table", "jobs"),
    ]
