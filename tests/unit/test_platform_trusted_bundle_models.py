"""Schema tests for durable trusted verification bundles."""

from typing import cast

from sqlalchemy import Table
from sqlalchemy.sql.sqltypes import Text

from vait.platform.persistence import registry_models


def test_trusted_bundle_model_uses_immutable_experiment_identity() -> None:
    """One experiment owns one canonical bundle with strict lineage columns."""
    model = getattr(registry_models, "TrustedVerificationBundleRow", None)
    assert model is not None, "TrustedVerificationBundleRow is not registered"
    table = cast(Table, model.__table__)

    assert table.name == "trusted_verification_bundles"
    assert set(table.columns.keys()) == {
        "experiment_id",
        "bundle_version",
        "contract_id",
        "contract_version",
        "source_revision",
        "fingerprint_sha256",
        "canonical_json",
    }
    assert list(table.primary_key.columns.keys()) == ["experiment_id"]
    assert {
        foreign_key.target_fullname for foreign_key in table.foreign_keys
    } == {"experiment_runs.experiment_id"}
    assert isinstance(table.c.canonical_json.type, Text)
    assert not any(column.name in {"updated_at", "mutable_payload"} for column in table.columns)

    constraint_names = {
        constraint.name for constraint in table.constraints
    }
    assert {
        "ck_trusted_verification_bundles_bundle_version_nonempty",
        "ck_trusted_verification_bundles_contract_id_nonempty",
        "ck_trusted_verification_bundles_contract_version_nonempty",
        "ck_trusted_verification_bundles_source_revision_length",
        "ck_trusted_verification_bundles_fingerprint_length",
        "ck_trusted_verification_bundles_canonical_json_nonempty",
    }.issubset(constraint_names)
