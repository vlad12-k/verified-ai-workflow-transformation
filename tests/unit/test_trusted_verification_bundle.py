"""Tests for immutable trusted verification bundles."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import FrozenInstanceError
from typing import cast
from uuid import UUID

import pytest

from vait.contracts.models import (
    Effect,
    ExactVerificationScope,
    Invariant,
    InvariantOperator,
    RiskConstraint,
    RiskLevel,
    TransformationContract,
    VerificationCase,
)
from vait.verification.trusted_bundle import TrustedVerificationBundle


def build_contract() -> TransformationContract:
    """Build a contract with an explicitly finite verification scope."""
    return TransformationContract(
        id="contract-v1",
        reference_implementation_id="reference-v1",
        candidate_implementation_id="candidate-v1",
        risk=RiskConstraint(level=RiskLevel.HIGH),
        exact_verification_scope=ExactVerificationScope(
            exhaustive_case_ids=frozenset({"case-1", "case-2"}),
        ),
        allowed_effects={Effect.NONE, Effect.READ},
    )


def build_cases() -> list[VerificationCase]:
    """Build the complete finite case snapshot."""
    return [
        VerificationCase(
            id="case-1",
            input_data={"record": {"values": [1, 2]}},
            risk_level=RiskLevel.LOW,
        ),
        VerificationCase(
            id="case-2",
            input_data={"record": {"values": [3, 4]}},
            risk_level=RiskLevel.HIGH,
        ),
    ]


def build_bundle() -> TrustedVerificationBundle:
    """Build a trusted bundle from validated domain inputs."""
    return TrustedVerificationBundle(
        experiment_id=UUID("c5e151b4-fd4b-4e79-9a14-0c9849f6bd2a"),
        contract=build_contract(),
        cases=build_cases(),
        contract_version="2026.1",
        source_revision="a" * 40,
    )


def test_bundle_has_versioned_deterministic_fingerprint() -> None:
    """Equivalent inputs produce the same versioned canonical fingerprint."""
    bundle = build_bundle()
    repeated = build_bundle()

    assert bundle.bundle_version == "1"
    assert bundle.fingerprint_sha256 == repeated.fingerprint_sha256
    assert len(bundle.fingerprint_sha256) == 64
    payload = bundle.to_payload()
    fingerprint = payload.pop("fingerprint_sha256")
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert fingerprint == hashlib.sha256(canonical).hexdigest()
    bundle.verify_integrity()
    assert TrustedVerificationBundle.from_json(bundle.to_json()).to_json() == (
        bundle.to_json()
    )


def test_case_order_is_canonicalised_by_case_id() -> None:
    """Equivalent finite case sets have the same canonical bundle digest."""
    contract = build_contract()
    cases = build_cases()
    forward = TrustedVerificationBundle(
        experiment_id=UUID("c5e151b4-fd4b-4e79-9a14-0c9849f6bd2a"),
        contract=contract,
        cases=cases,
        contract_version="2026.1",
        source_revision="a" * 40,
    )
    reversed_input = TrustedVerificationBundle(
        experiment_id=UUID("c5e151b4-fd4b-4e79-9a14-0c9849f6bd2a"),
        contract=build_contract(),
        cases=list(reversed(build_cases())),
        contract_version="2026.1",
        source_revision="a" * 40,
    )

    assert forward.fingerprint_sha256 == reversed_input.fingerprint_sha256
    assert [case.id for case in reversed_input.cases] == ["case-1", "case-2"]


def test_bundle_is_deeply_immutable_after_construction() -> None:
    """Mutating inputs and returned models cannot alter the stored snapshot."""
    contract = build_contract()
    cases = build_cases()
    bundle = TrustedVerificationBundle(
        experiment_id=UUID("c5e151b4-fd4b-4e79-9a14-0c9849f6bd2a"),
        contract=contract,
        cases=cases,
        contract_version="2026.1",
        source_revision="a" * 40,
    )
    original_json = bundle.to_json()

    contract.invariants.append(
        Invariant(
            id="new-invariant",
            path="result",
            operator=InvariantOperator.EXISTS,
        )
    )
    contract.allowed_effects.add(Effect.NETWORK)
    assert contract.exact_verification_scope is not None
    contract.exact_verification_scope.exhaustive_case_ids = frozenset({"changed"})
    original_record = cases[0].input_data["record"]
    assert isinstance(original_record, dict)
    original_values = original_record["values"]
    assert isinstance(original_values, list)
    original_values.append(99)

    returned_contract = bundle.contract
    returned_contract.invariants.clear()
    returned_contract.allowed_effects.add(Effect.WRITE)
    returned_case = bundle.cases[0]
    returned_record = returned_case.input_data["record"]
    assert isinstance(returned_record, dict)
    returned_values = returned_record["values"]
    assert isinstance(returned_values, list)
    returned_values.clear()

    exposed_payload = bundle.to_payload()
    exposed_payload["source_revision"] = "b" * 40

    assert bundle.to_json() == original_json
    bundle.verify_integrity()
    assert [case.id for case in bundle.cases] == ["case-1", "case-2"]


def test_bundle_rejects_top_level_field_reassignment() -> None:
    """Version and identity fields cannot be reassigned after construction."""
    bundle = build_bundle()

    with pytest.raises(FrozenInstanceError):
        setattr(bundle, "contract_version", "changed")  # noqa: B010


def test_integrity_check_rejects_in_memory_fingerprint_tampering() -> None:
    """Integrity verification detects deliberate low-level digest mutation."""
    bundle = build_bundle()
    tampered_fingerprint = (
        "0" * 64 if bundle.fingerprint_sha256 != "0" * 64 else "1" * 64
    )
    object.__setattr__(bundle, "fingerprint_sha256", tampered_fingerprint)

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        bundle.verify_integrity()


def test_bundle_rejects_duplicate_case_ids() -> None:
    """A finite snapshot cannot contain ambiguous duplicate case identities."""
    cases = build_cases()

    with pytest.raises(ValueError, match="case IDs must be unique"):
        TrustedVerificationBundle(
            experiment_id=UUID("c5e151b4-fd4b-4e79-9a14-0c9849f6bd2a"),
            contract=build_contract(),
            cases=[cases[0], cases[0]],
            contract_version="2026.1",
            source_revision="a" * 40,
        )


def test_bundle_rejects_missing_declared_exact_scope_cases() -> None:
    """Every exhaustive case declared by the contract must be snapshotted."""
    with pytest.raises(ValueError, match="omits declared exhaustive cases"):
        TrustedVerificationBundle(
            experiment_id=UUID("c5e151b4-fd4b-4e79-9a14-0c9849f6bd2a"),
            contract=build_contract(),
            cases=build_cases()[:1],
            contract_version="2026.1",
            source_revision="a" * 40,
        )


def test_bundle_rejects_unbounded_iterable_case_input() -> None:
    """A generator is not accepted in place of a finite sequence snapshot."""
    with pytest.raises(TypeError, match="finite sequence"):
        TrustedVerificationBundle(
            experiment_id=UUID("c5e151b4-fd4b-4e79-9a14-0c9849f6bd2a"),
            contract=build_contract(),
            cases=cast(
                Sequence[VerificationCase],
                (case for case in build_cases()),
            ),
            contract_version="2026.1",
            source_revision="a" * 40,
        )


def test_bundle_rejects_empty_case_snapshot() -> None:
    """A trusted bundle cannot omit the verification case snapshot."""
    with pytest.raises(ValueError, match="At least one verification case"):
        TrustedVerificationBundle(
            experiment_id=UUID("c5e151b4-fd4b-4e79-9a14-0c9849f6bd2a"),
            contract=build_contract(),
            cases=[],
            contract_version="2026.1",
            source_revision="a" * 40,
        )


def test_bundle_revalidates_mutated_contract_before_snapshotting() -> None:
    """A previously validated but subsequently invalid contract fails closed."""
    contract = build_contract()
    contract.candidate_implementation_id = contract.reference_implementation_id

    with pytest.raises(ValueError, match="Invalid transformation contract"):
        TrustedVerificationBundle(
            experiment_id=UUID("c5e151b4-fd4b-4e79-9a14-0c9849f6bd2a"),
            contract=contract,
            cases=build_cases(),
            contract_version="2026.1",
            source_revision="a" * 40,
        )


def test_bundle_rejects_missing_or_mismatched_payload_metadata() -> None:
    """Serialized bundles require every identity and version field to match."""
    missing_metadata = build_bundle().to_payload()
    del missing_metadata["contract_version"]

    with pytest.raises(ValueError, match="missing or unsupported fields"):
        TrustedVerificationBundle.from_payload(missing_metadata)

    missing_experiment_id = build_bundle().to_payload()
    del missing_experiment_id["experiment_id"]

    with pytest.raises(ValueError, match="missing or unsupported fields"):
        TrustedVerificationBundle.from_payload(missing_experiment_id)

    unsupported_version = build_bundle().to_payload()
    unsupported_version["bundle_version"] = "2"

    with pytest.raises(ValueError, match="Unsupported trusted verification bundle"):
        TrustedVerificationBundle.from_payload(unsupported_version)

    malformed_revision = build_bundle().to_payload()
    malformed_revision["source_revision"] = "not-a-revision"

    with pytest.raises(ValueError, match="source_revision"):
        TrustedVerificationBundle.from_payload(malformed_revision)


def test_bundle_rejects_fingerprint_and_content_tampering() -> None:
    """Changes to the digest or covered metadata are detected on import."""
    changed_fingerprint = build_bundle().to_payload()
    original_fingerprint = changed_fingerprint["fingerprint_sha256"]
    changed_fingerprint["fingerprint_sha256"] = (
        "0" * 64 if original_fingerprint != "0" * 64 else "1" * 64
    )

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        TrustedVerificationBundle.from_payload(changed_fingerprint)

    changed_content = build_bundle().to_payload()
    changed_content["source_revision"] = "b" * 40

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        TrustedVerificationBundle.from_payload(changed_content)

    changed_case = build_bundle().to_payload()
    case_payloads = changed_case["cases"]
    assert isinstance(case_payloads, list)
    first_case = case_payloads[0]
    assert isinstance(first_case, dict)
    input_data = first_case["input_data"]
    assert isinstance(input_data, dict)
    record = input_data["record"]
    assert isinstance(record, dict)
    values = record["values"]
    assert isinstance(values, list)
    values[0] = 999

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        TrustedVerificationBundle.from_payload(changed_case)


def test_bundle_rejects_unknown_contract_fields() -> None:
    """Ignored Pydantic extras cannot be smuggled through bundle import."""
    payload = build_bundle().to_payload()
    contract_payload = payload["contract"]
    assert isinstance(contract_payload, dict)
    contract_payload["unexpected"] = "not part of the contract schema"

    with pytest.raises(ValueError, match="not canonical"):
        TrustedVerificationBundle.from_payload(payload)


def test_bundle_import_rejects_noncanonical_coerced_field_types() -> None:
    """Canonical import distinguishes integers from booleans after validation."""
    contract = build_contract()
    contract.risk.max_output_mismatches = 1
    bundle = TrustedVerificationBundle(
        experiment_id=UUID("c5e151b4-fd4b-4e79-9a14-0c9849f6bd2a"),
        contract=contract,
        cases=build_cases(),
        contract_version="2026.1",
        source_revision="a" * 40,
    )
    payload = bundle.to_payload()
    contract_payload = payload["contract"]
    assert isinstance(contract_payload, dict)
    risk_payload = contract_payload["risk"]
    assert isinstance(risk_payload, dict)
    risk_payload["max_output_mismatches"] = True

    with pytest.raises(ValueError, match="not canonical"):
        TrustedVerificationBundle.from_payload(payload)


def test_bundle_json_rejects_duplicate_object_keys() -> None:
    """Ambiguous JSON object keys are rejected rather than silently collapsed."""
    with pytest.raises(ValueError, match="Invalid trusted verification bundle JSON"):
        TrustedVerificationBundle.from_json(
            '{"bundle_version":"1","bundle_version":"1"}',
        )


def test_bundle_json_is_canonical() -> None:
    """The transport form uses sorted keys and compact separators."""
    bundle = build_bundle()
    payload = json.loads(bundle.to_json())

    assert bundle.to_json() == json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
