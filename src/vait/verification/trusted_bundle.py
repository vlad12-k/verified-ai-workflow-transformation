"""Immutable, versioned snapshots of trusted verification inputs."""

import hashlib
import hmac
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Self
from uuid import UUID

from pydantic import ValidationError

from vait.contracts.models import TransformationContract, VerificationCase

_SOURCE_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True, init=False)
class TrustedVerificationBundle:
    """Immutable canonical snapshot for a finite verification scope.

    Case order is canonicalised by case ID. The SHA-256 fingerprint detects
    changes to the canonical content; it does not authenticate the bundle's
    origin.
    """

    BUNDLE_VERSION: ClassVar[str] = "1"
    _FINGERPRINT_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "bundle_version",
            "experiment_id",
            "contract_version",
            "source_revision",
            "contract",
            "cases",
        }
    )
    _PAYLOAD_FIELDS: ClassVar[frozenset[str]] = _FINGERPRINT_FIELDS | {
        "fingerprint_sha256",
    }

    bundle_version: str
    experiment_id: UUID
    contract_version: str
    source_revision: str
    fingerprint_sha256: str
    _contract_snapshot_json: bytes = field(repr=False)
    _case_snapshot_jsons: tuple[bytes, ...] = field(repr=False)

    def __init__(
        self,
        *,
        experiment_id: UUID,
        contract: TransformationContract,
        cases: Sequence[VerificationCase],
        contract_version: str,
        source_revision: str,
        bundle_version: str = BUNDLE_VERSION,
    ) -> None:
        """Validate and snapshot domain inputs without retaining mutable refs."""
        _validate_metadata(
            experiment_id=experiment_id,
            contract_version=contract_version,
            source_revision=source_revision,
            bundle_version=bundle_version,
        )
        if not isinstance(contract, TransformationContract):
            raise TypeError("contract must be a TransformationContract")
        if isinstance(cases, (str, bytes, bytearray)) or not isinstance(
            cases,
            Sequence,
        ):
            raise TypeError("cases must be a finite sequence of VerificationCase")

        try:
            validated_contract = TransformationContract.model_validate(
                contract.model_dump(mode="python"),
            )
        except (ValidationError, TypeError, ValueError):
            raise ValueError("Invalid transformation contract snapshot") from None

        try:
            case_inputs = tuple(cases)
        except (TypeError, ValueError):
            raise ValueError("Invalid verification case sequence") from None
        if not case_inputs:
            raise ValueError("At least one verification case is required")

        validated_cases: list[VerificationCase] = []
        for case in case_inputs:
            if not isinstance(case, VerificationCase):
                raise TypeError("cases must contain VerificationCase values")
            try:
                validated_cases.append(
                    VerificationCase.model_validate(case.model_dump(mode="python")),
                )
            except (ValidationError, TypeError, ValueError):
                raise ValueError("Invalid verification case snapshot") from None

        ordered_cases = tuple(sorted(validated_cases, key=lambda case: case.id))
        _validate_case_scope(validated_contract, ordered_cases)

        contract_payload = _contract_payload(validated_contract)
        case_payloads = tuple(_case_payload(case) for case in ordered_cases)
        contract_json = _canonical_json_bytes(contract_payload)
        case_jsons = tuple(_canonical_json_bytes(payload) for payload in case_payloads)
        fingerprint_payload = _build_fingerprint_payload(
            bundle_version=bundle_version,
            experiment_id=experiment_id,
            contract_version=contract_version,
            source_revision=source_revision,
            contract=contract_payload,
            cases=case_payloads,
        )

        object.__setattr__(self, "bundle_version", bundle_version)
        object.__setattr__(self, "experiment_id", experiment_id)
        object.__setattr__(self, "contract_version", contract_version)
        object.__setattr__(self, "source_revision", source_revision)
        object.__setattr__(self, "fingerprint_sha256", _fingerprint(fingerprint_payload))
        object.__setattr__(self, "_contract_snapshot_json", contract_json)
        object.__setattr__(self, "_case_snapshot_jsons", case_jsons)

    @property
    def contract(self) -> TransformationContract:
        """Return a fresh validated contract copy of the stored snapshot."""
        self.verify_integrity()
        return TransformationContract.model_validate_json(self._contract_snapshot_json)

    @property
    def cases(self) -> tuple[VerificationCase, ...]:
        """Return fresh validated case copies in canonical ID order."""
        self.verify_integrity()
        return tuple(
            VerificationCase.model_validate_json(snapshot)
            for snapshot in self._case_snapshot_jsons
        )

    def verify_integrity(self) -> None:
        """Raise ValueError unless the immutable snapshot matches its digest."""
        try:
            _validate_metadata(
                experiment_id=self.experiment_id,
                contract_version=self.contract_version,
                source_revision=self.source_revision,
                bundle_version=self.bundle_version,
            )
            contract = TransformationContract.model_validate_json(
                self._contract_snapshot_json,
            )
            cases = tuple(
                VerificationCase.model_validate_json(snapshot)
                for snapshot in self._case_snapshot_jsons
            )
            _validate_case_scope(contract, cases)
            canonical_contract = _canonical_json_bytes(_contract_payload(contract))
            canonical_cases = tuple(
                _canonical_json_bytes(_case_payload(case))
                for case in cases
            )
            if (
                canonical_contract != self._contract_snapshot_json
                or canonical_cases != self._case_snapshot_jsons
            ):
                raise ValueError("Snapshot is not canonical")
            if not isinstance(self.fingerprint_sha256, str) or not _FINGERPRINT_PATTERN.fullmatch(
                self.fingerprint_sha256,
            ):
                raise ValueError("Fingerprint format is invalid")
            expected_fingerprint = _fingerprint(self._fingerprint_payload())
        except (ValidationError, TypeError, ValueError, UnicodeDecodeError):
            raise ValueError("Trusted verification bundle integrity check failed") from None

        if not hmac.compare_digest(expected_fingerprint, self.fingerprint_sha256):
            raise ValueError("Trusted verification bundle fingerprint mismatch")

    def to_payload(self) -> dict[str, object]:
        """Return a detached canonical mapping suitable for transport."""
        self.verify_integrity()
        payload = self._fingerprint_payload()
        payload["fingerprint_sha256"] = self.fingerprint_sha256
        return payload

    def to_json(self) -> str:
        """Serialise the complete bundle as canonical UTF-8-compatible JSON."""
        return _canonical_json_bytes(self.to_payload()).decode("utf-8")

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> Self:
        """Validate a complete serialized payload and its supplied fingerprint."""
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        if set(payload) != cls._PAYLOAD_FIELDS:
            raise ValueError("Bundle payload has missing or unsupported fields")

        bundle_version = payload["bundle_version"]
        experiment_id_text = payload["experiment_id"]
        contract_version = payload["contract_version"]
        source_revision = payload["source_revision"]
        contract_data = payload["contract"]
        cases_data = payload["cases"]
        supplied_fingerprint = payload["fingerprint_sha256"]

        if not isinstance(bundle_version, str):
            raise ValueError("Bundle version metadata is invalid")
        if not isinstance(experiment_id_text, str):
            raise ValueError("Experiment ID metadata is invalid")
        if not isinstance(contract_version, str):
            raise ValueError("Contract version metadata is invalid")
        if not isinstance(source_revision, str):
            raise ValueError("Source revision metadata is invalid")
        if not isinstance(contract_data, Mapping):
            raise ValueError("Contract snapshot is invalid")
        if not isinstance(cases_data, list) or not all(
            isinstance(case, Mapping) for case in cases_data
        ):
            raise ValueError("Verification case snapshot is invalid")
        if not isinstance(supplied_fingerprint, str) or not _FINGERPRINT_PATTERN.fullmatch(
            supplied_fingerprint,
        ):
            raise ValueError("Bundle fingerprint metadata is invalid")

        try:
            experiment_id = UUID(experiment_id_text)
        except ValueError:
            raise ValueError("Experiment ID metadata is invalid") from None
        if str(experiment_id) != experiment_id_text:
            raise ValueError("Experiment ID metadata is not canonical")

        try:
            contract = TransformationContract.model_validate(contract_data)
            cases = tuple(
                VerificationCase.model_validate(case)
                for case in cases_data
            )
        except (ValidationError, TypeError, ValueError):
            raise ValueError("Invalid contract or verification case payload") from None

        bundle = cls(
            experiment_id=experiment_id,
            contract=contract,
            cases=cases,
            contract_version=contract_version,
            source_revision=source_revision,
            bundle_version=bundle_version,
        )
        canonical_payload = bundle._fingerprint_payload()
        supplied_content = {
            key: payload[key]
            for key in cls._FINGERPRINT_FIELDS
        }
        try:
            canonical_bytes = _canonical_json_bytes(canonical_payload)
            supplied_bytes = _canonical_json_bytes(
                _plain_json_value(supplied_content),
            )
        except ValueError:
            raise ValueError("Bundle payload is not canonical") from None
        if canonical_bytes != supplied_bytes:
            raise ValueError("Bundle payload is not canonical")
        if not hmac.compare_digest(bundle.fingerprint_sha256, supplied_fingerprint):
            raise ValueError("Trusted verification bundle fingerprint mismatch")
        return bundle

    @classmethod
    def from_json(cls, document: str) -> Self:
        """Parse strict JSON and validate the serialized bundle."""
        if not isinstance(document, str):
            raise TypeError("document must be a string")
        try:
            payload = json.loads(
                document,
                object_pairs_hook=_reject_duplicate_object_keys,
            )
        except (json.JSONDecodeError, ValueError):
            raise ValueError("Invalid trusted verification bundle JSON") from None
        if not isinstance(payload, Mapping):
            raise ValueError("Bundle JSON must contain an object")
        return cls.from_payload(payload)

    def _fingerprint_payload(self) -> dict[str, object]:
        """Reconstruct the digest payload without exposing stored references."""
        return {
            "bundle_version": self.bundle_version,
            "experiment_id": str(self.experiment_id),
            "contract_version": self.contract_version,
            "source_revision": self.source_revision,
            "contract": _decode_json_object(self._contract_snapshot_json),
            "cases": [
                _decode_json_object(snapshot)
                for snapshot in self._case_snapshot_jsons
            ],
        }


def _validate_metadata(
    *,
    experiment_id: UUID,
    contract_version: str,
    source_revision: str,
    bundle_version: str,
) -> None:
    """Reject absent, malformed or unsupported identity metadata."""
    if not isinstance(experiment_id, UUID):
        raise TypeError("experiment_id must be a UUID")
    if not isinstance(contract_version, str) or not contract_version.strip():
        raise ValueError("contract_version is required")
    if not isinstance(source_revision, str) or not _SOURCE_REVISION_PATTERN.fullmatch(
        source_revision,
    ):
        raise ValueError("source_revision must be a 40-character Git revision")
    if bundle_version != TrustedVerificationBundle.BUNDLE_VERSION:
        raise ValueError("Unsupported trusted verification bundle version")


def _validate_case_scope(
    contract: TransformationContract,
    cases: Sequence[VerificationCase],
) -> None:
    """Require unique cases and complete coverage of any declared exact scope."""
    if not cases:
        raise ValueError("At least one verification case is required")
    case_ids = tuple(case.id for case in cases)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Verification case IDs must be unique")

    scope = contract.exact_verification_scope
    if scope is not None and not scope.exhaustive_case_ids.issubset(case_ids):
        raise ValueError("Verification case snapshot omits declared exhaustive cases")


def _contract_payload(contract: TransformationContract) -> dict[str, object]:
    """Return a JSON-ready contract with unordered fields normalised."""
    payload = contract.model_dump(mode="json")
    payload["allowed_effects"] = sorted(effect.value for effect in contract.allowed_effects)
    if contract.exact_verification_scope is not None:
        scope = payload.get("exact_verification_scope")
        if not isinstance(scope, dict):
            raise ValueError("Invalid exact verification scope snapshot")
        scope["exhaustive_case_ids"] = sorted(
            contract.exact_verification_scope.exhaustive_case_ids,
        )
    return payload


def _case_payload(case: VerificationCase) -> dict[str, object]:
    """Return a JSON-ready case snapshot."""
    return case.model_dump(mode="json")


def _build_fingerprint_payload(
    *,
    bundle_version: str,
    experiment_id: UUID,
    contract_version: str,
    source_revision: str,
    contract: dict[str, object],
    cases: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Build the exact versioned content covered by the digest."""
    return {
        "bundle_version": bundle_version,
        "experiment_id": str(experiment_id),
        "contract_version": contract_version,
        "source_revision": source_revision,
        "contract": contract,
        "cases": list(cases),
    }


def _canonical_json_bytes(payload: object) -> bytes:
    """Encode JSON deterministically, rejecting non-standard numeric values."""
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise ValueError("Bundle content cannot be canonically serialized") from None


def _plain_json_value(value: object) -> object:
    """Convert mappings to plain JSON values without coercing scalar types."""
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("JSON object keys must be strings")
            normalized[key] = _plain_json_value(item)
        return normalized
    if isinstance(value, list):
        return [_plain_json_value(item) for item in value]
    if type(value) in {str, int, float, bool, type(None)}:
        return value
    raise ValueError("Value is not a canonical JSON type")


def _decode_json_object(snapshot: bytes) -> dict[str, object]:
    """Decode an internal serialized model snapshot as an object."""
    value = json.loads(snapshot)
    if not isinstance(value, dict):
        raise ValueError("Bundle snapshot must be a JSON object")
    return value


def _fingerprint(payload: object) -> str:
    """Return SHA-256 over canonical payload bytes."""
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _reject_duplicate_object_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    """Reject ambiguous duplicate keys in serialized JSON objects."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result
