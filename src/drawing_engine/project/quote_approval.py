"""Immutable engineer approvals and approved-only quote summaries.

The estimate comparison remains the source of eligible drawing-calculated
values.  This module records approvals in a separate, content-addressed
overlay; it never writes approval state back into the comparison artifact.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "0.1.0"
LAYER = "quote_approval_overlay"
QUANTITY_UNITS = {
    "concrete_volume_m3": "m3",
    "reinforcement_centerline_m": "m",
    "reinforcement_fabrication_length_m": "m",
    "reinforcement_mass_kg": "kg",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DECISION_INPUT_KEYS = {
    "element_id",
    "quantity_field",
    "value",
    "unit",
    "reason",
    "drawing_evidence",
}
_DECISION_KEYS = _DECISION_INPUT_KEYS | {"engineer", "approved_at"}
_ROOT_KEYS = {
    "schema_version",
    "layer",
    "estimate_comparison_sha256",
    "estimate_comparison_byte_count",
    "frozen_engineering_graph_sha256",
    "decisions",
    "contract",
    "record_sha256",
}
_CONTRACT = {
    "comparison_artifact_mutated": False,
    "engineering_graph_mutated": False,
    "declared_values_used": False,
    "arbitrary_quote_values_allowed": False,
    "missing_values_substituted_with_zero": False,
    "only_explicit_approvals_are_exported": True,
}


Artifact = bytes | bytearray | Path


def _artifact_bytes(artifact: Artifact) -> bytes:
    if isinstance(artifact, Path):
        return artifact.read_bytes()
    if isinstance(artifact, (bytes, bytearray)):
        return bytes(artifact)
    raise TypeError("estimate comparison must be supplied as raw bytes or a Path")


def _comparison(artifact: Artifact) -> tuple[bytes, dict[str, Any]]:
    raw = _artifact_bytes(artifact)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("estimate comparison must be valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("estimate comparison must be a JSON object")
    return raw, payload


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _record_sha256(record: Mapping[str, Any]) -> str:
    content = {key: deepcopy(value) for key, value in record.items() if key != "record_sha256"}
    return sha256(_canonical_json(content)).hexdigest()


def estimate_comparison_sha256(artifact: Artifact) -> str:
    """Hash the exact serialized comparison bytes, including whitespace."""

    return sha256(_artifact_bytes(artifact)).hexdigest()


def _valid_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _calculated_by_element(comparison: Mapping[str, Any], errors: list[str]) -> dict[str, Mapping[str, Any]]:
    pages = comparison.get("pages")
    if not isinstance(pages, list):
        errors.append("estimate comparison pages must be a list")
        return {}
    calculated_by_element: dict[str, Mapping[str, Any]] = {}
    for index, page in enumerate(pages):
        if not isinstance(page, Mapping):
            errors.append(f"estimate comparison pages[{index}] must be an object")
            continue
        element_id = page.get("element_id")
        calculated = page.get("calculated_from_drawing")
        if not isinstance(element_id, str) or not element_id:
            errors.append(f"estimate comparison pages[{index}].element_id is required")
            continue
        if element_id in calculated_by_element:
            errors.append(f"estimate comparison contains duplicate element_id: {element_id}")
            continue
        if not isinstance(calculated, Mapping):
            errors.append(f"estimate comparison element {element_id} has no calculated_from_drawing object")
            continue
        if calculated.get("element_id") != element_id:
            errors.append(f"estimate comparison element {element_id} has a mismatched calculated element_id")
            continue
        calculated_by_element[element_id] = calculated
    return calculated_by_element


def create_quote_approval_record(
    estimate_comparison: Artifact,
    *,
    engineer: str,
    decisions: Iterable[Mapping[str, Any]],
    approved_at: str | None = None,
) -> dict[str, Any]:
    """Create a validated approval overlay for exact drawing-derived values.

    Each decision input must contain exactly ``element_id``, ``quantity_field``,
    ``value``, ``unit``, ``reason``, and ``drawing_evidence``.  Engineer and
    timestamp are injected into every decision so each exported row is
    independently auditable.
    """

    if not isinstance(engineer, str) or not engineer.strip():
        raise ValueError("engineer is required")
    timestamp = approved_at or datetime.now().astimezone().isoformat(timespec="seconds")
    raw, comparison = _comparison(estimate_comparison)
    frozen_hash = (comparison.get("frozen_calculation") or {}).get("sha256")
    rows = []
    for index, decision in enumerate(decisions):
        if not isinstance(decision, Mapping):
            raise ValueError(f"decisions[{index}] must be an object")
        if set(decision) != _DECISION_INPUT_KEYS:
            raise ValueError(
                f"decisions[{index}] must contain exactly {sorted(_DECISION_INPUT_KEYS)}"
            )
        row = deepcopy(dict(decision))
        row["engineer"] = engineer.strip()
        row["approved_at"] = timestamp
        rows.append(row)
    record = {
        "schema_version": SCHEMA_VERSION,
        "layer": LAYER,
        "estimate_comparison_sha256": sha256(raw).hexdigest(),
        "estimate_comparison_byte_count": len(raw),
        "frozen_engineering_graph_sha256": frozen_hash,
        "decisions": rows,
        "contract": deepcopy(_CONTRACT),
    }
    record["record_sha256"] = _record_sha256(record)
    validation = validate_quote_approval_record(raw, record)
    if validation["status"] != "pass":
        raise ValueError("invalid quote approval record: " + "; ".join(validation["errors"]))
    return record


def validate_quote_approval_record(
    estimate_comparison: Artifact,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate artifact hashes, exact values, units, evidence, and uniqueness."""

    errors: list[str] = []
    raw, comparison = _comparison(estimate_comparison)
    actual_comparison_hash = sha256(raw).hexdigest()
    frozen_hash = (comparison.get("frozen_calculation") or {}).get("sha256")

    if not isinstance(record, Mapping):
        return {"status": "fail", "errors": ["approval record must be an object"]}
    unknown_root_keys = set(record) - _ROOT_KEYS
    if unknown_root_keys:
        errors.append(f"unsupported approval fields: {sorted(unknown_root_keys)}")
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported approval schema_version")
    if record.get("layer") != LAYER:
        errors.append(f"approval layer must be {LAYER}")
    if record.get("estimate_comparison_sha256") != actual_comparison_hash:
        errors.append("approval estimate comparison hash does not match the exact artifact bytes")
    if record.get("estimate_comparison_byte_count") != len(raw):
        errors.append("approval estimate comparison byte count does not match the artifact")
    if not isinstance(frozen_hash, str) or not _SHA256_RE.fullmatch(frozen_hash):
        errors.append("estimate comparison has no valid frozen engineering graph SHA-256")
    if record.get("frozen_engineering_graph_sha256") != frozen_hash:
        errors.append("approval frozen engineering graph hash does not match the comparison")
    if record.get("contract") != _CONTRACT:
        errors.append("approval contract is missing or altered")
    try:
        expected_record_hash = _record_sha256(record)
    except (TypeError, ValueError):
        expected_record_hash = None
        errors.append("approval record contains non-canonical JSON values")
    if record.get("record_sha256") != expected_record_hash:
        errors.append("approval record SHA-256 does not match its content")

    calculated_by_element = _calculated_by_element(comparison, errors)
    decisions = record.get("decisions")
    if not isinstance(decisions, list):
        errors.append("decisions must be a list")
        decisions = []
    if not decisions:
        errors.append("at least one explicit approval decision is required")

    approved_targets: set[tuple[str, str]] = set()
    for index, decision in enumerate(decisions):
        prefix = f"decisions[{index}]"
        if not isinstance(decision, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        unknown_decision_keys = set(decision) - _DECISION_KEYS
        missing_decision_keys = _DECISION_KEYS - set(decision)
        if unknown_decision_keys:
            errors.append(f"{prefix} contains unsupported fields: {sorted(unknown_decision_keys)}")
        if missing_decision_keys:
            errors.append(f"{prefix} is missing fields: {sorted(missing_decision_keys)}")

        element_id = decision.get("element_id")
        quantity_field = decision.get("quantity_field")
        target = (str(element_id or ""), str(quantity_field or ""))
        if target in approved_targets:
            errors.append(f"duplicate approval decision for {target[0]}.{target[1]}")
        approved_targets.add(target)

        calculated = calculated_by_element.get(str(element_id))
        if calculated is None:
            errors.append(f"{prefix}.element_id is not in the estimate comparison")
        if quantity_field not in QUANTITY_UNITS:
            errors.append(f"{prefix}.quantity_field is not approvable in v0")
            continue
        expected_unit = QUANTITY_UNITS[quantity_field]
        if decision.get("unit") != expected_unit:
            errors.append(f"{prefix}.unit must be {expected_unit} for {quantity_field}")

        value = decision.get("value")
        if not _finite_number(value):
            errors.append(f"{prefix}.value must be a finite, non-null number")
        if calculated is not None:
            if quantity_field not in calculated:
                errors.append(f"{prefix}.quantity_field is absent from calculated_from_drawing")
            else:
                calculated_value = calculated.get(quantity_field)
                if not _finite_number(calculated_value):
                    errors.append(f"{prefix} cannot approve an unknown calculated_from_drawing value")
                elif not _finite_number(value) or value != calculated_value:
                    errors.append(
                        f"{prefix}.value must exactly equal calculated_from_drawing.{quantity_field}"
                    )

        if not isinstance(decision.get("engineer"), str) or not decision.get("engineer", "").strip():
            errors.append(f"{prefix}.engineer is required")
        if not _valid_timestamp(decision.get("approved_at")):
            errors.append(f"{prefix}.approved_at must be an ISO-8601 timestamp with timezone")
        if not isinstance(decision.get("reason"), str) or not decision.get("reason", "").strip():
            errors.append(f"{prefix}.reason is required")

        evidence = decision.get("drawing_evidence")
        if not isinstance(evidence, list) or not evidence or not all(
            isinstance(item, str) and item.strip() for item in evidence
        ):
            errors.append(f"{prefix}.drawing_evidence must be a non-empty list of strings")
        else:
            folded = [item.casefold() for item in evidence]
            if any("declared" in item or "schedule" in item for item in folded):
                errors.append(f"{prefix}.drawing_evidence cannot cite declared schedule evidence")
            if len(set(evidence)) != len(evidence):
                errors.append(f"{prefix}.drawing_evidence contains duplicates")

    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "decision_count": len(decisions),
        "estimate_comparison_sha256": actual_comparison_hash,
        "frozen_engineering_graph_sha256": frozen_hash,
        "declared_values_used": False,
    }


def build_quote_summary(
    estimate_comparison: Artifact,
    approval_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Export only explicitly approved values; never synthesize missing rows."""

    raw = _artifact_bytes(estimate_comparison)
    validation = validate_quote_approval_record(raw, approval_record)
    if validation["status"] != "pass":
        raise ValueError("invalid quote approval record: " + "; ".join(validation["errors"]))

    approved_values = [deepcopy(dict(item)) for item in approval_record["decisions"]]
    grouped: dict[tuple[str, str], list[float]] = {}
    for item in approved_values:
        key = (item["quantity_field"], item["unit"])
        grouped.setdefault(key, []).append(float(item["value"]))
    totals = [
        {
            "quantity_field": quantity_field,
            "value": math.fsum(values),
            "unit": unit,
            "approved_element_count": len(values),
        }
        for (quantity_field, unit), values in sorted(grouped.items())
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "approved_quote_summary",
        "estimate_comparison_sha256": sha256(raw).hexdigest(),
        "frozen_engineering_graph_sha256": approval_record["frozen_engineering_graph_sha256"],
        "approval_record_sha256": approval_record["record_sha256"],
        "approved_values": approved_values,
        "totals": totals,
        "contract": {
            "contains_only_explicit_approvals": True,
            "declared_values_exported": False,
            "unknowns_exported_as_zero": False,
        },
    }


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Atomically create a JSON artifact and refuse to replace an existing one."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    return path


def write_quote_approval_record(
    path: Path,
    estimate_comparison: Artifact,
    record: Mapping[str, Any],
) -> Path:
    """Validate and persist an immutable approval overlay without overwriting."""

    validation = validate_quote_approval_record(estimate_comparison, record)
    if validation["status"] != "pass":
        raise ValueError("invalid quote approval record: " + "; ".join(validation["errors"]))
    return _write_new_json(path, record)


def write_quote_summary(
    path: Path,
    estimate_comparison: Artifact,
    approval_record: Mapping[str, Any],
) -> Path:
    """Validate and persist an approved-only quote summary without overwriting."""

    return _write_new_json(path, build_quote_summary(estimate_comparison, approval_record))
