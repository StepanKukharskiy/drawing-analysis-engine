"""Discipline-neutral takeoff records, reconciliation, and approval.

Adapters preserve their native evidence and publish a small common record set.
Calculated and declared lines remain independent until an explicit, mutually
unique scope-match certificate closes.  No zero, match, discrepancy, reviewed
value, or approval is synthesized from an empty channel.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime
from hashlib import sha256
import json
import math
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "0.1.0"
LAYER = "shared_takeoff_intelligence_catalog"
ADAPTER_LAYER = "takeoff_intelligence_adapter"
DISCIPLINES = {"concrete", "rebar", "mep"}
EPISTEMIC_STATES = {
    "direct",
    "observed",
    "derived",
    "inferred",
    "convention_dependent",
    "unknown",
    "calculated",
    "declared",
    "reviewed",
    "approved",
}
RECORD_CONTRACTS = {
    "takeoff_occurrence": {
        "record_version": SCHEMA_VERSION,
        "required_fields": [
            "id", "discipline", "document_key", "item_type", "item_subtype",
            "description", "source", "native_record_refs", "evidence_refs",
            "epistemic_state", "unresolved_reasons",
        ],
    },
    "takeoff_physical_item": {
        "record_version": SCHEMA_VERSION,
        "required_fields": [
            "id", "discipline", "document_key", "occurrence_refs",
            "native_record_refs", "evidence_refs", "epistemic_state",
        ],
    },
    "takeoff_calculated_line": {
        "record_version": SCHEMA_VERSION,
        "required_fields": [
            "id", "discipline", "document_key", "physical_item_ref",
            "item_type", "metric_kind", "value", "unit", "scope_ref",
            "native_record_refs", "evidence_refs", "epistemic_state",
        ],
    },
    "takeoff_declared_line": {
        "record_version": SCHEMA_VERSION,
        "required_fields": [
            "id", "discipline", "document_key", "occurrence_ref", "item_type",
            "metric_kind", "value", "unit", "scope_ref", "native_record_refs",
            "evidence_refs", "epistemic_state",
        ],
    },
    "takeoff_comparison": {
        "record_version": SCHEMA_VERSION,
        "required_fields": [
            "id", "discipline", "document_key", "calculated_line_ref",
            "declared_line_ref", "match_certificate_ref", "metric_kind", "unit",
            "calculated_value", "declared_value", "delta", "absolute_tolerance",
            "status", "severity", "evidence_refs",
        ],
    },
    "takeoff_approval": {
        "record_version": SCHEMA_VERSION,
        "required_fields": [
            "id", "document_key", "calculated_line_ref", "comparison_ref",
            "value", "unit", "engineer", "approved_at", "reason",
            "evidence_refs", "epistemic_state",
        ],
    },
}


def canonical_sha256(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def stable_id(kind: str, *parts: object) -> str:
    return f"{kind}.{canonical_sha256(parts)[:20]}"


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _record_errors(
    row: Mapping[str, Any],
    record_type: str,
    *,
    allow_zero: bool = True,
) -> list[str]:
    identifier = str(row.get("id") or f"missing-{record_type}-id")
    errors = []
    contract = RECORD_CONTRACTS[record_type]
    if row.get("record_type") != record_type:
        errors.append(f"{identifier}: record_type mismatch")
    if row.get("record_version") != contract["record_version"]:
        errors.append(f"{identifier}: record_version mismatch")
    for key in contract["required_fields"]:
        if key not in row:
            errors.append(f"{identifier}: missing {key}")
    if row.get("discipline") not in DISCIPLINES:
        errors.append(f"{identifier}: unsupported discipline")
    if not isinstance(row.get("document_key"), str) or not row.get("document_key"):
        errors.append(f"{identifier}: document_key is required")
    if row.get("epistemic_state") not in EPISTEMIC_STATES:
        errors.append(f"{identifier}: invalid epistemic_state")
    if not isinstance(row.get("evidence_refs"), list) or not row.get("evidence_refs"):
        errors.append(f"{identifier}: evidence_refs are required")
    if len(row.get("evidence_refs", [])) != len(set(row.get("evidence_refs", []))):
        errors.append(f"{identifier}: evidence_refs contain duplicates")
    if record_type in {"takeoff_calculated_line", "takeoff_declared_line"}:
        if not _finite(row.get("value")) or (not allow_zero and float(row["value"]) <= 0):
            errors.append(f"{identifier}: finite value is required")
        if not isinstance(row.get("unit"), str) or not row.get("unit"):
            errors.append(f"{identifier}: unit is required")
        if not isinstance(row.get("scope_ref"), str) or not row.get("scope_ref"):
            errors.append(f"{identifier}: explicit scope_ref is required")
    return errors


def validate_takeoff_adapter(payload: Mapping[str, Any]) -> list[str]:
    errors = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("adapter schema_version mismatch")
    if payload.get("layer") != ADAPTER_LAYER:
        errors.append("adapter layer mismatch")
    adapter = payload.get("adapter", {})
    if adapter.get("discipline") not in DISCIPLINES:
        errors.append("adapter discipline mismatch")
    if not isinstance(adapter.get("name"), str) or not adapter.get("name"):
        errors.append("adapter name is required")
    digest = payload.get("native_payload_ref", {}).get("payload_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        errors.append("native_payload_ref.payload_sha256 must be a SHA-256 digest")
    if payload.get("native_records_preserved_unchanged") is not True:
        errors.append("native records must be preserved unchanged")
    if "native_records" not in payload:
        errors.append("native_records are required")
    groups = {
        "occurrences": "takeoff_occurrence",
        "physical_items": "takeoff_physical_item",
        "calculated_lines": "takeoff_calculated_line",
        "declared_lines": "takeoff_declared_line",
    }
    all_ids = []
    for key, record_type in groups.items():
        rows = list(payload.get(key, []))
        all_ids.extend(str(row.get("id")) for row in rows)
        for row in rows:
            errors.extend(_record_errors(row, record_type))
            if row.get("discipline") != adapter.get("discipline"):
                errors.append(f"{row.get('id')}: cross-discipline adapter record")
    if len(all_ids) != len(set(all_ids)):
        errors.append("adapter record IDs must be unique")
    occurrence_ids = {str(row.get("id")) for row in payload.get("occurrences", [])}
    physical_ids = {str(row.get("id")) for row in payload.get("physical_items", [])}
    for row in payload.get("physical_items", []):
        refs = list(map(str, row.get("occurrence_refs", [])))
        if not refs or any(ref not in occurrence_ids for ref in refs):
            errors.append(f"{row.get('id')}: physical item occurrence membership is invalid")
    for row in payload.get("calculated_lines", []):
        if str(row.get("physical_item_ref")) not in physical_ids:
            errors.append(f"{row.get('id')}: calculated line lacks a physical item")
        if row.get("epistemic_state") != "calculated":
            errors.append(f"{row.get('id')}: calculated line state mismatch")
    for row in payload.get("declared_lines", []):
        occurrence_ref = row.get("occurrence_ref")
        if occurrence_ref is not None and str(occurrence_ref) not in occurrence_ids:
            errors.append(f"{row.get('id')}: declared occurrence ref is invalid")
        if row.get("epistemic_state") != "declared":
            errors.append(f"{row.get('id')}: declared line state mismatch")
    contract = payload.get("contract", {})
    for key in (
        "calculated_and_declared_channels_separate",
        "missing_values_not_substituted_with_zero",
        "approval_not_inferred",
    ):
        if contract.get(key) is not True:
            errors.append(f"adapter contract.{key} must be true")
    return errors


def _match_certificates(
    calculated: Sequence[Mapping[str, Any]],
    declared: Sequence[Mapping[str, Any]],
    certificates: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, tuple[str, Mapping[str, Any]]], list[str]]:
    errors = []
    calculated_by_id = {str(row["id"]): row for row in calculated}
    declared_by_id = {str(row["id"]): row for row in declared}
    result: dict[str, tuple[str, Mapping[str, Any]]] = {}
    used_declared = set()
    for certificate in certificates:
        identifier = str(certificate.get("id") or "missing-match-certificate-id")
        calculated_ref = str(certificate.get("calculated_line_ref") or "")
        declared_ref = str(certificate.get("declared_line_ref") or "")
        calc = calculated_by_id.get(calculated_ref)
        decl = declared_by_id.get(declared_ref)
        if certificate.get("state") != "accepted":
            errors.append(f"{identifier}: match certificate is not accepted")
            continue
        if calc is None or decl is None:
            errors.append(f"{identifier}: match certificate has unknown line refs")
            continue
        if calculated_ref in result or declared_ref in used_declared:
            errors.append(f"{identifier}: line pairing is not mutually unique")
            continue
        comparable = all(
            calc.get(key) == decl.get(key)
            for key in ("discipline", "document_key", "item_type", "metric_kind", "unit")
        )
        if not comparable:
            errors.append(f"{identifier}: matched lines are incompatible")
            continue
        evidence_refs = list(certificate.get("evidence_refs", []))
        if not evidence_refs or len(evidence_refs) != len(set(evidence_refs)):
            errors.append(f"{identifier}: match evidence is missing or duplicated")
            continue
        result[calculated_ref] = (declared_ref, certificate)
        used_declared.add(declared_ref)
    return result, errors


def build_takeoff_comparisons(
    *,
    calculated_lines: Sequence[Mapping[str, Any]],
    declared_lines: Sequence[Mapping[str, Any]],
    match_certificates: Iterable[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare only explicitly scope-matched lines; preserve unmatched sides."""

    matches, errors = _match_certificates(
        calculated_lines, declared_lines, match_certificates
    )
    if errors:
        raise ValueError("invalid takeoff match certificates:\n" + "\n".join(errors))
    declared_by_id = {str(row["id"]): row for row in declared_lines}
    used_declared = {declared_ref for declared_ref, _ in matches.values()}
    comparisons = []
    for calculated in calculated_lines:
        calculated_ref = str(calculated["id"])
        matched = matches.get(calculated_ref)
        if matched is None:
            comparisons.append({
                "record_type": "takeoff_comparison",
                "record_version": SCHEMA_VERSION,
                "id": stable_id("takeoff_comparison", calculated_ref, None),
                "discipline": calculated["discipline"],
                "document_key": calculated["document_key"],
                "calculated_line_ref": calculated_ref,
                "declared_line_ref": None,
                "match_certificate_ref": None,
                "metric_kind": calculated["metric_kind"],
                "unit": calculated["unit"],
                "calculated_value": calculated["value"],
                "declared_value": None,
                "delta": None,
                "absolute_tolerance": None,
                "status": "calculated_unmatched",
                "severity": "unknown",
                "evidence_refs": list(calculated["evidence_refs"]),
            })
            continue
        declared_ref, certificate = matched
        declared = declared_by_id[declared_ref]
        delta = float(calculated["value"]) - float(declared["value"])
        tolerance = float(certificate.get("absolute_tolerance") or 0.0)
        status = "match" if abs(delta) <= tolerance else "discrepancy"
        comparisons.append({
            "record_type": "takeoff_comparison",
            "record_version": SCHEMA_VERSION,
            "id": stable_id("takeoff_comparison", calculated_ref, declared_ref),
            "discipline": calculated["discipline"],
            "document_key": calculated["document_key"],
            "calculated_line_ref": calculated_ref,
            "declared_line_ref": declared_ref,
            "match_certificate_ref": str(certificate["id"]),
            "metric_kind": calculated["metric_kind"],
            "unit": calculated["unit"],
            "calculated_value": calculated["value"],
            "declared_value": declared["value"],
            "delta": round(delta, 9),
            "absolute_tolerance": tolerance,
            "status": status,
            "severity": "pass" if status == "match" else "review",
            "evidence_refs": sorted({
                *map(str, calculated["evidence_refs"]),
                *map(str, declared["evidence_refs"]),
                *map(str, certificate["evidence_refs"]),
            }),
        })
    for declared in declared_lines:
        declared_ref = str(declared["id"])
        if declared_ref in used_declared:
            continue
        comparisons.append({
            "record_type": "takeoff_comparison",
            "record_version": SCHEMA_VERSION,
            "id": stable_id("takeoff_comparison", None, declared_ref),
            "discipline": declared["discipline"],
            "document_key": declared["document_key"],
            "calculated_line_ref": None,
            "declared_line_ref": declared_ref,
            "match_certificate_ref": None,
            "metric_kind": declared["metric_kind"],
            "unit": declared["unit"],
            "calculated_value": None,
            "declared_value": declared["value"],
            "delta": None,
            "absolute_tolerance": None,
            "status": "declared_unmatched",
            "severity": "unknown",
            "evidence_refs": list(declared["evidence_refs"]),
        })
    comparisons.sort(key=lambda row: (row["document_key"], row["discipline"], row["id"]))
    activation = {
        "state": (
            "active_scoped_lines_available"
            if calculated_lines or declared_lines
            else "inactive_no_scoped_calculated_or_declared_lines"
        ),
        "calculated_line_count": len(calculated_lines),
        "declared_line_count": len(declared_lines),
        "comparison_count": len(comparisons),
        "empty_channels_do_not_create_comparisons": True,
    }
    return comparisons, activation


def build_takeoff_intelligence_catalog(
    *,
    adapters: Sequence[Mapping[str, Any]],
    match_certificates: Iterable[Mapping[str, Any]] = (),
    approval_records: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    errors = [
        f"adapter[{index}]: {error}"
        for index, adapter in enumerate(adapters)
        for error in validate_takeoff_adapter(adapter)
    ]
    if errors:
        raise ValueError("invalid takeoff adapter payload:\n" + "\n".join(errors))
    occurrences = [deepcopy(row) for adapter in adapters for row in adapter.get("occurrences", [])]
    physical_items = [deepcopy(row) for adapter in adapters for row in adapter.get("physical_items", [])]
    calculated = [deepcopy(row) for adapter in adapters for row in adapter.get("calculated_lines", [])]
    declared = [deepcopy(row) for adapter in adapters for row in adapter.get("declared_lines", [])]
    certificates = [deepcopy(dict(row)) for row in match_certificates]
    comparisons, activation = build_takeoff_comparisons(
        calculated_lines=calculated,
        declared_lines=declared,
        match_certificates=certificates,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "layer": LAYER,
        "record_contracts": deepcopy(RECORD_CONTRACTS),
        "adapter_contract_refs": [
            {
                "adapter_name": adapter["adapter"]["name"],
                "discipline": adapter["adapter"]["discipline"],
                "payload_sha256": canonical_sha256(adapter),
                "native_payload_sha256": adapter["native_payload_ref"]["payload_sha256"],
            }
            for adapter in adapters
        ],
        "occurrences": occurrences,
        "physical_items": physical_items,
        "calculated_lines": calculated,
        "declared_lines": declared,
        "match_certificates": certificates,
        "comparisons": comparisons,
        "approval_records": [deepcopy(row) for row in approval_records],
        "activation": activation,
        "summary": {
            "discipline_counts": dict(sorted(Counter(
                row["discipline"] for row in occurrences
            ).items())),
            "occurrence_count": len(occurrences),
            "physical_item_count": len(physical_items),
            "calculated_line_count": len(calculated),
            "declared_line_count": len(declared),
            "comparison_status_counts": dict(sorted(Counter(
                row["status"] for row in comparisons
            ).items())),
            "approval_count": len(approval_records),
        },
        "contract": {
            "discipline_native_records_preserved": True,
            "calculated_declared_reviewed_and_approved_separate": True,
            "explicit_mutually_unique_scope_match_required": True,
            "empty_channels_do_not_create_comparisons": True,
            "missing_values_not_substituted_with_zero": True,
            "approval_not_inferred": True,
        },
    }
    validation = validate_takeoff_intelligence_catalog(payload)
    if validation:
        raise ValueError("invalid takeoff catalog:\n" + "\n".join(validation))
    return payload


def create_takeoff_approval(
    catalog: Mapping[str, Any],
    *,
    calculated_line_ref: str,
    engineer: str,
    approved_at: str,
    reason: str,
    evidence_refs: Sequence[str],
) -> dict[str, Any]:
    lines = {
        str(row.get("id")): row for row in catalog.get("calculated_lines", [])
    }
    line = lines.get(calculated_line_ref)
    if line is None:
        raise ValueError("approval requires an existing calculated line")
    if not engineer.strip() or not reason.strip() or not evidence_refs:
        raise ValueError("approval requires engineer, reason, and evidence")
    try:
        timestamp = datetime.fromisoformat(approved_at.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError("approval timestamp must be ISO-8601") from exc
    if timestamp.tzinfo is None:
        raise ValueError("approval timestamp must include a timezone")
    comparison = next(
        (
            row
            for row in catalog.get("comparisons", [])
            if row.get("calculated_line_ref") == calculated_line_ref
        ),
        None,
    )
    return {
        "record_type": "takeoff_approval",
        "record_version": SCHEMA_VERSION,
        "id": stable_id(
            "takeoff_approval", calculated_line_ref, engineer, approved_at
        ),
        "document_key": line["document_key"],
        "calculated_line_ref": calculated_line_ref,
        "comparison_ref": comparison.get("id") if comparison else None,
        "value": line["value"],
        "unit": line["unit"],
        "engineer": engineer.strip(),
        "approved_at": approved_at,
        "reason": reason.strip(),
        "evidence_refs": sorted(set(map(str, evidence_refs))),
        "epistemic_state": "approved",
    }


def validate_takeoff_intelligence_catalog(payload: Mapping[str, Any]) -> list[str]:
    errors = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("catalog schema_version mismatch")
    if payload.get("layer") != LAYER:
        errors.append("catalog layer mismatch")
    if payload.get("record_contracts") != RECORD_CONTRACTS:
        errors.append("record contracts mismatch")
    groups = {
        "occurrences": "takeoff_occurrence",
        "physical_items": "takeoff_physical_item",
        "calculated_lines": "takeoff_calculated_line",
        "declared_lines": "takeoff_declared_line",
    }
    all_ids = []
    for key, record_type in groups.items():
        for row in payload.get(key, []):
            errors.extend(_record_errors(row, record_type))
            all_ids.append(str(row.get("id")))
    comparisons = list(payload.get("comparisons", []))
    approvals = list(payload.get("approval_records", []))
    all_ids.extend(str(row.get("id")) for row in [*comparisons, *approvals])
    if len(all_ids) != len(set(all_ids)):
        errors.append("catalog record IDs must be globally unique")
    occurrence_ids = {str(row.get("id")) for row in payload.get("occurrences", [])}
    physical_ids = {str(row.get("id")) for row in payload.get("physical_items", [])}
    calculated_by_id = {str(row.get("id")): row for row in payload.get("calculated_lines", [])}
    declared_by_id = {
        str(row.get("id")): row for row in payload.get("declared_lines", [])
    }
    declared_ids = set(declared_by_id)
    comparison_ids = {str(row.get("id")) for row in comparisons}
    certificates = list(payload.get("match_certificates", []))
    certificate_ids = {str(row.get("id")) for row in certificates}
    certificate_by_id = {str(row.get("id")): row for row in certificates}
    if len(certificate_ids) != len(certificates):
        errors.append("match certificate IDs must be unique")
    certificate_pairs = [
        (str(row.get("calculated_line_ref")), str(row.get("declared_line_ref")))
        for row in certificates
    ]
    if (
        len({pair[0] for pair in certificate_pairs}) != len(certificate_pairs)
        or len({pair[1] for pair in certificate_pairs}) != len(certificate_pairs)
    ):
        errors.append("match certificate pairing must be mutually unique")
    for row in certificates:
        identifier = str(row.get("id"))
        if row.get("state") != "accepted":
            errors.append(f"{identifier}: match certificate must be accepted")
        if str(row.get("calculated_line_ref")) not in calculated_by_id:
            errors.append(f"{identifier}: unknown calculated line")
        if str(row.get("declared_line_ref")) not in declared_ids:
            errors.append(f"{identifier}: unknown declared line")
        tolerance = row.get("absolute_tolerance", 0.0)
        if not _finite(tolerance) or float(tolerance) < 0:
            errors.append(f"{identifier}: invalid absolute tolerance")
        if not row.get("evidence_refs"):
            errors.append(f"{identifier}: match evidence is required")
    for row in payload.get("physical_items", []):
        if not row.get("occurrence_refs") or any(
            str(ref) not in occurrence_ids for ref in row.get("occurrence_refs", [])
        ):
            errors.append(f"{row.get('id')}: invalid occurrence membership")
    for row in payload.get("calculated_lines", []):
        if str(row.get("physical_item_ref")) not in physical_ids:
            errors.append(f"{row.get('id')}: calculated line lacks physical item")
        if row.get("epistemic_state") != "calculated":
            errors.append(f"{row.get('id')}: calculated state mismatch")
    for row in comparisons:
        identifier = str(row.get("id"))
        if row.get("record_type") != "takeoff_comparison" or row.get(
            "record_version"
        ) != SCHEMA_VERSION:
            errors.append(f"{identifier}: comparison contract mismatch")
        calculated_ref = row.get("calculated_line_ref")
        declared_ref = row.get("declared_line_ref")
        status = row.get("status")
        if calculated_ref is not None and str(calculated_ref) not in calculated_by_id:
            errors.append(f"{identifier}: unknown calculated line")
        if declared_ref is not None and str(declared_ref) not in declared_ids:
            errors.append(f"{identifier}: unknown declared line")
        expected_presence = {
            "calculated_unmatched": (True, False),
            "declared_unmatched": (False, True),
            "match": (True, True),
            "discrepancy": (True, True),
        }.get(status)
        if expected_presence is None or expected_presence != (
            calculated_ref is not None,
            declared_ref is not None,
        ):
            errors.append(f"{identifier}: comparison side/status mismatch")
        calculated = calculated_by_id.get(str(calculated_ref))
        declared = declared_by_id.get(str(declared_ref))
        source_line = calculated or declared
        if source_line is not None:
            for key in ("discipline", "document_key", "metric_kind", "unit"):
                if row.get(key) != source_line.get(key):
                    errors.append(f"{identifier}: comparison {key} mismatch")
            if not row.get("evidence_refs"):
                errors.append(f"{identifier}: comparison evidence is required")
        if status in {"calculated_unmatched", "declared_unmatched"}:
            if (
                row.get("match_certificate_ref") is not None
                or row.get("delta") is not None
                or row.get("absolute_tolerance") is not None
            ):
                errors.append(f"{identifier}: unmatched comparison gained pairing data")
            if status == "calculated_unmatched" and (
                calculated is None
                or row.get("calculated_value") != calculated.get("value")
                or row.get("declared_value") is not None
            ):
                errors.append(f"{identifier}: calculated-unmatched value mismatch")
            if status == "declared_unmatched" and (
                declared is None
                or row.get("declared_value") != declared.get("value")
                or row.get("calculated_value") is not None
            ):
                errors.append(f"{identifier}: declared-unmatched value mismatch")
            if row.get("severity") != "unknown":
                errors.append(f"{identifier}: unmatched severity must be unknown")
        else:
            certificate_ref = str(row.get("match_certificate_ref") or "")
            if certificate_ref not in certificate_ids or not _finite(row.get("delta")):
                errors.append(f"{identifier}: matched comparison lacks certificate or delta")
            if not _finite(row.get("absolute_tolerance")) or float(
                row.get("absolute_tolerance")
            ) < 0:
                errors.append(f"{identifier}: matched tolerance is invalid")
            if calculated is not None and declared is not None:
                certificate = certificate_by_id.get(certificate_ref)
                if certificate is not None and (
                    str(certificate.get("calculated_line_ref")) != str(calculated_ref)
                    or str(certificate.get("declared_line_ref")) != str(declared_ref)
                    or not _finite(row.get("absolute_tolerance"))
                    or float(certificate.get("absolute_tolerance", 0.0))
                    != float(row.get("absolute_tolerance") or 0.0)
                ):
                    errors.append(f"{identifier}: comparison certificate mismatch")
                if (
                    row.get("calculated_value") != calculated.get("value")
                    or row.get("declared_value") != declared.get("value")
                ):
                    errors.append(f"{identifier}: matched values mismatch")
                delta = round(
                    float(calculated["value"]) - float(declared["value"]), 9
                )
                if row.get("delta") != delta:
                    errors.append(f"{identifier}: comparison delta mismatch")
                if _finite(row.get("absolute_tolerance")):
                    expected_status = (
                        "match"
                        if abs(delta) <= float(row["absolute_tolerance"])
                        else "discrepancy"
                    )
                    if status != expected_status:
                        errors.append(f"{identifier}: comparison status mismatch")
                    expected_severity = (
                        "pass" if expected_status == "match" else "review"
                    )
                    if row.get("severity") != expected_severity:
                        errors.append(f"{identifier}: comparison severity mismatch")
    compared_calculated = [
        str(row["calculated_line_ref"])
        for row in comparisons
        if row.get("calculated_line_ref") is not None
    ]
    compared_declared = [
        str(row["declared_line_ref"])
        for row in comparisons
        if row.get("declared_line_ref") is not None
    ]
    if sorted(compared_calculated) != sorted(calculated_by_id):
        errors.append("each calculated line must appear in exactly one comparison")
    if sorted(compared_declared) != sorted(declared_ids):
        errors.append("each declared line must appear in exactly one comparison")
    for row in approvals:
        identifier = str(row.get("id"))
        if row.get("record_type") != "takeoff_approval" or row.get(
            "record_version"
        ) != SCHEMA_VERSION:
            errors.append(f"{identifier}: approval contract mismatch")
        calculated = calculated_by_id.get(str(row.get("calculated_line_ref")))
        if calculated is None:
            errors.append(f"{identifier}: approval lacks calculated line")
        elif row.get("value") != calculated.get("value") or row.get("unit") != calculated.get("unit"):
            errors.append(f"{identifier}: approval changed calculated value")
        elif row.get("document_key") != calculated.get("document_key"):
            errors.append(f"{identifier}: approval document mismatch")
        comparison_ref = str(row.get("comparison_ref") or "")
        comparison = next(
            (item for item in comparisons if str(item.get("id")) == comparison_ref),
            None,
        )
        if (
            comparison_ref not in comparison_ids
            or comparison is None
            or str(comparison.get("calculated_line_ref"))
            != str(row.get("calculated_line_ref"))
        ):
            errors.append(f"{identifier}: approval comparison ref is invalid")
        if row.get("epistemic_state") != "approved":
            errors.append(f"{identifier}: approval state mismatch")
        if not isinstance(row.get("engineer"), str) or not row.get("engineer").strip():
            errors.append(f"{identifier}: approval engineer is required")
        if not isinstance(row.get("reason"), str) or not row.get("reason").strip():
            errors.append(f"{identifier}: approval reason is required")
        if not row.get("evidence_refs") or len(row.get("evidence_refs", [])) != len(
            set(row.get("evidence_refs", []))
        ):
            errors.append(f"{identifier}: approval evidence is required")
        try:
            approved_at = datetime.fromisoformat(
                str(row.get("approved_at")).replace("Z", "+00:00")
            )
        except ValueError:
            approved_at = None
        if approved_at is None or approved_at.tzinfo is None:
            errors.append(f"{identifier}: approval timestamp is invalid")
    calculated_count = len(calculated_by_id)
    declared_count = len(declared_ids)
    activation = payload.get("activation", {})
    expected_state = (
        "active_scoped_lines_available"
        if calculated_count or declared_count
        else "inactive_no_scoped_calculated_or_declared_lines"
    )
    if activation != {
        "state": expected_state,
        "calculated_line_count": calculated_count,
        "declared_line_count": declared_count,
        "comparison_count": len(comparisons),
        "empty_channels_do_not_create_comparisons": True,
    }:
        errors.append("activation summary mismatch")
    expected_summary = {
        "discipline_counts": dict(sorted(Counter(
            row.get("discipline") for row in payload.get("occurrences", [])
        ).items())),
        "occurrence_count": len(payload.get("occurrences", [])),
        "physical_item_count": len(payload.get("physical_items", [])),
        "calculated_line_count": calculated_count,
        "declared_line_count": declared_count,
        "comparison_status_counts": dict(sorted(Counter(
            row.get("status") for row in comparisons
        ).items())),
        "approval_count": len(approvals),
    }
    if payload.get("summary") != expected_summary:
        errors.append("catalog summary mismatch")
    adapter_refs = list(payload.get("adapter_contract_refs", []))
    if len(adapter_refs) != len({
        (row.get("discipline"), row.get("native_payload_sha256"))
        for row in adapter_refs
    }):
        errors.append("adapter contract refs must be unique")
    for row in adapter_refs:
        if row.get("discipline") not in DISCIPLINES:
            errors.append("adapter contract discipline mismatch")
        for key in ("payload_sha256", "native_payload_sha256"):
            if not isinstance(row.get(key), str) or len(row[key]) != 64:
                errors.append(f"adapter contract {key} must be a SHA-256 digest")
    contract = payload.get("contract", {})
    for key in (
        "discipline_native_records_preserved",
        "calculated_declared_reviewed_and_approved_separate",
        "explicit_mutually_unique_scope_match_required",
        "empty_channels_do_not_create_comparisons",
        "missing_values_not_substituted_with_zero",
        "approval_not_inferred",
    ):
        if contract.get(key) is not True:
            errors.append(f"catalog contract.{key} must be true")
    return errors
