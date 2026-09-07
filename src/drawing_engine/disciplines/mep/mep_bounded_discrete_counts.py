"""M7B bounded discrete physical-identity and count certificates.

Only accepted M4 equipment, valve, fitting, and damper relations are eligible.
Each occurrence must have one M4 target. Duplicate projections and physical
identity require explicit, mutually unique evidence; missing or cross-sheet
ambiguous evidence abstains. Route length is outside this layer.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Mapping, Sequence

from src.drawing_engine.disciplines.mep.mep_attribute_binding import validate_mep_attribute_bindings
from src.drawing_engine.disciplines.mep.mep_occurrence_coverage_audit import (
    validate_mep_occurrence_coverage_audit,
)
from src.drawing_engine.disciplines.mep.mep_sheet_registry import validate_mep_sheet_registry


SCHEMA_VERSION = "0.1.0"
LAYER = "mep_bounded_discrete_counts"
DISCRETE_RELATION_TYPES = {
    "equipment_endpoint": "equipment",
    "valve": "valve",
    "fitting": "fitting",
    "damper": "damper",
}


def _sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _id(kind: str, *parts: object) -> str:
    return f"{kind}.{_sha256(parts)[:20]}"


def _semantic_candidate(relation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in relation.get("candidate", {}).items()
        if key not in {"raw_text", "terminology_entry_ref"}
    }


def _observed_occurrences(
    attribute_bindings: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output = []
    for relation in attribute_bindings.get("relations", []):
        relation_type = str(relation.get("relation_type"))
        if relation.get("state") != "accepted" or relation_type not in DISCRETE_RELATION_TYPES:
            continue
        relation_ref = str(relation.get("id"))
        output.append({
            "record_type": "mep_discrete_item_occurrence",
            "record_version": SCHEMA_VERSION,
            "id": _id("mep_discrete_item_occurrence", relation_ref),
            "category": DISCRETE_RELATION_TYPES[relation_type],
            "item_subtype": str(relation.get("candidate", {}).get("kind") or relation_type),
            "page_ref": str(relation.get("page_ref")),
            "m4_relation_ref": relation_ref,
            "m4_target_kind": relation.get("target_kind"),
            "m4_target_refs": sorted(str(ref) for ref in relation.get("target_refs", [])),
            "candidate": _semantic_candidate(relation),
            "counts": {
                "observed_occurrence_count": 1,
                "deduplicated_projected_count": None,
                "physical_instance_count": None,
                "calculated_count": None,
            },
            "epistemic_state": "derived",
            "evidence_refs": sorted({
                relation_ref,
                *[str(ref) for ref in relation.get("target_refs", [])],
                *[str(ref) for ref in relation.get("binding_evidence_refs", [])],
                *[str(ref) for ref in relation.get("proposal_evidence_refs", [])],
            }),
            "physical_identity_established": False,
            "quantity_eligible": False,
        })
    return sorted(output, key=lambda row: (row["page_ref"], row["category"], row["id"]))


def _abstention(
    occurrence_refs: Sequence[str], evidence_ref: str | None, reasons: Sequence[str]
) -> dict[str, Any]:
    return {
        "record_type": "mep_discrete_count_abstention",
        "record_version": SCHEMA_VERSION,
        "id": _id("mep_discrete_count_abstention", sorted(occurrence_refs), evidence_ref, sorted(reasons)),
        "occurrence_refs": sorted(occurrence_refs),
        "identity_evidence_ref": evidence_ref,
        "state": "abstained",
        "reasons": sorted(set(reasons)),
        "deduplicated_projected_count": None,
        "physical_instance_count": None,
        "calculated_count": None,
        "quantity_eligible": False,
    }


def build_mep_bounded_discrete_counts(
    *,
    sheet_registry: Mapping[str, Any],
    attribute_bindings: Mapping[str, Any],
    occurrence_coverage_audit: Mapping[str, Any],
    identity_evidence: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Certify bounded discrete counts and abstain on every unresolved identity."""

    upstream_errors = [
        *[f"M1: {error}" for error in validate_mep_sheet_registry(sheet_registry)],
        *[f"M4: {error}" for error in validate_mep_attribute_bindings(attribute_bindings)],
        *[
            f"coverage: {error}"
            for error in validate_mep_occurrence_coverage_audit(occurrence_coverage_audit)
        ],
    ]
    if upstream_errors:
        raise ValueError("invalid M7B input:\n" + "\n".join(upstream_errors))
    document_keys = {
        str(payload.get("document", {}).get("document_key"))
        for payload in (sheet_registry, attribute_bindings, occurrence_coverage_audit)
    }
    if len(document_keys) != 1:
        raise ValueError("M1, M4, and coverage-audit document keys must match")
    if occurrence_coverage_audit.get("m1_contract_ref", {}).get("payload_sha256") != _sha256(sheet_registry):
        raise ValueError("coverage audit does not bind the supplied M1 payload")
    if occurrence_coverage_audit.get("m4_contract_ref", {}).get("payload_sha256") != _sha256(attribute_bindings):
        raise ValueError("coverage audit does not bind the supplied M4 payload")

    occurrences = _observed_occurrences(attribute_bindings)
    occurrence_by_relation = {
        row["m4_relation_ref"]: row for row in occurrences
    }
    accepted_registration_refs = {
        str(row.get("id"))
        for row in sheet_registry.get("adjoining_sheet_transforms", [])
        if row.get("state") == "accepted"
    }
    evidence_rows = sorted(
        (deepcopy(dict(row)) for row in identity_evidence),
        key=lambda row: str(row.get("id")),
    )
    relation_evidence_membership: dict[str, list[str]] = {}
    for row in evidence_rows:
        for ref in set(str(value) for value in row.get("m4_relation_refs", [])):
            relation_evidence_membership.setdefault(ref, []).append(str(row.get("id")))

    certificates = []
    physical_items = []
    calculated_counts = []
    abstentions = []
    processed_relations: set[str] = set()
    for evidence in evidence_rows:
        evidence_ref = str(evidence.get("id"))
        relation_refs = sorted(set(str(ref) for ref in evidence.get("m4_relation_refs", [])))
        occurrence_rows = [occurrence_by_relation.get(ref) for ref in relation_refs]
        occurrence_refs = sorted(row["id"] for row in occurrence_rows if row is not None)
        reasons = []
        if not evidence_ref or not relation_refs:
            reasons.append("identity_evidence_requires_relation_refs")
        if any(row is None for row in occurrence_rows):
            reasons.append("identity_evidence_references_noneligible_m4_relation")
        if any(len(relation_evidence_membership.get(ref, [])) != 1 for ref in relation_refs):
            reasons.append("ambiguous_repetition_or_multiple_identity_evidence")
        if any(len(row["m4_target_refs"]) != 1 for row in occurrence_rows if row):
            reasons.append("m4_target_identity_not_unique")
        categories = {row["category"] for row in occurrence_rows if row}
        subtypes = {row["item_subtype"] for row in occurrence_rows if row}
        candidates = {_sha256(row["candidate"]) for row in occurrence_rows if row}
        if len(categories) != 1 or len(subtypes) != 1 or len(candidates) != 1:
            reasons.append("incompatible_discrete_item_identity")
        canonicalization = evidence.get("canonicalization", {})
        physical_identity = evidence.get("physical_identity", {})
        multiple = len(relation_refs) > 1
        if evidence.get("state") != "observed":
            reasons.append("identity_evidence_not_observed")
        if canonicalization.get("duplicate_projection") is not multiple:
            reasons.append("duplicate_projection_cardinality_mismatch")
        for key in ("mutually_unique", "target_signatures_match", "no_alternative_pairing"):
            if canonicalization.get(key) is not True:
                reasons.append(f"canonicalization_{key}_not_closed")
        if physical_identity.get("unique_physical_target") is not True:
            reasons.append("unique_physical_target_not_closed")
        if physical_identity.get("repetition_resolved") is not True:
            reasons.append("ambiguous_repetition")
        pages = {row["page_ref"] for row in occurrence_rows if row}
        cross_sheet = len(pages) > 1
        cross_sheet_state = physical_identity.get("cross_sheet_identity")
        registration_refs = sorted(
            str(ref) for ref in physical_identity.get("accepted_registration_refs", [])
        )
        if cross_sheet:
            if cross_sheet_state != "accepted_same_physical_item":
                reasons.append("cross_sheet_identity_unresolved")
            if not registration_refs or any(
                ref not in accepted_registration_refs for ref in registration_refs
            ):
                reasons.append("accepted_cross_sheet_registration_missing")
        elif cross_sheet_state != "not_applicable":
            reasons.append("cross_sheet_identity_state_mismatch")
        if not evidence.get("evidence_refs"):
            reasons.append("identity_evidence_refs_missing")

        processed_relations.update(ref for ref in relation_refs if ref in occurrence_by_relation)
        if reasons:
            abstentions.append(_abstention(occurrence_refs, evidence_ref, reasons))
            continue
        certificate = {
            "record_type": "mep_discrete_identity_certificate",
            "record_version": SCHEMA_VERSION,
            "id": _id("mep_discrete_identity_certificate", evidence_ref, relation_refs),
            "state": "accepted",
            "category": next(iter(categories)),
            "item_subtype": next(iter(subtypes)),
            "occurrence_refs": occurrence_refs,
            "m4_relation_refs": relation_refs,
            "canonical_target_refs": sorted(row["m4_target_refs"][0] for row in occurrence_rows),
            "identity_evidence_ref": evidence_ref,
            "counts": {
                "observed_occurrence_count": len(occurrence_rows),
                "deduplicated_projected_count": 1,
                "physical_instance_count": 1,
            },
            "certificates": {
                "unique_m4_target_identity": True,
                "duplicate_projection_canonicalized_or_not_required": True,
                "mutually_unique": True,
                "matching_target_signatures": True,
                "no_alternative_pairing": True,
                "repetition_resolved": True,
                "cross_sheet_identity_resolved_or_not_applicable": True,
            },
            "evidence_refs": sorted({
                evidence_ref,
                *[str(ref) for ref in evidence.get("evidence_refs", [])],
                *registration_refs,
                *relation_refs,
                *[ref for row in occurrence_rows for ref in row["evidence_refs"]],
            }),
            "quantity_eligible": False,
        }
        physical_item_id = _id("mep_discrete_physical_item", certificate["id"])
        physical_item = {
            "record_type": "mep_discrete_physical_item",
            "record_version": SCHEMA_VERSION,
            "id": physical_item_id,
            "category": certificate["category"],
            "item_subtype": certificate["item_subtype"],
            "occurrence_refs": occurrence_refs,
            "identity_certificate_ref": certificate["id"],
            "counts": deepcopy(certificate["counts"]),
            "physical_identity_established": True,
            "calculated_count_eligible": True,
            "installed_length_eligible": False,
            "evidence_refs": deepcopy(certificate["evidence_refs"]),
            "quantity_eligible": False,
        }
        calculated_count = {
            "record_type": "mep_calculated_discrete_count",
            "record_version": SCHEMA_VERSION,
            "id": _id("mep_calculated_discrete_count", physical_item_id),
            "physical_item_ref": physical_item_id,
            "identity_certificate_ref": certificate["id"],
            "category": certificate["category"],
            "item_subtype": certificate["item_subtype"],
            "value": 1,
            "unit": "ea",
            "state": "calculated",
            "epistemic_state": "derived",
            "approved_for_quote": False,
            "evidence_refs": [certificate["id"], physical_item_id],
            "quantity_eligible": False,
        }
        certificates.append(certificate)
        physical_items.append(physical_item)
        calculated_counts.append(calculated_count)

    for relation_ref, occurrence in sorted(occurrence_by_relation.items()):
        if relation_ref not in processed_relations:
            abstentions.append(_abstention(
                [occurrence["id"]], None, ["physical_identity_evidence_missing"]
            ))

    coverage_abstentions = [
        {
            "coverage_finding_ref": str(row.get("id")),
            "page_ref": str(row.get("page_ref")),
            "categories": deepcopy(row.get("categories", [])),
            "reason": "reviewed_hint_lacks_accepted_m4_target",
            "item_occurrence_established": False,
            "calculated_count": None,
        }
        for row in occurrence_coverage_audit.get("reviewed_unresolved_findings", [])
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "layer": LAYER,
        "document": deepcopy(dict(sheet_registry.get("document", {}))),
        "m1_contract_ref": {
            "layer": sheet_registry.get("layer"),
            "schema_version": sheet_registry.get("schema_version"),
            "payload_sha256": _sha256(sheet_registry),
        },
        "m4_contract_ref": {
            "layer": attribute_bindings.get("layer"),
            "schema_version": attribute_bindings.get("schema_version"),
            "payload_sha256": _sha256(attribute_bindings),
        },
        "coverage_audit_contract_ref": {
            "layer": occurrence_coverage_audit.get("layer"),
            "schema_version": occurrence_coverage_audit.get("schema_version"),
            "payload_sha256": _sha256(occurrence_coverage_audit),
        },
        "identity_evidence_sha256": _sha256(evidence_rows),
        "identity_evidence": evidence_rows,
        "observed_discrete_occurrences": occurrences,
        "identity_certificates": sorted(certificates, key=lambda row: row["id"]),
        "physical_items": sorted(physical_items, key=lambda row: row["id"]),
        "calculated_count_records": sorted(calculated_counts, key=lambda row: row["id"]),
        "abstentions": sorted(abstentions, key=lambda row: row["id"]),
        "coverage_abstentions": coverage_abstentions,
        "summary": {
            "observed_discrete_occurrence_count": len(occurrences),
            "accepted_identity_certificate_count": len(certificates),
            "deduplicated_projected_item_count": len(physical_items),
            "physical_item_count": len(physical_items),
            "calculated_count_record_count": len(calculated_counts),
            "calculated_count_total": sum(row["value"] for row in calculated_counts),
            "identity_abstention_count": len(abstentions),
            "coverage_abstention_count": len(coverage_abstentions),
        },
        "exchange_contract": {
            "discrete_items_only": True,
            "unique_m4_target_required": True,
            "duplicate_projection_canonicalization_required": True,
            "observed_deduplicated_and_physical_counts_separate": True,
            "calculated_count_requires_physical_identity": True,
            "ambiguous_repetition_abstains": True,
            "cross_sheet_identity_ambiguity_abstains": True,
            "reviewed_coverage_hints_are_not_occurrences": True,
            "installed_route_length_emitted": False,
            "projected_route_length_emitted": False,
            "bounded_local_route_length_emitted": False,
            "declared_data_used": False,
            "approved_for_quote": False,
        },
        "quantity_eligible": False,
    }
    errors = validate_mep_bounded_discrete_counts(payload)
    if errors:
        raise ValueError("\n".join(errors))
    return payload


def validate_mep_bounded_discrete_counts(payload: Mapping[str, Any]) -> list[str]:
    errors = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if payload.get("layer") != LAYER:
        errors.append("layer mismatch")
    for name in ("m1_contract_ref", "m4_contract_ref", "coverage_audit_contract_ref"):
        digest = payload.get(name, {}).get("payload_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            errors.append(f"{name}.payload_sha256 must be a SHA-256 digest")
    if payload.get("identity_evidence_sha256") != _sha256(
        payload.get("identity_evidence", [])
    ):
        errors.append("identity evidence digest mismatch")
    forbidden_keys = {
        "installed_length",
        "installed_length_m",
        "projected_length",
        "projected_length_m",
        "bounded_local_length",
        "bounded_local_length_m",
        "material_mass",
        "cost",
    }

    def walk(value: object, path: str = "") -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                location = f"{path}.{key}" if path else str(key)
                if key in forbidden_keys:
                    errors.append(f"{location}: forbidden M7B route/quantity field")
                walk(child, location)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(payload)
    occurrences = list(payload.get("observed_discrete_occurrences", []))
    occurrence_ids = {str(row.get("id")) for row in occurrences}
    if len(occurrence_ids) != len(occurrences):
        errors.append("discrete occurrence IDs must be unique")
    for row in occurrences:
        identifier = str(row.get("id"))
        if row.get("category") not in set(DISCRETE_RELATION_TYPES.values()):
            errors.append(f"{identifier}: unsupported discrete category")
        if row.get("counts") != {
            "observed_occurrence_count": 1,
            "deduplicated_projected_count": None,
            "physical_instance_count": None,
            "calculated_count": None,
        }:
            errors.append(f"{identifier}: occurrence count channels changed")
        if row.get("physical_identity_established") is not False:
            errors.append(f"{identifier}: occurrence cannot establish physical identity")
    certificates = list(payload.get("identity_certificates", []))
    certificate_ids = {str(row.get("id")) for row in certificates}
    if len(certificate_ids) != len(certificates):
        errors.append("identity certificate IDs must be unique")
    certified_occurrence_refs = []
    for row in certificates:
        identifier = str(row.get("id"))
        refs = [str(ref) for ref in row.get("occurrence_refs", [])]
        if not refs or any(ref not in occurrence_ids for ref in refs):
            errors.append(f"{identifier}: unknown occurrence ref")
        counts = row.get("counts", {})
        if counts.get("observed_occurrence_count") != len(refs):
            errors.append(f"{identifier}: observed count mismatch")
        if counts.get("deduplicated_projected_count") != 1:
            errors.append(f"{identifier}: deduplicated count must be one")
        if counts.get("physical_instance_count") != 1:
            errors.append(f"{identifier}: physical count must be one")
        if not all(row.get("certificates", {}).values()):
            errors.append(f"{identifier}: every identity gate must close")
        certified_occurrence_refs.extend(refs)
    if len(certified_occurrence_refs) != len(set(certified_occurrence_refs)):
        errors.append("an occurrence cannot belong to multiple accepted identities")
    physical_items = list(payload.get("physical_items", []))
    physical_ids = {str(row.get("id")) for row in physical_items}
    if len(physical_ids) != len(physical_items):
        errors.append("physical item IDs must be unique")
    if len(physical_items) != len(certificates):
        errors.append("each identity certificate must produce one physical item")
    for row in physical_items:
        certificate_ref = str(row.get("identity_certificate_ref"))
        if certificate_ref not in certificate_ids:
            errors.append(f"{row.get('id')}: identity certificate missing")
        else:
            certificate = next(
                item for item in certificates if str(item.get("id")) == certificate_ref
            )
            if row.get("occurrence_refs") != certificate.get("occurrence_refs"):
                errors.append(f"{row.get('id')}: certificate occurrence membership mismatch")
            if row.get("counts") != certificate.get("counts"):
                errors.append(f"{row.get('id')}: certificate count channels mismatch")
        if row.get("physical_identity_established") is not True:
            errors.append(f"{row.get('id')}: physical identity must be explicit")
        if row.get("calculated_count_eligible") is not True:
            errors.append(f"{row.get('id')}: calculated count eligibility missing")
        if row.get("installed_length_eligible") is not False:
            errors.append(f"{row.get('id')}: installed length must remain ineligible")
    counts = list(payload.get("calculated_count_records", []))
    count_ids = {str(row.get("id")) for row in counts}
    if len(count_ids) != len(counts):
        errors.append("calculated count IDs must be unique")
    if len(counts) != len(physical_items):
        errors.append("each physical item must produce one calculated count")
    for row in counts:
        physical_ref = str(row.get("physical_item_ref"))
        if physical_ref not in physical_ids:
            errors.append(f"{row.get('id')}: physical item ref missing")
        else:
            physical = next(
                item for item in physical_items if str(item.get("id")) == physical_ref
            )
            if row.get("identity_certificate_ref") != physical.get("identity_certificate_ref"):
                errors.append(f"{row.get('id')}: identity certificate chain mismatch")
        if row.get("value") != 1 or row.get("unit") != "ea" or row.get("state") != "calculated":
            errors.append(f"{row.get('id')}: calculated discrete count must be 1 ea")
        if row.get("approved_for_quote") is not False:
            errors.append(f"{row.get('id')}: quote approval must remain false")
    for row in payload.get("abstentions", []):
        if any(row.get(key) is not None for key in (
            "deduplicated_projected_count", "physical_instance_count", "calculated_count"
        )):
            errors.append(f"{row.get('id')}: abstention count channels must remain null")
    coverage_abstentions = list(payload.get("coverage_abstentions", []))
    coverage_finding_refs = {
        str(row.get("coverage_finding_ref")) for row in coverage_abstentions
    }
    if len(coverage_finding_refs) != len(coverage_abstentions):
        errors.append("coverage abstention finding refs must be unique")
    for row in coverage_abstentions:
        if row.get("reason") != "reviewed_hint_lacks_accepted_m4_target":
            errors.append(f"{row.get('coverage_finding_ref')}: coverage reason mismatch")
        if row.get("item_occurrence_established") is not False:
            errors.append(f"{row.get('coverage_finding_ref')}: hint cannot become an occurrence")
        if row.get("calculated_count") is not None:
            errors.append(f"{row.get('coverage_finding_ref')}: hint count must remain null")
    contract = payload.get("exchange_contract", {})
    for key in (
        "installed_route_length_emitted",
        "projected_route_length_emitted",
        "bounded_local_route_length_emitted",
        "declared_data_used",
        "approved_for_quote",
    ):
        if contract.get(key) is not False:
            errors.append(f"exchange_contract.{key} must remain false")
    summary = payload.get("summary", {})
    expected = {
        "observed_discrete_occurrence_count": len(occurrences),
        "accepted_identity_certificate_count": len(certificates),
        "deduplicated_projected_item_count": len(physical_items),
        "physical_item_count": len(physical_items),
        "calculated_count_record_count": len(counts),
        "calculated_count_total": sum(row.get("value", 0) for row in counts),
        "identity_abstention_count": len(payload.get("abstentions", [])),
        "coverage_abstention_count": len(coverage_abstentions),
    }
    if summary != expected:
        errors.append("summary mismatch")
    if payload.get("quantity_eligible") is not False:
        errors.append("catalog-level quantity eligibility must remain false")
    return errors
