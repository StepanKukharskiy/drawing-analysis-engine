"""M7A structured MEP item catalog with an additive bounded-M7B envelope.

M7A is a read-only adapter over accepted M1 page metadata and M4 item/route
relations.  Optional M5A records may enrich an occurrence with a measured
physical envelope dimension and resolved local elevation, but never with a
projected or installed length.  Observed occurrences, deduplicated projected
counts, physical counts, and calculated/declared/reviewed/approved values stay
separate.  M7B may add physical-item and calculated discrete-count records only
after its identity certificate closes; occurrence records remain frozen at the
M7A 0.1 contract.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from src.drawing_engine.disciplines.mep.mep_attribute_binding import validate_mep_attribute_bindings
from src.drawing_engine.disciplines.mep.mep_bounded_discrete_counts import validate_mep_bounded_discrete_counts
from src.drawing_engine.disciplines.mep.mep_bounded_local_3d import validate_mep_bounded_local_3d_segments
from src.drawing_engine.disciplines.mep.mep_occurrence_coverage_audit import validate_mep_occurrence_coverage_audit
from src.drawing_engine.disciplines.mep.mep_sheet_registry import validate_mep_sheet_registry


SCHEMA_VERSION = "0.2.0"
M7A_RECORD_VERSION = "0.1.0"
LAYER = "mep_structured_item_catalog"
METHOD_VERSION = "1.0.0"

RECORD_CONTRACTS = {
    "mep_item_occurrence": {
        "record_version": M7A_RECORD_VERSION,
        "required_fields": [
            "id",
            "item_type",
            "item_subtype",
            "system",
            "description",
            "source",
            "typed_dimensions",
            "elevations",
            "manufacturer",
            "model_number",
            "specification_reference",
            "mounting_type",
            "counts",
            "evidence_refs",
            "epistemic_state",
            "unresolved_reasons",
            "authority",
        ],
        "nullable_fields": [
            "system",
            "manufacturer",
            "model_number",
            "specification_reference",
            "mounting_type",
            "counts.deduplicated_projected_count",
            "counts.physical_instance_count",
        ],
    },
    "mep_physical_item": {
        "record_version": M7A_RECORD_VERSION,
        "required_fields": [
            "id",
            "item_occurrence_refs",
            "identity_certificate_refs",
            "counts",
            "epistemic_state",
            "unresolved_reasons",
            "authority",
        ],
        "nullable_fields": ["counts.physical_instance_count"],
    },
    "mep_takeoff_line": {
        "record_version": M7A_RECORD_VERSION,
        "required_fields": [
            "id",
            "item_occurrence_ref",
            "item_type",
            "item_subtype",
            "system",
            "source",
            "counts",
            "value_channels",
            "unresolved_reasons",
            "authority",
        ],
        "value_channels": [
            "calculated",
            "declared",
            "reviewed",
            "approved_for_quote",
        ],
    },
    "mep_calculated_discrete_count": {
        "record_version": "0.1.0",
        "required_fields": [
            "id",
            "physical_item_ref",
            "identity_certificate_ref",
            "category",
            "item_subtype",
            "value",
            "unit",
            "state",
            "evidence_refs",
        ],
    },
}

_ROUTE_ATTRIBUTES = {"route_system", "route_size", "route_elevation"}
_ITEM_RELATION_TYPES = {
    "equipment_endpoint": ("equipment", "connected_equipment"),
    "valve": ("valve", "inline_valve"),
    "fitting": ("fitting", "inline_fitting"),
    "damper": ("damper", "inline_damper"),
    "riser_drop": ("route_transition", "riser_drop"),
}
_FALSE_AUTHORITY_KEYS = (
    "deduplicated_projected_count_established",
    "physical_item_identity_established",
    "physical_instance_count_established",
    "calculated_quantity_established",
    "installed_length_emitted",
    "declared_data_extracted",
    "approved_for_quote",
    "quantity_eligible",
)


def _stable_id(kind: str, *parts: object) -> str:
    encoded = json.dumps(
        parts, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"{kind}.{hashlib.sha256(encoded).hexdigest()[:20]}"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _semantic_candidate(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(item)
        for key, item in value.items()
        if key not in {"raw_text", "terminology_entry_ref"}
    }


def _semantic_key(value: Mapping[str, Any]) -> str:
    return json.dumps(
        _semantic_candidate(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _page_source(page: Mapping[str, Any]) -> dict[str, Any]:
    fields = page.get("fields", {})
    sheet = fields.get("sheet_number", {})
    return {
        "page_ref": str(page.get("page_ref")),
        "pdf_page_number": int(page.get("page_number")),
        "drawing_sheet_number": sheet.get("value"),
        "drawing_sheet_number_state": str(sheet.get("state") or "unknown"),
        "sheet_role": str(page.get("role") or "unknown"),
        "sheet_record_ref": str(page.get("id")),
    }


def _attribute_value(
    relations: Iterable[Mapping[str, Any]], relation_type: str
) -> tuple[dict[str, Any] | None, list[str], str | None]:
    rows = [row for row in relations if row.get("relation_type") == relation_type]
    values = {_semantic_key(row.get("candidate", {})) for row in rows}
    refs = sorted(str(row.get("id")) for row in rows)
    if not rows:
        return None, refs, f"missing_{relation_type}"
    if len(values) != 1:
        return None, refs, f"conflicting_{relation_type}"
    return _semantic_candidate(rows[0].get("candidate", {})), refs, None


def _description(system: Mapping[str, Any] | None, size: Mapping[str, Any] | None) -> str:
    system_text = str((system or {}).get("kind") or "unclassified route").replace("_", " ")
    if size and size.get("value") is not None and size.get("unit"):
        size_text = f"{size['value']} {size['unit']}"
        return f"{system_text} {size_text} routed material"
    return f"{system_text} routed material"


def _route_subtype(size: Mapping[str, Any] | None) -> str:
    kind = str((size or {}).get("kind") or "")
    if kind == "rectangular_duct_size":
        return "duct_route"
    if kind in {"nominal_size", "pipe_size", "diameter"}:
        return "pipe_route"
    return "route"


def _typed_size(size: Mapping[str, Any] | None, relation_refs: list[str]) -> list[dict[str, Any]]:
    if size is None:
        return []
    row = {
        "dimension_type": str(size.get("kind") or "size"),
        "value": size.get("value"),
        "unit": size.get("unit"),
        "designation": size.get("designation"),
        "width": size.get("width"),
        "height": size.get("height"),
        "state": "derived",
        "physical_dimension": False,
        "evidence_refs": relation_refs,
    }
    return [row]


def _typed_elevation(
    elevation: Mapping[str, Any] | None, relation_refs: list[str]
) -> list[dict[str, Any]]:
    if elevation is None:
        return []
    return [{
        "elevation_type": str(elevation.get("basis") or "unknown"),
        "value": elevation.get("value"),
        "unit": elevation.get("unit"),
        "state": "derived",
        "evidence_refs": relation_refs,
    }]


def _m5a_by_composite(
    bounded_local_3d: Mapping[str, Any] | None,
) -> dict[str, Mapping[str, Any]]:
    output = {}
    if bounded_local_3d is None:
        return output
    for segment in bounded_local_3d.get("bounded_local_3d_segments", []):
        if segment.get("state") != "accepted":
            continue
        for ref in segment.get("source_route_composite_refs", []):
            output[str(ref)] = segment
    return output


def _metadata_unresolved() -> list[str]:
    return [
        "manufacturer_not_observed",
        "model_number_not_observed",
        "specification_reference_not_observed",
        "mounting_type_not_observed",
        "deduplicated_projected_count_not_established",
        "physical_instance_count_not_established",
        "calculated_quantity_not_established",
        "declared_quantity_not_extracted",
        "review_not_recorded",
        "quote_approval_not_recorded",
    ]


def _authority() -> dict[str, bool]:
    return {key: False for key in _FALSE_AUTHORITY_KEYS}


def _counts() -> dict[str, int | None]:
    return {
        "observed_occurrence_count": 1,
        "deduplicated_projected_count": None,
        "physical_instance_count": None,
    }


def _value_channels() -> dict[str, dict[str, Any]]:
    return {
        "calculated": {
            "value": None,
            "unit": None,
            "state": "unknown",
            "reason": "m7b_calculated_quantity_not_established",
        },
        "declared": {
            "value": None,
            "unit": None,
            "state": "unknown",
            "reason": "m7c_declared_data_not_extracted",
        },
        "reviewed": {
            "value": None,
            "unit": None,
            "state": "unknown",
            "reason": "engineer_review_not_recorded",
        },
        "approved_for_quote": {
            "value": None,
            "unit": None,
            "state": "unknown",
            "reason": "quote_approval_not_recorded",
        },
    }


def _route_occurrences(
    *,
    pages: Mapping[str, Mapping[str, Any]],
    bindings: Mapping[str, Any],
    m5a: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    relations_by_target: dict[str, list[Mapping[str, Any]]] = {}
    for relation in bindings.get("relations", []):
        if relation.get("state") != "accepted" or relation.get("relation_type") not in _ROUTE_ATTRIBUTES:
            continue
        for target_ref in relation.get("target_refs", []):
            relations_by_target.setdefault(str(target_ref), []).append(relation)
    composites = {
        str(row.get("id")): row
        for row in bindings.get("outlined_route_composites", [])
        if row.get("state") == "accepted"
    }
    output = []
    for target_ref in sorted(relations_by_target):
        relations = relations_by_target[target_ref]
        composite = composites.get(target_ref)
        if composite is None:
            continue
        page_ref = str(composite.get("page_ref"))
        page = pages.get(page_ref)
        if page is None:
            continue
        system, system_refs, system_reason = _attribute_value(relations, "route_system")
        size, size_refs, size_reason = _attribute_value(relations, "route_size")
        elevation, elevation_refs, elevation_reason = _attribute_value(
            relations, "route_elevation"
        )
        typed_dimensions = _typed_size(size, size_refs)
        elevations = _typed_elevation(elevation, elevation_refs)
        enrichment = m5a.get(target_ref)
        enrichment_refs: list[str] = []
        if enrichment is not None:
            dimension = enrichment.get("physical_envelope_dimension", {})
            typed_dimensions.append({
                "dimension_type": str(dimension.get("kind") or "physical_envelope"),
                "value": dimension.get("representative_outer_width_m"),
                "unit": "m",
                "designation": "drawing_measured",
                "width": None,
                "height": None,
                "state": "derived",
                "physical_dimension": True,
                "evidence_refs": sorted({
                    str(enrichment.get("id")),
                    *[str(ref) for ref in enrichment.get("source_primitive_refs", [])],
                }),
            })
            local_elevation = enrichment.get("elevation", {})
            elevations.append({
                "elevation_type": "centreline",
                "value": local_elevation.get("centreline_elevation_m"),
                "unit": "m",
                "state": "derived",
                "derivation": local_elevation.get("derivation"),
                "evidence_refs": [str(enrichment.get("id"))],
            })
            enrichment_refs.append(str(enrichment.get("id")))
        evidence_refs = sorted({
            target_ref,
            *[str(ref) for ref in composite.get("member_fragment_refs", [])],
            *[str(ref) for ref in composite.get("member_source_primitive_refs", [])],
            *[str(row.get("id")) for row in relations],
            *[str(ref) for row in relations for ref in row.get("proposal_evidence_refs", [])],
            *enrichment_refs,
        })
        unresolved = _metadata_unresolved()
        for reason in (system_reason, size_reason, elevation_reason):
            if reason:
                unresolved.append(reason)
        occurrence_id = _stable_id("mep_item_occurrence", target_ref, page_ref)
        output.append({
            "record_type": "mep_item_occurrence",
            "record_version": M7A_RECORD_VERSION,
            "id": occurrence_id,
            "item_type": "routed_material",
            "item_subtype": _route_subtype(size),
            "system": system,
            "description": _description(system, size),
            "description_state": "derived",
            "source": _page_source(page),
            "target_kind": "route_composite",
            "target_refs": [target_ref],
            "source_m4_relation_ref": None,
            "typed_dimensions": typed_dimensions,
            "elevations": elevations,
            "manufacturer": None,
            "model_number": None,
            "specification_reference": None,
            "mounting_type": None,
            "counts": _counts(),
            "associated_bounded_local_3d_segment_ref": (
                str(enrichment.get("id")) if enrichment is not None else None
            ),
            "evidence_refs": evidence_refs,
            "epistemic_state": "derived",
            "unresolved_reasons": sorted(set(unresolved)),
            "authority": _authority(),
            "quantity_eligible": False,
        })
    return output


def _accessory_occurrences(
    *, pages: Mapping[str, Mapping[str, Any]], bindings: Mapping[str, Any]
) -> list[dict[str, Any]]:
    output = []
    for relation in sorted(bindings.get("relations", []), key=lambda row: str(row.get("id"))):
        relation_type = str(relation.get("relation_type") or "")
        if relation.get("state") != "accepted" or relation_type not in _ITEM_RELATION_TYPES:
            continue
        page_ref = str(relation.get("page_ref"))
        page = pages.get(page_ref)
        if page is None:
            continue
        item_type, default_subtype = _ITEM_RELATION_TYPES[relation_type]
        candidate = _semantic_candidate(relation.get("candidate", {}))
        raw_text = str(relation.get("candidate", {}).get("raw_text") or "").strip()
        subtype = str(candidate.get("kind") or default_subtype)
        occurrence_id = _stable_id("mep_item_occurrence", relation.get("id"), page_ref)
        evidence_refs = sorted({
            str(relation.get("id")),
            str(relation.get("proposal_ref")),
            *[str(ref) for ref in relation.get("target_refs", [])],
            *[str(ref) for ref in relation.get("target_fragment_refs", [])],
            *[str(ref) for ref in relation.get("proposal_evidence_refs", [])],
            *[str(ref) for ref in relation.get("binding_evidence_refs", [])],
        })
        output.append({
            "record_type": "mep_item_occurrence",
            "record_version": M7A_RECORD_VERSION,
            "id": occurrence_id,
            "item_type": item_type,
            "item_subtype": subtype,
            "system": None,
            "description": raw_text or subtype.replace("_", " "),
            "description_state": "observed" if raw_text else "derived",
            "source": _page_source(page),
            "target_kind": relation.get("target_kind"),
            "target_refs": sorted(str(ref) for ref in relation.get("target_refs", [])),
            "source_m4_relation_ref": str(relation.get("id")),
            "typed_dimensions": [],
            "elevations": [],
            "manufacturer": None,
            "model_number": None,
            "specification_reference": None,
            "mounting_type": None,
            "counts": _counts(),
            "associated_bounded_local_3d_segment_ref": None,
            "evidence_refs": evidence_refs,
            "epistemic_state": str(relation.get("epistemic_state") or "derived"),
            "unresolved_reasons": sorted(set([
                *_metadata_unresolved(),
                "system_not_bound_to_item_target",
            ])),
            "authority": _authority(),
            "quantity_eligible": False,
        })
    return output


def _takeoff_line(occurrence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "record_type": "mep_takeoff_line",
        "record_version": M7A_RECORD_VERSION,
        "id": _stable_id("mep_takeoff_line", occurrence.get("id")),
        "item_occurrence_ref": str(occurrence.get("id")),
        "physical_item_ref": None,
        "item_type": occurrence.get("item_type"),
        "item_subtype": occurrence.get("item_subtype"),
        "system": deepcopy(occurrence.get("system")),
        "description": occurrence.get("description"),
        "source": deepcopy(occurrence.get("source")),
        "counts": deepcopy(occurrence.get("counts")),
        "value_channels": _value_channels(),
        "evidence_refs": deepcopy(occurrence.get("evidence_refs", [])),
        "epistemic_state": "unknown",
        "unresolved_reasons": deepcopy(occurrence.get("unresolved_reasons", [])),
        "authority": _authority(),
        "quantity_eligible": False,
    }


def _m7b_catalog_records(
    occurrences: Sequence[Mapping[str, Any]],
    bounded_discrete_counts: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if bounded_discrete_counts is None:
        return [], []
    catalog_occurrence_by_relation = {
        str(row.get("source_m4_relation_ref")): row
        for row in occurrences
        if row.get("source_m4_relation_ref")
    }
    m7b_occurrences = {
        str(row.get("id")): row
        for row in bounded_discrete_counts.get("observed_discrete_occurrences", [])
    }
    physical_items = []
    physical_ref_map = {}
    for source in bounded_discrete_counts.get("physical_items", []):
        item_occurrence_refs = []
        for m7b_occurrence_ref in source.get("occurrence_refs", []):
            m7b_occurrence = m7b_occurrences.get(str(m7b_occurrence_ref), {})
            catalog_occurrence = catalog_occurrence_by_relation.get(
                str(m7b_occurrence.get("m4_relation_ref"))
            )
            if catalog_occurrence is None:
                raise ValueError(
                    f"M7B physical item cannot map occurrence {m7b_occurrence_ref} to M7A"
                )
            item_occurrence_refs.append(str(catalog_occurrence.get("id")))
        identifier = _stable_id("mep_physical_item", source.get("id"))
        physical_ref_map[str(source.get("id"))] = identifier
        physical_items.append({
            "record_type": "mep_physical_item",
            "record_version": M7A_RECORD_VERSION,
            "id": identifier,
            "item_occurrence_refs": sorted(item_occurrence_refs),
            "identity_certificate_refs": [str(source.get("identity_certificate_ref"))],
            "source_m7b_physical_item_ref": str(source.get("id")),
            "category": source.get("category"),
            "item_subtype": source.get("item_subtype"),
            "counts": deepcopy(source.get("counts", {})),
            "epistemic_state": "derived",
            "unresolved_reasons": [
                "engineer_review_not_recorded",
                "quote_approval_not_recorded",
            ],
            "evidence_refs": deepcopy(source.get("evidence_refs", [])),
            "authority": {
                "physical_item_identity_established": True,
                "physical_instance_count_established": True,
                "calculated_count_established": True,
                "installed_length_emitted": False,
                "approved_for_quote": False,
                "quantity_eligible": False,
            },
            "quantity_eligible": False,
        })
    calculated_counts = []
    for source in bounded_discrete_counts.get("calculated_count_records", []):
        physical_item_ref = physical_ref_map.get(str(source.get("physical_item_ref")))
        if physical_item_ref is None:
            raise ValueError("M7B calculated count lacks a mapped physical item")
        calculated_counts.append({
            **deepcopy(dict(source)),
            "id": _stable_id("mep_calculated_discrete_count", source.get("id")),
            "physical_item_ref": physical_item_ref,
            "source_m7b_calculated_count_ref": str(source.get("id")),
        })
    return (
        sorted(physical_items, key=lambda row: row["id"]),
        sorted(calculated_counts, key=lambda row: row["id"]),
    )


def build_mep_item_catalog(
    *,
    sheet_registry: Mapping[str, Any],
    attribute_bindings: Mapping[str, Any],
    bounded_local_3d: Mapping[str, Any] | None = None,
    occurrence_coverage_audit: Mapping[str, Any] | None = None,
    bounded_discrete_counts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the catalog from M1/M4, optional M5A, and bounded M7B counts."""

    upstream_errors = [
        *[f"M1: {error}" for error in validate_mep_sheet_registry(sheet_registry)],
        *[f"M4: {error}" for error in validate_mep_attribute_bindings(attribute_bindings)],
    ]
    if bounded_local_3d is not None:
        upstream_errors.extend(
            f"M5A: {error}"
            for error in validate_mep_bounded_local_3d_segments(bounded_local_3d)
        )
    if occurrence_coverage_audit is not None:
        upstream_errors.extend(
            f"coverage: {error}"
            for error in validate_mep_occurrence_coverage_audit(occurrence_coverage_audit)
        )
    if bounded_discrete_counts is not None:
        upstream_errors.extend(
            f"M7B: {error}"
            for error in validate_mep_bounded_discrete_counts(bounded_discrete_counts)
        )
    if (occurrence_coverage_audit is None) != (bounded_discrete_counts is None):
        upstream_errors.append("coverage audit and M7B counts must be supplied together")
    if upstream_errors:
        raise ValueError("invalid upstream MEP payload:\n" + "\n".join(upstream_errors))
    payloads = [sheet_registry, attribute_bindings]
    if bounded_local_3d is not None:
        payloads.append(bounded_local_3d)
    if occurrence_coverage_audit is not None:
        payloads.append(occurrence_coverage_audit)
    if bounded_discrete_counts is not None:
        payloads.append(bounded_discrete_counts)
    document_keys = {
        str(payload.get("document", {}).get("document_key"))
        for payload in payloads
        if payload.get("document", {}).get("document_key")
    }
    if len(document_keys) != 1:
        raise ValueError("M1, M4, optional M5A, coverage, and M7B document keys must match")
    if occurrence_coverage_audit is not None:
        if occurrence_coverage_audit.get("m1_contract_ref", {}).get("payload_sha256") != _canonical_sha256(sheet_registry):
            raise ValueError("coverage audit does not bind the supplied M1 payload")
        if occurrence_coverage_audit.get("m4_contract_ref", {}).get("payload_sha256") != _canonical_sha256(attribute_bindings):
            raise ValueError("coverage audit does not bind the supplied M4 payload")
    if bounded_discrete_counts is not None:
        if bounded_discrete_counts.get("coverage_audit_contract_ref", {}).get("payload_sha256") != _canonical_sha256(occurrence_coverage_audit):
            raise ValueError("M7B does not bind the supplied coverage audit")

    page_rows = sorted(
        sheet_registry.get("pages", []), key=lambda row: int(row.get("page_number", 0))
    )
    pages = {str(row.get("page_ref")): row for row in page_rows}
    occurrences = [
        *_route_occurrences(
            pages=pages,
            bindings=attribute_bindings,
            m5a=_m5a_by_composite(bounded_local_3d),
        ),
        *_accessory_occurrences(pages=pages, bindings=attribute_bindings),
    ]
    occurrences.sort(key=lambda row: (
        int(row["source"]["pdf_page_number"]),
        str(row["item_type"]),
        str(row["id"]),
    ))
    physical_items, calculated_discrete_counts = _m7b_catalog_records(
        occurrences, bounded_discrete_counts
    )
    occurrence_refs_by_page: dict[str, list[str]] = {}
    for row in occurrences:
        occurrence_refs_by_page.setdefault(str(row["source"]["page_ref"]), []).append(str(row["id"]))
    coverage_by_page = {
        str(row.get("page_ref")): row
        for row in (occurrence_coverage_audit or {}).get("pages", [])
    }
    catalog_pages = []
    for page in page_rows:
        source = _page_source(page)
        refs = sorted(occurrence_refs_by_page.get(source["page_ref"], []))
        coverage_page = coverage_by_page.get(source["page_ref"], {})
        catalog_pages.append({
            "record_type": "mep_catalog_page",
            "record_version": M7A_RECORD_VERSION,
            **source,
            "item_occurrence_refs": refs,
            "item_occurrence_count": len(refs),
            "coverage_state": (
                "accepted_m4_item_evidence_available"
                if refs
                else "no_accepted_m4_item_evidence"
            ),
            "coverage_audit_state": coverage_page.get("coverage_state"),
            "reviewed_unresolved_finding_count": len(
                coverage_page.get("reviewed_unresolved_finding_refs", [])
            ),
            "m4_discrete_target_review_state": coverage_page.get(
                "m4_discrete_target_review_state"
            ),
            "m4_discrete_target_relation_count": len(
                coverage_page.get("m4_discrete_target_relation_refs", [])
            ),
            "unresolved_reasons": (
                [] if refs else ["no_accepted_m4_item_relations_on_page"]
            ),
            "quantity_eligible": False,
        })
    takeoff_lines = [_takeoff_line(row) for row in occurrences]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "layer": LAYER,
        "document": deepcopy(dict(sheet_registry.get("document", {}))),
        "record_contracts": deepcopy(RECORD_CONTRACTS),
        "m1_contract_ref": {
            "layer": sheet_registry.get("layer"),
            "schema_version": sheet_registry.get("schema_version"),
            "payload_sha256": _canonical_sha256(sheet_registry),
        },
        "m4_contract_ref": {
            "layer": attribute_bindings.get("layer"),
            "schema_version": attribute_bindings.get("schema_version"),
            "payload_sha256": _canonical_sha256(attribute_bindings),
        },
        "m5a_contract_ref": (
            {
                "layer": bounded_local_3d.get("layer"),
                "schema_version": bounded_local_3d.get("schema_version"),
                "payload_sha256": _canonical_sha256(bounded_local_3d),
                "enrichment_scope": "physical_dimensions_and_local_elevation_only",
            }
            if bounded_local_3d is not None
            else None
        ),
        "coverage_audit_contract_ref": (
            {
                "layer": occurrence_coverage_audit.get("layer"),
                "schema_version": occurrence_coverage_audit.get("schema_version"),
                "payload_sha256": _canonical_sha256(occurrence_coverage_audit),
                "accepted_set_scope": occurrence_coverage_audit.get(
                    "coverage_conclusion", {}
                ).get("accepted_set_scope"),
                "document_occurrence_completeness": occurrence_coverage_audit.get(
                    "coverage_conclusion", {}
                ).get("document_occurrence_completeness"),
                "m4_covered_page_count": occurrence_coverage_audit.get(
                    "summary", {}
                ).get("m4_covered_page_count"),
                "uncovered_item_bearing_page_count": occurrence_coverage_audit.get(
                    "summary", {}
                ).get("uncovered_item_bearing_page_count"),
                "reviewed_negative_target_page_count": occurrence_coverage_audit.get(
                    "summary", {}
                ).get("reviewed_negative_target_page_count"),
                "candidate_inventory_not_established_page_count": occurrence_coverage_audit.get(
                    "summary", {}
                ).get("candidate_inventory_not_established_page_count"),
            }
            if occurrence_coverage_audit is not None
            else None
        ),
        "m7b_contract_ref": (
            {
                "layer": bounded_discrete_counts.get("layer"),
                "schema_version": bounded_discrete_counts.get("schema_version"),
                "payload_sha256": _canonical_sha256(bounded_discrete_counts),
                "scope": "bounded_discrete_counts_only",
                "coverage_abstention_count": len(
                    bounded_discrete_counts.get("coverage_abstentions", [])
                ),
            }
            if bounded_discrete_counts is not None
            else None
        ),
        "pages": catalog_pages,
        "item_occurrences": occurrences,
        "physical_items": physical_items,
        "calculated_discrete_counts": calculated_discrete_counts,
        "takeoff_lines": takeoff_lines,
        "summary": {
            "page_count": len(catalog_pages),
            "page_with_item_count": sum(bool(row["item_occurrence_refs"]) for row in catalog_pages),
            "item_occurrence_count": len(occurrences),
            "routed_material_occurrence_count": sum(row["item_type"] == "routed_material" for row in occurrences),
            "physical_item_count": len(physical_items),
            "takeoff_line_count": len(takeoff_lines),
            "calculated_value_count": len(calculated_discrete_counts),
            "declared_value_count": 0,
            "coverage_abstention_count": (
                len(bounded_discrete_counts.get("coverage_abstentions", []))
                if bounded_discrete_counts is not None
                else 0
            ),
        },
        "exchange_contract": {
            "all_registered_pages_preserved": True,
            "accepted_m4_relations_are_only_item_source": True,
            "m5a_enrichment_is_dimension_and_elevation_only": True,
            "observed_deduplicated_and_physical_counts_separate": True,
            "calculated_declared_reviewed_and_approved_values_separate": True,
            "missing_product_metadata_is_explicit_null": True,
            "projected_length_emitted": False,
            "bounded_local_length_emitted": False,
            "installed_length_emitted": False,
            "m7b_bounded_discrete_counts_integrated": bounded_discrete_counts is not None,
            "physical_count_emitted": bool(physical_items),
            "calculated_quantity_emitted": bool(calculated_discrete_counts),
            "declared_data_extracted": False,
            "schedule_values_used": False,
            "quantity_eligible": False,
        },
        "quantity_eligible": False,
    }
    errors = validate_mep_item_catalog(payload)
    if errors:
        raise ValueError("\n".join(errors))
    return payload


def _walk(value: object, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], str, object]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            yield path, name, child
            yield from _walk(child, (*path, name))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, (*path, f"[{index}]"))


def _finite_or_none(value: object) -> bool:
    return value is None or (
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    )


def validate_mep_item_catalog(payload: Mapping[str, Any]) -> list[str]:
    """Validate M7A catalog closure plus optional bounded M7B counts."""

    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if payload.get("layer") != LAYER:
        errors.append("layer mismatch")
    if payload.get("record_contracts") != RECORD_CONTRACTS:
        errors.append("record_contracts mismatch")
    for name in ("m1_contract_ref", "m4_contract_ref"):
        digest = payload.get(name, {}).get("payload_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            errors.append(f"{name}.payload_sha256 must be a SHA-256 digest")
    m5a_ref = payload.get("m5a_contract_ref")
    if m5a_ref is not None:
        digest = m5a_ref.get("payload_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            errors.append("m5a_contract_ref.payload_sha256 must be a SHA-256 digest")
        if m5a_ref.get("enrichment_scope") != "physical_dimensions_and_local_elevation_only":
            errors.append("m5a_contract_ref enrichment scope mismatch")
    coverage_ref = payload.get("coverage_audit_contract_ref")
    m7b_ref = payload.get("m7b_contract_ref")
    if (coverage_ref is None) != (m7b_ref is None):
        errors.append("coverage_audit_contract_ref and m7b_contract_ref must appear together")
    for name, ref in (("coverage_audit_contract_ref", coverage_ref), ("m7b_contract_ref", m7b_ref)):
        if ref is not None:
            digest = ref.get("payload_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                errors.append(f"{name}.payload_sha256 must be a SHA-256 digest")
    if coverage_ref is not None:
        if coverage_ref.get("accepted_set_scope") != "complete_for_frozen_m4_payload_only":
            errors.append("coverage accepted-set scope mismatch")
        if coverage_ref.get("document_occurrence_completeness") != "not_established":
            errors.append("document occurrence completeness must remain unresolved")
        for key in (
            "m4_covered_page_count",
            "uncovered_item_bearing_page_count",
            "reviewed_negative_target_page_count",
            "candidate_inventory_not_established_page_count",
        ):
            if not isinstance(coverage_ref.get(key), int) or coverage_ref.get(key) < 0:
                errors.append(f"coverage {key} must be a nonnegative integer")
    if m7b_ref is not None and m7b_ref.get("scope") != "bounded_discrete_counts_only":
        errors.append("M7B catalog scope mismatch")

    forbidden_keys = {
        "installed_length",
        "installed_length_m",
        "projected_length",
        "projected_length_m",
        "bounded_local_length",
        "bounded_local_length_m",
        "confirmed_clash",
        "calculated_severity",
        "material_mass",
        "cost",
    }
    false_flags = {
        "projected_length_emitted",
        "bounded_local_length_emitted",
        "installed_length_emitted",
        "declared_data_extracted",
        "schedule_values_used",
        "quantity_eligible",
    }
    for path, key, value in _walk(payload):
        location = ".".join((*path, key))
        if key in forbidden_keys:
            errors.append(f"{location}: forbidden M7A/M7B output")
        if key in false_flags and value is not False:
            errors.append(f"{location}: authority flag must remain false")

    pages = list(payload.get("pages", []))
    page_refs = [str(row.get("page_ref")) for row in pages]
    page_numbers = [row.get("pdf_page_number") for row in pages]
    if len(page_refs) != len(set(page_refs)) or len(page_numbers) != len(set(page_numbers)):
        errors.append("catalog pages must have unique refs and PDF page numbers")
    if sorted(page_numbers) != list(range(1, len(pages) + 1)):
        errors.append("catalog pages must cover every registered PDF page in order")

    occurrences = list(payload.get("item_occurrences", []))
    occurrence_ids = [str(row.get("id")) for row in occurrences]
    if len(occurrence_ids) != len(set(occurrence_ids)):
        errors.append("item occurrence IDs must be unique")
    occurrence_by_id = {str(row.get("id")): row for row in occurrences}
    page_ref_set = set(page_refs)
    page_memberships: list[str] = []
    for page in pages:
        refs = [str(ref) for ref in page.get("item_occurrence_refs", [])]
        page_memberships.extend(refs)
        if page.get("item_occurrence_count") != len(refs):
            errors.append(f"{page.get('page_ref')}: item occurrence count mismatch")
        if any(ref not in occurrence_by_id for ref in refs):
            errors.append(f"{page.get('page_ref')}: unknown item occurrence ref")
        expected_state = "accepted_m4_item_evidence_available" if refs else "no_accepted_m4_item_evidence"
        if page.get("coverage_state") != expected_state:
            errors.append(f"{page.get('page_ref')}: coverage state mismatch")
        if page.get("quantity_eligible") is not False:
            errors.append(f"{page.get('page_ref')}: quantity_eligible must be false")
        audit_state = page.get("coverage_audit_state")
        hint_count = page.get("reviewed_unresolved_finding_count")
        target_review_state = page.get("m4_discrete_target_review_state")
        target_relation_count = page.get("m4_discrete_target_relation_count")
        if coverage_ref is not None:
            if audit_state not in {
                "not_applicable_divider",
                "current_m4_covered_subset",
                "not_m4_occurrence_covered",
            }:
                errors.append(f"{page.get('page_ref')}: coverage audit state missing")
            if not isinstance(hint_count, int) or hint_count < 0:
                errors.append(f"{page.get('page_ref')}: reviewed finding count is invalid")
            if target_review_state not in {
                None,
                "reviewed_no_acceptable_discrete_target",
                "candidate_inventory_not_established",
            }:
                errors.append(f"{page.get('page_ref')}: invalid M4 target review state")
            if not isinstance(target_relation_count, int) or target_relation_count < 0:
                errors.append(f"{page.get('page_ref')}: M4 target relation count is invalid")
        elif (
            audit_state is not None
            or hint_count != 0
            or target_review_state is not None
            or target_relation_count != 0
        ):
            errors.append(f"{page.get('page_ref')}: coverage audit fields require a contract")
    if sorted(page_memberships) != sorted(occurrence_ids):
        errors.append("every item occurrence must belong to exactly one catalog page")

    for row in occurrences:
        identifier = str(row.get("id"))
        for key in RECORD_CONTRACTS["mep_item_occurrence"]["required_fields"]:
            if key not in row:
                errors.append(f"{identifier}: missing required field {key}")
        if row.get("record_type") != "mep_item_occurrence":
            errors.append(f"{identifier}: record_type mismatch")
        if not isinstance(row.get("item_type"), str) or not row.get("item_type"):
            errors.append(f"{identifier}: item_type is required")
        if not isinstance(row.get("item_subtype"), str) or not row.get("item_subtype"):
            errors.append(f"{identifier}: item_subtype is required")
        if not isinstance(row.get("description"), str) or not row.get("description"):
            errors.append(f"{identifier}: description is required")
        if row.get("system") is not None and not isinstance(row.get("system"), Mapping):
            errors.append(f"{identifier}: system must be explicit null or an object")
        source = row.get("source", {})
        if str(source.get("page_ref")) not in page_ref_set:
            errors.append(f"{identifier}: source page is not registered")
        if not isinstance(source.get("pdf_page_number"), int):
            errors.append(f"{identifier}: PDF page number is required")
        if "drawing_sheet_number" not in source:
            errors.append(f"{identifier}: drawing sheet number field is required")
        for key in ("manufacturer", "model_number", "specification_reference", "mounting_type"):
            if key not in row or (row.get(key) is not None and not isinstance(row.get(key), str)):
                errors.append(f"{identifier}: {key} must be explicit null or string")
        counts = row.get("counts", {})
        if counts.get("observed_occurrence_count") != 1:
            errors.append(f"{identifier}: observed occurrence count must be one")
        if counts.get("deduplicated_projected_count") is not None:
            errors.append(f"{identifier}: M7A cannot establish deduplicated projected count")
        if counts.get("physical_instance_count") is not None:
            errors.append(f"{identifier}: M7A cannot establish physical instance count")
        if not row.get("evidence_refs"):
            errors.append(f"{identifier}: evidence refs are required")
        if not row.get("unresolved_reasons"):
            errors.append(f"{identifier}: unresolved reasons are required")
        authority = row.get("authority", {})
        for key in _FALSE_AUTHORITY_KEYS:
            if authority.get(key) is not False:
                errors.append(f"{identifier}: authority.{key} must be explicit false")
        for dimension in row.get("typed_dimensions", []):
            if not _finite_or_none(dimension.get("value")):
                errors.append(f"{identifier}: invalid typed dimension value")
        for elevation in row.get("elevations", []):
            if not _finite_or_none(elevation.get("value")):
                errors.append(f"{identifier}: invalid elevation value")

    physical_items = list(payload.get("physical_items", []))
    physical_ids = [str(row.get("id")) for row in physical_items]
    if len(physical_ids) != len(set(physical_ids)):
        errors.append("physical item IDs must be unique")
    for row in physical_items:
        identifier = str(row.get("id"))
        if m7b_ref is None:
            errors.append(f"{identifier}: physical item requires an M7B contract")
        if row.get("record_type") != "mep_physical_item":
            errors.append(f"{identifier}: physical item record_type mismatch")
        refs = [str(ref) for ref in row.get("item_occurrence_refs", [])]
        if not refs or any(ref not in occurrence_by_id for ref in refs):
            errors.append(f"{identifier}: physical item occurrence refs are invalid")
        if not row.get("identity_certificate_refs"):
            errors.append(f"{identifier}: physical identity certificate refs are required")
        counts = row.get("counts", {})
        if counts.get("observed_occurrence_count") != len(refs):
            errors.append(f"{identifier}: observed physical-item count mismatch")
        if counts.get("deduplicated_projected_count") != 1:
            errors.append(f"{identifier}: deduplicated projected count must be one")
        if counts.get("physical_instance_count") != 1:
            errors.append(f"{identifier}: physical instance count must be one")
        authority = row.get("authority", {})
        if authority.get("physical_item_identity_established") is not True:
            errors.append(f"{identifier}: physical identity authority missing")
        if authority.get("calculated_count_established") is not True:
            errors.append(f"{identifier}: calculated count authority missing")
        for key in ("installed_length_emitted", "approved_for_quote", "quantity_eligible"):
            if authority.get(key) is not False:
                errors.append(f"{identifier}: authority.{key} must remain false")

    calculated_discrete_counts = list(payload.get("calculated_discrete_counts", []))
    calculated_ids = [str(row.get("id")) for row in calculated_discrete_counts]
    if len(calculated_ids) != len(set(calculated_ids)):
        errors.append("calculated discrete count IDs must be unique")
    for row in calculated_discrete_counts:
        identifier = str(row.get("id"))
        for key in RECORD_CONTRACTS["mep_calculated_discrete_count"]["required_fields"]:
            if key not in row:
                errors.append(f"{identifier}: missing calculated count field {key}")
        if str(row.get("physical_item_ref")) not in set(physical_ids):
            errors.append(f"{identifier}: calculated count physical item ref is invalid")
        if row.get("value") != 1 or row.get("unit") != "ea" or row.get("state") != "calculated":
            errors.append(f"{identifier}: bounded discrete count must be 1 ea")
        if row.get("approved_for_quote") is not False:
            errors.append(f"{identifier}: approved_for_quote must remain false")

    lines = list(payload.get("takeoff_lines", []))
    line_ids = [str(row.get("id")) for row in lines]
    if len(line_ids) != len(set(line_ids)):
        errors.append("takeoff line IDs must be unique")
    for row in lines:
        identifier = str(row.get("id"))
        for key in RECORD_CONTRACTS["mep_takeoff_line"]["required_fields"]:
            if key not in row:
                errors.append(f"{identifier}: missing required field {key}")
        if row.get("record_type") != "mep_takeoff_line":
            errors.append(f"{identifier}: record_type mismatch")
        if str(row.get("item_occurrence_ref")) not in occurrence_by_id:
            errors.append(f"{identifier}: unknown item occurrence ref")
        channels = row.get("value_channels", {})
        for name in RECORD_CONTRACTS["mep_takeoff_line"]["value_channels"]:
            channel = channels.get(name, {})
            if channel.get("value") is not None or channel.get("unit") is not None:
                errors.append(f"{identifier}: {name} value must remain null at M7A")
            if channel.get("state") != "unknown" or not channel.get("reason"):
                errors.append(f"{identifier}: {name} unresolved state missing")
        authority = row.get("authority", {})
        for key in _FALSE_AUTHORITY_KEYS:
            if authority.get(key) is not False:
                errors.append(f"{identifier}: authority.{key} must be explicit false")
    if len(lines) != len(occurrences):
        errors.append("M7A requires one unresolved takeoff line per item occurrence")

    summary = payload.get("summary", {})
    expected = {
        "page_count": len(pages),
        "page_with_item_count": sum(bool(row.get("item_occurrence_refs")) for row in pages),
        "item_occurrence_count": len(occurrences),
        "routed_material_occurrence_count": sum(row.get("item_type") == "routed_material" for row in occurrences),
        "physical_item_count": len(physical_items),
        "takeoff_line_count": len(lines),
        "calculated_value_count": len(calculated_discrete_counts),
        "declared_value_count": 0,
        "coverage_abstention_count": (
            m7b_ref.get("coverage_abstention_count") if m7b_ref is not None else 0
        ),
    }
    if m7b_ref is None and expected["coverage_abstention_count"] != 0:
        errors.append("coverage abstentions require an M7B contract")
    if not isinstance(expected["coverage_abstention_count"], int) or expected["coverage_abstention_count"] < 0:
        errors.append("coverage_abstention_count must be a nonnegative integer")
    if summary != expected:
        errors.append("summary mismatch")
    contract = payload.get("exchange_contract", {})
    for key in false_flags:
        if contract.get(key) is not False:
            errors.append(f"exchange_contract.{key} must be explicit false")
    integrated = m7b_ref is not None
    if contract.get("m7b_bounded_discrete_counts_integrated") is not integrated:
        errors.append("M7B integration flag mismatch")
    if contract.get("physical_count_emitted") is not bool(physical_items):
        errors.append("physical_count_emitted flag mismatch")
    if contract.get("calculated_quantity_emitted") is not bool(calculated_discrete_counts):
        errors.append("calculated_quantity_emitted flag mismatch")
    return errors


def flatten_mep_item_catalog(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return one flat review row per item, plus one explicit row per empty page."""

    errors = validate_mep_item_catalog(payload)
    if errors:
        raise ValueError("invalid M7A payload:\n" + "\n".join(errors))
    occurrences = {
        str(row.get("id")): row for row in payload.get("item_occurrences", [])
    }
    physical_items = list(payload.get("physical_items", []))
    physical_by_occurrence = {
        str(ref): row
        for row in physical_items
        for ref in row.get("item_occurrence_refs", [])
    }
    calculated_by_physical = {
        str(row.get("physical_item_ref")): row
        for row in payload.get("calculated_discrete_counts", [])
    }
    coverage_ref = payload.get("coverage_audit_contract_ref") or {}
    m7b_ref = payload.get("m7b_contract_ref") or {}
    rows = []
    for page in payload.get("pages", []):
        refs = [str(ref) for ref in page.get("item_occurrence_refs", [])]
        source = {
            "pdf_page_number": page.get("pdf_page_number"),
            "page_ref": page.get("page_ref"),
            "drawing_sheet_number": page.get("drawing_sheet_number"),
            "sheet_role": page.get("sheet_role"),
            "document_occurrence_completeness": coverage_ref.get(
                "document_occurrence_completeness"
            ),
            "m7b_coverage_abstention_count": m7b_ref.get(
                "coverage_abstention_count", 0
            ),
        }
        if not refs:
            rows.append({
                "row_type": "page_without_accepted_item_evidence",
                **source,
                "item_occurrence_id": None,
                "physical_item_id": None,
                "item_type": None,
                "item_subtype": None,
                "system": None,
                "description": None,
                "nominal_dimension": None,
                "physical_dimension": None,
                "elevation": None,
                "manufacturer": None,
                "model_number": None,
                "specification_reference": None,
                "mounting_type": None,
                "observed_occurrence_count": 0,
                "deduplicated_projected_count": None,
                "physical_instance_count": None,
                "calculated_value": None,
                "calculated_unit": None,
                "unresolved_reasons": "no_accepted_m4_item_relations_on_page",
                "evidence_refs": "",
            })
            continue
        for ref in refs:
            item = occurrences[ref]
            physical_item = physical_by_occurrence.get(str(item.get("id")))
            dimensions = item.get("typed_dimensions", [])
            nominal = next((row for row in dimensions if not row.get("physical_dimension")), None)
            physical = next((row for row in dimensions if row.get("physical_dimension")), None)
            elevations = item.get("elevations", [])
            rows.append({
                "row_type": "item_occurrence",
                **source,
                "item_occurrence_id": item.get("id"),
                "physical_item_id": (
                    physical_item.get("id") if physical_item is not None else None
                ),
                "item_type": item.get("item_type"),
                "item_subtype": item.get("item_subtype"),
                "system": (item.get("system") or {}).get("kind"),
                "description": item.get("description"),
                "nominal_dimension": (
                    f"{nominal.get('value')} {nominal.get('unit')}" if nominal else None
                ),
                "physical_dimension": (
                    f"{physical.get('value')} {physical.get('unit')}" if physical else None
                ),
                "elevation": " | ".join(
                    f"{row.get('elevation_type')}={row.get('value')} {row.get('unit')}"
                    for row in elevations
                ) or None,
                "manufacturer": item.get("manufacturer"),
                "model_number": item.get("model_number"),
                "specification_reference": item.get("specification_reference"),
                "mounting_type": item.get("mounting_type"),
                "observed_occurrence_count": item.get("counts", {}).get("observed_occurrence_count"),
                "deduplicated_projected_count": item.get("counts", {}).get("deduplicated_projected_count"),
                "physical_instance_count": item.get("counts", {}).get("physical_instance_count"),
                "calculated_value": None,
                "calculated_unit": None,
                "unresolved_reasons": " | ".join(item.get("unresolved_reasons", [])),
                "evidence_refs": " | ".join(item.get("evidence_refs", [])),
            })
    for physical_item in physical_items:
        member_occurrences = [
            occurrences[str(ref)] for ref in physical_item.get("item_occurrence_refs", [])
        ]
        first = member_occurrences[0]
        calculated = calculated_by_physical.get(str(physical_item.get("id")))
        rows.append({
            "row_type": "physical_item",
            "pdf_page_number": " | ".join(
                str(row.get("source", {}).get("pdf_page_number"))
                for row in member_occurrences
            ),
            "page_ref": " | ".join(
                str(row.get("source", {}).get("page_ref")) for row in member_occurrences
            ),
            "drawing_sheet_number": " | ".join(
                str(row.get("source", {}).get("drawing_sheet_number"))
                for row in member_occurrences
            ),
            "sheet_role": " | ".join(sorted({
                str(row.get("source", {}).get("sheet_role"))
                for row in member_occurrences
            })),
            "document_occurrence_completeness": coverage_ref.get(
                "document_occurrence_completeness"
            ),
            "m7b_coverage_abstention_count": m7b_ref.get(
                "coverage_abstention_count", 0
            ),
            "item_occurrence_id": " | ".join(
                str(row.get("id")) for row in member_occurrences
            ),
            "physical_item_id": physical_item.get("id"),
            "item_type": first.get("item_type"),
            "item_subtype": first.get("item_subtype"),
            "system": (first.get("system") or {}).get("kind"),
            "description": first.get("description"),
            "nominal_dimension": None,
            "physical_dimension": None,
            "elevation": None,
            "manufacturer": first.get("manufacturer"),
            "model_number": first.get("model_number"),
            "specification_reference": first.get("specification_reference"),
            "mounting_type": first.get("mounting_type"),
            "observed_occurrence_count": physical_item.get("counts", {}).get(
                "observed_occurrence_count"
            ),
            "deduplicated_projected_count": physical_item.get("counts", {}).get(
                "deduplicated_projected_count"
            ),
            "physical_instance_count": physical_item.get("counts", {}).get(
                "physical_instance_count"
            ),
            "calculated_value": calculated.get("value") if calculated else None,
            "calculated_unit": calculated.get("unit") if calculated else None,
            "unresolved_reasons": " | ".join(
                physical_item.get("unresolved_reasons", [])
            ),
            "evidence_refs": " | ".join(physical_item.get("evidence_refs", [])),
        })
    return rows
