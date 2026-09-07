"""Adapt accepted drawing-calculated concrete objects to the shared contract."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from src.drawing_engine.disciplines.concrete.physical_object_quantity_promotion import (
    aggregate_calculated_concrete_quantities,
)
from src.drawing_engine.project.takeoff_intelligence import (
    ADAPTER_LAYER,
    SCHEMA_VERSION,
    canonical_sha256,
    stable_id,
    validate_takeoff_adapter,
)


def build_concrete_takeoff_adapter(
    engineering_graph: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish only accepted physical-object volumes; schedules are never read."""

    document_key = str(engineering_graph.get("document_key") or "")
    if not document_key:
        raise ValueError("engineering graph document_key is required")
    occurrences = []
    physical_items = []
    calculated_lines = []
    native_records = []
    seen_object_refs = set()
    for page_index, page in enumerate(engineering_graph.get("pages", []), start=1):
        promotion = page.get("physical_object_quantity_promotion") or {}
        if promotion.get("status") != "accepted":
            continue
        contract = promotion.get("contract") or {}
        if (
            promotion.get("quantity_eligible") is not True
            or contract.get("one_physical_object_per_equivalence_class") is not True
            or contract.get("construction_regions_are_never_aggregated_separately") is not True
            or contract.get("equivalence_members_are_never_aggregated_separately") is not True
            or contract.get("schedule_values_used") is not False
        ):
            raise ValueError("concrete promotion authority contract is not closed")
        physical = promotion.get("accepted_physical_object") or {}
        if physical.get("state") != "accepted" or physical.get("quantity_eligible") is not True:
            raise ValueError("accepted concrete promotion lacks one physical object")
        object_ref = str(physical.get("id") or "")
        scope_ref = str(physical.get("physical_object_scope_ref") or "")
        if not object_ref or not scope_ref or object_ref in seen_object_refs:
            raise ValueError("concrete physical object identity is missing or duplicated")
        quantities = aggregate_calculated_concrete_quantities(
            promotion.get("calculated_concrete_quantities", [])
        )
        object_quantities = [
            row for row in quantities if str(row.get("physical_object_ref")) == object_ref
        ]
        if len(object_quantities) != 1 or len(quantities) != 1:
            raise ValueError("concrete adapter requires one unique quantity per promoted object")
        quantity = object_quantities[0]
        if str(quantity.get("physical_object_scope_ref")) != scope_ref:
            raise ValueError("concrete quantity scope does not match the physical object")
        page_number = int(promotion.get("page") or page_index)
        evidence_refs = sorted({
            *map(str, physical.get("evidence_refs", [])),
            *map(str, quantity.get("evidence_refs", [])),
            object_ref,
            scope_ref,
            str(quantity.get("id")),
        })
        occurrence_id = stable_id(
            "takeoff_occurrence", document_key, "concrete", object_ref
        )
        physical_item_id = stable_id(
            "takeoff_physical_item", document_key, "concrete", object_ref
        )
        occurrences.append({
            "record_type": "takeoff_occurrence",
            "record_version": SCHEMA_VERSION,
            "id": occurrence_id,
            "discipline": "concrete",
            "document_key": document_key,
            "item_type": "concrete",
            "item_subtype": "accepted_constructive_union",
            "description": "Drawing-calculated concrete physical object",
            "source": {
                "pdf_page_number": page_number,
                "page_ref": f"page.{page_number}",
                "drawing_sheet_number": None,
            },
            "native_record_refs": [object_ref, scope_ref],
            "evidence_refs": evidence_refs,
            "epistemic_state": "derived",
            "unresolved_reasons": (
                ["absolute_orientation_unresolved"]
                if physical.get("geometry", {}).get("absolute_orientation_resolved")
                is False
                else []
            ),
        })
        physical_items.append({
            "record_type": "takeoff_physical_item",
            "record_version": SCHEMA_VERSION,
            "id": physical_item_id,
            "discipline": "concrete",
            "document_key": document_key,
            "occurrence_refs": [occurrence_id],
            "native_record_refs": [object_ref, scope_ref],
            "evidence_refs": evidence_refs,
            "epistemic_state": "derived",
        })
        calculated_lines.append({
            "record_type": "takeoff_calculated_line",
            "record_version": SCHEMA_VERSION,
            "id": stable_id(
                "takeoff_calculated_line", document_key, quantity.get("id"), "net_volume"
            ),
            "discipline": "concrete",
            "document_key": document_key,
            "physical_item_ref": physical_item_id,
            "item_type": "concrete",
            "metric_kind": "net_volume",
            "value": quantity["net_concrete_m3"],
            "unit": "m3",
            "scope_ref": scope_ref,
            "native_record_refs": [str(quantity["id"])],
            "evidence_refs": evidence_refs,
            "epistemic_state": "calculated",
        })
        native_records.extend([deepcopy(physical), deepcopy(quantity)])
        seen_object_refs.add(object_ref)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "layer": ADAPTER_LAYER,
        "adapter": {
            "name": "accepted_concrete_physical_object_adapter",
            "version": "1.0.0",
            "discipline": "concrete",
        },
        "native_payload_ref": {
            "layer": "object_agnostic_engineering_graph",
            "payload_sha256": canonical_sha256(engineering_graph),
        },
        "native_records": native_records,
        "native_records_preserved_unchanged": True,
        "occurrences": occurrences,
        "physical_items": physical_items,
        "calculated_lines": calculated_lines,
        "declared_lines": [],
        "contract": {
            "calculated_and_declared_channels_separate": True,
            "missing_values_not_substituted_with_zero": True,
            "approval_not_inferred": True,
            "accepted_physical_object_required": True,
            "one_quantity_per_physical_object": True,
            "construction_regions_not_counted_separately": True,
            "equivalence_members_not_counted_separately": True,
            "schedule_values_used": False,
        },
    }
    errors = validate_takeoff_adapter(payload)
    if errors:
        raise ValueError("invalid concrete takeoff adapter:\n" + "\n".join(errors))
    return payload
