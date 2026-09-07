"""Project frozen M7A records into shared takeoff roles without rewriting M7A."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from src.drawing_engine.disciplines.mep.mep_item_catalog import validate_mep_item_catalog
from src.drawing_engine.project.takeoff_intelligence import (
    ADAPTER_LAYER,
    SCHEMA_VERSION,
    canonical_sha256,
    stable_id,
    validate_takeoff_adapter,
)


def build_mep_takeoff_adapter(catalog: Mapping[str, Any]) -> dict[str, Any]:
    errors = validate_mep_item_catalog(catalog)
    if errors:
        raise ValueError("invalid M7A catalog:\n" + "\n".join(errors))
    document_key = str(catalog.get("document", {}).get("document_key") or "")
    occurrence_map = {}
    occurrences = []
    for source in catalog.get("item_occurrences", []):
        source_ref = str(source["id"])
        identifier = stable_id(
            "takeoff_occurrence", document_key, "mep", source_ref
        )
        occurrence_map[source_ref] = identifier
        occurrences.append({
            "record_type": "takeoff_occurrence",
            "record_version": SCHEMA_VERSION,
            "id": identifier,
            "discipline": "mep",
            "document_key": document_key,
            "item_type": source["item_type"],
            "item_subtype": source["item_subtype"],
            "description": source["description"],
            "source": deepcopy(source["source"]),
            "native_record_refs": [source_ref],
            "evidence_refs": sorted(set(map(str, source["evidence_refs"]))),
            "epistemic_state": source["epistemic_state"],
            "unresolved_reasons": deepcopy(source["unresolved_reasons"]),
        })
    physical_map = {}
    physical_items = []
    for source in catalog.get("physical_items", []):
        source_ref = str(source["id"])
        identifier = stable_id(
            "takeoff_physical_item", document_key, "mep", source_ref
        )
        physical_map[source_ref] = identifier
        physical_items.append({
            "record_type": "takeoff_physical_item",
            "record_version": SCHEMA_VERSION,
            "id": identifier,
            "discipline": "mep",
            "document_key": document_key,
            "occurrence_refs": [
                occurrence_map[str(ref)] for ref in source["item_occurrence_refs"]
            ],
            "native_record_refs": [source_ref],
            "evidence_refs": sorted(set(map(str, source["evidence_refs"]))),
            "epistemic_state": source["epistemic_state"],
        })
    calculated_lines = []
    for source in catalog.get("calculated_discrete_counts", []):
        source_ref = str(source["id"])
        native_physical_ref = str(source["physical_item_ref"])
        calculated_lines.append({
            "record_type": "takeoff_calculated_line",
            "record_version": SCHEMA_VERSION,
            "id": stable_id(
                "takeoff_calculated_line", document_key, "mep", source_ref
            ),
            "discipline": "mep",
            "document_key": document_key,
            "physical_item_ref": physical_map[native_physical_ref],
            "item_type": source["category"],
            "metric_kind": "physical_count",
            "value": source["value"],
            "unit": source["unit"],
            "scope_ref": native_physical_ref,
            "native_record_refs": [source_ref],
            "evidence_refs": sorted(set(map(str, source["evidence_refs"]))),
            "epistemic_state": "calculated",
        })
    native_records = {
        "item_occurrences": deepcopy(catalog.get("item_occurrences", [])),
        "physical_items": deepcopy(catalog.get("physical_items", [])),
        "takeoff_lines": deepcopy(catalog.get("takeoff_lines", [])),
        "calculated_discrete_counts": deepcopy(
            catalog.get("calculated_discrete_counts", [])
        ),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "layer": ADAPTER_LAYER,
        "adapter": {
            "name": "mep_m7a_passthrough_adapter",
            "version": "1.0.0",
            "discipline": "mep",
        },
        "native_payload_ref": {
            "layer": catalog.get("layer"),
            "schema_version": catalog.get("schema_version"),
            "payload_sha256": canonical_sha256(catalog),
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
            "m7a_records_reused_without_mutation": True,
            "unresolved_m7a_takeoff_lines_not_promoted": True,
            "installed_route_length_not_derived": True,
        },
    }
    errors = validate_takeoff_adapter(payload)
    if errors:
        raise ValueError("invalid MEP takeoff adapter:\n" + "\n".join(errors))
    return payload
