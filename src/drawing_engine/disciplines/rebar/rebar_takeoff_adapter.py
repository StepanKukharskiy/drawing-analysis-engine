"""Adapt frozen rebar results to the discipline-neutral takeoff contract.

The adapter republishes existing closure decisions; it does not solve family
identity, multiplicity, fabrication geometry, diameter, grade, or mass.  In
particular, profile estimates and convention-dependent mass stay outside the
calculated channel.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping

from src.drawing_engine.project.takeoff_intelligence import (
    ADAPTER_LAYER,
    SCHEMA_VERSION,
    canonical_sha256,
    stable_id,
    validate_takeoff_adapter,
)


_CLOSED_STATES = {"direct", "observed", "derived"}


def _positive(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _refs(*groups: object) -> list[str]:
    values = []
    for group in groups:
        if group is None:
            continue
        if isinstance(group, (list, tuple, set)):
            values.extend(str(value) for value in group if value is not None)
        else:
            values.append(str(group))
    return sorted(set(values))


def _source(page_number: int) -> dict[str, Any]:
    return {"pdf_page": page_number, "drawing_sheet": None}


def _occurrence(
    *,
    document_key: str,
    page_number: int,
    native_ref: str,
    native_id: str,
    mark: object,
    subtype: str,
    evidence_refs: list[str],
    epistemic_state: str,
    unresolved_reasons: list[str],
) -> dict[str, Any]:
    identifier = stable_id(
        "takeoff_occurrence", document_key, page_number, native_ref
    )
    return {
        "record_type": "takeoff_occurrence",
        "record_version": SCHEMA_VERSION,
        "id": identifier,
        "discipline": "rebar",
        "document_key": document_key,
        "item_type": "reinforcement_bar",
        "item_subtype": subtype,
        "description": (
            f"Reinforcement bar family {mark}"
            if mark is not None
            else "Unresolved reinforcement observations"
        ),
        "source": _source(page_number),
        "native_record_refs": [native_ref],
        "evidence_refs": evidence_refs or [native_id],
        "epistemic_state": epistemic_state,
        "unresolved_reasons": _refs(unresolved_reasons),
        "mark": None if mark is None else str(mark),
    }


def _physical_item(
    *,
    document_key: str,
    occurrence: Mapping[str, Any],
    native_ref: str,
    evidence_refs: list[str],
    subtype: str,
    mark: object,
    object_scope_ref: str | None,
    count: object,
    count_state: object,
    diameter_mm: object,
    diameter_state: object,
    steel_grade: object = None,
    steel_grade_state: object = "unknown",
) -> dict[str, Any]:
    identifier = stable_id(
        "takeoff_physical_item", occurrence["id"], object_scope_ref, mark
    )
    return {
        "record_type": "takeoff_physical_item",
        "record_version": SCHEMA_VERSION,
        "id": identifier,
        "discipline": "rebar",
        "document_key": document_key,
        "occurrence_refs": [occurrence["id"]],
        "native_record_refs": [native_ref],
        "evidence_refs": evidence_refs,
        "epistemic_state": "derived",
        "item_type": "reinforcement_bar",
        "item_subtype": subtype,
        "mark": None if mark is None else str(mark),
        "object_scope_ref": object_scope_ref,
        "physical_count": {"value": count, "state": count_state},
        "diameter_mm": {"value": diameter_mm, "state": diameter_state or "unknown"},
        "steel_grade": {"value": steel_grade, "state": steel_grade_state or "unknown"},
    }


def _calculated_line(
    *,
    document_key: str,
    physical_item: Mapping[str, Any],
    metric_kind: str,
    value: float | int,
    unit: str,
    native_refs: list[str],
    evidence_refs: list[str],
) -> dict[str, Any]:
    return {
        "record_type": "takeoff_calculated_line",
        "record_version": SCHEMA_VERSION,
        "id": stable_id(
            "takeoff_calculated_line", physical_item["id"], metric_kind, unit
        ),
        "discipline": "rebar",
        "document_key": document_key,
        "physical_item_ref": physical_item["id"],
        "item_type": "reinforcement_bar",
        "metric_kind": metric_kind,
        "value": value,
        "unit": unit,
        "scope_ref": physical_item["id"],
        "native_record_refs": native_refs,
        "evidence_refs": evidence_refs,
        "epistemic_state": "calculated",
    }


def build_rebar_takeoff_adapter(engineering_graph: Mapping[str, Any]) -> dict[str, Any]:
    """Publish only already-closed rebar identity and strict quantity rows."""

    document_key = engineering_graph.get("document_key")
    if not isinstance(document_key, str) or not document_key:
        raise ValueError("rebar adapter requires an engineering document_key")
    pages = engineering_graph.get("pages")
    if not isinstance(pages, list):
        raise ValueError("rebar adapter requires engineering pages")

    occurrences: list[dict[str, Any]] = []
    physical_items: list[dict[str, Any]] = []
    calculated_lines: list[dict[str, Any]] = []
    native_records: list[dict[str, Any]] = []

    for page_index, page in enumerate(pages):
        page_occurrence_start = len(occurrences)
        page_number = int(page.get("page") or page_index + 1)
        program = page.get("rebar_program") or {}
        native = program.get("native_vector_detail_linking") or {}
        families = list(native.get("physical_families") or [])
        native_records.append({
            "page_number": page_number,
            "physical_families": deepcopy(families),
            "quantity_families": deepcopy(
                (page.get("reinforcement_quantities") or {}).get("families") or []
            ),
            "quantity_elements": deepcopy(
                (page.get("reinforcement_quantities") or {}).get("elements") or []
            ),
            "native_vector_detail_linking_status": native.get("status"),
            "reinforcement_quantities_status": (
                page.get("reinforcement_quantities") or {}
            ).get("status"),
        })
        groups_by_id = {
            str(group.get("id")): group for group in program.get("groups") or []
        }
        family_items: dict[str, dict[str, Any]] = {}

        for family_index, family in enumerate(families):
            family_id = str(family.get("id") or f"family.{family_index + 1}")
            native_ref = (
                f"pages[{page_index}].rebar_program.native_vector_detail_linking."
                f"physical_families[{family_index}]"
            )
            resolved = (
                family.get("constraint_status") == "resolved"
                and family.get("count_state") in _CLOSED_STATES
                and _positive(family.get("count"))
            )
            reasons = [] if resolved else _refs(
                family.get("constraint_reason"),
                (family.get("multiplicity") or {}).get("reason"),
            )
            if not resolved and not reasons:
                reasons = ["family_identity_or_multiplicity_unresolved"]
            evidence = _refs(
                family_id,
                family.get("detail_ids"),
                family.get("placement_association_ids"),
                family.get("fragment_ids"),
                family.get("component_ids"),
                family.get("view_ids"),
                family.get("identity_evidence_refs"),
                (family.get("multiplicity") or {}).get("evidence_refs"),
            )
            group = groups_by_id.get(str(family.get("source_group_id")))
            topology = (
                (((group or {}).get("topology") or {}).get("family") or {}).get(
                    "value"
                )
            )
            occurrence = _occurrence(
                document_key=document_key,
                page_number=page_number,
                native_ref=native_ref,
                native_id=family_id,
                mark=family.get("mark"),
                subtype=str(topology or family.get("state") or "bar_family"),
                evidence_refs=evidence,
                epistemic_state=(
                    "derived"
                    if family.get("state") == "derived_cross_view_family"
                    else "observed"
                ),
                unresolved_reasons=reasons,
            )
            occurrences.append(occurrence)
            if not resolved:
                continue
            diameter = (((group or {}).get("bar_spec") or {}).get("diameter_mm") or {})
            grade = (((group or {}).get("bar_spec") or {}).get("steel_grade") or {})
            physical = _physical_item(
                document_key=document_key,
                occurrence=occurrence,
                native_ref=native_ref,
                evidence_refs=evidence,
                subtype=str(topology or "bar_family"),
                mark=family.get("mark"),
                object_scope_ref=None,
                count=family.get("count"),
                count_state=family.get("count_state"),
                diameter_mm=diameter.get("value"),
                diameter_state=diameter.get("state"),
                steel_grade=grade.get("value"),
                steel_grade_state=grade.get("state"),
            )
            physical_items.append(physical)
            family_items[family_id] = physical

        quantities = page.get("reinforcement_quantities") or {}
        quantity_families = list(quantities.get("families") or [])
        for row_index, row in enumerate(quantity_families):
            family_id = str(row.get("physical_family_id") or "")
            physical = family_items.get(family_id)
            if physical is None:
                continue
            native_ref = f"pages[{page_index}].reinforcement_quantities.families[{row_index}]"
            evidence = _refs(row.get("evidence_refs"), family_id, row.get("group_id"))
            if row.get("count_state") in _CLOSED_STATES and _positive(
                row.get("count")
            ):
                calculated_lines.append(_calculated_line(
                    document_key=document_key,
                    physical_item=physical,
                    metric_kind="physical_count",
                    value=int(row["count"]),
                    unit="ea",
                    native_refs=[native_ref],
                    evidence_refs=evidence,
                ))
            if row.get("fabrication_state") == "resolved" and _positive(
                row.get("fabrication_length_total_mm")
            ):
                calculated_lines.append(_calculated_line(
                    document_key=document_key,
                    physical_item=physical,
                    metric_kind="fabrication_length",
                    value=round(float(row["fabrication_length_total_mm"]) / 1000.0, 9),
                    unit="m",
                    native_refs=[native_ref],
                    evidence_refs=evidence,
                ))
            mass_inputs_closed = (
                row.get("mass_state") in _CLOSED_STATES
                and row.get("diameter_state") in _CLOSED_STATES
                and row.get("steel_grade_state") in _CLOSED_STATES
                and _positive(row.get("mass_kg"))
            )
            if mass_inputs_closed:
                calculated_lines.append(_calculated_line(
                    document_key=document_key,
                    physical_item=physical,
                    metric_kind="mass",
                    value=float(row["mass_kg"]),
                    unit="kg",
                    native_refs=[native_ref],
                    evidence_refs=evidence,
                ))

        detail_elements = list(quantities.get("elements") or [])
        detail_contract = quantities.get("contract") or {}
        detail_validation = quantities.get("validation") or {}
        detail_takeoff_closed = (
            quantities.get("status") == "resolved_drawing_takeoff"
            and detail_validation.get("all_object_bindings_unique") is True
            and detail_validation.get("scene_path_count_matches_takeoff_count") is True
            and detail_contract.get("schedule_values_used") is False
        )
        if detail_takeoff_closed:
            for element_index, element in enumerate(detail_elements):
                native_ref = (
                    f"pages[{page_index}].reinforcement_quantities."
                    f"elements[{element_index}]"
                )
                native_id = f"{element.get('object_instance_id')}:{element.get('mark')}"
                evidence = _refs(
                    native_id,
                    element.get("object_instance_id"),
                    element.get("evidence_refs"),
                )
                occurrence = _occurrence(
                    document_key=document_key,
                    page_number=page_number,
                    native_ref=native_ref,
                    native_id=native_id,
                    mark=element.get("mark"),
                    subtype=str(element.get("role") or "bar_family"),
                    evidence_refs=evidence,
                    epistemic_state="derived",
                    unresolved_reasons=[],
                )
                occurrences.append(occurrence)
                physical = _physical_item(
                    document_key=document_key,
                    occurrence=occurrence,
                    native_ref=native_ref,
                    evidence_refs=evidence,
                    subtype=str(element.get("role") or "bar_family"),
                    mark=element.get("mark"),
                    object_scope_ref=str(element.get("object_instance_id")),
                    count=element.get("count"),
                    count_state="derived",
                    diameter_mm=element.get("diameter_mm"),
                    diameter_state=element.get("diameter_state"),
                )
                physical_items.append(physical)
                if (
                    detail_contract.get("counts_are_section_and_spacing_derived")
                    is True
                    and _positive(element.get("count"))
                ):
                    calculated_lines.append(_calculated_line(
                        document_key=document_key,
                        physical_item=physical,
                        metric_kind="physical_count",
                        value=int(element["count"]),
                        unit="ea",
                        native_refs=[native_ref],
                        evidence_refs=evidence,
                    ))
                if (
                    detail_contract.get("lengths_are_direct_detail_annotations")
                    is True
                    and _positive(element.get("total_length_m"))
                ):
                    calculated_lines.append(_calculated_line(
                        document_key=document_key,
                        physical_item=physical,
                        metric_kind="fabrication_length",
                        value=float(element["total_length_m"]),
                        unit="m",
                        native_refs=[native_ref],
                        evidence_refs=evidence,
                    ))
                strict_mass = (
                    element.get("diameter_state") in _CLOSED_STATES
                    and quantities.get("mass_status") == "resolved"
                    and detail_validation.get("all_diameters_resolved") is True
                    and detail_validation.get("all_diameter_conventions_closed")
                    is True
                    and _positive(element.get("mass_kg"))
                )
                if strict_mass:
                    calculated_lines.append(_calculated_line(
                        document_key=document_key,
                        physical_item=physical,
                        metric_kind="mass",
                        value=float(element["mass_kg"]),
                        unit="kg",
                        native_refs=[native_ref],
                        evidence_refs=evidence,
                    ))

        if len(occurrences) == page_occurrence_start:
            reason = (
                native.get("reason")
                or quantities.get("reason")
                or "no physical rebar family closed"
            )
            native_ref = f"pages[{page_index}].rebar_program.native_vector_detail_linking"
            occurrences.append(_occurrence(
                document_key=document_key,
                page_number=page_number,
                native_ref=native_ref,
                native_id=f"page.{page_number}.rebar_observations",
                mark=None,
                subtype="unresolved_observation_scope",
                evidence_refs=[native_ref],
                epistemic_state="unknown",
                unresolved_reasons=_refs(reason, program.get("unknowns")),
            ))

    payload = {
        "schema_version": SCHEMA_VERSION,
        "layer": ADAPTER_LAYER,
        "adapter": {
            "name": "object_agnostic_rebar",
            "version": "1.0.0",
            "discipline": "rebar",
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
        "summary": {
            "occurrence_count": len(occurrences),
            "physical_item_count": len(physical_items),
            "calculated_line_count": len(calculated_lines),
            "unresolved_occurrence_count": sum(bool(row["unresolved_reasons"]) for row in occurrences),
        },
        "contract": {
            "calculated_and_declared_channels_separate": True,
            "missing_values_not_substituted_with_zero": True,
            "approval_not_inferred": True,
            "profile_estimates_excluded": True,
            "convention_dependent_mass_excluded": True,
            "projected_geometry_is_not_fabrication_length": True,
        },
    }
    errors = validate_takeoff_adapter(payload)
    if errors:
        raise ValueError("invalid rebar takeoff adapter:\n" + "\n".join(errors))
    return payload
