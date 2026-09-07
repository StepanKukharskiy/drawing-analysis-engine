"""Promote one accepted Slice 2 prism into the existing Step 4 solid kernel.

Only Slice 2-accepted identity and placement evidence is consumed.  This
adapter assigns the physical component and transform IDs, builds the already
certified orthogonal prism, and replays the existing watertightness, interface,
analytic-volume, and two-view projection gates.  It does not write an
estimation quantity or classify the component material.
"""

from __future__ import annotations

from hashlib import sha256
import math
from typing import Any, Mapping

from src.drawing_engine.disciplines.concrete.generic_profile_extrusion_solver import _extruded_mesh
from src.drawing_engine.disciplines.concrete.multi_component_solid_kernel import solve_multi_component_solid
from src.drawing_engine.disciplines.concrete.unsigned_bounded_sweep_certificate import (
    validate_unsigned_bounded_sweep_certificate,
)


SCHEMA_VERSION = "0.1.0"


def _stable_id(kind: str, page_number: int, *parts: Any) -> str:
    encoded = "\0".join((str(page_number), *(str(part) for part in parts))).encode("utf-8")
    return f"{kind}.page_{page_number:04d}.evidence_{sha256(encoded).hexdigest()[:16]}"


def _abstain(page_number: int, reason_code: str, refs: list[str]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "physical_component_step4_replay",
        "page": page_number,
        "status": "insufficient_constraints",
        "reason_code": reason_code,
        "physical_components": [],
        "physical_component_transforms": [],
        "kernel_result": None,
        "solid_preview": None,
        "gross_envelope_volume": None,
        "evidence_refs": sorted(set(refs)),
        "contract": {
            "slice2_acceptance_required": True,
            "step4_kernel_invoked": False,
            "quantity_writes_allowed": False,
            "material_classification_inferred": False,
            "schedule_values_used": False,
        },
    }


def replay_single_prism_step4(
    hypothesis_generation: Mapping[str, Any],
    placement_evidence: Mapping[str, Any],
    physical_component_reclosure: Mapping[str, Any],
    view_frame_graph: Mapping[str, Any],
    *,
    page_number: int,
) -> dict[str, Any]:
    """Invoke Step 4 for exactly one accepted orthogonal prism component."""

    accepted = [
        item
        for item in physical_component_reclosure.get("certificates", []) or []
        if item.get("status") == "reclosed_pass" and item.get("slice3_input_eligible") is True
    ]
    if len(accepted) != 1 or placement_evidence.get("status") != "resolved":
        return _abstain(
            page_number,
            "unique_slice2_component_unresolved",
            [str(item.get("id")) for item in accepted if item.get("id")],
        )
    certificate = accepted[0]
    hypothesis_ref = str(certificate.get("hypothesis_ref") or "")
    hypotheses = {
        str(item.get("id")): item
        for item in hypothesis_generation.get("hypotheses", []) or []
        if item.get("id") is not None
    }
    hypothesis = hypotheses.get(hypothesis_ref)
    placements = [
        item
        for item in placement_evidence.get("component_placements", []) or []
        if item.get("state") == "resolved" and str(item.get("hypothesis_ref")) == hypothesis_ref
    ]
    groupings = [
        item
        for item in placement_evidence.get("component_groupings", []) or []
        if item.get("state") == "resolved"
        and hypothesis_ref in set(map(str, item.get("hypothesis_refs", []) or []))
    ]
    if hypothesis is None or len(placements) != 1 or len(groupings) != 1:
        return _abstain(page_number, "slice2_promotion_evidence_incomplete", [hypothesis_ref])
    grouping, placement = groupings[0], placements[0]
    interfaces = grouping.get("interfaces", {}) or {}
    if (
        interfaces.get("state") != "resolved"
        or interfaces.get("completeness") != "complete"
        or interfaces.get("no_external_interfaces_required") is not True
        or interfaces.get("records", [])
    ):
        return _abstain(page_number, "single_component_interface_certificate_missing", [hypothesis_ref])

    supporting_ref = str(placement.get("supporting_projection_hypothesis_ref") or "")
    supporting = hypotheses.get(supporting_ref)
    if supporting is None:
        return _abstain(page_number, "supporting_projection_missing", [hypothesis_ref, supporting_ref])
    rows = [hypothesis, supporting]
    by_role = {str(item.get("projection_role")): item for item in rows}
    if set(by_role) != {"parent_projection", "section_projection"}:
        return _abstain(page_number, "orthogonal_projection_roles_unresolved", [hypothesis_ref, supporting_ref])

    relation_ref = str(placement.get("relation_ref") or "")
    relations = {
        str(item.get("id")): item
        for item in view_frame_graph.get("relations", []) or []
        if item.get("id") is not None
    }
    relation = relations.get(relation_ref)
    scopes = {
        str(item.get("id")): item
        for item in view_frame_graph.get("object_scopes", []) or []
        if item.get("id") is not None
    }
    scope = scopes.get(str(hypothesis.get("physical_scope_ref") or ""))
    coordinate_ref = str((scope or {}).get("shared_coordinate_scope_id") or "")
    coordinates = {
        str(item.get("id")): item
        for item in view_frame_graph.get("shared_coordinate_system", {}).get("scopes", []) or []
        if item.get("id") is not None
    }
    coordinate = coordinates.get(coordinate_ref)
    if relation is None or coordinate is None or coordinate.get("relative_orientation_state") != "resolved":
        return _abstain(
            page_number,
            "resolved_scope_orientation_missing",
            [hypothesis_ref, relation_ref, coordinate_ref],
        )
    mappings = {
        str(item.get("view_id")): item
        for item in coordinate.get("view_axis_mappings", []) or []
        if item.get("state") == "resolved_relative"
    }
    parent_view = str(relation.get("parent_view_id") or "")
    section_view = str(relation.get("section_view_id") or "")
    if parent_view not in mappings or section_view not in mappings:
        return _abstain(page_number, "two_view_axis_mapping_missing", [parent_view, section_view])

    extents: dict[str, list[float]] = {}
    for role, view_ref in (("parent_projection", parent_view), ("section_projection", section_view)):
        item = by_role[role]
        profile_refs = list(map(str, item.get("profile_refs", []) or []))
        metrics = (
            item.get("compatibility_certificate", {})
            .get("metric_extents", {})
            .get("profiles", {})
        ) or {}
        if len(profile_refs) != 1 or profile_refs[0] not in metrics:
            return _abstain(page_number, "unique_metric_profile_missing", [str(item.get("id"))])
        metric = metrics[profile_refs[0]]
        mapping = mappings[view_ref]
        for display_axis, value_key in (("u", "width_mm"), ("v", "height_mm")):
            object_axis, value = str(mapping.get(display_axis) or ""), metric.get(value_key)
            if (
                object_axis not in {"X", "Y", "Z"}
                or not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                return _abstain(page_number, "metric_axis_extent_invalid", [profile_refs[0]])
            extents.setdefault(object_axis, []).append(float(value))
    if set(extents) != {"X", "Y", "Z"} or any(
        max(values) - min(values) > max(2.0, 0.015 * max(values))
        for values in extents.values()
    ):
        return _abstain(page_number, "three_axis_metric_extent_reclosure_failed", [hypothesis_ref, supporting_ref])
    # The existing extrusion helper serializes vertices at 0.0001 mm.  Use the
    # same precision for the analytic record so mesh/analytic replay compares
    # like with like rather than reporting a serialization residual.
    dimensions = {
        axis: round(sum(values) / len(values), 4) for axis, values in extents.items()
    }

    relative_transform = placement.get("relative_transform", {}) or {}
    offsets = relative_transform.get("offsets_mm", {}) or {}
    signs = relative_transform.get("axis_signs", {}) or {}
    unsigned_certificate = placement.get("unsigned_bounded_sweep_certificate") or {}
    unsigned_mode = bool(
        unsigned_certificate.get("status") == "accepted"
        and not validate_unsigned_bounded_sweep_certificate(unsigned_certificate)
        and relative_transform.get("state") == "resolved_up_to_reflection"
        and str(unsigned_certificate.get("hypothesis_ref") or "") == hypothesis_ref
        and str(unsigned_certificate.get("physical_scope_ref") or "")
        == str(hypothesis.get("physical_scope_ref") or "")
    )
    if unsigned_mode and any(
        axis not in unsigned_certificate.get("extents_by_axis_mm", {})
        or abs(
            float(unsigned_certificate["extents_by_axis_mm"][axis])
            - float(dimensions[axis])
        )
        > max(2.0, 0.015 * float(dimensions[axis]))
        for axis in ("X", "Y", "Z")
    ):
        return _abstain(
            page_number,
            "unsigned_bounded_sweep_extent_contradiction",
            [str(placement.get("id")), str(unsigned_certificate.get("id"))],
        )
    canonical_signs = relative_transform.get("canonical_axis_signs", {}) or {}
    if (
        set(offsets) != {"X", "Y", "Z"}
        or set(signs) != {"X", "Y", "Z"}
        or (
            any(signs[axis] != 1 for axis in signs)
            if not unsigned_mode
            else any(signs[axis] is not None for axis in signs)
            or canonical_signs != {"X": 1, "Y": 1, "Z": 1}
        )
    ):
        return _abstain(page_number, "proper_component_transform_unresolved", [str(placement.get("id"))])

    evidence_refs = sorted(
        {
            hypothesis_ref,
            supporting_ref,
            str(certificate.get("id") or ""),
            str(placement.get("id") or ""),
            str(grouping.get("id") or ""),
            relation_ref,
            coordinate_ref,
            *map(str, placement.get("evidence_refs", []) or []),
        }
        - {""}
    )
    component_ref = _stable_id("physical_component", page_number, hypothesis_ref, supporting_ref)
    transform_ref = _stable_id("physical_component_transform", page_number, component_ref, *evidence_refs)
    shared_scope_origin = [float(offsets[axis]) for axis in ("X", "Y", "Z")]
    # A single component has free global translation.  Replay it in a local
    # origin gauge for numerical stability while retaining the independently
    # certified shared-scope position as explicit transform evidence.
    origin = [0.0, 0.0, 0.0]
    mesh = _extruded_mesh(
        [(0.0, 0.0), (dimensions["X"], 0.0), (dimensions["X"], dimensions["Z"]), (0.0, dimensions["Z"])],
        dimensions["Y"],
    )
    transform = {
        "id": transform_ref,
        "record_type": "physical_component_transform",
        "record_version": SCHEMA_VERSION,
        "component_ref": component_ref,
        "state": "canonical_relative_preview" if unsigned_mode else "resolved",
        "placement_role": "canonical_relative_preview" if unsigned_mode else "physical",
        "origin_xyz_mm": origin,
        "shared_scope_origin_xyz_mm": shared_scope_origin,
        "local_axes_xyz": {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]},
        "axis_signs": "unresolved_reflection" if unsigned_mode else "resolved",
        "orientation_sensitivity": "sign_invariant" if unsigned_mode else "orientation_sensitive",
        "unsigned_bounded_sweep_certificate_ref": (
            unsigned_certificate.get("id") if unsigned_mode else None
        ),
        "evidence_refs": evidence_refs,
    }
    volume_mm3 = dimensions["X"] * dimensions["Y"] * dimensions["Z"]
    component = {
        "id": component_ref,
        "mesh": {
            "vertices_xyz_mm": mesh["vertices_xyz_mm"],
            "triangles": mesh["triangles"],
        },
        "transform": transform,
        "analytic_volume": {
            "value_mm3": volume_mm3,
            "basis": "reclosed_orthogonal_profile_extents",
            "evidence_refs": evidence_refs,
        },
        "evidence_refs": evidence_refs,
    }
    unit = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}
    supplied_views = []
    for view_ref in (parent_view, section_view):
        mapping = mappings[view_ref]
        u_axis, v_axis = str(mapping["u"]), str(mapping["v"])
        u0, v0 = 0.0, 0.0
        du, dv = dimensions[u_axis], dimensions[v_axis]
        supplied_views.append(
            {
                "id": view_ref,
                "origin_xyz_mm": [0, 0, 0],
                "u_axis_xyz": unit[u_axis],
                "v_axis_xyz": unit[v_axis],
                "tolerance_mm": 0.5,
                "evidence_refs": evidence_refs,
                "component_projections": [
                    {
                        "component_ref": component_ref,
                        "polygons_uv_mm": [
                            [[u0, v0], [u0 + du, v0], [u0 + du, v0 + dv], [u0, v0 + dv]]
                        ],
                        "evidence_refs": evidence_refs,
                    }
                ],
            }
        )
    try:
        kernel_result = solve_multi_component_solid([component], [], supplied_views)
    except (ValueError, TypeError, KeyError) as error:
        result = _abstain(page_number, "step4_kernel_replay_failed", evidence_refs)
        result["status"] = "reclosed_fail"
        result["reason"] = str(error)
        return result

    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "physical_component_step4_replay",
        "page": page_number,
        "status": "reclosed_pass",
        "reason_code": None,
        "physical_components": [
            {
                "id": component_ref,
                "source_hypothesis_ref": hypothesis_ref,
                "supporting_projection_hypothesis_ref": supporting_ref,
                "profile_refs": list(map(str, hypothesis.get("profile_refs", []) or [])),
                "state": "canonical_relative_preview" if unsigned_mode else "resolved",
                "evidence_refs": evidence_refs,
            }
        ],
        "physical_component_transforms": [transform],
        "kernel_result": kernel_result,
        "solid_preview": {
            "mesh": kernel_result["assembly_mesh"],
            "source": (
                "accepted_unsigned_bounded_sweep_replayed_in_canonical_relative_gauge"
                if unsigned_mode
                else "accepted_slice2_component_replayed_through_step4_kernel"
            ),
            "validation": {
                "components": kernel_result["components"],
                "interfaces": kernel_result["shared_interfaces"],
                "overlaps": kernel_result["pair_validations"],
                "volume": kernel_result["volume_validation"],
                "reprojections": kernel_result["supplied_view_reprojections"],
            },
        },
        "gross_envelope_volume": {
            "state": "derived",
            "classification": "generic_gross_envelope",
            "value_mm3": volume_mm3,
            "value_m3": volume_mm3 / 1_000_000_000.0,
            "orientation_invariance": "certified" if unsigned_mode else "signed_orientation_resolved",
            "evidence_refs": evidence_refs,
        },
        "evidence_refs": evidence_refs,
        "contract": {
            "slice2_acceptance_required": True,
            "step4_kernel_invoked": True,
            "quantity_writes_allowed": False,
            "material_classification_inferred": False,
            "schedule_values_used": False,
            "signed_physical_placement_resolved": not unsigned_mode,
            "canonical_relative_preview_only": unsigned_mode,
        },
    }
