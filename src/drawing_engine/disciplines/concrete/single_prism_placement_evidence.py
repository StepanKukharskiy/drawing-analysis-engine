"""Derive fail-closed Slice 2 evidence for one orthogonal prismatic component.

This is a deliberately narrow, drawing-neutral certificate producer.  It
recognises one parent projection and one section projection in an already
accepted physical scope.  The two profiles are evidence for one component,
never two additive occurrences.  Missing orientation, non-rectangular
topology, non-unique extent correspondence, or incomplete dimension support
causes an abstention.
"""

from __future__ import annotations

from hashlib import sha256
import math
from typing import Any, Iterable, Mapping

from src.drawing_engine.core.dimension_attachment import DimensionAttachment
from src.drawing_engine.disciplines.concrete.unsigned_bounded_sweep_certificate import certify_unsigned_bounded_sweep


SCHEMA_VERSION = "0.1.0"


def _stable_id(kind: str, page_number: int, *parts: Any) -> str:
    encoded = "\0".join((str(page_number), *(str(part) for part in parts))).encode("utf-8")
    return f"{kind}.page_{page_number:04d}.evidence_{sha256(encoded).hexdigest()[:16]}"


def _profiles(assembly: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(profile["id"]): profile
        for scope in assembly.get("scopes", []) or []
        for profile in scope.get("profiles", []) or []
        if isinstance(profile, Mapping) and profile.get("id") is not None
    }


def _metric(hypothesis: Mapping[str, Any]) -> tuple[str, float, float] | None:
    profile_refs = list(map(str, hypothesis.get("profile_refs", []) or []))
    metrics = (
        hypothesis.get("compatibility_certificate", {})
        .get("metric_extents", {})
        .get("profiles", {})
    ) or {}
    if len(profile_refs) != 1 or profile_refs[0] not in metrics:
        return None
    row = metrics[profile_refs[0]]
    width, height = row.get("width_mm"), row.get("height_mm")
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
        for value in (width, height)
    ):
        return None
    return profile_refs[0], float(width), float(height)


def _matching_axis_pair(
    parent: tuple[str, float, float],
    section: tuple[str, float, float],
) -> tuple[int, int] | None:
    matches = []
    for parent_axis in (0, 1):
        for section_axis in (0, 1):
            left, right = parent[parent_axis + 1], section[section_axis + 1]
            if abs(left - right) <= max(2.0, 0.015 * max(left, right)):
                matches.append((parent_axis, section_axis))
    return matches[0] if len(matches) == 1 else None


def _profile_dimension(
    profile: Mapping[str, Any],
    dimensions: Mapping[str, DimensionAttachment],
    value_mm: float,
) -> DimensionAttachment | None:
    candidates = [
        dimensions[ref]
        for ref in map(
            str,
            profile.get("dimension_certificate", {}).get("dimension_refs", []) or [],
        )
        if ref in dimensions
        and dimensions[ref].status == "accepted"
        and abs(float(dimensions[ref].value_mm) - value_mm)
        <= max(2.0, 0.015 * max(float(dimensions[ref].value_mm), value_mm))
    ]
    return candidates[0] if len(candidates) == 1 else None


def _abstention(page_number: int, reason_code: str, evidence_refs: Iterable[Any]) -> dict[str, Any]:
    refs = sorted({str(ref) for ref in evidence_refs if ref is not None and str(ref)})
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "single_prism_component_placement_evidence",
        "page": page_number,
        "status": "insufficient_constraints",
        "reason_code": reason_code,
        "component_groupings": [],
        "component_placements": [],
        "evidence_refs": refs,
        "contract": {
            "single_component_only": True,
            "separate_profiles_are_additive": False,
            "signed_orientation_required": False,
            "orientation_sensitive_placement_requires_signed_orientation": True,
            "unsigned_bounded_sweep_certificate_accepted": False,
            "quantity_eligible": False,
            "schedule_values_used": False,
        },
    }


def derive_single_prism_placement_evidence(
    hypothesis_generation: Mapping[str, Any],
    scoped_profile_assembly: Mapping[str, Any],
    view_frame_graph: Mapping[str, Any],
    dimensions: Iterable[DimensionAttachment],
    *,
    page_number: int,
) -> dict[str, Any]:
    """Return one complete grouping/placement record or a precise abstention."""

    hypotheses = hypothesis_generation.get("hypotheses", []) or []
    by_scope: dict[str, list[Mapping[str, Any]]] = {}
    for item in hypotheses:
        by_scope.setdefault(str(item.get("physical_scope_ref") or ""), []).append(item)
    scopes = {
        str(item.get("id")): item
        for item in view_frame_graph.get("object_scopes", []) or []
        if item.get("id") is not None
    }
    coordinates = {
        str(item.get("id")): item
        for item in view_frame_graph.get("shared_coordinate_system", {}).get("scopes", []) or []
        if item.get("id") is not None
    }
    relations = {
        str(item.get("id")): item
        for item in view_frame_graph.get("relations", []) or []
        if item.get("id") is not None
    }
    profiles = _profiles(scoped_profile_assembly)
    dimensions_by_id = {
        str(item.attachment_id): item for item in dimensions if item.status == "accepted"
    }

    eligible = []
    for physical_scope_ref, rows in sorted(by_scope.items()):
        parent_rows = [item for item in rows if item.get("projection_role") == "parent_projection"]
        section_rows = [item for item in rows if item.get("projection_role") == "section_projection"]
        if not parent_rows or not section_rows:
            continue
        scope = scopes.get(physical_scope_ref)
        coordinate_ref = str((scope or {}).get("shared_coordinate_scope_id") or "")
        coordinate = coordinates.get(coordinate_ref)
        orientation = (coordinate or {}).get("signed_orientation_certificate", {}) or {}
        if orientation.get("status") == "contradiction":
            continue
        relation_refs = list(map(str, (scope or {}).get("relation_refs", []) or []))
        accepted_relations = [
            relations[ref]
            for ref in relation_refs
            if ref in relations
            and relations[ref].get("type") == "cut_at"
            and relations[ref].get("state") == "accepted"
            and relations[ref].get("integration_certificate", {}).get("status") == "passed"
        ]
        if len(accepted_relations) != 1:
            continue
        relation = accepted_relations[0]
        pair_candidates = []
        for parent in parent_rows:
            for section in section_rows:
                if (
                    str(parent.get("source_view_ref")) != str(relation.get("parent_view_id"))
                    or str(section.get("source_view_ref")) != str(relation.get("section_view_id"))
                ):
                    continue
                parent_metric, section_metric = _metric(parent), _metric(section)
                if parent_metric is None or section_metric is None:
                    continue
                shared_axes = _matching_axis_pair(parent_metric, section_metric)
                if shared_axes is None:
                    continue
                parent_profile = profiles.get(parent_metric[0])
                section_profile = profiles.get(section_metric[0])
                if (
                    parent_profile is None
                    or section_profile is None
                    or len(parent_profile.get("ordered_boundary_display", []) or []) != 4
                    or len(section_profile.get("ordered_boundary_display", []) or []) != 4
                ):
                    continue
                parent_dimensions = [
                    _profile_dimension(parent_profile, dimensions_by_id, parent_metric[index + 1])
                    for index in (0, 1)
                ]
                depth_value = section_metric[2 - shared_axes[1]]
                depth_dimension = _profile_dimension(section_profile, dimensions_by_id, depth_value)
                if depth_dimension is None or any(item is None for item in parent_dimensions):
                    continue
                pair_candidates.append(
                    (
                        parent,
                        section,
                        parent_metric,
                        section_metric,
                        shared_axes,
                        parent_profile,
                        section_profile,
                        parent_dimensions,
                        depth_value,
                        depth_dimension,
                    )
                )
        if len(pair_candidates) != 1:
            continue
        (
            parent,
            section,
            parent_metric,
            section_metric,
            shared_axes,
            parent_profile,
            section_profile,
            parent_dimensions,
            depth_value,
            depth_dimension,
        ) = pair_candidates[0]
        reprojection_refs = list(map(str, (scope or {}).get("reprojection_validation_refs", []) or []))
        validations = list((coordinate or {}).get("reprojection_validations", []) or [])
        if not reprojection_refs or len(validations) != len(reprojection_refs):
            continue
        position_refs = sorted(
            {
                str(ref)
                for validation in validations
                if validation.get("status") == "pass"
                for ref in validation.get("evidence_refs", []) or []
                if str(ref) in dimensions_by_id
            }
        )
        cut_rows = [
            item
            for item in (coordinate or {}).get("cut_plane_constraints", []) or []
            if str(item.get("relation_id")) == str(relation.get("id"))
        ]
        correspondences = [
            item
            for item in (coordinate or {}).get("contour_correspondences", []) or []
            if item.get("state") == "accepted"
        ]
        if (
            (orientation.get("status") == "accepted" and len(position_refs) < 2)
            or len(cut_rows) != 1
            or len(correspondences) != 1
        ):
            continue
        cut_candidates = cut_rows[0].get("coordinate_candidates_mm", []) or []
        signed = correspondences[0].get("selected", {}).get("signed_transform", {}) or {}
        signed_mapping = signed.get("child_coordinate_to_parent", {}) or {}
        signed_mode = bool(
            orientation.get("status") == "accepted"
            and len(cut_candidates) == 1
            and isinstance(cut_candidates[0], (int, float))
            and math.isfinite(float(cut_candidates[0]))
            and signed.get("state") == "resolved"
            and signed_mapping.get("sign") in {-1, 1}
            and isinstance(signed_mapping.get("offset_mm"), (int, float))
            and math.isfinite(float(signed_mapping["offset_mm"]))
        )
        mirror_candidates = signed.get("candidates", []) or []
        if not signed_mode and (
            orientation.get("status") == "accepted"
            or signed.get("state") != "unresolved"
            or len(mirror_candidates) != 2
            or {item.get("sign") for item in mirror_candidates} != {-1, 1}
            or any(
                not isinstance(item.get("offset_mm"), (int, float))
                or isinstance(item.get("offset_mm"), bool)
                or not math.isfinite(float(item["offset_mm"]))
                for item in mirror_candidates
            )
        ):
            continue
        mappings = {
            str(item.get("view_id")): item
            for item in (coordinate or {}).get("view_axis_mappings", []) or []
        }
        parent_mapping = mappings.get(str(relation.get("parent_view_id")), {})
        section_mapping = mappings.get(str(relation.get("section_view_id")), {})
        if parent_mapping.get("state") != "resolved_relative" or section_mapping.get("state") != "resolved_relative":
            continue
        parent_scale = float(parent_profile["dimension_certificate"]["scale_points_per_mm"])
        section_scale = float(section_profile["dimension_certificate"]["scale_points_per_mm"])
        parent_box = list(map(float, parent_profile.get("bbox_display", [])))
        section_box = list(map(float, section_profile.get("bbox_display", [])))
        if len(parent_box) != 4 or len(section_box) != 4 or parent_scale <= 0 or section_scale <= 0:
            continue
        offsets = {
            str(parent_mapping["u"]): parent_box[0] / parent_scale,
            str(parent_mapping["v"]): parent_box[1] / parent_scale,
            str(section_mapping["v"]): section_box[1] / section_scale,
        }
        object_axes = {
            str(item.get(axis))
            for item in (parent_mapping, section_mapping)
            for axis in ("u", "v", "normal")
            if item.get(axis)
        }
        if set(offsets) != object_axes or len(object_axes) != 3:
            continue
        validation = validations[0]
        shared_axis = str(validation.get("shared_axis") or "u")
        position_axis = str(parent_mapping.get(shared_axis) or parent_mapping.get("u"))
        cut_axis = str(cut_rows[0].get("object_axis") or parent_mapping.get("v"))
        if not position_axis or not cut_axis or position_axis == cut_axis:
            continue
        parent_axis_names = [
            str(parent_mapping.get("u") or ""),
            str(parent_mapping.get("v") or ""),
        ]
        section_depth_display_axis = "u" if 1 - shared_axes[1] == 0 else "v"
        depth_axis = str(section_mapping.get(section_depth_display_axis) or "")
        if not depth_axis or set(parent_axis_names + [depth_axis]) != object_axes:
            continue
        extents_by_axis = {
            parent_axis_names[0]: parent_metric[1],
            parent_axis_names[1]: parent_metric[2],
            depth_axis: depth_value,
        }
        dimension_refs_by_axis = {
            parent_axis_names[index]: [parent_dimensions[index].attachment_id]
            for index in (0, 1)
        }
        dimension_refs_by_axis[depth_axis] = [depth_dimension.attachment_id]
        unsigned_certificate = None
        if not signed_mode:
            unsigned_certificate = certify_unsigned_bounded_sweep(
                page_number=page_number,
                hypothesis_ref=str(parent.get("id") or ""),
                physical_scope_ref=physical_scope_ref,
                orientation_certificate=orientation,
                signed_transform=signed,
                extents_by_axis_mm=extents_by_axis,
                plan_bounded_axes=parent_axis_names,
                dimension_refs_by_axis=dimension_refs_by_axis,
                external_interface_count=0,
                evidence_refs=[
                    coordinate_ref,
                    relation.get("id"),
                    correspondences[0].get("id"),
                    *reprojection_refs,
                ],
            )
            if unsigned_certificate is None:
                continue
        position = (
            {
                position_axis: float(validation.get("parent_value_mm")),
                cut_axis: float(cut_candidates[0]),
            }
            if signed_mode
            else {axis: float(offsets[axis]) for axis in parent_axis_names}
        )
        placement_position_refs = (
            position_refs
            if signed_mode
            else sorted(
                {
                    parent_dimensions[0].attachment_id,
                    parent_dimensions[1].attachment_id,
                }
            )
        )
        eligible.append(
            {
                "physical_scope_ref": physical_scope_ref,
                "scope": scope,
                "coordinate_ref": coordinate_ref,
                "coordinate": coordinate,
                "relation": relation,
                "parent": parent,
                "section": section,
                "parent_profile": parent_profile,
                "section_profile": section_profile,
                "depth_value": depth_value,
                "depth_dimension_ref": depth_dimension.attachment_id,
                "position_refs": placement_position_refs,
                "position": position,
                "offsets": offsets,
                "object_axes": object_axes,
                "reprojection_refs": reprojection_refs,
                "signed_mapping": signed_mapping,
                "signed_transform_candidates": mirror_candidates,
                "cut_coordinate": float(cut_candidates[0]) if signed_mode else None,
                "cut_coordinate_candidates": [float(value) for value in cut_candidates],
                "correspondence_ref": str(correspondences[0].get("id") or ""),
                "orientation_ref": str(orientation.get("id") or ""),
                "orientation_mode": "signed" if signed_mode else "unsigned_bounded_sweep",
                "unsigned_bounded_sweep_certificate": unsigned_certificate,
            }
        )

    if len(eligible) != 1:
        return _abstention(
            page_number,
            "unique_single_prism_placement_unresolved",
            [item.get("id") for item in hypotheses],
        )

    item = eligible[0]
    parent, section, relation = item["parent"], item["section"], item["relation"]
    selected_ref, supporting_ref = str(parent["id"]), str(section["id"])
    independent_refs = sorted(
        {
            str(relation["id"]),
            item["coordinate_ref"],
            item["orientation_ref"],
            item["correspondence_ref"],
            supporting_ref,
            str((item["unsigned_bounded_sweep_certificate"] or {}).get("id") or ""),
            *item["reprojection_refs"],
        }
        - {""}
    )
    grouping_id = _stable_id(
        "physical_component_grouping_evidence",
        page_number,
        item["physical_scope_ref"],
        selected_ref,
        supporting_ref,
        *independent_refs,
    )
    grouping = {
        "id": grouping_id,
        "record_type": "physical_component_grouping_evidence",
        "record_version": SCHEMA_VERSION,
        "state": "resolved",
        "physical_scope_ref": item["physical_scope_ref"],
        "completeness": "complete",
        "hypothesis_refs": [selected_ref],
        "excluded_hypothesis_records": [
            {
                "hypothesis_ref": str(hypothesis["id"]),
                "state": "resolved",
                "disposition": (
                    "supporting_projection_only"
                    if str(hypothesis["id"]) == supporting_ref
                    else "rejected_alternative"
                ),
                "evidence_refs": independent_refs,
            }
            for hypothesis in hypotheses
            if str(hypothesis.get("physical_scope_ref") or "") == item["physical_scope_ref"]
            and str(hypothesis.get("id") or "") != selected_ref
        ],
        "separately_drawn_profiles_are_additive": False,
        "component_identity_records": [
            {
                "hypothesis_ref": selected_ref,
                "state": "resolved",
                "separately_drawn_profile_basis": False,
                "identity_evidence_refs": independent_refs,
            }
        ],
        "interfaces": {
            "state": "resolved",
            "completeness": "complete",
            "no_external_interfaces_required": True,
            "records": [],
        },
        "evidence_refs": independent_refs,
    }
    placement_refs = sorted(
        {
            *independent_refs,
            item["depth_dimension_ref"],
            *item["position_refs"],
        }
    )
    placement_id = _stable_id(
        "component_placement_evidence", page_number, selected_ref, *placement_refs
    )
    placement = {
        "id": placement_id,
        "record_type": "component_placement_evidence",
        "record_version": SCHEMA_VERSION,
        "state": "resolved",
        "hypothesis_ref": selected_ref,
        "supporting_projection_hypothesis_ref": supporting_ref,
        "physical_scope_ref": item["physical_scope_ref"],
        "profile_refs": list(map(str, parent.get("profile_refs", []) or [])),
        "relation_ref": str(relation["id"]),
        "parent_view_ref": str(relation["parent_view_id"]),
        "section_view_ref": str(relation["section_view_id"]),
        "depth": {
            "state": "resolved",
            "value_mm": item["depth_value"],
            "dimension_refs": [item["depth_dimension_ref"]],
        },
        "position": {
            "state": "resolved",
            "coordinates_mm": dict(sorted(item["position"].items())),
            "dimension_refs": item["position_refs"],
        },
        "orientation_sensitivity": (
            "orientation_sensitive"
            if item["orientation_mode"] == "signed"
            else "sign_invariant"
        ),
        "unsigned_bounded_sweep_certificate": item["unsigned_bounded_sweep_certificate"],
        "relative_transform": {
            "state": (
                "resolved"
                if item["orientation_mode"] == "signed"
                else "resolved_up_to_reflection"
            ),
            "axis_signs": (
                {axis: 1 for axis in sorted(item["object_axes"])}
                if item["orientation_mode"] == "signed"
                else {axis: None for axis in sorted(item["object_axes"])}
            ),
            "canonical_axis_signs": {axis: 1 for axis in sorted(item["object_axes"])},
            "offsets_mm": {
                axis: round(float(item["offsets"][axis]), 6)
                for axis in sorted(item["object_axes"])
            },
            "section_to_parent": {
                "state": "resolved" if item["orientation_mode"] == "signed" else "unresolved_reflection",
                "sign": (
                    int(item["signed_mapping"]["sign"])
                    if item["orientation_mode"] == "signed"
                    else None
                ),
                "offset_mm": (
                    float(item["signed_mapping"]["offset_mm"])
                    if item["orientation_mode"] == "signed"
                    else None
                ),
                "candidates": item["signed_transform_candidates"],
            },
            "cut_plane_coordinate_mm": item["cut_coordinate"],
            "cut_plane_coordinate_candidates_mm": item["cut_coordinate_candidates"],
        },
        "metric_extent_refs": item["reprojection_refs"],
        "evidence_refs": placement_refs,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "single_prism_component_placement_evidence",
        "page": page_number,
        "status": "resolved",
        "reason_code": None,
        "component_groupings": [grouping],
        "component_placements": [placement],
        "evidence_refs": placement_refs,
        "contract": {
            "single_component_only": True,
            "separate_profiles_are_additive": False,
            "signed_orientation_required": item["orientation_mode"] == "signed",
            "orientation_sensitive_placement_requires_signed_orientation": True,
            "unsigned_bounded_sweep_certificate_accepted": item["orientation_mode"] == "unsigned_bounded_sweep",
            "quantity_eligible": False,
            "schedule_values_used": False,
        },
    }
