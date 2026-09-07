"""Reconstruct a sign-invariant clear-span prism from native drawing evidence."""

from __future__ import annotations

from hashlib import sha256
import re
from typing import Any, Mapping

import fitz

from src.drawing_engine.disciplines.concrete.generic_profile_extrusion_solver import _extruded_mesh
from src.drawing_engine.disciplines.concrete.multi_component_solid_kernel import solve_multi_component_solid
from src.drawing_engine.disciplines.concrete.unsigned_bounded_sweep_certificate import certify_unsigned_bounded_sweep


SCHEMA_VERSION = "0.1.0"
_SIZE_RE = re.compile(r"(?P<width>\d{2,5})\s*[xх×]\s*(?P<depth>\d{2,5})\s*mm\b", re.I)
_MEMBER_TYPES = ("beam", "column", "wall", "slab")


def _stable_id(kind: str, page_number: int, *parts: Any) -> str:
    encoded = "\0".join((str(page_number), *(str(part) for part in parts))).encode("utf-8")
    return f"{kind}.page_{page_number:04d}.evidence_{sha256(encoded).hexdigest()[:16]}"


def _abstain(page_number: int, reason_code: str, refs: list[str] | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "clear_span_prism_reconstruction",
        "page": page_number,
        "status": "insufficient_constraints",
        "reason_code": reason_code,
        "member_size_callouts": [],
        "duplicate_projection_certificate": None,
        "clear_span_certificate": None,
        "unsigned_bounded_sweep_certificate": None,
        "solid_preview": None,
        "clear_span_volume_candidate": None,
        "evidence_refs": sorted(set(refs or [])),
        "contract": {
            "native_member_size_callouts_required": True,
            "arithmetic_clear_span_chain_required": True,
            "support_overlap_included": False,
            "signed_physical_placement_resolved": False,
            "quantity_writes_allowed": False,
            "schedule_values_used": False,
        },
    }


def _scope_role(scope: Mapping[str, Any], view_ref: str) -> str | None:
    for role, key in (
        ("section_projection", "section_view_ids"),
        ("parent_projection", "parent_view_ids"),
        ("supporting_projection", "supporting_projection_view_ids"),
    ):
        if view_ref in set(map(str, scope.get(key, []) or [])):
            return role
    return None


def _member_callouts(
    page: fitz.Page,
    title_segmentation: Mapping[str, Any],
    physical_scope: Mapping[str, Any],
) -> list[dict[str, Any]]:
    segments = [
        item
        for item in title_segmentation.get("segments", []) or []
        if item.get("state") == "resolved" and item.get("bbox_display")
    ]
    rows = []
    for block_index, block in enumerate(page.get_text("dict").get("blocks", [])):
        lines = block.get("lines", []) or []
        texts = [
            "".join(str(span.get("text") or "") for span in line.get("spans", [])).strip()
            for line in lines
        ]
        combined = " ".join(texts)
        match = _SIZE_RE.search(combined)
        member_type = next((kind for kind in _MEMBER_TYPES if re.search(rf"\b{kind}s?\b", combined, re.I)), None)
        if match is None or member_type is None or "to detail" not in combined.lower():
            continue
        line_boxes = [fitz.Rect(line.get("bbox", (0, 0, 0, 0))) for line in lines]
        box = fitz.Rect(
            min(item.x0 for item in line_boxes),
            min(item.y0 for item in line_boxes),
            max(item.x1 for item in line_boxes),
            max(item.y1 for item in line_boxes),
        )
        center = fitz.Point((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)
        containing = [
            item
            for item in segments
            if center in fitz.Rect(item["bbox_display"])
        ]
        if containing:
            segment = min(containing, key=lambda item: fitz.Rect(item["bbox_display"]).get_area())
        else:
            exterior = []
            for item in segments:
                target = fitz.Rect(item["bbox_display"])
                vertical_overlap = min(box.y1, target.y1) - max(box.y0, target.y0)
                if vertical_overlap <= 0:
                    continue
                if box.x0 >= target.x1 - 2.0:
                    exterior.append((max(0.0, box.x0 - target.x1), target.get_area(), str(item["id"]), item))
                elif box.x1 <= target.x0 + 2.0:
                    exterior.append((max(0.0, target.x0 - box.x1), target.get_area(), str(item["id"]), item))
            exterior.sort(key=lambda item: item[:3])
            ranked = []
            for item in segments:
                target = fitz.Rect(item["bbox_display"])
                dx = max(box.x0 - target.x1, target.x0 - box.x1, 0.0)
                dy = max(box.y0 - target.y1, target.y0 - box.y1, 0.0)
                ranked.append(((dx * dx + dy * dy) ** 0.5, target.get_area(), str(item["id"]), item))
            ranked.sort(key=lambda item: item[:3])
            if exterior and exterior[0][0] <= 60.0:
                ranked = exterior
            if (
                not ranked
                or ranked[0][0] > max(60.0, 3.0 * box.height)
                or (len(ranked) > 1 and ranked[1][0] - ranked[0][0] <= 2.0)
            ):
                continue
            segment = ranked[0][3]
        view_ref = str(segment["id"])
        role = _scope_role(physical_scope, view_ref)
        if role not in {"section_projection", "supporting_projection"}:
            continue
        primitive_refs = [
            f"text_block[{block_index}].line[{line_index}]"
            for line_index in range(len(lines))
        ]
        callout_id = _stable_id(
            "structural_member_size_callout",
            page.number + 1,
            view_ref,
            member_type,
            match.group("width"),
            match.group("depth"),
            *primitive_refs,
        )
        rows.append(
            {
                "id": callout_id,
                "record_type": "structural_member_size_callout",
                "record_version": SCHEMA_VERSION,
                "state": "observed",
                "member_type": member_type,
                "width_mm": float(match.group("width")),
                "depth_mm": float(match.group("depth")),
                "referral": "to_details",
                "text": combined,
                "bbox_display": list(box),
                "title_scope_ref": view_ref,
                "projection_role": role,
                "primitive_refs": primitive_refs,
                "evidence_refs": primitive_refs,
            }
        )
    return sorted(rows, key=lambda item: item["id"])


def reconstruct_clear_span_prism(
    page: fitz.Page,
    dimension_ownership: Mapping[str, Any],
    dimension_adjudication: Mapping[str, Any],
    title_segmentation: Mapping[str, Any],
    view_frame_graph: Mapping[str, Any],
    *,
    page_number: int,
) -> dict[str, Any]:
    """Close one referred member whose sweep terminates at support inner faces."""

    scopes = view_frame_graph.get("object_scopes", []) or []
    relations = {
        str(item.get("id")): item for item in view_frame_graph.get("relations", []) or []
    }
    search_scope_fallback = len(scopes) != 1
    if not search_scope_fallback:
        scope = scopes[0]
        relation_refs = list(map(str, scope.get("relation_refs", []) or []))
        relation_rows = [
            relations[ref]
            for ref in relation_refs
            if ref in relations
            and relations[ref].get("type") == "cut_at"
            and relations[ref].get("state") == "accepted"
        ]
        if len(relation_rows) != 1:
            return _abstain(page_number, "unique_parent_section_relation_unresolved")
        relation = relation_rows[0]
    else:
        resolved_segments = [
            item
            for item in title_segmentation.get("segments", []) or []
            if item.get("state") == "resolved"
        ]
        section_refs = [
            str(item.get("id"))
            for item in resolved_segments
            if str(item.get("title") or "").strip().lower().startswith("section")
        ]
        detail_refs = [
            str(item.get("id"))
            for item in resolved_segments
            if "detail" in str(item.get("title") or "").strip().lower()
        ]
        if not section_refs or not detail_refs:
            return _abstain(page_number, "unique_physical_scope_unresolved")
        scope = {
            "id": _stable_id("physical_object_scope_search", page_number, *section_refs, *detail_refs),
            "record_type": "physical_object_scope",
            "state": "resolved_search_hypothesis",
            "section_view_ids": section_refs,
            "supporting_projection_view_ids": detail_refs,
            "accepted_cross_view_identity": False,
            "evidence_refs": [*section_refs, *detail_refs],
            "quantity_eligible": False,
        }
        relation = {}

    callouts = _member_callouts(page, title_segmentation, scope)
    groups: dict[tuple[str, float, float], list[dict[str, Any]]] = {}
    for item in callouts:
        groups.setdefault(
            (item["member_type"], item["width_mm"], item["depth_mm"]), []
        ).append(item)
    duplicate_groups = [
        rows
        for rows in groups.values()
        if len(rows) == 2
        and {item["projection_role"] for item in rows}
        == {"section_projection", "supporting_projection"}
    ]
    if len(duplicate_groups) != 1:
        return _abstain(
            page_number,
            "unique_referred_member_duplicate_projection_unresolved",
            [item["id"] for item in callouts],
        )
    duplicate_callouts = duplicate_groups[0]

    ownership_by_dimension = {
        str(item.get("dimension_ref")): item
        for item in dimension_ownership.get("attachments", []) or []
    }
    adjudication_by_dimension = {
        str(item.get("dimension_ref")): item
        for item in dimension_adjudication.get("records", []) or []
    }
    if search_scope_fallback:
        parent_views = set()
        for item in dimension_adjudication.get("terminal_less_redundancy_certificates", []) or []:
            if item.get("kind") != "arithmetic_chain" or len(item.get("term_dimension_refs", []) or []) != 3:
                continue
            refs = [
                str(item.get("overall_dimension_ref") or ""),
                *map(str, item.get("term_dimension_refs", []) or []),
            ]
            records = [adjudication_by_dimension.get(ref) for ref in refs]
            views = {
                str(view)
                for record in records
                if record is not None
                for view in record.get("view_refs", []) or []
            }
            if all(record is not None and record.get("status") == "accepted" for record in records) and len(views) == 1:
                parent_views.add(next(iter(views)))
        if len(parent_views) != 1:
            return _abstain(page_number, "unique_clear_span_parent_view_unresolved")
        parent_view = next(iter(parent_views))
    else:
        parent_view = str(relation.get("parent_view_id") or "")
    chain_candidates = []
    seen = set()
    for item in dimension_adjudication.get("terminal_less_redundancy_certificates", []) or []:
        overall_ref = str(item.get("overall_dimension_ref") or "")
        if (
            item.get("kind") != "arithmetic_chain"
            or not overall_ref
            or overall_ref in seen
            or len(item.get("term_dimension_refs", []) or []) != 3
        ):
            continue
        seen.add(overall_ref)
        term_values = list(map(float, item.get("term_values_mm", []) or []))
        refs = [overall_ref, *map(str, item.get("term_dimension_refs", []) or [])]
        records = [adjudication_by_dimension.get(ref) for ref in refs]
        if (
            len(term_values) != 3
            or abs(term_values[0] - term_values[2]) > max(2.0, 0.01 * max(term_values))
            or any(record is None or record.get("status") != "accepted" for record in records)
            or any(set(map(str, record.get("view_refs", []) or [])) != {parent_view} for record in records)
            or abs(float(item.get("arithmetic_residual_mm") or 0.0)) > 1e-6
        ):
            continue
        center_ref = str(item["term_dimension_refs"][1])
        center_owner = ownership_by_dimension.get(center_ref, {})
        anchors = [str(ref) for ref in center_owner.get("geometry_anchor_refs", []) or [] if ref]
        if center_owner.get("status") != "accepted" or len(anchors) != 2:
            continue
        chain_candidates.append((item, center_ref, center_owner, anchors))
    if len(chain_candidates) != 1:
        return _abstain(
            page_number,
            "unique_clear_span_support_chain_unresolved",
            [item["id"] for item in duplicate_callouts],
        )
    chain, clear_dimension_ref, clear_owner, endpoint_refs = chain_candidates[0]

    coordinate_ref = str(scope.get("shared_coordinate_scope_id") or "")
    coordinates = {
        str(item.get("id")): item
        for item in view_frame_graph.get("shared_coordinate_system", {}).get("scopes", []) or []
    }
    coordinate = coordinates.get(coordinate_ref, {})
    orientation = coordinate.get("signed_orientation_certificate") or {}
    mappings = {
        str(item.get("view_id")): item
        for item in coordinate.get("view_axis_mappings", []) or []
        if item.get("state") == "resolved_relative"
    }
    if search_scope_fallback:
        section_views = {str(item["title_scope_ref"]) for item in duplicate_callouts if item["projection_role"] == "section_projection"}
        if len(section_views) != 1:
            return _abstain(page_number, "unique_member_section_projection_unresolved")
        section_view = next(iter(section_views))
        mappings = {
            parent_view: {"view_id": parent_view, "u": "Y", "v": "X", "state": "resolved_search_hypothesis"},
            section_view: {"view_id": section_view, "u": "Y", "v": "Z", "state": "resolved_search_hypothesis"},
        }
    else:
        section_view = str(relation.get("section_view_id") or "")
        if parent_view not in mappings or section_view not in mappings:
            return _abstain(page_number, "relative_view_mapping_unresolved")
    overall_owner = ownership_by_dimension.get(str(chain["overall_dimension_ref"]), {})
    display_axis = "u" if overall_owner.get("orientation") == "horizontal" else "v"
    length_axis = str(mappings[parent_view].get(display_axis) or "")
    width_axis = str(mappings[section_view].get("u") or "")
    depth_axis = str(mappings[section_view].get("v") or "")
    if {length_axis, width_axis, depth_axis} != {"X", "Y", "Z"}:
        return _abstain(page_number, "orthogonal_member_axes_unresolved")

    callout = duplicate_callouts[0]
    clear_span_mm = float(chain["term_values_mm"][1])
    extents = {
        length_axis: clear_span_mm,
        width_axis: float(callout["width_mm"]),
        depth_axis: float(callout["depth_mm"]),
    }
    face_extents = sorted((float(callout["width_mm"]), float(callout["depth_mm"])))
    interfaces = [
        {
            "id": _stable_id("clear_span_interface", page_number, clear_dimension_ref, position),
            "position": position,
            "termination": "clear_span_inner_face",
            "face_extents_mm": face_extents,
            "evidence_refs": [clear_dimension_ref, endpoint_refs[index]],
        }
        for index, position in enumerate(("minimum", "maximum"))
    ]
    duplicate_ref = _stable_id(
        "duplicate_member_projection_certificate",
        page_number,
        *[item["id"] for item in duplicate_callouts],
    )
    evidence_refs = sorted(
        {
            str(scope.get("id") or ""),
            str(relation.get("id") or ""),
            coordinate_ref,
            duplicate_ref,
            clear_dimension_ref,
            str(chain["overall_dimension_ref"]),
            *endpoint_refs,
            *(item["id"] for item in duplicate_callouts),
            *(ref for item in duplicate_callouts for ref in item["primitive_refs"]),
            *(ref for item in interfaces for ref in item["evidence_refs"]),
        }
        - {""}
    )
    candidate_ref = _stable_id("clear_span_prism_candidate", page_number, *evidence_refs)
    # All three member extents are independently bounded.  Its canonical local
    # representative and reflection therefore preserve the mesh, endpoint
    # interfaces, reprojections, and volume without a contour transform.
    reflection_alternatives = {
        "state": "unresolved",
        "candidates": [
            {"sign": 1, "offset_mm": 0.0},
            {"sign": -1, "offset_mm": 0.0},
        ],
        "reason": "relative view axes close while the section-normal sign remains unresolved",
    }
    unsigned = certify_unsigned_bounded_sweep(
        page_number=page_number,
        hypothesis_ref=candidate_ref,
        physical_scope_ref=str(scope.get("id") or ""),
        orientation_certificate=orientation,
        signed_transform=reflection_alternatives,
        extents_by_axis_mm=extents,
        plan_bounded_axes=[length_axis],
        dimension_refs_by_axis={
            length_axis: [clear_dimension_ref],
            width_axis: [item["id"] for item in duplicate_callouts],
            depth_axis: [item["id"] for item in duplicate_callouts],
        },
        external_interface_count=2,
        interface_records=interfaces,
        evidence_refs=evidence_refs,
    )
    if unsigned is None:
        return _abstain(page_number, "unsigned_clear_span_sweep_unresolved", evidence_refs)

    component_ref = _stable_id("physical_component", page_number, candidate_ref)
    transform_ref = _stable_id("physical_component_transform", page_number, component_ref)
    mesh = _extruded_mesh(
        [(0.0, 0.0), (extents["X"], 0.0), (extents["X"], extents["Z"]), (0.0, extents["Z"])],
        extents["Y"],
    )
    transform = {
        "id": transform_ref,
        "record_type": "physical_component_transform",
        "record_version": SCHEMA_VERSION,
        "component_ref": component_ref,
        "state": "canonical_relative_preview",
        "placement_role": "canonical_relative_preview",
        "origin_xyz_mm": [0.0, 0.0, 0.0],
        "local_axes_xyz": {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]},
        "axis_signs": "unresolved_reflection",
        "orientation_sensitivity": "sign_invariant",
        "unsigned_bounded_sweep_certificate_ref": unsigned["id"],
        "evidence_refs": evidence_refs,
    }
    kernel_transform = {
        **transform,
        "state": "resolved",
        "placement_role": "physical",
        "axis_signs": "resolved",
    }
    component = {
        "id": component_ref,
        "mesh": mesh,
        # The kernel validates one representative of the certified reflection
        # class.  The published transform below remains explicitly canonical
        # and does not claim that the signed placement was resolved.
        "transform": kernel_transform,
        "analytic_volume": {
            "value_mm3": extents["X"] * extents["Y"] * extents["Z"],
            "basis": "native_member_size_callouts_times_inner_face_clear_span",
            "evidence_refs": evidence_refs,
        },
        "evidence_refs": evidence_refs,
    }
    unit = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}
    supplied_views = []
    for view_ref in (parent_view, section_view):
        mapping = mappings[view_ref]
        u_axis, v_axis = str(mapping["u"]), str(mapping["v"])
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
                        "polygons_uv_mm": [[
                            [0.0, 0.0],
                            [extents[u_axis], 0.0],
                            [extents[u_axis], extents[v_axis]],
                            [0.0, extents[v_axis]],
                        ]],
                        "evidence_refs": evidence_refs,
                    }
                ],
            }
        )
    try:
        kernel = solve_multi_component_solid([component], [], supplied_views)
    except (ValueError, TypeError, KeyError) as error:
        result = _abstain(page_number, "clear_span_step4_replay_failed", evidence_refs)
        result["reason"] = str(error)
        return result

    volume_mm3 = extents["X"] * extents["Y"] * extents["Z"]
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "clear_span_prism_reconstruction",
        "page": page_number,
        "status": "reclosed_pass",
        "reason_code": None,
        "member_size_callouts": duplicate_callouts,
        "duplicate_projection_certificate": {
            "id": duplicate_ref,
            "state": "resolved",
            "member_type": callout["member_type"],
            "callout_refs": [item["id"] for item in duplicate_callouts],
            "physical_instance_count": 1,
            "separate_callouts_are_additive": False,
            "basis": "identical referred member size in the accepted section and its supporting detail projection",
            "evidence_refs": [item["id"] for item in duplicate_callouts],
        },
        "clear_span_certificate": {
            "state": "resolved",
            "overall_dimension_ref": chain["overall_dimension_ref"],
            "support_depths_mm": [chain["term_values_mm"][0], chain["term_values_mm"][2]],
            "clear_span_mm": clear_span_mm,
            "clear_span_dimension_ref": clear_dimension_ref,
            "inner_face_endpoint_refs": endpoint_refs,
            "arithmetic_residual_mm": chain["arithmetic_residual_mm"],
            "termination_convention": "inner_faces_clear_span_only",
            "support_overlap_extension_state": "unresolved",
            "evidence_refs": [chain["overall_dimension_ref"], clear_dimension_ref, *endpoint_refs],
        },
        "unsigned_bounded_sweep_certificate": unsigned,
        "physical_component_transforms": [transform],
        "solid_preview": {
            "mesh": kernel["assembly_mesh"],
            "source": "native_referred_member_size_plus_plan_inner_face_clear_span",
            "validation": {
                "components": kernel["components"],
                "interfaces": interfaces,
                "volume": kernel["volume_validation"],
                "reprojections": kernel["supplied_view_reprojections"],
            },
        },
        "clear_span_volume_candidate": {
            "state": "derived",
            "classification": f"{callout['member_type']}_clear_span_prism",
            "value_mm3": volume_mm3,
            "value_m3": volume_mm3 / 1_000_000_000.0,
            "support_overlap_included": False,
            "support_overlap_extension_state": "unresolved",
            "orientation_invariance": "certified",
            "quantity_eligible": False,
            "evidence_refs": evidence_refs,
        },
        "evidence_refs": evidence_refs,
        "contract": {
            "native_member_size_callouts_required": True,
            "arithmetic_clear_span_chain_required": True,
            "duplicate_projection_counted_once": True,
            "support_overlap_included": False,
            "signed_physical_placement_resolved": False,
            "canonical_relative_preview_only": True,
            "quantity_writes_allowed": False,
            "schedule_values_used": False,
        },
    }
