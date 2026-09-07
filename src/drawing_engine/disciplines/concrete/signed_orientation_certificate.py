"""Fail-closed signed orientation closure for shared coordinate scopes.

The shared coordinate solver intentionally establishes only an unsigned
orthographic gauge.  This module is the sole upstream boundary that may close
the remaining signs.  It uses native cutting-plane arrowheads for the plane
normal and independent asymmetric native landmarks for the contour transform;
component placement hypotheses are deliberately not an input.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Any


SCHEMA_VERSION = "0.1.0"
_POINT_TOLERANCE = 1.25


def _point(segment: Mapping[str, Any], endpoint: str) -> tuple[float, float]:
    return tuple(map(float, segment[f"{endpoint}_display"]))


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.dist(left, right)


def _same_style(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    def key(segment: Mapping[str, Any]) -> tuple[Any, ...]:
        style = segment.get("style", {}) or {}
        stroke = style.get("stroke")
        return (
            None if stroke is None else tuple(round(float(value), 2) for value in stroke),
            None if style.get("width") is None else round(float(style["width"]), 2),
            str(style.get("dash") or "").replace(" ", ""),
        )

    return key(left) == key(right)


def _shared_tip(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
    candidates = []
    for left_name in ("start", "end"):
        for right_name in ("start", "end"):
            left_point, right_point = _point(left, left_name), _point(right, right_name)
            separation = _distance(left_point, right_point)
            if separation > _POINT_TOLERANCE:
                continue
            tip = ((left_point[0] + right_point[0]) / 2.0, (left_point[1] + right_point[1]) / 2.0)
            left_tail = _point(left, "end" if left_name == "start" else "start")
            right_tail = _point(right, "end" if right_name == "start" else "start")
            candidates.append((separation, tip, left_tail, right_tail))
    if not candidates:
        return None
    _, tip, left_tail, right_tail = min(candidates, key=lambda item: item[0])
    return tip, left_tail, right_tail


def _arrowhead_direction(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    stem: Mapping[str, Any],
    trace_orientation: str,
) -> int | None:
    """Return the native display-normal sign for one unambiguous chevron."""

    if left.get("axis") != "oblique" or right.get("axis") != "oblique":
        return None
    if not _same_style(left, right) or not _same_style(left, stem):
        return None
    shared = _shared_tip(left, right)
    if shared is None:
        return None
    tip, left_tail, right_tail = shared
    normal_index = 1 if trace_orientation == "horizontal" else 0
    tangent_index = 1 - normal_index
    stem_endpoints = (_point(stem, "start"), _point(stem, "end"))
    if min(_distance(tip, endpoint) for endpoint in stem_endpoints) > 2.0 * _POINT_TOLERANCE:
        return None
    left_tangent = left_tail[tangent_index] - tip[tangent_index]
    right_tangent = right_tail[tangent_index] - tip[tangent_index]
    if left_tangent * right_tangent >= 0:
        return None
    tail_normal = ((left_tail[normal_index] + right_tail[normal_index]) / 2.0) - tip[normal_index]
    tangent_spread = abs(left_tangent - right_tangent)
    if abs(tail_normal) < _POINT_TOLERANCE or tangent_spread < 2.0 * _POINT_TOLERANCE:
        return None
    # Arrow direction is from the two tails toward their shared tip.
    return 1 if -tail_normal > 0 else -1


def _oriented_cutting_plane_certificate(
    constraint: Mapping[str, Any],
    plane: Mapping[str, Any],
    native_segments: list[Mapping[str, Any]],
) -> dict[str, Any]:
    trace = constraint.get("trace", {}) or {}
    orientation = str(trace.get("orientation") or "")
    trace_refs = set(map(str, trace.get("primitive_refs", []) or []))
    perpendicular = "vertical" if orientation == "horizontal" else "horizontal"
    trace_segments = [
        item
        for item in native_segments
        if orientation in {"horizontal", "vertical"}
        and (
            str(item.get("drawing_ref")) in trace_refs
            or str(item.get("primitive_ref")) in trace_refs
            or str(item.get("id")) in trace_refs
        )
    ]
    stems = [item for item in trace_segments if item.get("axis") == perpendicular]
    if not stems:
        tangent_index = 0 if orientation == "horizontal" else 1
        normal_index = 1 - tangent_index
        trace_lines = [item for item in trace_segments if item.get("axis") == orientation]
        trace_endpoints = [
            _point(item, endpoint)
            for item in trace_lines
            for endpoint in ("start", "end")
        ]
        stems = [
            item
            for item in native_segments
            if item.get("axis") == perpendicular
            and any(_same_style(item, trace_line) for trace_line in trace_lines)
            and any(
                min(_point(item, "start")[normal_index], _point(item, "end")[normal_index]) - _POINT_TOLERANCE
                <= endpoint[normal_index]
                <= max(_point(item, "start")[normal_index], _point(item, "end")[normal_index]) + _POINT_TOLERANCE
                and abs(
                    (_point(item, "start")[tangent_index] + _point(item, "end")[tangent_index]) / 2.0
                    - endpoint[tangent_index]
                )
                <= _POINT_TOLERANCE
                for endpoint in trace_endpoints
            )
        ]
    marker_records = []
    excluded = {str(item.get("id")) for item in stems}
    for stem in sorted(stems, key=lambda item: str(item.get("id"))):
        stem_length = max(float(stem.get("length_points") or 0.0), 1.0)
        endpoints = (_point(stem, "start"), _point(stem, "end"))
        nearby = [
            item
            for item in native_segments
            if str(item.get("id")) not in excluded
            and item.get("axis") == "oblique"
            and float(item.get("length_points") or 0.0) <= 1.5 * stem_length
            and min(
                _distance(_point(item, endpoint_name), stem_endpoint)
                for endpoint_name in ("start", "end")
                for stem_endpoint in endpoints
            )
            <= 1.5 * stem_length
        ]
        arrowheads = []
        for left_index, left in enumerate(nearby):
            for right in nearby[left_index + 1 :]:
                direction = _arrowhead_direction(left, right, stem, orientation)
                if direction is None:
                    continue
                arrowheads.append(
                    {
                        "display_normal_sign": direction,
                        "segment_refs": sorted([str(left.get("id")), str(right.get("id"))]),
                    }
                )
        unique = {
            (item["display_normal_sign"], tuple(item["segment_refs"])): item
            for item in arrowheads
        }
        directions = sorted({item["display_normal_sign"] for item in unique.values()})
        marker_records.append(
            {
                "stem_ref": str(stem.get("id")),
                "stem_primitive_ref": str(stem.get("primitive_ref")),
                "arrowhead_candidates": sorted(
                    unique.values(),
                    key=lambda item: (item["display_normal_sign"], item["segment_refs"]),
                ),
                "direction_candidates": directions,
                "state": "resolved" if len(directions) == 1 else "unresolved",
            }
        )
    resolved_directions = [
        item["direction_candidates"][0]
        for item in marker_records
        if item["state"] == "resolved"
    ]
    signed_coordinate = plane.get("signed_coordinate_in_parent_gauge_mm")
    coordinate_available = (
        isinstance(signed_coordinate, (int, float))
        and not isinstance(signed_coordinate, bool)
        and math.isfinite(float(signed_coordinate))
    )
    if len(stems) < 2 or len(resolved_directions) != len(stems):
        status, reason = "insufficient_constraints", "missing_unique_native_arrowhead"
        selected_direction = None
    elif len(set(resolved_directions)) > 1:
        status, reason = "insufficient_constraints", "bidirectional_or_inconsistent_native_markers"
        selected_direction = None
    elif not coordinate_available:
        status, reason = "insufficient_constraints", "signed_cut_plane_coordinate_unavailable"
        selected_direction = None
    else:
        status, reason = "pass", "all native endpoint arrowheads select one viewing direction"
        selected_direction = resolved_directions[0]
    selected_coordinate = (
        round(float(signed_coordinate), 6)
        if status == "pass" and coordinate_available
        else None
    )
    return {
        "status": status,
        "reason": reason,
        "trace_orientation": orientation or None,
        "cut_normal_display_axis": "v" if orientation == "horizontal" else "u" if orientation == "vertical" else None,
        "selected_viewing_sign": selected_direction,
        "selected_coordinate_mm": selected_coordinate,
        "marker_records": marker_records,
        "evidence_refs": sorted(
            {
                *trace_refs,
                *[item["stem_ref"] for item in marker_records],
                *[
                    ref
                    for item in marker_records
                    for candidate in item["arrowhead_candidates"]
                    for ref in candidate["segment_refs"]
                ],
            }
        ),
    }


def _hausdorff(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return math.inf
    return max(
        max(min(abs(value - candidate) for candidate in right) for value in left),
        max(min(abs(value - candidate) for candidate in left) for value in right),
    )


def _dimension_endpoint_landmarks(
    correspondence: Mapping[str, Any],
    attachments: Mapping[str, Mapping[str, Any]],
    frames: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    parent_id, child_id = str(correspondence.get("parent_view_id")), str(correspondence.get("child_view_id"))
    by_side: dict[str, dict[float, list[dict[str, Any]]]] = {
        "parent": {},
        "child": {},
    }
    for attachment_ref, attachment in attachments.items():
        if attachment.get("status") != "accepted":
            continue
        if len(set(map(str, attachment.get("owner_entity_refs", []) or []))) != 1:
            continue
        endpoints = attachment.get("measured_endpoints", []) or []
        if len(endpoints) != 2 or any(item.get("state") != "resolved" for item in endpoints):
            continue
        view_refs = set(map(str, attachment.get("view_refs", []) or []))
        if view_refs == {parent_id}:
            side_name, orientation = "parent", correspondence.get("parent_display_orientation")
            scale = frames.get(parent_id, {}).get("scale", {}).get("value_points_per_mm")
        elif view_refs == {child_id}:
            side_name, orientation = "child", correspondence.get("child_display_orientation")
            scale = frames.get(child_id, {}).get("scale", {}).get("value_points_per_mm")
        else:
            continue
        if attachment.get("orientation") != orientation:
            continue
        if not isinstance(scale, (int, float)) or scale <= 0:
            continue
        value_mm = attachment.get("value_mm")
        if not isinstance(value_mm, (int, float)) or isinstance(value_mm, bool) or not math.isfinite(float(value_mm)):
            continue
        coordinate_index = 0 if orientation == "horizontal" else 1
        record = {
            "attachment_ref": str(attachment_ref),
            "coordinates_mm": sorted(
                float(item["point_display"][coordinate_index]) / float(scale)
                for item in endpoints
            ),
            "evidence_refs": sorted(
                {
                    str(attachment_ref),
                    *[
                        str(item.get("selected_geometry_anchor_ref"))
                        for item in endpoints
                        if item.get("selected_geometry_anchor_ref")
                    ],
                }
            ),
        }
        by_side[side_name].setdefault(round(float(value_mm), 3), []).append(record)
    landmarks = []
    for value_mm in sorted(set(by_side["parent"]) & set(by_side["child"])):
        parent_rows = by_side["parent"][value_mm]
        child_rows = by_side["child"][value_mm]
        if len(parent_rows) != 1 or len(child_rows) != 1:
            continue
        parent, child = parent_rows[0], child_rows[0]
        landmarks.append(
            {
                "kind": "uniquely_owned_dimension_endpoints",
                "value_mm": value_mm,
                "parent_coordinates_mm": parent["coordinates_mm"],
                "child_coordinates_mm": child["coordinates_mm"],
                "evidence_refs": sorted({*parent["evidence_refs"], *child["evidence_refs"]}),
            }
        )
    return landmarks


def _signed_shared_axis_certificate(
    correspondence: Mapping[str, Any] | None,
    attachments: Mapping[str, Mapping[str, Any]],
    frames: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    selected = (correspondence or {}).get("selected") or {}
    transform = selected.get("signed_transform", {}) or {}
    candidates = transform.get("candidates", []) or []
    if transform.get("state") == "resolved" and transform.get("child_coordinate_to_parent"):
        candidates = [transform["child_coordinate_to_parent"]]
    parent_signature = selected.get("parent_signature", {}) or {}
    child_signature = selected.get("child_signature", {}) or {}
    parent_coordinates = list(map(float, parent_signature.get("coordinates_mm", []) or []))
    child_coordinates = list(map(float, child_signature.get("coordinates_mm", []) or []))
    tolerance = float(selected.get("tolerance_mm") or 0.0)
    landmarks = []
    if parent_coordinates and child_coordinates:
        kinds = ["native_contour_junctions"]
        if "cutting-plane" in str(parent_signature.get("basis", "")):
            kinds.append("cut_intersection_anchors")
        if len(parent_coordinates) >= 3 and len(child_coordinates) >= 3:
            kinds.append("ordered_internal_levels_or_subdivisions")
        landmarks.append(
            {
                "kind": "+".join(kinds),
                "parent_coordinates_mm": parent_coordinates,
                "child_coordinates_mm": child_coordinates,
                "evidence_refs": sorted(
                    {
                        *selected.get("parent_geometry_refs", []),
                        *selected.get("child_geometry_refs", []),
                    }
                ),
            }
        )
    landmarks.extend(
        _dimension_endpoint_landmarks(correspondence or {}, attachments, frames)
    )
    evaluations = []
    for candidate in candidates:
        sign, offset = candidate.get("sign"), candidate.get("offset_mm")
        if sign not in {-1, 1} or not isinstance(offset, (int, float)):
            continue
        checks = []
        for landmark in landmarks:
            transformed = [sign * value + float(offset) for value in landmark["child_coordinates_mm"]]
            residual = _hausdorff(landmark["parent_coordinates_mm"], transformed)
            checks.append(
                {
                    "kind": landmark["kind"],
                    "status": "pass" if residual <= tolerance else "fail",
                    "residual_mm": round(residual, 6),
                    "tolerance_mm": round(tolerance, 6),
                    "evidence_refs": landmark["evidence_refs"],
                }
            )
        evaluations.append(
            {
                "sign": sign,
                "offset_mm": round(float(offset), 6),
                "status": "pass" if checks and all(item["status"] == "pass" for item in checks) else "fail",
                "landmark_checks": checks,
            }
        )
    passing = [item for item in evaluations if item["status"] == "pass"]
    if len(passing) == 1:
        status, reason, selected_transform = "pass", "exactly one signed transform reprojects every native landmark", {
            "sign": passing[0]["sign"],
            "offset_mm": passing[0]["offset_mm"],
        }
    elif len(passing) > 1:
        status, reason, selected_transform = "ambiguous", "forward and mirrored transforms both reproject every available landmark", None
    elif evaluations:
        status, reason, selected_transform = "contradiction", "no signed transform reprojects every native landmark", None
    else:
        status, reason, selected_transform = "insufficient_constraints", "no signed transform and paired native landmarks are available", None
    return {
        "status": status,
        "reason": reason,
        "selected_transform": selected_transform,
        "candidate_evaluations": evaluations,
        "landmark_kinds": sorted({item["kind"] for item in landmarks}),
        "evidence_refs": sorted({ref for item in landmarks for ref in item["evidence_refs"]}),
    }


def certify_signed_orientation(
    frames: list[dict[str, Any]],
    shared_coordinates: dict[str, Any],
    dimension_ownership: Mapping[str, Any] | None = None,
    native_segments: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Attach one signed-orientation publication boundary to every scope."""

    segments = list(native_segments)
    frames_by_id = {str(item.get("view_id")): item for item in frames}
    constraints = {
        str(item.get("id")): item for item in shared_coordinates.get("constraints", []) or []
    }
    attachments = {
        str(item.get("id")): item
        for item in (dimension_ownership or {}).get("attachments", []) or []
    }
    records = []
    for scope in shared_coordinates.get("scopes", []) or []:
        planes = {
            str(item.get("relation_id")): item
            for item in scope.get("cut_plane_constraints", []) or []
        }
        correspondences = {
            str(item.get("constraint_id")): item
            for item in scope.get("contour_correspondences", []) or []
        }
        constraint_certificates = []
        for constraint_id in scope.get("constraint_ids", []) or []:
            constraint = constraints.get(str(constraint_id))
            if constraint is None or not str(constraint.get("mode", "")).startswith("cut_"):
                continue
            plane = planes.get(str(constraint.get("source_relation_id")))
            if plane is None:
                # Current coordinate records predate source_relation_id and
                # have a one-to-one relation-to-cut-constraint order.
                matching = list(planes.values())
                plane = matching[0] if len(matching) == 1 else {}
            marker = _oriented_cutting_plane_certificate(constraint, plane, segments)
            landmark = _signed_shared_axis_certificate(
                correspondences.get(str(constraint_id)), attachments, frames_by_id
            )
            if marker["status"] == "pass" and landmark["status"] == "pass":
                status, reason = "accepted", "cut normal, plane coordinate, and signed shared-axis transform close independently"
            elif landmark["status"] == "contradiction":
                status, reason = "contradiction", "signed native landmark reprojection contradicts every transform"
            else:
                status, reason = "insufficient_constraints", "orientation evidence does not uniquely close both required signs"
            constraint_certificates.append(
                {
                    "constraint_id": str(constraint_id),
                    "status": status,
                    "reason": reason,
                    "oriented_cutting_plane": marker,
                    "signed_shared_axis": landmark,
                }
            )
        if not constraint_certificates:
            continue
        if constraint_certificates and all(item["status"] == "accepted" for item in constraint_certificates):
            scope_status = "accepted"
        elif any(item["status"] == "contradiction" for item in constraint_certificates):
            scope_status = "contradiction"
        else:
            scope_status = "insufficient_constraints"
        record = {
            "id": f"signed_orientation_certificate.{len(records) + 1:04d}",
            "shared_coordinate_scope_id": str(scope.get("id")),
            "status": scope_status,
            "constraint_certificates": constraint_certificates,
            "selected_orientation": (
                {
                    "relative_transforms": [
                        item["signed_shared_axis"]["selected_transform"]
                        for item in constraint_certificates
                    ],
                    "cut_plane_coordinates_mm": [
                        item["oriented_cutting_plane"]["selected_coordinate_mm"]
                        for item in constraint_certificates
                    ],
                }
                if scope_status == "accepted"
                else None
            ),
            "quantity_aggregation_eligible": False,
        }
        scope["signed_orientation_certificate"] = record
        scope["relative_orientation_state"] = "resolved" if scope_status == "accepted" else "unresolved"
        if scope_status == "accepted":
            for certificate in constraint_certificates:
                transform = certificate["signed_shared_axis"]["selected_transform"]
                correspondence = correspondences.get(certificate["constraint_id"])
                if correspondence is not None and correspondence.get("selected") is not None:
                    correspondence["selected"]["signed_transform"] = {
                        "state": "resolved",
                        "child_coordinate_to_parent": dict(transform),
                        "candidates": [dict(transform)],
                        "reason": "scope-level signed-orientation certificate uniquely reprojects independent native landmarks",
                    }
                coordinate = certificate["oriented_cutting_plane"]["selected_coordinate_mm"]
                if coordinate is not None:
                    constraint = constraints.get(certificate["constraint_id"], {})
                    relation_id = str(constraint.get("source_relation_id") or "")
                    for plane in scope.get("cut_plane_constraints", []) or []:
                        if relation_id and str(plane.get("relation_id") or "") != relation_id:
                            continue
                        plane["coordinate_candidates_mm"] = [coordinate]
                        plane["viewing_sign_state"] = "resolved"
                        plane["viewing_sign"] = certificate["oriented_cutting_plane"]["selected_viewing_sign"]
                        plane["orientation_certificate_ref"] = record["id"]
            scope["gauge"]["relative_axis_sign_state"] = "resolved"
            scope["signed_contour_transform_count"] = sum(
                (item.get("selected") or {}).get("signed_transform", {}).get("state") == "resolved"
                for item in scope.get("contour_correspondences", []) or []
            )
            scope["path_reprojection_eligible"] = scope["signed_contour_transform_count"] > 0
        records.append(record)
    summary = {
        "certificate_count": len(records),
        "accepted_count": sum(item["status"] == "accepted" for item in records),
        "insufficient_constraints_count": sum(item["status"] == "insufficient_constraints" for item in records),
        "contradiction_count": sum(item["status"] == "contradiction" for item in records),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "layer": "signed_orientation_certificate",
        "status": "resolved_subset" if summary["accepted_count"] else "unresolved",
        "records": records,
        "summary": summary,
        "contract": {
            "native_marker_geometry_only": True,
            "missing_inconsistent_or_bidirectional_markers_abstain": True,
            "exactly_one_landmark_transform_required": True,
            "page_or_title_placement_selects_sign": False,
            "component_placement_selects_sign": False,
            "published_once_per_shared_coordinate_scope": True,
            "quantities_read": False,
            "quantities_written": False,
        },
    }
    shared_coordinates["signed_orientation"] = result
    shared_coordinates.setdefault("summary", {})["signed_orientation_accepted_scope_count"] = summary["accepted_count"]
    contour_summary = shared_coordinates.get("contour_correspondence", {}).get("summary")
    if isinstance(contour_summary, dict):
        contour_summary["signed_contour_transform_count"] = sum(
            (item.get("selected") or {}).get("signed_transform", {}).get("state") == "resolved"
            for scope in shared_coordinates.get("scopes", []) or []
            for item in scope.get("contour_correspondences", []) or []
        )
        contour_summary["path_reprojection_eligible_scope_count"] = sum(
            bool(scope.get("path_reprojection_eligible"))
            for scope in shared_coordinates.get("scopes", []) or []
        )
    return result
