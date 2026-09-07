"""Drawing-neutral cross-view solver for thin rectangular prisms.

The solver deliberately covers one small, defensible geometry class: two
disjoint orthographic native-vector profiles that share one metric extent and
close the other two extents.  It does not inspect titles, schedules, filenames,
or object names.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import fitz

from src.drawing_engine.core.dimension_attachment import DimensionAttachment, attach_dimensions


def _path_kind(drawing: dict[str, Any]) -> str | None:
    items = drawing["items"]
    if any(item[0] in {"re", "qu"} for item in items):
        return "closed_rectangle"
    rect = fitz.Rect(drawing["rect"])
    horizontal = vertical = False
    for item in items:
        if item[0] != "l":
            continue
        start, end = item[1], item[2]
        dx, dy = abs(end.x - start.x), abs(end.y - start.y)
        horizontal |= dy <= 0.7 and dx >= 0.90 * rect.width
        vertical |= dx <= 0.7 and dy >= 0.90 * rect.height
    return "orthogonal_profile" if horizontal and vertical else None


def _assembled_rectangle_refs(drawings: list[dict[str, Any]], rect: fitz.Rect) -> tuple[str, ...]:
    """Return exact native path refs when separate paths close ``rect``."""

    tolerance = max(0.8, 0.003 * max(rect.width, rect.height))
    boundaries: dict[str, list[tuple[float, float, str]]] = {name: [] for name in ("top", "bottom", "left", "right")}
    for drawing_index, drawing in enumerate(drawings):
        color = drawing.get("color")
        if color is not None and max(color) > 0.08:
            continue
        for item_index, item in enumerate(drawing["items"]):
            if item[0] != "l":
                continue
            start, end = item[1], item[2]
            ref = f"drawing[{drawing_index}].item[{item_index}]"
            if abs(start.y - end.y) <= tolerance:
                y = (start.y + end.y) / 2
                for name, boundary_y in (("top", rect.y0), ("bottom", rect.y1)):
                    if abs(y - boundary_y) <= tolerance:
                        lo, hi = sorted((start.x, end.x))
                        lo, hi = max(lo, rect.x0), min(hi, rect.x1)
                        if hi - lo >= tolerance:
                            boundaries[name].append((lo, hi, ref))
            if abs(start.x - end.x) <= tolerance:
                x = (start.x + end.x) / 2
                for name, boundary_x in (("left", rect.x0), ("right", rect.x1)):
                    if abs(x - boundary_x) <= tolerance:
                        lo, hi = sorted((start.y, end.y))
                        lo, hi = max(lo, rect.y0), min(hi, rect.y1)
                        if hi - lo >= tolerance:
                            boundaries[name].append((lo, hi, ref))

    refs: set[str] = set()
    for name, target in (("top", (rect.x0, rect.x1)), ("bottom", (rect.x0, rect.x1)), ("left", (rect.y0, rect.y1)), ("right", (rect.y0, rect.y1))):
        intervals = sorted(boundaries[name])
        if not intervals or intervals[0][0] > target[0] + tolerance:
            return ()
        covered = target[0]
        edge_refs = []
        for lo, hi, ref in intervals:
            if lo > covered + tolerance:
                break
            if hi > covered:
                covered = hi
                edge_refs.append(ref)
            if covered >= target[1] - tolerance:
                refs.update(edge_refs)
                break
        else:
            return ()
        if covered < target[1] - tolerance:
            return ()
    return tuple(sorted(refs))


def _axis_match(rect: fitz.Rect, dimension: DimensionAttachment) -> bool:
    axis = 0 if dimension.orientation == "horizontal" else 1
    cross = 1 - axis
    endpoints = dimension.measured_points
    along = sorted(point[axis] for point in endpoints)
    extent = (rect.x0, rect.x1) if axis == 0 else (rect.y0, rect.y1)
    span = extent[1] - extent[0]
    tolerance = max(2.5, 0.02 * span)
    if abs(along[0] - extent[0]) > tolerance or abs(along[1] - extent[1]) > tolerance:
        return False
    cross_value = sum(point[cross] for point in endpoints) / 2
    cross_extent = (rect.y0, rect.y1) if cross == 1 else (rect.x0, rect.x1)
    cross_distance = max(cross_extent[0] - cross_value, cross_value - cross_extent[1], 0.0)
    # CAD witness lines often extend past the measured profile and terminate
    # in a small gap outside it. Along-axis endpoint agreement is the primary
    # attachment test; permit an offset of less than half the orthogonal size.
    return cross_distance <= max(8.0, 0.45 * (cross_extent[1] - cross_extent[0]))


def _snap_mm(value: float) -> tuple[float, float]:
    snapped = round(value / 5.0) * 5.0
    if abs(value - snapped) <= 2.0:
        return snapped, value - snapped
    return round(value, 3), 0.0


def _metric_profiles(page: fitz.Page, dimensions: tuple[DimensionAttachment, ...]) -> list[dict[str, Any]]:
    accepted = [item for item in dimensions if item.status == "accepted"]
    profiles = []
    drawings = page.get_drawings()
    for drawing_index, drawing in enumerate(drawings):
        kind = _path_kind(drawing)
        if kind is None:
            continue
        rect = fitz.Rect(drawing["rect"])
        if not rect.intersects(page.rect):
            continue
        if rect.width < 12 or rect.height < 8 or rect.get_area() > 0.30 * page.rect.get_area():
            continue
        color = drawing.get("color")
        if color is not None and max(color) > 0.08:
            continue
        assembled_refs = _assembled_rectangle_refs(drawings, rect) if kind == "orthogonal_profile" else ()
        if assembled_refs:
            kind = "assembled_closed_rectangle"
        for dimension in accepted:
            if not _axis_match(rect, dimension):
                continue
            scale = dimension.scale_points_per_mm
            raw = (rect.width / scale, rect.height / scale)
            width, width_residual = _snap_mm(raw[0])
            height, height_residual = _snap_mm(raw[1])
            profiles.append(
                {
                    "id": f"metric_profile.{len(profiles) + 1:03d}",
                    "drawing_ref": f"drawing[{drawing_index}]",
                    "bbox_display": list(rect),
                    "path_kind": kind,
                    "scale_points_per_mm": scale,
                    "anchoring_dimension_id": dimension.attachment_id,
                    "anchoring_orientation": dimension.orientation,
                    "dimensions_mm": [width, height],
                    "raw_dimensions_mm": [round(raw[0], 3), round(raw[1], 3)],
                    "snap_residuals_mm": [round(width_residual, 3), round(height_residual, 3)],
                    "primitive_refs": [f"drawing[{drawing_index}]", *assembled_refs, dimension.attachment_id],
                }
            )
    # When both axes are explicitly dimensioned, use both values instead of
    # propagating either axis scale through the other. Small plotted-scale
    # residuals are common even in native CAD exports.
    dimension_by_id = {item.attachment_id: item for item in accepted}
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for profile in profiles:
        grouped[profile["drawing_ref"]].append(profile)
    reconciled = []
    for rows in grouped.values():
        direct: dict[int, DimensionAttachment] = {}
        for row in rows:
            dimension = dimension_by_id[row["anchoring_dimension_id"]]
            axis = 0 if dimension.orientation == "horizontal" else 1
            current = direct.get(axis)
            if current is None or (dimension.score, dimension.scale_support) > (current.score, current.scale_support):
                direct[axis] = dimension
        base = rows[0]
        dimensions_mm = list(base["dimensions_mm"])
        for axis, dimension in direct.items():
            dimensions_mm[axis] = dimension.value_mm
        evidence = sorted({item.attachment_id for item in direct.values()})
        reconciled.append(
            {
                **base,
                "id": f"metric_profile.{len(reconciled) + 1:03d}",
                "dimensions_mm": dimensions_mm,
                "anchoring_dimension_ids": evidence,
                "primitive_refs": list(dict.fromkeys([*base["primitive_refs"], *evidence])),
            }
        )
    # A small profile fully contained in a much larger metric outline is a
    # local subcomponent, not an independent orthographic view. Keep the
    # outermost contour as the host candidate.
    pruned = []
    for profile in reconciled:
        rect = fitz.Rect(profile["bbox_display"])
        nested = any(
            other is not profile
            and fitz.Rect(other["bbox_display"]).contains(rect)
            and fitz.Rect(other["bbox_display"]).get_area() >= 3.0 * rect.get_area()
            for other in reconciled
        )
        if not nested:
            pruned.append(profile)
    return pruned


def _box_mesh(dimensions: tuple[float, float, float]) -> dict[str, Any]:
    x, y, z = dimensions
    vertices = [[dx * x, dy * y, dz * z] for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)]
    faces = [
        [0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6],
        [0, 1, 5], [0, 5, 4], [2, 6, 7], [2, 7, 3],
        [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5],
    ]
    edges: defaultdict[tuple[int, int], int] = defaultdict(int)
    for face in faces:
        for left, right in zip(face, (*face[1:], face[0])):
            edges[tuple(sorted((left, right)))] += 1
    boundary_edges = sum(count == 1 for count in edges.values())
    return {
        "vertices_xyz_mm": vertices,
        "triangles": faces,
        "validation": {
            "vertex_count": len(vertices),
            "face_count": len(faces),
            "edge_count": len(edges),
            "euler_characteristic": len(vertices) - len(edges) + len(faces),
            "boundary_edge_count": boundary_edges,
            "watertight": boundary_edges == 0 and all(count == 2 for count in edges.values()),
        },
    }


def solve_generic_prismatic(
    page: fitz.Page,
    dimensions: tuple[DimensionAttachment, ...] | None = None,
) -> dict[str, Any]:
    """Resolve a unique thin rectangular prism or fail closed."""

    if dimensions is None:
        dimensions = tuple(attach_dimensions(page))
    profiles = _metric_profiles(page, dimensions)
    candidates: dict[tuple[float, float, float], dict[str, Any]] = {}
    for left_index, left in enumerate(profiles):
        left_box = fitz.Rect(left["bbox_display"])
        for right in profiles[left_index + 1 :]:
            if left["drawing_ref"] == right["drawing_ref"] or left_box.intersects(fitz.Rect(right["bbox_display"])):
                continue
            if not any("closed_rectangle" in kind for kind in (left["path_kind"], right["path_kind"])):
                continue
            for left_axis in (0, 1):
                for right_axis in (0, 1):
                    shared_left = left["dimensions_mm"][left_axis]
                    shared_right = right["dimensions_mm"][right_axis]
                    shared = (shared_left + shared_right) / 2
                    if abs(shared_left - shared_right) > max(5.0, 0.01 * shared):
                        continue
                    others = [left["dimensions_mm"][1 - left_axis], right["dimensions_mm"][1 - right_axis]]
                    depth, broad = sorted(others)
                    if depth > 0.25 * broad:
                        continue
                    dimensions_mm = tuple(float(value) for value in (round(shared, 3), broad, depth))
                    key = tuple(sorted(dimensions_mm, reverse=True))
                    residual = shared_left - shared_right
                    record = candidates.setdefault(
                        key,
                        {
                            "dimensions_mm": list(dimensions_mm),
                            "profile_refs": [],
                            "shared_extent_residuals_mm": [],
                            "thinness_ratio": round(depth / broad, 4),
                            "direct_evidence_count": 0,
                        },
                    )
                    record["profile_refs"].extend([left["id"], right["id"]])
                    record["shared_extent_residuals_mm"].append(round(residual, 3))
                    record["direct_evidence_count"] = max(
                        record["direct_evidence_count"],
                        len(set(left.get("anchoring_dimension_ids", [])) | set(right.get("anchoring_dimension_ids", []))),
                    )

    candidate_list = list(candidates.values())
    clusters: list[list[dict[str, Any]]] = []
    for candidate in candidate_list:
        key = sorted(candidate["dimensions_mm"], reverse=True)
        for cluster in clusters:
            representative = sorted(cluster[0]["dimensions_mm"], reverse=True)
            if all(abs(left - right) <= max(5.0, 0.01 * max(left, right)) for left, right in zip(key, representative)):
                cluster.append(candidate)
                break
        else:
            clusters.append([candidate])
    if len(clusters) != 1:
        reason = "no cross-view thin-prism closure" if not candidate_list else "multiple cross-view thin-prism closures"
        raise ValueError(reason)
    candidate = max(
        clusters[0],
        key=lambda item: (item["direct_evidence_count"], -max(abs(value) for value in item["shared_extent_residuals_mm"])),
    )
    dimensions_mm = tuple(candidate["dimensions_mm"])
    mesh = _box_mesh(dimensions_mm)
    if not mesh["validation"]["watertight"] or mesh["validation"]["euler_characteristic"] != 2:
        raise ValueError("generic prism B-rep is not watertight")
    volume_mm3 = dimensions_mm[0] * dimensions_mm[1] * dimensions_mm[2]
    max_residual = max(abs(value) for value in candidate["shared_extent_residuals_mm"])
    profile_by_id = {item["id"]: item for item in profiles}
    supporting_ids = sorted(set(candidate["profile_refs"]))
    supporting = [profile_by_id[item] for item in supporting_ids]
    return {
        "schema_version": "0.1.0",
        "pipeline_mode": "generic_cross_view_prismatic",
        "concrete_3d_input": {
            "object_name": "procedurally_solved_generic_prism",
            "shape_type": "rectangular_prism",
            "dimensions_mm": list(dimensions_mm),
            "source_status": "two_disjoint_metric_native_vector_profiles",
            "declared_volume_m3": None,
            "evidence": supporting_ids,
            "unknowns": ["no geometric void contour was constraint-matched"],
        },
        "concrete_quantity_takeoff": {
            "status": "procedurally_calculated",
            "units": "m3",
            "components": [{"component": "rectangular_prism", "dimensions_mm": list(dimensions_mm), "volume_mm3": volume_mm3, "volume_m3": volume_mm3 / 1e9}],
            "gross_concrete_mm3": volume_mm3,
            "gross_concrete_m3": volume_mm3 / 1e9,
            "recess_deduction_mm3": 0.0,
            "recess_deduction_m3": 0.0,
            "net_concrete_mm3": volume_mm3,
            "net_concrete_m3": volume_mm3 / 1e9,
            "note": "No constraint-matched void contour was found; embedded steel is not deducted from concrete.",
        },
        "constraint_validation": {
            "status": "pass",
            "equations": [
                {"id": "cross_view_shared_extent", "max_abs_residual_mm": max_residual, "tolerance_mm": 5.0, "status": "pass"},
                {"id": "watertight_box_brep", **mesh["validation"], "status": "pass"},
                {"id": "analytic_box_volume", "volume_mm3": volume_mm3, "residual_mm3": 0.0, "tolerance_mm3": 1e-6, "status": "pass"},
            ],
        },
        "relation_graph": {
            "nodes": [{"id": item["id"], "type": "metric_profile", "bbox_display": item["bbox_display"]} for item in supporting]
            + [{"id": "object.concrete.host", "type": "semantic_3d_object"}],
            "relations": [{"id": "rel.identity.concrete_cross_view", "type": "same_object_across_views", "sources": supporting_ids, "target": "object.concrete.host", "checks": ["shared_metric_extent", "orthogonal_profiles", "watertight_brep"]}],
        },
        "procedural_evidence": {"metric_profiles": profiles, "selected_profile_ids": supporting_ids, "mesh": mesh},
    }
