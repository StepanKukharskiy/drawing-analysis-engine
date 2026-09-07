"""Drawing-neutral solver for a dimensioned profile extruded across another view.

This solver recognizes a deliberately narrow geometric construction: a simple
closed, non-rectangular native-vector profile and a disjoint, mostly filled
orthographic outline that share one plotted extent.  Independent dimension
chains must anchor both views and agree on drawing scale.  Titles, schedules,
filenames, and object-class labels are never inspected.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any

import fitz

from src.drawing_engine.core.dimension_attachment import DimensionAttachment, attach_dimensions


POINT_TOLERANCE = 0.65


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _signed_area(points: list[tuple[float, float]]) -> float:
    return sum(
        left[0] * right[1] - right[0] * left[1]
        for left, right in zip(points, (*points[1:], points[0]))
    ) / 2.0


def _clean_closed_walk(points: list[tuple[float, float]]) -> list[tuple[float, float]] | None:
    """Remove zero-area CAD retraces while retaining real profile vertices."""

    cleaned = []
    for point in points:
        if not cleaned or _distance(cleaned[-1], point) > POINT_TOLERANCE:
            cleaned.append(point)
    if len(cleaned) < 4 or _distance(cleaned[0], cleaned[-1]) > POINT_TOLERANCE:
        return None
    cleaned[-1] = cleaned[0]
    changed = True
    while changed:
        changed = False
        for index in range(1, len(cleaned) - 1):
            if _distance(cleaned[index - 1], cleaned[index + 1]) <= POINT_TOLERANCE:
                del cleaned[index : index + 2]
                changed = True
                break
    if len(cleaned) < 5 or _distance(cleaned[0], cleaned[-1]) > POINT_TOLERANCE:
        return None
    return cleaned[:-1]


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], d: tuple[float, float]
) -> bool:
    return _orientation(a, b, c) * _orientation(a, b, d) < -1e-7 and _orientation(c, d, a) * _orientation(c, d, b) < -1e-7


def _is_simple(points: list[tuple[float, float]]) -> bool:
    count = len(points)
    for left_index in range(count):
        a, b = points[left_index], points[(left_index + 1) % count]
        for right_index in range(left_index + 1, count):
            if right_index in {left_index, (left_index + 1) % count} or (right_index + 1) % count == left_index:
                continue
            c, d = points[right_index], points[(right_index + 1) % count]
            if _segments_intersect(a, b, c, d):
                return False
    return True


def _native_loops(page: fitz.Page) -> list[dict[str, Any]]:
    loops = []
    for drawing_index, drawing in enumerate(page.get_drawings()):
        color = drawing.get("color")
        if color is not None and max(color) > 0.08:
            continue
        walk: list[tuple[float, float]] = []
        valid = True
        for item in drawing["items"]:
            if item[0] != "l":
                valid = False
                break
            start, end = (float(item[1].x), float(item[1].y)), (float(item[2].x), float(item[2].y))
            if not walk:
                walk.extend((start, end))
            elif _distance(walk[-1], start) <= POINT_TOLERANCE:
                walk.append(end)
            else:
                valid = False
                break
        points = _clean_closed_walk(walk) if valid else None
        if points is None or len(points) < 4 or not _is_simple(points):
            continue
        rect = fitz.Rect(drawing["rect"])
        if rect.width < 20 or rect.height < 12 or rect.get_area() > 0.20 * page.rect.get_area():
            continue
        area = abs(_signed_area(points))
        if area < 80:
            continue
        fill_ratio = area / rect.get_area()
        diagonal_count = sum(
            abs(right[0] - left[0]) > POINT_TOLERANCE and abs(right[1] - left[1]) > POINT_TOLERANCE
            for left, right in zip(points, (*points[1:], points[0]))
        )
        loops.append(
            {
                "id": f"native_loop.{len(loops) + 1:03d}",
                "drawing_ref": f"drawing[{drawing_index}]",
                "bbox_display": list(rect),
                "points_display": [[round(x, 4), round(y, 4)] for x, y in points],
                "area_points2": area,
                "fill_ratio": fill_ratio,
                "diagonal_count": diagonal_count,
                "stroke_width_points": float(drawing.get("width") or 0),
                "primitive_refs": [f"drawing[{drawing_index}]"],
            }
        )
    return loops


def _scale_cluster(dimensions: tuple[DimensionAttachment, ...]) -> tuple[float, list[DimensionAttachment]]:
    accepted = [item for item in dimensions if item.status == "accepted"]
    clusters = []
    for seed in accepted:
        members = [
            item
            for item in accepted
            if abs(item.scale_points_per_mm - seed.scale_points_per_mm) <= 0.02 * seed.scale_points_per_mm
        ]
        clusters.append(members)
    if not clusters:
        raise ValueError("no accepted metric scale for profile extrusion")
    members = max(clusters, key=lambda items: (len(items), sum(item.score for item in items)))
    if len(members) < 2:
        raise ValueError("profile extrusion requires repeated metric scale support")
    scale = statistics.median(item.scale_points_per_mm for item in members)
    spread = max(item.scale_points_per_mm for item in members) - min(item.scale_points_per_mm for item in members)
    if spread > 0.025 * scale:
        raise ValueError("profile extrusion dimension scales disagree")
    return scale, members


def _dimension_touches_loop(loop: dict[str, Any], dimension: DimensionAttachment) -> bool:
    rect = fitz.Rect(loop["bbox_display"])
    points = [tuple(point) for point in loop["points_display"]]
    axis = 0 if dimension.orientation == "horizontal" else 1
    cross_axis = 1 - axis
    coordinates = [point[axis] for point in points]
    endpoints = dimension.measured_points
    if not all(min(abs(endpoint[axis] - value) for value in coordinates) <= 1.4 for endpoint in endpoints):
        return False
    cross_value = sum(endpoint[cross_axis] for endpoint in endpoints) / 2
    cross_extent = (rect.y0, rect.y1) if cross_axis == 1 else (rect.x0, rect.x1)
    cross_distance = max(cross_extent[0] - cross_value, cross_value - cross_extent[1], 0.0)
    return cross_distance <= max(8.0, 0.12 * (cross_extent[1] - cross_extent[0]))


def _point_in_triangle(point: tuple[float, float], a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> bool:
    cross1 = _orientation(a, b, point)
    cross2 = _orientation(b, c, point)
    cross3 = _orientation(c, a, point)
    return min(cross1, cross2, cross3) >= -1e-8


def _triangulate(points: list[tuple[float, float]]) -> list[tuple[int, int, int]]:
    if _signed_area(points) < 0:
        points.reverse()
    remaining = list(range(len(points)))
    triangles: list[tuple[int, int, int]] = []
    guard = 0
    while len(remaining) > 3 and guard < len(points) ** 2:
        guard += 1
        for position, middle in enumerate(remaining):
            left = remaining[position - 1]
            right = remaining[(position + 1) % len(remaining)]
            if _orientation(points[left], points[middle], points[right]) <= 1e-8:
                continue
            if any(
                candidate not in {left, middle, right}
                and _point_in_triangle(points[candidate], points[left], points[middle], points[right])
                for candidate in remaining
            ):
                continue
            triangles.append((left, middle, right))
            del remaining[position]
            break
        else:
            raise ValueError("profile polygon cannot be triangulated")
    if len(remaining) != 3:
        raise ValueError("profile polygon triangulation did not close")
    triangles.append(tuple(remaining))
    return triangles


def _extruded_mesh(points_mm: list[tuple[float, float]], depth_mm: float) -> dict[str, Any]:
    if _signed_area(points_mm) < 0:
        points_mm = list(reversed(points_mm))
    cap = _triangulate(points_mm.copy())
    count = len(points_mm)
    vertices = [[x, 0.0, z] for x, z in points_mm] + [[x, depth_mm, z] for x, z in points_mm]
    faces = [list(item) for item in cap] + [[a + count, c + count, b + count] for a, b, c in cap]
    for index in range(count):
        following = (index + 1) % count
        faces.extend(([index, index + count, following + count], [index, following + count, following]))
    edges: defaultdict[tuple[int, int], int] = defaultdict(int)
    for face in faces:
        for left, right in zip(face, (*face[1:], face[0])):
            edges[tuple(sorted((left, right)))] += 1
    boundary_edges = sum(value == 1 for value in edges.values())
    tetrahedral_volume = abs(
        sum(
            sum(
                vertices[face[0]][axis]
                * (
                    vertices[face[1]][(axis + 1) % 3] * vertices[face[2]][(axis + 2) % 3]
                    - vertices[face[1]][(axis + 2) % 3] * vertices[face[2]][(axis + 1) % 3]
                )
                for axis in range(3)
            )
            for face in faces
        )
        / 6.0
    )
    return {
        "vertices_xyz_mm": [[round(value, 4) for value in vertex] for vertex in vertices],
        "triangles": faces,
        "validation": {
            "vertex_count": len(vertices),
            "face_count": len(faces),
            "edge_count": len(edges),
            "euler_characteristic": len(vertices) - len(edges) + len(faces),
            "boundary_edge_count": boundary_edges,
            "watertight": boundary_edges == 0 and all(value == 2 for value in edges.values()),
            "tetrahedral_volume_mm3": tetrahedral_volume,
        },
    }


def solve_generic_profile_extrusion(
    page: fitz.Page,
    dimensions: tuple[DimensionAttachment, ...] | None = None,
) -> dict[str, Any]:
    """Resolve one uniquely supported profile extrusion or abstain."""

    if dimensions is None:
        dimensions = tuple(attach_dimensions(page))
    scale, scale_dimensions = _scale_cluster(dimensions)
    loops = _native_loops(page)
    candidates = []
    for profile in loops:
        if not (0.08 <= profile["fill_ratio"] <= 0.72) or profile["diagonal_count"] < 1:
            continue
        profile_anchors = [item for item in scale_dimensions if _dimension_touches_loop(profile, item)]
        if not profile_anchors:
            continue
        profile_rect = fitz.Rect(profile["bbox_display"])
        for companion in loops:
            if companion is profile or companion["fill_ratio"] < 0.84:
                continue
            companion_rect = fitz.Rect(companion["bbox_display"])
            if profile_rect.intersects(companion_rect):
                continue
            companion_anchors = [item for item in scale_dimensions if _dimension_touches_loop(companion, item)]
            if not companion_anchors:
                continue
            profile_extents = (profile_rect.width, profile_rect.height)
            companion_extents = (companion_rect.width, companion_rect.height)
            for profile_axis in (0, 1):
                for companion_axis in (0, 1):
                    shared_left, shared_right = profile_extents[profile_axis], companion_extents[companion_axis]
                    tolerance = max(1.5, 0.012 * (shared_left + shared_right) / 2)
                    if abs(shared_left - shared_right) > tolerance:
                        continue
                    width_similarity = min(profile["stroke_width_points"], companion["stroke_width_points"]) / max(
                        profile["stroke_width_points"], companion["stroke_width_points"], 1e-6
                    )
                    if width_similarity < 0.72:
                        continue
                    depth_raw = companion_extents[1 - companion_axis] / scale
                    nearby_values = [item.value_mm for item in scale_dimensions if abs(item.value_mm - depth_raw) <= 0.02 * depth_raw]
                    depth_mm = float(statistics.median(nearby_values)) if nearby_values else round(depth_raw, 3)
                    candidates.append(
                        {
                            "profile": profile,
                            "companion": companion,
                            "profile_axis": profile_axis,
                            "companion_axis": companion_axis,
                            "depth_mm": depth_mm,
                            "depth_raw_mm": depth_raw,
                            "shared_extent_residual_points": shared_left - shared_right,
                            "shared_extent_tolerance_points": tolerance,
                            "dimension_ids": sorted(
                                {item.attachment_id for item in (*profile_anchors, *companion_anchors)}
                            ),
                        }
                    )
    unique: dict[tuple[str, str, int, int], dict[str, Any]] = {
        (item["profile"]["id"], item["companion"]["id"], item["profile_axis"], item["companion_axis"]): item
        for item in candidates
    }
    if len(unique) != 1:
        reason = "no dimensioned cross-view profile extrusion closure" if not unique else "multiple dimensioned cross-view profile extrusion closures"
        raise ValueError(reason)
    candidate = next(iter(unique.values()))
    profile = candidate["profile"]
    rect = fitz.Rect(profile["bbox_display"])
    points_mm = [((point[0] - rect.x0) / scale, (rect.y1 - point[1]) / scale) for point in profile["points_display"]]
    area_mm2 = abs(_signed_area(points_mm))
    depth_mm = candidate["depth_mm"]
    analytic_volume = area_mm2 * depth_mm
    mesh = _extruded_mesh(points_mm, depth_mm)
    validation = mesh["validation"]
    volume_residual = validation["tetrahedral_volume_mm3"] - analytic_volume
    if not validation["watertight"] or validation["euler_characteristic"] != 2:
        raise ValueError("profile extrusion B-rep is not watertight")
    if abs(volume_residual) > max(1.0, 1e-8 * analytic_volume):
        raise ValueError("profile extrusion analytic and mesh volumes disagree")
    companion = candidate["companion"]
    selected_ids = [profile["id"], companion["id"]]
    evidence_profiles = [
        {
            **profile,
            "role": "extruded_profile",
            "dimensions_mm": [round(rect.width / scale, 3), round(rect.height / scale, 3)],
            "anchoring_dimension_ids": candidate["dimension_ids"],
        },
        {
            **companion,
            "role": "orthographic_depth_outline",
            "dimensions_mm": [
                round(fitz.Rect(companion["bbox_display"]).width / scale, 3),
                round(fitz.Rect(companion["bbox_display"]).height / scale, 3),
            ],
            "anchoring_dimension_ids": candidate["dimension_ids"],
        },
    ]
    dimensions_mm = {
        "profile_bbox_width_mm": round(rect.width / scale, 3),
        "profile_bbox_height_mm": round(rect.height / scale, 3),
        "extrusion_depth_mm": depth_mm,
        "profile_area_mm2": round(area_mm2, 3),
    }
    return {
        "schema_version": "0.1.0",
        "pipeline_mode": "generic_dimensioned_profile_extrusion",
        "concrete_3d_input": {
            "object_name": "procedurally_solved_generic_profile_extrusion",
            "shape_type": "extruded_profile",
            "dimensions_mm": dimensions_mm,
            "profile_points_xz_mm": [[round(x, 3), round(z, 3)] for x, z in points_mm],
            "source_status": "two_disjoint_metric_native_vector_profiles",
            "declared_volume_m3": None,
            "evidence": [*selected_ids, *candidate["dimension_ids"]],
            "unknowns": ["no geometric void contour was constraint-matched"],
        },
        "concrete_quantity_takeoff": {
            "status": "procedurally_calculated",
            "units": "m3",
            "components": [
                {
                    "component": "extruded_profile",
                    "profile_area_mm2": area_mm2,
                    "extrusion_depth_mm": depth_mm,
                    "volume_mm3": analytic_volume,
                    "volume_m3": analytic_volume / 1e9,
                }
            ],
            "gross_concrete_mm3": analytic_volume,
            "gross_concrete_m3": analytic_volume / 1e9,
            "recess_deduction_mm3": 0.0,
            "recess_deduction_m3": 0.0,
            "net_concrete_mm3": analytic_volume,
            "net_concrete_m3": analytic_volume / 1e9,
            "note": "Native-vector profile area multiplied by cross-view extrusion depth; schedules are excluded.",
        },
        "constraint_validation": {
            "status": "pass",
            "equations": [
                {
                    "id": "dimension_scale_consensus",
                    "scale_points_per_mm": scale,
                    "supporting_dimension_ids": candidate["dimension_ids"],
                    "status": "pass",
                },
                {
                    "id": "cross_view_shared_plotted_extent",
                    "residual_points": candidate["shared_extent_residual_points"],
                    "tolerance_points": candidate["shared_extent_tolerance_points"],
                    "status": "pass",
                },
                {"id": "watertight_profile_extrusion_brep", **validation, "status": "pass"},
                {
                    "id": "analytic_profile_area_times_depth",
                    "volume_mm3": analytic_volume,
                    "mesh_residual_mm3": volume_residual,
                    "status": "pass",
                },
            ],
        },
        "relation_graph": {
            "nodes": [
                {"id": profile["id"], "type": "metric_profile", "bbox_display": profile["bbox_display"]},
                {"id": companion["id"], "type": "metric_orthographic_outline", "bbox_display": companion["bbox_display"]},
                {"id": "object.concrete.host", "type": "semantic_3d_object"},
            ],
            "relations": [
                {
                    "id": "rel.identity.profile_extrusion",
                    "type": "same_object_across_views",
                    "sources": selected_ids,
                    "target": "object.concrete.host",
                    "checks": ["shared_plotted_extent", "independent_metric_anchors", "watertight_brep"],
                }
            ],
        },
        "procedural_evidence": {
            "metric_profiles": evidence_profiles,
            "selected_profile_ids": selected_ids,
            "dimension_ids": candidate["dimension_ids"],
            "drawing_scale_points_per_mm": scale,
            "mesh": mesh,
        },
    }
