"""View-scoped extrusion solver for repeated physical-object hypotheses.

The rule is drawing neutral: one closed metric profile is paired with one
three-sided orthographic depth outline.  A dimension must attach to the open
outline's minor extent, both views must share the other plotted extent, and
every resulting mesh must pass independent watertight and volume checks.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable

import fitz

from src.drawing_engine.core.dimension_attachment import DimensionAttachment
from src.drawing_engine.disciplines.concrete.generic_profile_extrusion_solver import _extruded_mesh, _native_loops, _signed_area


def _contained(outer: fitz.Rect, inner: fitz.Rect, tolerance: float = 2.0) -> bool:
    return (outer + (-tolerance, -tolerance, tolerance, tolerance)).contains(inner)


def _three_sided_outlines(page: fitz.Page, scope: fitz.Rect) -> list[dict[str, Any]]:
    rows = []
    for drawing_index, drawing in enumerate(page.get_drawings()):
        rect = fitz.Rect(drawing["rect"])
        items = drawing.get("items", [])
        if not _contained(scope, rect) or len(items) != 3 or any(item[0] != "l" for item in items):
            continue
        if rect.width < 8 or rect.height < 20 or rect.height / max(rect.width, 0.1) < 2.2:
            continue
        segments = [(item[1], item[2]) for item in items]
        if any(abs(a.x - b.x) > 0.7 and abs(a.y - b.y) > 0.7 for a, b in segments):
            continue
        connected = all(segments[index][1].distance_to(segments[index + 1][0]) <= 0.8 for index in range(2))
        if not connected or abs(segments[0][0].y - segments[-1][1].y) > 0.8:
            continue
        rows.append(
            {
                "id": f"open_depth_outline.drawing_{drawing_index}",
                "drawing_ref": f"drawing[{drawing_index}]",
                "bbox_display": list(rect),
                "primitive_refs": [f"drawing[{drawing_index}]"],
                "stroke_width_points": float(drawing.get("width") or 0.0),
                "points_display": [[float(segments[0][0].x), float(segments[0][0].y)], *[[float(b.x), float(b.y)] for _, b in segments]],
            }
        )
    return rows


def _depth_dimension(outline: dict[str, Any], dimensions: Iterable[DimensionAttachment], scope: fitz.Rect) -> list[DimensionAttachment]:
    rect = fitz.Rect(outline["bbox_display"])
    matches = []
    for dimension in dimensions:
        if dimension.status != "accepted" or dimension.orientation != "horizontal":
            continue
        if not scope.intersects(fitz.Rect(dimension.text_bbox)):
            continue
        endpoints = dimension.measured_points
        endpoint_x = sorted(point[0] for point in endpoints)
        if max(abs(endpoint_x[0] - rect.x0), abs(endpoint_x[1] - rect.x1)) > 1.5:
            continue
        if abs(abs(endpoints[0][1] - endpoints[1][1])) > 1.0:
            continue
        graphical_scale = rect.width / max(float(dimension.value_mm), 0.1)
        if abs(graphical_scale - dimension.scale_points_per_mm) > 0.02 * dimension.scale_points_per_mm:
            continue
        matches.append(dimension)
    return matches


def _profile_scale_support(
    profile: dict[str, Any],
    dimensions: Iterable[DimensionAttachment],
    scope: fitz.Rect,
    target_scale: float,
) -> list[DimensionAttachment]:
    rect = fitz.Rect(profile["bbox_display"])
    rows = []
    for dimension in dimensions:
        if dimension.status != "accepted" or not scope.intersects(fitz.Rect(dimension.text_bbox)):
            continue
        if abs(dimension.scale_points_per_mm - target_scale) > 0.02 * target_scale:
            continue
        measured = fitz.Rect(dimension.measured_points[0], dimension.measured_points[1]).normalize()
        if _contained(rect + (-8, -8, 8, 8), measured):
            rows.append(dimension)
    return rows


def _combine_meshes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    components = []
    offset_x = 0.0
    total_volume = 0.0
    for row in rows:
        mesh = row["mesh"]
        local = mesh["vertices_xyz_mm"]
        width = max(point[0] for point in local) - min(point[0] for point in local)
        shifted = [[point[0] + offset_x, point[1], point[2]] for point in local]
        base = len(vertices)
        vertices.extend(shifted)
        faces.extend([[base + index for index in face] for face in mesh["triangles"]])
        components.append(
            {
                "object_instance_id": row["object_instance_id"],
                "vertex_start": base,
                "vertex_count": len(local),
                "display_offset_x_mm": offset_x,
                "display_offset_state": "presentation_only_not_physical_placement",
            }
        )
        offset_x += width + 500.0
        total_volume += float(mesh["validation"]["tetrahedral_volume_mm3"])
    edges: defaultdict[tuple[int, int], int] = defaultdict(int)
    for face in faces:
        for left, right in zip(face, (*face[1:], face[0])):
            edges[tuple(sorted((left, right)))] += 1
    return {
        "vertices_xyz_mm": vertices,
        "triangles": faces,
        "components": components,
        "validation": {
            "vertex_count": len(vertices),
            "face_count": len(faces),
            "edge_count": len(edges),
            "component_count": len(rows),
            "euler_characteristic": len(vertices) - len(edges) + len(faces),
            "boundary_edge_count": sum(value == 1 for value in edges.values()),
            "watertight": bool(rows) and all(value == 2 for value in edges.values()),
            "tetrahedral_volume_mm3": total_volume,
        },
    }


def solve_object_instance_extrusions(
    page: fitz.Page,
    dimensions: tuple[DimensionAttachment, ...],
    object_graph: dict[str, Any],
    views: list[dict[str, Any]],
) -> dict[str, Any]:
    """Solve every paired instance or abstain from the page-level total."""

    instances = object_graph.get("instances", [])
    if not instances:
        raise ValueError("no procedural object-instance pairs for scoped extrusion")
    view_by_id = {item["id"]: item for item in views}
    native_loops = _native_loops(page)
    solved = []
    failures = []
    for instance in instances:
        primary_scope = fitz.Rect(view_by_id[instance["primary_view_id"]]["bbox_display"])
        section_scope = fitz.Rect(view_by_id[instance["section_view_id"]]["bbox_display"])
        profiles = [
            item
            for item in native_loops
            if _contained(primary_scope, fitz.Rect(item["bbox_display"]))
            and 0.08 <= item["fill_ratio"] <= 0.72
            and item["diagonal_count"] >= 1
        ]
        outlines = _three_sided_outlines(page, section_scope)
        candidates = []
        relation_certificate = instance.get("certificate", {}) or {}
        relation_scale_eligible = bool(
            instance.get("state") == "certified"
            and relation_certificate.get("status") == "passed"
            and relation_certificate.get("contour_correspondence_ref")
            and relation_certificate.get("reprojection_validation_refs")
            and float(instance.get("extrusion_depth_mm") or 0.0) > 0
        )
        for profile in profiles:
            profile_rect = fitz.Rect(profile["bbox_display"])
            for outline in outlines:
                outline_rect = fitz.Rect(outline["bbox_display"])
                shared_tolerance = max(1.5, 0.012 * (profile_rect.height + outline_rect.height) / 2)
                shared_residual = profile_rect.height - outline_rect.height
                if abs(shared_residual) > shared_tolerance:
                    continue
                width_similarity = min(profile["stroke_width_points"], outline["stroke_width_points"]) / max(
                    profile["stroke_width_points"], outline["stroke_width_points"], 1e-6
                )
                if width_similarity < 0.72:
                    continue
                for depth in _depth_dimension(outline, dimensions, section_scope):
                    scale = float(depth.scale_points_per_mm)
                    anchors = _profile_scale_support(profile, dimensions, primary_scope, scale)
                    certified_scale = bool(
                        relation_scale_eligible
                        and abs(float(instance["extrusion_depth_mm"]) - float(depth.value_mm))
                        <= max(1.0, 0.005 * float(depth.value_mm))
                    )
                    if not anchors and not certified_scale:
                        continue
                    candidates.append(
                        (
                            profile,
                            outline,
                            depth,
                            anchors,
                            shared_residual,
                            shared_tolerance,
                            {
                                "status": "passed",
                                "method": (
                                    "accepted_profile_dimension"
                                    if anchors
                                    else "relation_certified_native_silhouette_scale_transfer"
                                ),
                                "evidence_refs": (
                                    [item.attachment_id for item in anchors]
                                    if anchors
                                    else [
                                        str(relation_certificate["contour_correspondence_ref"]),
                                        *map(str, relation_certificate["reprojection_validation_refs"]),
                                        *map(str, relation_certificate.get("metric_span_refs", [])),
                                    ]
                                ),
                            },
                        )
                    )
        if len(candidates) != 1:
            failures.append(f"{instance['id']}: {'no' if not candidates else 'multiple'} scoped extrusion closures")
            continue
        profile, outline, depth, anchors, shared_residual, shared_tolerance, scale_certificate = candidates[0]
        scale = float(depth.scale_points_per_mm)
        rect = fitz.Rect(profile["bbox_display"])
        points_mm = [((point[0] - rect.x0) / scale, (rect.y1 - point[1]) / scale) for point in profile["points_display"]]
        area_mm2 = abs(_signed_area(points_mm))
        depth_mm = float(depth.value_mm)
        analytic_volume = area_mm2 * depth_mm
        mesh = _extruded_mesh(points_mm, depth_mm)
        volume_residual = float(mesh["validation"]["tetrahedral_volume_mm3"]) - analytic_volume
        if not mesh["validation"]["watertight"] or abs(volume_residual) > max(1.0, 1e-8 * analytic_volume):
            failures.append(f"{instance['id']}: mesh validation failed")
            continue
        solved.append(
            {
                "object_instance_id": instance["id"],
                "profile": profile,
                "outline": outline,
                "depth_dimension": depth,
                "profile_anchors": anchors,
                "profile_scale_certificate": scale_certificate,
                "scale_points_per_mm": scale,
                "profile_points_xz_mm": [[round(x, 3), round(z, 3)] for x, z in points_mm],
                "profile_area_mm2": area_mm2,
                "depth_mm": depth_mm,
                "volume_mm3": analytic_volume,
                "mesh": mesh,
                "shared_extent_residual_points": shared_residual,
                "shared_extent_tolerance_points": shared_tolerance,
            }
        )
    if failures or len(solved) != len(instances):
        raise ValueError("; ".join(failures) or "not every object instance closed")

    combined_mesh = _combine_meshes(solved)
    total = sum(item["volume_mm3"] for item in solved)
    if not combined_mesh["validation"]["watertight"] or abs(combined_mesh["validation"]["tetrahedral_volume_mm3"] - total) > 1.0:
        raise ValueError("combined object-instance mesh validation failed")
    components = [
        {
            "component": item["object_instance_id"],
            "profile_area_mm2": item["profile_area_mm2"],
            "extrusion_depth_mm": item["depth_mm"],
            "volume_mm3": item["volume_mm3"],
            "volume_m3": item["volume_mm3"] / 1e9,
        }
        for item in solved
    ]
    evidence_profiles = []
    selected_ids = []
    for item in solved:
        profile_id = f"{item['object_instance_id']}.profile"
        outline_id = f"{item['object_instance_id']}.depth_outline"
        selected_ids.extend((profile_id, outline_id))
        evidence_profiles.extend(
            (
                {
                    **item["profile"],
                    "id": profile_id,
                    "role": "extruded_profile",
                    "dimensions_mm": [
                        round(fitz.Rect(item["profile"]["bbox_display"]).width / item["scale_points_per_mm"], 3),
                        round(fitz.Rect(item["profile"]["bbox_display"]).height / item["scale_points_per_mm"], 3),
                    ],
                    "object_instance_id": item["object_instance_id"],
                },
                {
                    **item["outline"],
                    "id": outline_id,
                    "role": "orthographic_depth_outline",
                    "dimensions_mm": [item["depth_mm"], round(fitz.Rect(item["outline"]["bbox_display"]).height / item["scale_points_per_mm"], 3)],
                    "object_instance_id": item["object_instance_id"],
                },
            )
        )
    return {
        "schema_version": "0.1.0",
        "pipeline_mode": "procedural_object_instance_profile_extrusions",
        "concrete_3d_input": {
            "object_name": "procedurally_solved_object_instance_collection",
            "shape_type": "multi_object_extrusion_collection",
            "objects": [
                {
                    "object_instance_id": item["object_instance_id"],
                    "shape_type": "extruded_profile",
                    "profile_points_xz_mm": item["profile_points_xz_mm"],
                    "profile_area_mm2": item["profile_area_mm2"],
                    "extrusion_depth_mm": item["depth_mm"],
                }
                for item in solved
            ],
            "declared_volume_m3": None,
            "unknowns": ["relative physical placement between separate objects is not defined by the sheet"],
        },
        "concrete_quantity_takeoff": {
            "status": "procedurally_calculated",
            "units": "m3",
            "components": components,
            "gross_concrete_mm3": total,
            "gross_concrete_m3": total / 1e9,
            "recess_deduction_mm3": 0.0,
            "recess_deduction_m3": 0.0,
            "net_concrete_mm3": total,
            "net_concrete_m3": total / 1e9,
            "note": "Per-object native profile area multiplied by independently dimensioned paired-view depth; schedules are excluded.",
        },
        "constraint_validation": {
            "status": "pass",
            "object_instance_count": len(solved),
            "all_instances_closed": True,
            "combined_mesh": combined_mesh["validation"],
        },
        "procedural_evidence": {
            "metric_profiles": evidence_profiles,
            "selected_profile_ids": selected_ids,
            "dimension_ids": sorted(
                {item["depth_dimension"].attachment_id for item in solved}
                | {dimension.attachment_id for item in solved for dimension in item["profile_anchors"]}
                | {
                    str(ref)
                    for item in solved
                    for ref in item["profile_scale_certificate"].get("evidence_refs", [])
                    if str(ref).startswith("dimension_")
                }
            ),
            "object_instance_solutions": [
                {
                    "object_instance_id": item["object_instance_id"],
                    "profile_id": f"{item['object_instance_id']}.profile",
                    "depth_outline_id": f"{item['object_instance_id']}.depth_outline",
                    "depth_dimension_id": item["depth_dimension"].attachment_id,
                    "depth_mm": item["depth_mm"],
                    "profile_scale_certificate": item["profile_scale_certificate"],
                    "shared_extent_residual_points": item["shared_extent_residual_points"],
                    "shared_extent_tolerance_points": item["shared_extent_tolerance_points"],
                    "volume_m3": item["volume_mm3"] / 1e9,
                }
                for item in solved
            ],
            "mesh": combined_mesh,
        },
    }
