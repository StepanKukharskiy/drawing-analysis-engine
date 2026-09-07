"""Deterministic fabrication length from dimension-attached detail geometry."""

from __future__ import annotations

import math
from typing import Any


def _angle(start: list[float], end: list[float]) -> float:
    return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 180.0


def _unique_value(rows: list[dict[str, Any]], role: str) -> float | None:
    values = {
        round(float(item["value_mm"]), 6)
        for item in rows
        if item.get("semantic_role") == role and item.get("value_mm") is not None
    }
    return next(iter(values)) if len(values) == 1 else None


def _raster_open_rectangular_hook_solution(detail: dict[str, Any]) -> dict[str, Any] | None:
    """Recover a dimensioned open loop while keeping bend radius symbolic.

    Raster fabrication sketches are frequently resized independently in X and
    Y to fit a table row.  Pixel lengths and apparent radii are therefore not
    metric evidence.  Orthogonal overall dimensions and repeated hook
    projections can still close the sharp-vertex topology.  Exact cutting
    length is emitted only if a centerline bend radius is independently
    supplied or explicitly annotated.
    """

    raster = [
        geometry
        for geometry in detail.get("geometry_paths", [])
        if geometry.get("source_type") == "embedded_raster_detail"
    ]
    if not raster:
        return None
    observations = detail.get("raster_fabrication_dimension_observations", [])
    width = _unique_value(observations, "overall_width_dimension")
    height = _unique_value(observations, "overall_height_dimension")
    hook = _unique_value(observations, "hook_projection_dimension")
    if width is None or height is None or hook is None:
        return None

    counts = {"horizontal": 0, "vertical": 0, "diagonal": 0}
    curve_count = 0
    for geometry in raster:
        curve_count += len(geometry.get("curve_candidates", []))
        for segment in geometry.get("segments", []):
            if segment.get("kind") != "line_candidate":
                continue
            angle = _angle(segment["start_display"], segment["end_display"])
            if min(angle, 180.0 - angle) <= 8.0:
                counts["horizontal"] += 1
            elif abs(angle - 90.0) <= 8.0:
                counts["vertical"] += 1
            elif min(abs(angle - 45.0), abs(angle - 135.0)) <= 10.0:
                counts["diagonal"] += 1
    topology_checks = {
        "orthogonal_loop_boundaries": counts["horizontal"] >= 4 and counts["vertical"] >= 4,
        "two_diagonal_hook_boundaries": counts["diagonal"] >= 2,
        "rounded_bend_evidence": curve_count >= 2,
        "overall_width_unique": width > 0,
        "overall_height_unique": height > 0,
        "hook_projection_unique": hook > 0,
    }
    if not all(topology_checks.values()):
        return None

    bend_angles = [90.0, 90.0, 90.0, 135.0, 135.0]
    sharp_length = 2.0 * (width + height) + 2.0 * math.hypot(hook, hook)
    deduction_coefficient = sum(
        2.0 * math.tan(math.radians(angle) / 2.0) - math.radians(angle)
        for angle in bend_angles
    )
    explicit_radii = {
        round(float(item["value_mm"]), 6)
        for item in observations
        if item.get("kind") == "bend_radius"
        and item.get("value_mm") is not None
        and item.get("confidence", 0.0) >= 0.90
    }
    supplied_radius = detail.get("centerline_bend_radius_mm")
    if supplied_radius is not None:
        explicit_radii.add(round(float(supplied_radius), 6))
    radius = next(iter(explicit_radii)) if len(explicit_radii) == 1 else None
    evidence_refs = [
        item["id"]
        for item in observations
        if item.get("semantic_role") in {
            "overall_width_dimension",
            "overall_height_dimension",
            "hook_projection_dimension",
        }
        and item.get("id")
    ]
    common = {
        "state": "derived",
        "topology": "open_rectangular_loop_with_two_diagonal_hooks",
        "metric_geometry": {
            "overall_width_mm": width,
            "overall_height_mm": height,
            "hook_projection_mm": hook,
            "hook_leg_sharp_length_mm": round(math.hypot(hook, hook), 3),
            "bend_angles_deg": bend_angles,
        },
        "sharp_vertex_length_mm": round(sharp_length, 3),
        "centerline_radius_coefficient": round(deduction_coefficient, 6),
        "length_expression": f"{sharp_length:.3f} - {deduction_coefficient:.6f} * R_cl_mm",
        "topology_validation": {"status": "pass", "checks": topology_checks},
        "evidence": {
            "raster_dimension_refs": evidence_refs,
            "line_orientation_support": counts,
            "curve_candidate_count": curve_count,
        },
        "closure_validation": {
            "linear_dimensions_assigned": True,
            "topology_closed": True,
            "centerline_radius_assigned": radius is not None,
            "schedule_values_used": False,
        },
    }
    if radius is None:
        return {
            **common,
            "status": "convention_dependent",
            "length_mm": None,
            "reason": "linear dimensions and hook topology close, but centerline bend radius is not explicitly specified",
        }
    length = sharp_length - deduction_coefficient * radius
    if not 0 < length < sharp_length:
        return {**common, "status": "unresolved", "length_mm": None, "reason": "the supplied centerline bend radius makes the cutting-length equation invalid"}
    return {
        **common,
        "status": "resolved",
        "length_mm": round(length, 3),
        "centerline_bend_radius_mm": radius,
        "basis": "dimension-anchored sharp topology minus circular-bend tangent deductions",
    }


def solve_detail_fabrication_geometry(detail: dict[str, Any]) -> dict[str, Any]:
    """Solve a minimal, fully dimensioned line/arc centerline graph.

    Native centerline paths are eligible. Raster Hough fragments are retained
    as observations but are not promoted because paired outlines, leaders and
    dimension lines cannot yet be separated without ambiguity.
    """

    raster_solution = _raster_open_rectangular_hook_solution(detail)
    if raster_solution is not None:
        return raster_solution

    native_segments = [
        segment
        for geometry in detail.get("geometry_paths", [])
        if geometry.get("source_type") != "embedded_raster_detail"
        for segment in geometry.get("segments", [])
        if segment.get("kind") in {"line", "cubic_bezier"}
    ]
    raster_segments = [
        segment
        for geometry in detail.get("geometry_paths", [])
        if geometry.get("source_type") == "embedded_raster_detail"
        for segment in geometry.get("segments", [])
    ]
    dimensions = [item for item in detail.get("fabrication_dimensions", []) if item.get("status") == "accepted"]
    evidence = {
        "native_segment_count": len(native_segments),
        "raster_segment_candidate_count": len(raster_segments),
        "attached_dimension_ids": [item["dimension_id"] for item in dimensions],
    }
    if not native_segments:
        return {
            "status": "unresolved",
            "length_mm": None,
            "reason": "no unambiguous native centerline segment graph; raster fragments remain candidates",
            "evidence": evidence,
        }
    lines = [item for item in native_segments if item["kind"] == "line"]
    curves = [item for item in native_segments if item["kind"] == "cubic_bezier"]
    orientations: dict[str, list[dict[str, Any]]] = {"horizontal": [], "vertical": []}
    for line in lines:
        angle = _angle(line["start_display"], line["end_display"])
        if min(angle, 180.0 - angle) <= 3.0:
            orientations["horizontal"].append(line)
        elif abs(angle - 90.0) <= 3.0:
            orientations["vertical"].append(line)
        else:
            return {"status": "unresolved", "length_mm": None, "reason": "inclined leg lacks an attached along-leg dimension", "evidence": evidence}
    dimension_by_orientation: dict[str, list[dict[str, Any]]] = {"horizontal": [], "vertical": []}
    for dimension in dimensions:
        if dimension.get("orientation") in dimension_by_orientation:
            dimension_by_orientation[dimension["orientation"]].append(dimension)
    line_total = 0.0
    assignments = []
    for orientation, rows in orientations.items():
        if not rows:
            continue
        candidates = dimension_by_orientation[orientation]
        if len(rows) != 1 or len(candidates) != 1:
            return {
                "status": "unresolved",
                "length_mm": None,
                "reason": f"{orientation} legs and attached dimensions do not form a one-to-one constraint",
                "evidence": evidence,
            }
        value = float(candidates[0]["value_mm"])
        line_total += value
        assignments.append({"segment_kind": "line", "orientation": orientation, "value_mm": value, "dimension_id": candidates[0]["dimension_id"]})
    radius_values = [
        float(item["value_mm"])
        for item in detail.get("raster_fabrication_dimension_observations", [])
        if item.get("kind") == "bend_radius" and item.get("value_mm")
    ]
    arc_total = 0.0
    if curves:
        if len(radius_values) != len(curves):
            return {"status": "unresolved", "length_mm": None, "reason": "every bend needs one explicit radius", "evidence": evidence}
        for curve, radius in zip(curves, radius_values):
            start = curve["start_display"]
            c1 = curve["control_1_display"]
            c2 = curve["control_2_display"]
            end = curve["end_display"]
            first_tangent = _angle(start, c1)
            last_tangent = _angle(c2, end)
            sweep = abs(first_tangent - last_tangent)
            sweep = min(sweep, 180.0 - sweep)
            if not 5.0 <= sweep <= 175.0:
                return {"status": "unresolved", "length_mm": None, "reason": "cubic bend does not yield a stable tangent sweep", "evidence": evidence}
            arc_length = radius * math.radians(sweep)
            arc_total += arc_length
            assignments.append({"segment_kind": "circular_arc", "radius_mm": radius, "sweep_deg": sweep, "value_mm": arc_length})
    if not assignments:
        return {"status": "unresolved", "length_mm": None, "reason": "no fabrication leg was dimensionally assigned", "evidence": evidence}
    return {
        "status": "resolved",
        "state": "derived",
        "length_mm": round(line_total + arc_total, 3),
        "basis": "sum of one-to-one attached leg dimensions and explicit-radius circular arc lengths",
        "assignments": assignments,
        "closure_validation": {"all_native_segments_assigned": True, "schedule_values_used": False},
        "evidence": evidence,
    }
