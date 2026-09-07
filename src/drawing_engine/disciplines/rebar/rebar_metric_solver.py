"""Dimension-anchored metric reconstruction for simple reinforcement groups.

Drawing coordinates are observations.  Accepted host dimensions define the
metric frame, while repeated bar projections and section end projections add
soft geometric constraints.  A result is accepted only when the reconstructed
bar reprojects into every supporting view within tolerance.
"""

from __future__ import annotations

from collections import defaultdict
import math
import statistics
from typing import Any

import fitz

from src.drawing_engine.core.dimension_attachment import DimensionAttachment


def _line_segments(page: fitz.Page) -> list[dict[str, Any]]:
    segments = []
    for drawing_index, drawing in enumerate(page.get_drawings()):
        width = float(drawing.get("width") or 0.0)
        for item_index, item in enumerate(drawing.get("items", [])):
            if item[0] != "l":
                continue
            a, b = item[1], item[2]
            segments.append(
                {
                    "start": [float(a.x), float(a.y)],
                    "end": [float(b.x), float(b.y)],
                    "width_pt": width,
                    "primitive_ref": f"drawing[{drawing_index}].item[{item_index}]",
                }
            )
    return segments


def _elevation_endpoint_candidates(
    page: fitz.Page,
    elevation_span_display: list[float],
    host_top_y: float,
    host_bottom_y: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    span_left, span_right = sorted(float(value) for value in elevation_span_display)
    host_span = host_bottom_y - host_top_y
    horizontal_padding = max(8.0, (span_right - span_left) * 0.35)
    candidates = []
    all_segments = _line_segments(page)
    for segment in all_segments:
        (x0, y0), (x1, y1) = segment["start"], segment["end"]
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        center_x = (x0 + x1) / 2
        top, bottom = sorted((y0, y1))
        if segment["width_pt"] < 1.2 or dx > 0.5 or dy < 0.70 * host_span:
            continue
        if not (span_left - horizontal_padding <= center_x <= span_right + horizontal_padding):
            continue
        if top < host_top_y - 0.04 * host_span or bottom > host_bottom_y + 0.04 * host_span:
            continue
        candidates.append(
            {
                **segment,
                "x_display": center_x,
                "top_y_display": top,
                "bottom_y_display": bottom,
                "length_display": bottom - top,
            }
        )
    return candidates, all_segments


def _endpoint_consensus(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clusters: list[list[dict[str, Any]]] = []
    for candidate in sorted(candidates, key=lambda row: (row["top_y_display"], row["bottom_y_display"])):
        matching = next(
            (
                cluster
                for cluster in clusters
                if abs(statistics.median(row["top_y_display"] for row in cluster) - candidate["top_y_display"]) <= 2.5
                and abs(statistics.median(row["bottom_y_display"] for row in cluster) - candidate["bottom_y_display"]) <= 2.5
            ),
            None,
        )
        if matching is None:
            clusters.append([candidate])
        else:
            matching.append(candidate)
    if not clusters:
        return []
    return max(
        clusters,
        key=lambda cluster: (
            len({round(row["x_display"], 1) for row in cluster}),
            len(cluster),
            statistics.median(row["length_display"] for row in cluster),
        ),
    )


def _section_position_validation(
    section_observations: dict[str, Any],
    expected_xy_mm: list[list[float]],
    diameter_mm: float,
    host_width_mm: float,
) -> dict[str, Any]:
    expected = sorted(([float(x), float(y)] for x, y in expected_xy_mm), key=lambda point: (point[0], point[1]))
    rows = []
    for section in section_observations.get("sections", []):
        host_bbox = section.get("host_bbox_display")
        if not host_bbox:
            continue
        box = fitz.Rect(host_bbox)
        if box.width <= 0 or box.height <= 0 or abs(box.width / box.height - 1.0) > 0.05:
            continue
        scale = box.width / host_width_mm
        center = [(box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2]
        points = []
        refs = []
        for candidate in section.get("candidates", []):
            if candidate.get("kind") != "filled_end_projection":
                continue
            candidate_box = fitz.Rect(candidate.get("source_bbox_display", candidate["bbox_display"]))
            raw_diameter = max(candidate_box.width, candidate_box.height) / scale
            if abs(raw_diameter - diameter_mm) > max(1.0, diameter_mm * 0.08):
                continue
            points.append(
                [
                    ((candidate_box.x0 + candidate_box.x1) / 2 - center[0]) / scale,
                    -((candidate_box.y0 + candidate_box.y1) / 2 - center[1]) / scale,
                ]
            )
            refs.append(candidate["primitive_ref"])
        points.sort(key=lambda point: (point[0], point[1]))
        if len(points) != len(expected):
            continue
        residuals = [math.dist(left, right) for left, right in zip(expected, points)]
        rows.append(
            {
                "section_id": section["section_id"],
                "point_count": len(points),
                "max_position_residual_mm": round(max(residuals, default=0.0), 3),
                "mean_position_residual_mm": round(statistics.mean(residuals), 3),
                "primitive_refs": refs,
            }
        )
    tolerance = max(2.0, diameter_mm * 0.15)
    passing = [row for row in rows if row["max_position_residual_mm"] <= tolerance]
    return {
        "tested_sections": rows,
        "passing_section_count": len(passing),
        "max_position_residual_mm": None if not passing else max(row["max_position_residual_mm"] for row in passing),
        "tolerance_mm": tolerance,
        "status": "pass" if len(passing) >= 2 else "fail",
    }


def _endpoint_attachment_count(
    selected: list[dict[str, Any]],
    all_segments: list[dict[str, Any]],
) -> tuple[int, list[str]]:
    selected_refs = {row["primitive_ref"] for row in selected}
    attachments = set()
    for bar in selected:
        endpoints = ([bar["x_display"], bar["top_y_display"]], [bar["x_display"], bar["bottom_y_display"]])
        for segment in all_segments:
            if segment["primitive_ref"] in selected_refs or segment["width_pt"] < 1.2:
                continue
            dx = abs(segment["end"][0] - segment["start"][0])
            dy = abs(segment["end"][1] - segment["start"][1])
            if dx <= max(0.5, 0.08 * dy):
                continue
            if any(
                min(math.dist(endpoint, segment["start"]), math.dist(endpoint, segment["end"])) <= 2.5
                for endpoint in endpoints
            ):
                attachments.add(segment["primitive_ref"])
    return len(attachments), sorted(attachments)


def solve_dimension_anchored_straight_group(
    page: fitz.Page,
    group: dict[str, Any],
    elevation_span_display: list[float],
    height_dimension: DimensionAttachment,
    host_width_mm: float,
    section_observations: dict[str, Any],
) -> dict[str, Any]:
    """Solve a straight group's Z endpoints in an accepted host metric frame."""

    host_top_y = min(point[1] for point in height_dimension.measured_points)
    host_bottom_y = max(point[1] for point in height_dimension.measured_points)
    scale = (host_bottom_y - host_top_y) / float(height_dimension.value_mm)
    candidates, all_segments = _elevation_endpoint_candidates(
        page, elevation_span_display, host_top_y, host_bottom_y
    )
    selected = _endpoint_consensus(candidates)
    section_validation = _section_position_validation(
        section_observations,
        group.get("xy_mm", []),
        float(group.get("diameter_mm") or 0.0),
        host_width_mm,
    )
    if not selected:
        return {
            "status": "unresolved",
            "reason": "no unique long straight elevation endpoint consensus",
            "metric_frame": {
                "axis": "Z",
                "host_dimension_id": height_dimension.attachment_id,
                "points_per_mm": scale,
            },
            "section_validation": section_validation,
            "schedule_values_used": False,
        }

    top_y = statistics.median(row["top_y_display"] for row in selected)
    bottom_y = statistics.median(row["bottom_y_display"] for row in selected)
    unique_projection_count = len({round(row["x_display"], 1) for row in selected})
    z_start = (host_bottom_y - bottom_y) / scale
    z_end = (host_bottom_y - top_y) / scale
    raw_length = z_end - z_start
    nearest_millimetre = round(raw_length)
    length = float(nearest_millimetre) if abs(raw_length - nearest_millimetre) <= 0.5 else raw_length
    endpoint_residuals = [
        max(abs(row["top_y_display"] - top_y), abs(row["bottom_y_display"] - bottom_y)) / scale
        for row in selected
    ]
    off_axis_count, off_axis_refs = _endpoint_attachment_count(selected, all_segments)
    reprojection_tolerance = max(2.0, float(group.get("diameter_mm") or 0.0) * 0.15)
    reprojection = {
        "view": "elevation_projection_family",
        "supporting_primitive_count": len(selected),
        "unique_projected_axis_count": unique_projection_count,
        "max_endpoint_residual_mm": round(max(endpoint_residuals, default=0.0), 3),
        "orientation_residual_deg": 0.0,
        "tolerance_mm": reprojection_tolerance,
        "primitive_refs": [row["primitive_ref"] for row in selected],
    }
    checks = {
        "accepted_host_dimension": height_dimension.status == "accepted",
        "elevation_endpoint_consensus": len(selected) >= 2 and unique_projection_count >= 2,
        "section_position_consistency": section_validation["status"] == "pass",
        "reprojection_within_tolerance": reprojection["max_endpoint_residual_mm"] <= reprojection_tolerance,
        "no_off_axis_endpoint_attachment": off_axis_count == 0,
    }
    status = "pass" if all(checks.values()) else "fail"
    primitive_refs = [row["primitive_ref"] for row in selected]
    anchors = [
        {
            "id": "metric_anchor.host.z.base",
            "type": "semantic_metric_anchor",
            "reference": {"object": "object.concrete.host", "feature": "host.base"},
            "coordinate": {"axis": "Z", "value_mm": 0.0},
            "state": "direct",
            "evidence_refs": [height_dimension.attachment_id, *height_dimension.evidence],
        },
        {
            "id": "metric_anchor.host.z.top",
            "type": "semantic_metric_anchor",
            "reference": {"object": "object.concrete.host", "feature": "host.top"},
            "coordinate": {"axis": "Z", "value_mm": float(height_dimension.value_mm)},
            "state": "direct",
            "evidence_refs": [height_dimension.attachment_id, *height_dimension.evidence],
        },
        {
            "id": "metric_anchor.rebar.start",
            "type": "offset_anchor",
            "reference": "metric_anchor.host.z.base",
            "coordinate": {"axis": "Z", "value_mm": round(z_start, 3)},
            "offset_mm": round(z_start, 3),
            "state": "derived",
            "evidence_refs": primitive_refs,
        },
        {
            "id": "metric_anchor.rebar.end",
            "type": "offset_anchor",
            "reference": "metric_anchor.host.z.top",
            "coordinate": {"axis": "Z", "value_mm": round(z_end, 3)},
            "offset_mm": round(z_end - float(height_dimension.value_mm), 3),
            "state": "derived",
            "evidence_refs": primitive_refs,
        },
    ]
    return {
        "status": status,
        "topology_family": "straight",
        "metric_frame": {
            "axis": "Z",
            "host_dimension_id": height_dimension.attachment_id,
            "host_top_y_display": host_top_y,
            "host_bottom_y_display": host_bottom_y,
            "host_height_mm": float(height_dimension.value_mm),
            "points_per_mm": scale,
            "state": "direct",
        },
        "anchors": anchors,
        "solved_centerline": {
            "z_start_mm": round(z_start, 3),
            "z_end_mm": round(z_end, 3),
            "installed_length_each_mm": round(length, 3),
            "raw_installed_length_each_mm": round(raw_length, 3),
            "reporting_increment_mm": 1.0,
            "rounding_residual_mm": round(length - raw_length, 3),
            "state": "derived",
        },
        "elevation_observations": {
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "primitive_refs": primitive_refs,
            "off_axis_endpoint_attachment_count": off_axis_count,
            "off_axis_endpoint_attachment_refs": off_axis_refs,
        },
        "section_validation": section_validation,
        "reprojection": reprojection,
        "validation": {"status": status, "checks": checks},
        "fabrication_eligibility": {
            "status": "pass" if status == "pass" else "unresolved",
            "installed_equals_cutting": status == "pass",
            "basis": "straight unbent topology validated by elevation endpoints and consistent section projections",
        },
        "schedule_values_used": False,
    }
