"""Measured hatch and architectural-boundary certificates for MEP candidates.

The checks operate on frozen projected geometry plus independent M1 grid-axis
evidence.  They never use a route/system label to nominate negative geometry.
Every supplied entity receives one measured result for each negative family;
an absent match is therefore a recorded evaluation, not a literal assumed zero.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from statistics import median
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "0.1.0"
LAYER = "mep_negative_drawing_content_certificates"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _stable_id(kind: str, *parts: Any) -> str:
    return f"{kind}.{hashlib.sha256(_canonical(parts).encode()).hexdigest()[:20]}"


def _distance_to_line(point: Sequence[float], start: Sequence[float],
                      end: Sequence[float]) -> float:
    dx, dy = float(end[0]) - float(start[0]), float(end[1]) - float(start[1])
    denominator = dx * dx + dy * dy
    if denominator <= 1e-18:
        return math.dist(point[:2], start[:2])
    cross = abs(dx * (float(start[1]) - float(point[1]))
                - (float(start[0]) - float(point[0])) * dy)
    return cross / math.sqrt(denominator)


def _path_metrics(polylines: Sequence[Sequence[Sequence[float]]], tolerance: float) -> dict[str, Any]:
    points = [point for path in polylines for point in path]
    segments = [(a, b) for path in polylines for a, b in zip(path, path[1:])]
    length = sum(math.dist(a[:2], b[:2]) for a, b in segments)
    if not points or not segments:
        return {"bbox_display": None, "path_display_points": 0.0,
                "straight": False, "orientation_degrees": None,
                "axis": None, "projection_interval": None,
                "perpendicular_coordinate": None, "closed_rectilinear": False}
    bbox = [min(point[0] for point in points), min(point[1] for point in points),
            max(point[0] for point in points), max(point[1] for point in points)]
    start, end = points[0], points[-1]
    chord = math.dist(start[:2], end[:2])
    straight_residual = max((_distance_to_line(point, start, end) for point in points), default=0.0)
    straight = chord > tolerance and straight_residual <= max(tolerance * 2.0, chord * .001)
    angle = None
    axis = None
    projection = None
    perpendicular = None
    if straight:
        angle = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 180.0
        if min(angle, 180.0 - angle) <= 1.0:
            axis = "horizontal"
            projection = [min(start[0], end[0]), max(start[0], end[0])]
            perpendicular = (start[1] + end[1]) / 2.0
        elif abs(angle - 90.0) <= 1.0:
            axis = "vertical"
            projection = [min(start[1], end[1]), max(start[1], end[1])]
            perpendicular = (start[0] + end[0]) / 2.0
    rectilinear = bool(segments) and all(
        abs(a[0] - b[0]) <= tolerance or abs(a[1] - b[1]) <= tolerance
        for a, b in segments)
    closed = (math.dist(points[0][:2], points[-1][:2]) <= tolerance
              and len(segments) >= 4 and rectilinear)
    return {
        "bbox_display": [round(float(value), 6) for value in bbox],
        "path_display_points": round(length, 6),
        "straight": straight,
        "straightness_residual_display_points": round(straight_residual, 6),
        "orientation_degrees": round(angle, 6) if angle is not None else None,
        "axis": axis,
        "projection_interval": [round(value, 6) for value in projection] if projection else None,
        "perpendicular_coordinate": round(perpendicular, 6) if perpendicular is not None else None,
        "closed_rectilinear": closed,
    }


def _interval_overlap(left: Sequence[float], right: Sequence[float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def _accepted_grid_axes(grid_axes: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in grid_axes
            if row.get("state") == "observed"
            and row.get("method") == "opposing_equal_label_alignment_v1"
            and row.get("opposing_label_pair_count", 0) >= 1
            and row.get("axis_group_evidence_refs")
            and len(row.get("opposing_label_span_display", [])) == 2]


def _hatch_clusters(prepared: Sequence[Mapping[str, Any]], tolerance: float) -> dict[str, dict[str, Any]]:
    groups: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for row in prepared:
        metrics = row["geometry_metrics"]
        if not metrics["straight"] or metrics["projection_interval"] is None:
            continue
        style = row.get("non_colour_style") or {}
        width = float(style.get("width_display_points") or tolerance)
        key = (metrics["axis"], round(float(metrics["orientation_degrees"] or 0.0), 0),
               round(width, 3), style.get("dash_pattern"))
        groups[key].append(row)
    accepted: dict[str, dict[str, Any]] = {}
    for key, rows in groups.items():
        if len(rows) < 8:
            continue
        for seed in rows:
            seed_metrics = seed["geometry_metrics"]
            seed_interval = seed_metrics["projection_interval"]
            seed_length = seed_metrics["path_display_points"]
            peers = []
            for row in rows:
                metrics = row["geometry_metrics"]
                length = metrics["path_display_points"]
                overlap = _interval_overlap(seed_interval, metrics["projection_interval"])
                if (min(seed_length, length) > tolerance
                        and overlap / min(seed_length, length) >= .72
                        and .55 <= length / seed_length <= 1.82):
                    peers.append(row)
            coordinates = sorted({round(float(row["geometry_metrics"]["perpendicular_coordinate"]), 6)
                                  for row in peers})
            if len(coordinates) < 8:
                continue
            gaps = [right - left for left, right in zip(coordinates, coordinates[1:])
                    if right - left > tolerance]
            if len(gaps) < 7:
                continue
            pitch = median(gaps)
            pitch_residual = max(abs(gap - pitch) for gap in gaps)
            style_width = max(tolerance, float((seed.get("non_colour_style") or {})
                                               .get("width_display_points") or tolerance))
            if (pitch <= max(style_width * 16.0, seed_length * .12)
                    and pitch_residual <= max(tolerance * 4.0, pitch * .25)):
                evidence_refs = sorted({ref for row in peers for ref in row["source_primitive_refs"]})
                record = {
                    "kind": "regular_parallel_hatch_lattice",
                    "peer_entity_refs": sorted(row["id"] for row in peers),
                    "source_primitive_refs": evidence_refs,
                    "parallel_member_count": len(coordinates),
                    "median_pitch_display_points": round(pitch, 6),
                    "maximum_pitch_residual_display_points": round(pitch_residual, 6),
                    "non_colour_style_key": list(key),
                }
                for row in peers:
                    accepted[row["id"]] = record
    return accepted


def build_negative_content_certificates(
    *, entities: Sequence[Mapping[str, Any]], grid_axes: Sequence[Mapping[str, Any]],
    page_rect_display: Sequence[float], precision_display_points: float,
) -> dict[str, Any]:
    """Evaluate every supplied entity against measured negative motifs."""
    tolerance = max(float(precision_display_points), 1e-6)
    prepared = []
    for entity in entities:
        metrics = _path_metrics(entity.get("polylines_display", []), tolerance)
        prepared.append({**entity, "geometry_metrics": metrics})
    hatch_by_entity = _hatch_clusters(prepared, tolerance)
    axes = _accepted_grid_axes(grid_axes)
    page_width = float(page_rect_display[2]) - float(page_rect_display[0])
    page_height = float(page_rect_display[3]) - float(page_rect_display[1])
    records = []
    for entity in prepared:
        metrics = entity["geometry_metrics"]
        architectural_matches = []
        accepted_architectural_matches = []
        if metrics["straight"] and metrics["axis"]:
            for axis in axes:
                if axis.get("orientation_display") != metrics["axis"]:
                    continue
                residual = abs(float(metrics["perpendicular_coordinate"])
                               - float(axis["coordinate_display"]))
                span = axis["opposing_label_span_display"]
                overlap = _interval_overlap(metrics["projection_interval"], span)
                length = max(tolerance, metrics["path_display_points"])
                line_width = float((entity.get("non_colour_style") or {})
                                   .get("width_display_points") or tolerance)
                if residual <= max(tolerance * 4.0, line_width * 2.0) and overlap / length >= .80:
                    match = {
                        "kind": "opposing_label_grid_axis_coincidence",
                        "grid_axis_ref": axis["id"],
                        "grid_axis_label": axis.get("label"),
                        "coordinate_residual_display_points": round(residual, 6),
                        "overlap_ratio": round(overlap / length, 6),
                        "axis_span_coverage_ratio": round(
                            overlap / max(tolerance, float(span[1]) - float(span[0])), 6),
                        "evidence_refs": sorted(set(axis.get("evidence_refs", []))
                                                | set(axis.get("axis_group_evidence_refs", []))),
                    }
                    architectural_matches.append(match)
                    if match["axis_span_coverage_ratio"] >= .60:
                        accepted_architectural_matches.append(match)
        if metrics["closed_rectilinear"] and metrics["bbox_display"]:
            box = metrics["bbox_display"]
            width, height = box[2] - box[0], box[3] - box[1]
            if (width >= max(tolerance * 20, page_width * .005)
                    and height >= max(tolerance * 20, page_height * .005)):
                match = {
                    "kind": "closed_rectilinear_drawing_boundary",
                    "bbox_display": box,
                    "evidence_refs": list(entity["source_primitive_refs"]),
                }
                architectural_matches.append(match)
                accepted_architectural_matches.append(match)
        hatch_match = hatch_by_entity.get(entity["id"])
        records.append({
            "id": _stable_id("mep_negative_content_evaluation", entity["id"],
                             architectural_matches, hatch_match),
            "record_type": "mep_negative_content_evaluation",
            "state": "measured",
            "page_ref": entity.get("page_ref"),
            "entity_ref": entity["id"],
            "entity_kind": entity.get("entity_kind"),
            "source_primitive_refs": sorted(entity.get("source_primitive_refs", [])),
            "geometry_metrics": metrics,
            "hatch_certificate": ({"state": "accepted", **hatch_match}
                                  if hatch_match else {"state": "no_match"}),
            "architectural_boundary_certificate": (
                {"state": "accepted", "matches": architectural_matches}
                if accepted_architectural_matches else
                {"state": "review_candidate", "matches": architectural_matches}
                if architectural_matches else {"state": "no_match", "matches": []}),
            "route_identity_established": False,
            "quantity_eligible": False,
        })
    hatch_refs = {ref for row in records if row["hatch_certificate"]["state"] == "accepted"
                  for ref in row["source_primitive_refs"]}
    architectural_refs = {
        ref for row in records
        if row["architectural_boundary_certificate"]["state"] == "accepted"
        for ref in row["source_primitive_refs"]}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "layer": LAYER,
        "page_ref": records[0].get("page_ref") if records else None,
        "evaluations": records,
        "summary": {
            "evaluated_entity_count": len(records),
            "evaluated_source_primitive_count": len({ref for row in records
                                                      for ref in row["source_primitive_refs"]}),
            "hatch_entity_count": sum(row["hatch_certificate"]["state"] == "accepted"
                                      for row in records),
            "hatch_source_primitive_count": len(hatch_refs),
            "architectural_boundary_entity_count": sum(
                row["architectural_boundary_certificate"]["state"] == "accepted"
                for row in records),
            "architectural_boundary_review_candidate_count": sum(
                row["architectural_boundary_certificate"]["state"] == "review_candidate"
                for row in records),
            "architectural_boundary_source_primitive_count": len(architectural_refs),
        },
        "measurement_contract": {
            "every_supplied_entity_evaluated_once": len(records) == len(entities)
                and len({row["entity_ref"] for row in records}) == len(entities),
            "hatch_method": "regular parallel same-style lattice with bounded pitch residual",
            "architectural_methods": [
                "M1 opposing-label grid-axis coincidence",
                "closed rectilinear drawing boundary",
            ],
            "precision_display_points": tolerance,
            "fixed_page_coordinates_used": False,
            "route_or_system_identity_used_to_nominate_negative_geometry": False,
        },
        "acceptance_gate": {
            "every_entity_measured": len(records) == len(entities)
                and all(row["state"] == "measured" for row in records),
            "literal_zero_counts_used": False,
            "status": "accepted_measured_negative_content_certificates",
        },
    }
    return payload


def validate_negative_content_certificates(payload: Mapping[str, Any]) -> list[str]:
    errors = []
    rows = payload.get("evaluations", [])
    if payload.get("layer") != LAYER:
        errors.append("unexpected layer")
    if len(rows) != len({row.get("entity_ref") for row in rows}):
        errors.append("negative-content entities are missing or duplicated")
    if not all(row.get("state") == "measured" for row in rows):
        errors.append("every negative-content evaluation must be measured")
    for row in rows:
        if row.get("hatch_certificate", {}).get("state") not in {"accepted", "no_match"}:
            errors.append(f"invalid hatch state: {row.get('entity_ref')}")
        if row.get("architectural_boundary_certificate", {}).get("state") not in {
                "accepted", "review_candidate", "no_match"}:
            errors.append(f"invalid architectural state: {row.get('entity_ref')}")
    if payload.get("measurement_contract", {}).get(
            "route_or_system_identity_used_to_nominate_negative_geometry") is not False:
        errors.append("negative certificates cannot use route identity")
    return errors
