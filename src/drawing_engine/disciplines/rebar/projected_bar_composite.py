"""Provenance-preserving composite projected-bar proposals.

Native PDF paths remain immutable observations. This module may group two
near-coincident, uniquely paired, same-mark path components as two strokes of
one projected bar representation. Distant or non-unique same-mark pairs stay
review candidates and never become physical placements or quantities.
"""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Sequence


def _digest(*parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()[:20]


def _angle_difference(left: float, right: float) -> float:
    difference = abs(float(left) - float(right)) % 180.0
    return min(difference, 180.0 - difference)


def _style_compatible(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_width = float(left.get("width_pt") or 0.0)
    right_width = float(right.get("width_pt") or 0.0)
    width_tolerance = max(0.35, 0.25 * max(left_width, right_width, 0.1))
    return (
        abs(left_width - right_width) <= width_tolerance
        and str(left.get("dashes") or "") == str(right.get("dashes") or "")
        and str(left.get("stroke")) == str(right.get("stroke"))
    )


def _representative(
    component: Mapping[str, Any],
    fragments: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    rows = [
        fragments[str(fragment_id)]
        for fragment_id in component.get("fragment_ids", []) or []
        if str(fragment_id) in fragments
        and fragments[str(fragment_id)].get("geometry", {}).get("kind") == "line"
    ]
    return max(
        rows,
        key=lambda item: float(item.get("geometry", {}).get("length_points") or 0.0),
        default=None,
    )


def _marks(component: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        str(mark)
        for mark in component.get("mark_hypotheses", []) or []
        if mark
    } | {
        str(mark)
        for row in rows
        for mark in row.get("mark_hypotheses", []) or []
        if mark
    }


def _projection_metrics(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    left_geometry = left["geometry"]
    right_geometry = right["geometry"]
    left_points = left_geometry["points_display"]
    right_points = right_geometry["points_display"]
    angle = math.radians(float(left_geometry["angle_deg"]))
    direction = (math.cos(angle), math.sin(angle))
    normal = (-direction[1], direction[0])

    def interval(points: Sequence[Sequence[float]], axis: tuple[float, float]) -> tuple[float, float]:
        values = [float(point[0]) * axis[0] + float(point[1]) * axis[1] for point in points]
        return min(values), max(values)

    left_axis = interval(left_points, direction)
    right_axis = interval(right_points, direction)
    left_normal = interval(left_points, normal)
    right_normal = interval(right_points, normal)
    left_length = max(left_axis[1] - left_axis[0], 0.0)
    right_length = max(right_axis[1] - right_axis[0], 0.0)
    overlap = max(0.0, min(left_axis[1], right_axis[1]) - max(left_axis[0], right_axis[0]))
    normal_left = (left_normal[0] + left_normal[1]) / 2.0
    normal_right = (right_normal[0] + right_normal[1]) / 2.0
    return {
        "angle_difference_deg": _angle_difference(
            float(left_geometry["angle_deg"]), float(right_geometry["angle_deg"])
        ),
        "length_ratio": min(left_length, right_length) / max(left_length, right_length, 0.001),
        "axis_overlap_ratio": overlap / max(min(left_length, right_length), 0.001),
        "axis_endpoint_mismatch_points": abs(left_axis[0] - right_axis[0])
        + abs(left_axis[1] - right_axis[1]),
        "perpendicular_separation_points": abs(normal_left - normal_right),
        "max_visible_length_points": max(left_length, right_length),
    }


def build_composite_projected_bar_proposals(
    fragments: Sequence[Mapping[str, Any]],
    components: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return accepted double-stroke groups and unresolved same-mark pairs."""

    fragment_by_id = {str(item["id"]): item for item in fragments}
    eligible_components = []
    for component in components:
        if len(component.get("view_ids", []) or []) != 1:
            continue
        representative = _representative(component, fragment_by_id)
        if representative is None:
            continue
        component_fragments = [
            fragment_by_id[str(fragment_id)]
            for fragment_id in component.get("fragment_ids", []) or []
            if str(fragment_id) in fragment_by_id
        ]
        marks = _marks(component, component_fragments)
        if marks:
            eligible_components.append((component, representative, component_fragments, marks))

    candidates = []
    local_pair_ids_by_component: defaultdict[str, set[str]] = defaultdict(set)
    for left_index, (left_component, left, left_rows, left_marks) in enumerate(eligible_components):
        for right_component, right, right_rows, right_marks in eligible_components[left_index + 1 :]:
            if left_component["view_ids"] != right_component["view_ids"]:
                continue
            shared_marks = sorted(left_marks & right_marks, key=lambda value: (len(value), value))
            if len(shared_marks) != 1:
                continue
            if not _style_compatible(left.get("style", {}), right.get("style", {})):
                continue
            metrics = _projection_metrics(left, right)
            width = max(
                float(left.get("style", {}).get("width_pt") or 0.0),
                float(right.get("style", {}).get("width_pt") or 0.0),
            )
            endpoint_tolerance = max(2.0, 0.02 * metrics["max_visible_length_points"])
            separation_limit = max(3.5, 1.25 * width)
            local_double_stroke = (
                metrics["angle_difference_deg"] <= 2.0
                and metrics["length_ratio"] >= 0.98
                and metrics["axis_overlap_ratio"] >= 0.98
                and metrics["axis_endpoint_mismatch_points"] <= endpoint_tolerance
                and 0.2 <= metrics["perpendicular_separation_points"] <= separation_limit
            )
            component_ids = sorted((str(left_component["id"]), str(right_component["id"])))
            pair_id = "composite_projected_bar." + _digest(shared_marks[0], component_ids)
            candidate = {
                "id": pair_id,
                "mark": shared_marks[0],
                "view_id": str(left_component["view_ids"][0]),
                "component_ids": component_ids,
                "fragment_ids": sorted({str(row["id"]) for row in [*left_rows, *right_rows]}),
                "primitive_refs": sorted(
                    {
                        str(reference)
                        for row in [*left_rows, *right_rows]
                        for reference in (row.get("primitive_ref"), row.get("source_path_ref"))
                        if reference is not None
                    }
                ),
                "geometry_metrics": {key: round(value, 6) for key, value in metrics.items()},
                "local_double_stroke_candidate": local_double_stroke,
                "state": "review_candidate",
                "epistemic_state": "unknown",
                "certificate": None,
                "reason_not_accepted": (
                    "near-coincident geometry is not a unique one-to-one pairing"
                    if local_double_stroke
                    else "same mark is candidate generation only; paths lack a local double-stroke certificate"
                ),
            }
            candidates.append(candidate)
            if local_double_stroke:
                for component_id in component_ids:
                    local_pair_ids_by_component[component_id].add(pair_id)

    for candidate in candidates:
        if not candidate["local_double_stroke_candidate"]:
            continue
        component_ids = candidate["component_ids"]
        unique = all(local_pair_ids_by_component[component_id] == {candidate["id"]} for component_id in component_ids)
        if unique:
            candidate.update(
                {
                    "state": "accepted",
                    "epistemic_state": "derived",
                    "certificate": "unique_near_coincident_parallel_same_mark_double_stroke",
                    "reason_not_accepted": None,
                }
            )

    candidates.sort(key=lambda item: (item["view_id"], len(item["mark"]), item["mark"], item["component_ids"]))
    return {
        "schema_version": "0.1.0",
        "layer": "composite_projected_bar_proposals",
        "proposals": candidates,
        "summary": {
            "proposal_count": len(candidates),
            "accepted_count": sum(item["state"] == "accepted" for item in candidates),
            "review_candidate_count": sum(item["state"] != "accepted" for item in candidates),
        },
        "contract": {
            "native_fragments_mutated": False,
            "accepted_group_is_projected_representation_only": True,
            "physical_placement_identity_inferred": False,
            "quantities_changed": False,
            "same_mark_alone_accepts_group": False,
            "schedule_values_used": False,
        },
    }
