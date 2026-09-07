"""Fail-closed native-contour correspondence across orthographic views."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from copy import deepcopy
import math
from itertools import combinations
from typing import Any

from src.drawing_engine.core.vector_topology import unique_points


def _view_contours(
    views: Iterable[Mapping[str, Any]],
    contours: Mapping[str, Mapping[str, Any]],
    native_segments: Iterable[Mapping[str, Any]] = (),
) -> dict[str, list[Mapping[str, Any]]]:
    segments_by_id = {
        str(item["id"]): item
        for item in native_segments
        if item.get("id") is not None
    }
    output = {}
    for view in views:
        rows = []
        for contour_id in view.get("contour_refs", []) or []:
            contour = contours.get(str(contour_id))
            if contour is None:
                continue
            topology = contour.get("topology", {})
            closed = bool(contour.get("closed"))
            simple_open = (
                str(contour.get("kind", "")).startswith("composite_open")
                and int(topology.get("endpoint_vertex_count", 0)) == 2
                and int(topology.get("branch_vertex_count", 0)) == 0
            )
            if int(topology.get("segment_count", 0)) < 3 or not (closed or simple_open):
                continue
            # Composite topology records deliberately reference the immutable
            # page-global segments instead of duplicating them.  Rehydrate the
            # geometry for reprojection while preserving those exact IDs.
            if not contour.get("segments_display") and contour.get("segment_refs"):
                segment_rows = [
                    segments_by_id[str(ref)]
                    for ref in contour.get("segment_refs", [])
                    if str(ref) in segments_by_id
                ]
                if len(segment_rows) == len(contour.get("segment_refs", [])):
                    contour = {**dict(contour), "segments_display": segment_rows}
            rows.append(contour)
        output[str(view["id"])] = rows
    return output


def _axis_candidates(
    rows: Iterable[Mapping[str, Any]],
    view: Mapping[str, Any],
    orientation: str,
    ownership_by_contour: Mapping[str, list[Mapping[str, Any]]],
) -> list[Mapping[str, Any]]:
    view_box = view.get("bbox_display") or (0, 0, 0, 0)
    axis_index = 0 if orientation == "horizontal" else 1
    view_extent = float(view_box[axis_index + 2]) - float(view_box[axis_index])
    output = []
    for contour in rows:
        box = contour.get("bbox_display") or (0, 0, 0, 0)
        extent = float(box[axis_index + 2]) - float(box[axis_index])
        owned = bool(_owner_support(str(contour["id"]), orientation, ownership_by_contour))
        if view_extent <= 0 or extent >= 0.20 * view_extent or owned:
            output.append(contour)
    return output


def _orientation(mode: str) -> tuple[str, str, str]:
    if mode == "cut_horizontal":
        return "horizontal", "horizontal", "u"
    if mode == "cut_vertical":
        # A vertical section trace preserves the displayed vertical axis in
        # an end/transverse projection.  The orthogonal 200 mm extent is the
        # extrusion depth, not the shared profile axis.
        return "vertical", "vertical", "v"
    if mode == "pair_side":
        return "vertical", "vertical", "v"
    return "horizontal", "horizontal", "u"


def _signature(
    contour: Mapping[str, Any],
    orientation: str,
    points_per_mm: float,
) -> dict[str, Any] | None:
    coordinate_index = 0 if orientation == "horizontal" else 1
    coordinates = [
        float(point[coordinate_index]) / points_per_mm
        for segment in contour.get("segments_display", []) or []
        for point in (segment.get("start_display"), segment.get("end_display"))
        if point is not None
    ]
    if len(coordinates) < 3:
        return None
    tolerance = max(0.5, 0.8 / points_per_mm)
    coordinates = unique_points(coordinates, tolerance)
    if len(coordinates) < 2:
        return None
    minimum, maximum = coordinates[0], coordinates[-1]
    return {
        "coordinates_mm": coordinates,
        "normalised_forward_mm": [value - minimum for value in coordinates],
        "normalised_reverse_mm": [maximum - value for value in reversed(coordinates)],
        "minimum_mm_in_view_gauge": minimum,
        "maximum_mm_in_view_gauge": maximum,
        "span_mm": maximum - minimum,
        "level_count": len(coordinates),
    }


def _projected_silhouette_signature(
    contour: Mapping[str, Any],
    orientation: str,
    points_per_mm: float,
) -> dict[str, Any] | None:
    """Project one closed native contour onto a display axis.

    This is intentionally narrower than a bounding-box comparison: the
    extrema are derived from the immutable contour segments, retain their
    exact geometry references, and are used only inside an independently
    accepted title/marker relation when the annotation trace lies outside the
    parent contour.  The caller still requires a unique child contour and an
    independently dimensioned transverse extent.
    """

    if not contour.get("closed"):
        return None
    coordinate_index = 0 if orientation == "horizontal" else 1
    segment_rows = list(contour.get("segments_display", []) or [])
    coordinates = [
        float(point[coordinate_index]) / points_per_mm
        for segment in segment_rows
        for point in (segment.get("start_display"), segment.get("end_display"))
        if point is not None
    ]
    if len(coordinates) < 3:
        return None
    minimum, maximum = min(coordinates), max(coordinates)
    if maximum - minimum <= 0:
        return None
    geometry_refs = sorted(
        {
            str(segment.get("id") or segment.get("primitive_ref"))
            for segment in segment_rows
            if segment.get("id") or segment.get("primitive_ref")
        }
    )
    return {
        "coordinates_mm": [minimum, maximum],
        "normalised_forward_mm": [0.0, maximum - minimum],
        "normalised_reverse_mm": [0.0, maximum - minimum],
        "minimum_mm_in_view_gauge": minimum,
        "maximum_mm_in_view_gauge": maximum,
        "span_mm": maximum - minimum,
        "level_count": 2,
        "basis": "native closed-contour orthogonal silhouette projection",
        "child_geometry_refs": geometry_refs,
    }


def _slice_signature(
    contour: Mapping[str, Any],
    cut_orientation: str,
    cut_coordinate_display: float,
    points_per_mm: float,
) -> dict[str, Any] | None:
    """Intersect a native profile with the accepted parent-view cut line."""

    intersections = []
    intersection_refs = set()
    for segment in contour.get("segments_display", []) or []:
        points = segment.get("sample_points_display") or [
            segment.get("start_display"),
            segment.get("end_display"),
        ]
        points = [point for point in points if point is not None]
        for start, end in zip(points, points[1:]):
            x0, y0 = map(float, start)
            x1, y1 = map(float, end)
            if cut_orientation == "horizontal":
                if abs(y1 - y0) <= 1e-9:
                    if abs(y0 - cut_coordinate_display) <= 0.8:
                        intersections.extend((x0, x1))
                        intersection_refs.add(str(segment.get("id") or segment.get("primitive_ref") or ""))
                    continue
                ratio = (cut_coordinate_display - y0) / (y1 - y0)
                if -1e-9 <= ratio <= 1.0 + 1e-9:
                    intersections.append(x0 + ratio * (x1 - x0))
                    intersection_refs.add(str(segment.get("id") or segment.get("primitive_ref") or ""))
            else:
                if abs(x1 - x0) <= 1e-9:
                    if abs(x0 - cut_coordinate_display) <= 0.8:
                        intersections.extend((y0, y1))
                        intersection_refs.add(str(segment.get("id") or segment.get("primitive_ref") or ""))
                    continue
                ratio = (cut_coordinate_display - x0) / (x1 - x0)
                if -1e-9 <= ratio <= 1.0 + 1e-9:
                    intersections.append(y0 + ratio * (y1 - y0))
                    intersection_refs.add(str(segment.get("id") or segment.get("primitive_ref") or ""))
    tolerance = max(0.5, 0.8 / points_per_mm)
    coordinates = unique_points((value / points_per_mm for value in intersections), tolerance)
    if len(coordinates) < 2:
        return None
    minimum, maximum = coordinates[0], coordinates[-1]
    return {
        "coordinates_mm": coordinates,
        "normalised_forward_mm": [value - minimum for value in coordinates],
        "normalised_reverse_mm": [maximum - value for value in reversed(coordinates)],
        "minimum_mm_in_view_gauge": minimum,
        "maximum_mm_in_view_gauge": maximum,
        "span_mm": maximum - minimum,
        "level_count": len(coordinates),
        "basis": "native profile intersection at accepted cutting-plane coordinate",
        "cut_coordinate_display": round(cut_coordinate_display, 6),
        "parent_geometry_refs": sorted(intersection_refs - {""}),
    }


def _trace_signature(parent_cut: Mapping[str, Any], points_per_mm: float) -> dict[str, Any] | None:
    box = parent_cut.get("trace_bbox_display") or []
    if len(box) != 4:
        return None
    coordinate_index = 0 if parent_cut.get("orientation") == "horizontal" else 1
    coordinates = sorted(
        [float(box[coordinate_index]) / points_per_mm, float(box[coordinate_index + 2]) / points_per_mm]
    )
    if coordinates[1] - coordinates[0] <= 0:
        return None
    return {
        "coordinates_mm": coordinates,
        "normalised_forward_mm": [0.0, coordinates[1] - coordinates[0]],
        "normalised_reverse_mm": [0.0, coordinates[1] - coordinates[0]],
        "minimum_mm_in_view_gauge": coordinates[0],
        "maximum_mm_in_view_gauge": coordinates[1],
        "span_mm": coordinates[1] - coordinates[0],
        "level_count": 2,
        "basis": "accepted cutting-plane terminal span",
        "cut_coordinate_display": round(float(parent_cut["coordinate_display"]), 6),
    }


def _segment_cut_intersections(
    segment: Mapping[str, Any],
    cut_orientation: str,
    coordinate: float,
) -> list[float]:
    points = segment.get("sample_points_display") or [segment.get("start_display"), segment.get("end_display")]
    points = [point for point in points if point is not None]
    output = []
    for start, end in zip(points, points[1:]):
        x0, y0 = map(float, start)
        x1, y1 = map(float, end)
        if cut_orientation == "horizontal":
            if abs(y1 - y0) <= 1e-9:
                continue
            ratio = (coordinate - y0) / (y1 - y0)
            if -1e-9 <= ratio <= 1.0 + 1e-9:
                output.append(x0 + ratio * (x1 - x0))
        else:
            if abs(x1 - x0) <= 1e-9:
                continue
            ratio = (coordinate - x0) / (x1 - x0)
            if -1e-9 <= ratio <= 1.0 + 1e-9:
                output.append(y0 + ratio * (y1 - y0))
    return output


def _style_key(segment: Mapping[str, Any]) -> tuple[Any, ...]:
    style = segment.get("style", {}) or {}
    stroke = style.get("stroke")
    return (
        None if stroke is None else tuple(round(float(value), 2) for value in stroke),
        None if style.get("width") is None else round(float(style["width"]), 2),
        str(style.get("dash") or "").replace(" ", ""),
    )


def _cut_edge_signatures(
    native_segments: Iterable[Mapping[str, Any]],
    view_box: Iterable[float],
    parent_cut: Mapping[str, Any],
    points_per_mm: float,
) -> list[dict[str, Any]]:
    box = list(map(float, view_box))
    cut_orientation = str(parent_cut["orientation"])
    coordinate = float(parent_cut["coordinate_display"])
    excluded = set(map(str, parent_cut.get("primitive_refs", [])))
    grouped: dict[tuple[Any, ...], list[tuple[float, str]]] = defaultdict(list)
    for segment in native_segments:
        if str(segment.get("primitive_ref")) in excluded or str(segment.get("drawing_ref")) in excluded:
            continue
        segment_box = segment.get("bbox_display") or (0, 0, 0, 0)
        if (
            float(segment_box[2]) < box[0]
            or float(segment_box[0]) > box[2]
            or float(segment_box[3]) < box[1]
            or float(segment_box[1]) > box[3]
        ):
            continue
        for value in _segment_cut_intersections(segment, cut_orientation, coordinate):
            if cut_orientation == "horizontal" and not box[0] - 1 <= value <= box[2] + 1:
                continue
            if cut_orientation == "vertical" and not box[1] - 1 <= value <= box[3] + 1:
                continue
            grouped[_style_key(segment)].append((value, str(segment["id"])))
    signatures = []
    for style, observations in grouped.items():
        levels: list[dict[str, Any]] = []
        for coordinate_value, segment_ref in sorted(observations):
            if levels and abs(coordinate_value - levels[-1]["coordinate_display"]) <= 0.8:
                levels[-1]["segment_refs"].append(segment_ref)
                continue
            levels.append({"coordinate_display": coordinate_value, "segment_refs": [segment_ref]})
        for left, right in combinations(levels, 2):
            values = sorted(
                [left["coordinate_display"] / points_per_mm, right["coordinate_display"] / points_per_mm]
            )
            signatures.append(
                {
                    "coordinates_mm": values,
                    "normalised_forward_mm": [0.0, values[1] - values[0]],
                    "normalised_reverse_mm": [0.0, values[1] - values[0]],
                    "minimum_mm_in_view_gauge": values[0],
                    "maximum_mm_in_view_gauge": values[1],
                    "span_mm": values[1] - values[0],
                    "level_count": 2,
                    "basis": "style-compatible native edge pair intersected by accepted cutting plane",
                    "cut_coordinate_display": round(coordinate, 6),
                    "parent_geometry_refs": sorted({*left["segment_refs"], *right["segment_refs"]}),
                    "style_signature": list(style),
                }
            )
    return signatures


def _view_edge_signatures(
    native_segments: Iterable[Mapping[str, Any]],
    view_box: Iterable[float],
    orientation: str,
    points_per_mm: float,
) -> list[dict[str, Any]]:
    box = list(map(float, view_box))
    coordinate_index = 0 if orientation == "horizontal" else 1
    view_extent = box[coordinate_index + 2] - box[coordinate_index]
    output = []
    for segment in native_segments:
        if segment.get("axis") != orientation:
            continue
        segment_box = list(map(float, segment.get("bbox_display") or (0, 0, 0, 0)))
        if (
            segment_box[0] < box[0] - 1
            or segment_box[2] > box[2] + 1
            or segment_box[1] < box[1] - 1
            or segment_box[3] > box[3] + 1
        ):
            continue
        start = float(segment["start_display"][coordinate_index])
        end = float(segment["end_display"][coordinate_index])
        if abs(end - start) < 0.20 * max(view_extent, 1.0):
            continue
        values = sorted([start / points_per_mm, end / points_per_mm])
        output.append(
            {
                "coordinates_mm": values,
                "normalised_forward_mm": [0.0, values[1] - values[0]],
                "normalised_reverse_mm": [0.0, values[1] - values[0]],
                "minimum_mm_in_view_gauge": values[0],
                "maximum_mm_in_view_gauge": values[1],
                "span_mm": values[1] - values[0],
                "level_count": 2,
                "basis": "long native edge inside the child projection",
                "child_geometry_refs": [str(segment["id"])],
                "style_signature": list(_style_key(segment)),
            }
        )
    return output


def _hausdorff(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return math.inf
    return max(
        max(min(abs(value - candidate) for candidate in right) for value in left),
        max(min(abs(value - candidate) for candidate in left) for value in right),
    )


def _metric_signature(signature: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(round(float(value), 6) for value in signature.get("normalised_forward_mm", []) or []),
        tuple(round(float(value), 6) for value in signature.get("normalised_reverse_mm", []) or []),
        round(float(signature.get("span_mm") or 0.0), 6),
        int(signature.get("level_count") or 0),
    )


def _transform_alternatives(candidate: Mapping[str, Any]) -> tuple[tuple[int, float], ...]:
    transform = candidate.get("signed_transform", {}) or {}
    alternatives = transform.get("candidates", []) or []
    if transform.get("state") == "resolved":
        alternatives = [transform.get("child_coordinate_to_parent", {}) or {}]
    output = []
    for item in alternatives:
        sign, offset = item.get("sign"), item.get("offset_mm")
        if sign not in {-1, 1} or not isinstance(offset, (int, float)):
            return ()
        output.append((int(sign), round(float(offset), 6)))
    return tuple(sorted(output))


def _representation_kind(candidate: Mapping[str, Any]) -> str:
    parent_basis = str(candidate.get("parent_signature", {}).get("basis") or "")
    if candidate.get("parent_contour_id") is not None and "cutting-plane" in parent_basis:
        return "closed_contour"
    if candidate.get("parent_contour_id") is None and parent_basis == "style-compatible native edge pair intersected by accepted cutting plane":
        return "native_edge_pair"
    return "other"


def _canonical_correspondence_key(candidate: Mapping[str, Any]) -> tuple[Any, ...] | None:
    """Return an exact semantic key, never a span-only similarity key."""

    parent = candidate.get("parent_signature", {}) or {}
    child = candidate.get("child_signature", {}) or {}
    parent_edges = tuple(sorted(map(str, candidate.get("parent_geometry_refs", []) or [])))
    alternatives = _transform_alternatives(candidate)
    cut_coordinate = parent.get("cut_coordinate_display")
    if (
        candidate.get("state") != "pass"
        or _representation_kind(candidate) not in {"closed_contour", "native_edge_pair"}
        or not parent_edges
        or not isinstance(cut_coordinate, (int, float))
        or len(alternatives) not in {1, 2}
    ):
        return None
    return (
        round(float(cut_coordinate), 6),
        parent_edges,
        _metric_signature(parent),
        str(candidate.get("child_contour_id") or ""),
        tuple(sorted(map(str, candidate.get("child_geometry_refs", []) or []))),
        _metric_signature(child),
        alternatives,
    )


def _canonicalize_correspondence_candidates(
    candidates: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Collapse only proven cross-representation duplicates and merge provenance."""

    rows = [deepcopy(dict(item)) for item in candidates]
    by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    unkeyed = []
    for item in rows:
        key = _canonical_correspondence_key(item)
        if key is None:
            unkeyed.append(item)
        else:
            by_key[key].append(item)
    canonical = list(unkeyed)
    collapsed = 0
    canonicalized_groups = 0
    for key in sorted(by_key, key=repr):
        group = by_key[key]
        kinds = {_representation_kind(item) for item in group}
        if kinds != {"closed_contour", "native_edge_pair"}:
            canonical.extend(group)
            continue
        representative = min(
            group,
            key=lambda item: (
                _representation_kind(item) != "closed_contour",
                str(item.get("parent_contour_id") or ""),
                str(item.get("child_contour_id") or ""),
                tuple(item.get("primitive_refs", []) or []),
            ),
        )
        representations = sorted(
            (
                {
                    "kind": _representation_kind(item),
                    "parent_contour_id": item.get("parent_contour_id"),
                    "child_contour_id": item.get("child_contour_id"),
                    "parent_geometry_refs": sorted(map(str, item.get("parent_geometry_refs", []) or [])),
                    "child_geometry_refs": sorted(map(str, item.get("child_geometry_refs", []) or [])),
                    "primitive_refs": sorted(map(str, item.get("primitive_refs", []) or [])),
                }
                for item in group
            ),
            key=lambda item: (
                item["kind"],
                str(item["parent_contour_id"] or ""),
                str(item["child_contour_id"] or ""),
                item["primitive_refs"],
            ),
        )
        representative["primitive_refs"] = sorted(
            {str(ref) for item in group for ref in item.get("primitive_refs", []) or []}
        )
        representative["parent_geometry_refs"] = sorted(
            {str(ref) for item in group for ref in item.get("parent_geometry_refs", []) or []}
        )
        representative["child_geometry_refs"] = sorted(
            {str(ref) for item in group for ref in item.get("child_geometry_refs", []) or []}
        )
        representative["owned_dimension_refs"] = sorted(
            {str(ref) for item in group for ref in item.get("owned_dimension_refs", []) or []}
        )
        representative["canonicalization"] = {
            "state": "canonical",
            "representation_count": len(group),
            "representation_kinds": sorted(kinds),
            "canonical_native_edge_support": list(key[1]),
            "cut_coordinate_display": key[0],
            "signed_transform_alternatives": [
                {"sign": sign, "offset_mm": offset} for sign, offset in key[6]
            ],
            "source_representations": representations,
            "provenance_merged": True,
            "mirror_resolved_by_canonicalization": False,
        }
        canonical.append(representative)
        collapsed += len(group) - 1
        canonicalized_groups += 1
    canonical.sort(
        key=lambda item: (
            item["state"] != "pass",
            item["objective"],
            str(item.get("parent_contour_id") or ""),
            str(item.get("child_contour_id") or ""),
            tuple(item.get("primitive_refs", []) or []),
        )
    )
    return canonical, {
        "representation_candidate_count": len(rows),
        "representation_passing_candidate_count": sum(
            item.get("state") == "pass" for item in rows
        ),
        "canonical_candidate_count": len(canonical),
        "canonical_correspondence_count": sum(
            item.get("state") == "pass" for item in canonical
        ),
        "canonicalized_group_count": canonicalized_groups,
        "collapsed_representation_count": collapsed,
    }


def _owner_support(
    contour_id: str,
    orientation: str,
    ownership_by_contour: Mapping[str, list[Mapping[str, Any]]],
) -> list[Mapping[str, Any]]:
    return [
        item
        for item in ownership_by_contour.get(contour_id, [])
        if item.get("status") == "accepted"
        and item.get("orientation") == orientation
        and item.get("semantic_role") == "overall_shared_axis_span"
    ]


def _pair(
    parent: Mapping[str, Any] | None,
    child: Mapping[str, Any] | None,
    parent_orientation: str,
    child_orientation: str,
    parent_scale: float,
    child_scale: float,
    ownership_by_contour: Mapping[str, list[Mapping[str, Any]]],
    parent_cut: Mapping[str, Any] | None = None,
    parent_signature_override: Mapping[str, Any] | None = None,
    child_signature_override: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    scale_transfer = None
    parent_signature = dict(parent_signature_override) if parent_signature_override is not None else None
    if parent is not None and parent_signature is None:
        sliced_signature = (
            _slice_signature(
                parent,
                str(parent_cut["orientation"]),
                float(parent_cut["coordinate_display"]),
                parent_scale,
            )
            if parent_cut is not None
            else None
        )
        parent_signature = sliced_signature or (
            _signature(parent, parent_orientation, parent_scale)
            if parent_cut is None
            else None
        )
        if sliced_signature is None and parent_cut is not None and child is not None:
            raw_parent = _projected_silhouette_signature(parent, parent_orientation, 1.0)
            raw_child = _projected_silhouette_signature(child, child_orientation, 1.0)
            if raw_parent is not None and raw_child is not None:
                display_residual = abs(raw_parent["span_mm"] - raw_child["span_mm"])
                display_tolerance = max(0.8, 0.015 * max(raw_parent["span_mm"], raw_child["span_mm"]))
                if display_residual <= display_tolerance:
                    parent_signature = _projected_silhouette_signature(
                        parent, parent_orientation, child_scale
                    )
                    child_signature_override = _projected_silhouette_signature(
                        child, child_orientation, child_scale
                    )
                    scale_transfer = {
                        "status": "passed",
                        "source_view_scale_points_per_mm": child_scale,
                        "parent_scale_points_per_mm": child_scale,
                        "display_span_residual_points": round(display_residual, 6),
                        "display_span_tolerance_points": round(display_tolerance, 6),
                        "basis": "unique native silhouette reprojection at the independently dimensioned section scale",
                    }
    if parent_signature is None and parent_cut is not None:
        parent_signature = _trace_signature(parent_cut, parent_scale)
    child_signature = (
        dict(child_signature_override)
        if child_signature_override is not None
        else _signature(child, child_orientation, child_scale) if child is not None else None
    )
    if parent_signature is None or child_signature is None:
        return None
    forward_residual = _hausdorff(
        parent_signature["normalised_forward_mm"],
        child_signature["normalised_forward_mm"],
    )
    reverse_residual = _hausdorff(
        parent_signature["normalised_forward_mm"],
        child_signature["normalised_reverse_mm"],
    )
    signature_residual = min(forward_residual, reverse_residual)
    span_residual = abs(parent_signature["span_mm"] - child_signature["span_mm"])
    tolerance = max(2.0, 0.015 * max(parent_signature["span_mm"], child_signature["span_mm"]))
    parent_owners = (
        _owner_support(str(parent["id"]), parent_orientation, ownership_by_contour)
        if parent is not None
        else []
    )
    child_owners = (
        _owner_support(str(child["id"]), child_orientation, ownership_by_contour)
        if child is not None
        else []
    )
    owned_values = [float(item["value_mm"]) for item in (*parent_owners, *child_owners)]
    owned_residual = max(owned_values) - min(owned_values) if len(owned_values) >= 2 else None
    owner_conflict = owned_residual is not None and owned_residual > tolerance
    passed = span_residual <= tolerance and signature_residual <= tolerance and not owner_conflict
    sign_margin = abs(forward_residual - reverse_residual)
    sign = 1 if forward_residual < reverse_residual else -1
    sign_state = "resolved" if passed and sign_margin > max(1.0, tolerance / 2.0) else "unresolved"
    if sign > 0:
        offset = (
            parent_signature["minimum_mm_in_view_gauge"]
            - child_signature["minimum_mm_in_view_gauge"]
        )
    else:
        offset = (
            parent_signature["minimum_mm_in_view_gauge"]
            + child_signature["maximum_mm_in_view_gauge"]
        )
    objective = span_residual + signature_residual - min(2, len(parent_owners) + len(child_owners)) * 0.5
    return {
        "parent_contour_id": None if parent is None else str(parent["id"]),
        "child_contour_id": None if child is None else str(child["id"]),
        "state": "pass" if passed else "fail",
        "parent_signature": parent_signature,
        "child_signature": child_signature,
        "span_residual_mm": round(span_residual, 6),
        "path_signature_residual_mm": round(signature_residual, 6),
        "forward_residual_mm": round(forward_residual, 6),
        "reverse_residual_mm": round(reverse_residual, 6),
        "tolerance_mm": round(tolerance, 6),
        "owned_dimension_residual_mm": None if owned_residual is None else round(owned_residual, 6),
        "owned_dimension_refs": [str(item["id"]) for item in (*parent_owners, *child_owners)],
        "signed_transform": {
            "state": sign_state,
            "child_coordinate_to_parent": {
                "sign": sign if sign_state == "resolved" else None,
                "offset_mm": round(offset, 6) if sign_state == "resolved" else None,
            },
            "candidates": (
                []
                if sign_state == "resolved"
                else [
                    {
                        "sign": 1,
                        "offset_mm": round(
                            parent_signature["minimum_mm_in_view_gauge"]
                            - child_signature["minimum_mm_in_view_gauge"],
                            6,
                        ),
                    },
                    {
                        "sign": -1,
                        "offset_mm": round(
                            parent_signature["minimum_mm_in_view_gauge"]
                            + child_signature["maximum_mm_in_view_gauge"],
                            6,
                        ),
                    },
                ]
            ),
            "reason": (
                "asymmetric native contour levels select one signed alignment"
                if sign_state == "resolved"
                else "forward and mirrored native contour alignments remain equivalent"
            ),
        },
        "objective": round(objective, 6),
        "projection_mode": (
            "orthogonal_silhouette"
            if scale_transfer is not None
            else "cut_intersection" if parent_cut is not None else "path_alignment"
        ),
        "scale_transfer_certificate": scale_transfer,
        "primitive_refs": sorted(
            {
                *(parent.get("primitive_refs", []) if parent is not None else parent_cut.get("primitive_refs", []) if parent_cut else []),
                *(child.get("primitive_refs", []) if child is not None else []),
                *[
                    str(segment.get("primitive_ref"))
                    for contour in tuple(item for item in (parent, child) if item is not None)
                    for segment in contour.get("segments_display", []) or []
                    if segment.get("primitive_ref")
                ],
                *(
                    parent_signature.get("parent_geometry_refs", [])
                    if parent_signature is not None
                    else []
                ),
                *child_signature.get("child_geometry_refs", []),
            }
        ),
        "parent_geometry_refs": list(parent_signature.get("parent_geometry_refs", [])),
        "child_geometry_refs": list(child_signature.get("child_geometry_refs", [])),
    }


def solve_contour_correspondence(
    frames: list[dict[str, Any]],
    shared_coordinates: dict[str, Any],
    contours: Iterable[Mapping[str, Any]],
    views: Iterable[Mapping[str, Any]],
    dimension_ownership: Mapping[str, Any] | None = None,
    native_segments: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Attach unique path-derived correspondence records to coordinate scopes."""

    native_segments = list(native_segments)
    contours_by_id = {str(item["id"]): item for item in contours}
    views_by_id = {str(item["id"]): item for item in views}
    contours_by_view = _view_contours(views, contours_by_id, native_segments)
    frames_by_view = {str(item["view_id"]): item for item in frames}
    constraints_by_id = {
        str(item["id"]): item for item in shared_coordinates.get("constraints", [])
    }
    ownership_by_contour: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for attachment in (dimension_ownership or {}).get("attachments", []) or []:
        for owner_ref in attachment.get("owner_entity_refs", []) or []:
            if str(owner_ref) in contours_by_id:
                ownership_by_contour[str(owner_ref)].append(attachment)
    all_records = []
    for scope in shared_coordinates.get("scopes", []) or []:
        scope_records = []
        for constraint_id in scope.get("constraint_ids", []) or []:
            constraint = constraints_by_id.get(str(constraint_id))
            if constraint is None:
                continue
            parent_id, child_id = str(constraint["parent_view_id"]), str(constraint["child_view_id"])
            parent_frame, child_frame = frames_by_view.get(parent_id), frames_by_view.get(child_id)
            if parent_frame is None or child_frame is None:
                continue
            parent_scale = parent_frame.get("scale", {}).get("value_points_per_mm")
            child_scale = child_frame.get("scale", {}).get("value_points_per_mm")
            if not child_scale:
                continue
            parent_orientation, child_orientation, shared_axis = _orientation(str(constraint["mode"]))
            trace = constraint.get("trace", {})
            trace_box = trace.get("bbox_display") or []
            parent_cut = None
            if str(constraint["mode"]).startswith("cut_") and len(trace_box) == 4:
                coordinate_index = 1 if trace.get("orientation") == "horizontal" else 0
                parent_cut = {
                    "orientation": trace.get("orientation"),
                    "coordinate_display": (
                        float(trace_box[coordinate_index]) + float(trace_box[coordinate_index + 2])
                    )
                    / 2.0,
                    "trace_bbox_display": list(map(float, trace_box)),
                    "primitive_refs": list(trace.get("primitive_refs", [])),
                }
            if not parent_scale:
                if parent_cut is None:
                    continue
                # The parent scale may be transferred only by the exact
                # native silhouette gate inside _pair.  Until that passes,
                # the child scale is merely a computational candidate.
                parent_scale = child_scale
            parent_contours = _axis_candidates(
                contours_by_view.get(parent_id, []),
                views_by_id.get(parent_id, {}),
                parent_orientation,
                ownership_by_contour,
            )
            child_contours = _axis_candidates(
                contours_by_view.get(child_id, []),
                views_by_id.get(child_id, {}),
                child_orientation,
                ownership_by_contour,
            )
            candidates = [
                pair
                for parent in parent_contours
                for child in child_contours
                for pair in [
                    _pair(
                        parent,
                        child,
                        parent_orientation,
                        child_orientation,
                        float(parent_scale),
                        float(child_scale),
                        ownership_by_contour,
                        parent_cut,
                    )
                ]
                if pair is not None
            ]
            edge_signatures = []
            silhouette_pass = any(
                item["state"] == "pass"
                and item.get("projection_mode") == "orthogonal_silhouette"
                for item in candidates
            )
            if parent_cut is not None and not silhouette_pass:
                edge_signatures = _cut_edge_signatures(
                    native_segments,
                    views_by_id.get(parent_id, {}).get("bbox_display", (0, 0, 0, 0)),
                    parent_cut,
                    float(parent_scale),
                )
                candidates.extend(
                    pair
                    for signature in edge_signatures
                    for child in child_contours
                    for pair in [
                        _pair(
                            None,
                            child,
                            parent_orientation,
                            child_orientation,
                            float(parent_scale),
                            float(child_scale),
                            ownership_by_contour,
                            parent_cut,
                            signature,
                        )
                    ]
                    if pair is not None
                )
            if parent_cut is not None and not any(item["state"] == "pass" for item in candidates):
                child_edge_signatures = _view_edge_signatures(
                    native_segments,
                    views_by_id.get(child_id, {}).get("bbox_display", (0, 0, 0, 0)),
                    child_orientation,
                    float(child_scale),
                )
                candidates.extend(
                    pair
                    for parent_signature in edge_signatures
                    for child_signature in child_edge_signatures
                    if parent_signature.get("style_signature") == child_signature.get("style_signature")
                    for pair in [
                        _pair(
                            None,
                            None,
                            parent_orientation,
                            child_orientation,
                            float(parent_scale),
                            float(child_scale),
                            ownership_by_contour,
                            parent_cut,
                            parent_signature,
                            child_signature,
                        )
                    ]
                    if pair is not None
                )
            if parent_cut is not None and not any(item["state"] == "pass" for item in candidates):
                candidates.extend(
                    [
                    pair
                    for child in child_contours
                    for pair in [
                        _pair(
                            None,
                            child,
                            parent_orientation,
                            child_orientation,
                            float(parent_scale),
                            float(child_scale),
                            ownership_by_contour,
                            parent_cut,
                        )
                    ]
                    if pair is not None
                    ]
                )
            candidates, canonicalization = _canonicalize_correspondence_candidates(candidates)
            passing = [item for item in candidates if item["state"] == "pass"]
            selected = None
            ambiguity_margin = None
            if passing:
                ambiguity_margin = (
                    passing[1]["objective"] - passing[0]["objective"]
                    if len(passing) > 1
                    else math.inf
                )
                if len(passing) == 1 or ambiguity_margin > max(1.0, passing[0]["tolerance_mm"] / 2.0):
                    selected = passing[0]
            record = {
                "id": f"contour_correspondence.{len(all_records) + 1:04d}",
                "coordinate_scope_id": scope["id"],
                "constraint_id": constraint_id,
                "parent_view_id": parent_id,
                "child_view_id": child_id,
                "shared_object_axis": shared_axis,
                "parent_display_orientation": parent_orientation,
                "child_display_orientation": child_orientation,
                "state": "accepted" if selected is not None else "candidate" if candidates else "unknown",
                "selected": selected,
                "candidate_count": len(candidates),
                "passing_candidate_count": len(passing),
                "representation_candidate_count": canonicalization["representation_candidate_count"],
                "representation_passing_candidate_count": canonicalization[
                    "representation_passing_candidate_count"
                ],
                "canonical_candidate_count": canonicalization["canonical_candidate_count"],
                "canonical_correspondence_count": canonicalization[
                    "canonical_correspondence_count"
                ],
                "canonicalized_group_count": canonicalization["canonicalized_group_count"],
                "collapsed_representation_count": canonicalization["collapsed_representation_count"],
                "ambiguity_margin": None if ambiguity_margin is None or math.isinf(ambiguity_margin) else round(ambiguity_margin, 6),
                "candidates": candidates[:12],
                "reason": (
                    "one native contour pair uniquely passes path-coordinate reprojection"
                    if selected is not None
                    else "multiple native contour pairs pass within tolerance"
                    if passing
                    else "no closed native contour pair passes path-coordinate reprojection"
                ),
            }
            scope_records.append(record)
            all_records.append(record)
            if selected is not None and (selected.get("scale_transfer_certificate") or {}).get("status") == "passed":
                scope.setdefault("reprojection_validations", []).append(
                    {
                        "id": f"{record['id']}.silhouette_reprojection",
                        "status": "pass",
                        "shared_axis": shared_axis,
                        "residual_mm": selected["span_residual_mm"],
                        "tolerance_mm": selected["tolerance_mm"],
                        "evidence_refs": list(selected.get("primitive_refs", [])),
                        "reason": "unique native closed-contour silhouette reprojects at the independently resolved section scale",
                    }
                )
        scope["contour_correspondences"] = scope_records
        scope["actual_contour_reprojection_count"] = sum(item["state"] == "accepted" for item in scope_records)
        scope["signed_contour_transform_count"] = sum(
            (item.get("selected") or {}).get("signed_transform", {}).get("state") == "resolved"
            for item in scope_records
        )
        scope["path_reprojection_eligible"] = (
            scope.get("state") == "resolved_relative"
            and scope["signed_contour_transform_count"] > 0
        )
    summary = {
        "contour_correspondence_count": len(all_records),
        "accepted_contour_correspondence_count": sum(item["state"] == "accepted" for item in all_records),
        "signed_contour_transform_count": sum(
            (item.get("selected") or {}).get("signed_transform", {}).get("state") == "resolved"
            for item in all_records
        ),
        "path_reprojection_eligible_scope_count": sum(
            item.get("path_reprojection_eligible", False)
            for item in shared_coordinates.get("scopes", []) or []
        ),
        "canonicalized_group_count": sum(
            item.get("canonicalized_group_count", 0) for item in all_records
        ),
        "collapsed_representation_count": sum(
            item.get("collapsed_representation_count", 0) for item in all_records
        ),
    }
    shared_coordinates["contour_correspondence"] = {
        "schema_version": "0.1.0",
        "layer": "native_contour_correspondence",
        "status": "resolved_subset" if summary["accepted_contour_correspondence_count"] else "unresolved",
        "records": all_records,
        "summary": summary,
        "validation": {
            "bounding_box_similarity_alone_is_not_correspondence": True,
            "native_path_coordinates_are_compared": True,
            "multiple_passing_pairs_abstain": True,
            "equivalent_cross_representations_are_canonicalized": True,
            "canonicalization_requires_exact_native_edge_and_metric_support": True,
            "canonicalization_never_resolves_mirror": True,
            "mirrored_sign_ambiguity_is_preserved": True,
            "schedule_values_used": False,
            "filename_dispatch_used": False,
            "object_class_template_used": False,
        },
    }
    shared_coordinates.setdefault("summary", {}).update(summary)
    return shared_coordinates["contour_correspondence"]


def reclose_contour_correspondence(
    frames: list[dict[str, Any]],
    shared_coordinates: Mapping[str, Any] | None,
    contours: Iterable[Mapping[str, Any]],
    views: Iterable[Mapping[str, Any]],
    dimension_ownership: Mapping[str, Any] | None = None,
    native_segments: Iterable[Mapping[str, Any]] = (),
    *,
    certified_constraint_ids: Iterable[str] = (),
    translation_errors: Iterable[str] = (),
    input_uniqueness_errors: Iterable[str] = (),
    coordinate_reclosure_status: str = "insufficient_constraints",
) -> dict[str, Any]:
    """Re-run native path correspondence over a certified replay slice."""

    constraint_ids = sorted({str(item) for item in certified_constraint_ids})
    errors = sorted({str(item) for item in translation_errors if str(item)})
    uniqueness_errors = sorted({str(item) for item in input_uniqueness_errors if str(item)})
    coordinate_copy = deepcopy(dict(shared_coordinates or {}))
    result = None
    records: list[Mapping[str, Any]] = []

    if not constraint_ids:
        status = "insufficient_constraints"
        reason = "no certified contour constraints are available"
    elif uniqueness_errors or coordinate_reclosure_status == "reclosed_fail":
        status = "reclosed_fail"
        reason = "certified section identity or coordinate topology fails uniqueness"
    elif errors or coordinate_reclosure_status != "reclosed_pass" or not coordinate_copy:
        status = "insufficient_constraints"
        reason = "certified constraints do not provide complete native coordinate and contour inputs"
    else:
        result = solve_contour_correspondence(
            deepcopy(frames),
            coordinate_copy,
            [deepcopy(dict(item)) for item in contours],
            [deepcopy(dict(item)) for item in views],
            deepcopy(dict(dimension_ownership or {})),
            [deepcopy(dict(item)) for item in native_segments],
        )
        records = result.get("records", []) or []
        ambiguous = [
            item
            for item in records
            if item.get("state") != "accepted" and item.get("passing_candidate_count", 0) > 1
        ]
        failed = [
            item
            for item in records
            if item.get("state") != "accepted"
            and item.get("candidate_count", 0) > 0
            and item.get("passing_candidate_count", 0) == 0
        ]
        missing = [item for item in records if item.get("candidate_count", 0) == 0]
        if ambiguous or failed:
            status = "reclosed_fail"
            reason = "native path reprojection is conflicting or not unique"
        elif not records or missing:
            status = "insufficient_constraints"
            reason = "native contour paths do not support every certified coordinate constraint"
        else:
            status = "reclosed_pass"
            reason = "native path, uniqueness, metric, and reprojection gates reclosed"

    candidate_count = sum(int(item.get("candidate_count", 0)) for item in records)
    accepted_count = sum(item.get("state") == "accepted" for item in records)
    ambiguous_count = sum(
        item.get("state") != "accepted" and item.get("passing_candidate_count", 0) > 1
        for item in records
    )
    failed_candidate_count = sum(
        item.get("state") != "accepted"
        and item.get("candidate_count", 0) > 0
        and item.get("passing_candidate_count", 0) == 0
        for item in records
    )
    return {
        "schema_version": "0.1.0",
        "stage": "contour_correspondence",
        "status": status,
        "reason": reason,
        "certified_constraint_ids": constraint_ids,
        "gates": {
            "coordinate": {"status": coordinate_reclosure_status},
            "native_path": {
                "status": "pass" if records and candidate_count else "insufficient",
                "record_count": len(records),
                "candidate_count": candidate_count,
            },
            "uniqueness": {
                "status": "fail" if uniqueness_errors or ambiguous_count else "pass" if records else "insufficient",
                "errors": uniqueness_errors,
                "ambiguous_record_count": ambiguous_count,
            },
            "reprojection": {
                "status": (
                    "fail"
                    if failed_candidate_count
                    else "pass"
                    if records and accepted_count == len(records)
                    else "insufficient"
                ),
                "accepted_count": accepted_count,
                "failed_candidate_count": failed_candidate_count,
            },
        },
        "translation": {
            "status": "pass" if not errors else "insufficient",
            "errors": errors,
        },
        "contour_correspondence": result,
        "contract": {
            "certified_constraints_only": True,
            "canonical_ids_resolved_against_frozen_native_pages": True,
            "native_path_coordinates_compared": True,
            "multiple_passing_pairs_abstain": True,
            "mirrored_sign_ambiguity_preserved": True,
            "quantities_read": False,
            "quantities_written": False,
        },
    }
