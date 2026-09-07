"""M3 provenance-preserving projected MEP route observations.

This layer records page-local route geometry and topology only.  Native PDF
vectors remain primary; quality-gated raster lines can enter the same record
shape with their confidence, transform, crop, and model provenance intact.
Endpoint connectivity is limited to shared display-space vertices.  A
geometric intersection away from endpoints is an unconnected crossing, while
three or more fragments sharing one endpoint vertex form a branch
observation.  Gaps and repeated shapes remain candidates, never authority for
system identity, continuation, fabrication, installation, or quantity.
"""

from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import fitz

from src.drawing_engine.core.vector_topology import extract_page_topology


SCHEMA_VERSION = "0.1.0"
LAYER = "mep_projected_route_observations"


def _stable_id(kind: str, *parts: object) -> str:
    encoded = json.dumps(parts, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return f"{kind}.{hashlib.sha256(encoded).hexdigest()[:20]}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _round_point(point: Iterable[object], digits: int = 6) -> list[float]:
    x, y = point
    return [round(float(x), digits), round(float(y), digits)]


def _bbox(points: Iterable[Iterable[object]]) -> list[float]:
    rows = [_round_point(point) for point in points]
    return [
        round(min(point[0] for point in rows), 6),
        round(min(point[1] for point in rows), 6),
        round(max(point[0] for point in rows), 6),
        round(max(point[1] for point in rows), 6),
    ]


def _path_length(points: list[list[float]]) -> float:
    return sum(math.dist(left, right) for left, right in zip(points, points[1:]))


def _normalise_style(style: Mapping[str, Any] | None) -> dict[str, Any]:
    source = dict(style or {})
    stroke = source.get("stroke")
    fill = source.get("fill")
    width = source.get("width", source.get("width_points"))
    return {
        "stroke": None if stroke is None else [round(float(value), 6) for value in stroke],
        "fill": None if fill is None else [round(float(value), 6) for value in fill],
        "width_display_points": None if width is None else round(float(width), 6),
        "dash_pattern": source.get("dash", source.get("dash_pattern")),
        "color_only_identity_eligible": False,
    }


def _scale_from_page_scope(page_scope: Mapping[str, Any]) -> float | None:
    value = (
        page_scope.get("fields", {})
        .get("scale", {})
        .get("drawing_inches_per_paper_inch")
    )
    if value is None:
        return None
    try:
        scale = float(value)
    except (TypeError, ValueError):
        return None
    return scale if math.isfinite(scale) and scale > 0 else None


def _metric_observation(length_points: float, page_scope: Mapping[str, Any]) -> dict[str, Any]:
    scale = _scale_from_page_scope(page_scope)
    if scale is None:
        return {
            "state": "unknown",
            "measurement": "local_2d_projected_path",
            "projected_path_m": None,
            "drawing_inches_per_paper_inch": None,
            "basis": "M1 sheet scale is unresolved",
            "dimensionality": "two_dimensional_projection",
        }
    projected_metres = length_points / 72.0 * scale * 0.0254
    return {
        "state": "derived",
        "measurement": "local_2d_projected_path",
        "projected_path_m": round(projected_metres, 8),
        "drawing_inches_per_paper_inch": round(scale, 8),
        "basis": "display points converted through evidence-backed M1 sheet scale",
        "dimensionality": "two_dimensional_projection",
    }


def _page_ownership(
    page_scope: Mapping[str, Any], package_ref: str | None
) -> dict[str, Any]:
    sheet_field = page_scope.get("fields", {}).get("sheet_number", {})
    return {
        "page_ref": str(page_scope["page_ref"]),
        "page_number": int(page_scope["page_number"]),
        "view_scope_ref": str(page_scope.get("id") or page_scope["page_ref"]),
        "view_role": str(page_scope.get("role") or "unknown"),
        "sheet_number": sheet_field.get("value"),
        "sheet_number_evidence_refs": sorted(
            str(ref) for ref in sheet_field.get("evidence_refs", [])
        ),
        "package_ref": package_ref,
    }


def _native_candidates(
    page_scope: Mapping[str, Any], topology: Mapping[str, Any]
) -> list[dict[str, Any]]:
    page_ref = str(page_scope["page_ref"])
    output = []
    for segment in topology.get("segments", []):
        points = segment.get("sample_points_display") or [
            segment["start_display"],
            segment["end_display"],
        ]
        points = [_round_point(point) for point in points]
        source_ref = str(segment["id"])
        output.append(
            {
                "source_kind": "native_pdf_vector",
                "source_primitive_ref": source_ref,
                "geometry_kind": str(segment.get("kind") or "line"),
                "points_display": points,
                "control_points_display": [
                    _round_point(point)
                    for point in segment.get("control_points_display", [])
                ],
                "style": _normalise_style(segment.get("style")),
                "confidence": 1.0,
                "provenance": {
                    "method": "native_pdf_vector_topology",
                    "source_primitive_ref": source_ref,
                    "drawing_ref": segment.get("drawing_ref"),
                    "primitive_ref": segment.get("primitive_ref"),
                    "item_index": segment.get("item_index"),
                    "part_index": segment.get("part_index"),
                    "coordinate_space": "rotation-normalized PyMuPDF page display coordinates",
                    "coordinate_transform": None,
                    "source_crop_display": None,
                    "model_provenance": None,
                    **({'native_metadata': copy.deepcopy(segment['native_metadata'])}
                       if segment.get('native_metadata') else {}),
                },
                "candidate_id": _stable_id(
                    "mep_route_primitive",
                    page_ref,
                    "native_pdf_vector",
                    source_ref,
                    points,
                ),
            }
        )
    return output


def _raster_candidates(
    page_scope: Mapping[str, Any], fallback: Mapping[str, Any]
) -> list[dict[str, Any]]:
    page_ref = str(page_scope["page_ref"])
    output = []
    shared_transform = fallback.get("coordinate_transform")
    shared_model = fallback.get("model_provenance")
    for observation in fallback.get("observations", []) or []:
        geometry = observation.get("geometry_display") or {}
        geometry_kind = str(geometry.get("kind") or "")
        if observation.get("kind") != "line_segment" and geometry_kind not in {
            "line",
            "cubic",
        }:
            continue
        points = geometry.get("points_display") or observation.get("points_display") or []
        if len(points) < 2:
            continue
        points = [_round_point(point) for point in points]
        source_ref = str(observation["id"])
        output.append(
            {
                "source_kind": "raster_vector_candidate",
                "source_primitive_ref": source_ref,
                "geometry_kind": geometry_kind or "line",
                "points_display": points,
                "control_points_display": [
                    _round_point(point)
                    for point in geometry.get("control_points_display", [])
                ],
                "style": _normalise_style(observation.get("style")),
                "confidence": round(float(observation.get("confidence", 0.0)), 6),
                "provenance": {
                    "method": str(observation.get("method") or "unknown_raster_method"),
                    "source_primitive_ref": source_ref,
                    "drawing_ref": None,
                    "primitive_ref": source_ref,
                    "item_index": None,
                    "part_index": None,
                    "coordinate_space": "rotation-normalized PyMuPDF page display coordinates",
                    "coordinate_transform": observation.get(
                        "coordinate_transform", shared_transform
                    ),
                    "source_crop_display": observation.get("source_crop_display"),
                    "model_provenance": observation.get(
                        "model_provenance", shared_model
                    ),
                },
                "candidate_id": _stable_id(
                    "mep_route_primitive",
                    page_ref,
                    "raster_vector_candidate",
                    source_ref,
                    points,
                    observation.get("method"),
                ),
            }
        )
    return output


def _cluster_endpoints(
    page_ref: str,
    candidates: list[dict[str, Any]],
    tolerance: float,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], str]]:
    endpoints = []
    for candidate in candidates:
        for role, point in (
            ("start", candidate["points_display"][0]),
            ("end", candidate["points_display"][-1]),
        ):
            endpoints.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "role": role,
                    "point": point,
                }
            )
    endpoints.sort(
        key=lambda item: (
            item["point"][0],
            item["point"][1],
            item["candidate_id"],
            item["role"],
        )
    )
    parent = list(range(len(endpoints)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    cell_size = max(tolerance, 1e-9)
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, item in enumerate(endpoints):
        x, y = item["point"]
        cell = (math.floor(x / cell_size), math.floor(y / cell_size))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in cells.get((cell[0] + dx, cell[1] + dy), []):
                    if math.dist(item["point"], endpoints[other]["point"]) <= tolerance:
                        union(index, other)
        cells[cell].append(index)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(endpoints)):
        groups[find(index)].append(index)
    vertices = []
    assignments = {}
    for rows in groups.values():
        x = sum(endpoints[index]["point"][0] for index in rows) / len(rows)
        y = sum(endpoints[index]["point"][1] for index in rows) / len(rows)
        point = [round(x, 6), round(y, 6)]
        vertex_id = _stable_id("mep_route_vertex", page_ref, point)
        member_keys = []
        for index in rows:
            item = endpoints[index]
            key = (item["candidate_id"], item["role"])
            assignments[key] = vertex_id
            member_keys.append(key)
        vertices.append(
            {
                "record_type": "mep_route_vertex_observation",
                "record_version": SCHEMA_VERSION,
                "id": vertex_id,
                "page_ref": page_ref,
                "point_display": point,
                "endpoint_refs": [],
                "fragment_refs": sorted({key[0] for key in member_keys}),
                "degree": len({key[0] for key in member_keys}),
                "state": "observed",
                "quantity_eligible": False,
            }
        )
    vertices.sort(key=lambda item: (item["point_display"], item["id"]))
    return vertices, assignments


def _fragment_records(
    page_scope: Mapping[str, Any],
    package_ref: str | None,
    candidates: list[dict[str, Any]],
    assignments: Mapping[tuple[str, str], str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ownership = _page_ownership(page_scope, package_ref)
    fragments = []
    endpoints = []
    for candidate in sorted(candidates, key=lambda item: item["candidate_id"]):
        fragment_id = candidate["candidate_id"]
        start_vertex = assignments[(fragment_id, "start")]
        end_vertex = assignments[(fragment_id, "end")]
        length_points = _path_length(candidate["points_display"])
        fragment = {
            "record_type": "mep_route_fragment_observation",
            "record_version": SCHEMA_VERSION,
            "id": fragment_id,
            "page_ref": str(page_scope["page_ref"]),
            "page_number": int(page_scope["page_number"]),
            "source_kind": candidate["source_kind"],
            "source_primitive_ref": candidate["source_primitive_ref"],
            "geometry": {
                "kind": candidate["geometry_kind"],
                "points_display": candidate["points_display"],
                "control_points_display": candidate["control_points_display"],
                "bbox_display": _bbox(candidate["points_display"]),
                "path_display_points": round(length_points, 6),
            },
            "endpoint_vertex_refs": [start_vertex, end_vertex],
            "style": candidate["style"],
            "confidence": candidate["confidence"],
            "provenance": candidate["provenance"],
            "ownership": dict(ownership),
            "local_metric_observation": _metric_observation(length_points, page_scope),
            "route_family_ref": None,
            "state": "observed",
            "quantity_eligible": False,
        }
        fragments.append(fragment)
        for role, point, vertex_ref in (
            ("start", candidate["points_display"][0], start_vertex),
            ("end", candidate["points_display"][-1], end_vertex),
        ):
            endpoints.append(
                {
                    "record_type": "mep_route_endpoint_observation",
                    "record_version": SCHEMA_VERSION,
                    "id": _stable_id("mep_route_endpoint", fragment_id, role),
                    "page_ref": str(page_scope["page_ref"]),
                    "fragment_ref": fragment_id,
                    "source_primitive_ref": candidate["source_primitive_ref"],
                    "role": role,
                    "point_display": point,
                    "vertex_ref": vertex_ref,
                    "state": "observed",
                    "quantity_eligible": False,
                }
            )
    return fragments, endpoints


def _branch_records(
    page_ref: str,
    vertices: list[dict[str, Any]],
    fragments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {fragment["id"]: fragment for fragment in fragments}
    output = []
    for vertex in vertices:
        directions = []
        for fragment_ref in vertex["fragment_refs"]:
            fragment = by_id[fragment_ref]
            if fragment["endpoint_vertex_refs"][0] == vertex["id"]:
                point, interior = (
                    fragment["geometry"]["points_display"][0],
                    fragment["geometry"]["points_display"][1],
                )
            else:
                point, interior = (
                    fragment["geometry"]["points_display"][-1],
                    fragment["geometry"]["points_display"][-2],
                )
            angle = math.atan2(interior[1] - point[1], interior[0] - point[0]) % (2.0 * math.pi)
            if not any(
                min(abs(angle - kept), 2.0 * math.pi - abs(angle - kept)) <= math.radians(3.0)
                for kept in directions
            ):
                directions.append(angle)
        if len(directions) < 3:
            continue
        output.append({
            "record_type": "mep_route_branch_observation",
            "record_version": SCHEMA_VERSION,
            "id": _stable_id("mep_route_branch", page_ref, vertex["id"]),
            "page_ref": page_ref,
            "vertex_ref": vertex["id"],
            "point_display": vertex["point_display"],
            "fragment_refs": vertex["fragment_refs"],
            "degree": vertex["degree"],
            "distinct_incident_direction_count": len(directions),
            "basis": "three_or_more_fragments_share_one_page_global_endpoint_vertex",
            "state": "observed",
            "quantity_eligible": False,
        })
    return output


def _segment_intersection(
    left_start: list[float],
    left_end: list[float],
    right_start: list[float],
    right_end: list[float],
    *,
    epsilon: float = 1e-8,
) -> tuple[list[float], float, float] | None:
    ax, ay = left_start
    bx, by = left_end
    cx, cy = right_start
    dx, dy = right_end
    rx, ry = bx - ax, by - ay
    sx, sy = dx - cx, dy - cy
    denominator = rx * sy - ry * sx
    if abs(denominator) <= epsilon:
        return None
    qpx, qpy = cx - ax, cy - ay
    left_ratio = (qpx * sy - qpy * sx) / denominator
    right_ratio = (qpx * ry - qpy * rx) / denominator
    if epsilon < left_ratio < 1.0 - epsilon and epsilon < right_ratio < 1.0 - epsilon:
        return (
            [round(ax + left_ratio * rx, 6), round(ay + left_ratio * ry, 6)],
            left_ratio,
            right_ratio,
        )
    return None


def _crossing_records(
    page_ref: str, fragments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    records = {}
    cell_size = 64.0
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, fragment in enumerate(fragments):
        x0, y0, x1, y1 = fragment["geometry"]["bbox_display"]
        for cell_x in range(math.floor(x0 / cell_size), math.floor(x1 / cell_size) + 1):
            for cell_y in range(math.floor(y0 / cell_size), math.floor(y1 / cell_size) + 1):
                cells[(cell_x, cell_y)].append(index)
    pairs = {
        (min(left, right), max(left, right))
        for rows in cells.values()
        for position, left in enumerate(rows)
        for right in rows[position + 1 :]
        if left != right
    }
    for left_index, right_index in sorted(pairs):
        left, right = fragments[left_index], fragments[right_index]
        if set(left["endpoint_vertex_refs"]) & set(right["endpoint_vertex_refs"]):
            continue
        left_points = left["geometry"]["points_display"]
        right_points = right["geometry"]["points_display"]
        for left_a, left_b in zip(left_points, left_points[1:]):
            for right_a, right_b in zip(right_points, right_points[1:]):
                intersection = _segment_intersection(left_a, left_b, right_a, right_b)
                if intersection is None:
                    continue
                point, _, _ = intersection
                refs = sorted([left["id"], right["id"]])
                key = (tuple(refs), tuple(point))
                records[key] = {
                    "record_type": "mep_route_crossing_observation",
                    "record_version": SCHEMA_VERSION,
                    "id": _stable_id("mep_route_crossing", page_ref, refs, point),
                    "page_ref": page_ref,
                    "point_display": point,
                    "fragment_refs": refs,
                    "connectivity": "unconnected",
                    "basis": "interior geometric intersection has no shared endpoint vertex",
                    "state": "observed",
                    "quantity_eligible": False,
                }
    return sorted(records.values(), key=lambda item: (item["point_display"], item["id"]))


def _unit_from_endpoint(fragment: Mapping[str, Any], role: str) -> tuple[float, float]:
    points = fragment["geometry"]["points_display"]
    endpoint, interior = (points[0], points[1]) if role == "start" else (points[-1], points[-2])
    dx, dy = interior[0] - endpoint[0], interior[1] - endpoint[1]
    magnitude = math.hypot(dx, dy)
    return (0.0, 0.0) if magnitude <= 1e-12 else (dx / magnitude, dy / magnitude)


def _style_comparison(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    non_color_fields = ("width_display_points", "dash_pattern")
    known = [
        field
        for field in non_color_fields
        if left.get(field) is not None and right.get(field) is not None
    ]
    return {
        "non_color_fields_compared": known,
        "non_color_match": bool(known)
        and all(left.get(field) == right.get(field) for field in known),
        "stroke_color_match": left.get("stroke") is not None
        and left.get("stroke") == right.get("stroke"),
        "color_only_identity_eligible": False,
    }


def _gap_records(
    page_ref: str,
    fragments: list[dict[str, Any]],
    endpoints: list[dict[str, Any]],
    *,
    gap_tolerance: float,
    minimum_alignment_cosine: float = 0.965,
) -> list[dict[str, Any]]:
    by_fragment = {fragment["id"]: fragment for fragment in fragments}
    output = []
    cell_size = max(gap_tolerance, 1e-9)
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, endpoint in enumerate(endpoints):
        x, y = endpoint["point_display"]
        cells[(math.floor(x / cell_size), math.floor(y / cell_size))].append(index)
    pairs = set()
    for left_index, left in enumerate(endpoints):
        x, y = left["point_display"]
        cell = (math.floor(x / cell_size), math.floor(y / cell_size))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for right_index in cells.get((cell[0] + dx, cell[1] + dy), []):
                    if right_index > left_index:
                        pairs.add((left_index, right_index))
    for left_index, right_index in sorted(pairs):
        left, right = endpoints[left_index], endpoints[right_index]
        if left["fragment_ref"] == right["fragment_ref"] or left["vertex_ref"] == right["vertex_ref"]:
            continue
        distance = math.dist(left["point_display"], right["point_display"])
        if distance <= 1e-9 or distance > gap_tolerance:
            continue
        left_fragment = by_fragment[left["fragment_ref"]]
        right_fragment = by_fragment[right["fragment_ref"]]
        left_unit = _unit_from_endpoint(left_fragment, left["role"])
        right_unit = _unit_from_endpoint(right_fragment, right["role"])
        alignment = abs(left_unit[0] * right_unit[0] + left_unit[1] * right_unit[1])
        if alignment < minimum_alignment_cosine:
            continue
        gap_unit = (
            (right["point_display"][0] - left["point_display"][0]) / distance,
            (right["point_display"][1] - left["point_display"][1]) / distance,
        )
        gap_alignment = min(
            abs(left_unit[0] * gap_unit[0] + left_unit[1] * gap_unit[1]),
            abs(right_unit[0] * gap_unit[0] + right_unit[1] * gap_unit[1]),
        )
        if gap_alignment < minimum_alignment_cosine:
            continue
        refs = sorted([left["id"], right["id"]])
        output.append(
            {
                "record_type": "mep_route_gap_observation",
                "record_version": SCHEMA_VERSION,
                "id": _stable_id("mep_route_gap", page_ref, refs),
                "page_ref": page_ref,
                "endpoint_refs": refs,
                "fragment_refs": sorted([left["fragment_ref"], right["fragment_ref"]]),
                "distance_display_points": round(distance, 6),
                "alignment_cosine": round(alignment, 6),
                "gap_axis_alignment_cosine": round(gap_alignment, 6),
                "style_comparison": _style_comparison(
                    left_fragment["style"], right_fragment["style"]
                ),
                "state": "candidate",
                "interpretation": "unresolved_page_local_gap",
                "quantity_eligible": False,
            }
        )
    return sorted(output, key=lambda item: (item["distance_display_points"], item["id"]))


def _shape_signature(fragment: Mapping[str, Any]) -> tuple[Any, ...]:
    points = fragment["geometry"]["points_display"]
    origin = points[0]
    forward = tuple(
        (round(point[0] - origin[0], 3), round(point[1] - origin[1], 3))
        for point in points
    )
    reverse_origin = points[-1]
    reverse = tuple(
        (round(point[0] - reverse_origin[0], 3), round(point[1] - reverse_origin[1], 3))
        for point in reversed(points)
    )
    geometry = min(forward, reverse)
    return (fragment["geometry"]["kind"], geometry)


def _repetition_records(
    page_ref: str, fragments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for fragment in fragments:
        groups[_shape_signature(fragment)].append(fragment)
    output = []
    for signature, rows in sorted(groups.items(), key=lambda item: repr(item[0])):
        if len(rows) < 2:
            continue
        refs = sorted(row["id"] for row in rows)
        non_color_styles = {
            (row["style"]["width_display_points"], str(row["style"]["dash_pattern"]))
            for row in rows
        }
        colors = {str(row["style"]["stroke"]) for row in rows}
        output.append(
            {
                "record_type": "mep_route_repetition_observation",
                "record_version": SCHEMA_VERSION,
                "id": _stable_id("mep_route_repetition", page_ref, refs),
                "page_ref": page_ref,
                "fragment_refs": refs,
                "occurrence_count": len(refs),
                "basis": "translation-invariant equal projected geometry",
                "non_color_style_consensus": len(non_color_styles) == 1,
                "stroke_color_consensus": len(colors) == 1,
                "color_only_identity_eligible": False,
                "state": "observed",
                "quantity_eligible": False,
            }
        )
    return output


def build_mep_route_page(
    *,
    page_scope: Mapping[str, Any],
    native_topology: Mapping[str, Any] | None = None,
    raster_fallback: Mapping[str, Any] | None = None,
    package_ref: str | None = None,
    vertex_tolerance_display_points: float = 0.75,
    gap_tolerance_display_points: float = 6.0,
) -> dict[str, Any]:
    """Build one M3 page from an M1 page scope and geometry observations."""

    if page_scope.get("record_type") != "mep_sheet_page_record":
        raise ValueError("page_scope must be an M1 mep_sheet_page_record")
    if (
        not math.isfinite(float(vertex_tolerance_display_points))
        or float(vertex_tolerance_display_points) <= 0
        or not math.isfinite(float(gap_tolerance_display_points))
        or float(gap_tolerance_display_points) <= 0
    ):
        raise ValueError("route topology tolerances must be finite and positive")
    page_ref = str(page_scope["page_ref"])
    candidates = [
        *_native_candidates(page_scope, native_topology or {}),
        *_raster_candidates(page_scope, raster_fallback or {}),
    ]
    ids = [item["candidate_id"] for item in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{page_ref}: duplicate route primitive IDs")
    vertices, assignments = _cluster_endpoints(
        page_ref, candidates, vertex_tolerance_display_points
    )
    fragments, endpoints = _fragment_records(
        page_scope, package_ref, candidates, assignments
    )
    endpoint_by_id = {item["id"]: item for item in endpoints}
    for vertex in vertices:
        vertex["endpoint_refs"] = sorted(
            endpoint["id"]
            for endpoint in endpoints
            if endpoint["vertex_ref"] == vertex["id"]
        )
    branches = _branch_records(page_ref, vertices, fragments)
    crossings = _crossing_records(page_ref, fragments)
    gaps = _gap_records(
        page_ref,
        fragments,
        endpoints,
        gap_tolerance=gap_tolerance_display_points,
    )
    repetitions = _repetition_records(page_ref, fragments)
    return {
        "record_type": "mep_route_page_observations",
        "record_version": SCHEMA_VERSION,
        "id": _stable_id("mep_route_page", page_ref),
        "page_ref": page_ref,
        "page_number": int(page_scope["page_number"]),
        "scope": _page_ownership(page_scope, package_ref),
        "fragments": fragments,
        "vertices": vertices,
        "endpoints": sorted(endpoints, key=lambda item: item["id"]),
        "gaps": gaps,
        "branches": branches,
        "crossings": crossings,
        "repetitions": repetitions,
        **({'native_metadata_inventory': copy.deepcopy(native_topology['native_metadata_inventory'])}
           if native_topology and native_topology.get('native_metadata_inventory') else {}),
        "tolerances": {
            "vertex_display_points": float(vertex_tolerance_display_points),
            "gap_display_points": float(gap_tolerance_display_points),
        },
        "summary": {
            "fragment_count": len(fragments),
            "native_fragment_count": sum(
                item["source_kind"] == "native_pdf_vector" for item in fragments
            ),
            "raster_fragment_count": sum(
                item["source_kind"] == "raster_vector_candidate" for item in fragments
            ),
            "endpoint_count": len(endpoint_by_id),
            "vertex_count": len(vertices),
            "gap_candidate_count": len(gaps),
            "branch_observation_count": len(branches),
            "unconnected_crossing_count": len(crossings),
            "repetition_observation_count": len(repetitions),
        },
        "authority_boundary": {
            "page_local_projected_observations_only": True,
            "shared_endpoint_is_the_only_connectivity_basis": True,
            "geometric_crossings_are_unconnected": True,
            "color_alone_never_establishes_identity": True,
            "engineering_outputs_enabled": False,
        },
        "quantity_eligible": False,
    }


def build_mep_route_page_from_denominator(
    *, page_scope: Mapping[str, Any], descriptor_pack: Any,
    disposition_pack: Any, package_ref: str | None = None,
    vertex_tolerance_display_points: float = .05,
    gap_tolerance_display_points: float = .05,
) -> dict[str, Any]:
    """Build M3 route rows from an exhaustive compact source denominator.

    Only the singular ``route_evidence`` disposition is expanded into the
    route topology.  All other source primitives remain present through the
    immutable descriptor/disposition pack reference and exhaustive counts.
    This prevents an envelope or leader proposal from defining M3's source
    existence denominator.
    """
    from src.drawing_engine.disciplines.mep.mep_source_primitive_denominator import validate_source_denominator

    errors = validate_source_denominator(
        descriptor_pack=descriptor_pack, disposition_pack=disposition_pack)
    if errors:
        raise ValueError("source denominator is not complete: " + "; ".join(errors))
    if descriptor_pack.manifest.get("page_ref") != str(page_scope["page_ref"]):
        raise ValueError("source denominator belongs to a different M1 page")
    segments = []
    dispositions = disposition_pack.records()
    with descriptor_pack.payload_path.open("rb") as payload_stream:
        for descriptor, disposition in zip(descriptor_pack.descriptors(), dispositions, strict=True):
            if descriptor["descriptor_ordinal"] != disposition["descriptor_ordinal"]:
                raise ValueError("descriptor/disposition ordinal mismatch")
            if disposition["primary_disposition"] != "route_evidence":
                continue
            row = descriptor_pack.row(descriptor, payload_stream)
            segments.append(row["source_native_segment"])
    expected = disposition_pack.manifest["disposition_counts"].get("route_evidence", 0)
    if len(segments) != expected:
        raise ValueError("expanded M3 route count differs from denominator disposition")
    page = build_mep_route_page(
        page_scope=page_scope, native_topology={"segments": segments},
        package_ref=package_ref,
        vertex_tolerance_display_points=vertex_tolerance_display_points,
        gap_tolerance_display_points=gap_tolerance_display_points)
    page["source_denominator"] = {
        "layer": "mep_source_primitive_denominator",
        "native_descriptor_sha256": descriptor_pack.manifest["descriptor_sha256"],
        "native_payload_sha256": descriptor_pack.manifest["payload_sha256"],
        "source_disposition_sha256": disposition_pack.manifest["sha256"],
        "native_segment_count": len(descriptor_pack),
        "primary_disposition_counts": copy.deepcopy(
            disposition_pack.manifest["disposition_counts"]),
        "unaccounted_segment_count": disposition_pack.manifest["unaccounted_segment_count"],
        "expanded_route_evidence_count": len(segments),
        "source_existence_selected_by_envelope_or_leader": False,
        "role_classification_provisional": True,
    }
    page["authority_boundary"].update({
        "complete_native_source_denominator_referenced": True,
        "non_route_source_geometry_preserved_in_compact_denominator": True,
        "route_evidence_is_not_semantic_route_certification": True,
    })
    return page


def _package_index(sheet_registry: Mapping[str, Any]) -> dict[str, str]:
    output = {}
    for package in sheet_registry.get("packages", []):
        package_ref = str(package["id"])
        for page_ref in [
            package.get("divider_page_ref"),
            *package.get("member_page_refs", []),
        ]:
            if page_ref is not None:
                output[str(page_ref)] = package_ref
    return output


def build_mep_route_graph(
    *,
    sheet_registry: Mapping[str, Any],
    page_inputs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compose M3 pages while reusing M1 page and package ownership."""

    if sheet_registry.get("layer") != "mep_sheet_coordinate_registry":
        raise ValueError("sheet_registry must be the M1 MEP sheet coordinate registry")
    scopes = {str(page["page_ref"]): page for page in sheet_registry.get("pages", [])}
    unknown = sorted(set(page_inputs) - set(scopes))
    if unknown:
        raise ValueError(f"page inputs reference unknown M1 scopes: {unknown}")
    packages = _package_index(sheet_registry)
    pages = []
    for page_ref, scope in sorted(
        scopes.items(), key=lambda item: int(item[1]["page_number"])
    ):
        page_input = page_inputs.get(page_ref, {})
        pages.append(
            build_mep_route_page(
                page_scope=scope,
                native_topology=page_input.get("native_topology"),
                raster_fallback=page_input.get("raster_fallback"),
                package_ref=packages.get(page_ref),
                vertex_tolerance_display_points=float(
                    page_input.get("vertex_tolerance_display_points", 0.75)
                ),
                gap_tolerance_display_points=float(
                    page_input.get("gap_tolerance_display_points", 6.0)
                ),
            )
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "layer": LAYER,
        "document": dict(sheet_registry.get("document", {})),
        "sheet_registry_ref": {
            "schema_version": sheet_registry.get("schema_version"),
            "layer": sheet_registry.get("layer"),
            "document_key": sheet_registry.get("document", {}).get("document_key"),
        },
        "pages": pages,
        "summary": {
            "page_count": len(pages),
            "fragment_count": sum(page["summary"]["fragment_count"] for page in pages),
            "native_fragment_count": sum(
                page["summary"]["native_fragment_count"] for page in pages
            ),
            "raster_fragment_count": sum(
                page["summary"]["raster_fragment_count"] for page in pages
            ),
            "gap_candidate_count": sum(
                page["summary"]["gap_candidate_count"] for page in pages
            ),
            "branch_observation_count": sum(
                page["summary"]["branch_observation_count"] for page in pages
            ),
            "unconnected_crossing_count": sum(
                page["summary"]["unconnected_crossing_count"] for page in pages
            ),
            "repetition_observation_count": sum(
                page["summary"]["repetition_observation_count"] for page in pages
            ),
        },
        "exchange_contract": {
            "m1_page_and_package_scopes_reused": True,
            "native_and_raster_records_have_schema_parity": True,
            "source_primitive_provenance_preserved": True,
            "local_projected_metrics_only": True,
            "engineering_outputs_enabled": False,
            "schedule_values_used": False,
        },
        "quantity_eligible": False,
    }
    errors = validate_mep_route_graph(payload)
    if errors:
        raise ValueError("\n".join(errors))
    return payload


def extract_mep_route_graph(
    pdf_path: Path | str,
    sheet_registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract native M3 inputs and invoke raster fallback only when M1 routes it."""

    source = Path(pdf_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    registry_document = sheet_registry.get("document", {})
    expected_bytes = registry_document.get("source_bytes")
    if expected_bytes is not None and int(expected_bytes) != source.stat().st_size:
        raise ValueError("PDF byte count does not match M1 sheet registry provenance")
    expected_sha256 = registry_document.get("source_pdf_sha256")
    if expected_sha256 is not None and str(expected_sha256) != _file_sha256(source):
        raise ValueError("PDF sha256 does not match M1 sheet registry provenance")
    scopes_by_number = {
        int(page["page_number"]): page for page in sheet_registry.get("pages", [])
    }
    page_inputs = {}
    with fitz.open(source) as document:
        if len(document) != len(scopes_by_number):
            raise ValueError("PDF page count does not match M1 sheet registry")
        for page in document:
            page_number = int(page.number) + 1
            scope = scopes_by_number.get(page_number)
            if scope is None:
                raise ValueError(f"M1 sheet registry lacks page {page_number}")
            page_input: dict[str, Any] = {
                "native_topology": extract_page_topology(page),
            }
            base_route = (
                scope.get("quality_route", {})
                .get("base_structural_quality_route", {})
                .get("route")
            )
            if base_route == "raster_reconstruct":
                from src.drawing_engine.core.raster_fallback import reconstruct_raster_observations

                page_input["raster_fallback"] = reconstruct_raster_observations(
                    page,
                    scope.get("quality_route", {}).get(
                        "base_structural_quality_route"
                    ),
                )
            page_inputs[str(scope["page_ref"])] = page_input
    return build_mep_route_graph(
        sheet_registry=sheet_registry,
        page_inputs=page_inputs,
    )


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _walk_items(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            yield path, name, child
            yield from _walk_items(child, (*path, name))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_items(child, (*path, f"[{index}]"))


def validate_mep_route_graph(payload: Mapping[str, Any]) -> list[str]:
    """Validate provenance, topology references, and the M3 authority boundary."""

    errors = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if payload.get("layer") != LAYER:
        errors.append("layer mismatch")
    forbidden_keys = {
        "accepted_route_ref",
        "route_identity",
        "physical_continuation",
        "installed_length",
        "physical_length",
        "fitting_count",
        "quantity",
    }
    allowed_false_authority_keys = {
        "physical_continuation_established",
        "route_identity_established",
        "installed_length_emitted",
        "engineering_outputs_enabled",
    }
    forbidden_prefixes = (
        "accepted_route_",
        "installed_length_",
        "physical_length_",
        "fitting_count_",
        "route_identity_",
        "physical_continuation_",
    )
    present_forbidden = sorted(
        {
            key
            for key in _walk_keys(payload)
            if key in forbidden_keys
            or (
                key.startswith(forbidden_prefixes)
                and key not in allowed_false_authority_keys
            )
        }
    )
    if present_forbidden:
        errors.append(f"forbidden engineering output keys: {present_forbidden}")
    for path, key, value in _walk_items(payload):
        location = ".".join((*path, key))
        if key == "quantity_eligible" and value is not False:
            errors.append(f"{location}: quantity_eligible must be false")
        if key in allowed_false_authority_keys and value is not False:
            errors.append(f"{location}: authority boolean must be false")
        if key == "route_family_ref" and value is not None:
            errors.append(f"{location}: route family identity was assigned")
        if key == "color_only_identity_eligible" and value is not False:
            errors.append(f"{location}: color-only identity became eligible")
    expected_exchange = {
        "m1_page_and_package_scopes_reused": True,
        "native_and_raster_records_have_schema_parity": True,
        "source_primitive_provenance_preserved": True,
        "local_projected_metrics_only": True,
        "engineering_outputs_enabled": False,
        "schedule_values_used": False,
    }
    exchange = payload.get("exchange_contract", {})
    for key, expected in expected_exchange.items():
        if exchange.get(key) is not expected:
            errors.append(f"exchange_contract.{key}: expected {expected}")
    pages = list(payload.get("pages", []))
    page_refs = [str(page.get("page_ref")) for page in pages]
    if len(page_refs) != len(set(page_refs)):
        errors.append("page_ref values are not unique")
    document_page_count = payload.get("document", {}).get("page_count")
    if document_page_count is not None and int(document_page_count) != len(pages):
        errors.append("document page_count does not match M3 pages")
    all_fragment_ids = set()
    all_vertex_ids = set()
    for page in pages:
        page_ref = str(page.get("page_ref"))
        if page.get("record_type") != "mep_route_page_observations":
            errors.append(f"{page_ref}: wrong page record_type")
        if page.get("scope", {}).get("page_ref") != page_ref:
            errors.append(f"{page_ref}: M1 page scope mismatch")
        expected_page_boundary = {
            "page_local_projected_observations_only": True,
            "shared_endpoint_is_the_only_connectivity_basis": True,
            "geometric_crossings_are_unconnected": True,
            "color_alone_never_establishes_identity": True,
            "engineering_outputs_enabled": False,
        }
        boundary = page.get("authority_boundary", {})
        for key, expected in expected_page_boundary.items():
            if boundary.get(key) is not expected:
                errors.append(f"{page_ref}.authority_boundary.{key}: expected {expected}")
        fragments = list(page.get("fragments", []))
        vertices = list(page.get("vertices", []))
        endpoints = list(page.get("endpoints", []))
        fragment_ids = [str(item.get("id")) for item in fragments]
        vertex_ids = [str(item.get("id")) for item in vertices]
        endpoint_ids = [str(item.get("id")) for item in endpoints]
        if len(fragment_ids) != len(set(fragment_ids)):
            errors.append(f"{page_ref}: duplicate fragment IDs")
        if len(vertex_ids) != len(set(vertex_ids)):
            errors.append(f"{page_ref}: duplicate vertex IDs")
        if len(endpoint_ids) != len(set(endpoint_ids)):
            errors.append(f"{page_ref}: duplicate endpoint IDs")
        if all_fragment_ids.intersection(fragment_ids):
            errors.append(f"{page_ref}: fragment IDs are not page-global unique")
        if all_vertex_ids.intersection(vertex_ids):
            errors.append(f"{page_ref}: vertex IDs are not page-global unique")
        all_fragment_ids.update(fragment_ids)
        all_vertex_ids.update(vertex_ids)
        fragment_set, vertex_set, endpoint_set = (
            set(fragment_ids),
            set(vertex_ids),
            set(endpoint_ids),
        )
        fragment_by_id = {str(item["id"]): item for item in fragments}
        for fragment in fragments:
            if fragment.get("page_ref") != page_ref:
                errors.append(f"{fragment.get('id')}: wrong page ownership")
            if fragment.get("quantity_eligible") is not False:
                errors.append(f"{fragment.get('id')}: became quantity eligible")
            if fragment.get("source_kind") not in {
                "native_pdf_vector",
                "raster_vector_candidate",
            }:
                errors.append(f"{fragment.get('id')}: unknown observation source")
            if len(fragment.get("endpoint_vertex_refs", [])) != 2 or any(
                str(ref) not in vertex_set
                for ref in fragment.get("endpoint_vertex_refs", [])
            ):
                errors.append(f"{fragment.get('id')}: invalid endpoint vertex refs")
            provenance = fragment.get("provenance", {})
            if str(provenance.get("source_primitive_ref")) != str(
                fragment.get("source_primitive_ref")
            ):
                errors.append(f"{fragment.get('id')}: source provenance was not preserved")
            if fragment.get("source_kind") == "raster_vector_candidate":
                if provenance.get("coordinate_transform") is None:
                    errors.append(f"{fragment.get('id')}: raster transform missing")
                if provenance.get("model_provenance") is None:
                    errors.append(f"{fragment.get('id')}: raster model provenance missing")
            metric = fragment.get("local_metric_observation", {})
            if metric.get("measurement") != "local_2d_projected_path":
                errors.append(f"{fragment.get('id')}: non-local metric observation")
        source_shapes = defaultdict(set)
        for fragment in fragments:
            source_shapes[fragment.get("source_kind")].add(tuple(sorted(fragment.keys())))
        if any(len(shapes) != 1 for shapes in source_shapes.values()):
            errors.append(f"{page_ref}: inconsistent fragment schemas within a source")
        if len(source_shapes) > 1 and len({next(iter(shapes)) for shapes in source_shapes.values()}) != 1:
            errors.append(f"{page_ref}: native/raster fragment schema mismatch")
        for endpoint in endpoints:
            if str(endpoint.get("fragment_ref")) not in fragment_set:
                errors.append(f"{endpoint.get('id')}: unknown fragment")
            if str(endpoint.get("vertex_ref")) not in vertex_set:
                errors.append(f"{endpoint.get('id')}: unknown vertex")
        for vertex in vertices:
            if any(str(ref) not in fragment_set for ref in vertex.get("fragment_refs", [])):
                errors.append(f"{vertex.get('id')}: unknown incident fragment")
            if any(str(ref) not in endpoint_set for ref in vertex.get("endpoint_refs", [])):
                errors.append(f"{vertex.get('id')}: unknown incident endpoint")
            if int(vertex.get("degree", -1)) != len(set(vertex.get("fragment_refs", []))):
                errors.append(f"{vertex.get('id')}: degree mismatch")
        for branch in page.get("branches", []):
            vertex_ref = str(branch.get("vertex_ref"))
            vertex = next((item for item in vertices if item["id"] == vertex_ref), None)
            if vertex is None or int(vertex.get("degree", 0)) < 3:
                errors.append(f"{branch.get('id')}: branch lacks shared endpoint degree")
        for crossing in page.get("crossings", []):
            refs = [str(ref) for ref in crossing.get("fragment_refs", [])]
            if len(refs) != 2 or any(ref not in fragment_set for ref in refs):
                errors.append(f"{crossing.get('id')}: invalid crossing fragments")
            elif set(fragment_by_id[refs[0]]["endpoint_vertex_refs"]) & set(
                fragment_by_id[refs[1]]["endpoint_vertex_refs"]
            ):
                errors.append(f"{crossing.get('id')}: connected fragments labeled as crossing")
            if crossing.get("connectivity") != "unconnected":
                errors.append(f"{crossing.get('id')}: crossing connectivity promoted")
    return errors
