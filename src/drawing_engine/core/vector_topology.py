"""Shared native-vector segment and vertex topology.

PDF drawings frequently split one physical edge across several path items or
several drawing records.  This module gives those observations stable segment
and display-space vertex identifiers without assigning an object class.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math
from typing import Any, Iterable

import fitz

from src.drawing_engine.core.pdf_native_metadata import drawing_metadata, get_native_drawings


def _point(value: Any) -> tuple[float, float]:
    if hasattr(value, "x"):
        return (float(value.x), float(value.y))
    return (float(value[0]), float(value[1]))


def _axis(start: tuple[float, float], end: tuple[float, float]) -> str:
    dx, dy = abs(end[0] - start[0]), abs(end[1] - start[1])
    if dy <= 0.7 and dx >= 1.0:
        return "horizontal"
    if dx <= 0.7 and dy >= 1.0:
        return "vertical"
    return "oblique"


def _cubic_points(item: tuple[Any, ...], steps: int = 16) -> list[tuple[float, float]]:
    p0, p1, p2, p3 = (_point(item[index]) for index in range(1, 5))
    output = []
    for index in range(steps + 1):
        t = index / steps
        u = 1.0 - t
        output.append(
            (
                u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
                u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1],
            )
        )
    return output


def _item_parts(item: tuple[Any, ...]) -> list[dict[str, Any]]:
    kind = item[0]
    if kind == "l":
        start, end = _point(item[1]), _point(item[2])
        return [{"kind": "line", "start": start, "end": end, "control_points": []}]
    if kind == "re":
        # get_drawings() normalizes compact C-drawing rectangles.  Keep that
        # exact geometry when consuming get_cdrawings() path callbacks.
        rect = fitz.Rect(item[1]).normalize()
        points = [(rect.x0, rect.y0), (rect.x1, rect.y0), (rect.x1, rect.y1), (rect.x0, rect.y1)]
        return [
            {"kind": "line", "start": start, "end": end, "control_points": []}
            for start, end in zip(points, points[1:] + points[:1])
        ]
    if kind == "qu":
        quad = fitz.Quad(item[1])
        points = [_point(quad.ul), _point(quad.ur), _point(quad.lr), _point(quad.ll)]
        return [
            {"kind": "line", "start": start, "end": end, "control_points": []}
            for start, end in zip(points, points[1:] + points[:1])
        ]
    if kind == "c":
        points = _cubic_points(item)
        return [
            {
                "kind": "cubic",
                "start": points[0],
                "end": points[-1],
                "control_points": [_point(item[2]), _point(item[3])],
                "sample_points": points,
            }
        ]
    return []


def _length(part: dict[str, Any]) -> float:
    points = part.get("sample_points") or [part["start"], part["end"]]
    return sum(math.dist(left, right) for left, right in zip(points, points[1:]))


def _assign_vertices(segments: list[dict[str, Any]], tolerance: float) -> list[dict[str, Any]]:
    """Attach stable page-global vertex IDs using a small spatial hash."""

    endpoints = sorted(
        (
            float(point[0]),
            float(point[1]),
            segment_index,
            endpoint_name,
        )
        for segment_index, segment in enumerate(segments)
        for endpoint_name, point in (("start", segment["start_display"]), ("end", segment["end_display"]))
    )
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    vertices: list[dict[str, Any]] = []
    assignments: dict[tuple[int, str], int] = {}
    for x, y, segment_index, endpoint_name in endpoints:
        cell = (math.floor(x / tolerance), math.floor(y / tolerance))
        candidates = [
            vertex_index
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for vertex_index in cells.get((cell[0] + dx, cell[1] + dy), [])
            if math.dist((x, y), tuple(vertices[vertex_index]["point_display"])) <= tolerance
        ]
        if candidates:
            vertex_index = min(
                candidates,
                key=lambda index: (math.dist((x, y), tuple(vertices[index]["point_display"])), index),
            )
            vertex = vertices[vertex_index]
            count = vertex.pop("_count")
            vertex["point_display"] = [
                round((vertex["point_display"][0] * count + x) / (count + 1), 6),
                round((vertex["point_display"][1] * count + y) / (count + 1), 6),
            ]
            vertex["_count"] = count + 1
        else:
            vertex_index = len(vertices)
            vertices.append(
                {
                    "id": f"geometry_vertex.{vertex_index + 1:06d}",
                    "point_display": [round(x, 6), round(y, 6)],
                    "_count": 1,
                }
            )
            cells[cell].append(vertex_index)
        assignments[(segment_index, endpoint_name)] = vertex_index
    for segment_index, segment in enumerate(segments):
        segment["start_vertex_id"] = vertices[assignments[(segment_index, "start")]]["id"]
        segment["end_vertex_id"] = vertices[assignments[(segment_index, "end")]]["id"]
    for vertex in vertices:
        vertex.pop("_count", None)
    return vertices


def iter_native_segments(drawings, drawing_filter=None, *, drawing_index_offset=0):
    """Yield canonical native segment records before optional vertex assembly.

    A filter receives the original drawing record, without renumbering any
    primitive. Selected records do not establish full-page topology coverage.
    """
    for relative_index, drawing in enumerate(drawings):
        drawing_index = drawing_index_offset + relative_index
        if drawing_filter is not None and not drawing_filter(drawing):
            continue
        drawing_ref = f"drawing[{drawing_index}]"
        metadata = drawing_metadata(drawing)
        for item_index, item in enumerate(drawing.get("items", [])):
            primitive_ref = f"{drawing_ref}.item[{item_index}]"
            for part_index, part in enumerate(_item_parts(item)):
                start, end = part["start"], part["end"]
                segment = {
                    "id": f"{primitive_ref}.segment[{part_index}]",
                    "drawing_ref": drawing_ref,
                    "primitive_ref": primitive_ref,
                    "item_index": item_index,
                    "part_index": part_index,
                    "kind": part["kind"],
                    "start_display": [round(start[0], 6), round(start[1], 6)],
                    "end_display": [round(end[0], 6), round(end[1], 6)],
                    "control_points_display": [list(map(float, point)) for point in part.get("control_points", [])],
                    "sample_points_display": [list(map(float, point)) for point in part.get("sample_points", [])],
                    "axis": _axis(start, end),
                    "length_points": round(_length(part), 6),
                    "bbox_display": [
                        round(min(start[0], end[0], *(point[0] for point in part.get("sample_points", []))), 6),
                        round(min(start[1], end[1], *(point[1] for point in part.get("sample_points", []))), 6),
                        round(max(start[0], end[0], *(point[0] for point in part.get("sample_points", []))), 6),
                        round(max(start[1], end[1], *(point[1] for point in part.get("sample_points", []))), 6),
                    ],
                    "style": {
                        "width": drawing.get("width"),
                        "stroke": drawing.get("color"),
                        "fill": drawing.get("fill"),
                        "dash": drawing.get("dashes"),
                    },
                }
                if metadata:
                    segment['native_metadata'] = deepcopy(metadata)
                yield segment


def extract_page_topology(page: fitz.Page, vertex_tolerance: float = 0.75,
                          *, include_native_metadata: bool = False) -> dict[str, Any]:
    """Return all supported native path segments with page-global vertices."""

    drawings, metadata_inventory = get_native_drawings(page, extended_metadata=include_native_metadata)
    drawing_styles = {
        f"drawing[{index}]": {
            "width": drawing.get("width"),
            "stroke": drawing.get("color"),
            "fill": drawing.get("fill"),
            "dash": drawing.get("dashes"),
            "close_path": bool(drawing.get("closePath")),
            "rect_display": list(fitz.Rect(drawing["rect"])),
        }
        for index, drawing in enumerate(drawings)
    }
    segments = list(iter_native_segments(drawings))
    vertices = _assign_vertices(segments, vertex_tolerance)
    incident: dict[str, list[str]] = defaultdict(list)
    for segment in segments:
        incident[segment["start_vertex_id"]].append(segment["id"])
        incident[segment["end_vertex_id"]].append(segment["id"])
    for vertex in vertices:
        vertex["segment_ids"] = sorted(incident[vertex["id"]])
        vertex["degree"] = len(vertex["segment_ids"])
    return {
        "schema_version": "0.1.0",
        "segments": segments,
        "vertices": vertices,
        "drawing_styles": drawing_styles,
        "vertex_tolerance_points": vertex_tolerance,
        **({'native_metadata_inventory': metadata_inventory} if include_native_metadata
           or metadata_inventory['optional_content_groups'] else {}),
    }


def topology_for_drawing(topology: dict[str, Any], drawing_ref: str) -> dict[str, Any]:
    """Return ordered segments and graph invariants for one PDF drawing path."""

    segments = [segment for segment in topology["segments"] if segment["drawing_ref"] == drawing_ref]
    vertex_ids = sorted(
        {vertex_id for segment in segments for vertex_id in (segment["start_vertex_id"], segment["end_vertex_id"])}
    )
    degrees = {vertex_id: 0 for vertex_id in vertex_ids}
    adjacency: dict[str, set[str]] = defaultdict(set)
    for segment in segments:
        left, right = segment["start_vertex_id"], segment["end_vertex_id"]
        degrees[left] += 1
        degrees[right] += 1
        adjacency[left].add(right)
        adjacency[right].add(left)
    components = 0
    visited = set()
    for seed in vertex_ids:
        if seed in visited:
            continue
        components += 1
        stack = [seed]
        while stack:
            vertex_id = stack.pop()
            if vertex_id in visited:
                continue
            visited.add(vertex_id)
            stack.extend(adjacency[vertex_id] - visited)
    cycle_rank = max(0, len(segments) - len(vertex_ids) + components)
    style = topology["drawing_styles"].get(drawing_ref, {})
    graph_closed = bool(segments) and all(degree >= 2 and degree % 2 == 0 for degree in degrees.values())
    return {
        "segments": segments,
        "vertex_ids": vertex_ids,
        "segment_count": len(segments),
        "vertex_count": len(vertex_ids),
        "connected_component_count": components,
        "endpoint_vertex_count": sum(degree == 1 for degree in degrees.values()),
        "branch_vertex_count": sum(degree > 2 for degree in degrees.values()),
        "cycle_rank": cycle_rank,
        "closed": bool(style.get("close_path") or graph_closed),
    }


def _style_key(style: dict[str, Any]) -> tuple[Any, ...]:
    stroke = style.get("stroke")
    stroke_key = None if stroke is None else tuple(round(float(value), 2) for value in stroke)
    width = style.get("width")
    return (
        stroke_key,
        None if width is None else round(float(width), 2),
        str(style.get("dash") or "").replace(" ", ""),
    )


def composite_topologies(topology: dict[str, Any]) -> list[dict[str, Any]]:
    """Join compatible path records through shared page-global vertices."""

    segments = topology.get("segments", [])
    parent = list(range(len(segments)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    by_style_vertex: dict[tuple[Any, ...], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, segment in enumerate(segments):
        style = topology.get("drawing_styles", {}).get(segment["drawing_ref"], {})
        key = _style_key(style)
        for vertex_id in (segment["start_vertex_id"], segment["end_vertex_id"]):
            by_style_vertex[key][vertex_id].append(index)
    for vertices in by_style_vertex.values():
        for rows in vertices.values():
            for index in rows[1:]:
                union(rows[0], index)
    components: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, segment in enumerate(segments):
        components[find(index)].append(segment)
    output = []
    for rows in components.values():
        drawing_refs = sorted({item["drawing_ref"] for item in rows})
        if len(drawing_refs) < 2 or len(rows) < 3:
            continue
        degrees: dict[str, int] = defaultdict(int)
        for segment in rows:
            degrees[segment["start_vertex_id"]] += 1
            degrees[segment["end_vertex_id"]] += 1
        endpoints = sum(value == 1 for value in degrees.values())
        branches = sum(value > 2 for value in degrees.values())
        closed = bool(rows) and endpoints == 0 and branches == 0 and all(value == 2 for value in degrees.values())
        if not closed and endpoints != 2:
            continue
        x_values = [float(point[0]) for segment in rows for point in (segment["start_display"], segment["end_display"])]
        y_values = [float(point[1]) for segment in rows for point in (segment["start_display"], segment["end_display"])]
        bbox = [min(x_values), min(y_values), max(x_values), max(y_values)]
        if bbox[2] - bbox[0] < 2 or bbox[3] - bbox[1] < 2:
            continue
        output.append(
            {
                "segments": sorted(rows, key=lambda item: (item["drawing_ref"], item["item_index"], item["part_index"])),
                "drawing_refs": drawing_refs,
                "vertex_ids": sorted(degrees),
                "bbox_display": [round(value, 6) for value in bbox],
                "closed": closed,
                "segment_count": len(rows),
                "vertex_count": len(degrees),
                "connected_component_count": 1,
                "endpoint_vertex_count": endpoints,
                "branch_vertex_count": branches,
                "cycle_rank": max(0, len(rows) - len(degrees) + 1),
            }
        )
    return sorted(output, key=lambda item: (*item["bbox_display"], item["drawing_refs"]))


def segments_by_id(topology: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {segment["id"]: segment for segment in topology.get("segments", [])}


def vertices_by_id(topology: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {vertex["id"]: vertex for vertex in topology.get("vertices", [])}


def primitive_segment_index(topology: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for segment in topology.get("segments", []):
        output[segment["primitive_ref"]].append(segment)
    return dict(output)


def drawing_segment_index(topology: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for segment in topology.get("segments", []):
        output[segment["drawing_ref"]].append(segment)
    return dict(output)


def point_distance_to_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    dx, dy = bx - ax, by - ay
    squared = dx * dx + dy * dy
    if squared <= 1e-12:
        return math.hypot(px - ax, py - ay)
    ratio = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / squared))
    return math.hypot(px - (ax + ratio * dx), py - (ay + ratio * dy))


def unique_points(values: Iterable[float], tolerance: float) -> list[float]:
    output = []
    for value in sorted(float(item) for item in values):
        if not output or abs(value - output[-1]) > tolerance:
            output.append(value)
        else:
            output[-1] = (output[-1] + value) / 2.0
    return output
