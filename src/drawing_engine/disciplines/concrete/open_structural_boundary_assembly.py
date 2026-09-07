"""Assemble evidence-backed open structural boundaries inside section scopes.

This Step 5 layer works upstream of closed-contour selection.  It decomposes
page-global native edges at real intersections, extracts branch-free boundary
chains, and preserves every missing landing/support closure as an explicit
interface port.  It never inserts a geometric bridge or authorizes quantity.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
import statistics
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "0.1.0"


def _stable_id(page_number: int, kind: str, *parts: Any) -> str:
    value = "\0".join((str(page_number), kind, *(str(part) for part in parts)))
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{kind}.page_{page_number:04d}.evidence_{digest}"


def _field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _drawing_ref(ref: Any) -> str:
    return str(ref).split(".item[", 1)[0]


def _primitive_ref(ref: Any) -> str:
    return str(ref).split(".segment[", 1)[0]


def _normal_style(style: Mapping[str, Any] | Iterable[Any]) -> tuple[Any, ...]:
    if isinstance(style, Mapping):
        stroke, width, dash = style.get("stroke"), style.get("width"), style.get("dash")
    else:
        stroke, width, dash = list(style)
    return (
        tuple(round(float(value), 3) for value in (stroke or ())),
        round(float(width or 0.0), 3),
        str(dash or "").replace(" ", ""),
    )


def _structural_styles_and_edges(
    view_frame_graph: Mapping[str, Any], scope_ref: str
) -> tuple[set[tuple[Any, ...]], set[str], list[str]]:
    styles: set[tuple[Any, ...]] = set()
    edges: set[str] = set()
    refs: set[str] = set()
    for coordinate_scope in view_frame_graph.get("shared_coordinate_system", {}).get("scopes", []) or []:
        for correspondence in coordinate_scope.get("contour_correspondences", []) or []:
            if str(correspondence.get("child_view_id")) != scope_ref:
                continue
            refs.add(str(correspondence.get("id")))
            for candidate in correspondence.get("candidates", []) or []:
                if candidate.get("state") != "pass":
                    continue
                signature = (candidate.get("child_signature") or {}).get("style_signature")
                if signature:
                    styles.add(_normal_style(signature))
                edges.update(map(str, candidate.get("child_geometry_refs", []) or []))
    return styles, edges, sorted(refs)


def _excluded_native_refs(dimensions: Iterable[Any], rebar_path_graph: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    annotation = set()
    for dimension in dimensions:
        baseline = _field(dimension, "baseline")
        if baseline is not None:
            annotation.add(_primitive_ref(_field(baseline, "primitive_ref", "")))
        for line in _field(dimension, "extension_lines", ()) or ():
            annotation.add(_primitive_ref(_field(line, "primitive_ref", "")))
        annotation.update(_primitive_ref(ref) for ref in _field(dimension, "terminal_refs", ()) or ())
    evidence_backed_fragment_ids = {
        str(fragment_ref)
        for component in rebar_path_graph.get("components", []) or []
        if component.get("mark_hypotheses") or component.get("resolved_group_ids")
        for fragment_ref in component.get("fragment_ids", []) or []
    }
    fragments = rebar_path_graph.get("fragments", []) or []
    evidence_backed_fragment_ids.update(
        str(item.get("id"))
        for item in fragments
        if item.get("mark_hypotheses") or item.get("resolved_group_ids")
    )
    rebar = {
        _primitive_ref(item.get("primitive_ref"))
        for item in fragments
        if item.get("primitive_ref") and str(item.get("id")) in evidence_backed_fragment_ids
    }
    return {ref for ref in annotation if ref}, {ref for ref in rebar if ref}


def _metric_evidence(local: Mapping[str, Any]) -> tuple[float | None, float | None, list[str]]:
    scales = []
    refs = []
    for item in (local.get("local_ownership") or {}).get("attachments", []) or []:
        value = float(item.get("value_mm") or 0.0)
        endpoints = item.get("measured_endpoints", []) or []
        if value <= 0 or len(endpoints) != 2:
            continue
        points = [endpoint.get("point_display") for endpoint in endpoints]
        if not all(points):
            continue
        distance = math.dist(tuple(map(float, points[0])), tuple(map(float, points[1])))
        if distance > 0:
            scales.append(distance / value)
            refs.append(str(item.get("dimension_ref")))
    scale = statistics.median(scales) if scales else None
    tread_values = [
        float(item.get("value_mm"))
        for item in local.get("repeated_projected_measurement_certificates", []) or []
        if item.get("orientation") == "horizontal" and float(item.get("value_mm") or 0) > 0
    ]
    return scale, min(tread_values) if tread_values else None, sorted(set(refs))


def _stepped_structural_edge_refs(
    segments: Iterable[Mapping[str, Any]],
    scale: float | None,
    tread_mm: float | None,
) -> set[str]:
    """Preserve structural stair paths that also serve as dimension targets."""

    if scale is None or tread_mm is None:
        return set()
    tread_points = scale * tread_mm
    by_drawing: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in segments:
        by_drawing[str(item.get("drawing_ref"))].append(item)
    accepted = set()
    for rows in by_drawing.values():
        horizontal_treads = sum(
            item.get("axis") == "horizontal"
            and abs(float(item.get("length_points") or 0.0) - tread_points)
            <= max(2.0, 0.12 * tread_points)
            for item in rows
        )
        vertical_edges = sum(item.get("axis") == "vertical" for item in rows)
        if horizontal_treads >= 3 and vertical_edges >= 3:
            accepted.update(str(item.get("id")) for item in rows)
    return accepted


def _line_intersection(left: Mapping[str, Any], right: Mapping[str, Any], tolerance: float) -> tuple[float, float, list[float]] | None:
    if left.get("kind") != "line" or right.get("kind") != "line":
        return None
    p = tuple(map(float, left["start_display"]))
    q = tuple(map(float, right["start_display"]))
    r = (float(left["end_display"][0]) - p[0], float(left["end_display"][1]) - p[1])
    s = (float(right["end_display"][0]) - q[0], float(right["end_display"][1]) - q[1])
    cross = r[0] * s[1] - r[1] * s[0]
    qp = (q[0] - p[0], q[1] - p[1])
    if abs(cross) <= 1e-9:
        return None
    t = (qp[0] * s[1] - qp[1] * s[0]) / cross
    u = (qp[0] * r[1] - qp[1] * r[0]) / cross
    left_slack = tolerance / max(math.hypot(*r), 1e-9)
    right_slack = tolerance / max(math.hypot(*s), 1e-9)
    if not (-left_slack <= t <= 1.0 + left_slack and -right_slack <= u <= 1.0 + right_slack):
        return None
    t, u = min(1.0, max(0.0, t)), min(1.0, max(0.0, u))
    return t, u, [p[0] + t * r[0], p[1] + t * r[1]]


def _split_intersections(segments: list[dict[str, Any]], tolerance: float, page_number: int) -> tuple[list[dict[str, Any]], int]:
    cuts: dict[str, list[tuple[float, list[float]]]] = {
        str(item["id"]): [(0.0, list(item["start_display"])), (1.0, list(item["end_display"]))]
        for item in segments
    }
    intersection_keys = set()
    for left, right in itertools.combinations(segments, 2):
        result = _line_intersection(left, right, tolerance)
        if result is None:
            continue
        left_t, right_t, point = result
        cuts[str(left["id"])].append((left_t, point))
        cuts[str(right["id"])].append((right_t, point))
        if 1e-8 < left_t < 1.0 - 1e-8 or 1e-8 < right_t < 1.0 - 1e-8:
            intersection_keys.add((round(point[0] / tolerance), round(point[1] / tolerance)))

    endpoint_vertices = {
        (round(float(point[0]) / tolerance), round(float(point[1]) / tolerance)): str(vertex)
        for item in segments
        for point, vertex in (
            (item["start_display"], item["start_vertex_id"]),
            (item["end_display"], item["end_vertex_id"]),
        )
    }
    output = []
    for segment in sorted(segments, key=lambda item: str(item["id"])):
        unique = {}
        for parameter, point in cuts[str(segment["id"])]:
            unique[round(parameter, 9)] = (parameter, point)
        ordered = [unique[key] for key in sorted(unique)]
        for index, ((start_t, start), (end_t, end)) in enumerate(zip(ordered, ordered[1:])):
            if math.dist(start, end) <= tolerance * 0.25:
                continue
            vertices = []
            for point in (start, end):
                key = (round(point[0] / tolerance), round(point[1] / tolerance))
                vertices.append(
                    endpoint_vertices.get(key)
                    or _stable_id(page_number, "structural_intersection_vertex", key[0], key[1])
                )
            ref = str(segment["id"])
            output.append(
                {
                    **segment,
                    "id": ref if len(ordered) == 2 else f"{ref}.split[{index}]",
                    "parent_native_edge_ref": ref,
                    "start_display": [round(float(value), 6) for value in start],
                    "end_display": [round(float(value), 6) for value in end],
                    "start_vertex_id": vertices[0],
                    "end_vertex_id": vertices[1],
                    "length_points": round(math.dist(start, end), 6),
                    "split_parameter_range": [round(start_t, 9), round(end_t, 9)],
                }
            )
    return output, len(intersection_keys)


def _ordered_chains(edges: list[dict[str, Any]], page_number: int) -> list[dict[str, Any]]:
    by_vertex: dict[str, list[int]] = defaultdict(list)
    for index, edge in enumerate(edges):
        by_vertex[str(edge["start_vertex_id"])].append(index)
        by_vertex[str(edge["end_vertex_id"])].append(index)
    unused = set(range(len(edges)))
    chains = []

    def walk(seed: int, start_vertex: str) -> tuple[list[int], list[str]]:
        selected, vertices = [], [start_vertex]
        current_edge, current_vertex = seed, start_vertex
        while current_edge in unused:
            unused.remove(current_edge)
            selected.append(current_edge)
            edge = edges[current_edge]
            current_vertex = (
                str(edge["end_vertex_id"])
                if str(edge["start_vertex_id"]) == current_vertex
                else str(edge["start_vertex_id"])
            )
            vertices.append(current_vertex)
            candidates = [item for item in by_vertex[current_vertex] if item in unused]
            if len(by_vertex[current_vertex]) != 2 or len(candidates) != 1:
                break
            current_edge = candidates[0]
        return selected, vertices

    starts = sorted(
        (vertex, index)
        for vertex, indices in by_vertex.items()
        if len(indices) != 2
        for index in indices
    )
    while unused:
        available = next(((vertex, index) for vertex, index in starts if index in unused), None)
        if available is None:
            seed = min(unused)
            available = (str(edges[seed]["start_vertex_id"]), seed)
        indices, vertices = walk(available[1], available[0])
        rows = [edges[index] for index in indices]
        refs = [str(item["parent_native_edge_ref"]) for item in rows]
        chains.append(
            {
                "id": _stable_id(page_number, "branch_free_structural_chain", *refs, *vertices),
                "state": "derived",
                "split_edge_refs": [str(item["id"]) for item in rows],
                "source_edge_refs": sorted(set(refs)),
                "ordered_vertex_refs": vertices,
                "ordered_points_display": [
                    list(rows[0]["start_display"] if str(rows[0]["start_vertex_id"]) == vertices[0] else rows[0]["end_display"]),
                    *[
                        list(edge["end_display"] if str(edge["start_vertex_id"]) == vertices[index] else edge["start_display"])
                        for index, edge in enumerate(rows)
                    ],
                ],
                "endpoint_vertex_refs": [] if vertices[0] == vertices[-1] else [vertices[0], vertices[-1]],
                "closed": vertices[0] == vertices[-1],
                "branch_free": True,
                "quantity_eligible": False,
            }
        )
    return sorted(chains, key=lambda item: item["id"])


def _axis(left: list[float], right: list[float], tolerance: float = 0.7) -> str:
    dx, dy = abs(right[0] - left[0]), abs(right[1] - left[1])
    if dy <= tolerance and dx > tolerance:
        return "horizontal"
    if dx <= tolerance and dy > tolerance:
        return "vertical"
    return "oblique"


def _role_chains(chains: list[dict[str, Any]], scale: float, tread_mm: float, page_number: int) -> list[dict[str, Any]]:
    output = []
    tread_points = scale * tread_mm
    for chain in chains:
        points = chain["ordered_points_display"]
        source_refs = chain["source_edge_refs"]
        axes = [_axis(left, right) for left, right in zip(points, points[1:])]
        lengths = [math.dist(left, right) for left, right in zip(points, points[1:])]
        matching_treads = {
            index
            for index, (axis, length) in enumerate(zip(axes, lengths))
            if axis == "horizontal" and abs(length - tread_points) <= max(2.0, 0.12 * tread_points)
        }
        stair_indices = set()
        if len(matching_treads) >= 3:
            first, last = min(matching_treads), max(matching_treads)
            stair_indices.update(range(first, last + 1))
            if first and axes[first - 1] == "vertical":
                stair_indices.add(first - 1)
            if last + 1 < len(axes) and axes[last + 1] == "vertical":
                stair_indices.add(last + 1)
        waist_indices = {
            index
            for index, (axis, left, right) in enumerate(zip(axes, points, points[1:]))
            if axis == "oblique"
            and abs(right[0] - left[0]) >= 3.0 * tread_points
            and abs(right[1] - left[1]) >= tread_points
        }
        roles = ["support_or_landing_boundary"] * len(axes)
        for index in stair_indices:
            roles[index] = "stepped_upper_surface"
        for index in waist_indices:
            roles[index] = "waist_underside"
        start = 0
        while start < len(roles):
            role = roles[start]
            end = start + 1
            while end < len(roles) and roles[end] == role:
                end += 1
            selected_points = points[start : end + 1]
            selected_split_refs = chain["split_edge_refs"][start:end]
            selected_axes = axes[start:end]
            selected_box = _bbox(selected_points)
            annotation_like = (
                role == "support_or_landing_boundary"
                and sum(axis == "oblique" for axis in selected_axes) >= 2
                and max(selected_box[2] - selected_box[0], selected_box[3] - selected_box[1])
                < 2.0 * tread_points
            )
            if annotation_like:
                role = "internal_annotation_line"
            selected_source = sorted(
                {
                    str(ref).split(".split[", 1)[0]
                    for ref in selected_split_refs
                }
            )
            role_id = _stable_id(page_number, "structural_boundary_chain", role, *selected_split_refs)
            output.append(
                {
                    "id": role_id,
                    "state": "derived",
                    "role": role,
                    "boundary_eligible": not annotation_like,
                    "parent_branch_free_chain_ref": chain["id"],
                    "split_edge_refs": selected_split_refs,
                    "source_edge_refs": selected_source or source_refs,
                    "ordered_points_display": selected_points,
                    "ordered_vertex_refs": chain["ordered_vertex_refs"][start : end + 1],
                    "endpoint_ports": [
                        {
                            "id": _stable_id(page_number, "structural_interface_port", role_id, side),
                            "side": side,
                            "point_display": list(selected_points[index]),
                            "vertex_ref": chain["ordered_vertex_refs"][start if index == 0 else end],
                            "state": "open_interface",
                            "interface_kind": "landing_or_support_interface",
                        }
                        for side, index in (("start", 0), ("end", -1))
                    ],
                    "quantity_eligible": False,
                }
            )
            start = end
    return sorted(output, key=lambda item: item["id"])


def _bbox(points: Iterable[Iterable[float]]) -> list[float]:
    rows = [list(map(float, point)) for point in points]
    return [min(p[0] for p in rows), min(p[1] for p in rows), max(p[0] for p in rows), max(p[1] for p in rows)]


def _support_paths(
    start_vertex: str,
    end_vertex: str,
    supports: list[dict[str, Any]],
    *,
    limit: int = 3,
    max_chains: int = 8,
) -> list[list[tuple[dict[str, Any], bool]]]:
    by_vertex: dict[str, list[tuple[dict[str, Any], bool, str]]] = defaultdict(list)
    for chain in supports:
        ports = chain.get("endpoint_ports", []) or []
        if len(ports) != 2:
            continue
        left, right = str(ports[0]["vertex_ref"]), str(ports[1]["vertex_ref"])
        by_vertex[left].append((chain, True, right))
        by_vertex[right].append((chain, False, left))
    solutions = []

    def search(vertex: str, selected: list[tuple[dict[str, Any], bool]], used: set[str]) -> None:
        if len(solutions) >= limit or len(selected) > max_chains:
            return
        if vertex == end_vertex:
            solutions.append(selected)
            return
        for chain, forward, target in sorted(by_vertex.get(vertex, []), key=lambda item: item[0]["id"]):
            if chain["id"] in used:
                continue
            search(target, [*selected, (chain, forward)], {*used, chain["id"]})

    search(start_vertex, [], set())
    return solutions


def _native_support_closure(
    step: Mapping[str, Any],
    waist: Mapping[str, Any],
    supports: list[dict[str, Any]],
) -> dict[str, Any]:
    step_ports = step.get("endpoint_ports", []) or []
    waist_ports = waist.get("endpoint_ports", []) or []
    configurations = []
    for mapping in ((0, 1), (1, 0)):
        joins = []
        for step_index, waist_index in enumerate(mapping):
            paths = _support_paths(
                str(step_ports[step_index]["vertex_ref"]),
                str(waist_ports[waist_index]["vertex_ref"]),
                supports,
            )
            joins.append(paths[0] if len(paths) == 1 else None)
        used = [
            {chain["id"] for chain, _ in path}
            for path in joins
            if path is not None
        ]
        non_overlapping = len(used) < 2 or not (used[0] & used[1])
        certified_count = sum(path is not None for path in joins) if non_overlapping else 0
        configurations.append(
            {
                "mapping": mapping,
                "joins": joins if non_overlapping else [None, None],
                "certified_count": certified_count,
                "support_chain_count": sum(len(path or []) for path in joins),
            }
        )
    configurations.sort(
        key=lambda item: (-item["certified_count"], item["support_chain_count"], item["mapping"])
    )
    best = configurations[0]
    tied = [
        item
        for item in configurations[1:]
        if (item["certified_count"], item["support_chain_count"])
        == (best["certified_count"], best["support_chain_count"])
    ]
    if tied:
        best = {"mapping": (0, 1), "joins": [None, None], "certified_count": 0, "support_chain_count": 0}
    support_refs = [
        chain["id"]
        for path in best["joins"]
        if path is not None
        for chain, _ in path
    ]
    connected = {
        ("stepped_upper_surface", index)
        for index, path in enumerate(best["joins"])
        if path is not None
    } | {
        ("waist_underside", best["mapping"][index])
        for index, path in enumerate(best["joins"])
        if path is not None
    }
    boundary_points: list[list[float]] = []
    if best["certified_count"] == 2:
        boundary_points = [list(point) for point in step["ordered_points_display"]]

        def append_path(path: list[tuple[dict[str, Any], bool]], reverse: bool = False) -> None:
            rows = list(reversed(path)) if reverse else path
            for chain, forward in rows:
                points = list(chain["ordered_points_display"])
                use_forward = forward if not reverse else not forward
                oriented = points if use_forward else list(reversed(points))
                boundary_points.extend(list(point) for point in oriented[1:])

        append_path(best["joins"][1])
        waist_points = list(waist["ordered_points_display"])
        if best["mapping"][1] == 1:
            waist_points.reverse()
        boundary_points.extend(list(point) for point in waist_points[1:])
        append_path(best["joins"][0], reverse=True)
        if len(boundary_points) > 1 and math.dist(boundary_points[0], boundary_points[-1]) <= 1e-6:
            boundary_points.pop()
    return {
        "closed": best["certified_count"] == 2,
        "support_boundary_chain_refs": support_refs,
        "connected_port_keys": connected,
        "certified_native_support_join_count": best["certified_count"],
        "interface_mapping": list(best["mapping"]),
        "ambiguous_support_path": bool(tied),
        "ordered_boundary_display": boundary_points,
    }


def _flight_proposals(role_chains: list[dict[str, Any]], scale: float, tread_mm: float, metric_refs: list[str], page_number: int) -> list[dict[str, Any]]:
    steps = [item for item in role_chains if item["role"] == "stepped_upper_surface"]
    waists = [item for item in role_chains if item["role"] == "waist_underside"]
    supports = [
        item
        for item in role_chains
        if item["role"] == "support_or_landing_boundary" and item.get("boundary_eligible")
    ]
    proposals = []
    for step in steps:
        step_box = _bbox(step["ordered_points_display"])
        compatible = []
        for waist in waists:
            waist_box = _bbox(waist["ordered_points_display"])
            overlap = min(step_box[2], waist_box[2]) - max(step_box[0], waist_box[0])
            step_dx = step["ordered_points_display"][-1][0] - step["ordered_points_display"][0][0]
            step_dy = step["ordered_points_display"][-1][1] - step["ordered_points_display"][0][1]
            waist_dx = waist["ordered_points_display"][-1][0] - waist["ordered_points_display"][0][0]
            waist_dy = waist["ordered_points_display"][-1][1] - waist["ordered_points_display"][0][1]
            same_slope_sign = step_dx * step_dy * waist_dx * waist_dy > 0
            if overlap >= 3.0 * tread_mm * scale and same_slope_sign:
                distance = sum(
                    min(math.dist(port["point_display"], other["point_display"]) for other in waist["endpoint_ports"])
                    for port in step["endpoint_ports"]
                )
                compatible.append((distance, waist))
        if not compatible:
            continue
        compatible.sort(key=lambda item: (item[0], item[1]["id"]))
        best_distance = compatible[0][0]
        selected = [item for distance, item in compatible if distance <= best_distance + max(2.0, 0.02 * tread_mm * scale)]
        for waist in selected:
            proposal_id = _stable_id(page_number, "flight_boundary_proposal", step["id"], waist["id"])
            support_closure = _native_support_closure(step, waist, supports)
            ports = [
                {**port, "id": _stable_id(page_number, "flight_interface_port", proposal_id, source, port["side"]), "source_boundary_role": source}
                for source, boundary in (("stepped_upper_surface", step), ("waist_underside", waist))
                for port_index, port in enumerate(boundary["endpoint_ports"])
                if (source, port_index) not in support_closure["connected_port_keys"]
            ]
            support_by_id = {item["id"]: item for item in supports}
            support_edges = {
                ref
                for chain_ref in support_closure["support_boundary_chain_refs"]
                for ref in support_by_id[chain_ref]["source_edge_refs"]
            }
            evidence_refs = sorted({*step["source_edge_refs"], *waist["source_edge_refs"], *support_edges, *metric_refs})
            closed = support_closure["closed"]
            proposals.append(
                {
                    "id": proposal_id,
                    "record_type": "interface_bounded_flight_proposal",
                    "state": "closed" if closed else "incomplete",
                    "stepped_surface_chain_ref": step["id"],
                    "waist_underside_chain_ref": waist["id"],
                    "support_or_landing_boundary_refs": support_closure["support_boundary_chain_refs"],
                    "interface_ports": ports,
                    "closure": {
                        "closed": closed,
                        "branch_free": True,
                        "unique_completion": closed and not support_closure["ambiguous_support_path"],
                        "scale_bounded": True,
                        "invented_bridge_count": 0,
                        "explicit_open_interface_count": len(ports),
                        "certified_native_support_join_count": support_closure["certified_native_support_join_count"],
                        "ambiguous_support_path": support_closure["ambiguous_support_path"],
                    },
                    "join_certificate": {
                        "compatible_metric_scale": True,
                        "repeated_tread_station_agreement": True,
                        "compatible_native_style": True,
                        "horizontal_station_overlap": True,
                        "unique_waist_assignment": len(selected) == 1,
                        "native_support_interface_mapping": support_closure["interface_mapping"],
                    },
                    "dimension_certificate": {
                        "scale_points_per_mm": round(scale, 9),
                        "tread_run_mm": tread_mm,
                        "dimension_refs": metric_refs,
                    },
                    "source_edge_refs": sorted({*step["source_edge_refs"], *waist["source_edge_refs"], *support_edges}),
                    "primitive_refs": sorted(
                        {
                            _primitive_ref(ref)
                            for ref in {*step["source_edge_refs"], *waist["source_edge_refs"], *support_edges}
                        }
                    ),
                    "ordered_boundary_display": support_closure["ordered_boundary_display"],
                    "derived_bridges": [],
                    "evidence_refs": evidence_refs,
                    "quantity_eligible": False,
                }
            )
    return sorted(proposals, key=lambda item: item["id"])


def assemble_open_structural_boundaries(
    topology: Mapping[str, Any],
    title_segmentation: Mapping[str, Any],
    local_dimension_reclosure: Mapping[str, Any],
    view_frame_graph: Mapping[str, Any],
    dimensions: Iterable[Any],
    rebar_path_graph: Mapping[str, Any],
    *,
    page_number: int,
) -> dict[str, Any]:
    """Publish closed profiles or explicit incomplete structural proposals."""

    scopes = {
        str(item.get("id")): item
        for item in title_segmentation.get("segments", []) or []
        if item.get("state") == "resolved"
    }
    annotation_refs, rebar_refs = _excluded_native_refs(dimensions, rebar_path_graph)
    results = []
    for local in local_dimension_reclosure.get("scope_results", []) or []:
        scope_ref = str(local.get("scope_ref"))
        scope = scopes.get(scope_ref)
        styles, reprojection_edges, correspondence_refs = _structural_styles_and_edges(view_frame_graph, scope_ref)
        scale, tread_mm, metric_refs = _metric_evidence(local)
        allowed_drawings = (
            set(map(str, scope.get("primitive_refs", []) or []))
            - set(map(str, scope.get("excluded_primitive_refs", []) or []))
            if scope
            else set()
        )
        scoped_edges = [
            dict(item)
            for item in topology.get("segments", []) or []
            if str(item.get("drawing_ref")) in allowed_drawings
        ]
        structural_supports = [
            item
            for item in scoped_edges
            if _normal_style(item.get("style") or {}) in styles
        ]
        stepped_dual_role_refs = _stepped_structural_edge_refs(
            structural_supports,
            scale,
            tread_mm,
        )
        eligible = [
            item
            for item in structural_supports
            if (
                _primitive_ref(item.get("id")) not in annotation_refs
                or str(item.get("id")) in stepped_dual_role_refs
            )
            and _primitive_ref(item.get("id")) not in rebar_refs
        ]
        eligible_refs = {str(item.get("id")) for item in eligible}
        missing_reprojection = sorted(reprojection_edges - eligible_refs)
        split_edges: list[dict[str, Any]] = []
        branch_chains: list[dict[str, Any]] = []
        role_chains: list[dict[str, Any]] = []
        proposals: list[dict[str, Any]] = []
        intersection_count = 0
        reason = None
        if scope is None or local.get("status") != "resolved_subset":
            reason = "title_scope_metric_reclosure_unresolved"
        elif not styles:
            reason = "no_existing_structural_style_certificate"
        elif scale is None or tread_mm is None:
            reason = "metric_scale_or_repeated_tread_unresolved"
        elif missing_reprojection:
            reason = "accepted_reprojection_edges_not_eligible"
        else:
            split_edges, intersection_count = _split_intersections(
                eligible,
                float(topology.get("vertex_tolerance_points") or 0.75),
                page_number,
            )
            branch_chains = _ordered_chains(split_edges, page_number)
            role_chains = _role_chains(branch_chains, scale, tread_mm, page_number)
            proposals = _flight_proposals(role_chains, scale, tread_mm, metric_refs, page_number)
            if not proposals:
                reason = "no_interface_bounded_flight_proposal"
        closed_profiles = [item for item in proposals if item["closure"]["closed"]]
        transition = (
            f"{len(structural_supports)} native structural supports -> {len(eligible)} eligible edges -> "
            f"{len(role_chains)} branch-free chains -> {len(proposals)} interface-bounded flight proposals -> "
            f"{len(closed_profiles)} closed profiles"
        )
        results.append(
            {
                "scope_ref": scope_ref,
                "status": "closed_profiles_available" if closed_profiles else "incomplete_proposals" if proposals else "insufficient_constraints",
                "reason_code": None if closed_profiles else reason or "flight_interfaces_remain_open",
                "transition": transition,
                "scope_native_edge_count": len(scoped_edges),
                "native_structural_support_count": len(structural_supports),
                "eligible_edge_count": len(eligible),
                "split_edge_count": len(split_edges),
                "derived_intersection_vertex_count": intersection_count,
                "branch_free_chain_count": len(role_chains),
                "maximal_branch_free_chain_count": len(branch_chains),
                "interface_bounded_flight_proposal_count": len(proposals),
                "closed_profile_count": len(closed_profiles),
                "structural_style_signatures": [[list(item[0]), item[1], item[2]] for item in sorted(styles)],
                "contour_correspondence_refs": correspondence_refs,
                "reprojection_edge_refs": sorted(reprojection_edges),
                "reprojection_edges_missing_from_eligible": missing_reprojection,
                "excluded_annotation_edge_count": sum(_primitive_ref(item.get("id")) in annotation_refs for item in structural_supports),
                "dual_role_structural_override_count": sum(
                    _primitive_ref(item.get("id")) in annotation_refs
                    and str(item.get("id")) in stepped_dual_role_refs
                    for item in structural_supports
                ),
                "dual_role_structural_override_edge_refs": sorted(
                    str(item.get("id"))
                    for item in structural_supports
                    if _primitive_ref(item.get("id")) in annotation_refs
                    and str(item.get("id")) in stepped_dual_role_refs
                ),
                "excluded_rebar_edge_count": sum(_primitive_ref(item.get("id")) in rebar_refs for item in structural_supports),
                "metric_scale_points_per_mm": None if scale is None else round(scale, 9),
                "tread_run_mm": tread_mm,
                "branch_free_chains": role_chains,
                "flight_boundary_proposals": proposals,
                "closed_profiles": closed_profiles,
                "quantity_eligible": False,
            }
        )
    summary = {
        "scope_count": len(results),
        "native_structural_support_count": sum(item["native_structural_support_count"] for item in results),
        "eligible_edge_count": sum(item["eligible_edge_count"] for item in results),
        "branch_free_chain_count": sum(item["branch_free_chain_count"] for item in results),
        "interface_bounded_flight_proposal_count": sum(item["interface_bounded_flight_proposal_count"] for item in results),
        "closed_profile_count": sum(item["closed_profile_count"] for item in results),
        "transition": results[0]["transition"] if len(results) == 1 else None,
        "boundary_role_counts": dict(sorted(Counter(chain["role"] for item in results for chain in item["branch_free_chains"]).items())),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "open_structural_boundary_assembly",
        "page": page_number,
        "status": "closed_profiles_available" if summary["closed_profile_count"] else "incomplete_proposals" if summary["interface_bounded_flight_proposal_count"] else "insufficient_constraints",
        "scope_results": results,
        "summary": summary,
        "contract": {
            "page_global_native_edges_required": True,
            "structural_style_comes_from_existing_correspondence_evidence": True,
            "hard_coded_line_width_forbidden": True,
            "intersections_split_into_stable_vertices": True,
            "dimension_rebar_and_annotation_edges_excluded": True,
            "open_interfaces_remain_explicit": True,
            "closure_bridges_invented": False,
            "landing_or_mesh_construction_performed": False,
            "quantity_eligible": False,
            "schedule_values_used": False,
        },
    }
