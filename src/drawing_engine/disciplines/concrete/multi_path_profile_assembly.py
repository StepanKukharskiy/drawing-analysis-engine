"""Fail-closed assembly of concrete-profile boundaries split across PDF paths.

The kernel operates only on the page-global native topology.  It may bridge a
small drafting gap, but it does not assign a physical object or authorize a
quantity.  A boundary is resolved only when compatible native paths have one
branch-free completion supported by a repeated metric scale.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import statistics
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "0.1.0"


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _dimension_ref(item: Any) -> str:
    return str(_value(item, "attachment_id", _value(item, "id", "")))


def _dimension_scale(item: Any) -> float | None:
    scale = _value(item, "scale_points_per_mm")
    if scale is not None and float(scale) > 0:
        return float(scale)
    points = _value(item, "measured_points_display", _value(item, "measured_points"))
    value_mm = _value(item, "value_mm")
    if not points or len(points) != 2 or value_mm is None or float(value_mm) <= 0:
        return None
    return math.dist(tuple(map(float, points[0])), tuple(map(float, points[1]))) / float(value_mm)


def _metric_certificate(
    dimensions: Iterable[Any],
    tolerance_ratio: float,
) -> dict[str, Any]:
    rows = []
    for item in dimensions:
        state = str(_value(item, "status", _value(item, "state", "")))
        ref = _dimension_ref(item)
        scale = _dimension_scale(item)
        if state not in {"accepted", "detector_qualified", "resolved"} or not ref or scale is None:
            continue
        rows.append((ref, scale))
    rows = sorted({(ref, scale) for ref, scale in rows})
    by_scale = sorted(rows, key=lambda row: (row[1], row[0]))
    clusters: list[list[tuple[str, float]]] = []
    for left in range(len(by_scale)):
        for right in range(left + 2, len(by_scale) + 1):
            cluster = by_scale[left:right]
            median = statistics.median(scale for _, scale in cluster)
            spread = cluster[-1][1] - cluster[0][1]
            if spread > tolerance_ratio * median:
                break
            if cluster not in clusters:
                clusters.append(cluster)
    eligible = [cluster for cluster in clusters if len({ref for ref, _ in cluster}) >= 2]
    if not eligible:
        return {
            "passed": False,
            "reason": "fewer than two accepted dimension chains support one metric scale",
            "dimension_refs": [],
            "independent_chain_count": 0,
            "scale_points_per_mm": None,
            "scale_spread_ratio": None,
        }
    eligible.sort(
        key=lambda cluster: (
            -len({ref for ref, _ in cluster}),
            statistics.median(scale for _, scale in cluster),
            tuple(ref for ref, _ in cluster),
        )
    )
    selected = eligible[0]
    support = len({ref for ref, _ in selected})
    tied = [
        cluster
        for cluster in eligible[1:]
        if len({ref for ref, _ in cluster}) == support
        and abs(
            statistics.median(scale for _, scale in cluster)
            - statistics.median(scale for _, scale in selected)
        )
        > tolerance_ratio * statistics.median(scale for _, scale in selected)
    ]
    if tied:
        return {
            "passed": False,
            "reason": "multiple equally supported metric scales remain",
            "dimension_refs": [],
            "independent_chain_count": support,
            "scale_points_per_mm": None,
            "scale_spread_ratio": None,
        }
    scales = [scale for _, scale in selected]
    median = statistics.median(scales)
    return {
        "passed": True,
        "reason": None,
        "dimension_refs": sorted({ref for ref, _ in selected}),
        "independent_chain_count": support,
        "scale_points_per_mm": round(median, 9),
        "scale_spread_ratio": round((max(scales) - min(scales)) / median, 9),
    }


def _style_key(segment: Mapping[str, Any], topology: Mapping[str, Any]) -> tuple[Any, ...]:
    style = segment.get("style") or topology.get("drawing_styles", {}).get(segment.get("drawing_ref"), {})
    stroke = style.get("stroke")
    fill = style.get("fill")
    return (
        None if stroke is None else tuple(round(float(value), 2) for value in stroke),
        None if fill is None else tuple(round(float(value), 2) for value in fill),
        None if style.get("width") is None else round(float(style["width"]), 2),
        str(style.get("dash") or "").replace(" ", ""),
    )


def _selected_segments(
    topology: Mapping[str, Any],
    allowed_primitive_refs: Iterable[str] | None,
) -> list[dict[str, Any]]:
    if allowed_primitive_refs is None:
        return [dict(item) for item in topology.get("segments", [])]
    allowed = {str(item) for item in allowed_primitive_refs}
    return [
        dict(item)
        for item in topology.get("segments", [])
        if str(item.get("id")) in allowed
        or str(item.get("primitive_ref")) in allowed
        or str(item.get("drawing_ref")) in allowed
    ]


def _native_components(
    segments: list[dict[str, Any]],
    topology: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_style: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for segment in segments:
        by_style[_style_key(segment, topology)].append(segment)
    output = []
    for style_key, rows in sorted(by_style.items(), key=lambda item: repr(item[0])):
        by_vertex: dict[str, list[int]] = defaultdict(list)
        for index, segment in enumerate(rows):
            by_vertex[str(segment["start_vertex_id"])].append(index)
            by_vertex[str(segment["end_vertex_id"])].append(index)
        remaining = set(range(len(rows)))
        while remaining:
            seed = min(remaining)
            indices = set()
            stack = [seed]
            while stack:
                index = stack.pop()
                if index in indices:
                    continue
                indices.add(index)
                segment = rows[index]
                for vertex in (str(segment["start_vertex_id"]), str(segment["end_vertex_id"])):
                    stack.extend(candidate for candidate in by_vertex[vertex] if candidate not in indices)
            remaining -= indices
            members = sorted(
                (rows[index] for index in indices),
                key=lambda item: str(item["id"]),
            )
            degrees: dict[str, int] = defaultdict(int)
            for segment in members:
                degrees[str(segment["start_vertex_id"])] += 1
                degrees[str(segment["end_vertex_id"])] += 1
            endpoints = sorted(vertex for vertex, degree in degrees.items() if degree == 1)
            points = [
                tuple(map(float, point))
                for segment in members
                for point in (segment["start_display"], segment["end_display"])
            ]
            output.append(
                {
                    "segments": members,
                    "style_key": style_key,
                    "degrees": dict(degrees),
                    "endpoints": endpoints,
                    "branch_vertex_ids": sorted(vertex for vertex, degree in degrees.items() if degree > 2),
                    "drawing_refs": sorted({str(item["drawing_ref"]) for item in members}),
                    "bbox_display": [
                        min(point[0] for point in points),
                        min(point[1] for point in points),
                        max(point[0] for point in points),
                        max(point[1] for point in points),
                    ],
                }
            )
    return sorted(
        output,
        key=lambda item: (
            repr(item["style_key"]),
            tuple(str(segment["id"]) for segment in item["segments"]),
        ),
    )


def _vertex_points(topology: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    return {
        str(item["id"]): tuple(map(float, item["point_display"]))
        for item in topology.get("vertices", [])
    }


def _bbox_union(boxes: Iterable[Iterable[float]]) -> list[float]:
    rows = [list(map(float, box)) for box in boxes]
    return [
        min(row[0] for row in rows),
        min(row[1] for row in rows),
        max(row[2] for row in rows),
        max(row[3] for row in rows),
    ]


def _record_namespace(page_number: int | None, scope_ref: str | None) -> str:
    page = "unknown" if page_number is None else f"{int(page_number):04d}"
    scope_digest = hashlib.sha256(str(scope_ref or "unscoped").encode("utf-8")).hexdigest()[:12]
    return f"page_{page}.scope_{scope_digest}"


def _bridge_id(namespace: str, left: str, right: str) -> str:
    evidence = "\0".join((namespace, *sorted((left, right)))).encode("utf-8")
    return f"derived_profile_bridge.{namespace}.evidence_{hashlib.sha256(evidence).hexdigest()[:16]}"


def _bridge_candidates(
    components: list[dict[str, Any]],
    points: Mapping[str, tuple[float, float]],
    scale: float,
    max_bridge_mm: float,
    max_relative_gap: float,
    namespace: str,
) -> list[dict[str, Any]]:
    candidates = []
    maximum_points = max_bridge_mm * scale
    cells: dict[tuple[Any, int, int], list[tuple[int, str]]] = defaultdict(list)
    eligible = []
    for component_index, component in enumerate(components):
        if len(component["endpoints"]) != 2 or component["branch_vertex_ids"]:
            continue
        for vertex in component["endpoints"]:
            point = points[vertex]
            cell = (component["style_key"], math.floor(point[0] / maximum_points), math.floor(point[1] / maximum_points))
            cells[cell].append((component_index, vertex))
            eligible.append((component_index, vertex, cell))
    seen = set()
    for left_index, left_vertex, cell in sorted(eligible, key=lambda item: (item[0], item[1])):
        left = components[left_index]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for right_index, right_vertex in cells.get((cell[0], cell[1] + dx, cell[2] + dy), []):
                    if right_index <= left_index:
                        continue
                    pair = (left_vertex, right_vertex)
                    if pair in seen:
                        continue
                    seen.add(pair)
                    right = components[right_index]
                    box = _bbox_union((left["bbox_display"], right["bbox_display"]))
                    span = max(box[2] - box[0], box[3] - box[1])
                    limit_points = min(maximum_points, max_relative_gap * span)
                    distance = math.dist(points[left_vertex], points[right_vertex])
                    if distance > limit_points + 1e-9:
                        continue
                    candidates.append(
                        {
                            "id": _bridge_id(namespace, left_vertex, right_vertex),
                            "left_component": left_index,
                            "right_component": right_index,
                            "start_vertex_id": left_vertex,
                            "end_vertex_id": right_vertex,
                            "start_display": list(points[left_vertex]),
                            "end_display": list(points[right_vertex]),
                            "length_points": round(distance, 9),
                            "length_mm": round(distance / scale, 6),
                            "limit_points": round(limit_points, 9),
                            "max_bridge_mm": max_bridge_mm,
                            "max_relative_gap": max_relative_gap,
                        }
                    )
    return sorted(candidates, key=lambda item: (item["start_vertex_id"], item["end_vertex_id"], item["id"]))


def _component_clusters(component_count: int, bridges: list[dict[str, Any]]) -> list[list[int]]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for bridge in bridges:
        left, right = bridge["left_component"], bridge["right_component"]
        adjacency[left].add(right)
        adjacency[right].add(left)
    clusters = []
    remaining = set(range(component_count))
    while remaining:
        seed = min(remaining)
        cluster = set()
        stack = [seed]
        while stack:
            current = stack.pop()
            if current in cluster:
                continue
            cluster.add(current)
            stack.extend(adjacency[current] - cluster)
        remaining -= cluster
        clusters.append(sorted(cluster))
    return clusters


def _unique_matchings(
    component_ids: list[int],
    components: list[dict[str, Any]],
    bridges: list[dict[str, Any]],
    limit: int = 1025,
) -> tuple[list[list[dict[str, Any]]], bool]:
    endpoint_component = {
        vertex: component_id
        for component_id in component_ids
        for vertex in components[component_id]["endpoints"]
    }
    endpoints = sorted(endpoint_component)
    by_endpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for bridge in bridges:
        if bridge["left_component"] not in component_ids or bridge["right_component"] not in component_ids:
            continue
        by_endpoint[bridge["start_vertex_id"]].append(bridge)
        by_endpoint[bridge["end_vertex_id"]].append(bridge)
    solutions: list[list[dict[str, Any]]] = []
    truncated = False

    def search(unmatched: set[str], selected: list[dict[str, Any]]) -> None:
        nonlocal truncated
        if len(solutions) >= limit:
            truncated = True
            return
        if not unmatched:
            adjacency: dict[int, set[int]] = defaultdict(set)
            for bridge in selected:
                left, right = bridge["left_component"], bridge["right_component"]
                adjacency[left].add(right)
                adjacency[right].add(left)
            visited = set()
            stack = [component_ids[0]]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                stack.extend(adjacency[current] - visited)
            if visited == set(component_ids):
                solutions.append(sorted(selected, key=lambda item: item["id"]))
            return
        def available(item: str) -> int:
            return sum(
                (row["end_vertex_id"] if row["start_vertex_id"] == item else row["start_vertex_id"])
                in unmatched
                for row in by_endpoint[item]
            )

        endpoint = min(unmatched, key=lambda item: (available(item), item))
        for bridge in by_endpoint[endpoint]:
            other = bridge["end_vertex_id"] if bridge["start_vertex_id"] == endpoint else bridge["start_vertex_id"]
            if other not in unmatched:
                continue
            search(unmatched - {endpoint, other}, [*selected, bridge])

    search(set(endpoints), [])
    return solutions, truncated


def _edge_rows(components: Iterable[dict[str, Any]], bridges: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for component in components:
        for segment in component["segments"]:
            rows.append(
                {
                    "id": str(segment["id"]),
                    "start_vertex_id": str(segment["start_vertex_id"]),
                    "end_vertex_id": str(segment["end_vertex_id"]),
                    "start_display": list(segment["start_display"]),
                    "end_display": list(segment["end_display"]),
                    "derived": False,
                }
            )
    for bridge in bridges:
        rows.append(
            {
                "id": bridge["id"],
                "start_vertex_id": bridge["start_vertex_id"],
                "end_vertex_id": bridge["end_vertex_id"],
                "start_display": bridge["start_display"],
                "end_display": bridge["end_display"],
                "derived": True,
            }
        )
    return rows


def _ordered_boundary(edges: list[dict[str, Any]]) -> list[list[float]] | None:
    by_vertex: dict[str, list[int]] = defaultdict(list)
    for index, edge in enumerate(edges):
        by_vertex[edge["start_vertex_id"]].append(index)
        by_vertex[edge["end_vertex_id"]].append(index)
    if not edges or any(len(indices) != 2 for indices in by_vertex.values()):
        return None
    first_vertex = min(by_vertex)
    current_vertex = first_vertex
    unused = set(range(len(edges)))
    points: list[list[float]] = []
    while unused:
        candidates = sorted(index for index in by_vertex[current_vertex] if index in unused)
        if not candidates:
            return None
        index = candidates[0]
        edge = edges[index]
        unused.remove(index)
        forward = edge["start_vertex_id"] == current_vertex
        points.append(list(edge["start_display"] if forward else edge["end_display"]))
        current_vertex = edge["end_vertex_id"] if forward else edge["start_vertex_id"]
    if current_vertex != first_vertex or len(points) < 3:
        return None
    return points


def _orientation(a: list[float], b: list[float], c: list[float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _proper_intersection(a: list[float], b: list[float], c: list[float], d: list[float]) -> bool:
    return _orientation(a, b, c) * _orientation(a, b, d) < -1e-9 and _orientation(c, d, a) * _orientation(c, d, b) < -1e-9


def _simple_polygon(points: list[list[float]]) -> bool:
    count = len(points)
    signed_area = sum(
        points[index][0] * points[(index + 1) % count][1]
        - points[(index + 1) % count][0] * points[index][1]
        for index in range(count)
    ) / 2.0
    if abs(signed_area) <= 1e-8:
        return False
    for left in range(count):
        a, b = points[left], points[(left + 1) % count]
        for right in range(left + 1, count):
            if right in {left, (left + 1) % count} or (right + 1) % count == left:
                continue
            if _proper_intersection(a, b, points[right], points[(right + 1) % count]):
                return False
    return True


def _profile_record(
    page_number: int | None,
    scope_ref: str | None,
    components: list[dict[str, Any]],
    bridges: list[dict[str, Any]],
    metric: dict[str, Any],
) -> dict[str, Any] | None:
    edges = _edge_rows(components, bridges)
    boundary = _ordered_boundary(edges)
    if boundary is None or not _simple_polygon(boundary):
        return None
    source_segments = sorted(
        (segment for component in components for segment in component["segments"]),
        key=lambda item: str(item["id"]),
    )
    primitive_refs = sorted({str(item["primitive_ref"]) for item in source_segments})
    drawing_refs = sorted({str(item["drawing_ref"]) for item in source_segments})
    source_edge_refs = [str(item["id"]) for item in source_segments]
    bridge_refs = [str(item["id"]) for item in bridges]
    namespace = _record_namespace(page_number, scope_ref)
    identity_evidence = "\0".join(
        (namespace, *source_edge_refs, *bridge_refs, *metric["dimension_refs"])
    ).encode("utf-8")
    profile_id = (
        f"assembled_profile_candidate.{namespace}."
        f"evidence_{hashlib.sha256(identity_evidence).hexdigest()[:16]}"
    )
    bbox = _bbox_union(component["bbox_display"] for component in components)
    return {
        "record_type": "assembled_profile_candidate",
        "record_version": SCHEMA_VERSION,
        "id": profile_id,
        "page": page_number,
        "state": "resolved",
        "scope_ref": scope_ref,
        "quantity_eligible": False,
        "bbox_display": [round(value, 6) for value in bbox],
        "primitive_refs": primitive_refs,
        "drawing_refs": drawing_refs,
        "source_edge_refs": source_edge_refs,
        "derived_bridge_refs": bridge_refs,
        "derived_bridges": bridges,
        "ordered_boundary_display": [[round(value, 6) for value in point] for point in boundary],
        "closure": {
            "closed": True,
            "branch_free": True,
            "unique_completion": True,
            "scale_bounded": all(bridge["length_mm"] <= bridge["max_bridge_mm"] for bridge in bridges),
            "dimensionally_redundant": metric["passed"],
        },
        "dimension_certificate": metric,
        "evidence_refs": sorted({*source_edge_refs, *primitive_refs, *bridge_refs, *metric["dimension_refs"]}),
        "epistemic_state": "derived",
        "basis": "unique branch-free compatible-style completion over page-global native vertices",
    }


def assemble_multi_path_profiles(
    topology: Mapping[str, Any],
    dimensions: Iterable[Any],
    *,
    page_number: int | None = None,
    scope_ref: str | None = None,
    allowed_primitive_refs: Iterable[str] | None = None,
    max_bridge_mm: float = 25.0,
    max_relative_gap: float = 0.025,
    scale_tolerance_ratio: float = 0.025,
) -> dict[str, Any]:
    """Assemble uniquely closable multi-path profiles and preserve abstentions."""

    if max_bridge_mm <= 0 or not 0 < max_relative_gap <= 1 or not 0 < scale_tolerance_ratio <= 0.25:
        raise ValueError("profile assembly tolerances must be positive and bounded")
    metric = _metric_certificate(dimensions, scale_tolerance_ratio)
    base = {
        "schema_version": SCHEMA_VERSION,
        "layer": "multi_path_profile_assembly",
        "page": page_number,
        "scope_ref": scope_ref,
        "profiles": [],
        "abstentions": [],
        "contract": {
            "page_global_native_vertices_required": True,
            "compatible_styles_required": True,
            "derived_bridges_are_evidence": True,
            "physical_object_identity_established": False,
            "quantity_eligible": False,
            "schedule_values_used": False,
        },
    }
    if not metric["passed"]:
        base["status"] = "abstained"
        base["abstentions"].append(
            {
                "reason_code": "insufficient_dimensional_redundancy",
                "reason": metric["reason"],
                "dimension_certificate": metric,
                "evidence_refs": [],
            }
        )
        base["summary"] = {"resolved_count": 0, "abstention_count": 1}
        return base
    segments = _selected_segments(topology, allowed_primitive_refs)
    components = _native_components(segments, topology)
    points = _vertex_points(topology)
    namespace = _record_namespace(page_number, scope_ref)
    consumed: set[int] = set()

    for component_index, component in enumerate(components):
        if len(component["drawing_refs"]) < 2:
            continue
        if component["branch_vertex_ids"]:
            base["abstentions"].append(
                {
                    "reason_code": "branched_native_boundary",
                    "reason": "native path graph contains a vertex with degree greater than two",
                    "component_drawing_refs": component["drawing_refs"],
                    "branch_vertex_refs": component["branch_vertex_ids"],
                    "evidence_refs": sorted(str(item["id"]) for item in component["segments"]),
                }
            )
            consumed.add(component_index)
            continue
        if component["endpoints"]:
            continue
        record = _profile_record(page_number, scope_ref, [component], [], metric)
        if record is not None:
            base["profiles"].append(record)
            consumed.add(component_index)

    bridges = _bridge_candidates(
        components,
        points,
        float(metric["scale_points_per_mm"]),
        max_bridge_mm,
        max_relative_gap,
        namespace,
    )
    for component_ids in _component_clusters(len(components), bridges):
        if any(component_id in consumed for component_id in component_ids):
            continue
        selected = [components[component_id] for component_id in component_ids]
        drawing_refs = sorted({ref for component in selected for ref in component["drawing_refs"]})
        if len(drawing_refs) < 2:
            continue
        evidence_refs = sorted(
            str(segment["id"])
            for component in selected
            for segment in component["segments"]
        )
        branch_refs = sorted({ref for component in selected for ref in component["branch_vertex_ids"]})
        if branch_refs:
            base["abstentions"].append(
                {
                    "reason_code": "branched_native_boundary",
                    "reason": "native path graph contains a vertex with degree greater than two",
                    "component_drawing_refs": drawing_refs,
                    "branch_vertex_refs": branch_refs,
                    "evidence_refs": evidence_refs,
                }
            )
            continue
        if any(len(component["endpoints"]) != 2 for component in selected):
            base["abstentions"].append(
                {
                    "reason_code": "incomplete_profile_completion",
                    "reason": "not every compatible native component is one branch-free open path",
                    "component_drawing_refs": drawing_refs,
                    "evidence_refs": evidence_refs,
                }
            )
            continue
        if len(component_ids) < 2:
            base["abstentions"].append(
                {
                    "reason_code": "incomplete_profile_completion",
                    "reason": "the compatible multi-path boundary remains open within the metric gap bound",
                    "component_drawing_refs": drawing_refs,
                    "candidate_bridge_refs": [],
                    "evidence_refs": evidence_refs,
                }
            )
            continue
        if len(component_ids) > 12:
            base["abstentions"].append(
                {
                    "reason_code": "completion_search_not_bounded",
                    "reason": "more than twelve mutually bridgeable native components require profile scoping",
                    "component_drawing_refs": drawing_refs,
                    "evidence_refs": evidence_refs,
                }
            )
            continue
        matchings, truncated = _unique_matchings(component_ids, components, bridges)
        valid = []
        for matching in matchings:
            record = _profile_record(page_number, scope_ref, selected, matching, metric)
            if record is not None:
                valid.append(record)
        if len(valid) == 1 and not truncated:
            base["profiles"].append(valid[0])
        else:
            base["abstentions"].append(
                {
                    "reason_code": (
                        "completion_search_not_bounded"
                        if truncated
                        else "multiply_closable_profile"
                        if len(valid) > 1
                        else "incomplete_profile_completion"
                    ),
                    "reason": (
                        "completion enumeration exceeded the fail-closed search bound"
                        if truncated
                        else "more than one scale-bounded branch-free completion remains"
                        if len(valid) > 1
                        else "no scale-bounded branch-free completion closes a simple profile"
                    ),
                    "component_drawing_refs": drawing_refs,
                    "candidate_bridge_refs": sorted(
                        bridge["id"]
                        for bridge in bridges
                        if bridge["left_component"] in component_ids and bridge["right_component"] in component_ids
                    ),
                    "evidence_refs": evidence_refs,
                }
            )

    if not base["profiles"] and not base["abstentions"] and len({str(item.get("drawing_ref")) for item in segments}) >= 2:
        base["abstentions"].append(
            {
                "reason_code": "incomplete_profile_completion",
                "reason": "the scoped native paths have no unique compatible-style completion within the metric gap bound",
                "component_drawing_refs": sorted({str(item.get("drawing_ref")) for item in segments}),
                "candidate_bridge_refs": [bridge["id"] for bridge in bridges],
                "evidence_refs": sorted(str(item["id"]) for item in segments),
            }
        )

    base["profiles"] = sorted(base["profiles"], key=lambda item: item["id"])
    base["abstentions"] = sorted(
        base["abstentions"],
        key=lambda item: (
            str(item.get("reason_code")),
            tuple(map(str, item.get("evidence_refs", []) or [])),
        ),
    )
    base["status"] = "resolved" if base["profiles"] and not base["abstentions"] else "partial" if base["profiles"] else "abstained"
    base["summary"] = {
        "resolved_count": len(base["profiles"]),
        "abstention_count": len(base["abstentions"]),
        "source_segment_count": len(segments),
        "native_component_count": len(components),
        "candidate_bridge_count": len(bridges),
    }
    return base


def validate_multi_path_profile_assembly(payload: Mapping[str, Any]) -> list[str]:
    """Validate closure certificates without accepting physical identity."""

    errors = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    if payload.get("layer") != "multi_path_profile_assembly":
        errors.append("layer must be multi_path_profile_assembly")
    contract = payload.get("contract") or {}
    if contract.get("quantity_eligible") is not False:
        errors.append("assembly contract must remain quantity-ineligible")
    if contract.get("physical_object_identity_established") is not False:
        errors.append("assembly contract must not establish physical object identity")
    if contract.get("schedule_values_used") is not False:
        errors.append("assembly contract must exclude schedule values")
    for index, profile in enumerate(payload.get("profiles", []) or []):
        prefix = f"profiles[{index}]"
        if profile.get("record_type") != "assembled_profile_candidate":
            errors.append(f"{prefix}.record_type must be assembled_profile_candidate")
        if profile.get("record_version") != SCHEMA_VERSION:
            errors.append(f"{prefix}.record_version must be {SCHEMA_VERSION}")
        if profile.get("state") != "resolved":
            errors.append(f"{prefix}.state must be resolved")
        if profile.get("quantity_eligible") is not False:
            errors.append(f"{prefix} must remain quantity-ineligible")
        source_refs = set(map(str, profile.get("source_edge_refs", []) or []))
        bridge_refs = set(map(str, profile.get("derived_bridge_refs", []) or []))
        dimension_refs = set(map(str, (profile.get("dimension_certificate") or {}).get("dimension_refs", []) or []))
        evidence_refs = set(map(str, profile.get("evidence_refs", []) or []))
        if not source_refs:
            errors.append(f"{prefix}.source_edge_refs must preserve native edges")
        if not source_refs | bridge_refs | dimension_refs <= evidence_refs:
            errors.append(f"{prefix}.evidence_refs omit source, bridge, or dimension evidence")
        derived_ids = {str(item.get("id")) for item in profile.get("derived_bridges", []) or []}
        if derived_ids != bridge_refs:
            errors.append(f"{prefix}.derived_bridge_refs do not match derived_bridges")
        closure = profile.get("closure") or {}
        for field in (
            "closed",
            "branch_free",
            "unique_completion",
            "scale_bounded",
            "dimensionally_redundant",
        ):
            if closure.get(field) is not True:
                errors.append(f"{prefix}.closure.{field} must be true")
        certificate = profile.get("dimension_certificate") or {}
        if certificate.get("passed") is not True or int(certificate.get("independent_chain_count") or 0) < 2:
            errors.append(f"{prefix}.dimension_certificate requires two independent chains")
    return errors
