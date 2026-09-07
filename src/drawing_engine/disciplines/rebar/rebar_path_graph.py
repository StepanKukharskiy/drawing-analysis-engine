"""Drawing-neutral reinforcement path graph.

The graph deliberately treats native PDF curves as observations rather than
physical bars.  It records source fragments, endpoint connectivity, short-gap
continuation hypotheses, local metric scales, and mark-leader evidence.  Only
non-branching, uniquely connected components may become metric path
hypotheses; projected lengths are never promoted to cutting lengths here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import statistics
from typing import Any, Iterable

import fitz

from src.drawing_engine.core.cross_view_identity import extract_thin_segments, trace_leader
from src.drawing_engine.core.dimension_attachment import DimensionAttachment
from src.drawing_engine.core.leader_target_ranker import load_default_model, rank_candidates
from src.drawing_engine.disciplines.rebar.projected_bar_composite import build_composite_projected_bar_proposals


def _point_segment_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    dx, dy = bx - ax, by - ay
    denominator = dx * dx + dy * dy
    if denominator == 0:
        return math.dist(point, start)
    fraction = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denominator))
    return math.dist(point, (ax + fraction * dx, ay + fraction * dy))


def _polyline_distance(point: tuple[float, float], points: list[list[float]]) -> float:
    return min(
        (_point_segment_distance(point, tuple(left), tuple(right)) for left, right in zip(points, points[1:])),
        default=math.inf,
    )


def _polyline_length(points: list[list[float]]) -> float:
    return sum(math.dist(left, right) for left, right in zip(points, points[1:]))


def _cubic_points(item: tuple[Any, ...], steps: int = 10) -> list[list[float]]:
    p0, p1, p2, p3 = item[1], item[2], item[3], item[4]
    output = []
    for index in range(steps + 1):
        t = index / steps
        u = 1.0 - t
        output.append(
            [
                u**3 * p0.x + 3 * u * u * t * p1.x + 3 * u * t * t * p2.x + t**3 * p3.x,
                u**3 * p0.y + 3 * u * u * t * p1.y + 3 * u * t * t * p2.y + t**3 * p3.y,
            ]
        )
    return output


def _item_points(item: tuple[Any, ...]) -> tuple[str, list[list[float]]] | None:
    if item[0] == "l":
        return "line", [[float(item[1].x), float(item[1].y)], [float(item[2].x), float(item[2].y)]]
    if item[0] == "c":
        return "cubic", _cubic_points(item)
    return None


def _smallest_view(point: fitz.Point, views: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [view for view in views if point in fitz.Rect(view["bbox_display"])]
    return min(candidates, key=lambda item: fitz.Rect(item["bbox_display"]).get_area()) if candidates else None


def _containing_contours(point: fitz.Point, contours: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        item
        for item in contours
        if item.get("closed") and fitz.Rect(item["bbox_display"]).get_area() >= 100 and point in fitz.Rect(item["bbox_display"])
    ]
    return sorted(selected, key=lambda item: fitz.Rect(item["bbox_display"]).get_area())[:6]


def _angle(points: list[list[float]]) -> float:
    dx = points[-1][0] - points[0][0]
    dy = points[-1][1] - points[0][1]
    return math.degrees(math.atan2(dy, dx)) % 180.0


def _angle_difference(left: float, right: float) -> float:
    delta = abs(left - right) % 180.0
    return min(delta, 180.0 - delta)


def _style_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    width_left = float(left.get("width_pt") or 0.0)
    width_right = float(right.get("width_pt") or 0.0)
    if max(width_left, width_right) > 0 and abs(width_left - width_right) > max(0.35, 0.45 * max(width_left, width_right)):
        return False
    return left.get("stroke") == right.get("stroke") or left.get("stroke") is None or right.get("stroke") is None


def _metric_scales(
    dimensions: Iterable[DimensionAttachment],
    views: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: defaultdict[str, list[DimensionAttachment]] = defaultdict(list)
    for item in dimensions:
        if item.status != "accepted" or item.scale_points_per_mm <= 0:
            continue
        center = fitz.Point((item.text_bbox[0] + item.text_bbox[2]) / 2, (item.text_bbox[1] + item.text_bbox[3]) / 2)
        view = _smallest_view(center, views)
        if view is not None:
            grouped[view["id"]].append(item)
    output = {}
    for view_id, rows in grouped.items():
        values = [float(item.scale_points_per_mm) for item in rows]
        median = statistics.median(values)
        residuals = [abs(value - median) / median for value in values]
        consistent = [item for item, residual in zip(rows, residuals) if residual <= 0.04]
        if not consistent:
            continue
        selected_values = [float(item.scale_points_per_mm) for item in consistent]
        selected_median = statistics.median(selected_values)
        output[view_id] = {
            "points_per_mm": selected_median,
            "support_count": len(consistent),
            "state": "direct_consensus" if len(consistent) >= 2 else "direct_single_dimension",
            "max_relative_residual": round(max((abs(value - selected_median) / selected_median for value in selected_values), default=0.0), 6),
            "dimension_refs": [item.attachment_id for item in consistent],
        }
    return output


def _repetition_groups(fragments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: defaultdict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for fragment in fragments:
        if fragment["geometry"]["kind"] != "line":
            continue
        length = fragment["geometry"]["length_points"]
        if length <= 0:
            continue
        key = (
            fragment.get("view_id"),
            round(fragment["geometry"]["angle_deg"] / 5.0),
            round(math.log(max(length, 1.0), 1.25)),
            round(float(fragment["style"].get("width_pt") or 0.0) / 0.25),
        )
        buckets[key].append(fragment)
    groups = []
    for rows in buckets.values():
        if len(rows) < 3:
            continue
        group_id = f"repetition_group.{len(groups) + 1:04d}"
        for row in rows:
            row["repetition_group_ids"].append(group_id)
            row["candidate_score"] = min(1.0, row["candidate_score"] + 0.24)
        groups.append(
            {
                "id": group_id,
                "fragment_ids": [row["id"] for row in rows],
                "view_id": rows[0].get("view_id"),
                "count": len(rows),
                "orientation_deg": round(statistics.median(row["geometry"]["angle_deg"] for row in rows), 3),
                "median_length_points": round(statistics.median(row["geometry"]["length_points"] for row in rows), 3),
                "state": "observed_repetition",
            }
        )
    return groups


def _extract_fragments(
    page: fitz.Page,
    dimensions: Iterable[DimensionAttachment],
    views: list[dict[str, Any]],
    contours: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    excluded = {
        ref
        for item in dimensions
        if item.status == "accepted"
        for ref in (item.baseline.primitive_ref, *(line.primitive_ref for line in item.extension_lines), *item.terminal_refs)
    }
    contour_by_ref = {
        ref: contour
        for contour in contours
        for ref in contour.get("primitive_refs", [])
    }
    page_diagonal = math.hypot(page.rect.width, page.rect.height)
    fragments = []
    for drawing_index, drawing in enumerate(page.get_drawings()):
        drawing_ref = f"drawing[{drawing_index}]"
        for item_index, item in enumerate(drawing.get("items", [])):
            parsed = _item_points(item)
            if parsed is None:
                continue
            kind, points = parsed
            length = _polyline_length(points)
            if length < max(2.5, 0.0012 * page_diagonal):
                continue
            primitive_ref = f"{drawing_ref}.item[{item_index}]"
            if primitive_ref in excluded:
                continue
            midpoint = fitz.Point(
                sum(point[0] for point in points) / len(points),
                sum(point[1] for point in points) / len(points),
            )
            view = _smallest_view(midpoint, views)
            if view is not None and view["role_hypothesis"] == "table_or_grid_candidate":
                continue
            hosts = _containing_contours(midpoint, contours)
            score = 0.10
            role = None if view is None else view["role_hypothesis"]
            if role == "reinforcement_view_candidate":
                score += 0.30
            elif role in {"section_view_candidate", "plan_view_candidate"}:
                score += 0.19
            elif role == "drawing_view_candidate":
                score += 0.12
            width = float(drawing.get("width") or 0.0)
            if width >= 1.0:
                score += 0.20
            elif width >= 0.50:
                score += 0.10
            if drawing.get("fill") is not None:
                score += 0.08
            if hosts:
                score += 0.15
            if length >= 0.08 * min(
                fitz.Rect(view["bbox_display"]).width if view else page.rect.width,
                fitz.Rect(view["bbox_display"]).height if view else page.rect.height,
            ):
                score += 0.08
            if drawing_ref in contour_by_ref:
                score -= 0.35
            fragments.append(
                {
                    "id": f"path_fragment.{len(fragments) + 1:06d}",
                    "primitive_ref": primitive_ref,
                    "source_path_ref": drawing_ref,
                    "source_item_index": item_index,
                    "view_id": None if view is None else view["id"],
                    "view_role": role,
                    "host_contour_ids": [host["id"] for host in hosts],
                    "geometry": {
                        "kind": kind,
                        "points_display": [[round(value, 4) for value in point] for point in points],
                        "length_points": round(length, 4),
                        "angle_deg": round(_angle(points), 4),
                    },
                    "style": {
                        "width_pt": width,
                        "stroke": drawing.get("color"),
                        "fill": drawing.get("fill"),
                        "dashes": drawing.get("dashes"),
                    },
                    "candidate_score": max(0.0, round(score, 4)),
                    "repetition_group_ids": [],
                    "mark_hypotheses": [],
                    "state": "observed",
                }
            )
    deduplicated: dict[tuple[Any, ...], dict[str, Any]] = {}
    for fragment in fragments:
        points = fragment["geometry"]["points_display"]
        forward = tuple(round(value, 2) for point in points for value in point)
        reverse = tuple(round(value, 2) for point in reversed(points) for value in point)
        signature = (
            fragment.get("view_id"),
            fragment["geometry"]["kind"],
            min(forward, reverse),
            round(float(fragment["style"].get("width_pt") or 0.0), 2),
        )
        existing = deduplicated.get(signature)
        if existing is None:
            fragment["duplicate_primitive_refs"] = []
            deduplicated[signature] = fragment
        else:
            existing["duplicate_primitive_refs"].append(fragment["primitive_ref"])
            existing["candidate_score"] = max(existing["candidate_score"], fragment["candidate_score"])
    fragments = list(deduplicated.values())
    repetition = _repetition_groups(fragments)
    fragments = [item for item in fragments if item["candidate_score"] >= 0.30 or item["repetition_group_ids"]]
    fragments.sort(key=lambda item: item["primitive_ref"])
    for index, fragment in enumerate(fragments, start=1):
        fragment["id"] = f"path_fragment.{index:06d}"
    surviving = {item["id"] for item in fragments}
    # IDs changed after filtering, so rebuild repetition membership from refs.
    by_ref = {item["primitive_ref"]: item for item in fragments}
    filtered_groups = []
    for group in repetition:
        refs = [row["primitive_ref"] for row in fragments if group["id"] in row["repetition_group_ids"]]
        ids = [by_ref[ref]["id"] for ref in refs if ref in by_ref]
        if len(ids) >= 3:
            group["fragment_ids"] = ids
            filtered_groups.append(group)
    return fragments, filtered_groups


def _endpoint_records(fragments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for fragment in fragments:
        points = fragment["geometry"]["points_display"]
        for endpoint_index, point in enumerate((points[0], points[-1])):
            records.append(
                {
                    "fragment_id": fragment["id"],
                    "endpoint_index": endpoint_index,
                    "point": point,
                    "view_id": fragment.get("view_id"),
                }
            )
    return records


def _endpoint_clusters(endpoints: list[dict[str, Any]], tolerance: float) -> list[list[dict[str, Any]]]:
    cell = tolerance
    buckets: defaultdict[tuple[int, int, str | None], list[int]] = defaultdict(list)
    clusters: list[list[dict[str, Any]]] = []
    centers: list[tuple[float, float]] = []
    for endpoint in endpoints:
        x, y = endpoint["point"]
        key_x, key_y = round(x / cell), round(y / cell)
        matches = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                matches.extend(buckets.get((key_x + dx, key_y + dy, endpoint["view_id"]), []))
        match = next((index for index in matches if math.dist((x, y), centers[index]) <= tolerance), None)
        if match is None:
            match = len(clusters)
            clusters.append([])
            centers.append((x, y))
            buckets[(key_x, key_y, endpoint["view_id"])].append(match)
        clusters[match].append(endpoint)
        centers[match] = (
            sum(row["point"][0] for row in clusters[match]) / len(clusters[match]),
            sum(row["point"][1] for row in clusters[match]) / len(clusters[match]),
        )
    return clusters


def _connections(fragments: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {item["id"]: item for item in fragments}
    endpoints = _endpoint_records(fragments)
    junctions = []
    connections = []
    connected_endpoints: set[tuple[str, int]] = set()
    for cluster in _endpoint_clusters(endpoints, 1.8):
        unique = {(row["fragment_id"], row["endpoint_index"]): row for row in cluster}
        if len(unique) < 2:
            continue
        rows = list(unique.values())
        center = [statistics.mean(row["point"][0] for row in rows), statistics.mean(row["point"][1] for row in rows)]
        junction_id = f"path_junction.{len(junctions) + 1:06d}"
        junctions.append(
            {
                "id": junction_id,
                "point_display": [round(center[0], 4), round(center[1], 4)],
                "endpoint_refs": [f"{row['fragment_id']}.endpoint[{row['endpoint_index']}]" for row in rows],
                "degree": len(rows),
                "state": "observed",
            }
        )
        for left_index, left in enumerate(rows):
            for right in rows[left_index + 1 :]:
                left_fragment, right_fragment = by_id[left["fragment_id"]], by_id[right["fragment_id"]]
                compatible = _style_compatible(left_fragment["style"], right_fragment["style"])
                same_source = left_fragment["source_path_ref"] == right_fragment["source_path_ref"]
                unique_pair = len(rows) == 2
                accepted = compatible and (unique_pair or same_source)
                relation = "source_path_continuity" if same_source else "unique_endpoint_connection" if unique_pair else "ambiguous_junction"
                connections.append(
                    {
                        "id": f"path_connection.{len(connections) + 1:06d}",
                        "from_fragment_id": left["fragment_id"],
                        "to_fragment_id": right["fragment_id"],
                        "junction_id": junction_id,
                        "relation": relation,
                        "gap_points": round(math.dist(left["point"], right["point"]), 4),
                        "style_compatible": compatible,
                        "state": "accepted" if accepted else "ambiguous",
                    }
                )
                if accepted:
                    connected_endpoints.update(((left["fragment_id"], left["endpoint_index"]), (right["fragment_id"], right["endpoint_index"])))

    unmatched = [row for row in endpoints if (row["fragment_id"], row["endpoint_index"]) not in connected_endpoints]
    candidates: defaultdict[tuple[str, int], list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    gap_limit = 8.0
    endpoint_grid: defaultdict[tuple[int, int, str | None], list[int]] = defaultdict(list)
    for endpoint_index, endpoint in enumerate(unmatched):
        x, y = endpoint["point"]
        endpoint_grid[(math.floor(x / gap_limit), math.floor(y / gap_limit), endpoint["view_id"])].append(endpoint_index)
    for left_index, left in enumerate(unmatched):
        left_fragment = by_id[left["fragment_id"]]
        x, y = left["point"]
        cell_x, cell_y = math.floor(x / gap_limit), math.floor(y / gap_limit)
        neighbor_indexes = {
            candidate_index
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for candidate_index in endpoint_grid.get((cell_x + dx, cell_y + dy, left["view_id"]), [])
            if candidate_index > left_index
        }
        for right_index in sorted(neighbor_indexes):
            right = unmatched[right_index]
            if left["view_id"] != right["view_id"] or left["fragment_id"] == right["fragment_id"]:
                continue
            gap = math.dist(left["point"], right["point"])
            if not 1.8 < gap <= gap_limit:
                continue
            right_fragment = by_id[right["fragment_id"]]
            if not _style_compatible(left_fragment["style"], right_fragment["style"]):
                continue
            if _angle_difference(left_fragment["geometry"]["angle_deg"], right_fragment["geometry"]["angle_deg"]) > 5.0:
                continue
            candidates[(left["fragment_id"], left["endpoint_index"])].append((gap, right))
            candidates[(right["fragment_id"], right["endpoint_index"])].append((gap, left))
    used: set[tuple[str, int]] = set()
    for endpoint_key, ranked in sorted(candidates.items()):
        if endpoint_key in used:
            continue
        ranked.sort(key=lambda item: item[0])
        gap, other = ranked[0]
        other_key = (other["fragment_id"], other["endpoint_index"])
        reverse = sorted(candidates.get(other_key, []), key=lambda item: item[0])
        reciprocal = reverse and (reverse[0][1]["fragment_id"], reverse[0][1]["endpoint_index"]) == endpoint_key
        margin = (ranked[1][0] - gap) if len(ranked) > 1 else math.inf
        reverse_margin = (reverse[1][0] - gap) if len(reverse) > 1 else math.inf
        accepted = reciprocal and min(margin, reverse_margin) >= 1.0
        left_fragment_id, left_endpoint_index = endpoint_key
        connections.append(
            {
                "id": f"path_connection.{len(connections) + 1:06d}",
                "from_fragment_id": left_fragment_id,
                "to_fragment_id": other["fragment_id"],
                "junction_id": None,
                "relation": "short_collinear_occlusion_bridge",
                "endpoint_refs": [
                    f"{left_fragment_id}.endpoint[{left_endpoint_index}]",
                    f"{other['fragment_id']}.endpoint[{other['endpoint_index']}]",
                ],
                "gap_points": round(gap, 4),
                "ambiguity_margin_points": None if math.isinf(min(margin, reverse_margin)) else round(min(margin, reverse_margin), 4),
                "style_compatible": True,
                "state": "accepted" if accepted else "ambiguous",
            }
        )
        if accepted:
            used.update((endpoint_key, other_key))
    return junctions, connections


def _components(
    fragments: list[dict[str, Any]],
    connections: list[dict[str, Any]],
    scales: dict[str, dict[str, Any]],
    views: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    view_by_id = {item["id"]: item for item in views}
    parent = {item["id"]: item["id"] for item in fragments}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    by_id = {item["id"]: item for item in fragments}
    accepted = []
    for item in connections:
        if item["state"] != "accepted":
            continue
        left = by_id[item["from_fragment_id"]]
        right = by_id[item["to_fragment_id"]]
        left_marks = set(left.get("mark_hypotheses", []))
        right_marks = set(right.get("mark_hypotheses", []))
        same_source = left.get("source_path_ref") == right.get("source_path_ref")
        # A leader-bound mark must not leak through a projected crossing into
        # an unrelated path.  Preserve native source-path continuity and
        # explicitly shared identities; all other marked/unmarked junctions
        # remain observations rather than physical centerline connections.
        if (left_marks or right_marks) and not same_source and not (left_marks & right_marks):
            item["component_connection_state"] = "rejected_identity_bridge"
            item["component_connection_reason"] = "mark identity cannot propagate through an unmarked or differently marked projection crossing"
            continue
        item["component_connection_state"] = "accepted"
        accepted.append(item)
    for connection in accepted:
        union(connection["from_fragment_id"], connection["to_fragment_id"])
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for fragment in fragments:
        grouped[find(fragment["id"])].append(fragment)
    output = []
    for rows in grouped.values():
        ids = {item["id"] for item in rows}
        edges = [item for item in accepted if item["from_fragment_id"] in ids and item["to_fragment_id"] in ids]
        degree = {item: 0 for item in ids}
        for edge in edges:
            degree[edge["from_fragment_id"]] += 1
            degree[edge["to_fragment_id"]] += 1
        if any(value > 2 for value in degree.values()):
            topology = "branched"
        elif len(rows) == 1:
            topology = "single_fragment"
        elif degree and all(value == 2 for value in degree.values()):
            topology = "closed_cycle"
        elif sum(value == 1 for value in degree.values()) == 2 and all(value <= 2 for value in degree.values()):
            topology = "open_chain"
        else:
            topology = "disconnected_or_ambiguous"
        view_ids = sorted({item["view_id"] for item in rows if item.get("view_id")})
        view_id = view_ids[0] if len(view_ids) == 1 else None
        scale = scales.get(view_id or "")
        visible_points = sum(item["geometry"]["length_points"] for item in rows)
        bridge_points = sum(item["gap_points"] for item in edges if item["relation"] == "short_collinear_occlusion_bridge")
        total_points = visible_points + bridge_points
        mark_hypotheses = sorted({mark for item in rows for mark in item.get("mark_hypotheses", [])})
        points = [point for item in rows for point in item["geometry"]["points_display"]]
        x0 = min(point[0] for point in points)
        y0 = min(point[1] for point in points)
        x1 = max(point[0] for point in points)
        y1 = max(point[1] for point in points)
        width = max(x1 - x0, 0.0)
        height = max(y1 - y0, 0.0)
        major = max(width, height)
        minor = min(width, height)
        aspect = major / max(minor, 0.5)
        roles = sorted({item.get("view_role") for item in rows if item.get("view_role")})
        view = view_by_id.get(view_id or "")
        if topology == "closed_cycle" and major <= 20.0 and "section_view_candidate" in roles:
            dimensionality = "axis_end_projection_0d"
        elif topology in {"single_fragment", "open_chain"} and aspect >= 6.0:
            dimensionality = "axis_projection_1d"
        elif topology == "closed_cycle" or (width >= 2.5 and height >= 2.5 and aspect < 6.0):
            dimensionality = "planar_path_projection_2d"
        else:
            dimensionality = "ambiguous_projection"
        normalized_center = None
        if view is not None:
            view_box = fitz.Rect(view["bbox_display"])
            if view_box.width > 0 and view_box.height > 0:
                normalized_center = [
                    round(((x0 + x1) / 2 - view_box.x0) / view_box.width, 4),
                    round(((y0 + y1) / 2 - view_box.y0) / view_box.height, 4),
                ]
        component_id = f"path_component.{len(output) + 1:06d}"
        output.append(
            {
                "id": component_id,
                "fragment_ids": sorted(ids),
                "connection_ids": [item["id"] for item in edges],
                "view_ids": view_ids,
                "host_contour_ids": sorted({host for item in rows for host in item["host_contour_ids"]}),
                "repetition_group_ids": sorted({group for item in rows for group in item["repetition_group_ids"]}),
                "mark_hypotheses": mark_hypotheses,
                "topology": topology,
                "bbox_display": [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)],
                "projection_dimensionality": {
                    "value": dimensionality,
                    "state": "inferred" if dimensionality != "ambiguous_projection" else "unknown",
                    "basis": "native path topology, component aspect ratio, and view role",
                    "aspect_ratio": round(aspect, 4),
                    "view_role_hypotheses": roles,
                    "normalized_center_in_view": normalized_center,
                },
                "style_signature": {
                    "median_width_pt": round(statistics.median(float(item["style"].get("width_pt") or 0.0) for item in rows), 4),
                    "stroke_values": sorted({str(item["style"].get("stroke")) for item in rows}),
                },
                "candidate_score": round(max(item.get("candidate_score", 0.0) for item in rows), 4),
                "projected_length": {
                    "visible_points": round(visible_points, 4),
                    "bridged_gap_points": round(bridge_points, 4),
                    "total_points": round(total_points, 4),
                    "value_mm": None if scale is None else round(total_points / scale["points_per_mm"], 3),
                    "status": "metric_observation" if scale is not None else "display_space_observation",
                    "metric_scale": scale,
                },
                "physical_path_state": (
                    "candidate_nonbranching_projection"
                    if topology in {"single_fragment", "open_chain", "closed_cycle"}
                    else "unresolved_topology"
                ),
                "installed_centerline": {
                    "value_mm": None,
                    "status": "unresolved",
                    "reason": "cross-view identity and projection dimensionality are not yet uniquely established",
                },
                "fabrication_cutting": {
                    "value_mm": None,
                    "status": "unresolved",
                    "reason": "bend, hook, and fabrication convention are outside the observation graph",
                },
            }
        )
    return output


def enrich_projection_identities(path_graph: dict[str, Any]) -> None:
    """Attach conservative cross-view identity families to an existing graph.

    Exact group provenance is accepted.  A unique same-mark relation is still
    only a review candidate because one physical mark may have many instances.
    Unmarked geometry produces reciprocal-best candidates, never accepted
    identity, so similar projections cannot silently inflate quantities.
    """

    fragments = {item["id"]: item for item in path_graph.get("fragments", [])}
    components = path_graph.get("components", [])
    for component in components:
        component["resolved_group_ids"] = sorted(
            {
                group_id
                for fragment_id in component.get("fragment_ids", [])
                for group_id in fragments.get(fragment_id, {}).get("resolved_group_ids", [])
            }
        )

    identities: list[dict[str, Any]] = []
    family_by_group: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for component in components:
        for group_id in component.get("resolved_group_ids", []):
            family_by_group[group_id].append(component)
    for group_id, rows in sorted(family_by_group.items()):
        views = sorted({view for item in rows for view in item.get("view_ids", [])})
        identities.append(
            {
                "id": f"projection_identity.{len(identities) + 1:05d}",
                "component_ids": sorted(item["id"] for item in rows),
                "view_ids": views,
                "group_id": group_id,
                "mark": next((mark for item in rows for mark in item.get("mark_hypotheses", [])), None),
                "projection_dimensionalities": sorted(
                    {item["projection_dimensionality"]["value"] for item in rows}
                ),
                "state": "accepted" if len(views) >= 2 else "within_view_provenance",
                "score": 1.0,
                "basis": "exact source primitive provenance to one independently resolved group",
            }
        )

    already_linked = {component_id for item in identities for component_id in item["component_ids"]}
    candidates = [
        item
        for item in components
        if item["id"] not in already_linked
        and len(item.get("view_ids", [])) == 1
        and item.get("physical_path_state") == "candidate_nonbranching_projection"
        and (
            item.get("mark_hypotheses")
            or item.get("repetition_group_ids")
            or item.get("candidate_score", 0.0) >= 0.62
        )
    ]
    by_view: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_view[item["view_ids"][0]].append(item)
    candidates = [
        item
        for view_id in sorted(by_view)
        for item in sorted(by_view[view_id], key=lambda row: row.get("candidate_score", 0.0), reverse=True)[:20]
    ]
    candidate_by_id = {item["id"]: item for item in candidates}
    scope_required = bool(path_graph.get("object_instance_scopes") or path_graph.get("coordinate_scopes"))
    coordinate_scopes = {
        str(item["coordinate_scope_id"]): item
        for item in path_graph.get("coordinate_scopes", [])
    }

    def actual_reprojection(
        left: dict[str, Any],
        right: dict[str, Any],
        shared_scope_ids: set[str],
    ) -> dict[str, Any]:
        left_view, right_view = left["view_ids"][0], right["view_ids"][0]
        for scope_id in sorted(shared_scope_ids):
            scope = coordinate_scopes.get(scope_id, {})
            scales = scope.get("view_scales_points_per_mm", {})
            for record in scope.get("contour_correspondences", []) or []:
                if record.get("state") != "accepted" or {
                    str(record.get("parent_view_id")),
                    str(record.get("child_view_id")),
                } != {left_view, right_view}:
                    continue
                transform = record.get("selected", {}).get("signed_transform", {})
                if transform.get("state") != "resolved":
                    return {
                        "status": "unknown",
                        "coordinate_scope_id": scope_id,
                        "contour_correspondence_id": record.get("id"),
                        "reason": "contour path matches, but mirrored view sign remains unresolved",
                    }
                parent_view = str(record["parent_view_id"])
                child_view = str(record["child_view_id"])
                parent_component = left if left_view == parent_view else right
                child_component = left if left_view == child_view else right
                parent_scale, child_scale = scales.get(parent_view), scales.get(child_view)
                if not parent_scale or not child_scale:
                    continue
                parent_index = 0 if record["parent_display_orientation"] == "horizontal" else 1
                child_index = 0 if record["child_display_orientation"] == "horizontal" else 1
                parent_box, child_box = parent_component["bbox_display"], child_component["bbox_display"]
                parent_coordinate = (float(parent_box[parent_index]) + float(parent_box[parent_index + 2])) / (2.0 * float(parent_scale))
                child_coordinate = (float(child_box[child_index]) + float(child_box[child_index + 2])) / (2.0 * float(child_scale))
                mapping = transform["child_coordinate_to_parent"]
                reprojected = float(mapping["sign"]) * child_coordinate + float(mapping["offset_mm"])
                residual = abs(parent_coordinate - reprojected)
                tolerance = max(2.0, float(record.get("selected", {}).get("tolerance_mm", 2.0)))
                return {
                    "status": "pass" if residual <= tolerance else "fail",
                    "coordinate_scope_id": scope_id,
                    "contour_correspondence_id": record.get("id"),
                    "shared_object_axis": record.get("shared_object_axis"),
                    "parent_coordinate_mm": round(parent_coordinate, 6),
                    "reprojected_child_coordinate_mm": round(reprojected, 6),
                    "residual_mm": round(residual, 6),
                    "tolerance_mm": round(tolerance, 6),
                    "reason": "component centers agree under the unique native-contour transform" if residual <= tolerance else "component centers disagree under the unique native-contour transform",
                }
        return {
            "status": "unknown",
            "reason": "no unique signed native-contour transform connects these views",
        }

    def pair_score(left: dict[str, Any], right: dict[str, Any]) -> tuple[float, list[str], dict[str, Any]]:
        if left["view_ids"][0] == right["view_ids"][0]:
            return 0.0, [], {"status": "not_applicable"}
        left_instances = set(left.get("object_instance_ids", []))
        right_instances = set(right.get("object_instance_ids", []))
        left_coordinates = set(left.get("coordinate_scope_ids", []))
        right_coordinates = set(right.get("coordinate_scope_ids", []))
        shared_instances = left_instances.intersection(right_instances)
        shared_coordinates = left_coordinates.intersection(right_coordinates)
        if scope_required and not shared_instances and not shared_coordinates:
            return 0.0, [], {"status": "unknown", "reason": "components do not share an object or coordinate scope"}
        reprojection = actual_reprojection(left, right, shared_coordinates)
        if reprojection["status"] == "fail":
            return 0.0, [], reprojection
        score = 0.0
        factors = []
        if shared_instances:
            score += 0.18
            factors.append("same procedural object-instance scope")
        if shared_coordinates:
            score += 0.18
            factors.append("same reprojection-validated coordinate scope")
        if reprojection["status"] == "pass":
            score += 0.18
            factors.append("native path center passes the signed contour transform")
        common_marks = set(left.get("mark_hypotheses", [])) & set(right.get("mark_hypotheses", []))
        if common_marks:
            score += 0.50
            factors.append("same evidence-backed mark token")
        left_dim = left["projection_dimensionality"]["value"]
        right_dim = right["projection_dimensionality"]["value"]
        if {left_dim, right_dim} == {"axis_projection_1d", "axis_end_projection_0d"}:
            score += 0.25
            factors.append("complementary axis and end projections")
        elif left_dim == right_dim and left_dim != "ambiguous_projection":
            score += 0.12
            factors.append("compatible projection dimensionality")
        if left.get("topology") == right.get("topology"):
            score += 0.10
            factors.append("matching path topology")
        if left.get("repetition_group_ids") and right.get("repetition_group_ids"):
            score += 0.10
            factors.append("repeated in both views")
        left_width = left.get("style_signature", {}).get("median_width_pt")
        right_width = right.get("style_signature", {}).get("median_width_pt")
        if left_width is not None and right_width is not None and abs(left_width - right_width) <= max(0.18, 0.25 * max(left_width, right_width, 0.1)):
            score += 0.08
            factors.append("compatible native line style")
        left_center = left["projection_dimensionality"].get("normalized_center_in_view")
        right_center = right["projection_dimensionality"].get("normalized_center_in_view")
        if left_center and right_center and math.dist(left_center, right_center) <= 0.18:
            score += 0.08
            factors.append("compatible normalized host position")
        left_length = left.get("projected_length", {}).get("value_mm")
        right_length = right.get("projected_length", {}).get("value_mm")
        if left_dim == right_dim and left_length and right_length:
            ratio = min(left_length, right_length) / max(left_length, right_length)
            if ratio >= 0.94:
                score += 0.10
                factors.append("matching metric projected extent")
        return min(score, 0.99), factors, reprojection

    ranked: defaultdict[str, list[tuple[float, dict[str, Any], list[str], dict[str, Any]]]] = defaultdict(list)
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            score, factors, reprojection = pair_score(left, right)
            if score >= 0.45:
                ranked[left["id"]].append((score, right, factors, reprojection))
                ranked[right["id"]].append((score, left, factors, reprojection))
    emitted: set[tuple[str, str]] = set()
    for component_id, rows in sorted(ranked.items()):
        rows.sort(key=lambda item: (-item[0], item[1]["id"]))
        score, other, factors, reprojection = rows[0]
        reverse = sorted(ranked.get(other["id"], []), key=lambda item: (-item[0], item[1]["id"]))
        if not reverse or reverse[0][1]["id"] != component_id:
            continue
        pair = tuple(sorted((component_id, other["id"])))
        if pair in emitted:
            continue
        emitted.add(pair)
        left = candidate_by_id[component_id]
        marks = sorted(set(left.get("mark_hypotheses", [])) & set(other.get("mark_hypotheses", [])))
        margin = score - rows[1][0] if len(rows) > 1 else score
        identities.append(
            {
                "id": f"projection_identity.{len(identities) + 1:05d}",
                "component_ids": list(pair),
                "view_ids": sorted({left["view_ids"][0], other["view_ids"][0]}),
                "group_id": None,
                "mark": marks[0] if len(marks) == 1 else None,
                "projection_dimensionalities": sorted(
                    {left["projection_dimensionality"]["value"], other["projection_dimensionality"]["value"]}
                ),
                "state": "review_candidate",
                "score": round(score, 3),
                "ambiguity_margin": round(margin, 3),
                "basis": factors,
                "actual_path_reprojection": reprojection,
                "reason_not_accepted": "similarity is not unique physical-instance identity",
                "object_instance_ids": sorted(
                    set(left.get("object_instance_ids", [])).intersection(other.get("object_instance_ids", []))
                ),
                "coordinate_scope_ids": sorted(
                    set(left.get("coordinate_scope_ids", [])).intersection(other.get("coordinate_scope_ids", []))
                ),
            }
        )
    path_graph["cross_view_projection_identities"] = identities
    path_graph.setdefault("summary", {})["projection_dimensionality_counts"] = dict(
        sorted(Counter(item["projection_dimensionality"]["value"] for item in components).items())
    )
    path_graph["summary"]["accepted_cross_view_identity_count"] = sum(
        item["state"] == "accepted" for item in identities
    )
    path_graph["summary"]["review_cross_view_identity_count"] = sum(
        item["state"] == "review_candidate" for item in identities
    )


def _attach_mark_hypotheses(
    page: fitz.Page,
    fragments: list[dict[str, Any]],
    text_roles: list[dict[str, Any]],
    components: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    traces = []
    learned_model = load_default_model()
    thin = extract_thin_segments(page)
    component_by_fragment = {
        fragment_id: component
        for component in components
        for fragment_id in component.get("fragment_ids", [])
    }
    for role in text_roles:
        if role.get("resolved_role") != "identifier_candidate":
            continue
        trace = trace_leader(fitz.Rect(role["bbox_display"]), thin, search_radius=150, max_hops=9)
        if not trace["terminals"]:
            continue
        leader_refs = {item["drawing_ref"] for item in trace.get("segments", [])}
        ranked = []
        for fragment in fragments:
            if fragment.get("primitive_ref") in leader_refs:
                continue
            distance = min(
                (_polyline_distance(tuple(terminal), fragment["geometry"]["points_display"]) for terminal in trace["terminals"]),
                default=math.inf,
            )
            if distance <= 12.0:
                ranked.append((distance, fragment))
        ranked.sort(key=lambda item: item[0])
        if not ranked:
            continue
        ranked_components: dict[str, tuple[float, dict[str, Any], dict[str, Any]]] = {}
        for distance, fragment in ranked:
            component = component_by_fragment.get(fragment["id"])
            component_id = component["id"] if component is not None else fragment["id"]
            if component_id not in ranked_components or distance < ranked_components[component_id][0]:
                ranked_components[component_id] = (distance, fragment, component or {})
        component_rows = sorted(ranked_components.values(), key=lambda item: (item[0], item[1]["id"]))
        margin = component_rows[1][0] - component_rows[0][0] if len(component_rows) > 1 else math.inf
        accepted = component_rows[0][0] <= 4.0 and margin >= 2.0
        learned_ranking = None
        if learned_model is not None:
            terminal = min(
                trace["terminals"],
                key=lambda point: min(
                    _polyline_distance(tuple(point), fragment["geometry"]["points_display"])
                    for _, fragment in ranked
                ),
            )
            learned_rows = rank_candidates(
                learned_model,
                tuple(terminal),
                [fragment for _, fragment in ranked],
                view_bbox=list(page.rect),
            )
            learned_ranking = {
                "model_sha256": learned_model["model_sha256"],
                "ranking_only": True,
                "winner_fragment_id": learned_rows[0]["candidate_id"] if learned_rows else None,
                "winner_probability": learned_rows[0]["probability"] if learned_rows else None,
                "probability_margin": (
                    round(learned_rows[0]["probability"] - learned_rows[1]["probability"], 6)
                    if len(learned_rows) > 1
                    else learned_rows[0]["probability"] if learned_rows else None
                ),
                "deterministic_distance_winner_preserved": ranked[0][1]["id"],
            }
        hypothesis = {
            "id": f"path_mark_hypothesis.{len(traces) + 1:05d}",
            "token": role["text"],
            "text_role_id": role["id"],
            "bbox_display": role.get("bbox_display"),
            "source_role_basis": role.get("basis"),
            "fragment_id": component_rows[0][1]["id"],
            "component_id": component_rows[0][2].get("id"),
            "terminal_distance_points": round(component_rows[0][0], 3),
            "ambiguity_margin_points": None if math.isinf(margin) else round(margin, 3),
            "leader_trace": trace,
            "state": "accepted" if accepted else "ambiguous",
            "uniqueness_basis": "unique connected path component rather than individual parallel outline fragment",
            "semantic_value": role.get("semantic_value"),
            "count_spacing_observation": role.get("paired_count_spacing"),
            "learned_ranking": learned_ranking,
        }
        traces.append(hypothesis)
        if accepted:
            component_rows[0][1]["mark_hypotheses"].append(role["text"])
            component = component_rows[0][2]
            component["mark_hypotheses"] = sorted(
                set(component.get("mark_hypotheses", [])) | {role["text"]},
                key=lambda value: (len(value), value),
            )
    return traces


def _apply_section_mark_hypotheses(
    fragments: list[dict[str, Any]],
    section_observations: dict[str, Any] | None,
) -> None:
    """Make exact section leader terminals authoritative for source paths."""

    marks_by_source: defaultdict[str, set[str]] = defaultdict(set)
    for section in (section_observations or {}).get("sections", []):
        for identity in section.get("leader_identities", []):
            for primitive_ref in identity.get("candidate_refs", []):
                marks_by_source[primitive_ref].add(str(identity["mark_text"]))
    for fragment in fragments:
        marks = marks_by_source.get(fragment.get("source_path_ref", ""))
        if not marks:
            continue
        fragment["mark_hypotheses"] = sorted(marks, key=lambda value: (len(value), value))
        fragment["mark_assignment_basis"] = "exclusive section mark plus first connected heavy-geometry leader contact"


def build_rebar_path_graph(
    page: fitz.Page,
    dimensions: Iterable[DimensionAttachment],
    views: list[dict[str, Any]],
    contours: list[dict[str, Any]],
    text_roles: list[dict[str, Any]],
    section_observations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the generic path graph before any solid or object solver gate."""

    dimension_rows = tuple(dimensions)
    fragments, repetition = _extract_fragments(page, dimension_rows, views, contours)
    _apply_section_mark_hypotheses(fragments, section_observations)
    junctions, connections = _connections(fragments)
    scales = _metric_scales(dimension_rows, views)
    components = _components(fragments, connections, scales, views)
    mark_hypotheses = _attach_mark_hypotheses(page, fragments, text_roles, components)
    composite_projected_bars = build_composite_projected_bar_proposals(fragments, components)
    metric_components = [item for item in components if item["projected_length"]["value_mm"] is not None]
    nonbranching = [item for item in components if item["physical_path_state"] == "candidate_nonbranching_projection"]
    projected_by_view: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for component in metric_components:
        if len(component["view_ids"]) == 1:
            projected_by_view[component["view_ids"][0]].append(component)
    graph = {
        "schema_version": "0.1.0",
        "layer": "rebar_physical_path_graph",
        "status": "candidate_paths_observed" if fragments else "unresolved",
        "fragments": fragments,
        "junctions": junctions,
        "connections": connections,
        "components": components,
        "composite_projected_bars": composite_projected_bars,
        "repetition_groups": repetition,
        "mark_hypotheses": mark_hypotheses,
        "metric_frames": scales,
        "projected_length_observations_by_view": [
            {
                "view_id": view_id,
                "component_count": len(rows),
                "sum_projected_length_mm": round(sum(float(item["projected_length"]["value_mm"]) for item in rows), 3),
                "state": "non_aggregable_observation",
                "reason": "components may be duplicate projections, non-rebar graphics, or coincident physical instances",
            }
            for view_id, rows in sorted(projected_by_view.items())
        ],
        "summary": {
            "fragment_count": len(fragments),
            "junction_count": len(junctions),
            "accepted_connection_count": sum(item["state"] == "accepted" for item in connections),
            "ambiguous_connection_count": sum(item["state"] != "accepted" for item in connections),
            "component_count": len(components),
            "accepted_composite_projected_bar_count": composite_projected_bars["summary"]["accepted_count"],
            "review_composite_projected_bar_count": composite_projected_bars["summary"]["review_candidate_count"],
            "nonbranching_projection_count": len(nonbranching),
            "metric_projected_component_count": len(metric_components),
            "installed_centerline_resolved_count": 0,
        },
        "validation": {
            "duplicate_fragment_refs": len(fragments) - len({item["primitive_ref"] for item in fragments}),
            "all_accepted_connections_reference_fragments": all(
                item["from_fragment_id"] in {row["id"] for row in fragments}
                and item["to_fragment_id"] in {row["id"] for row in fragments}
                for item in connections
                if item["state"] == "accepted"
            ),
            "schedule_values_used": False,
        },
        "contract": {
            "object_template_required": False,
            "host_contours_are_constraints_not_bar_paths": True,
            "visible_projection_is_not_physical_length": True,
            "cross_view_identity_required_before_deduplication": True,
            "composite_projected_bars_do_not_change_quantities": True,
            "fabrication_rules_are_separate": True,
            "schedule_values_used": False,
        },
    }
    enrich_projection_identities(graph)
    return graph
