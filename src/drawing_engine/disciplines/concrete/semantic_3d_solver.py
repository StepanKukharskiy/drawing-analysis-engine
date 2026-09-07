"""Drawing-neutral constraint solver for procedural semantic 3D records.

The solver consumes native PDF vectors/text plus reusable region and dimension
proposals. It closes only relations supported by geometry and equations; all
remaining reinforcement groups are emitted as unresolved rather than filled by
sheet-specific constants.
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict, deque
from math import dist, hypot
from typing import Any

import fitz

from src.drawing_engine.disciplines.concrete.concrete_reconstruction import analytic_volume_mm3, engineering_mesh, spec_from_semantics
from src.drawing_engine.core.cross_view_identity import extract_thin_segments, trace_leader
from src.drawing_engine.core.dimension_attachment import DimensionAttachment, attach_dimensions
from src.drawing_engine.disciplines.rebar.reinforcement_scene import reinforcement_from_semantics, reinforcement_payload
from src.drawing_engine.disciplines.rebar.rebar_metric_solver import solve_dimension_anchored_straight_group
from src.drawing_engine.disciplines.rebar.section_rebar_extraction import extract_section_rebar
from src.drawing_engine.core.semantic_region_grouping import RegionProposal, propose_semantic_regions


STANDARD_DIAMETERS_MM = (6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 32, 36, 40)


def _snap(value: float, choices: tuple[int, ...] = STANDARD_DIAMETERS_MM) -> int:
    return min(choices, key=lambda item: abs(item - value))


def _inside(rect: fitz.Rect, point: tuple[float, float]) -> bool:
    return fitz.Point(*point) in rect


def _host_section(
    page: fitz.Page,
    region: RegionProposal,
    dimensions: tuple[DimensionAttachment, ...],
) -> tuple[fitz.Rect, DimensionAttachment]:
    box = fitz.Rect(region.bbox)
    candidates: list[fitz.Rect] = []
    for drawing in page.get_drawings():
        rect = fitz.Rect(drawing["rect"])
        if rect.width <= 0 or rect.height <= 0 or not box.contains(rect):
            continue
        ratio = min(rect.width, rect.height) / max(rect.width, rect.height)
        if ratio >= 0.98 and 0.20 * box.width <= rect.width <= 0.70 * box.width:
            candidates.append(rect)
    if not candidates:
        raise ValueError("no square host-section outline found")
    outline = max(candidates, key=lambda rect: rect.get_area())
    local = [
        item
        for item in dimensions
        if item.status == "accepted"
        and item.orientation == "horizontal"
        and all(_inside(box, point) for point in item.measured_points)
        and abs(abs(item.measured_points[1][0] - item.measured_points[0][0]) - outline.width) <= 4.0
    ]
    if not local:
        raise ValueError("host outline has no attached horizontal dimension")
    dimension = min(local, key=lambda item: abs(item.scale_points_per_mm - outline.width / item.value_mm))
    return outline, dimension


def _dimension_chain(
    region: RegionProposal,
    dimensions: tuple[DimensionAttachment, ...],
    expected_scale: float,
) -> dict[str, Any]:
    box = fitz.Rect(region.bbox)
    horizontal = [
        item
        for item in dimensions
        if item.status == "accepted"
        and item.orientation == "horizontal"
        and abs(item.scale_points_per_mm / expected_scale - 1.0) <= 0.04
        and box.intersects(fitz.Rect(item.text_bbox))
    ]
    if len(horizontal) < 3:
        raise ValueError("final section has no compatible width chain")
    overall = max(horizontal, key=lambda item: item.value_mm)
    ox0, ox1 = sorted(point[0] for point in overall.measured_points)
    pieces = []
    for item in horizontal:
        x0, x1 = sorted(point[0] for point in item.measured_points)
        if item is overall or x0 < ox0 - 6 or x1 > ox1 + 6:
            continue
        pieces.append((x0, x1, item))

    queue: deque[tuple[float, list[DimensionAttachment]]] = deque([(ox0, [])])
    solution: list[DimensionAttachment] | None = None
    while queue:
        cursor, path = queue.popleft()
        if abs(cursor - ox1) <= 6 and path and abs(sum(item.value_mm for item in path) - overall.value_mm) <= 1:
            solution = path
            break
        for x0, x1, item in pieces:
            if item in path or abs(x0 - cursor) > 6 or x1 <= cursor + 2:
                continue
            queue.append((x1, [*path, item]))
    if solution is None:
        raise ValueError("final-section dimension chain does not close")
    return {
        "overall": overall,
        "pieces": solution,
        "equation": f"{' + '.join(str(int(item.value_mm)) for item in solution)} = {int(overall.value_mm)}",
        "residual_mm": sum(item.value_mm for item in solution) - overall.value_mm,
    }


def _profile_components(page: fitz.Page) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for drawing_index, drawing in enumerate(page.get_drawings()):
        width = float(drawing.get("width") or 0)
        color = drawing.get("color")
        if not (0.42 <= width <= 0.55) or color is None or max(color) > 0.05 or drawing.get("fill") is not None:
            continue
        for item_index, item in enumerate(drawing["items"]):
            if item[0] != "l":
                continue
            a, b = item[1], item[2]
            start, end = (float(a.x), float(a.y)), (float(b.x), float(b.y))
            if dist(start, end) < 5:
                continue
            segments.append({"start": start, "end": end, "ref": f"drawing[{drawing_index}].item[{item_index}]"})

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, segment in enumerate(segments):
        for point in (segment["start"], segment["end"]):
            buckets[(round(point[0]), round(point[1]))].append(index)
    adjacency: dict[int, set[int]] = defaultdict(set)
    for indices in buckets.values():
        for left in indices:
            adjacency[left].update(item for item in indices if item != left)

    components = []
    unseen = set(range(len(segments)))
    while unseen:
        seed = unseen.pop()
        stack = [seed]
        indices = {seed}
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    indices.add(neighbor)
                    stack.append(neighbor)
        selected = [segments[index] for index in indices]
        points = [point for segment in selected for point in (segment["start"], segment["end"])]
        bbox = fitz.Rect(min(x for x, _ in points), min(y for _, y in points), max(x for x, _ in points), max(y for _, y in points))
        diagonal = sum(abs(a[0] - b[0]) > 2 and abs(a[1] - b[1]) > 2 for a, b in ((item["start"], item["end"]) for item in selected))
        components.append({"segments": selected, "bbox": bbox, "diagonal_count": diagonal})
    return components


def _profile_closure(segments: list[dict[str, Any]]) -> dict[str, Any]:
    """Reconstruct closed profile cycles from native segment endpoints."""

    def key(point: tuple[float, float]) -> tuple[float, float]:
        return round(point[0], 1), round(point[1], 1)

    points: dict[tuple[float, float], tuple[float, float]] = {}
    edges: list[tuple[tuple[float, float], tuple[float, float]]] = []
    adjacency: dict[tuple[float, float], list[int]] = defaultdict(list)
    for segment in segments:
        left, right = key(segment["start"]), key(segment["end"])
        points.setdefault(left, segment["start"])
        points.setdefault(right, segment["end"])
        edge_index = len(edges)
        edges.append((left, right))
        adjacency[left].append(edge_index)
        adjacency[right].append(edge_index)

    unused = set(range(len(edges)))
    cycles: list[list[tuple[float, float]]] = []
    while unused:
        first_edge = min(unused)
        unused.remove(first_edge)
        start, current = edges[first_edge]
        polygon = [points[start], points[current]]
        previous_edge = first_edge
        while current != start:
            choices = [edge for edge in adjacency[current] if edge in unused and edge != previous_edge]
            if not choices:
                polygon = []
                break
            edge_index = min(choices)
            unused.remove(edge_index)
            left, right = edges[edge_index]
            current = right if left == current else left
            polygon.append(points[current])
            previous_edge = edge_index
            if len(polygon) > len(edges) + 1:
                polygon = []
                break
        if len(polygon) >= 4 and polygon[0] == polygon[-1]:
            cycles.append(polygon)

    def signed_area(polygon: list[tuple[float, float]]) -> float:
        return 0.5 * sum(
            left[0] * right[1] - right[0] * left[1]
            for left, right in zip(polygon, polygon[1:])
        )

    cycles.sort(key=lambda polygon: abs(signed_area(polygon)), reverse=True)
    selected = cycles[0] if cycles else []
    degrees = [len(rows) for rows in adjacency.values()]
    return {
        "status": "pass" if selected and all(value == 2 for value in degrees) else "review",
        "closed_cycle_count": len(cycles),
        "node_count": len(adjacency),
        "edge_count": len(edges),
        "all_nodes_degree_two": bool(degrees) and all(value == 2 for value in degrees),
        "polygon_points_display": [[float(x), float(y)] for x, y in selected],
        "signed_area_points2": signed_area(selected) if selected else None,
    }


def _formwork_profile(page: fitz.Page, height_dimension: DimensionAttachment, shaft_width_mm: float) -> dict[str, Any]:
    top_y = min(point[1] for point in height_dimension.measured_points)
    bottom_y = max(point[1] for point in height_dimension.measured_points)
    scale = (bottom_y - top_y) / height_dimension.value_mm
    candidates = [
        component
        for component in _profile_components(page)
        if component["bbox"].height >= 0.80 * (bottom_y - top_y)
        and 0.02 * page.rect.width <= component["bbox"].width <= 0.25 * page.rect.width
        and component["diagonal_count"] >= 1
    ]
    if not candidates:
        raise ValueError("no connected formwork profile matches the height dimension")
    component = max(candidates, key=lambda item: item["bbox"].height + item["bbox"].width)
    bbox = component["bbox"]
    closure = _profile_closure(component["segments"])
    vertical_totals: dict[float, float] = defaultdict(float)
    for segment in component["segments"]:
        a, b = segment["start"], segment["end"]
        if abs(a[0] - b[0]) <= 0.5:
            vertical_totals[round((a[0] + b[0]) / 2, 1)] += abs(a[1] - b[1])
    shaft_edges = sorted(
        (x for x, length in vertical_totals.items() if length >= 0.55 * bbox.height),
        key=lambda x: vertical_totals[x],
        reverse=True,
    )[:2]
    if len(shaft_edges) != 2:
        raise ValueError("formwork profile has no stable shaft-edge pair")
    left, right = sorted(shaft_edges)
    observed_width = (right - left) / scale
    if abs(observed_width / shaft_width_mm - 1) > 0.04:
        raise ValueError("formwork shaft width conflicts with section dimension")

    def z(y: float) -> float:
        return (bottom_y - y) / scale

    corbels = []
    for direction, shaft_edge, extreme in ((-1, left, bbox.x0), (1, right, bbox.x1)):
        if abs(extreme - shaft_edge) <= 4:
            continue
        wing_segments = [
            segment
            for segment in component["segments"]
            if (direction < 0 and min(segment["start"][0], segment["end"][0]) < left - 2)
            or (direction > 0 and max(segment["start"][0], segment["end"][0]) > right + 2)
        ]
        outer_vertical = [
            segment
            for segment in wing_segments
            if abs(segment["start"][0] - segment["end"][0]) <= 0.5
            and abs((segment["start"][0] + segment["end"][0]) / 2 - extreme) <= 1.5
        ]
        if not outer_vertical:
            raise ValueError("corbel profile lacks outer rectangular edge")
        outer_points = [point for segment in outer_vertical for point in (segment["start"], segment["end"])]
        all_points = [point for segment in wing_segments for point in (segment["start"], segment["end"])]
        corbels.append(
            {
                "direction_x": direction,
                "profile_projection_mm": abs(extreme - shaft_edge) / scale,
                "taper_bottom_z_mm": z(max(point[1] for point in all_points)),
                "rectangular_bottom_z_mm": z(max(point[1] for point in outer_points)),
                "top_z_mm": z(min(point[1] for point in outer_points)),
                "primitive_refs": sorted({segment["ref"] for segment in wing_segments}),
            }
        )
    if not corbels:
        raise ValueError("formwork profile contains no corbel wing")
    return {
        "bbox_display": list(bbox),
        "shaft_edges_x_display": [left, right],
        "height_scale_points_per_mm": scale,
        "corbels": corbels,
        "primitive_refs": sorted({segment["ref"] for segment in component["segments"]}),
        "segments_display": [
            {
                "start_display": list(segment["start"]),
                "end_display": list(segment["end"]),
                "primitive_ref": segment["ref"],
            }
            for segment in component["segments"]
        ],
        "closure_validation": {key: value for key, value in closure.items() if key != "polygon_points_display"},
        "polygon_points_display": closure["polygon_points_display"],
    }


def _attach_profile_levels_to_dimensions(
    profile: dict[str, Any],
    dimensions: tuple[DimensionAttachment, ...],
) -> dict[str, Any]:
    """Attach absolute and incremental dimensions to profile Z edges."""

    scale = profile["height_scale_points_per_mm"]
    compatible = [
        item
        for item in dimensions
        if item.status == "accepted"
        and abs(item.scale_points_per_mm / scale - 1.0) <= 0.04
    ]
    raw = [dict(item) for item in profile["corbels"]]
    resolved = []
    relations = []
    for index, item in enumerate(raw, start=1):
        base_raw = item["taper_bottom_z_mm"]
        lower_raw = item["rectangular_bottom_z_mm"] - item["taper_bottom_z_mm"]
        upper_raw = item["top_z_mm"] - item["rectangular_bottom_z_mm"]
        base_candidates = [candidate for candidate in compatible if candidate.value_mm > 1000 and abs(candidate.value_mm - base_raw) <= 5]
        delta_candidates = [
            candidate
            for candidate in compatible
            if 50 <= candidate.value_mm <= 1000
            and max(abs(candidate.value_mm - lower_raw), abs(candidate.value_mm - upper_raw)) <= 5
        ]
        base = min(base_candidates, key=lambda candidate: abs(candidate.value_mm - base_raw)) if base_candidates else None
        delta = min(
            delta_candidates,
            key=lambda candidate: abs(candidate.value_mm - lower_raw) + abs(candidate.value_mm - upper_raw),
        ) if delta_candidates else None
        taper = base.value_mm if base is not None else base_raw
        lower = delta.value_mm if delta is not None else lower_raw
        upper = delta.value_mm if delta is not None else upper_raw
        item.update(
            {
                "taper_bottom_z_mm": taper,
                "rectangular_bottom_z_mm": taper + lower,
                "top_z_mm": taper + lower + upper,
                "level_status": (
                    "dimension_attached"
                    if base is not None and delta is not None
                    else "profile_scaled"
                ),
                "level_dimension_ids": [
                    candidate.attachment_id
                    for candidate in (base, delta)
                    if candidate is not None
                ],
            }
        )
        resolved.append(item)
        relations.append(
            {
                "corbel_index": index,
                "raw_levels_mm": [base_raw, base_raw + lower_raw, base_raw + lower_raw + upper_raw],
                "resolved_levels_mm": [taper, taper + lower, taper + lower + upper],
                "base_dimension_id": None if base is None else base.attachment_id,
                "increment_dimension_id": None if delta is None else delta.attachment_id,
                "max_adjustment_mm": max(
                    abs(taper - base_raw),
                    abs(taper + lower - (base_raw + lower_raw)),
                    abs(taper + lower + upper - (base_raw + lower_raw + upper_raw)),
                ),
            }
        )
    return {"raw_corbels": raw, "resolved_corbels": resolved, "relations": relations}


def _concrete_quantity_takeoff(spec: Any) -> dict[str, Any]:
    shaft_mm3 = spec.width_x_mm * spec.depth_y_mm * spec.height_z_mm
    corbels = []
    for index, corbel in enumerate(spec.corbels, start=1):
        rectangular_height = corbel.top_z_mm - corbel.rectangular_bottom_z_mm
        triangular_height = corbel.rectangular_bottom_z_mm - corbel.taper_bottom_z_mm
        volume = corbel.projection_mm * corbel.depth_mm * (rectangular_height + triangular_height / 2)
        corbels.append(
            {
                "component": f"corbel.{index}",
                "direction_x": corbel.direction_x,
                "projection_mm": corbel.projection_mm,
                "depth_mm": corbel.depth_mm,
                "rectangular_height_mm": rectangular_height,
                "triangular_height_mm": triangular_height,
                "volume_mm3": volume,
                "volume_m3": volume / 1_000_000_000,
            }
        )
    recesses = []
    for index, recess in enumerate(spec.recesses, start=1):
        volume = recess.width_mm * recess.height_mm * recess.depth_mm
        recesses.append(
            {
                "component": f"recess.{index}",
                "size_mm": [recess.width_mm, recess.height_mm, recess.depth_mm],
                "deduction_mm3": volume,
                "deduction_m3": volume / 1_000_000_000,
            }
        )
    gross = shaft_mm3 + sum(item["volume_mm3"] for item in corbels)
    deduction = sum(item["deduction_mm3"] for item in recesses)
    net = gross - deduction
    return {
        "status": "procedurally_calculated",
        "units": "m3",
        "shaft": {"volume_mm3": shaft_mm3, "volume_m3": shaft_mm3 / 1_000_000_000},
        "corbels": corbels,
        "gross_concrete_mm3": gross,
        "gross_concrete_m3": gross / 1_000_000_000,
        "recess_deduction_mm3": deduction,
        "recess_deduction_m3": deduction / 1_000_000_000,
        "net_concrete_mm3": net,
        "net_concrete_m3": net / 1_000_000_000,
        "note": "Recess elevation uncertainty does not change its volume deduction; only resolved void size is deducted.",
    }


def _recesses(page: fitz.Page, regions: tuple[RegionProposal, ...], height_dimension: DimensionAttachment) -> dict[str, Any]:
    excluded = [
        fitz.Rect(item.bbox)
        for item in regions
        if item.kind in {"section_view", "bar_detail_group"}
    ]
    top_y = min(point[1] for point in height_dimension.measured_points)
    bottom_y = max(point[1] for point in height_dimension.measured_points)
    scale = (bottom_y - top_y) / height_dimension.value_mm
    thin = extract_thin_segments(page)
    observations = []
    pattern = re.compile(r"(?P<w>\d+)\s*[xх×]\s*(?P<h>\d+)\s*[xх×]\s*(?P<d>\d+)", re.I)
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
            match = pattern.search(text)
            if not match:
                continue
            box = fitz.Rect(line["bbox"])
            center = fitz.Point((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)
            if any(center in region for region in excluded):
                continue
            trace = trace_leader(box, thin, search_radius=200, max_hops=10)
            if not trace["terminals"]:
                continue
            terminal = max(trace["terminals"], key=lambda point: dist((center.x, center.y), point))
            if not (top_y <= terminal[1] <= bottom_y):
                continue
            observations.append(
                {
                    "size_mm": [int(match.group("w")), int(match.group("h")), int(match.group("d"))],
                    "terminal_display": terminal,
                    "center_z_candidate_mm": (bottom_y - terminal[1]) / scale,
                    "text_bbox_display": list(box),
                    "leader_refs": [segment["drawing_ref"] for segment in trace["segments"]],
                }
            )
    clusters: list[list[dict[str, Any]]] = []
    for item in sorted(observations, key=lambda value: value["center_z_candidate_mm"]):
        if not clusters or abs(statistics.median(row["center_z_candidate_mm"] for row in clusters[-1]) - item["center_z_candidate_mm"]) > 250:
            clusters.append([item])
        else:
            clusters[-1].append(item)
    return {
        "observations": observations,
        "clusters": [
            {
                "size_mm": cluster[0]["size_mm"],
                "center_z_mm": statistics.median(item["center_z_candidate_mm"] for item in cluster),
                "support": len(cluster),
                "residual_span_mm": max(item["center_z_candidate_mm"] for item in cluster) - min(item["center_z_candidate_mm"] for item in cluster),
            }
            for cluster in clusters
        ],
    }


def _longitudinal_bars(page: fitz.Page, outline: fitz.Rect, scale: float) -> dict[str, Any]:
    candidates = []
    for drawing_index, drawing in enumerate(page.get_drawings()):
        fill = drawing.get("fill")
        rect = fitz.Rect(drawing["rect"])
        if fill is None or max(fill) > 0.05 or not outline.contains(rect):
            continue
        if rect.width <= 0 or rect.height <= 0 or min(rect.width, rect.height) / max(rect.width, rect.height) < 0.80:
            continue
        raw = max(rect.width, rect.height) / scale
        diameter = _snap(raw)
        if diameter < 16 or abs(raw - diameter) > max(1.0, diameter * 0.06):
            continue
        candidates.append((drawing_index, rect, diameter, raw))
    if len(candidates) < 2:
        raise ValueError("no longitudinal end projections found")
    diameter = statistics.mode(item[2] for item in candidates)
    selected = [item for item in candidates if item[2] == diameter]
    center = fitz.Point((outline.x0 + outline.x1) / 2, (outline.y0 + outline.y1) / 2)
    return {
        "mark": "auto.longitudinal.primary",
        "role": "longitudinal",
        "count": len(selected),
        "diameter_mm": diameter,
        "xy_mm": [
            [
                ((rect.x0 + rect.x1) / 2 - center.x) / scale,
                -((rect.y0 + rect.y1) / 2 - center.y) / scale,
            ]
            for _, rect, _, _ in selected
        ],
        "projection_bboxes_display": [list(rect) for _, rect, _, _ in selected],
        "primitive_refs": [f"drawing[{index}]" for index, *_ in selected],
        "raw_diameter_samples_mm": [item[3] for item in selected],
    }


def _repeated_ties(page: fitz.Page, height_dimension: DimensionAttachment) -> dict[str, Any]:
    clusters: dict[tuple[float, float], list[tuple[float, float, str]]] = defaultdict(list)
    for drawing_index, drawing in enumerate(page.get_drawings()):
        width = float(drawing.get("width") or 0)
        if width <= 1.5:
            continue
        for item_index, item in enumerate(drawing["items"]):
            if item[0] != "l":
                continue
            a, b = item[1], item[2]
            dx, dy = abs(a.x - b.x), abs(a.y - b.y)
            if dy > 0.25 or not (20 <= dx <= 80):
                continue
            x0, x1 = sorted((float(a.x), float(b.x)))
            clusters[(round(x0, 1), round(x1, 1))].append(((a.y + b.y) / 2, width, f"drawing[{drawing_index}].item[{item_index}]"))
    span, items = max(clusters.items(), key=lambda pair: len({round(row[0], 2) for row in pair[1]}))
    refs_by_y: dict[float, set[str]] = defaultdict(set)
    for y, _, ref in items:
        refs_by_y[round(y, 2)].add(ref)
    ys = sorted(refs_by_y)
    if len(ys) < 10:
        raise ValueError("no dominant repeated tie span found")
    top_y = min(point[1] for point in height_dimension.measured_points)
    bottom_y = max(point[1] for point in height_dimension.measured_points)
    scale = (bottom_y - top_y) / height_dimension.value_mm
    return {
        "mark": "auto.transverse.primary",
        "role": "closed_transverse_tie",
        "count": len(ys),
        "span_display": list(span),
        "z_mm": [(bottom_y - y) / scale for y in ys],
        "instance_y_display": ys,
        "instance_refs": [sorted(refs_by_y[y]) for y in ys],
        "primitive_refs": sorted({row[2] for row in items}),
    }


def _tie_section_geometry(page: fitz.Page, outline: fitz.Rect, scale: float) -> dict[str, Any]:
    """Recover a closed tie centreline from paired native-vector boundaries.

    Native CAD exports draw a bent bar as two nearby boundaries.  Stroke width
    is intentionally ignored: the boundary separation carries the diameter,
    while the mean of each pair carries the fabrication centreline.
    """

    horizontal: list[tuple[float, str]] = []
    vertical: list[tuple[float, str]] = []
    margin = 10.0
    for drawing_index, drawing in enumerate(page.get_drawings()):
        if float(drawing.get("width") or 0) < 1.5:
            continue
        for item_index, item in enumerate(drawing["items"]):
            if item[0] != "l":
                continue
            a, b = item[1], item[2]
            dx, dy = abs(a.x - b.x), abs(a.y - b.y)
            ref = f"drawing[{drawing_index}].item[{item_index}]"
            if (
                dx >= 0.50 * outline.width
                and dy <= 0.25
                and outline.x0 - margin <= min(a.x, b.x)
                and max(a.x, b.x) <= outline.x1 + margin
                and outline.y0 - margin <= (a.y + b.y) / 2 <= outline.y1 + margin
            ):
                horizontal.append(((a.y + b.y) / 2, ref))
            if (
                dy >= 0.50 * outline.height
                and dx <= 0.25
                and outline.y0 - margin <= min(a.y, b.y)
                and max(a.y, b.y) <= outline.y1 + margin
                and outline.x0 - margin <= (a.x + b.x) / 2 <= outline.x1 + margin
            ):
                vertical.append(((a.x + b.x) / 2, ref))

    def unique_axes(items: list[tuple[float, str]]) -> list[tuple[float, list[str]]]:
        groups: list[list[tuple[float, str]]] = []
        for item in sorted(items):
            if not groups or abs(statistics.median(value for value, _ in groups[-1]) - item[0]) > 0.45:
                groups.append([item])
            else:
                groups[-1].append(item)
        return [
            (statistics.median(value for value, _ in group), sorted({ref for _, ref in group}))
            for group in groups
        ]

    h_axes = unique_axes(horizontal)
    v_axes = unique_axes(vertical)
    if len(h_axes) < 4 or len(v_axes) < 4:
        raise ValueError("section has no complete paired-boundary tie topology")

    # The two extreme pairs are the four long sides of the same closed bar.
    top_pair, bottom_pair = h_axes[:2], h_axes[-2:]
    left_pair, right_pair = v_axes[:2], v_axes[-2:]
    gaps = [
        top_pair[1][0] - top_pair[0][0],
        bottom_pair[1][0] - bottom_pair[0][0],
        left_pair[1][0] - left_pair[0][0],
        right_pair[1][0] - right_pair[0][0],
    ]
    raw_diameter = statistics.median(gaps) / scale
    diameter = _snap(raw_diameter)
    if abs(raw_diameter - diameter) > max(1.0, diameter * 0.08):
        raise ValueError("paired tie boundaries do not snap to one standard diameter")

    cx = (outline.x0 + outline.x1) / 2
    cy = (outline.y0 + outline.y1) / 2
    left = statistics.mean(value for value, _ in left_pair)
    right = statistics.mean(value for value, _ in right_pair)
    top = statistics.mean(value for value, _ in top_pair)
    bottom = statistics.mean(value for value, _ in bottom_pair)
    center_offset = ((left + right) / 2 - cx, -((top + bottom) / 2 - cy))
    if hypot(*center_offset) / scale > 8:
        raise ValueError("tie centreline is not concentric with its host section")
    refs = sorted(
        {
            ref
            for pair in (top_pair, bottom_pair, left_pair, right_pair)
            for _, pair_refs in pair
            for ref in pair_refs
        }
    )
    return {
        "diameter_mm": diameter,
        "raw_diameter_mm": raw_diameter,
        "half_x_mm": (right - left) / (2 * scale),
        "half_y_mm": (bottom - top) / (2 * scale),
        "center_offset_mm": [center_offset[0] / scale, center_offset[1] / scale],
        "primitive_refs": refs,
        "boundary_axes_display": {
            "horizontal": [top_pair[0][0], top_pair[1][0], bottom_pair[0][0], bottom_pair[1][0]],
            "vertical": [left_pair[0][0], left_pair[1][0], right_pair[0][0], right_pair[1][0]],
        },
    }


def _reinforcement_centerlines(
    longitudinal: dict[str, Any],
    ties: dict[str, Any],
    tie_geometry: dict[str, Any],
) -> dict[str, Any]:
    """Create only the 3D paths that are closed by section/elevation evidence."""

    z_values = ties["z_mm"]
    metric_solution = longitudinal.get("dimension_anchored_solution", {})
    if metric_solution.get("status") == "pass":
        solved = metric_solution["solved_centerline"]
        z0, z1 = solved["z_start_mm"], solved["z_end_mm"]
        longitudinal_evidence = [
            *metric_solution["elevation_observations"]["primitive_refs"],
            *(
                ref
                for row in metric_solution["section_validation"]["tested_sections"]
                for ref in row["primitive_refs"]
            ),
            metric_solution["metric_frame"]["host_dimension_id"],
            "dimension-anchored endpoints with cross-view reprojection validation",
        ]
    else:
        z0, z1 = min(z_values), max(z_values)
        longitudinal_evidence = ["fallback elevation envelope bounded by first and last resolved transverse projection"]
    records: list[dict[str, Any]] = []
    for instance, (x, y) in enumerate(longitudinal["xy_mm"], start=1):
        records.append(
            {
                "mark": longitudinal["mark"],
                "role": longitudinal["role"],
                "instance": instance,
                "diameter_mm": longitudinal["diameter_mm"],
                "placement_status": "drawing_constrained",
                "points_xyz_mm": [[x, y, z0], [x, y, z1]],
                "closed": False,
                "evidence": [
                    longitudinal["primitive_refs"][instance - 1],
                    *longitudinal_evidence,
                ],
            }
        )
    hx, hy = tie_geometry["half_x_mm"], tie_geometry["half_y_mm"]
    ox, oy = tie_geometry["center_offset_mm"]
    for instance, z in enumerate(z_values, start=1):
        records.append(
            {
                "mark": ties["mark"],
                "role": ties["role"],
                "instance": instance,
                "diameter_mm": tie_geometry["diameter_mm"],
                "placement_status": "drawing_constrained",
                "points_xyz_mm": [
                    [ox - hx, oy - hy, z],
                    [ox + hx, oy - hy, z],
                    [ox + hx, oy + hy, z],
                    [ox - hx, oy + hy, z],
                    [ox - hx, oy - hy, z],
                ],
                "closed": True,
                "evidence": [
                    *ties["instance_refs"][instance - 1],
                    *tie_geometry["primitive_refs"],
                ],
            }
        )
    paths = reinforcement_from_semantics({"reinforcement": records})
    return reinforcement_payload(paths)


def solve_semantic_3d(
    page: fitz.Page,
    dimensions: tuple[DimensionAttachment, ...] | None = None,
    text_roles: list[dict[str, Any]] | None = None,
    *,
    regions: tuple[Any, ...] | None = None,
    section_rebar: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a procedural concrete record plus resolved/partial reinforcement."""

    if dimensions is None:
        dimensions = attach_dimensions(page)
    accepted = tuple(item for item in dimensions if item.status == "accepted")
    if regions is None:
        regions = propose_semantic_regions(page, dimensions)
    sections = [item for item in regions if item.kind == "section_view"]
    if len(sections) < 2:
        raise ValueError("at least two section proposals are required")
    first_section, final_section = sections[0], sections[-1]
    outline, shaft_dimension = _host_section(page, first_section, accepted)
    shaft_width = shaft_dimension.value_mm
    section_scale = outline.width / shaft_width
    height_dimension = max(
        (item for item in accepted if item.orientation == "vertical"),
        key=lambda item: item.value_mm,
    )
    chain = _dimension_chain(final_section, accepted, section_scale)
    piece_values = [item.value_mm for item in chain["pieces"]]
    shaft_indices = [index for index, value in enumerate(piece_values) if abs(value - shaft_width) <= 1]
    if len(shaft_indices) != 1:
        raise ValueError("final-section chain does not contain one shaft width")
    projection_values = [value for index, value in enumerate(piece_values) if index not in shaft_indices]
    if not projection_values or max(projection_values) - min(projection_values) > 1:
        raise ValueError("corbel projection pieces are inconsistent")
    projection = statistics.median(projection_values)
    profile = _formwork_profile(page, height_dimension, shaft_width)
    level_attachment = _attach_profile_levels_to_dimensions(profile, accepted)
    profile["raw_corbels"] = level_attachment["raw_corbels"]
    profile["corbels"] = level_attachment["resolved_corbels"]
    profile["level_attachment_relations"] = level_attachment["relations"]
    profile_dimension_ids = sorted(
        {
            height_dimension.attachment_id,
            shaft_dimension.attachment_id,
            *(item.attachment_id for item in chain["pieces"]),
            *(
                dimension_id
                for relation in level_attachment["relations"]
                for dimension_id in (
                    relation.get("base_dimension_id"),
                    relation.get("increment_dimension_id"),
                )
                if dimension_id
            ),
        }
    )
    profile["calculation_contour"] = {
        "id": "calculation_contour.elevation.001",
        "role": "dimensioned_elevation_profile",
        "state": "derived",
        "bbox_display": profile["bbox_display"],
        "segments_display": profile["segments_display"],
        "polygon_points_display": profile["polygon_points_display"],
        "primitive_refs": profile["primitive_refs"],
        "dimension_refs": profile_dimension_ids,
        "dimensions": {
            "height_mm": height_dimension.value_mm,
            "shaft_width_mm": shaft_width,
            "projection_mm": projection,
            "profile_level_relations": level_attachment["relations"],
        },
        "closure_validation": profile["closure_validation"],
        "provenance": {
            "mode": "derived",
            "method": "connected native profile segments plus attached drawing dimensions",
            "schedule_values_used": False,
        },
    }
    if len(profile["corbels"]) != len(projection_values):
        raise ValueError("section-chain wing count conflicts with elevation profile")
    projection_residuals = [item["profile_projection_mm"] - projection for item in profile["corbels"]]
    if any(abs(value) > max(5.0, projection * 0.02) for value in projection_residuals):
        raise ValueError("section and elevation corbel projections conflict")

    recess_data = _recesses(page, regions, height_dimension)
    resolved_recesses = [cluster for cluster in recess_data["clusters"] if cluster["support"] >= 1]
    concrete_input = {
        "object_name": "procedurally_solved_concrete_host",
        "shaft": {
            "width_x_mm": shaft_width,
            "depth_y_mm": shaft_width,
            "height_z_mm": height_dimension.value_mm,
        },
        "corbels": [
            {
                "direction_x": item["direction_x"],
                "projection_mm": projection,
                "depth_mm": shaft_width,
                "taper_bottom_z_mm": item["taper_bottom_z_mm"],
                "rectangular_bottom_z_mm": item["rectangular_bottom_z_mm"],
                "top_z_mm": item["top_z_mm"],
                "status": "derived_from_connected_formwork_profile_and_closed_section_chain",
            }
            for item in profile["corbels"]
        ],
        "recesses": [
            {
                "center_x_mm": 0,
                "center_z_mm": item["center_z_mm"],
                "width_mm": item["size_mm"][0],
                "height_mm": item["size_mm"][1],
                "depth_mm": item["size_mm"][2],
                "face_y": 1,
                "status": "derived_from_native_size_text_and_leader_terminal; local labelled face defines +Y",
                "placement_precision": "approximate_leader_terminal",
                "uncertainty_mm": max(5.0, item["residual_span_mm"] / 2),
            }
            for item in resolved_recesses
        ],
        "declared_volume_m3": None,
        "source_status": "strict_procedural_native_vector_text_constraint_solution",
        "evidence": [
            f"host section dimension attachment {shaft_dimension.attachment_id}",
            f"height dimension attachment {height_dimension.attachment_id}",
            f"final section chain: {chain['equation']}",
            f"connected formwork profile with {len(profile['primitive_refs'])} primitive refs",
        ],
        "unknowns": [
            "recess leader terminals approximate elevation placement until dimension-to-recess-centre relations are explicit",
            "undimensioned embedded-item deduction policy",
        ],
    }
    spec = spec_from_semantics(concrete_input)
    solid, mesh = engineering_mesh(spec)
    quantity_takeoff = _concrete_quantity_takeoff(spec)

    longitudinal = _longitudinal_bars(page, outline, section_scale)
    ties = _repeated_ties(page, height_dimension)
    tie_geometry = _tie_section_geometry(page, outline, section_scale)
    ties.update(
        {
            "diameter_mm": tie_geometry["diameter_mm"],
            "section_centerline_half_size_mm": [tie_geometry["half_x_mm"], tie_geometry["half_y_mm"]],
            "section_boundary_refs": tie_geometry["primitive_refs"],
        }
    )
    if section_rebar is None:
        section_rebar = extract_section_rebar(page, regions, dimensions, text_roles)
    longitudinal["dimension_anchored_solution"] = solve_dimension_anchored_straight_group(
        page,
        longitudinal,
        ties["span_display"],
        height_dimension,
        shaft_width,
        section_rebar,
    )
    centerlines = _reinforcement_centerlines(longitudinal, ties, tie_geometry)
    reinforcement = {
        "resolved_groups": [longitudinal, ties],
        "centerline_scene": centerlines,
        "section_rebar_observations": section_rebar,
        "partial_groups": [],
        "unresolved_requirements": [
            "raster detail topology grammar for remaining bar marks",
            "leader-terminal multiplicity for U-bars",
            "out-of-plane placement for corbel reinforcement",
            "bend radii, hook rules, and fabrication centerline corrections",
        ],
        "full_3d_status": "partial_only",
    }
    profile_shaft_width = (profile["shaft_edges_x_display"][1] - profile["shaft_edges_x_display"][0]) / profile["height_scale_points_per_mm"]
    equations = [
            {"id": "section_width_chain", "equation": chain["equation"], "residual_mm": chain["residual_mm"], "tolerance_mm": 1.0},
            {"id": "shaft_cross_view_width", "section_mm": shaft_width, "profile_mm": profile_shaft_width, "residual_mm": profile_shaft_width - shaft_width, "tolerance_mm": max(5.0, shaft_width * 0.04)},
            {"id": "corbel_projection_cross_view", "section_mm": projection, "profile_residuals_mm": projection_residuals, "max_abs_residual_mm": max(abs(value) for value in projection_residuals), "tolerance_mm": max(5.0, projection * 0.02)},
            {
                "id": "analytic_mesh_volume",
                "analytic_mm3": analytic_volume_mm3(spec),
                "mesh_mm3": float(mesh.volume),
                "manifold_mm3": float(solid.volume()),
                "residual_mm3": float(mesh.volume) - analytic_volume_mm3(spec),
                "tolerance_mm3": 1e-2,
            },
            {
                "id": "longitudinal_count_vs_centerlines",
                "resolved_count": longitudinal["count"],
                "centerline_count": centerlines["counts_by_mark"][longitudinal["mark"]],
                "residual": centerlines["counts_by_mark"][longitudinal["mark"]] - longitudinal["count"],
                "tolerance": 0,
            },
            {
                "id": "longitudinal_dimension_anchored_reprojection",
                "residual_mm": longitudinal["dimension_anchored_solution"].get("reprojection", {}).get("max_endpoint_residual_mm", float("inf")),
                "tolerance_mm": longitudinal["dimension_anchored_solution"].get("reprojection", {}).get("tolerance_mm", 0.0),
            },
            {
                "id": "transverse_count_vs_centerlines",
                "resolved_count": ties["count"],
                "centerline_count": centerlines["counts_by_mark"][ties["mark"]],
                "residual": centerlines["counts_by_mark"][ties["mark"]] - ties["count"],
                "tolerance": 0,
            },
            {
                "id": "tie_boundary_diameter",
                "raw_mm": tie_geometry["raw_diameter_mm"],
                "snapped_mm": tie_geometry["diameter_mm"],
                "residual_mm": tie_geometry["raw_diameter_mm"] - tie_geometry["diameter_mm"],
                "tolerance_mm": 1.0,
            },
            {
                "id": "profile_level_dimension_attachment",
                "max_adjustment_mm": max(item["max_adjustment_mm"] for item in level_attachment["relations"]),
                "residual_mm": max(item["max_adjustment_mm"] for item in level_attachment["relations"]),
                "tolerance_mm": 5.0,
            },
        ]
    for equation in equations:
        if "max_abs_residual_mm" in equation:
            residual, tolerance = equation["max_abs_residual_mm"], equation["tolerance_mm"]
        elif "residual_mm3" in equation:
            residual, tolerance = abs(equation["residual_mm3"]), equation["tolerance_mm3"]
        elif "residual_mm" in equation:
            residual, tolerance = abs(equation["residual_mm"]), equation["tolerance_mm"]
        else:
            residual, tolerance = abs(equation["residual"]), equation["tolerance"]
        equation["status"] = "pass" if residual <= tolerance else "fail"
    constraints = {
        "status": "pass" if all(item["status"] == "pass" for item in equations) else "fail",
        "equations": equations,
    }
    if constraints["status"] != "pass":
        failed = ", ".join(item["id"] for item in equations if item["status"] != "pass")
        raise ValueError(f"semantic 3D constraint system did not validate: {failed}")
    relation_graph = {
        "nodes": [
            {"id": "view.section.host", "type": "section_object", "bbox_display": list(outline)},
            {"id": "view.elevation.host", "type": "elevation_object", "bbox_display": profile["bbox_display"]},
            {"id": "object.concrete.host", "type": "semantic_3d_object"},
            {"id": longitudinal["mark"], "type": "reinforcement_group"},
            *[
                {"id": anchor["id"], "type": anchor["type"], "coordinate": anchor["coordinate"], "state": anchor["state"]}
                for anchor in longitudinal["dimension_anchored_solution"].get("anchors", [])
            ],
            {"id": ties["mark"], "type": "reinforcement_group"},
        ],
        "relations": [
            {
                "id": "rel.dimension.shaft_width",
                "type": "dimension_attached_to_opposed_edges",
                "source": shaft_dimension.attachment_id,
                "target": "view.section.host",
                "semantic_edges": ["shaft.-X", "shaft.+X", "shaft.-Y", "shaft.+Y"],
                "value_mm": shaft_width,
                "provenance": {"mode": "derived", "method": "native_dimension_geometry"},
            },
            {
                "id": "rel.dimension.height",
                "type": "dimension_attached_to_opposed_edges",
                "source": height_dimension.attachment_id,
                "target": "view.elevation.host",
                "semantic_edges": ["shaft.base", "shaft.top"],
                "value_mm": height_dimension.value_mm,
                "provenance": {"mode": "derived", "method": "native_dimension_geometry"},
            },
            {
                "id": "rel.identity.concrete_cross_view",
                "type": "same_object_across_views",
                "sources": ["view.section.host", "view.elevation.host"],
                "target": "object.concrete.host",
                "checks": ["shaft_width", "wing_count", "wing_projection", "connected_profile"],
                "provenance": {"mode": "derived", "method": "constraint_consistent_view_matching"},
            },
            *[
                {
                    "id": f"rel.dimension.corbel_levels.{item['corbel_index']}",
                    "type": "dimensions_attached_to_profile_edges",
                    "sources": [
                        source
                        for source in (item["base_dimension_id"], item["increment_dimension_id"])
                        if source is not None
                    ],
                    "target": f"object.concrete.host.corbel.{item['corbel_index']}",
                    "resolved_levels_mm": item["resolved_levels_mm"],
                    "max_adjustment_mm": item["max_adjustment_mm"],
                    "provenance": {"mode": "derived", "method": "profile_edge_dimension_attachment"},
                }
                for item in level_attachment["relations"]
            ],
            {
                "id": "rel.rebar.longitudinal",
                "type": "projection_instances_to_straight_3d_group",
                "source": longitudinal["primitive_refs"],
                "target": longitudinal["mark"],
                "count": longitudinal["count"],
                "diameter_mm": longitudinal["diameter_mm"],
                "provenance": {"mode": "derived", "method": "filled_projection_geometry"},
            },
            {
                "id": "rel.rebar.longitudinal.metric_solution",
                "type": "dimension_anchored_straight_centerline",
                "sources": [anchor["id"] for anchor in longitudinal["dimension_anchored_solution"].get("anchors", [])],
                "target": longitudinal["mark"],
                "installed_length_each_mm": longitudinal["dimension_anchored_solution"].get("solved_centerline", {}).get("installed_length_each_mm"),
                "reprojection": longitudinal["dimension_anchored_solution"].get("reprojection"),
                "provenance": {"mode": "derived", "method": "hard_dimensions_soft_projection_constraints"},
            },
            {
                "id": "rel.rebar.transverse",
                "type": "section_topology_times_elevation_instances",
                "source": [*tie_geometry["primitive_refs"], *ties["primitive_refs"]],
                "target": ties["mark"],
                "topology": "closed_rectangular_centerline",
                "count": ties["count"],
                "diameter_mm": ties["diameter_mm"],
                "provenance": {"mode": "derived", "method": "paired_boundaries_and_repeated_span_matching"},
            },
        ],
    }
    return {
        "schema_version": "0.1.0",
        "pipeline_mode": "strict_procedural",
        "concrete_3d_input": concrete_input,
        "concrete_quantity_takeoff": quantity_takeoff,
        "reinforcement_3d_input": reinforcement,
        "constraint_validation": constraints,
        "relation_graph": relation_graph,
        "procedural_evidence": {
            "host_outline_display": list(outline),
            "profile": profile,
            "recess_detection": recess_data,
            "tie_section_geometry": tie_geometry,
            "accepted_dimension_ids": [item.attachment_id for item in accepted],
            "mesh": {
                "vertices_xyz_mm": [[float(value) for value in vertex] for vertex in mesh.vertices],
                "triangles": [[int(value) for value in face] for face in mesh.faces],
                "validation": {
                    "vertex_count": len(mesh.vertices),
                    "face_count": len(mesh.faces),
                    "watertight": bool(mesh.is_watertight),
                },
            },
        },
    }
