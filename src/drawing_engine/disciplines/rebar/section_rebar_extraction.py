"""Procedural section-host and native rebar-candidate extraction."""

from __future__ import annotations

import heapq
import re
import statistics
from math import hypot
from typing import Any

import fitz

from src.drawing_engine.core.cross_view_identity import extract_thin_segments, trace_leader
from src.drawing_engine.core.dimension_attachment import DimensionAttachment, attach_dimensions
from src.drawing_engine.core.semantic_region_grouping import RegionProposal


DRAWING_REF_RE = re.compile(r"^drawing\[(?P<index>\d+)\]$")


def _rect_distance(left: fitz.Rect, right: fitz.Rect) -> float:
    dx = max(left.x0 - right.x1, right.x0 - left.x1, 0.0)
    dy = max(left.y0 - right.y1, right.y0 - left.y1, 0.0)
    return hypot(dx, dy)


def _local_horizontal_dimensions(
    region: RegionProposal,
    dimensions: tuple[DimensionAttachment, ...],
) -> list[DimensionAttachment]:
    box = fitz.Rect(region.bbox)
    return [
        item
        for item in dimensions
        if item.status == "accepted"
        and item.orientation == "horizontal"
        and box.intersects(fitz.Rect(item.text_bbox))
    ]


def _local_vertical_dimensions(
    region: RegionProposal,
    dimensions: tuple[DimensionAttachment, ...],
) -> list[DimensionAttachment]:
    box = fitz.Rect(region.bbox)
    return [
        item
        for item in dimensions
        if item.status == "accepted"
        and item.orientation == "vertical"
        and box.intersects(fitz.Rect(item.text_bbox))
    ]


def _candidate_paths(page: fitz.Page, region: RegionProposal) -> list[dict[str, Any]]:
    box = fitz.Rect(region.bbox)
    candidates = []
    seen: set[tuple[Any, ...]] = set()
    for drawing_index, drawing in enumerate(page.get_drawings()):
        source_rect = fitz.Rect(drawing["rect"])
        width = float(drawing.get("width") or 0)
        fill = drawing.get("fill")
        padding = max(0.75, width / 2)
        rect = source_rect + (-padding, -padding, padding, padding)
        if not box.intersects(rect):
            continue
        black_fill = fill is not None and max(fill) <= 0.05
        if width < 1.5 and not black_fill:
            continue
        signature = (
            round(rect.x0, 1),
            round(rect.y0, 1),
            round(rect.x1, 1),
            round(rect.y1, 1),
            round(width, 1),
            black_fill,
        )
        if signature in seen:
            continue
        seen.add(signature)
        compact = rect.width > 0 and rect.height > 0 and min(rect.width, rect.height) / max(rect.width, rect.height) >= 0.75
        candidates.append(
            {
                "drawing_index": drawing_index,
                "primitive_ref": f"drawing[{drawing_index}]",
                "bbox_display": list(rect),
                "source_bbox_display": list(source_rect),
                "stroke_width_pt": width,
                "filled": black_fill,
                "compact": compact,
                "geometry_items": [
                    (
                        {
                            "kind": "line",
                            "start_display": [float(item[1].x), float(item[1].y)],
                            "end_display": [float(item[2].x), float(item[2].y)],
                        }
                        if item[0] == "l"
                        else {
                            "kind": "cubic_bezier",
                            "start_display": [float(item[1].x), float(item[1].y)],
                            "control_1_display": [float(item[2].x), float(item[2].y)],
                            "control_2_display": [float(item[3].x), float(item[3].y)],
                            "end_display": [float(item[4].x), float(item[4].y)],
                        }
                    )
                    for item in drawing.get("items", [])
                    if item[0] in {"l", "c"}
                ],
            }
        )
    return candidates


def _numeric_token_role(
    token: str,
    word_box: fitz.Rect,
    dimensions: tuple[DimensionAttachment, ...],
    text_roles: list[dict[str, Any]] | None,
) -> tuple[str, str]:
    role = next(
        (
            item["resolved_role"]
            for item in text_roles or []
            if item.get("text") == token
            and _rect_distance(word_box, fitz.Rect(item["bbox_display"])) <= 1.5
        ),
        None,
    )
    if role == "identifier_candidate":
        return "identifier_candidate", "exclusive semantic identifier role"
    matches = [
        item
        for item in dimensions
        if item.text.strip() == token
        and _rect_distance(word_box, fitz.Rect(item.text_bbox)) <= 1.5
    ]
    if any(item.status == "accepted" for item in matches):
        return "dimension", "token belongs to an accepted dimension chain"
    if any(item.score >= 0.95 for item in matches):
        return "dimension_candidate", "token has a high-confidence dimension-chain attachment"
    return "identifier_candidate", "numeric token requires a leader-to-rebar terminal"


def _section_host_bbox(
    page: fitz.Page,
    region: RegionProposal,
    dimensions: tuple[DimensionAttachment, ...],
    candidates: list[dict[str, Any]],
) -> tuple[fitz.Rect, str]:
    box = fitz.Rect(region.bbox)
    local = _local_horizontal_dimensions(region, dimensions)
    if not local:
        raise ValueError(f"{region.label} has no accepted horizontal host dimension")
    overall = max(local, key=lambda item: item.value_mm)
    scale = overall.scale_points_per_mm
    square_outlines = []
    for drawing in page.get_drawings():
        rect = fitz.Rect(drawing["rect"])
        width = float(drawing.get("width") or 0)
        if not box.contains(rect) or not (1.0 <= width <= 1.5) or rect.get_area() < 1000:
            continue
        ratio = min(rect.width, rect.height) / max(rect.width, rect.height)
        if ratio >= 0.92 and abs(rect.width / scale - overall.value_mm) <= max(10.0, 0.03 * overall.value_mm):
            square_outlines.append(rect)
    if square_outlines:
        return max(square_outlines, key=lambda rect: rect.get_area()), "closed_concrete_outline"

    x0, x1 = sorted((overall.measured_points[0][0], overall.measured_points[1][0]))
    seed_floor = box.y0 + 0.10 * box.height
    eligible = []
    for candidate in candidates:
        rect = fitz.Rect(candidate["bbox_display"])
        center = fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
        if not (x0 - 10 <= center.x <= x1 + 10):
            continue
        if not (seed_floor <= center.y <= box.y1 - 0.03 * box.height):
            continue
        if max(rect.width, rect.height) > max(120.0, (x1 - x0) * 0.60):
            continue
        eligible.append(rect)
    if not eligible:
        raise ValueError(f"{region.label} has no heavy reinforcement component near its overall dimension")
    vertical_dimensions = [
        item
        for item in _local_vertical_dimensions(region, dimensions)
        if item.value_mm <= overall.value_mm
    ]
    expected_depth = (
        # A complex section commonly contains several equally valid chained
        # depths (for example 125 + 125 + 150 = 400).  Their attachment scores
        # can all be 1.0, so choosing by score alone silently picks whichever
        # small band was extracted first.  The enclosing host is defined by
        # the largest accepted local depth; smaller chains remain dimensions
        # of subregions inside that host.
        max(vertical_dimensions, key=lambda item: (item.value_mm, item.score)).value_mm * scale
        if vertical_dimensions
        else None
    )
    filled_centers = []
    for candidate in candidates:
        rect = fitz.Rect(candidate["bbox_display"])
        center = fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
        if candidate["filled"] and candidate["compact"] and x0 - 10 <= center.x <= x1 + 10 and seed_floor <= center.y <= box.y1:
            filled_centers.append(center.y)
    if len(filled_centers) >= 4:
        center_y = (min(filled_centers) + max(filled_centers)) / 2
    else:
        bins: dict[int, list[float]] = {}
        for rect in eligible:
            center_y = (rect.y0 + rect.y1) / 2
            bins.setdefault(round(center_y / 25), []).append(center_y)
        center_y = statistics.median(max(bins.values(), key=len))
    if expected_depth is None:
        observed_span = max(rect.y1 for rect in eligible) - min(rect.y0 for rect in eligible)
        expected_depth = max(observed_span, 0.08 * (x1 - x0))
    y0, y1 = center_y - expected_depth / 2, center_y + expected_depth / 2
    return fitz.Rect(x0, y0, x1, y1), "overall_dimension_and_heavy_path_component"


def _leader_identities(
    page: fitz.Page,
    region: RegionProposal,
    candidates: list[dict[str, Any]],
    dimensions: tuple[DimensionAttachment, ...],
    text_roles: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    box = fitz.Rect(region.bbox)
    dimension_boxes = [fitz.Rect(item.text_bbox) for item in dimensions if item.status == "accepted"]
    thin = extract_thin_segments(page)
    identities = []
    rejected = []
    for word in page.get_text("words"):
        token = str(word[4]).strip()
        if not token.isdigit() or not (1 <= int(token) <= 99):
            continue
        word_box = fitz.Rect(word[:4])
        if not box.contains(word_box):
            continue
        semantic_role, semantic_basis = _numeric_token_role(token, word_box, dimensions, text_roles)
        if semantic_role != "identifier_candidate":
            rejected.append(
                {
                    "text": token,
                    "bbox_display": list(word_box),
                    "state": "rejected",
                    "reason": semantic_basis,
                }
            )
            continue
        trace = trace_leader(
            word_box,
            thin,
            connection_tolerance=1.8,
            search_radius=150,
            max_hops=9,
            continue_through_intersections=True,
        )
        matches, contact = _first_rebar_contact(word_box, trace, candidates)
        if matches:
            identities.append(
                {
                    "mark_text": token,
                    "text_bbox_display": list(word_box),
                    "candidate_refs": sorted(set(matches)),
                    "leader_segment_refs": sorted({item["drawing_ref"] for item in trace["segments"]}),
                    "terminal_count": len(trace["terminals"]),
                    "semantic_role": semantic_role,
                    "semantic_basis": semantic_basis,
                    "status": "leader_connected_candidate_identity",
                    "targeting_method": "first_connected_heavy_geometry_contact",
                    "target_contact": contact,
                }
            )
        else:
            rejected.append(
                {
                    "text": token,
                    "bbox_display": list(word_box),
                    "state": "rejected",
                    "reason": (
                        "leader trace has no terminal on a unique rebar candidate"
                        if trace["segments"]
                        else "no native leader begins at the token"
                    ),
                    "leader_trace": trace,
                }
            )
    return identities, rejected


def _first_rebar_contact(
    word_box: fitz.Rect,
    trace: dict[str, Any],
    candidates: list[dict[str, Any]],
    connection_tolerance: float = 1.8,
) -> tuple[list[str], dict[str, Any] | None]:
    """Stop a leader at its first connected heavy-geometry contact.

    Continuing through butt joints and intersections is necessary for broken
    PDF leaders, but the resulting trace can enter a dense reinforcement grid
    and expose many unrelated distant leaves.  Resolve that ambiguity using
    geodesic distance from the mark token over the traced thin-line graph and
    stop at the first contact with a heavy rebar candidate.  Coincident or
    paired PDF primitives at that one contact remain together as a coherent
    target set.
    """

    points: list[tuple[float, float]] = []
    adjacency: list[list[tuple[int, float]]] = []

    def node_id(point: list[float]) -> int:
        row = (float(point[0]), float(point[1]))
        for index, existing in enumerate(points):
            if hypot(row[0] - existing[0], row[1] - existing[1]) <= connection_tolerance:
                return index
        points.append(row)
        adjacency.append([])
        return len(points) - 1

    for segment in trace.get("segments", []):
        left = node_id(segment["start"])
        right = node_id(segment["end"])
        length = hypot(points[left][0] - points[right][0], points[left][1] - points[right][1])
        if length <= 0:
            continue
        adjacency[left].append((right, length))
        adjacency[right].append((left, length))
    if not points:
        return [], None

    seed_tolerance = max(12.0, float(trace.get("seed_distance_pt") or 0.0) + 2.0)
    seeds = [
        index
        for index, point in enumerate(points)
        if _rect_distance(
            fitz.Rect(point[0] - 0.25, point[1] - 0.25, point[0] + 0.25, point[1] + 0.25),
            word_box,
        )
        <= seed_tolerance
    ]
    if not seeds:
        return [], None
    distances = [float("inf")] * len(points)
    pending: list[tuple[float, int]] = []
    for seed in seeds:
        distances[seed] = 0.0
        heapq.heappush(pending, (0.0, seed))
    while pending:
        distance, index = heapq.heappop(pending)
        if distance != distances[index]:
            continue
        for target, length in adjacency[index]:
            candidate_distance = distance + length
            if candidate_distance < distances[target]:
                distances[target] = candidate_distance
                heapq.heappush(pending, (candidate_distance, target))

    contacts = []
    for index, point in enumerate(points):
        if distances[index] == float("inf"):
            continue
        point_box = fitz.Rect(point[0] - 0.5, point[1] - 0.5, point[0] + 0.5, point[1] + 0.5)
        for candidate in candidates:
            display_distance = _rect_distance(point_box, fitz.Rect(candidate["bbox_display"]))
            if display_distance <= 8.0:
                contacts.append(
                    {
                        "total_distance_pt": distances[index] + display_distance,
                        "path_distance_pt": distances[index],
                        "display_distance_pt": display_distance,
                        "point_display": [point[0], point[1]],
                        "primitive_ref": candidate["primitive_ref"],
                    }
                )
    if not contacts:
        return [], None
    contacts.sort(key=lambda item: (item["total_distance_pt"], item["primitive_ref"]))
    best = contacts[0]
    coherent = [
        item
        for item in contacts
        if item["total_distance_pt"] <= best["total_distance_pt"] + 1.5
    ]
    refs = sorted({item["primitive_ref"] for item in coherent})
    return refs, {
        "point_display": best["point_display"],
        "path_distance_pt": round(float(best["path_distance_pt"]), 4),
        "display_distance_pt": round(float(best["display_distance_pt"]), 4),
        "coherent_primitive_count": len(refs),
    }


def _paired_target_units(candidates: list[dict[str, Any]]) -> list[list[str]]:
    """Collapse nearby outline strokes into candidate physical target units."""

    rows = []
    for item in candidates:
        box = item.get("bbox_display") or []
        if len(box) != 4:
            continue
        width, height = box[2] - box[0], box[3] - box[1]
        if max(width, height) < 8.0 or max(width, height) / max(min(width, height), 0.5) < 6.0:
            continue
        rows.append(
            {
                "ref": item["primitive_ref"],
                "box": box,
                "vertical": height >= width,
                "center": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
            }
        )
    pairs: list[list[str]] = []
    used: set[int] = set()
    for index, left in enumerate(rows):
        if index in used:
            continue
        matches = []
        for other_index, right in enumerate(rows[index + 1 :], start=index + 1):
            if other_index in used or left["vertical"] != right["vertical"]:
                continue
            left_box, right_box = left["box"], right["box"]
            if left["vertical"]:
                overlap = max(0.0, min(left_box[3], right_box[3]) - max(left_box[1], right_box[1]))
                span = min(left_box[3] - left_box[1], right_box[3] - right_box[1])
                separation = abs(left["center"][0] - right["center"][0])
            else:
                overlap = max(0.0, min(left_box[2], right_box[2]) - max(left_box[0], right_box[0]))
                span = min(left_box[2] - left_box[0], right_box[2] - right_box[0])
                separation = abs(left["center"][1] - right["center"][1])
            if span > 0 and overlap / span >= 0.80 and 0.5 <= separation <= 8.0:
                matches.append((separation, other_index, right))
        if matches:
            _, other_index, right = min(matches)
            used.update((index, other_index))
            pairs.append(sorted((left["ref"], right["ref"])))

    grouped = {ref for pair in pairs for ref in pair}
    return pairs + [[item["primitive_ref"]] for item in candidates if item["primitive_ref"] not in grouped]


def _resolve_unique_occurrence_targets(
    identities: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> None:
    """Resolve overlapping leader contacts by a unique injective assignment.

    A continued leader trace can touch the neighbouring outlined bar before it
    reaches its own pair.  Each printed callout occurrence still targets one
    physical contour unit.  We therefore solve the local bipartite graph of
    callout occurrences and paired contour units, and rewrite nothing unless
    that graph has exactly one injective solution.
    """

    units = _paired_target_units(candidates)
    unit_by_ref = {ref: index for index, refs in enumerate(units) for ref in refs}
    options = {
        index: sorted({unit_by_ref[ref] for ref in identity.get("candidate_refs", []) if ref in unit_by_ref})
        for index, identity in enumerate(identities)
    }
    pending = {index for index, rows in options.items() if rows}
    while pending:
        seed = min(pending)
        occurrence_ids = {seed}
        unit_ids = set(options[seed])
        changed = True
        while changed:
            changed = False
            for index in sorted(pending - occurrence_ids):
                if unit_ids.intersection(options[index]):
                    occurrence_ids.add(index)
                    unit_ids.update(options[index])
                    changed = True
        pending.difference_update(occurrence_ids)
        if len(unit_ids) < len(occurrence_ids):
            continue

        ordered = sorted(occurrence_ids, key=lambda index: (len(options[index]), index))
        solutions: list[dict[int, int]] = []

        def visit(position: int, used: set[int], assignment: dict[int, int]) -> None:
            if len(solutions) > 1:
                return
            if position == len(ordered):
                solutions.append(dict(assignment))
                return
            occurrence = ordered[position]
            for unit in options[occurrence]:
                if unit in used:
                    continue
                assignment[occurrence] = unit
                visit(position + 1, used | {unit}, assignment)
                assignment.pop(occurrence, None)

        visit(0, set(), {})
        if len(solutions) != 1:
            continue
        for occurrence, unit in solutions[0].items():
            identity = identities[occurrence]
            original = list(identity.get("candidate_refs", []))
            identity["candidate_refs"] = list(units[unit])
            identity["targeting_method"] = "unique_global_leader_occurrence_matching"
            identity["target_assignment_certificate"] = {
                "status": "pass",
                "candidate_unit_count": len(unit_ids),
                "callout_occurrence_count": len(occurrence_ids),
                "original_candidate_refs": original,
                "assigned_candidate_refs": list(units[unit]),
                "basis": "unique injective callout-occurrence to paired-contour assignment",
            }


def extract_section_rebar(
    page: fitz.Page,
    regions: tuple[RegionProposal, ...],
    dimensions: tuple[DimensionAttachment, ...] | None = None,
    text_roles: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return exact native rebar candidates for every section component."""

    if dimensions is None:
        dimensions = tuple(attach_dimensions(page))
    output = []
    for region in (item for item in regions if item.kind == "section_view"):
        raw_candidates = _candidate_paths(page, region)
        try:
            host, host_method = _section_host_bbox(page, region, dimensions, raw_candidates)
        except ValueError as error:
            output.append(
                {
                    "section_id": region.proposal_id,
                    "label": region.label,
                    "status": "unresolved",
                    "reason": str(error),
                    "candidates": [],
                    "leader_identities": [],
                }
            )
            continue
        selected = []
        for candidate in raw_candidates:
            rect = fitz.Rect(candidate["bbox_display"])
            center = fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
            if center not in host + (-5, -5, 5, 5):
                continue
            if max(rect.width, rect.height) > 1.05 * max(host.width, host.height):
                continue
            candidate = dict(candidate)
            candidate["candidate_id"] = f"{region.proposal_id}.rebar.{len(selected) + 1:03d}"
            candidate["kind"] = "filled_end_projection" if candidate["filled"] and candidate["compact"] else "native_rebar_path"
            selected.append(candidate)
        identities, rejected_tokens = _leader_identities(page, region, selected, dimensions, text_roles)
        _resolve_unique_occurrence_targets(identities, selected)
        marks_by_ref: dict[str, set[str]] = {}
        for identity in identities:
            for ref in identity["candidate_refs"]:
                marks_by_ref.setdefault(ref, set()).add(identity["mark_text"])
        for candidate in selected:
            candidate["leader_marks"] = sorted(marks_by_ref.get(candidate["primitive_ref"], set()), key=int)
            candidate["state"] = "resolved" if candidate["leader_marks"] else "candidate"
            candidate["reason"] = (
                "native leader terminal and exclusive mark token"
                if candidate["leader_marks"]
                else "rebar-like native geometry inside a dimensioned section; physical identity unresolved"
            )
        output.append(
            {
                "section_id": region.proposal_id,
                "label": region.label,
                "status": "observed_candidates",
                "host_bbox_display": list(host),
                "host_method": host_method,
                "candidate_count": len(selected),
                "filled_projection_count": sum(item["kind"] == "filled_end_projection" for item in selected),
                "leader_identity_count": len(identities),
                "candidates": selected,
                "leader_identities": identities,
                "rejected_numeric_tokens": rejected_tokens,
                "provenance": {"mode": "derived", "method": "section_host_containment_and_native_style"},
            }
        )
    return {
        "sections": output,
        "resolved_section_count": sum(item["status"] == "observed_candidates" for item in output),
        "candidate_count": sum(item.get("candidate_count", 0) for item in output),
        "identity_count": sum(item.get("leader_identity_count", 0) for item in output),
        "scope": "native section rebar candidates; topology and fabrication identity remain separate claims",
    }
