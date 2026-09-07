"""Vector-first leader tracing and cross-view identity graphs.

This module deliberately separates two claims:

* a leader trace connects an explicit mark token to source geometry; and
* a cross-view identity edge connects that trace to the same mark's detail cell.

Fabrication-rule transfer is evaluated separately.  A bar can therefore have a
high-confidence within-sheet identity while a K1 length formula is rejected for
K7 because the topology changed.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import fitz


@dataclass(frozen=True)
class VectorSegment:
    start: tuple[float, float]
    end: tuple[float, float]
    drawing_index: int
    item_index: int
    width: float

    @property
    def length(self) -> float:
        return math.dist(self.start, self.end)

    def as_dict(self) -> dict:
        return {
            "start": list(self.start),
            "end": list(self.end),
            "drawing_ref": f"drawing[{self.drawing_index}].item[{self.item_index}]",
            "width_pt": self.width,
            "length_pt": self.length,
        }


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return math.dist(point, start)
    fraction = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    closest = (ax + fraction * dx, ay + fraction * dy)
    return math.dist(point, closest)


def _segment_in_window(segment: VectorSegment, window: fitz.Rect) -> bool:
    xs = (segment.start[0], segment.end[0])
    ys = (segment.start[1], segment.end[1])
    return max(xs) >= window.x0 and min(xs) <= window.x1 and max(ys) >= window.y0 and min(ys) <= window.y1


def _segment_angle(segment: VectorSegment) -> float:
    return math.degrees(
        math.atan2(segment.end[1] - segment.start[1], segment.end[0] - segment.start[0])
    ) % 180.0


def _segments_connect(
    left: VectorSegment,
    right: VectorSegment,
    tolerance: float,
) -> bool:
    """Connect endpoint, butt-joint, and short collinear-gap leader pieces.

    Native CAD exports frequently split a leader where it meets another line or
    leave a sub-point plotting gap.  Interior-to-interior crossings remain
    disconnected because a graphical crossing alone is not evidence of a
    leader junction.
    """

    endpoint_gap = min(
        math.dist(left.start, right.start),
        math.dist(left.start, right.end),
        math.dist(left.end, right.start),
        math.dist(left.end, right.end),
    )
    if endpoint_gap <= tolerance:
        return True
    endpoint_to_segment = min(
        _point_segment_distance(left.start, right.start, right.end),
        _point_segment_distance(left.end, right.start, right.end),
        _point_segment_distance(right.start, left.start, left.end),
        _point_segment_distance(right.end, left.start, left.end),
    )
    if endpoint_to_segment <= tolerance:
        return True
    angle_delta = abs(_segment_angle(left) - _segment_angle(right))
    angle_delta = min(angle_delta, 180.0 - angle_delta)
    return angle_delta <= 2.0 and endpoint_gap <= max(3.5, 2.5 * tolerance)


def extract_thin_segments(page: fitz.Page, *, max_width: float = 0.65) -> list[VectorSegment]:
    """Return display-space leader candidates from native PDF line items."""

    matrix = page.rotation_matrix
    segments: list[VectorSegment] = []
    seen: set[tuple[float, ...]] = set()
    for drawing_index, drawing in enumerate(page.get_drawings()):
        width = drawing.get("width")
        if width is None or width > max_width:
            continue
        for item_index, item in enumerate(drawing["items"]):
            if item[0] != "l":
                continue
            start_point = item[1] * matrix
            end_point = item[2] * matrix
            start = (float(start_point.x), float(start_point.y))
            end = (float(end_point.x), float(end_point.y))
            length = math.dist(start, end)
            if not 2.0 <= length <= 140.0:
                continue
            ordered = tuple(round(value, 2) for point in sorted((start, end)) for value in point)
            if ordered in seen:
                continue
            seen.add(ordered)
            segments.append(VectorSegment(start, end, drawing_index, item_index, float(width)))
    return segments


def _cluster_endpoints(segments: list[VectorSegment], tolerance: float = 1.5) -> list[dict]:
    clusters: list[dict] = []
    for segment_index, segment in enumerate(segments):
        for point in (segment.start, segment.end):
            match = next(
                (
                    cluster
                    for cluster in clusters
                    if math.dist(point, cluster["center"]) <= tolerance
                ),
                None,
            )
            if match is None:
                clusters.append({"points": [point], "segments": {segment_index}, "center": point})
                continue
            match["points"].append(point)
            match["segments"].add(segment_index)
            match["center"] = (
                sum(item[0] for item in match["points"]) / len(match["points"]),
                sum(item[1] for item in match["points"]) / len(match["points"]),
            )
    return clusters


def infer_terminals(
    segments: list[VectorSegment],
    label_center: tuple[float, float],
) -> tuple[list[tuple[float, float]], str]:
    """Find arrow tips; use distant leaves only when arrowheads are absent."""

    clusters = _cluster_endpoints(segments)
    arrow_tips: list[tuple[float, float]] = []
    for cluster in clusters:
        incident = [segments[index] for index in cluster["segments"]]
        short_count = sum(segment.length <= 7.0 for segment in incident)
        long_count = sum(segment.length > 10.0 for segment in incident)
        if short_count >= 2 and long_count >= 1:
            arrow_tips.append(cluster["center"])
    if arrow_tips:
        return sorted(arrow_tips, key=lambda point: (point[1], point[0])), "arrowhead"

    leaves = [
        cluster["center"]
        for cluster in clusters
        if len(cluster["segments"]) == 1 and math.dist(cluster["center"], label_center) > 25.0
    ]
    leaves.sort(key=lambda point: math.dist(point, label_center), reverse=True)
    return sorted(leaves[:4], key=lambda point: (point[1], point[0])), "distant_leaf"


def trace_leader(
    label_box: fitz.Rect,
    segments: list[VectorSegment],
    *,
    connection_tolerance: float = 1.35,
    search_radius: float = 150.0,
    max_hops: int = 6,
    continue_through_intersections: bool = False,
) -> dict:
    """Trace the connected thin-line component nearest an explicit mark token."""

    center = ((label_box.x0 + label_box.x1) / 2, (label_box.y0 + label_box.y1) / 2)
    window = fitz.Rect(
        label_box.x0 - search_radius,
        label_box.y0 - search_radius,
        label_box.x1 + search_radius,
        label_box.y1 + search_radius,
    )
    local = [segment for segment in segments if _segment_in_window(segment, window)]
    if not local:
        return {"segments": [], "terminals": [], "terminal_method": "none", "seed_distance_pt": None}

    label_window = label_box + (-1.5, -1.5, 1.5, 1.5)
    seed_candidates = [
        segment
        for segment in local
        if not (
            fitz.Point(*segment.start) in label_window
            and fitz.Point(*segment.end) in label_window
            and segment.length <= max(18.0, 1.25 * label_box.height)
        )
    ]
    if not seed_candidates:
        return {"segments": [], "terminals": [], "terminal_method": "none", "seed_distance_pt": None}
    seed = min(seed_candidates, key=lambda segment: _point_segment_distance(center, segment.start, segment.end))
    seed_distance = _point_segment_distance(center, seed.start, seed.end)
    if seed_distance > 16.0:
        return {"segments": [], "terminals": [], "terminal_method": "none", "seed_distance_pt": seed_distance}

    chosen = [seed]
    queue: deque[tuple[VectorSegment, int]] = deque([(seed, 0)])
    seen = {(seed.drawing_index, seed.item_index)}
    while queue:
        current, hops = queue.popleft()
        if hops >= max_hops:
            continue
        for candidate in local:
            key = (candidate.drawing_index, candidate.item_index)
            if key in seen:
                continue
            endpoint_gap = min(
                math.dist(current.start, candidate.start),
                math.dist(current.start, candidate.end),
                math.dist(current.end, candidate.start),
                math.dist(current.end, candidate.end),
            )
            connected = (
                _segments_connect(current, candidate, connection_tolerance)
                if continue_through_intersections
                else endpoint_gap <= connection_tolerance
            )
            if connected:
                seen.add(key)
                chosen.append(candidate)
                queue.append((candidate, hops + 1))

    terminals, terminal_method = infer_terminals(chosen, center)
    return {
        "segments": [segment.as_dict() for segment in chosen],
        "terminals": [list(point) for point in terminals],
        "terminal_method": terminal_method,
        "seed_distance_pt": seed_distance,
    }


def _view_for_box(box: fitz.Rect, view_boxes: Iterable[tuple[str, fitz.Rect]]) -> str:
    center = fitz.Point((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)
    matches = [(name, rect) for name, rect in view_boxes if center in rect]
    if not matches:
        return "unclassified_view"
    return min(matches, key=lambda item: item[1].get_area())[0]


def build_cross_view_graph(
    page: fitz.Page,
    *,
    sheet_key: str,
    detail_boxes: dict[str, fitz.Rect],
    view_boxes: Iterable[tuple[str, fitz.Rect]],
    marks: tuple[str, ...] = ("7", "8"),
) -> dict:
    """Build an auditable M7/M8 identity graph from native text and vectors."""

    thin_segments = extract_thin_segments(page)
    nodes: list[dict] = []
    edges: list[dict] = []
    mark_occurrences: dict[str, list[dict]] = {mark: [] for mark in marks}

    for mark in marks:
        detail = detail_boxes[mark]
        nodes.append(
            {
                "id": f"cross_view.mark_{mark}.detail_sketch",
                "mark": mark,
                "kind": "detail_sketch",
                "view": "view.bar_detail_sketches",
                "bbox_display": list(detail),
                "status": "observed",
            }
        )

    for word in page.get_text("words"):
        mark = word[4].strip()
        if mark not in marks:
            continue
        box = fitz.Rect(word[:4]) * page.rotation_matrix
        if box.x0 >= 1620:
            continue
        if 1340 <= box.x0 <= 1420 and box.y1 <= 1325:
            # The table row number is represented by the exact detail-cell node.
            continue
        view = _view_for_box(box, view_boxes)
        trace = trace_leader(box, thin_segments)
        occurrence_index = len(mark_occurrences[mark]) + 1
        node_id = f"cross_view.mark_{mark}.callout.{occurrence_index:02d}"
        node = {
            "id": node_id,
            "mark": mark,
            "kind": "projection_callout",
            "view": view,
            "bbox_display": list(box),
            "status": "observed" if trace["segments"] else "unknown",
            "leader_trace": trace,
        }
        nodes.append(node)
        mark_occurrences[mark].append(node)

    for mark in marks:
        detail_id = f"cross_view.mark_{mark}.detail_sketch"
        for occurrence_index, node in enumerate(mark_occurrences[mark], start=1):
            trace = node["leader_trace"]
            factors = {
                "explicit_mark_token": 0.45,
                "connected_native_leader": 0.25 if trace["segments"] else 0.0,
                "leader_terminal": 0.10 if trace["terminals"] else 0.0,
                "same_mark_detail_cell": 0.10,
                "recognized_drawing_view": 0.10 if node["view"] != "unclassified_view" else 0.0,
            }
            score = sum(factors.values())
            edges.append(
                {
                    "id": f"cross_view.mark_{mark}.identity.{occurrence_index:02d}",
                    "mark": mark,
                    "source": detail_id,
                    "target": node["id"],
                    "status": "accepted" if score >= 0.85 else "review",
                    "score": round(score, 3),
                    "factors": factors,
                    "note": "Same-mark detail-to-projection identity; fabrication formula transfer is a separate decision.",
                }
            )

    transfer_checks = [
        {
            "id": "cross_view.mark_7.k1_rule_transfer",
            "mark": "7",
            "status": "accepted",
            "score": 0.95,
            "evidence": ["same U-bar topology", "same diameter", "same dimensioned legs"],
            "note": "K1 M7 one-piece length rule remains geometrically compatible. Count is checked separately.",
        },
        {
            "id": "cross_view.mark_8.k1_rule_transfer",
            "mark": "8",
            "status": "accepted" if sheet_key == "k1" else "rejected",
            "score": 1.0 if sheet_key == "k1" else 0.35,
            "evidence": (
                ["development-sheet topology", "diameter and three-piece identity"]
                if sheet_key == "k1"
                else ["same mark token", "same diameter", "same count", "asymmetric topology conflict"]
            ),
            "note": (
                "K1 symmetric M8 rule is internally compatible."
                if sheet_key == "k1"
                else "Within-sheet M8 identity is accepted, but the symmetric K1 length formula is rejected."
            ),
        },
    ]
    return {
        "schema_version": "0.1.0",
        "method": "exact mark token + connected native thin-line component + arrowhead/leaf terminal inference",
        "thresholds": {"accepted": 0.85, "review": 0.60},
        "nodes": nodes,
        "identity_edges": edges,
        "fabrication_rule_transfer_checks": transfer_checks,
    }
