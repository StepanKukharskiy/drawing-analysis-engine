"""Dual-channel page-wide projected MEP route recovery.

Channel A replays independently certified parallel-outline corridors against
the current path-role pack.  Channel B reconstructs exact or strictly bounded
near-joined chains from anchored route-style authored paths.  Colour/style may
support a route candidate but never names its system; M4 remains the only
system/size/elevation authority.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from src.drawing_engine.disciplines.mep.mep_native_path_pack import NativePathPack
from src.drawing_engine.disciplines.mep.mep_page_wide_path_classification import (
    FLAG_ANCHORED_ROUTE_STYLE, FLAG_LEGACY_M3_CANDIDATE,
    PageWidePathRolePack,
)


SCHEMA_VERSION = "0.1.0"
LAYER = "mep_page_wide_projected_route_recovery"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"))


def _stable_id(kind: str, *parts: Any) -> str:
    return f"{kind}.{hashlib.sha256(_canonical(parts).encode()).hexdigest()[:20]}"


def _drawing_ordinal(source_ref: str) -> int:
    return int(source_ref.removeprefix("drawing[").split("]", 1)[0])


def _intervals(values: Iterable[int]) -> list[list[int]]:
    rows = sorted(set(values))
    if not rows:
        return []
    output = []
    start = end = rows[0]
    for value in rows[1:]:
        if value == end + 1:
            end = value
        else:
            output.append([start, end])
            start = end = value
    output.append([start, end])
    return output


def _near_text(point: Sequence[float], boxes: Sequence[Sequence[float]],
               factor: float = 1.25) -> bool:
    for box in boxes:
        height = max(1.0, float(box[3]) - float(box[1]))
        margin = factor * height
        if (box[0] - margin <= point[0] <= box[2] + margin
                and box[1] - margin <= point[1] <= box[3] + margin):
            return True
    return False


def _text_spatial_index(boxes: Sequence[Sequence[float]], *, cell_size: float = 64.0):
    cells: dict[tuple[int, int], list[Sequence[float]]] = defaultdict(list)
    for box in boxes:
        x0, y0 = math.floor(float(box[0]) / cell_size), math.floor(float(box[1]) / cell_size)
        x1, y1 = math.floor(float(box[2]) / cell_size), math.floor(float(box[3]) / cell_size)
        for x in range(x0 - 1, x1 + 2):
            for y in range(y0 - 1, y1 + 2):
                cells[(x, y)].append(box)
    return cells


def _near_indexed_text(point: Sequence[float], cells, *, cell_size: float = 64.0) -> bool:
    cell = (math.floor(float(point[0]) / cell_size),
            math.floor(float(point[1]) / cell_size))
    return _near_text(point, cells.get(cell, ()))


def _simplify(points: Sequence[Sequence[float]], tolerance: float) -> list[list[float]]:
    if len(points) <= 2:
        return [[round(float(x), 6), round(float(y), 6)] for x, y in points]
    output = [list(points[0])]
    for index, point in enumerate(points[1:-1], 1):
        a, b, c = output[-1], point, points[index + 1]
        ab = (b[0] - a[0], b[1] - a[1])
        bc = (c[0] - b[0], c[1] - b[1])
        cross = abs(ab[0] * bc[1] - ab[1] * bc[0])
        dot = ab[0] * bc[0] + ab[1] * bc[1]
        if cross <= tolerance * max(1.0, math.hypot(*ab), math.hypot(*bc)) and dot >= 0:
            continue
        output.append(list(point))
    output.append(list(points[-1]))
    return [[round(float(x), 6), round(float(y), 6)] for x, y in output]


@dataclass(slots=True)
class _Edge:
    path_ordinal: int
    drawing_ordinal: int
    style_id: int
    start: tuple[float, float]
    end: tuple[float, float]
    length: float
    source_segment_count: int
    legacy: bool
    oblique: bool


@dataclass(slots=True)
class _OutlineSide:
    path_ordinal: int
    drawing_ordinal: int
    first_descriptor_ordinal: int
    source_segment_count: int
    style_id: int
    start: tuple[float, float]
    end: tuple[float, float]
    length: float
    angle: float


def _directed_line(path: Mapping[str, Any]) -> tuple[
        tuple[float, float], tuple[float, float], float] | None:
    """Return one canonical straight authored path without naming its role."""
    if (path.get("kind_mask") != 1 or path.get("endpoint_closed")
            or path.get("style_id") is None):
        return None
    start = tuple(map(float, path["start_display"]))
    end = tuple(map(float, path["end_display"]))
    span = math.dist(start, end)
    length = float(path.get("source_segment_length_points") or 0.0)
    if span <= 1e-8 or length / span > 1.002:
        return None
    if end < start:
        start, end = end, start
    angle = math.atan2(end[1] - start[1], end[0] - start[0]) % math.pi
    return start, end, angle


def _outline_pair_metrics(left: _OutlineSide, right: _OutlineSide) -> dict[str, float] | None:
    angle_delta = abs(left.angle - right.angle)
    angle_delta = min(angle_delta, math.pi - angle_delta)
    if math.degrees(angle_delta) > .5:
        return None
    ux = math.cos((left.angle + right.angle) / 2.0)
    uy = math.sin((left.angle + right.angle) / 2.0)
    nx, ny = -uy, ux

    def project(point: tuple[float, float]) -> tuple[float, float]:
        return point[0] * ux + point[1] * uy, point[0] * nx + point[1] * ny

    left_start, left_end = project(left.start), project(left.end)
    right_start, right_end = project(right.start), project(right.end)
    if right_start[0] > right_end[0]:
        right_start, right_end = right_end, right_start
    if left_start[0] > left_end[0]:
        left_start, left_end = left_end, left_start
    along_residual = max(abs(left_start[0] - right_start[0]),
                         abs(left_end[0] - right_end[0]))
    separations = (abs(left_start[1] - right_start[1]),
                   abs(left_end[1] - right_end[1]))
    mean_separation = sum(separations) / 2.0
    separation_deviation = abs(separations[0] - separations[1])
    minimum_length = min(left.length, right.length)
    length_ratio = minimum_length / max(left.length, right.length, 1e-9)
    if (length_ratio < .98 or along_residual > .35
            or mean_separation <= .1
            or separation_deviation > max(.35, .08 * mean_separation)):
        return None
    return {
        "mean_separation_display_points": mean_separation,
        "maximum_separation_deviation_display_points": separation_deviation,
        "maximum_along_residual_display_points": along_residual,
        "maximum_tangent_difference_degrees": math.degrees(angle_delta),
        "minimum_length_display_points": minimum_length,
        "length_ratio": length_ratio,
    }


def discover_page_wide_outline_corridors(
    *, page_ref: str, path_pack: NativePathPack,
    role_pack: PageWidePathRolePack, styles: Sequence[Mapping[str, Any]],
    legacy_composites: Sequence[Mapping[str, Any]] = (),
    additional_style_ids: Iterable[int] = (),
    annotated_widths_by_style: Mapping[int, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Pair current authored paths without granting legacy M3 route authority.

    Duplicate paint paths are canonicalized before mutual-nearest pairing.  A
    pair establishes projected corridor geometry only; an anchored style can
    independently support MEP candidacy later, while M4 alone names a system.
    """
    sides = []
    additional_style_ids = set(map(int, additional_style_ids))
    annotated_widths_by_style = annotated_widths_by_style or {}
    allowed_roles = {
        "anchored_route_style_candidate",
        "outlined_corridor_source_candidate",
        "text_associated_stroke_candidate",
    }
    for path, role in zip(path_pack.records(), role_pack.records()):
        anchored = bool(role["flags"] & FLAG_ANCHORED_ROUTE_STYLE)
        additional = path.get("style_id") in additional_style_ids
        if not anchored and not additional:
            continue
        if anchored and role["role"] not in allowed_roles:
            continue
        if additional and role["role"] in {
                "excluded_non_view_content", "equipment_fitting_evidence",
                "annotation_dimension", "measured_hatch_candidate",
                "architectural_boundary_candidate", "drawing_furniture"}:
            continue
        directed = _directed_line(path)
        if directed is None:
            continue
        start, end, angle = directed
        width = float(styles[path["style_id"]].get("width") or .12)
        if float(path["source_segment_length_points"]) < max(2.0, 3.0 * width):
            continue
        sides.append(_OutlineSide(
            path_ordinal=int(path["path_ordinal"]),
            drawing_ordinal=int(path["drawing_ordinal"]),
            first_descriptor_ordinal=int(path["first_descriptor_ordinal"]),
            source_segment_count=int(path["source_segment_count"]),
            style_id=int(path["style_id"]), start=start, end=end,
            length=float(path["source_segment_length_points"]), angle=angle,
        ))

    # Repeated XObjects can paint one geometric side many times.  Keep every
    # source path but let one canonical side compete for a corridor pairing.
    duplicate_groups: dict[tuple[Any, ...], list[_OutlineSide]] = defaultdict(list)
    for side in sides:
        signature = (side.style_id, *(round(value, 3)
                                     for point in (side.start, side.end)
                                     for value in point))
        duplicate_groups[signature].append(side)
    canonical = [min(group, key=lambda row: row.path_ordinal)
                 for group in duplicate_groups.values()]
    duplicates_by_ordinal = {
        min(group, key=lambda row: row.path_ordinal).path_ordinal:
        sorted(row.path_ordinal for row in group)
        for group in duplicate_groups.values()
    }

    by_style: dict[int, list[int]] = defaultdict(list)
    for index, side in enumerate(canonical):
        by_style[side.style_id].append(index)
    alternatives: dict[int, list[tuple[float, int, dict[str, float]]]] = defaultdict(list)
    evaluated_pair_count = 0
    for style_id, indices in by_style.items():
        width = float(styles[style_id].get("width") or .12)
        cell_size = 64.0
        midpoint_cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index in indices:
            side = canonical[index]
            midpoint_cells[(
                math.floor((side.start[0] + side.end[0]) / 2.0 / cell_size),
                math.floor((side.start[1] + side.end[1]) / 2.0 / cell_size),
            )].append(index)
        for left_index in indices:
            left = canonical[left_index]
            midpoint = ((left.start[0] + left.end[0]) / 2.0,
                        (left.start[1] + left.end[1]) / 2.0)
            # Every accepted pair has <=.35 along residual and separation
            # bounded by this side's length.  This complete spatial window
            # only removes impossible comparisons; it changes no certificate.
            annotated_widths = annotated_widths_by_style.get(style_id, ())
            radius = max(12.0 * width, .08 * left.length,
                         max(annotated_widths, default=0.0) * 1.8) + 1.0
            possible = []
            for x in range(math.floor((midpoint[0] - radius) / cell_size),
                           math.floor((midpoint[0] + radius) / cell_size) + 1):
                for y in range(math.floor((midpoint[1] - radius) / cell_size),
                               math.floor((midpoint[1] + radius) / cell_size) + 1):
                    possible.extend(midpoint_cells.get((x, y), ()))
            for right_index in sorted(index for index in possible
                                      if index > left_index):
                right = canonical[right_index]
                metrics = _outline_pair_metrics(left, right)
                if metrics is None:
                    continue
                evaluated_pair_count += 1
                separation = metrics["mean_separation_display_points"]
                minimum_length = metrics["minimum_length_display_points"]
                annotated_width_match = any(
                    .55 * expected <= separation <= 1.8 * expected
                    for expected in annotated_widths)
                if ((separation > max(12.0 * width, .08 * minimum_length)
                     and not annotated_width_match)
                        or minimum_length < max(2.5 * separation, 3.0 * width, 2.0)):
                    continue
                score = (separation
                         + 4.0 * metrics["maximum_along_residual_display_points"]
                         + 2.0 * metrics["maximum_separation_deviation_display_points"])
                alternatives[left_index].append((score, right_index, metrics))
                alternatives[right_index].append((score, left_index, metrics))

    nearest: dict[int, tuple[int, dict[str, float]]] = {}
    ambiguous_side_count = 0
    for index, rows in alternatives.items():
        rows.sort(key=lambda row: (row[0], canonical[row[1]].path_ordinal))
        if len(rows) > 1 and rows[1][0] - rows[0][0] <= .05:
            ambiguous_side_count += 1
            continue
        nearest[index] = (rows[0][1], rows[0][2])

    legacy_by_drawings = {}
    for row in legacy_composites:
        drawings = tuple(sorted(_drawing_ordinal(ref) for ref in
                                row.get("member_source_primitive_refs", [])))
        if len(drawings) == 2:
            legacy_by_drawings[drawings] = row

    accepted_pairs = []
    paired = set()
    for left_index, (right_index, metrics) in sorted(nearest.items()):
        if left_index >= right_index or nearest.get(right_index, (None,))[0] != left_index:
            continue
        left, right = canonical[left_index], canonical[right_index]
        paired.update((left_index, right_index))
        left_duplicates = duplicates_by_ordinal[left.path_ordinal]
        right_duplicates = duplicates_by_ordinal[right.path_ordinal]
        legacy = next((legacy_by_drawings[key]
                       for key in legacy_by_drawings
                       if key[0] in {canonical[index].drawing_ordinal for index in (left_index,)}
                       and key[1] in {canonical[index].drawing_ordinal for index in (right_index,)}), None)
        # The common case above is exact; the explicit duplicate search keeps a
        # repeated projection from losing a previously bound M4 composite ID.
        if legacy is None:
            left_drawings = {side.drawing_ordinal for side in
                             duplicate_groups[(left.style_id, *(round(value, 3)
                                 for point in (left.start, left.end) for value in point))]}
            right_drawings = {side.drawing_ordinal for side in
                              duplicate_groups[(right.style_id, *(round(value, 3)
                                  for point in (right.start, right.end) for value in point))]}
            legacy = next((row for key, row in legacy_by_drawings.items()
                           if ((key[0] in left_drawings and key[1] in right_drawings)
                               or (key[1] in left_drawings and key[0] in right_drawings))), None)
        centreline = [[round((left.start[0] + right.start[0]) / 2.0, 6),
                       round((left.start[1] + right.start[1]) / 2.0, 6)],
                      [round((left.end[0] + right.end[0]) / 2.0, 6),
                       round((left.end[1] + right.end[1]) / 2.0, 6)]]
        if math.dist(centreline[0], centreline[1]) <= 1e-8:
            continue
        path_ordinals = sorted((*left_duplicates, *right_duplicates))
        descriptor_intervals = sorted([
            [side.first_descriptor_ordinal,
             side.first_descriptor_ordinal + side.source_segment_count - 1]
            for side in sides if side.path_ordinal in set(path_ordinals)
        ])
        accepted_pairs.append({
            "id": (legacy["id"] if legacy else _stable_id(
                "mep_page_wide_outline_corridor", page_ref, path_ordinals)),
            "record_type": "mep_page_wide_outline_corridor_candidate",
            "page_ref": page_ref,
            "state": "accepted_projected_corridor_geometry",
            "member_path_ordinals": path_ordinals,
            "canonical_side_path_ordinals": [left.path_ordinal, right.path_ordinal],
            "member_drawing_ordinals": sorted({
                side.drawing_ordinal for side in sides
                if side.path_ordinal in set(path_ordinals)}),
            "member_descriptor_ordinal_intervals": descriptor_intervals,
            "legacy_composite_ref": legacy["id"] if legacy else None,
            "derived_geometry": {
                "centreline_points_display": centreline,
                "corridor_width_display_points": round(
                    metrics["mean_separation_display_points"], 6),
                "projected_path_display_points": round(
                    math.dist(centreline[0], centreline[1]), 6),
            },
            "geometry_metrics": {key: round(value, 6)
                                 for key, value in metrics.items()},
            "method": {
                "name": "page_wide_mutually_unique_parallel_authored_paths",
                "version": "1.0.0",
            },
            "certificates": {
                "accepted_drawing_view_ownership": True,
                "anchored_route_style_correlation": (
                    left.style_id not in additional_style_ids),
                "annotation_nominated_neutral_style": (
                    left.style_id in additional_style_ids),
                "straight_path_geometry": True,
                "persistent_parallel_separation": True,
                "synchronized_endpoints": True,
                "mutual_unique_pairing": True,
                "duplicate_paint_paths_canonicalized": True,
                "legacy_M3_used_as_route_authority": False,
                "annotation_supported_width_search": any(
                    .55 * expected <= metrics["mean_separation_display_points"] <= 1.8 * expected
                    for expected in annotated_widths_by_style.get(left.style_id, ())),
            },
            "route_identity_established": False,
            "system_identity_established": False,
            "quantity_eligible": False,
        })
    accepted_pairs.sort(key=lambda row: (row["member_path_ordinals"], row["id"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "mep_page_wide_outline_corridor_discovery",
        "page_ref": page_ref,
        "accepted_corridors": accepted_pairs,
        "summary": {
            "eligible_authored_path_count": len(sides),
            "canonical_geometric_side_count": len(canonical),
            "duplicate_paint_path_count": len(sides) - len(canonical),
            "geometrically_evaluated_pair_count": evaluated_pair_count,
            "mutually_unique_corridor_count": len(accepted_pairs),
            "paired_canonical_side_count": len(paired),
            "unpaired_canonical_side_count": len(canonical) - len(paired),
            "ambiguous_canonical_side_count": ambiguous_side_count,
            "legacy_composite_match_count": sum(
                row["legacy_composite_ref"] is not None for row in accepted_pairs),
        },
        "authority": {
            "legacy_M3_controls_candidate_existence": False,
            "projected_corridor_geometry_established": True,
            "route_identity_established": False,
            "system_identity_established": False,
            "quantity_eligible": False,
        },
    }


def nominate_annotation_outline_styles(
    *, path_pack: NativePathPack, role_pack: PageWidePathRolePack,
    styles: Sequence[Mapping[str, Any]],
    route_annotations: Sequence[Mapping[str, Any]],
    drawing_inches_per_paper_inch: float,
    anchored_style_ids: Iterable[int] = (),
) -> dict[str, Any]:
    """Nominate neutral authored styles without granting them route status.

    Legacy M3 membership is used only to bound the style search.  A nomination
    also requires a current mutually parallel path pair, a drawing-scale size
    match, accepted-view ownership, and proximity to an explicit native route
    annotation.  Full current paths of nominated styles are replayed later.
    """
    anchored_style_ids = set(map(int, anchored_style_ids))
    sides = []
    coloured_current_sides = []
    blocked_roles = {
        "excluded_non_view_content", "equipment_fitting_evidence",
        "annotation_dimension", "measured_hatch_candidate",
        "architectural_boundary_candidate", "drawing_furniture",
    }
    for path, role in zip(path_pack.records(), role_pack.records()):
        if path.get("style_id") in anchored_style_ids:
            continue
        if role["role"] in blocked_roles:
            continue
        directed = _directed_line(path)
        if directed is None or float(path["source_segment_length_points"]) < 12.0:
            continue
        start, end, angle = directed
        side = _OutlineSide(
            path_ordinal=int(path["path_ordinal"]),
            drawing_ordinal=int(path["drawing_ordinal"]),
            first_descriptor_ordinal=int(path["first_descriptor_ordinal"]),
            source_segment_count=int(path["source_segment_count"]),
            style_id=int(path["style_id"]), start=start, end=end,
            length=float(path["source_segment_length_points"]), angle=angle,
        )
        if role["flags"] & FLAG_LEGACY_M3_CANDIDATE:
            sides.append(side)
        stroke = styles[side.style_id].get("stroke")
        if (stroke is not None
                and max(map(float, stroke)) - min(map(float, stroke)) >= .25):
            coloured_current_sides.append(side)

    def box_gap(side: _OutlineSide, box: Sequence[float]) -> float:
        x0, x1 = sorted((side.start[0], side.end[0]))
        y0, y1 = sorted((side.start[1], side.end[1]))
        dx = max(x0 - float(box[2]), float(box[0]) - x1, 0.0)
        dy = max(y0 - float(box[3]), float(box[1]) - y1, 0.0)
        return math.hypot(dx, dy)

    certificates = []
    current_style_evidence: dict[int, list[dict[str, Any]]] = defaultdict(list)
    nominated = set()
    scale = float(drawing_inches_per_paper_inch)
    if scale <= 0:
        raise ValueError("drawing scale must be positive")
    for annotation in route_annotations:
        nominal_inches = annotation.get("nominal_size_inches")
        box = annotation.get("bbox_display")
        if not isinstance(nominal_inches, (int, float)) or not box:
            continue
        expected_width = float(nominal_inches) / scale * 72.0
        local = [side for side in sides if box_gap(side, box) <= 100.0]
        candidates = []
        for position, left in enumerate(local):
            for right in local[position + 1:]:
                if left.style_id != right.style_id:
                    continue
                metrics = _outline_pair_metrics(left, right)
                if metrics is None:
                    continue
                separation = metrics["mean_separation_display_points"]
                if not (.55 * expected_width <= separation <= 1.8 * expected_width):
                    continue
                if metrics["minimum_length_display_points"] < max(
                        12.0, 3.0 * separation):
                    continue
                gap = min(box_gap(left, box), box_gap(right, box))
                score = gap + 12.0 * abs(math.log(separation / expected_width))
                candidates.append((score, left.style_id, left, right, metrics))
        candidates.sort(key=lambda row: (row[0], row[1], row[2].path_ordinal,
                                         row[3].path_ordinal))
        if candidates:
            best = candidates[0]
            competing_styles = {
                row[1] for row in candidates[1:] if row[0] - best[0] <= 3.0}
            if not competing_styles - {best[1]}:
                nominated.add(best[1])
                certificates.append({
                    "id": _stable_id("mep_annotation_outline_style", annotation.get("id"),
                                     best[1], best[2].path_ordinal,
                                     best[3].path_ordinal),
                    "state": "accepted_style_nomination",
                    "annotation_ref": annotation.get("id"),
                    "system_candidate": annotation.get("system_candidate"),
                    "nominal_size_inches": nominal_inches,
                    "style_id": best[1],
                    "candidate_side_path_ordinals": sorted(
                        (best[2].path_ordinal, best[3].path_ordinal)),
                    "annotation_gap_display_points": round(
                        min(box_gap(best[2], box), box_gap(best[3], box)), 6),
                    "expected_corridor_width_display_points": round(expected_width, 6),
                    "observed_corridor_width_display_points": round(
                        best[4]["mean_separation_display_points"], 6),
                    "legacy_M3_used_only_to_bound_style_search": True,
                    "route_identity_established": False,
                    "system_identity_established": False,
                    "quantity_eligible": False,
                })

        # A second, candidate-only channel prevents a missing legacy-M3 bit
        # from hiding a currently visible coloured pipe family.  It does not
        # choose one style or name one system: every retained style must have
        # its own current parallel-pair, drawing-scale width, accepted-view,
        # and annotation-proximity evidence.  Ambiguous supply/return styles
        # remain a dashed combined family in the audit.
        current_local = [side for side in coloured_current_sides
                         if box_gap(side, box) <= 110.0]
        by_style: dict[int, tuple[float, _OutlineSide, _OutlineSide,
                                  dict[str, float]]] = {}
        for position, left in enumerate(current_local):
            for right in current_local[position + 1:]:
                if left.style_id != right.style_id:
                    continue
                metrics = _outline_pair_metrics(left, right)
                if metrics is None:
                    continue
                separation = metrics["mean_separation_display_points"]
                if not (.55 * expected_width <= separation <= 1.8 * expected_width):
                    continue
                if metrics["minimum_length_display_points"] < max(
                        12.0, 3.0 * separation):
                    continue
                gap = min(box_gap(left, box), box_gap(right, box))
                score = gap + 12.0 * abs(math.log(separation / expected_width))
                previous = by_style.get(left.style_id)
                if previous is None or score < previous[0]:
                    by_style[left.style_id] = (score, left, right, metrics)
        for style_id, (score, left, right, metrics) in by_style.items():
            current_style_evidence[style_id].append({
                "annotation_ref": annotation.get("id"),
                "system_candidate": annotation.get("system_candidate"),
                "nominal_size_inches": float(nominal_inches),
                "candidate_side_path_ordinals": sorted(
                    (left.path_ordinal, right.path_ordinal)),
                "score": round(score, 6),
                "annotation_gap_display_points": round(
                    min(box_gap(left, box), box_gap(right, box)), 6),
                "observed_corridor_width_display_points": round(
                    metrics["mean_separation_display_points"], 6),
            })
    style_groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for certificate in certificates:
        style_groups[int(certificate["style_id"])].append(certificate)
    style_correlations = []
    for style_id, rows in sorted(style_groups.items()):
        systems = sorted({str(row["system_candidate"]) for row in rows
                          if row.get("system_candidate")})
        sizes = sorted({float(row["nominal_size_inches"]) for row in rows
                        if isinstance(row.get("nominal_size_inches"), (int, float))})
        if len(rows) < 3 or len(systems) != 1 or len(sizes) != 1:
            continue
        style_correlations.append({
            "id": _stable_id("mep_annotation_style_correlation", style_id,
                             systems, sizes, [row["id"] for row in rows]),
            "state": "supported_route_style_correlation",
            "style_id": style_id,
            "system_candidates": systems,
            "nominal_size_candidates_inches": sizes,
            "annotation_certificate_refs": sorted(row["id"] for row in rows),
            "minimum_independent_annotations": 3,
            "system_identity_established": False,
            "quantity_eligible": False,
        })
    existing_correlation_styles = {int(row["style_id"])
                                   for row in style_correlations}
    for style_id, evidence in sorted(current_style_evidence.items()):
        if style_id in existing_correlation_styles:
            continue
        annotation_refs = sorted({str(row["annotation_ref"])
                                  for row in evidence})
        systems = sorted({str(row["system_candidate"]) for row in evidence
                          if row.get("system_candidate")})
        sizes = sorted({float(row["nominal_size_inches"]) for row in evidence})
        families = {value.removesuffix("_supply").removesuffix("_return")
                    for value in systems}
        if len(annotation_refs) < 2 or len(sizes) != 1 or len(families) != 1:
            continue
        nominated.add(style_id)
        style_correlations.append({
            "id": _stable_id("mep_ambiguous_annotation_style_correlation",
                             style_id, systems, sizes, annotation_refs),
            "state": "supported_route_style_correlation",
            "style_id": style_id,
            "system_candidates": systems,
            "nominal_size_candidates_inches": sizes,
            "annotation_certificate_refs": annotation_refs,
            "current_pair_evidence": sorted(
                evidence, key=lambda row: (row["annotation_ref"], row["score"])),
            "minimum_independent_annotations": 2,
            "style_selection_unique": False,
            "system_identity_established": False,
            "quantity_eligible": False,
        })
    style_correlations.sort(key=lambda row: (int(row["style_id"]), row["id"]))
    return {
        "nominated_style_ids": sorted(nominated),
        "certificates": certificates,
        "supported_style_correlations": style_correlations,
        "summary": {
            "route_annotation_count": len(route_annotations),
            "accepted_style_nomination_count": len(certificates),
            "nominated_neutral_style_count": len(nominated),
            "supported_style_correlation_count": len(style_correlations),
        },
    }


def _join_tolerance(style: Mapping[str, Any]) -> float:
    width = float(style.get("width") or .12)
    # The 0.12 upper bound is the observed coordinate quantum on this source;
    # width only tightens it.  This is per style and never a global snap.
    return round(min(.12, max(.02, width / 6.0)), 6)


def _attribute_values(relations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output = {}
    for relation_type in ("route_system", "route_size", "route_elevation"):
        rows = [row for row in relations
                if row.get("state") == "accepted"
                and row.get("relation_type") == relation_type]
        values = {}
        for row in rows:
            candidate = row.get("candidate", {})
            value = (candidate.get("kind") if relation_type == "route_system"
                     else candidate.get("value") or candidate.get("raw_text"))
            values[_canonical(value)] = value
        output[relation_type] = {
            "state": "accepted" if len(values) == 1 else "unknown",
            "value": next(iter(values.values())) if len(values) == 1 else None,
            "relation_refs": sorted(row["id"] for row in rows),
            "reason": None if len(values) == 1 else (
                "conflicting_M4_values" if values else "no_accepted_M4_relation"),
        }
    return output


def _observed_connector_candidates(
    *, page_ref: str, route_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Classify only endpoint-incidence L and T geometry.

    This is an observed projected connector class, not a physical fitting,
    connection standard, SKU, deduplicated count, or purchase quantity.
    """
    endpoints = []
    for row in route_rows:
        if row.get("state") not in {
                "identified_mep_route", "supported_unidentified_mep_candidate"}:
            continue
        points = row.get("polyline_display") or []
        if len(points) < 2:
            continue
        for end_index, (point, inner) in enumerate(
                ((points[0], points[1]), (points[-1], points[-2]))):
            vector = (float(inner[0]) - float(point[0]),
                      float(inner[1]) - float(point[1]))
            magnitude = math.hypot(*vector)
            if magnitude <= 1e-8:
                continue
            endpoints.append({
                "component_ref": row["id"], "end_index": end_index,
                "point": tuple(map(float, point)),
                "direction": (vector[0] / magnitude, vector[1] / magnitude),
                "width": float(row.get("corridor_width_display_points") or 1.0),
            })
    if not endpoints:
        return []

    parent = list(range(len(endpoints)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    cell_size = 8.0
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, endpoint in enumerate(endpoints):
        point = endpoint["point"]
        cell = (math.floor(point[0] / cell_size), math.floor(point[1] / cell_size))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in cells.get((cell[0] + dx, cell[1] + dy), ()):
                    if endpoints[other]["component_ref"] == endpoint["component_ref"]:
                        continue
                    tolerance = min(8.0, max(3.0, .75 * (
                        endpoint["width"] + endpoints[other]["width"])))
                    if math.dist(point, endpoints[other]["point"]) <= tolerance:
                        union(index, other)
        cells[cell].append(index)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(endpoints)):
        groups[find(index)].append(index)

    def angle(left: Sequence[float], right: Sequence[float]) -> float:
        dot = max(-1.0, min(1.0, left[0] * right[0] + left[1] * right[1]))
        return math.degrees(math.acos(dot))

    output = []
    for indices in groups.values():
        by_component = {}
        for index in indices:
            by_component.setdefault(endpoints[index]["component_ref"], endpoints[index])
        incident = list(by_component.values())
        connector_class = None
        if len(incident) == 2:
            included_angle = angle(incident[0]["direction"], incident[1]["direction"])
            if 70.0 <= included_angle <= 110.0:
                connector_class = "elbow_L"
        elif len(incident) == 3:
            angles = {(left, right): angle(incident[left]["direction"],
                                           incident[right]["direction"])
                      for left in range(3) for right in range(left + 1, 3)}
            straight = [pair for pair, value in angles.items() if value >= 160.0]
            if len(straight) == 1:
                branch = next(index for index in range(3) if index not in straight[0])
                if all(70.0 <= angles[tuple(sorted((branch, arm)))] <= 110.0
                       for arm in straight[0]):
                    connector_class = "tee_T"
        if connector_class is None:
            continue
        point = [round(sum(row["point"][axis] for row in incident) / len(incident), 6)
                 for axis in (0, 1)]
        output.append({
            "id": _stable_id("mep_observed_projected_connector", page_ref,
                             connector_class, point,
                             sorted(row["component_ref"] for row in incident)),
            "record_type": "mep_observed_projected_connector_candidate",
            "page_ref": page_ref,
            "state": "observed_projected_connector_candidate",
            "generic_class": connector_class,
            "point_display": point,
            "incident_component_refs": sorted(
                row["component_ref"] for row in incident),
            "observed_count": 1,
            "deduplicated_count": None,
            "physical_count": None,
            "connection_standard": None,
            "manufacturer": None,
            "model": None,
            "sku": None,
            "engineer_review_required": True,
            "quantity_eligible": False,
        })
    return sorted(output, key=lambda row: (row["generic_class"],
                                           row["point_display"], row["id"]))


def _transverse_attachment_indices(
    route_rows: Sequence[Mapping[str, Any]],
) -> set[int]:
    """Measure short outlined bars crossing longer route interiors.

    Both segment parameters must be interior, which keeps endpoint branches and
    elbows out of this negative.  Similar-length crossing routes also remain
    untouched and disconnected.
    """
    segments = []
    for index, row in enumerate(route_rows):
        points = row.get("polyline_display") or []
        if len(points) != 2:
            continue
        a, b = tuple(map(float, points[0])), tuple(map(float, points[1]))
        length = math.dist(a, b)
        if length > 1e-8:
            segments.append((index, a, b, length))
    output = set()
    for position, (left_index, a, b, left_length) in enumerate(segments):
        ab = (b[0] - a[0], b[1] - a[1])
        for right_index, c, d, right_length in segments[position + 1:]:
            if min(left_length, right_length) > .35 * max(left_length, right_length):
                continue
            short_index, long_index = ((left_index, right_index)
                                      if left_length < right_length
                                      else (right_index, left_index))
            short_width = float(route_rows[short_index].get(
                "corridor_width_display_points") or 0.0)
            long_width = float(route_rows[long_index].get(
                "corridor_width_display_points") or 0.0)
            # A shorter crossing route is not thereby a hanger.  A transverse
            # attachment candidate must also be a thin bar whose full span is
            # local to the wider route; a large pipe crossing a small drain
            # must remain route geometry.
            if (min(short_width, long_width) <= 0
                    or short_width > .5 * long_width
                    or min(left_length, right_length) > 4.0 * long_width):
                continue
            cd = (d[0] - c[0], d[1] - c[1])
            denominator = ab[0] * cd[1] - ab[1] * cd[0]
            if abs(denominator) <= 1e-8:
                continue
            angle = math.degrees(math.acos(max(-1.0, min(1.0,
                abs((ab[0] * cd[0] + ab[1] * cd[1])
                    / (left_length * right_length))))))
            if not 70.0 <= angle <= 90.0:
                continue
            ac = (c[0] - a[0], c[1] - a[1])
            left_t = (ac[0] * cd[1] - ac[1] * cd[0]) / denominator
            right_t = (ac[0] * ab[1] - ac[1] * ab[0]) / denominator
            if not (.08 < left_t < .92 and .08 < right_t < .92):
                continue
            output.add(short_index)
    return output


def build_page_wide_route_recovery(
    *, page_ref: str, path_pack: NativePathPack,
    role_pack: PageWidePathRolePack, styles: Sequence[Mapping[str, Any]],
    text_boxes_display: Sequence[Sequence[float]],
    outlined_composites: Sequence[Mapping[str, Any]],
    m4_relations: Sequence[Mapping[str, Any]],
    fragment_source_refs: Mapping[str, str],
    interface_points: Sequence[Mapping[str, Any]] = (),
    problem_regions: Sequence[Mapping[str, Any]] = (),
    route_annotations: Sequence[Mapping[str, Any]] = (),
    drawing_inches_per_paper_inch: float | None = None,
    annotation_supported_style_ids: Iterable[int] = (),
    annotation_style_correlations: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    if role_pack.manifest.get("record_count") != len(path_pack):
        raise ValueError("route recovery path and role packs differ")
    text_cells = _text_spatial_index(text_boxes_display)

    edges: list[_Edge] = []
    path_roles: dict[int, str] = {}
    path_style_ids: dict[int, int | None] = {}
    drawing_to_path: dict[int, int] = {}
    needed_drawings = {
        int(drawing) for row in outlined_composites
        if row.get("page_ref") == page_ref
        for drawing in row.get("member_drawing_ordinals", [])
    }
    needed_drawings.update(
        _drawing_ordinal(ref) for row in outlined_composites
        if row.get("page_ref") == page_ref
        for ref in row.get("member_source_primitive_refs", []))
    needed_drawings.update(
        _drawing_ordinal(fragment_source_refs[fragment_ref])
        for relation in m4_relations
        if relation.get("page_ref") == page_ref and relation.get("state") == "accepted"
        for fragment_ref in relation.get("target_fragment_refs", [])
        if fragment_ref in fragment_source_refs)
    for path, role in zip(path_pack.records(), role_pack.records()):
        ordinal = path["path_ordinal"]
        if path["drawing_ordinal"] in needed_drawings:
            drawing_to_path[path["drawing_ordinal"]] = ordinal
            path_roles[ordinal] = role["role"]
            path_style_ids[ordinal] = path["style_id"]
        if role["role"] != "anchored_route_style_candidate":
            continue
        if path["style_id"] is None or path["source_segment_length_points"] <= 1e-8:
            continue
        edges.append(_Edge(
            path_ordinal=ordinal, drawing_ordinal=path["drawing_ordinal"],
            style_id=path["style_id"], start=tuple(path["start_display"]),
            end=tuple(path["end_display"]),
            length=float(path["source_segment_length_points"]),
            source_segment_count=int(path["source_segment_count"]),
            legacy=bool(role["flags"] & FLAG_LEGACY_M3_CANDIDATE),
            oblique=bool(path["axis_mask"] & 4),
        ))

    endpoint_buckets: dict[tuple[int, float, float], list[tuple[int, int]]] = defaultdict(list)
    for edge_index, edge in enumerate(edges):
        for endpoint_index, point in enumerate((edge.start, edge.end)):
            endpoint_buckets[(edge.style_id, round(point[0], 6),
                              round(point[1], 6))].append(
                                  (edge_index, endpoint_index))

    endpoint_join: dict[tuple[int, int], tuple[int, int, str, float]] = {}
    branch_buckets = []
    ambiguous_bucket_count = 0
    exact_join_count = near_join_count = 0
    for key, endpoints in endpoint_buckets.items():
        unique_edges = {row[0] for row in endpoints}
        if len(endpoints) == 2 and len(unique_edges) == 2:
            (left, left_end), (right, right_end) = endpoints
            a = (edges[left].start, edges[left].end)[left_end]
            b = (edges[right].start, edges[right].end)[right_end]
            residual = math.dist(a, b)
            if residual <= 1e-6:
                kind = "exact_join"
                endpoint_join[(left, left_end)] = (right, right_end, kind, residual)
                endpoint_join[(right, right_end)] = (left, left_end, kind, residual)
                exact_join_count += 1
            else:
                ambiguous_bucket_count += 1
        elif len(unique_edges) >= 3:
            points = [(edges[index].start, edges[index].end)[end]
                      for index, end in endpoints]
            branch_buckets.append({
                "key": key, "endpoints": endpoints,
                "point_display": [round(sum(p[0] for p in points) / len(points), 6),
                                  round(sum(p[1] for p in points) / len(points), 6)],
            })
        elif len(endpoints) > 1:
            ambiguous_bucket_count += 1

    # Mutually unique bounded near joins are evaluated only for exact-dangling
    # endpoints.  Adjacent cells are searched explicitly so bucket edges cannot
    # suppress a valid candidate.
    near_cells: dict[tuple[int, int, int], list[tuple[int, int]]] = defaultdict(list)
    dangling = []
    for key, endpoints in endpoint_buckets.items():
        if len(endpoints) != 1:
            continue
        endpoint = endpoints[0]
        if endpoint in endpoint_join:
            continue
        edge_index, endpoint_index = endpoint
        edge = edges[edge_index]
        point = (edge.start, edge.end)[endpoint_index]
        tolerance = _join_tolerance(styles[edge.style_id])
        cell = (edge.style_id, math.floor(point[0] / tolerance),
                math.floor(point[1] / tolerance))
        near_cells[cell].append(endpoint)
        dangling.append(endpoint)
    near_candidates: dict[tuple[int, int], list[tuple[int, int, float]]] = {}
    for endpoint in dangling:
        edge_index, endpoint_index = endpoint
        edge = edges[edge_index]
        point = (edge.start, edge.end)[endpoint_index]
        tolerance = _join_tolerance(styles[edge.style_id])
        cell = (edge.style_id, math.floor(point[0] / tolerance),
                math.floor(point[1] / tolerance))
        candidates = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in near_cells.get((cell[0], cell[1] + dx, cell[2] + dy), ()):
                    if other == endpoint or other[0] == edge_index:
                        continue
                    other_edge = edges[other[0]]
                    other_point = (other_edge.start, other_edge.end)[other[1]]
                    residual = math.dist(point, other_point)
                    if 1e-6 < residual <= tolerance:
                        candidates.append((other[0], other[1], residual))
        near_candidates[endpoint] = candidates
    seen_near = set()
    for endpoint, candidates in near_candidates.items():
        if len(candidates) != 1:
            if candidates:
                ambiguous_bucket_count += 1
            continue
        other = (candidates[0][0], candidates[0][1])
        reverse = near_candidates.get(other, [])
        if len(reverse) != 1 or (reverse[0][0], reverse[0][1]) != endpoint:
            continue
        key = tuple(sorted((endpoint, other)))
        if key in seen_near:
            continue
        residual = candidates[0][2]
        endpoint_join[endpoint] = (*other, "mutually_unique_near_join", residual)
        endpoint_join[other] = (*endpoint, "mutually_unique_near_join", residual)
        seen_near.add(key)
        near_join_count += 1

    adjacency: dict[int, set[int]] = defaultdict(set)
    for (edge_index, _), (other, _, _, _) in endpoint_join.items():
        adjacency[edge_index].add(other)
    edge_groups = []
    visited = bytearray(len(edges))
    for start in range(len(edges)):
        if visited[start]:
            continue
        visited[start] = 1
        queue = deque([start])
        group = []
        while queue:
            current = queue.popleft()
            group.append(current)
            for other in adjacency[current]:
                if not visited[other]:
                    visited[other] = 1
                    queue.append(other)
        edge_groups.append(sorted(group))

    components = []
    excluded_leaders = []
    symbol_candidates = []
    edge_to_component: dict[int, str] = {}
    path_outcomes: dict[int, str] = {}
    component_path_ordinals: dict[str, list[int]] = {}
    branch_edge_indices = {
        edge_index for row in branch_buckets for edge_index, _ in row["endpoints"]
    }
    for group in edge_groups:
        group_set = set(group)
        degree = {index: len(adjacency[index] & group_set) for index in group}
        start_edge = min((index for index in group if degree[index] < 2), default=min(group))
        joined_ends = {end for end in (0, 1) if (start_edge, end) in endpoint_join}
        enter = next((end for end in (0, 1) if end not in joined_ends), 0)
        ordered, points = [], []
        current, previous = start_edge, None
        join_kinds = Counter()
        join_residuals = []
        while current is not None and current not in ordered:
            ordered.append(current)
            edge = edges[current]
            a, b = (edge.start, edge.end) if enter == 0 else (edge.end, edge.start)
            if not points:
                points.append(a)
            points.append(b)
            link = endpoint_join.get((current, 1 - enter))
            if link is None or link[0] == previous:
                current = None
                continue
            other, other_end, kind, residual = link
            if other in ordered:
                current = None
                continue
            join_kinds[kind] += 1
            join_residuals.append(residual)
            previous, current, enter = current, other, other_end
        # Components with a rare non-chain topology remain covered but are not
        # rendered as one route.
        if len(ordered) != len(group):
            ordered.extend(sorted(group_set - set(ordered)))
        total_length = sum(edges[index].length for index in group)
        width = float(styles[edges[group[0]].style_id].get("width") or .12)
        closed = (all(degree[index] == 2 for index in group)
                  or (len(points) > 2 and math.dist(points[0], points[-1])
                      <= _join_tolerance(styles[edges[group[0]].style_id])))
        has_branch_terminal = any(edge_index in branch_edge_indices
                                  for edge_index in group)
        terminal_text = bool(points and (
            _near_indexed_text(points[0], text_cells)
            or _near_indexed_text(points[-1], text_cells)))
        oblique = any(edges[index].oblique for index in group)
        legacy_count = sum(edges[index].legacy for index in group)
        if closed and total_length <= max(120.0, 120 * width):
            state = "fitting_or_equipment_symbol_candidate"
            reason = "closed_anchored_style_component"
        elif has_branch_terminal:
            state = "unresolved_branch_or_ceiling_attachment"
            reason = "multi_endpoint_coincidence_is_not_route_support"
        elif (terminal_text and oblique and not has_branch_terminal
              and total_length <= 300.0):
            state = "text_leader_candidate_excluded_from_route"
            reason = "unbranched_oblique_chain_terminates_in_text_zone"
        elif (total_length >= max(3.0, 6 * width)
              and (len(group) >= 2 or total_length >= 12.0)):
            state = "supported_unidentified_mep_candidate"
            reason = "anchored_route_style_plus_bounded_topology"
        else:
            state = "unresolved_anchored_style_geometry"
            reason = "insufficient_route_extent_or_topology"
        path_ordinals = [edges[index].path_ordinal for index in group]
        for path_ordinal in path_ordinals:
            path_outcomes[path_ordinal] = state
        if state == "text_leader_candidate_excluded_from_route":
            excluded_leaders.append({
                "id": _stable_id("mep_text_leader_chain_candidate", page_ref,
                                 path_ordinals),
                "state": state,
                "path_ordinal_intervals": _intervals(path_ordinals),
                "authored_path_count": len(group),
                "source_segment_count": sum(
                    edges[index].source_segment_count for index in group),
                "polyline_display": _simplify(
                    points, _join_tolerance(styles[edges[group[0]].style_id])),
                "route_identity_established": False,
                "quantity_eligible": False,
            })
        elif state == "fitting_or_equipment_symbol_candidate":
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            symbol_candidates.append({
                "id": _stable_id("mep_anchored_symbol_candidate", page_ref,
                                 path_ordinals),
                "state": state,
                "generic_class": "UNKNOWN_CONNECTOR_OR_ELEMENT",
                "path_ordinal_intervals": _intervals(path_ordinals),
                "authored_path_count": len(group),
                "source_segment_count": sum(
                    edges[index].source_segment_count for index in group),
                "bbox_display": ([round(min(xs), 6), round(min(ys), 6),
                                  round(max(xs), 6), round(max(ys), 6)]
                                 if points else None),
                "polyline_display": _simplify(
                    points, _join_tolerance(styles[edges[group[0]].style_id])),
                "exact_identity_established": False,
                "engineer_review_required": True,
                "quantity_eligible": False,
            })
        if state != "supported_unidentified_mep_candidate":
            continue
        component_id = _stable_id(
            "mep_page_wide_route_component", page_ref, path_ordinals, state)
        component = {
            "id": component_id,
            "record_type": "mep_page_wide_route_component_candidate",
            "page_ref": page_ref, "channel": "single_centreline_or_microsegment",
            "state": state, "reason": reason,
            "style_id": edges[group[0]].style_id,
            "path_ordinal_intervals": _intervals(path_ordinals),
            "authored_path_count": len(group),
            "source_segment_count": sum(edges[index].source_segment_count for index in group),
            "legacy_M3_candidate_path_count": legacy_count,
            "legacy_M3_used_as_route_authority": False,
            "projected_path_display_points": round(total_length, 6),
            "polyline_display": _simplify(
                points, _join_tolerance(styles[edges[group[0]].style_id])),
            "exact_join_count": join_kinds["exact_join"],
            "certified_near_join_count": join_kinds["mutually_unique_near_join"],
            "maximum_near_join_residual_display_points": (
                round(max(join_residuals), 6) if join_residuals else 0.0),
            "branch_terminal_present": has_branch_terminal,
            "text_terminal_present": terminal_text,
            "attributes": _attribute_values([]),
            "physical_continuity_established": False,
            "installed_length": None, "purchase_length": None,
            "quantity_eligible": False,
        }
        components.append(component)
        component_path_ordinals[component_id] = path_ordinals
        for index in group:
            edge_to_component[index] = component_id

    path_to_component = {
        path_ordinal: component_id
        for component_id, ordinals in component_path_ordinals.items()
        for path_ordinal in ordinals
    }
    relations_for_component = defaultdict(list)
    for relation in m4_relations:
        if relation.get("page_ref") != page_ref or relation.get("state") != "accepted":
            continue
        targets = []
        for fragment_ref in relation.get("target_fragment_refs", []):
            source_ref = fragment_source_refs.get(fragment_ref)
            if source_ref is None:
                continue
            path_ordinal = drawing_to_path.get(_drawing_ordinal(source_ref))
            if path_ordinal is not None and path_ordinal in path_to_component:
                targets.append(path_to_component[path_ordinal])
        if targets and len(set(targets)) == 1:
            relations_for_component[targets[0]].append(relation)
    for component in components:
        relations = relations_for_component.get(component["id"], [])
        component["attributes"] = _attribute_values(relations)
        if (component["state"] == "supported_unidentified_mep_candidate"
                and component["attributes"]["route_system"]["state"] == "accepted"):
            component["state"] = "identified_mep_route"
            component["reason"] = "unique_accepted_M4_system_binding"
            for path_ordinal in component_path_ordinals[component["id"]]:
                path_outcomes[path_ordinal] = "identified_mep_route"

    component_by_id = {row["id"]: row for row in components}
    # A three-or-more endpoint coincidence is not a branch certificate.  The
    # same native topology is produced by hangers / ceiling attachments and
    # other transverse symbols.  Preserve it as a review candidate until an
    # independently typed tee, fitting, or port uniquely binds at this point.
    branch_or_attachment_candidates = []
    accepted_branches = []
    for branch in branch_buckets:
        incident = sorted({edge_to_component[index] for index, _ in branch["endpoints"]
                           if index in edge_to_component
                           and component_by_id[edge_to_component[index]]["state"] in {
                               "identified_mep_route",
                               "supported_unidentified_mep_candidate",
                           }})
        typed_candidates = [
            row for row in interface_points
            if row.get("point_display")
            and math.dist(branch["point_display"], row["point_display"]) <= 2.0
            and row.get("state") == "accepted"
            and row.get("interface_class") in {"tee", "branch_fitting", "typed_port"}
        ]
        candidate = {
            "id": _stable_id("mep_projected_branch_or_attachment_candidate",
                             page_ref, branch["point_display"], incident),
            "state": "unresolved_branch_or_ceiling_attachment",
            "point_display": branch["point_display"],
            "incident_component_refs": incident,
            "incident_path_ordinals": sorted({
                edges[index].path_ordinal for index, _ in branch["endpoints"]}),
            "candidate_typed_interface_refs": sorted(
                row["id"] for row in typed_candidates),
            "branch_identity_established": False,
            "system_propagation_permitted": False,
            "physical_continuity_established": False,
            "quantity_eligible": False,
        }
        branch_or_attachment_candidates.append(candidate)
        if len(incident) < 3 or len(typed_candidates) != 1:
            continue
        accepted_branches.append({
            **candidate,
            "id": _stable_id("mep_projected_branch_certificate", page_ref,
                             branch["point_display"], incident,
                             typed_candidates[0]["id"]),
            "state": "accepted_projected_branch_geometry",
            "accepted_typed_interface_ref": typed_candidates[0]["id"],
            "branch_identity_established": True,
        })

    interface_bindings = []
    for component in components:
        if component["state"] not in {
                "identified_mep_route", "supported_unidentified_mep_candidate"}:
            continue
        for terminal_index, point in enumerate((component["polyline_display"][0],
                                                component["polyline_display"][-1])):
            candidates = [row for row in interface_points
                          if row.get("point_display")
                          and math.dist(point, row["point_display"]) <= 2.0]
            accepted = [row for row in candidates if row.get("state") == "accepted"]
            state = "accepted" if len(accepted) == 1 and len(candidates) == 1 else "unresolved"
            interface_bindings.append({
                "id": _stable_id("mep_projected_interface_binding", component["id"],
                                 terminal_index, [row.get("id") for row in candidates]),
                "state": state, "component_ref": component["id"],
                "terminal_index": terminal_index, "point_display": point,
                "candidate_interface_refs": sorted(row["id"] for row in candidates),
                "accepted_interface_ref": accepted[0]["id"] if state == "accepted" else None,
                "accepted_interface_class": (
                    accepted[0].get("interface_class") if state == "accepted" else None),
                "reason": None if state == "accepted" else (
                    "no_interface_candidate" if not candidates
                    else "interface_not_uniquely_typed_and_accepted"),
                "system_propagation_permitted": False,
                "physical_continuity_established": False,
                "quantity_eligible": False,
            })

    relations_by_composite = defaultdict(list)
    for relation in m4_relations:
        if relation.get("page_ref") != page_ref or relation.get("state") != "accepted":
            continue
        for target in relation.get("target_refs", []):
            relations_by_composite[target].append(relation)
    outlined_rows = []
    for composite in outlined_composites:
        if (composite.get("page_ref") != page_ref
                or composite.get("state") not in {
                    "accepted", "accepted_projected_corridor_geometry"}):
            continue
        drawings = list(map(int, composite.get("member_drawing_ordinals", [])))
        if not drawings:
            drawings = [_drawing_ordinal(ref)
                        for ref in composite.get("member_source_primitive_refs", [])]
        ordinals = list(map(int, composite.get("member_path_ordinals", [])))
        if not ordinals:
            ordinals = [drawing_to_path.get(drawing) for drawing in drawings]
        if None in ordinals or any(path_roles.get(value) in {
                "excluded_non_view_content", "annotation_dimension",
                "measured_hatch_candidate",
                "architectural_boundary_candidate", "drawing_furniture",
        } for value in ordinals):
            continue
        attributes = _attribute_values(relations_by_composite[composite["id"]])
        # M4 names an identified system.  Unnamed outlined corridors are
        # evaluated below through a separate geometry/style seed-and-extension
        # certificate; the outlined pair alone is never enough.
        state = ("identified_mep_route" if attributes["route_system"]["state"] == "accepted"
                 else "unclassified_outlined_corridor")
        geometry = composite.get("derived_geometry", {})
        style_ids = {path_style_ids.get(value) for value in ordinals}
        style_id = next(iter(style_ids)) if len(style_ids) == 1 else None
        outlined_rows.append({
            "id": _stable_id("mep_page_wide_outline_route", composite["id"]),
            "record_type": "mep_page_wide_route_component_candidate",
            "page_ref": page_ref, "channel": "parallel_outline_corridor",
            "state": state,
            "outlined_composite_ref": composite["id"],
            "certificate_method": composite.get("method"),
            "certificate_flags": composite.get("certificates"),
            "source_path_ordinals": ordinals,
            "style_id": style_id,
            "source_segment_refs": composite.get("member_source_primitive_refs", []),
            "source_path_ordinal_intervals": _intervals(ordinals),
            "source_descriptor_ordinal_intervals": composite.get(
                "member_descriptor_ordinal_intervals", []),
            "polyline_display": geometry.get("centreline_points_display", []),
            "projected_path_display_points": geometry.get("projected_path_display_points"),
            "corridor_width_display_points": geometry.get(
                "corridor_width_display_points"),
            "attributes": attributes,
            "annotation_candidate_refs": [],
            "candidate_systems": [],
            "candidate_nominal_sizes_inches": [],
            "legacy_M3_membership_is_not_route_authority": True,
            "physical_continuity_established": False,
            "installed_length": None, "purchase_length": None,
            "quantity_eligible": False,
        })

    # Recover real outlined routes without allowing the repeated transverse
    # attachment glyphs to become seeds.  Only long, slender corridors seed
    # geometry.  Shorter corridors are admitted
    # only when a mutually unique same-style endpoint chain reaches a seed.
    # This uses the current outline certificate plus an independently frozen
    # anchored MEP style; neither colour nor connectivity names the system.
    anchored_style_ids = set(role_pack.manifest.get("anchored_style_ids", []))
    transverse_attachments = _transverse_attachment_indices(outlined_rows)
    annotation_seeds: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    annotation_supported_style_ids = set(map(int, annotation_supported_style_ids))
    annotation_target_style_ids = (
        annotation_supported_style_ids | anchored_style_ids)
    annotation_style_by_id = {
        int(row["style_id"]): row for row in annotation_style_correlations
        if row.get("state") == "supported_route_style_correlation"
    }
    scale = float(drawing_inches_per_paper_inch or 0.0)

    def row_box_gap(row: Mapping[str, Any], box: Sequence[float]) -> float:
        points = row.get("polyline_display") or []
        if not points:
            return math.inf
        x0 = min(float(point[0]) for point in points)
        y0 = min(float(point[1]) for point in points)
        x1 = max(float(point[0]) for point in points)
        y1 = max(float(point[1]) for point in points)
        dx = max(x0 - float(box[2]), float(box[0]) - x1, 0.0)
        dy = max(y0 - float(box[3]), float(box[1]) - y1, 0.0)
        return math.hypot(dx, dy)

    if scale > 0:
        for annotation in route_annotations:
            nominal = annotation.get("nominal_size_inches")
            box = annotation.get("bbox_display")
            if not isinstance(nominal, (int, float)) or not box:
                continue
            expected_width = float(nominal) / scale * 72.0
            candidates = []
            for index, row in enumerate(outlined_rows):
                if (annotation_target_style_ids
                        and row.get("style_id") not in annotation_target_style_ids):
                    continue
                width = float(row.get("corridor_width_display_points") or 0.0)
                length = float(row.get("projected_path_display_points") or 0.0)
                if width <= 0 or length < max(12.0, 3.0 * width):
                    continue
                if not (.55 * expected_width <= width <= 1.8 * expected_width):
                    continue
                gap = row_box_gap(row, box)
                if gap > 100.0:
                    continue
                score = gap + 12.0 * abs(math.log(width / expected_width))
                candidates.append((score, index))
            candidates.sort(key=lambda item: (item[0], outlined_rows[item[1]]["id"]))
            if not candidates:
                continue
            best_score, best_index = candidates[0]
            competing = [index for score, index in candidates[1:]
                         if score - best_score <= 3.0
                         and outlined_rows[index].get("style_id") !=
                         outlined_rows[best_index].get("style_id")]
            if competing:
                continue
            annotation_seeds[best_index].append(annotation)

    outline_seed_indices = set()
    for index, row in enumerate(outlined_rows):
        style_id = row.get("style_id")
        length = float(row.get("projected_path_display_points") or 0.0)
        width = float(row.get("corridor_width_display_points") or 0.0)
        source_stroke_width = float(
            styles[style_id].get("width") or .12) if style_id is not None else .12
        slender = bool(
            width >= max(1.0, 1.2 * source_stroke_width)
            and length / width >= 5.0 and length >= 12.0)
        geometry_seed = bool(
            style_id in anchored_style_ids and slender
            and index not in transverse_attachments)
        annotation_seed = bool(
            annotation_seeds.get(index) and index not in transverse_attachments)
        style_correlation = annotation_style_by_id.get(style_id, {})
        annotated_style_width_match = bool(scale > 0 and any(
            .55 * (float(size) / scale * 72.0) <= width <=
            1.8 * (float(size) / scale * 72.0)
            for size in style_correlation.get("nominal_size_candidates_inches", [])))
        annotation_style_seed = bool(
            style_correlation and annotated_style_width_match
            and length >= max(12.0, 2.5 * width)
            and index not in transverse_attachments)
        identified_seed = row["state"] == "identified_mep_route"
        row["route_candidate_support"] = {
            "anchored_route_style": style_id in anchored_style_ids,
            "long_slender_seed": geometry_seed,
            "explicit_annotation_and_width_seed": annotation_seed,
            "repeated_annotation_style_seed": annotation_style_seed,
            "repeated_annotation_style_width_match": annotated_style_width_match,
            "annotation_style_correlation_ref": (
                annotation_style_by_id[style_id]["id"]
                if annotation_style_seed else None),
            "annotation_refs": sorted(
                str(value.get("id")) for value in annotation_seeds.get(index, [])),
            "minimum_seed_length_display_points": 12.0,
            "minimum_corridor_width_display_points": round(
                max(1.0, 1.2 * source_stroke_width), 6),
            "minimum_length_to_width_ratio": 5.0,
            "length_to_corridor_width_ratio": (
                round(length / width, 6) if width > 0 else None),
            "connected_to_seed": False,
            "transverse_attachment_negative": index in transverse_attachments,
            "system_identity_from_colour_or_connectivity": False,
        }
        if geometry_seed or annotation_seed or annotation_style_seed or identified_seed:
            outline_seed_indices.add(index)

    route_style_ids = anchored_style_ids | {
        int(outlined_rows[index]["style_id"])
        for index in annotation_seeds
        if outlined_rows[index].get("style_id") is not None
    }
    outline_endpoints = []
    for index, row in enumerate(outlined_rows):
        points = row.get("polyline_display") or []
        if len(points) < 2 or row.get("style_id") not in route_style_ids:
            continue
        outline_endpoints.extend(((index, 0, points[0]),
                                  (index, 1, points[-1])))
    outline_near: dict[tuple[int, int], list[tuple[int, int, float]]] = defaultdict(list)
    for left_pos, (left, left_end, a) in enumerate(outline_endpoints):
        left_row = outlined_rows[left]
        left_width = float(left_row.get("corridor_width_display_points") or 0.0)
        for right, right_end, b in outline_endpoints[left_pos + 1:]:
            if left == right or left_row.get("style_id") != outlined_rows[right].get("style_id"):
                continue
            right_width = float(outlined_rows[right].get(
                "corridor_width_display_points") or 0.0)
            if min(left_width, right_width) <= 0:
                continue
            if max(left_width, right_width) / min(left_width, right_width) > 1.12:
                continue
            residual = math.dist(a, b)
            # Corridor midpoints accumulate the residuals of two independently
            # authored sidewalls.  Use a width-derived, strictly capped bound;
            # mutual uniqueness and endpoint-only matching still prohibit a
            # global snap or crossing join.
            tolerance = min(.5, max(.12, min(left_width, right_width) / 12.0))
            if residual <= tolerance:
                outline_near[(left, left_end)].append((right, right_end, residual))
                outline_near[(right, right_end)].append((left, left_end, residual))
    outline_adjacency: dict[int, set[int]] = defaultdict(set)
    for endpoint, candidates in outline_near.items():
        if len(candidates) != 1:
            continue
        other = (candidates[0][0], candidates[0][1])
        reverse = outline_near.get(other, [])
        if len(reverse) == 1 and (reverse[0][0], reverse[0][1]) == endpoint:
            outline_adjacency[endpoint[0]].add(other[0])

    outline_supported = set(outline_seed_indices)
    queue = deque(sorted(outline_seed_indices))
    while queue:
        current = queue.popleft()
        for other in outline_adjacency[current]:
            if other not in outline_supported:
                outline_supported.add(other)
                queue.append(other)

    reached_annotations: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for seed, annotations in annotation_seeds.items():
        reached = {seed}
        queue = deque([seed])
        while queue:
            current = queue.popleft()
            for other in outline_adjacency[current]:
                if other not in reached:
                    reached.add(other)
                    queue.append(other)
        for index in reached:
            reached_annotations[index].extend(annotations)
    for index, row in enumerate(outlined_rows):
        row["route_candidate_support"]["connected_to_seed"] = (
            index in outline_supported and index not in outline_seed_indices)
        if (index in outline_supported and index not in transverse_attachments
                and row["state"] == "unclassified_outlined_corridor"):
            row["state"] = "supported_unidentified_mep_candidate"
            row["reason"] = (
                "explicit_native_annotation_size_and_corridor_width"
                if annotation_seeds.get(index) else
                "repeated_native_annotation_route_style_correlation"
                if row.get("style_id") in annotation_style_by_id else
                "certified_outline_plus_anchored_style_long_slender_seed"
                if index in outline_seed_indices else
                "mutually_unique_same_style_endpoint_chain_to_route_seed")
        elif row["state"] == "unclassified_outlined_corridor":
            row["reason"] = (
                "short_transverse_corridor_crosses_longer_route_interior"
                if index in transverse_attachments else
                "outlined_pair_without_independent_route_support")
        annotations = reached_annotations.get(index, [])
        row["annotation_candidate_refs"] = sorted({
            str(value.get("id")) for value in annotations})
        style_correlation = annotation_style_by_id.get(row.get("style_id"), {})
        row["candidate_systems"] = sorted({
            str(value.get("system_candidate")) for value in annotations
            if value.get("system_candidate")} | set(
                style_correlation.get("system_candidates", [])))
        row["candidate_nominal_sizes_inches"] = sorted({
            float(value["nominal_size_inches"]) for value in annotations
            if isinstance(value.get("nominal_size_inches"), (int, float))} | set(
                style_correlation.get("nominal_size_candidates_inches", [])))

    observed_connectors = _observed_connector_candidates(
        page_ref=page_ref, route_rows=outlined_rows)

    problem_results = []
    for region in problem_regions:
        counts = Counter()
        for edge in edges:
            box = [min(edge.start[0], edge.end[0]), min(edge.start[1], edge.end[1]),
                   max(edge.start[0], edge.end[0]), max(edge.start[1], edge.end[1])]
            region_box = region["bbox_display"]
            if not (box[2] < region_box[0] or region_box[2] < box[0]
                    or box[3] < region_box[1] or region_box[3] < box[1]):
                counts[path_outcomes.get(edge.path_ordinal, "unaccounted")] += 1
        problem_results.append({**region, "anchored_path_outcome_counts": dict(sorted(counts.items()))})

    outcome_counts = Counter(path_outcomes.values())
    candidate_path_count = len(edges)
    payload = {
        "schema_version": SCHEMA_VERSION, "layer": LAYER, "page_ref": page_ref,
        "precision_characterization": {
            str(style_id): {
                "coordinate_quantum_upper_bound_display_points": .12,
                "line_width_display_points": styles[style_id].get("width"),
                "maximum_join_tolerance_display_points": _join_tolerance(styles[style_id]),
                "global_snap_tolerance_used": False,
            }
            for style_id in sorted({edge.style_id for edge in edges})
        },
        "single_centreline_components": components,
        "excluded_text_leader_candidates": excluded_leaders,
        "fitting_or_equipment_symbol_candidates": symbol_candidates,
        "outlined_corridor_components": outlined_rows,
        "observed_projected_connector_candidates": observed_connectors,
        "projected_branch_or_attachment_candidates": branch_or_attachment_candidates,
        "accepted_projected_branch_certificates": accepted_branches,
        "interface_bindings": interface_bindings,
        "problem_region_dispositions": problem_results,
        "coverage": {
            "native_source_segment_count": path_pack.manifest["source_segment_count"],
            "authored_path_count": len(path_pack),
            "path_role_count": role_pack.manifest["record_count"],
            "anchored_route_style_candidate_path_count": candidate_path_count,
            "anchored_candidate_outcome_path_counts": dict(sorted(outcome_counts.items())),
            "anchored_candidate_accounted_path_count": sum(outcome_counts.values()),
            "unaccounted_anchored_candidate_path_count": (
                candidate_path_count - sum(outcome_counts.values())),
            "identified_outline_route_count": sum(
                row["state"] == "identified_mep_route" for row in outlined_rows),
            "supported_unidentified_outline_route_count": sum(
                row["state"] == "supported_unidentified_mep_candidate"
                for row in outlined_rows),
            "unclassified_outline_corridor_count": sum(
                row["state"] == "unclassified_outlined_corridor" for row in outlined_rows),
            "long_slender_outline_seed_count": sum(
                row.get("route_candidate_support", {}).get("long_slender_seed") is True
                for row in outlined_rows),
            "measured_transverse_attachment_negative_count": len(
                transverse_attachments),
            "explicit_route_annotation_count": len(route_annotations),
            "annotation_width_seed_count": len(annotation_seeds),
            "exact_join_count": exact_join_count,
            "certified_near_join_count": near_join_count,
            "ambiguous_endpoint_bucket_count": ambiguous_bucket_count,
            "unresolved_branch_or_attachment_candidate_count": len(
                branch_or_attachment_candidates) - len(accepted_branches),
            "accepted_projected_branch_count": len(accepted_branches),
            "accepted_typed_interface_binding_count": sum(
                row["state"] == "accepted" for row in interface_bindings),
            "observed_projected_connector_candidate_counts": dict(sorted(Counter(
                row["generic_class"] for row in observed_connectors).items())),
        },
        "acceptance_gate": {
            "all_denominator_segments_accounted": (
                role_pack.manifest["source_segment_count"]
                == path_pack.manifest["source_segment_count"]),
            "every_authored_path_has_one_role": (
                role_pack.manifest["record_count"] == len(path_pack)),
            "every_anchored_candidate_has_one_outcome": (
                candidate_path_count == sum(outcome_counts.values())),
            "legacy_M3_direct_route_authority": False,
            "every_rendered_route_requires_current_certificate": True,
            "system_identity_from_colour": False,
            "installed_length": None, "purchase_length": None,
        },
        "authority": {
            "projected_route_candidates_established": True,
            "M4_only_names_systems": True,
            "physical_continuity_established": False,
            "installed_length_established": False,
            "purchase_length_established": False,
            "quantity_eligible": False,
        },
    }
    positive_gates = (
        "all_denominator_segments_accounted", "every_authored_path_has_one_role",
        "every_anchored_candidate_has_one_outcome",
        "every_rendered_route_requires_current_certificate",
    )
    negative_authority_gates = (
        "legacy_M3_direct_route_authority", "system_identity_from_colour",
    )
    payload["acceptance_gate"]["status"] = (
        "accepted_page5_source_coverage_for_diagnostic_render"
        if (all(payload["acceptance_gate"][key] is True for key in positive_gates)
            and all(payload["acceptance_gate"][key] is False
                    for key in negative_authority_gates))
        else "development_rejected")
    return payload


def validate_page_wide_route_recovery(payload: Mapping[str, Any]) -> list[str]:
    errors = []
    coverage = payload.get("coverage", {})
    gate = payload.get("acceptance_gate", {})
    if coverage.get("unaccounted_anchored_candidate_path_count") != 0:
        errors.append("anchored route-style paths remain unaccounted")
    for key in ("all_denominator_segments_accounted", "every_authored_path_has_one_role",
                "every_anchored_candidate_has_one_outcome",
                "every_rendered_route_requires_current_certificate"):
        if gate.get(key) is not True:
            errors.append(f"coverage gate failed: {key}")
    if gate.get("legacy_M3_direct_route_authority") is not False:
        errors.append("legacy M3 was granted direct route authority")
    if gate.get("system_identity_from_colour") is not False:
        errors.append("colour was granted system identity")
    for row in [*payload.get("single_centreline_components", []),
                *payload.get("outlined_corridor_components", [])]:
        if row.get("installed_length") is not None or row.get("purchase_length") is not None:
            errors.append(f"forbidden length authority: {row.get('id')}")
            break
    return errors
