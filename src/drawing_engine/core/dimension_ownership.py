"""Attach dimension proposals to the exact drawing geometry they measure.

Dimension-chain recognition proves that a numeric token belongs to a complete
dimension annotation.  It does not by itself prove which object, contour, or
bar leg owns that dimension.  This module follows the two extension lines to
their measured endpoints and records the native vector items touching those
points.  Ambiguous endpoint targets remain candidates.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from typing import Any, Iterable, Mapping

import fitz

from src.drawing_engine.core.dimension_attachment import DimensionAttachment
from src.drawing_engine.core.vector_topology import extract_page_topology, point_distance_to_segment, vertices_by_id


def _distance_to_segment(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    dx, dy = bx - ax, by - ay
    squared = dx * dx + dy * dy
    if squared <= 1e-12:
        return math.hypot(px - ax, py - ay)
    ratio = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / squared))
    return math.hypot(px - (ax + ratio * dx), py - (ay + ratio * dy))


def _axis(start: tuple[float, float], end: tuple[float, float]) -> str:
    dx, dy = abs(end[0] - start[0]), abs(end[1] - start[1])
    if dy <= 0.7 and dx >= 1.0:
        return "horizontal"
    if dx <= 0.7 and dy >= 1.0:
        return "vertical"
    return "oblique"


def _point(value: Any) -> tuple[float, float]:
    return (float(value.x), float(value.y))


def _cubic_points(item: tuple[Any, ...], steps: int = 12) -> list[tuple[float, float]]:
    p0, p1, p2, p3 = (_point(item[index]) for index in range(1, 5))
    points = []
    for index in range(steps + 1):
        t = index / steps
        u = 1.0 - t
        points.append(
            (
                u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
                u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1],
            )
        )
    return points


def _item_segments(item: tuple[Any, ...]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    kind = item[0]
    if kind == "l":
        return [(_point(item[1]), _point(item[2]))]
    if kind == "re":
        rect = fitz.Rect(item[1])
        points = [(rect.x0, rect.y0), (rect.x1, rect.y0), (rect.x1, rect.y1), (rect.x0, rect.y1)]
        return list(zip(points, points[1:] + points[:1]))
    if kind == "qu":
        quad = fitz.Quad(item[1])
        points = [_point(quad.ul), _point(quad.ur), _point(quad.lr), _point(quad.ll)]
        return list(zip(points, points[1:] + points[:1]))
    if kind == "c":
        points = _cubic_points(item)
        return list(zip(points, points[1:]))
    return []


def _native_segments(page: fitz.Page, native_topology: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    topology = native_topology or extract_page_topology(page)
    return [
        {
            "primitive_ref": segment["primitive_ref"],
            "drawing_ref": segment["drawing_ref"],
            "segment_index": segment["part_index"],
            "topology_segment_ref": segment["id"],
            "start_display": segment["start_display"],
            "end_display": segment["end_display"],
            "start_vertex_id": segment["start_vertex_id"],
            "end_vertex_id": segment["end_vertex_id"],
            "sample_points_display": segment.get("sample_points_display", []),
            "axis": segment["axis"],
        }
        for segment in topology.get("segments", [])
    ]


def _excluded(candidate_ref: str, drawing_ref: str, excluded_refs: set[str]) -> bool:
    return candidate_ref in excluded_refs or drawing_ref in excluded_refs


def _contour_index(contours: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for contour in contours:
        for primitive_ref in contour.get("primitive_refs", []):
            result.setdefault(primitive_ref.split(".item[")[0], []).append(contour["id"])
        for segment in contour.get("segments_display", []) or []:
            result.setdefault(str(segment.get("id")), []).append(contour["id"])
            result.setdefault(str(segment.get("primitive_ref")), []).append(contour["id"])
    return result


def _view_refs(point: tuple[float, float], views: Iterable[dict[str, Any]]) -> list[str]:
    location = fitz.Point(*point)
    containing = []
    for view in views:
        rect = fitz.Rect(view["bbox_display"]) + (-2, -2, 2, 2)
        if location in rect:
            containing.append((rect.get_area(), view["id"]))
    return [view_id for _, view_id in sorted(containing)]


def _endpoint_candidates(
    point: tuple[float, float],
    dimension: DimensionAttachment,
    segments: list[dict[str, Any]],
    contour_by_drawing: dict[str, list[str]],
    views: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    excluded = {
        dimension.baseline.primitive_ref,
        *(line.primitive_ref for line in dimension.extension_lines),
        *dimension.terminal_refs,
    }
    tolerance = max(1.25, min(3.0, dimension.scale_points_per_mm * 8.0))
    candidates = []
    target_axis = "vertical" if dimension.orientation == "horizontal" else "horizontal"
    for segment in segments:
        if _excluded(segment["primitive_ref"], segment["drawing_ref"], excluded):
            continue
        sample_points = [tuple(value) for value in segment.get("sample_points_display", [])]
        distance = min(
            (
                point_distance_to_segment(point, start, end)
                for start, end in zip(sample_points, sample_points[1:])
            ),
            default=point_distance_to_segment(
                point,
                tuple(segment["start_display"]),
                tuple(segment["end_display"]),
            ),
        )
        if distance > tolerance:
            continue
        orientation_support = 1.0 if segment["axis"] == target_axis else 0.72 if segment["axis"] == "oblique" else 0.60
        score = orientation_support * max(0.0, 1.0 - distance / (tolerance + 1e-9))
        vertex_tolerance = min(tolerance, 1.5)
        vertex_refs = sorted(
            vertex_id
            for vertex_id, endpoint in (
                (segment["start_vertex_id"], segment["start_display"]),
                (segment["end_vertex_id"], segment["end_display"]),
            )
            if math.dist(point, tuple(endpoint)) <= vertex_tolerance
        )
        contour_refs = sorted(
            {
                *contour_by_drawing.get(segment["drawing_ref"], []),
                *contour_by_drawing.get(segment["primitive_ref"], []),
                *contour_by_drawing.get(segment["topology_segment_ref"], []),
            }
        )
        candidates.append(
            {
                **segment,
                "distance_points": round(distance, 4),
                "score": round(score, 4),
                "contour_refs": contour_refs,
                "view_refs": _view_refs(point, views),
                "geometry_anchor_refs": vertex_refs or [segment["topology_segment_ref"]],
            }
        )
    candidates.sort(key=lambda item: (-item["score"], item["distance_points"], item["primitive_ref"], item["segment_index"]))
    deduplicated: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        current = deduplicated.get(candidate["primitive_ref"])
        if current is None or candidate["score"] > current["score"]:
            deduplicated[candidate["primitive_ref"]] = candidate
    return sorted(deduplicated.values(), key=lambda item: (-item["score"], item["distance_points"], item["primitive_ref"]))[:8]


def _unique_target(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if candidates[0]["score"] - candidates[1]["score"] >= 0.22:
        return candidates[0]
    return None


def _unique_native_target(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    target = _unique_target(candidates)
    if target is not None:
        return target
    axes = {item.get("axis") for item in candidates}
    if len(axes) == 1 and next(iter(axes)) in {"horizontal", "vertical"}:
        axis = next(iter(axes))
        cross_index = 1 if axis == "horizontal" else 0
        along_index = 0 if axis == "horizontal" else 1
        cross_values = [
            (float(item["start_display"][cross_index]) + float(item["end_display"][cross_index])) / 2.0
            for item in candidates
        ]
        intervals = [
            sorted(
                (
                    float(item["start_display"][along_index]),
                    float(item["end_display"][along_index]),
                )
            )
            for item in candidates
        ]
        common_start = max(item[0] for item in intervals)
        common_end = min(item[1] for item in intervals)
        if max(cross_values) - min(cross_values) <= 0.25 and common_start <= common_end + 0.25:
            return candidates[0]
    return None


def _cluster_endpoint_rows(rows: list[dict[str, Any]], tolerance: float = 1.25) -> list[list[dict[str, Any]]]:
    clusters: list[list[dict[str, Any]]] = []
    for row in sorted(rows, key=lambda item: (*item["point_display"], item["dimension_ref"], item["endpoint_index"])):
        cluster = next(
            (
                candidate
                for candidate in clusters
                if math.dist(
                    tuple(row["point_display"]),
                    (
                        sum(item["point_display"][0] for item in candidate) / len(candidate),
                        sum(item["point_display"][1] for item in candidate) / len(candidate),
                    ),
                )
                <= tolerance
            ),
            None,
        )
        if cluster is None:
            cluster = []
            clusters.append(cluster)
        cluster.append(row)
    return clusters


def _solve_endpoint_clusters(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve shared dimension endpoints against page-global vertices."""

    summaries = []
    for cluster_index, cluster in enumerate(_cluster_endpoint_rows(rows), start=1):
        anchor_support: dict[str, set[tuple[str, int]]] = defaultdict(set)
        anchor_score: dict[str, float] = defaultdict(float)
        for row in cluster:
            observation_key = (row["dimension_ref"], row["endpoint_index"])
            best_by_anchor: dict[str, float] = {}
            for candidate in row["candidates"]:
                for anchor in candidate.get("geometry_anchor_refs", []):
                    if not str(anchor).startswith("geometry_vertex."):
                        continue
                    best_by_anchor[anchor] = max(best_by_anchor.get(anchor, 0.0), float(candidate["score"]))
            for anchor, score in best_by_anchor.items():
                anchor_support[anchor].add(observation_key)
                anchor_score[anchor] += score
        ranked = sorted(
            anchor_support,
            key=lambda anchor: (-len(anchor_support[anchor]), -anchor_score[anchor], anchor),
        )
        winner = None
        if ranked:
            best = ranked[0]
            second = ranked[1] if len(ranked) > 1 else None
            support_margin = len(anchor_support[best]) - (len(anchor_support[second]) if second else 0)
            score_margin = anchor_score[best] - (anchor_score[second] if second else 0.0)
            if len(cluster) > 1 and len(anchor_support[best]) >= 2 and (support_margin > 0 or score_margin >= 0.22):
                winner = best
        for row in cluster:
            matching = [
                candidate
                for candidate in row["candidates"]
                if winner is not None and winner in candidate.get("geometry_anchor_refs", [])
            ]
            target = max(matching, key=lambda item: (item["score"], -item["distance_points"]), default=None)
            if target is None:
                target = _unique_native_target(row["candidates"])
            row["selected"] = target
            row["selected_geometry_anchor_ref"] = (
                winner
                if target is not None and winner in target.get("geometry_anchor_refs", [])
                else next(iter(target.get("geometry_anchor_refs", [])), None) if target is not None else None
            )
            row["selection_basis"] = (
                "shared page-global vertex supported by multiple dimension endpoints"
                if target is not None and winner is not None and winner in target.get("geometry_anchor_refs", [])
                else "unique local native target"
                if target is not None
                else "ambiguous native targets"
            )
        summaries.append(
            {
                "id": f"dimension_endpoint_cluster.{cluster_index:04d}",
                "point_display": [
                    round(sum(item["point_display"][0] for item in cluster) / len(cluster), 6),
                    round(sum(item["point_display"][1] for item in cluster) / len(cluster), 6),
                ],
                "observation_count": len(cluster),
                "selected_geometry_anchor_ref": winner,
                "state": "resolved" if winner is not None else "local_or_unresolved",
                "dimension_endpoint_refs": [
                    f"{item['dimension_ref']}.endpoint[{item['endpoint_index']}]" for item in cluster
                ],
            }
        )
    return summaries


def _semantic_role(
    dimension: DimensionAttachment,
    selected: list[dict[str, Any] | None],
    common_contours: set[str],
    contours_by_id: dict[str, dict[str, Any]],
) -> str:
    if len(common_contours) != 1 or not all(selected):
        return "geometry_span"
    contour = contours_by_id[next(iter(common_contours))]
    box = fitz.Rect(contour["bbox_display"])
    coordinate_index = 0 if dimension.orientation == "horizontal" else 1
    contour_extent = box.width if coordinate_index == 0 else box.height
    measured_extent = abs(
        float(dimension.measured_points[1][coordinate_index])
        - float(dimension.measured_points[0][coordinate_index])
    )
    tolerance = max(1.5, 0.02 * max(contour_extent, measured_extent, 1.0))
    return "overall_shared_axis_span" if abs(contour_extent - measured_extent) <= tolerance else "internal_contour_span"


def _unique_preliminary_view_owner(
    dimension: DimensionAttachment,
    views: Iterable[dict[str, Any]],
) -> str | None:
    """Return one smallest preliminary view that contains the whole chain."""

    rows = []
    text_box = fitz.Rect(dimension.text_bbox)
    text_center = fitz.Point(
        (text_box.x0 + text_box.x1) / 2.0,
        (text_box.y0 + text_box.y1) / 2.0,
    )
    for view in views:
        view_id = str(view.get("id"))
        box = fitz.Rect(view.get("bbox_display", ()))
        if box.is_empty:
            continue
        explicitly_local = dimension.attachment_id in set(map(str, view.get("dimension_refs", []) or []))
        padding = max(4.0, 0.025 * math.hypot(box.width, box.height))
        expanded = box + (-padding, -padding, padding, padding)
        targets_contained = all(fitz.Point(*point) in expanded for point in dimension.measured_points)
        text_contained = text_center in expanded
        if explicitly_local or (targets_contained and text_contained):
            rows.append((box.get_area(), view_id))
    rows.sort(key=lambda item: (item[0], item[1]))
    if not rows:
        return None
    if len(rows) > 1 and abs(rows[1][0] / max(rows[0][0], 1e-9) - 1.0) <= 0.05:
        return None
    return rows[0][1]


def _unique_view_from_refs(
    view_refs: Iterable[str],
    views: Iterable[dict[str, Any]],
) -> str | None:
    """Choose one smallest nested view, abstaining on peer competitors."""

    allowed = set(map(str, view_refs))
    rows = []
    for view in views:
        view_id = str(view.get("id"))
        if view_id not in allowed:
            continue
        box = fitz.Rect(view.get("bbox_display", ()))
        if not box.is_empty:
            rows.append((box.get_area(), view_id))
    rows.sort(key=lambda item: (item[0], item[1]))
    if not rows:
        return None
    if len(rows) > 1 and abs(rows[1][0] / max(rows[0][0], 1e-9) - 1.0) <= 0.05:
        return None
    return rows[0][1]


def resolve_dimension_ownership(
    page: fitz.Page,
    dimensions: Iterable[DimensionAttachment],
    views: Iterable[dict[str, Any]],
    contours: Iterable[dict[str, Any]],
    native_topology: dict[str, Any] | None = None,
    *,
    allowed_primitive_refs: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return fail-closed dimension-to-geometry ownership records."""

    views = list(views)
    contours = list(contours)
    contours_by_id = {str(item["id"]): item for item in contours}
    native_topology = native_topology or extract_page_topology(page)
    segments = _native_segments(page, native_topology)
    if allowed_primitive_refs is not None:
        allowed = set(map(str, allowed_primitive_refs))
        segments = [
            item
            for item in segments
            if str(item["primitive_ref"]) in allowed or str(item["drawing_ref"]) in allowed
        ]
    contour_by_drawing = _contour_index(contours)
    endpoint_rows_by_dimension: dict[str, list[dict[str, Any]]] = defaultdict(list)
    working = []
    for dimension in dimensions:
        for endpoint_index, measured_point in enumerate(dimension.measured_points):
            row = {
                "dimension_ref": dimension.attachment_id,
                "source_proposal_status": dimension.proposal_status or dimension.status,
                "endpoint_index": endpoint_index,
                "point_display": list(measured_point),
                "candidates": _endpoint_candidates(measured_point, dimension, segments, contour_by_drawing, views),
            }
            endpoint_rows_by_dimension[dimension.attachment_id].append(row)
        working.append(dimension)
    # Detector-qualified chains retain the exact ownership competition used by
    # the previous trusted path.  Lower-confidence proposals are still fully
    # evaluated, but in a separate pool so a newly admitted ambiguity cannot
    # displace a previously unique endpoint decision.
    trusted_refs = {
        dimension.attachment_id
        for dimension in working
        if (dimension.proposal_status or dimension.status) == "accepted"
    }
    trusted_rows = [
        row
        for dimension_ref, rows in endpoint_rows_by_dimension.items()
        if dimension_ref in trusted_refs
        for row in rows
    ]
    proposal_rows = [
        row
        for dimension_ref, rows in endpoint_rows_by_dimension.items()
        if dimension_ref not in trusted_refs
        for row in rows
    ]
    endpoint_clusters = [
        *_solve_endpoint_clusters(trusted_rows),
        *_solve_endpoint_clusters(proposal_rows),
    ]
    for cluster_index, cluster in enumerate(endpoint_clusters, start=1):
        cluster["id"] = f"dimension_endpoint_cluster.{cluster_index:04d}"
    attachments = []
    relations = []
    for dimension in working:
        endpoint_rows = []
        selected = []
        for row in endpoint_rows_by_dimension[dimension.attachment_id]:
            candidates = row["candidates"]
            target = row.get("selected")
            selected.append(target)
            endpoint_rows.append(
                {
                    "endpoint_index": row["endpoint_index"],
                    "point_display": row["point_display"],
                    "state": "resolved" if target is not None else "candidate" if candidates else "unknown",
                    "selected_primitive_ref": None if target is None else target["primitive_ref"],
                    "selected_topology_segment_ref": None if target is None else target["topology_segment_ref"],
                    "selected_geometry_anchor_ref": row.get("selected_geometry_anchor_ref"),
                    "selection_basis": row.get("selection_basis"),
                    "candidates": candidates,
                }
            )

        common_views: set[str] = set()
        common_contours: set[str] = set()
        if all(selected):
            common_views = set(selected[0]["view_refs"]) & set(selected[1]["view_refs"])
            common_contours = set(selected[0]["contour_refs"]) & set(selected[1]["contour_refs"])
        preliminary_view_owner = _unique_view_from_refs(common_views, views)
        unique_common_contour = next(iter(common_contours)) if len(common_contours) == 1 else None
        owner_refs = (
            [unique_common_contour]
            if unique_common_contour is not None and preliminary_view_owner is not None
            else [preliminary_view_owner]
            if preliminary_view_owner is not None
            else []
        )
        status = (
            "accepted"
            if all(selected) and preliminary_view_owner is not None
            else "candidate"
            if any(row["candidates"] for row in endpoint_rows)
            else "unresolved"
        )
        ownership_basis = (
            "unique_common_contour_inside_preliminary_view"
            if unique_common_contour is not None and preliminary_view_owner is not None
            else "unique_preliminary_view_from_native_endpoint_geometry"
            if preliminary_view_owner is not None
            else None
        )
        has_competing_endpoint_targets = any(row["state"] == "candidate" for row in endpoint_rows)
        legacy_explicit_view_fallback = (
            status != "accepted"
            and not has_competing_endpoint_targets
            and (dimension.proposal_status or dimension.status) == "accepted"
            and dimension.terminal_style != "terminal_less"
            and all(dimension.terminal_refs_by_endpoint)
        )
        if legacy_explicit_view_fallback:
            preliminary_view_owner = _unique_preliminary_view_owner(dimension, views)
            if preliminary_view_owner is not None:
                owner_refs = [preliminary_view_owner]
                status = "accepted"
                ownership_basis = "detector_qualified_explicit_chain_uniquely_scoped_to_preliminary_view"
        reason = None
        if status != "accepted":
            reason = (
                "measured endpoints touch multiple compatible primitives"
                if any(len(row["candidates"]) > 1 for row in endpoint_rows)
                else "both measured endpoints do not resolve uniquely inside one preliminary view"
            )
        ownership_id = f"dimension_ownership.{len(attachments) + 1:04d}"
        semantic_role = _semantic_role(dimension, selected, common_contours, contours_by_id)
        attachments.append(
            {
                "id": ownership_id,
                "dimension_ref": dimension.attachment_id,
                "value_mm": dimension.value_mm,
                "orientation": dimension.orientation,
                "status": status,
                "epistemic_state": "derived" if status == "accepted" else "unknown",
                "role": "contour_edge_span" if common_contours else "view_scoped_geometry_span_candidate",
                "semantic_role": semantic_role,
                "owner_entity_refs": owner_refs,
                "view_refs": [preliminary_view_owner] if preliminary_view_owner is not None else [],
                "ownership_basis": ownership_basis,
                "measured_endpoints": endpoint_rows,
                "primitive_refs": sorted({target["primitive_ref"] for target in selected if target is not None}),
                "topology_segment_refs": sorted(
                    {target["topology_segment_ref"] for target in selected if target is not None}
                ),
                "geometry_anchor_refs": [
                    row.get("selected_geometry_anchor_ref") for row in endpoint_rows
                ],
                "reason": reason,
            }
        )
        for owner_ref in owner_refs if status == "accepted" else []:
            relations.append(
                {
                    "id": f"dimension_relation.{len(relations) + 1:04d}",
                    "type": "dimension_of",
                    "from": dimension.attachment_id,
                    "to": owner_ref,
                    "state": "derived",
                    "basis": (
                        "detector-qualified explicit-terminal chain is uniquely scoped to one preliminary view; exact endpoint primitive ownership remains unresolved"
                        if ownership_basis
                        == "detector_qualified_explicit_chain_uniquely_scoped_to_preliminary_view"
                        else "dimension extension endpoints uniquely terminate on geometry inside one preliminary view"
                    ),
                    "evidence_refs": [ownership_id, *attachments[-1]["primitive_refs"]],
                }
            )

    conflicts = []
    by_anchor_pair: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for attachment in attachments:
        anchors = attachment.get("geometry_anchor_refs", [])
        if (
            attachment["status"] == "accepted"
            and len(anchors) == 2
            and all(str(anchor).startswith("geometry_vertex.") for anchor in anchors)
            and anchors[0] != anchors[1]
        ):
            by_anchor_pair[(attachment["orientation"], *sorted(anchors))].append(attachment)
    for key, rows in by_anchor_pair.items():
        values = [float(item["value_mm"]) for item in rows]
        tolerance = max(2.0, 0.02 * max(values))
        if max(values) - min(values) <= tolerance:
            continue
        conflict_id = f"dimension_global_conflict.{len(conflicts) + 1:03d}"
        conflicts.append(
            {
                "id": conflict_id,
                "state": "conflict",
                "geometry_anchor_pair": list(key[1:]),
                "dimension_refs": [item["dimension_ref"] for item in rows],
                "values_mm": values,
                "reason": "the same geometry-anchor pair has incompatible printed dimensions",
            }
        )
        rejected_ids = {item["id"] for item in rows}
        relations = [
            relation
            for relation in relations
            if not any(relation["id"].startswith("dimension_relation") and relation["from"] == item["dimension_ref"] for item in rows)
        ]
        for attachment in attachments:
            if attachment["id"] in rejected_ids:
                attachment["status"] = "candidate"
                attachment["epistemic_state"] = "unknown"
                attachment["owner_entity_refs"] = []
                attachment["reason"] = f"global dimension conflict {conflict_id}"

    status_counts = Counter(item["status"] for item in attachments)
    return {
        "schema_version": "0.1.0",
        "layer": "dimension_ownership",
        "status": "partial" if status_counts.get("accepted", 0) < len(attachments) else "resolved",
        "attachments": attachments,
        "relations": relations,
        "endpoint_clusters": endpoint_clusters,
        "conflicts": conflicts,
        "summary": {
            "dimension_count": len(attachments),
            "accepted_count": status_counts.get("accepted", 0),
            "candidate_count": status_counts.get("candidate", 0),
            "unresolved_count": status_counts.get("unresolved", 0),
            "endpoint_cluster_count": len(endpoint_clusters),
            "globally_resolved_endpoint_cluster_count": sum(
                item["state"] == "resolved" for item in endpoint_clusters
            ),
            "global_conflict_count": len(conflicts),
        },
        "contract": {
            "all_dimension_proposals_are_evaluated": True,
            "candidate_proposals_cannot_displace_detector_qualified_endpoint_decisions": True,
            "numeric_proximity_alone_is_not_ownership": True,
            "both_measured_endpoints_required": True,
            "one_preliminary_view_owner_required": True,
            "legacy_explicit_terminal_view_fallback_is_recorded": True,
            "ambiguous_targets_abstain": True,
            "shared_endpoint_decisions_use_page_global_vertex_ids": True,
            "incompatible_dimensions_on_one_anchor_pair_abstain": True,
        },
    }


def _owned_metric_rows(
    dimensions: Iterable[DimensionAttachment],
    attachments_by_dimension: Mapping[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for dimension in dimensions:
        ownership_rows = attachments_by_dimension.get(dimension.attachment_id, [])
        if len(ownership_rows) != 1:
            continue
        ownership = ownership_rows[0]
        view_refs = list(ownership.get("view_refs", []) or [])
        legacy_explicit_view_fallback = (
            dimension.terminal_style != "terminal_less"
            and ownership.get("ownership_basis")
            == "detector_qualified_explicit_chain_uniquely_scoped_to_preliminary_view"
        )
        if (
            ownership.get("status") != "accepted"
            or len(view_refs) != 1
            or (
                not legacy_explicit_view_fallback
                and not all(
                    endpoint.get("state") == "resolved"
                    for endpoint in ownership.get("measured_endpoints", [])
                )
            )
        ):
            continue
        along_axis = 0 if dimension.orientation == "horizontal" else 1
        cross_axis = 1 - along_axis
        coordinates = sorted(float(point[along_axis]) for point in dimension.measured_points)
        rows.append(
            {
                "dimension": dimension,
                "dimension_ref": dimension.attachment_id,
                "view_ref": view_refs[0],
                "orientation": dimension.orientation,
                "start": coordinates[0],
                "end": coordinates[1],
                "span": coordinates[1] - coordinates[0],
                "cross_coordinates": [float(point[cross_axis]) for point in dimension.measured_points],
                "extension_direction": (
                    1
                    if sum(float(point[cross_axis]) for point in dimension.measured_points) / 2
                    > (dimension.baseline.start[cross_axis] + dimension.baseline.end[cross_axis]) / 2
                    else -1
                ),
                "value_mm": float(dimension.value_mm),
                "scale": float(dimension.scale_points_per_mm),
            }
        )
    return rows


def _local_scale_certificates(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    certificates = {}
    for row in rows:
        support = [
            other
            for other in rows
            if other["view_ref"] == row["view_ref"]
            and abs(other["scale"] / max(row["scale"], 1e-12) - 1.0) <= 0.04
        ]
        refs = sorted({str(item["dimension_ref"]) for item in support})
        certificates[row["dimension_ref"]] = {
            "view_ref": row["view_ref"],
            "scale_points_per_mm": round(row["scale"], 8),
            "support_count": len(refs),
            "supporting_dimension_refs": refs,
            "relative_tolerance": 0.04,
            "passed": len(refs) >= 2,
        }
    return certificates


def _terminal_less_redundancy(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Certify terminal-less dimensions by arithmetic closure or repeated pitch."""

    evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_scope: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        # Collinear stations on opposite sides of the measured geometry can
        # describe unrelated objects (for example a plan width chain beside a
        # support-clear-span chain).  They must never certify one another.
        by_scope[(row["view_ref"], row["orientation"], row["extension_direction"])].append(row)

    for scope_rows in by_scope.values():
        # Adjacent equal intervals with equal printed values form a repeated
        # pitch certificate. Coincident duplicate annotations do not.
        for index, left in enumerate(scope_rows):
            for right in scope_rows[index + 1 :]:
                if left["dimension"].terminal_style != "terminal_less" or right["dimension"].terminal_style != "terminal_less":
                    continue
                point_tolerance = max(1.5, 0.015 * max(left["span"], right["span"], 1.0))
                value_tolerance = max(2.0, 0.01 * max(left["value_mm"], right["value_mm"]))
                adjacent = (
                    abs(left["end"] - right["start"]) <= point_tolerance
                    or abs(right["end"] - left["start"]) <= point_tolerance
                )
                if (
                    adjacent
                    and abs(left["span"] - right["span"]) <= point_tolerance
                    and abs(left["value_mm"] - right["value_mm"]) <= value_tolerance
                ):
                    refs = sorted((left["dimension_ref"], right["dimension_ref"]))
                    for row in (left, right):
                        evidence[row["dimension_ref"]].append(
                            {"kind": "repeated_pitch", "supporting_dimension_refs": refs}
                        )

        # An overall interval may close against one unique contiguous chain of
        # two or more shorter intervals whose printed values add to the total.
        for overall in scope_rows:
            point_tolerance = max(1.5, 0.015 * max(overall["span"], 1.0))
            children = [
                row
                for row in scope_rows
                if row["dimension_ref"] != overall["dimension_ref"]
                and row["span"] < overall["span"] - point_tolerance
                and row["start"] >= overall["start"] - point_tolerance
                and row["end"] <= overall["end"] + point_tolerance
            ]
            paths: set[tuple[str, ...]] = set()

            def walk(position: float, path: tuple[str, ...], total: float) -> None:
                if len(paths) > 1:
                    return
                if abs(position - overall["end"]) <= point_tolerance:
                    value_tolerance = max(2.0, 0.01 * overall["value_mm"])
                    if len(path) >= 2 and abs(total - overall["value_mm"]) <= value_tolerance:
                        paths.add(path)
                    return
                for child in sorted(children, key=lambda item: (item["start"], item["end"], item["dimension_ref"])):
                    if child["dimension_ref"] in path:
                        continue
                    if abs(child["start"] - position) <= point_tolerance and child["end"] > position + point_tolerance:
                        walk(child["end"], (*path, child["dimension_ref"]), total + child["value_mm"])

            walk(overall["start"], (), 0.0)
            if len(paths) != 1:
                continue
            path = next(iter(paths))
            refs = [overall["dimension_ref"], *path]
            by_ref = {row["dimension_ref"]: row for row in scope_rows}
            term_values_mm = [float(by_ref[ref]["value_mm"]) for ref in path]
            certificate = {
                "kind": "arithmetic_chain",
                "supporting_dimension_refs": refs,
                "overall_dimension_ref": overall["dimension_ref"],
                "term_dimension_refs": list(path),
                "term_values_mm": term_values_mm,
                "total_value_mm": float(overall["value_mm"]),
                "arithmetic_residual_mm": round(
                    sum(term_values_mm) - float(overall["value_mm"]), 6
                ),
                "extension_direction": int(overall["extension_direction"]),
            }
            for ref in refs:
                if by_ref[ref]["dimension"].terminal_style == "terminal_less":
                    evidence[ref].append(dict(certificate))

    result = {}
    for dimension_ref, rows_for_dimension in evidence.items():
        rows_for_dimension.sort(
            key=lambda item: (
                0 if item["kind"] == "arithmetic_chain" else 1,
                item["supporting_dimension_refs"],
            )
        )
        result[dimension_ref] = rows_for_dimension[0]
    return result


def adjudicate_dimension_proposals(
    dimensions: Iterable[DimensionAttachment],
    ownership: dict[str, Any],
) -> tuple[tuple[DimensionAttachment, ...], dict[str, Any], dict[str, Any]]:
    """Combine proposal evidence with unique ownership before acceptance.

    A detector score never accepts a dimension by itself. Both extension
    targets must close on native geometry inside one preliminary view, and the
    scale must repeat in that view. Terminal-less chains additionally need a
    closed arithmetic chain or adjacent repeated-pitch certificate.
    """

    dimensions = tuple(dimensions)
    result = deepcopy(ownership)
    attachments_by_dimension: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attachment in result.get("attachments", []) or []:
        attachments_by_dimension[str(attachment.get("dimension_ref"))].append(attachment)

    metric_rows = _owned_metric_rows(dimensions, attachments_by_dimension)
    local_scale_certificates = _local_scale_certificates(metric_rows)
    redundancy_certificates = _terminal_less_redundancy(metric_rows)
    records = []
    accepted_dimension_refs: set[str] = set()
    adjudicated = []
    for dimension in dimensions:
        ownership_rows = attachments_by_dimension.get(dimension.attachment_id, [])
        uniquely_owned = len(ownership_rows) == 1 and ownership_rows[0].get("status") == "accepted"
        ownership_row = ownership_rows[0] if len(ownership_rows) == 1 else None
        along_axis = 0 if dimension.orientation == "horizontal" else 1
        measured_span = abs(
            dimension.measured_points[1][along_axis]
            - dimension.measured_points[0][along_axis]
        )
        aligned = dimension.endpoint_alignment_residual <= max(3.0, measured_span * 0.03)
        different_cross_axis_targets = not aligned
        expected_extension_axis = "vertical" if dimension.orientation == "horizontal" else "horizontal"
        perpendicular_extensions = all(
            _axis(line.start, line.end) == expected_extension_axis
            for line in dimension.extension_lines
        )
        complete_terminals = all(dimension.terminal_refs_by_endpoint)
        terminal_less = dimension.terminal_style == "terminal_less"
        text_reliable = (
            dimension.text_method == "native_pdf_text"
            or dimension.text_confidence >= 0.80
        )
        exact_endpoints = bool(
            ownership_row
            and len(ownership_row.get("measured_endpoints", [])) == 2
            and all(
                endpoint.get("state") == "resolved"
                for endpoint in ownership_row.get("measured_endpoints", [])
            )
        )
        legacy_explicit_view_fallback = bool(
            ownership_row
            and not terminal_less
            and complete_terminals
            and ownership_row.get("ownership_basis")
            == "detector_qualified_explicit_chain_uniquely_scoped_to_preliminary_view"
        )
        unique_preliminary_view = bool(
            ownership_row and len(ownership_row.get("view_refs", []) or []) == 1
        )
        local_scale = local_scale_certificates.get(
            dimension.attachment_id,
            {
                "view_ref": None,
                "scale_points_per_mm": round(float(dimension.scale_points_per_mm), 8),
                "support_count": 0,
                "supporting_dimension_refs": [],
                "relative_tolerance": 0.04,
                "passed": False,
            },
        )
        redundancy = redundancy_certificates.get(dimension.attachment_id)
        legacy_scale_support_required = (
            2
            if dimension.text_method == "geometry_gated_ocr" and dimension.text_confidence >= 0.80
            else 3
        )
        legacy_explicit_page_scale = bool(
            not terminal_less
            and complete_terminals
            and (dimension.proposal_status or dimension.status) == "accepted"
            and dimension.scale_support >= legacy_scale_support_required
        )
        scale_certificate_passed = local_scale["passed"] or legacy_explicit_page_scale
        text_score = 0.40 if dimension.text_method == "native_pdf_text" else 0.35 if dimension.text_confidence >= 0.80 else 0.30
        terminal_score = 0.10 if complete_terminals else 0.05 if dimension.terminal_refs else 0.0
        adjudicated_score = text_score + 0.25 + 0.15 + terminal_score + (0.10 if scale_certificate_passed else 0.0)
        terminal_certificate = complete_terminals or (terminal_less and redundancy is not None)
        proposal_complete = (
            perpendicular_extensions
            and text_reliable
            and scale_certificate_passed
            and terminal_certificate
            and adjudicated_score >= 0.90
        )
        accepted = (
            uniquely_owned
            and (exact_endpoints or legacy_explicit_view_fallback)
            and unique_preliminary_view
            and proposal_complete
        )
        final_status = "accepted" if accepted else "ambiguous"
        adjudicated.append(
            replace(
                dimension,
                status=final_status,
                score=adjudicated_score,
                scale_support=local_scale["support_count"],
            )
        )
        if accepted:
            accepted_dimension_refs.add(dimension.attachment_id)
        evidence_refs = sorted(
            {
                dimension.attachment_id,
                *(dimension.terminal_refs if complete_terminals else ()),
                *(
                    [str(ownership_row.get("id"))]
                    if ownership_row is not None and ownership_row.get("id")
                    else []
                ),
                *(
                    str(ref)
                    for ref in (ownership_row or {}).get("geometry_anchor_refs", [])
                    if ref
                ),
                *((redundancy or {}).get("supporting_dimension_refs", [])),
                *local_scale.get("supporting_dimension_refs", []),
            }
        )
        reasons = []
        if not uniquely_owned:
            reasons.append("measured endpoints do not close on one geometry owner")
        if not exact_endpoints and not legacy_explicit_view_fallback:
            reasons.append("both extension targets do not resolve to exact native geometry")
        if not unique_preliminary_view:
            reasons.append("extension targets do not share one unique preliminary view")
        if not perpendicular_extensions:
            reasons.append("extension lines are not perpendicular to the dimension baseline")
        if not scale_certificate_passed:
            reasons.append("view-local scale does not repeat")
        if terminal_less and redundancy is None:
            reasons.append("terminal-less chain lacks arithmetic or repeated-pitch redundancy")
        if not terminal_less and not complete_terminals:
            reasons.append("explicit terminal evidence is incomplete")
        if not text_reliable or adjudicated_score < 0.90:
            reasons.append("dimension-chain evidence does not meet the 0.90 threshold")
        records.append(
            {
                "id": f"dimension_adjudication.{len(records) + 1:04d}",
                "dimension_ref": dimension.attachment_id,
                "proposal_status": dimension.proposal_status or dimension.status,
                "ownership_ref": ownership_row.get("id") if ownership_row else None,
                "view_refs": list((ownership_row or {}).get("view_refs", [])),
                "owner_entity_refs": list((ownership_row or {}).get("owner_entity_refs", [])),
                "status": final_status,
                "state": "derived" if accepted else "unknown",
                "evidence_refs": evidence_refs,
                "reason": None if accepted else "; ".join(reasons),
                "certificate": {
                    "proposal_complete": proposal_complete,
                    "unique_geometry_owner": uniquely_owned,
                    "ownership_basis": (ownership_row or {}).get("ownership_basis"),
                    "exact_endpoint_geometry_resolved": exact_endpoints,
                    "legacy_explicit_terminal_view_fallback": legacy_explicit_view_fallback,
                    "unique_preliminary_view_owner": unique_preliminary_view,
                    "perpendicular_extension_pair": perpendicular_extensions,
                    "terminal_style": dimension.terminal_style,
                    "complete_terminal_pair": complete_terminals,
                    "terminal_less_redundancy": redundancy,
                    "view_local_scale": local_scale,
                    "legacy_explicit_page_scale": {
                        "passed": legacy_explicit_page_scale,
                        "support_count": dimension.scale_support,
                        "required_support": legacy_scale_support_required,
                    },
                    "scale_certificate_basis": (
                        "view_local_repetition"
                        if local_scale["passed"]
                        else "legacy_explicit_terminal_page_consensus"
                        if legacy_explicit_page_scale
                        else None
                    ),
                    "global_score_threshold": 0.90,
                    "adjudicated_score": round(adjudicated_score, 6),
                    "projected_endpoints_permitted": True,
                    "different_cross_axis_targets": different_cross_axis_targets,
                    "endpoint_alignment_observation": aligned,
                    "text_reliable": text_reliable,
                    "legacy_solver_eligible": bool(
                        accepted and (dimension.proposal_status or dimension.status) == "accepted"
                    ),
                },
            }
        )

    for attachment in result.get("attachments", []) or []:
        dimension_ref = str(attachment.get("dimension_ref"))
        if dimension_ref in accepted_dimension_refs and attachment.get("status") == "accepted":
            attachment["adjudication_status"] = "accepted"
            attachment["epistemic_state"] = "derived"
        else:
            attachment["status"] = "candidate" if attachment.get("status") != "unresolved" else "unresolved"
            attachment["adjudication_status"] = "ambiguous"
            attachment["epistemic_state"] = "unknown"
            attachment["owner_entity_refs"] = []
            attachment["reason"] = attachment.get("reason") or "dimension proposal did not pass ownership adjudication"
    result["relations"] = [
        relation
        for relation in result.get("relations", []) or []
        if str(relation.get("from")) in accepted_dimension_refs
    ]
    status_counts = Counter(item.get("status") for item in result.get("attachments", []) or [])
    result["status"] = "resolved" if records and len(accepted_dimension_refs) == len(records) else "partial"
    result["summary"].update(
        {
            "accepted_count": status_counts.get("accepted", 0),
            "candidate_count": status_counts.get("candidate", 0),
            "unresolved_count": status_counts.get("unresolved", 0),
        }
    )
    result.setdefault("contract", {})["acceptance_requires_proposal_and_unique_ownership"] = True

    report = {
        "schema_version": "0.1.0",
        "layer": "dimension_adjudication",
        "status": "resolved" if records and len(accepted_dimension_refs) == len(records) else "partial",
        "records": records,
        "view_local_scale_certificates": [
            {"dimension_ref": dimension_ref, **certificate}
            for dimension_ref, certificate in sorted(local_scale_certificates.items())
        ],
        "terminal_less_redundancy_certificates": [
            {"dimension_ref": dimension_ref, **certificate}
            for dimension_ref, certificate in sorted(redundancy_certificates.items())
        ],
        "summary": {
            "proposal_count": len(records),
            "accepted_count": len(accepted_dimension_refs),
            "ambiguous_count": len(records) - len(accepted_dimension_refs),
            "ownership_evaluated_count": sum(bool(item["ownership_ref"]) for item in records),
            "view_local_scale_pass_count": sum(
                item["passed"] for item in local_scale_certificates.values()
            ),
            "terminal_less_redundancy_count": len(redundancy_certificates),
        },
        "contract": {
            "detector_score_alone_never_accepts": True,
            "unique_geometry_ownership_required": True,
            "unique_preliminary_view_required": True,
            "terminal_less_exact_endpoint_geometry_required": True,
            "legacy_explicit_terminal_view_fallback_is_recorded": True,
            "view_local_repeated_scale_required": True,
            "legacy_detector_qualified_explicit_terminals_retain_page_scale_consensus": True,
            "terminal_less_requires_arithmetic_or_repeated_pitch_redundancy": True,
            "projected_cross_axis_endpoints_allowed_when_uniquely_owned": True,
            "global_score_threshold_unchanged": 0.90,
            "newly_adjudicated_dimensions_require_relation_scope_before_legacy_solver_use": True,
            "all_proposals_reach_ownership": len(result.get("attachments", []) or []) == len(dimensions),
            "schedule_values_used": False,
        },
    }
    return tuple(adjudicated), result, report


def _scope_record_id(kind: str, scope_ref: str, index: int) -> str:
    scope_digest = sha256(scope_ref.encode("utf-8")).hexdigest()[:12]
    return f"{kind}.scope_{scope_digest}.{index:04d}"


def _dimension_annotation_refs(dimension: DimensionAttachment) -> set[str]:
    refs = {
        dimension.baseline.primitive_ref,
        *(line.primitive_ref for line in dimension.extension_lines),
        *dimension.terminal_refs,
    }
    return {str(ref).split(".item[", 1)[0] for ref in refs if ref}


def _dimension_is_inside_title_scope(
    dimension: DimensionAttachment,
    scope: Mapping[str, Any],
) -> bool:
    allowed = set(map(str, scope.get("primitive_refs", []) or []))
    excluded = set(map(str, scope.get("excluded_primitive_refs", []) or []))
    annotation_refs = _dimension_annotation_refs(dimension)
    if not annotation_refs or not annotation_refs <= allowed or annotation_refs & excluded:
        return False
    box = fitz.Rect(scope.get("bbox_display", ()))
    if box.is_empty:
        return False
    text = fitz.Rect(dimension.text_bbox)
    center = fitz.Point((text.x0 + text.x1) / 2.0, (text.y0 + text.y1) / 2.0)
    # Title segmentation is geometry-led. Numeric text may overhang a native
    # drawing boundary by sub-point font metrics while all three dimension
    # paths remain exact members of the scope.
    return center in (box + (-2.0, -2.0, 2.0, 2.0))


def _apply_projected_station_transfers(
    dimensions: Iterable[DimensionAttachment],
    ownership: dict[str, Any],
    *,
    scope_ref: str,
) -> list[dict[str, Any]]:
    """Reclose missing targets from a unique exact projected station.

    The transfer is title-scope local and does not claim that the donor and
    receiver points are one physical point. It only proves their common
    coordinate along a dimension axis for local arithmetic-chain replay.
    """

    dimensions_by_ref = {item.attachment_id: item for item in dimensions}
    attachments_by_ref = {
        str(item.get("dimension_ref")): item
        for item in ownership.get("attachments", []) or []
    }
    donors = []
    for dimension_ref, attachment in attachments_by_ref.items():
        dimension = dimensions_by_ref.get(dimension_ref)
        if dimension is None:
            continue
        along_axis = 0 if dimension.orientation == "horizontal" else 1
        cross_axis = 1 - along_axis
        for endpoint in attachment.get("measured_endpoints", []) or []:
            anchor = endpoint.get("selected_geometry_anchor_ref")
            if endpoint.get("state") != "resolved" or not anchor:
                continue
            point = list(map(float, endpoint.get("point_display", [])))
            if len(point) != 2:
                continue
            donors.append(
                {
                    "dimension_ref": dimension_ref,
                    "endpoint_index": int(endpoint.get("endpoint_index", 0)),
                    "orientation": dimension.orientation,
                    "scale": float(dimension.scale_points_per_mm),
                    "along": point[along_axis],
                    "cross": point[cross_axis],
                    "point_display": point,
                    "anchor_ref": str(anchor),
                    "primitive_ref": endpoint.get("selected_primitive_ref"),
                    "topology_segment_ref": endpoint.get("selected_topology_segment_ref"),
                }
            )

    transfers = []
    for dimension_ref, attachment in attachments_by_ref.items():
        dimension = dimensions_by_ref.get(dimension_ref)
        if dimension is None:
            continue
        along_axis = 0 if dimension.orientation == "horizontal" else 1
        cross_axis = 1 - along_axis
        tolerance = max(1.5, 0.015 * abs(
            float(dimension.measured_points[1][along_axis])
            - float(dimension.measured_points[0][along_axis])
        ))
        for endpoint in attachment.get("measured_endpoints", []) or []:
            if endpoint.get("state") == "resolved":
                continue
            point = list(map(float, endpoint.get("point_display", [])))
            if len(point) != 2:
                continue
            matches = [
                donor
                for donor in donors
                if donor["dimension_ref"] != dimension_ref
                and donor["orientation"] == dimension.orientation
                and abs(donor["scale"] / max(float(dimension.scale_points_per_mm), 1e-12) - 1.0) <= 0.04
                and abs(donor["along"] - point[along_axis]) <= tolerance
            ]
            anchors = {item["anchor_ref"] for item in matches}
            if len(anchors) != 1:
                continue
            selected = min(
                matches,
                key=lambda item: (
                    abs(item["along"] - point[along_axis]),
                    abs(item["cross"] - point[cross_axis]),
                    item["dimension_ref"],
                    item["endpoint_index"],
                ),
            )
            transfer_id = _scope_record_id(
                "title_scope_projected_station_transfer",
                scope_ref,
                len(transfers) + 1,
            )
            endpoint.update(
                {
                    "state": "resolved",
                    "selected_primitive_ref": selected["primitive_ref"],
                    "selected_topology_segment_ref": selected["topology_segment_ref"],
                    "selected_geometry_anchor_ref": selected["anchor_ref"],
                    "selection_basis": "unique exact same-axis station inside one title scope",
                    "projected_station_transfer_ref": transfer_id,
                }
            )
            transfers.append(
                {
                    "id": transfer_id,
                    "state": "derived",
                    "scope_ref": scope_ref,
                    "receiver_dimension_ref": dimension_ref,
                    "receiver_endpoint_index": int(endpoint.get("endpoint_index", 0)),
                    "receiver_point_display": point,
                    "donor_dimension_ref": selected["dimension_ref"],
                    "donor_endpoint_index": selected["endpoint_index"],
                    "donor_point_display": selected["point_display"],
                    "geometry_anchor_ref": selected["anchor_ref"],
                    "along_axis_residual_points": round(
                        abs(selected["along"] - point[along_axis]), 6
                    ),
                    "cross_axis_offset_points": round(
                        selected["cross"] - point[cross_axis], 6
                    ),
                    "physical_point_identity_established": False,
                    "quantity_eligible": False,
                    "evidence_refs": [
                        dimension_ref,
                        selected["dimension_ref"],
                        selected["anchor_ref"],
                    ],
                }
            )
        endpoints = attachment.get("measured_endpoints", []) or []
        if len(endpoints) == 2 and all(item.get("state") == "resolved" for item in endpoints):
            attachment["status"] = "accepted"
            attachment["epistemic_state"] = "derived"
            attachment["view_refs"] = [scope_ref]
            attachment["owner_entity_refs"] = [scope_ref]
            if any(item.get("projected_station_transfer_ref") for item in endpoints):
                attachment["ownership_basis"] = "title_scope_projected_station_transfer"
            attachment["geometry_anchor_refs"] = [
                item.get("selected_geometry_anchor_ref") for item in endpoints
            ]
            attachment["primitive_refs"] = sorted(
                {
                    str(item.get("selected_primitive_ref"))
                    for item in endpoints
                    if item.get("selected_primitive_ref")
                }
            )
            attachment["topology_segment_refs"] = sorted(
                {
                    str(item.get("selected_topology_segment_ref"))
                    for item in endpoints
                    if item.get("selected_topology_segment_ref")
                }
            )
            attachment["reason"] = None
    return transfers


def _promote_scope_local_repeated_measurements(
    dimensions: tuple[DimensionAttachment, ...],
    ownership: dict[str, Any],
    report: dict[str, Any],
    *,
    scope_ref: str,
) -> tuple[tuple[DimensionAttachment, ...], list[dict[str, Any]]]:
    """Accept exact repeated measurements only in the local certificate."""

    dimensions_by_ref = {item.attachment_id: item for item in dimensions}
    attachments_by_ref = {
        str(item.get("dimension_ref")): item
        for item in ownership.get("attachments", []) or []
    }
    records_by_ref = {
        str(item.get("dimension_ref")): item for item in report.get("records", []) or []
    }
    scale_by_ref = {
        str(item.get("dimension_ref")): item
        for item in report.get("view_local_scale_certificates", []) or []
    }
    eligible = []
    for dimension_ref, dimension in dimensions_by_ref.items():
        attachment = attachments_by_ref.get(dimension_ref, {})
        endpoints = attachment.get("measured_endpoints", []) or []
        if (
            dimension.terminal_style != "terminal_less"
            or len(endpoints) != 2
            or not all(item.get("state") == "resolved" for item in endpoints)
            or not scale_by_ref.get(dimension_ref, {}).get("passed")
        ):
            continue
        along_axis = 0 if dimension.orientation == "horizontal" else 1
        cross_axis = 1 - along_axis
        along = sorted(float(point[along_axis]) for point in dimension.measured_points)
        cross = sum(float(point[cross_axis]) for point in dimension.measured_points) / 2.0
        eligible.append(
            {
                "dimension_ref": dimension_ref,
                "dimension": dimension,
                "orientation": dimension.orientation,
                "value_mm": float(dimension.value_mm),
                "span_points": along[1] - along[0],
                "start": along[0],
                "end": along[1],
                "cross": cross,
                "scale": float(dimension.scale_points_per_mm),
            }
        )

    groups: list[list[dict[str, Any]]] = []
    consumed: set[str] = set()
    for left in eligible:
        if left["dimension_ref"] in consumed:
            continue
        matches = [left]
        for right in eligible:
            if right["dimension_ref"] == left["dimension_ref"]:
                continue
            point_tolerance = max(1.5, 0.015 * max(left["span_points"], right["span_points"], 1.0))
            value_tolerance = max(2.0, 0.01 * max(left["value_mm"], right["value_mm"]))
            if (
                right["orientation"] == left["orientation"]
                and abs(right["value_mm"] - left["value_mm"]) <= value_tolerance
                and abs(right["span_points"] - left["span_points"]) <= point_tolerance
                and abs(right["start"] - left["start"]) <= point_tolerance
                and abs(right["end"] - left["end"]) <= point_tolerance
                and abs(right["cross"] - left["cross"]) > 2.0 * point_tolerance
                and abs(right["scale"] / max(left["scale"], 1e-12) - 1.0) <= 0.04
            ):
                matches.append(right)
        unique = {item["dimension_ref"]: item for item in matches}
        if len(unique) >= 2:
            group = [unique[ref] for ref in sorted(unique)]
            groups.append(group)
            consumed.update(unique)

    certificates = []
    promoted_refs = set()
    for group in groups:
        refs = [item["dimension_ref"] for item in group]
        certificate = {
            "id": _scope_record_id(
                "title_scope_repeated_projected_measurement",
                scope_ref,
                len(certificates) + 1,
            ),
            "kind": "repeated_projected_measurement",
            "scope_ref": scope_ref,
            "state": "derived",
            "supporting_dimension_refs": refs,
            "value_mm": group[0]["value_mm"],
            "orientation": group[0]["orientation"],
            "physical_instance_identity_established": False,
            "additive_count_established": False,
            "quantity_eligible": False,
            "evidence_refs": refs,
        }
        certificates.append(certificate)
        for ref in refs:
            promoted_refs.add(ref)
            record = records_by_ref[ref]
            record["status"] = "accepted"
            record["state"] = "derived"
            record["reason"] = None
            record["certificate"]["terminal_less_redundancy"] = certificate
            record["certificate"]["scope_local_only"] = True
            record["certificate"]["legacy_solver_eligible"] = False
            attachment = attachments_by_ref[ref]
            attachment["status"] = "accepted"
            attachment["adjudication_status"] = "accepted"
            attachment["epistemic_state"] = "derived"
            attachment["owner_entity_refs"] = [scope_ref]
            attachment["view_refs"] = [scope_ref]
            attachment["reason"] = None

    promoted_dimensions = tuple(
        replace(item, status="accepted", score=max(float(item.score), 0.90))
        if item.attachment_id in promoted_refs
        else item
        for item in dimensions
    )
    accepted_refs = {
        str(item.get("dimension_ref"))
        for item in report.get("records", []) or []
        if item.get("status") == "accepted"
    }
    report["summary"]["accepted_count"] = len(accepted_refs)
    report["summary"]["ambiguous_count"] = len(dimensions) - len(accepted_refs)
    report["summary"]["scope_local_repeated_measurement_count"] = len(certificates)
    ownership["summary"]["accepted_count"] = len(accepted_refs)
    ownership["summary"]["candidate_count"] = sum(
        item.get("status") == "candidate" for item in ownership.get("attachments", []) or []
    )
    ownership["summary"]["unresolved_count"] = sum(
        item.get("status") == "unresolved" for item in ownership.get("attachments", []) or []
    )
    return promoted_dimensions, certificates


def reclose_title_scope_dimensions(
    page: fitz.Page,
    dimensions: Iterable[DimensionAttachment],
    global_ownership: Mapping[str, Any],
    global_adjudication: Mapping[str, Any],
    title_segmentation: Mapping[str, Any],
    target_scope_refs: Iterable[str],
    contours: Iterable[dict[str, Any]],
    native_topology: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay existing proposals inside exact title-scope membership.

    Global ownership and adjudication are immutable inputs. Accepted local
    records are reconstruction evidence only and can never enter legacy solid
    solvers or quantity aggregation directly.
    """

    dimensions = tuple(dimensions)
    target_refs = set(map(str, target_scope_refs))
    scopes = {
        str(item.get("id")): item
        for item in title_segmentation.get("segments", []) or []
        if item.get("state") == "resolved" and str(item.get("id")) in target_refs
    }
    global_ownership_by_ref = {
        str(item.get("dimension_ref")): item
        for item in global_ownership.get("attachments", []) or []
    }
    global_adjudication_by_ref = {
        str(item.get("dimension_ref")): item
        for item in global_adjudication.get("records", []) or []
    }
    scope_results = []
    for scope_ref in sorted(target_refs):
        scope = scopes.get(scope_ref)
        if scope is None:
            continue
        local_dimensions = tuple(
            item for item in dimensions if _dimension_is_inside_title_scope(item, scope)
        )
        if len(local_dimensions) < 2:
            scope_results.append(
                {
                    "scope_ref": scope_ref,
                    "status": "insufficient_constraints",
                    "reason_code": "fewer_than_two_exact_scope_dimension_proposals",
                    "selected_proposal_refs": [item.attachment_id for item in local_dimensions],
                    "accepted_dimension_refs": [],
                    "projected_station_transfers": [],
                    "repeated_projected_measurement_certificates": [],
                    "local_ownership": None,
                    "local_adjudication": None,
                }
            )
            continue
        local_view = {
            "id": scope_ref,
            "bbox_display": list(scope["bbox_display"]),
            "dimension_refs": [item.attachment_id for item in local_dimensions],
        }
        allowed = sorted(
            set(map(str, scope.get("primitive_refs", []) or []))
            - set(map(str, scope.get("excluded_primitive_refs", []) or []))
        )
        local_ownership = resolve_dimension_ownership(
            page,
            local_dimensions,
            [local_view],
            contours,
            native_topology=dict(native_topology),
            allowed_primitive_refs=allowed,
        )
        transfers = _apply_projected_station_transfers(
            local_dimensions,
            local_ownership,
            scope_ref=scope_ref,
        )
        adjudicated, local_ownership, local_adjudication = adjudicate_dimension_proposals(
            local_dimensions,
            local_ownership,
        )
        adjudicated, repeat_certificates = _promote_scope_local_repeated_measurements(
            adjudicated,
            local_ownership,
            local_adjudication,
            scope_ref=scope_ref,
        )
        accepted_refs = [item.attachment_id for item in adjudicated if item.status == "accepted"]
        unresolved_observations = [
            {
                "dimension_ref": item.attachment_id,
                "value_mm": float(item.value_mm),
                "orientation": item.orientation,
                "status": item.status,
                "reason": next(
                    (
                        record.get("reason")
                        for record in local_adjudication.get("records", []) or []
                        if str(record.get("dimension_ref")) == item.attachment_id
                    ),
                    None,
                ),
                "quantity_eligible": False,
            }
            for item in adjudicated
            if item.status != "accepted"
        ]
        arithmetic_by_overall = {}
        for item in local_adjudication.get("terminal_less_redundancy_certificates", []) or []:
            if item.get("kind") == "arithmetic_chain":
                arithmetic_by_overall.setdefault(str(item.get("overall_dimension_ref")), item)

        ownership_id_map = {}
        for index, item in enumerate(local_ownership.get("attachments", []) or [], start=1):
            old = str(item.get("id"))
            new = _scope_record_id("title_scope_dimension_ownership", scope_ref, index)
            ownership_id_map[old] = new
            item["id"] = new
            item["scope_ref"] = scope_ref
            item["scope_local_only"] = True
            item["quantity_eligible"] = False
        for index, item in enumerate(local_adjudication.get("records", []) or [], start=1):
            item["id"] = _scope_record_id("title_scope_dimension_adjudication", scope_ref, index)
            item["ownership_ref"] = ownership_id_map.get(
                str(item.get("ownership_ref")), item.get("ownership_ref")
            )
            item["global_status"] = global_adjudication_by_ref.get(
                str(item.get("dimension_ref")), {}
            ).get("status")
            item["global_ownership_status"] = global_ownership_by_ref.get(
                str(item.get("dimension_ref")), {}
            ).get("status")
            item["scope_ref"] = scope_ref
            item["scope_local_only"] = True
            item["quantity_eligible"] = False
        scope_results.append(
            {
                "scope_ref": scope_ref,
                "status": "resolved_subset" if accepted_refs else "insufficient_constraints",
                "reason_code": None if accepted_refs else "no_scope_local_dimensions_accepted",
                "selected_proposal_refs": [item.attachment_id for item in local_dimensions],
                "accepted_dimension_refs": accepted_refs,
                "unresolved_metric_observations": unresolved_observations,
                "observed_value_multiplicity": {
                    str(value): count
                    for value, count in sorted(
                        Counter(float(item.value_mm) for item in local_dimensions).items()
                    )
                },
                "projected_station_transfers": transfers,
                "repeated_projected_measurement_certificates": repeat_certificates,
                "independent_arithmetic_chain_certificates": list(arithmetic_by_overall.values()),
                "independent_metric_check_count": len(arithmetic_by_overall) + len(repeat_certificates),
                "local_ownership": local_ownership,
                "local_adjudication": local_adjudication,
                "global_dimension_states_preserved": True,
                "quantity_eligible": False,
            }
        )

    accepted_count = sum(
        len(item.get("accepted_dimension_refs", []) or []) for item in scope_results
    )
    return {
        "schema_version": "0.1.0",
        "layer": "title_scope_local_dimension_reclosure",
        "page": page.number + 1,
        "status": "resolved_subset" if accepted_count else "insufficient_constraints",
        "scope_results": scope_results,
        "summary": {
            "target_scope_count": len(target_refs),
            "evaluated_scope_count": len(scope_results),
            "resolved_scope_count": sum(item["status"] == "resolved_subset" for item in scope_results),
            "accepted_scope_local_dimension_count": accepted_count,
            "projected_station_transfer_count": sum(
                len(item.get("projected_station_transfers", []) or []) for item in scope_results
            ),
            "independent_metric_check_count": sum(
                int(item.get("independent_metric_check_count", 0)) for item in scope_results
            ),
        },
        "contract": {
            "existing_dimension_proposals_replayed": True,
            "exact_title_scope_primitive_membership_required": True,
            "global_dimension_states_preserved": True,
            "projected_station_transfer_is_not_physical_point_identity": True,
            "scope_local_acceptance_cannot_enter_legacy_solvers": True,
            "physical_component_identity_established": False,
            "quantity_eligible": False,
            "schedule_values_used": False,
        },
    }


def _raster_chain_view_ref(
    chain: Mapping[str, Any],
    views: Iterable[dict[str, Any]],
) -> str | None:
    points = [
        tuple(map(float, point))
        for key in ("dimension_points_display", "measured_points_display")
        for point in chain.get(key, []) or []
    ]
    if len(points) != 4:
        return None
    containing = []
    for view in views:
        if view.get("source_modality") != "raster":
            continue
        rect = fitz.Rect(view["bbox_display"]) + (-2.0, -2.0, 2.0, 2.0)
        if all(fitz.Point(*point) in rect for point in points):
            containing.append((float(rect.get_area()), str(view["id"])))
    if not containing:
        return None
    containing.sort()
    if len(containing) > 1 and abs(containing[1][0] - containing[0][0]) <= 1e-6:
        return None
    return containing[0][1]


def _raster_scale_observation(chain: Mapping[str, Any], value: float) -> float | None:
    points = chain.get("dimension_points_display", []) or []
    if len(points) != 2 or value <= 0:
        return None
    axis = 0 if chain.get("orientation") == "horizontal" else 1
    span = abs(float(points[1][axis]) - float(points[0][axis]))
    return span / value if span > 0 else None


def _raster_scale_consensus(
    candidates: list[dict[str, Any]],
    *,
    relative_tolerance: float = 0.04,
) -> dict[str, dict[str, Any]]:
    """Accept only repeated, mutually compatible scale observations per view."""

    by_view: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        if candidate["view_ref"] and candidate["scale_points_per_mm_observation"] is not None:
            by_view[candidate["view_ref"]].append(candidate)
    result: dict[str, dict[str, Any]] = {}
    for view_ref, rows in by_view.items():
        scales = [float(item["scale_points_per_mm_observation"]) for item in rows]
        compatible = len(rows) >= 2 and max(scales) - min(scales) <= relative_tolerance * max(scales)
        consensus = sum(scales) / len(scales) if compatible else None
        state = "accepted" if compatible else "unknown"
        reason = (
            None
            if compatible
            else "one local scale observation cannot establish metric scale"
            if len(rows) == 1
            else "local raster dimension scale observations conflict"
        )
        result[view_ref] = {
            "view_ref": view_ref,
            "status": state,
            "scale_points_per_mm": None if consensus is None else round(consensus, 8),
            "supporting_dimension_chain_refs": [item["dimension_chain_ref"] for item in rows],
            "observation_count": len(rows),
            "relative_tolerance": relative_tolerance,
            "reason": reason,
        }
    return result


def _raster_endpoint_candidates(
    point: tuple[float, float],
    orientation: str,
    excluded_primitive_refs: set[str],
    excluded_topology_edge_refs: set[str],
    contours: Iterable[dict[str, Any]],
    edge_ids: set[str],
    vertex_ids: set[str],
    tolerance: float,
) -> list[dict[str, Any]]:
    target_axis = "vertical" if orientation == "horizontal" else "horizontal"
    candidates = []
    for contour in contours:
        if contour.get("source_modality") != "raster" or not contour.get("topology_component_ref"):
            continue
        for segment in contour.get("segments_display", []) or []:
            edge_ref = str(segment.get("id"))
            primitive_ref = str(segment.get("primitive_ref"))
            if (
                edge_ref not in edge_ids
                or edge_ref in excluded_topology_edge_refs
                or primitive_ref in excluded_primitive_refs
            ):
                continue
            start = tuple(map(float, segment["start_display"]))
            end = tuple(map(float, segment["end_display"]))
            distance = point_distance_to_segment(point, start, end)
            if distance > tolerance:
                continue
            axis = str(segment.get("axis", "oblique"))
            orientation_support = 1.0 if axis == target_axis else 0.72 if axis == "oblique" else 0.60
            score = orientation_support * max(0.0, 1.0 - distance / (tolerance + 1e-9))
            nearby_vertices = sorted(
                (
                    (math.dist(point, endpoint), str(vertex_ref))
                    for vertex_ref, endpoint in (
                        (segment.get("start_vertex_id"), start),
                        (segment.get("end_vertex_id"), end),
                    )
                    if str(vertex_ref) in vertex_ids and math.dist(point, endpoint) <= tolerance
                ),
                key=lambda item: (item[0], item[1]),
            )
            candidates.append(
                {
                    "contour_ref": str(contour["id"]),
                    "primitive_ref": primitive_ref,
                    "topology_edge_ref": edge_ref,
                    "topology_vertex_ref": nearby_vertices[0][1] if nearby_vertices else None,
                    "geometry_anchor_ref": nearby_vertices[0][1] if nearby_vertices else edge_ref,
                    "axis": axis,
                    "start_display": list(start),
                    "end_display": list(end),
                    "distance_points": round(distance, 4),
                    "score": round(score, 4),
                }
            )
    candidates.sort(
        key=lambda item: (
            -item["score"],
            item["distance_points"],
            item["topology_edge_ref"],
            item["contour_ref"],
        )
    )
    deduplicated = {}
    for candidate in candidates:
        deduplicated.setdefault(candidate["topology_edge_ref"], candidate)
    return list(deduplicated.values())[:8]


def resolve_raster_dimension_ownership(
    dimension_topology: Mapping[str, Any],
    dimension_label_ocr: Mapping[str, Any],
    vector_topology: Mapping[str, Any],
    views: Iterable[dict[str, Any]],
    contours: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Form metric raster dimension candidates without enabling solvers."""

    views = list(views)
    contours = list(contours)
    observations = {
        str(item["id"]): item
        for item in dimension_label_ocr.get("observations", []) or []
    }
    chain_results = {
        str(item["dimension_chain_ref"]): item
        for item in dimension_label_ocr.get("chain_results", []) or []
    }
    edge_ids = {str(item["id"]) for item in vector_topology.get("edges", []) or []}
    vertex_ids = {str(item["id"]) for item in vector_topology.get("vertices", []) or []}
    candidates = []
    for index, chain in enumerate(dimension_topology.get("chains", []) or [], start=1):
        chain_ref = str(chain["id"])
        reading = chain_results.get(chain_ref, {})
        observation_ref = reading.get("selected_observation_ref")
        observation = observations.get(str(observation_ref)) if observation_ref is not None else None
        unique_reading = reading.get("status") == "unique_numeric_observation" and observation is not None
        value = float(observation["numeric_value_candidate"]) if unique_reading else None
        view_ref = _raster_chain_view_ref(chain, views)
        candidates.append(
            {
                "id": f"raster.dimension_candidate.{index:04d}",
                "dimension_chain_ref": chain_ref,
                "label_observation_ref": observation_ref if unique_reading else None,
                "ocr_status": reading.get("status", "no_numeric_observation"),
                "ocr_observation_refs": list(reading.get("observation_refs", []) or []),
                "orientation": chain.get("orientation"),
                "dimension_points_display": chain.get("dimension_points_display", []),
                "measured_points_display": chain.get("measured_points_display", []),
                "numeric_value_candidate": value,
                "value_mm": None,
                "view_ref": view_ref,
                "scale_points_per_mm_observation": (
                    _raster_scale_observation(chain, value) if value is not None else None
                ),
                "metric_scale": {"status": "unknown", "scale_points_per_mm": None},
                "status": "unknown",
                "epistemic_state": "unknown",
                "solver_eligible": False,
                "reason": reading.get("reason"),
            }
        )

    scale_scopes = _raster_scale_consensus(candidates)
    for candidate in candidates:
        scale_scope = scale_scopes.get(candidate["view_ref"])
        if scale_scope is not None:
            candidate["metric_scale"] = {
                "status": scale_scope["status"],
                "scale_points_per_mm": scale_scope["scale_points_per_mm"],
                "scope_view_ref": scale_scope["view_ref"],
                "supporting_dimension_chain_refs": scale_scope["supporting_dimension_chain_refs"],
                "reason": scale_scope["reason"],
            }
        if candidate["view_ref"] is None and candidate["numeric_value_candidate"] is not None:
            candidate["reason"] = "dimension chain does not have a unique local raster view"
        elif candidate["metric_scale"]["status"] != "accepted" and candidate["numeric_value_candidate"] is not None:
            candidate["reason"] = candidate["metric_scale"].get("reason")

    attachments = []
    relations = []
    topology_tolerance = vector_topology.get("snap_tolerance_display_points")
    tolerance = max(1.25, min(3.0, 2.0 * float(topology_tolerance or 1.0)))
    chain_by_ref = {
        str(item["id"]): item
        for item in dimension_topology.get("chains", []) or []
    }
    for candidate in candidates:
        chain = chain_by_ref[candidate["dimension_chain_ref"]]
        excluded_primitives = {
            str(reference)
            for reference in (
                chain.get("baseline_ref"),
                *(chain.get("extension_line_refs", []) or []),
                *(chain.get("terminal_refs", []) or []),
            )
            if reference is not None
        }
        excluded_edges = {str(item) for item in chain.get("topology_edge_refs", []) or []}
        endpoint_rows = []
        selected = []
        for endpoint_index, raw_point in enumerate(chain.get("measured_points_display", []) or []):
            point = tuple(map(float, raw_point))
            rows = _raster_endpoint_candidates(
                point,
                str(chain.get("orientation")),
                excluded_primitives,
                excluded_edges,
                contours,
                edge_ids,
                vertex_ids,
                tolerance,
            )
            target = _unique_target(rows)
            selected.append(target)
            endpoint_rows.append(
                {
                    "endpoint_index": endpoint_index,
                    "point_display": list(point),
                    "state": "resolved" if target is not None else "candidate" if rows else "unknown",
                    "selected_contour_ref": None if target is None else target["contour_ref"],
                    "selected_primitive_ref": None if target is None else target["primitive_ref"],
                    "selected_topology_edge_ref": None if target is None else target["topology_edge_ref"],
                    "selected_topology_vertex_ref": None if target is None else target["topology_vertex_ref"],
                    "selected_geometry_anchor_ref": None if target is None else target["geometry_anchor_ref"],
                    "candidates": rows,
                }
            )
        common_contours = (
            {item["contour_ref"] for item in selected if item is not None}
            if len(selected) == 2 and all(selected)
            else set()
        )
        unique_owner = len(common_contours) == 1
        accepted = (
            candidate["metric_scale"]["status"] == "accepted"
            and candidate["numeric_value_candidate"] is not None
            and len(endpoint_rows) == 2
            and all(selected)
            and unique_owner
        )
        if accepted:
            candidate["value_mm"] = candidate["numeric_value_candidate"]
            candidate["status"] = "accepted_dimension_ownership_candidate"
            candidate["epistemic_state"] = "derived"
            candidate["reason"] = None
        elif candidate["reason"] is None:
            candidate["reason"] = (
                "measured endpoint has ambiguous raster contour edges"
                if any(len(row["candidates"]) > 1 for row in endpoint_rows)
                else "both measured endpoints do not resolve to one common raster contour owner"
            )
        attachment = {
            "id": f"raster.dimension_ownership.{len(attachments) + 1:04d}",
            "dimension_candidate_ref": candidate["id"],
            "dimension_chain_ref": candidate["dimension_chain_ref"],
            "status": "accepted" if accepted else "unknown",
            "epistemic_state": "derived" if accepted else "unknown",
            "owner_entity_refs": sorted(common_contours) if accepted else [],
            "measured_endpoints": endpoint_rows,
            "primitive_refs": sorted({item["primitive_ref"] for item in selected if item is not None}),
            "topology_edge_refs": sorted({item["topology_edge_ref"] for item in selected if item is not None}),
            "topology_vertex_refs": [
                item["topology_vertex_ref"] if item is not None else None for item in selected
            ],
            "geometry_anchor_refs": [
                item["geometry_anchor_ref"] if item is not None else None for item in selected
            ],
            "solver_eligible": False,
            "reason": candidate["reason"],
        }
        attachments.append(attachment)
        if accepted:
            relations.append(
                {
                    "id": f"raster.dimension_relation.{len(relations) + 1:04d}",
                    "type": "raster_dimension_candidate_of",
                    "from": candidate["id"],
                    "to": attachment["owner_entity_refs"][0],
                    "state": "derived",
                    "basis": "repeated local raster scale and two uniquely traced endpoints on one contour",
                    "evidence_refs": [
                        candidate["dimension_chain_ref"],
                        candidate["label_observation_ref"],
                        *attachment["topology_edge_refs"],
                    ],
                    "solver_eligible": False,
                }
            )

    accepted_count = sum(item["status"] == "accepted" for item in attachments)
    return {
        "schema_version": "0.1.0",
        "layer": "raster_dimension_ownership",
        "status": "resolved_candidates" if accepted_count else "unknown",
        "dimension_candidates": candidates,
        "attachments": attachments,
        "relations": relations,
        "scale_scopes": list(scale_scopes.values()),
        "summary": {
            "dimension_candidate_count": len(candidates),
            "accepted_candidate_count": accepted_count,
            "unknown_candidate_count": len(candidates) - accepted_count,
            "accepted_scale_scope_count": sum(item["status"] == "accepted" for item in scale_scopes.values()),
            "single_scale_scope_count": sum(item["observation_count"] == 1 for item in scale_scopes.values()),
            "conflicting_ocr_candidate_count": sum(
                item["ocr_status"] == "ambiguous_conflicting_numeric_observations" for item in candidates
            ),
        },
        "contract": {
            "complete_raster_chain_required": True,
            "unique_crop_reading_required": True,
            "repeated_local_scale_consensus_required": True,
            "both_endpoints_require_stable_raster_edge_or_vertex_ids": True,
            "unique_common_contour_owner_required": True,
            "ambiguity_remains_unknown": True,
            "solver_eligible": False,
            "schedule_values_used": False,
        },
    }
