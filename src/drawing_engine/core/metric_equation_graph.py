"""Bind arithmetic dimension equations to native chains and view scopes.

Expressions such as ``100 x 52 = 5200`` are direct arithmetic evidence, but
the printed equality alone does not identify an object.  This module requires
the expression to sit on a complete native dimension chain, resolves a local
scale consensus, binds the chain to one semantic view, and only then proposes
cross-view metric scopes.  It never reads schedules or drawing titles.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import fitz

from src.drawing_engine.core.dimension_attachment import (
    DimensionAttachment,
    _axis,
    _intersection_points,
    _segments,
    _terminal_refs,
)
from src.drawing_engine.disciplines.rebar.rebar_program import extract_spacing_constraints


def _view_for_box(box: fitz.Rect, views: Iterable[dict[str, Any]], padding: float = 2.0) -> str | None:
    center = fitz.Point((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)
    candidates = []
    for view in views:
        view_box = fitz.Rect(view["bbox_display"]) + (-padding, -padding, padding, padding)
        if center in view_box:
            candidates.append((view_box.get_area(), str(view["id"])))
    return min(candidates)[1] if candidates else None


def _line_orientation(page: fitz.Page, box: fitz.Rect) -> str:
    center = fitz.Point((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)
    rows = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            line_box = fitz.Rect(line.get("bbox", (0, 0, 0, 0)))
            if center not in line_box + (-0.5, -0.5, 0.5, 0.5):
                continue
            direction = tuple(float(value) for value in line.get("dir", (1.0, 0.0)))
            rows.append((line_box.get_area(), "vertical" if abs(direction[1]) > abs(direction[0]) else "horizontal"))
    return min(rows)[1] if rows else ("vertical" if box.height > 1.8 * box.width else "horizontal")


def _chain_candidates(
    page: fitz.Page,
    constraint: dict[str, Any],
    views: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    segments = _segments(page)
    box = fitz.Rect(constraint["bbox_display"])
    orientation = _line_orientation(page, box)
    center = ((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)
    result = []
    for baseline in segments:
        if _axis(baseline) != orientation:
            continue
        if orientation == "horizontal":
            cross = (baseline.start[1] + baseline.end[1]) / 2
            if not (-6.0 <= cross - box.y1 <= 20.0):
                continue
            if not min(baseline.start[0], baseline.end[0]) - 5.0 <= center[0] <= max(baseline.start[0], baseline.end[0]) + 5.0:
                continue
        else:
            cross = (baseline.start[0] + baseline.end[0]) / 2
            if not (-6.0 <= cross - box.x1 <= 20.0):
                continue
            if not min(baseline.start[1], baseline.end[1]) - 5.0 <= center[1] <= max(baseline.start[1], baseline.end[1]) + 5.0:
                continue
        intersections = _intersection_points(baseline, orientation, segments)
        if len(intersections) < 2:
            continue
        first, last = intersections[0], intersections[-1]
        separation = abs(last[0] - first[0])
        if separation < max(10.0, baseline.length * 0.45):
            continue
        first_ticks = _terminal_refs(first[2], segments)
        last_ticks = _terminal_refs(last[2], segments)
        terminal_refs = sorted(set((*first_ticks, *last_ticks)))
        result.append(
            {
                "orientation": orientation,
                "baseline_ref": baseline.primitive_ref,
                "baseline_display": [list(baseline.start), list(baseline.end)],
                "extension_line_refs": [first[1].primitive_ref, last[1].primitive_ref],
                "extension_lines_display": [
                    [list(first[1].start), list(first[1].end)],
                    [list(last[1].start), list(last[1].end)],
                ],
                "terminal_refs": terminal_refs,
                "dimension_points_display": [list(first[2]), list(last[2])],
                "measured_points_display": [list(first[3]), list(last[3])],
                "separation_points": separation,
                "scale_points_per_mm": separation / float(constraint["extent_mm"]),
                "complete_terminal_pair": bool(first_ticks and last_ticks),
                "baseline_coverage": separation / max(baseline.length, 1e-9),
                "view_id": _view_for_box(box, views),
                "primitive_refs": [
                    baseline.primitive_ref,
                    first[1].primitive_ref,
                    last[1].primitive_ref,
                    *terminal_refs,
                ],
            }
        )
    return result


def _scale_support(
    candidate: dict[str, Any],
    all_candidates: list[tuple[str, dict[str, Any]]],
    dimensions: Iterable[DimensionAttachment],
) -> tuple[int, list[str]]:
    scale = float(candidate["scale_points_per_mm"])
    refs = {
        constraint_id
        for constraint_id, other in all_candidates
        if other.get("complete_terminal_pair")
        and abs(float(other["scale_points_per_mm"]) / scale - 1.0) <= 0.02
    }
    refs.update(
        item.attachment_id
        for item in dimensions
        if item.status == "accepted" and abs(float(item.scale_points_per_mm) / scale - 1.0) <= 0.02
    )
    return len(refs), sorted(refs)


def _resolve_chains(
    page: fitz.Page,
    constraints: list[dict[str, Any]],
    views: list[dict[str, Any]],
    dimensions: Iterable[DimensionAttachment],
) -> None:
    by_constraint = {item["id"]: _chain_candidates(page, item, views) for item in constraints}
    flattened = [(constraint_id, candidate) for constraint_id, rows in by_constraint.items() for candidate in rows]
    dimensions = tuple(dimensions)
    for constraint in constraints:
        ranked = []
        for candidate in by_constraint[constraint["id"]]:
            support, support_refs = _scale_support(candidate, flattened, dimensions)
            score = (
                (0.30 if constraint["arithmetic"]["status"] == "pass" else 0.0)
                + (0.30 if candidate["complete_terminal_pair"] else 0.0)
                + 0.15
                + 0.15
                + (0.10 if support >= 3 else 0.0)
            )
            ranked.append(
                (
                    score,
                    support,
                    abs(1.0 - float(candidate["baseline_coverage"])),
                    candidate["baseline_ref"],
                    {**candidate, "scale_support": support, "scale_support_refs": support_refs, "score": round(score, 3)},
                )
            )
        ranked.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
        best = ranked[0][4] if ranked else None
        runner = ranked[1][4] if len(ranked) > 1 else None
        unique = best is not None and (
            runner is None
            or best["scale_support"] > runner["scale_support"]
            or abs(float(best["scale_points_per_mm"]) / float(runner["scale_points_per_mm"]) - 1.0) <= 0.005
            and best["baseline_coverage"] > runner["baseline_coverage"] + 0.04
        )
        accepted = bool(
            best
            and unique
            and best["complete_terminal_pair"]
            and best["scale_support"] >= 3
            and constraint["arithmetic"]["status"] == "pass"
            and best["view_id"] is not None
        )
        constraint["status"] = "accepted" if accepted else "candidate" if best else "unresolved"
        constraint["epistemic_state"] = "derived" if accepted else "unknown"
        constraint["chain_attachment"] = best
        constraint["chain_candidates"] = [item[4] for item in ranked[:6]]
        constraint["view_id"] = None if best is None else best["view_id"]
        constraint["reason"] = None if accepted else (
            "no complete native dimension chain surrounds the arithmetic expression"
            if best is None
            else "multiple native chains or local scales remain compatible"
            if not unique
            else "native chain lacks repeated local scale support"
        )


def _extent_clusters(constraints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accepted = [item for item in constraints if item.get("status") == "accepted" and item.get("view_id")]
    clusters: list[list[dict[str, Any]]] = []
    for constraint in sorted(accepted, key=lambda item: (float(item["extent_mm"]), item["id"])):
        cluster = next(
            (
                rows
                for rows in clusters
                if abs(float(rows[0]["extent_mm"]) - float(constraint["extent_mm"]))
                <= max(2.0, 0.005 * float(constraint["extent_mm"]))
            ),
            None,
        )
        if cluster is None:
            cluster = []
            clusters.append(cluster)
        cluster.append(constraint)
    return [
        {
            "id": f"metric_extent_cluster.{index:03d}",
            "extent_mm": round(sum(float(item["extent_mm"]) for item in rows) / len(rows), 3),
            "constraint_ids": [item["id"] for item in rows],
            "view_ids": sorted({str(item["view_id"]) for item in rows}),
            "equation_signatures": sorted(
                {
                    (int(item["spacing_mm"]), int(item["interval_count"]), int(item["extent_mm"]))
                    for item in rows
                }
            ),
            "orientations_by_view": {
                view_id: sorted(
                    {
                        str(item["chain_attachment"]["orientation"])
                        for item in rows
                        if str(item["view_id"]) == view_id
                    }
                )
                for view_id in sorted({str(item["view_id"]) for item in rows})
            },
            "state": "derived",
        }
        for index, rows in enumerate(clusters, start=1)
    ]


def _shared_relations(clusters: list[dict[str, Any]], constraints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {item["id"]: item for item in constraints}
    relations = []
    for cluster in clusters:
        if len(cluster["view_ids"]) < 2:
            continue
        exact_by_view: dict[str, set[tuple[int, int, int]]] = defaultdict(set)
        for constraint_id in cluster["constraint_ids"]:
            item = by_id[constraint_id]
            exact_by_view[str(item["view_id"])].add(
                (int(item["spacing_mm"]), int(item["interval_count"]), int(item["extent_mm"]))
            )
        view_ids = cluster["view_ids"]
        for left_index, left in enumerate(view_ids):
            for right in view_ids[left_index + 1 :]:
                exact = sorted(exact_by_view[left].intersection(exact_by_view[right]))
                independently_factorised = len(cluster["equation_signatures"]) >= 2 and len(view_ids) >= 3
                accepted = bool(exact or independently_factorised)
                relations.append(
                    {
                        "id": f"shared_metric_extent.{len(relations) + 1:03d}",
                        "type": "shared_metric_extent",
                        "left_view_id": left,
                        "right_view_id": right,
                        "extent_cluster_id": cluster["id"],
                        "extent_mm": cluster["extent_mm"],
                        "status": "accepted" if accepted else "candidate",
                        "state": "derived" if accepted else "unknown",
                        "basis": (
                            "same complete arithmetic dimension chain in two views"
                            if exact
                            else "same extent independently factorised across at least three views"
                            if independently_factorised
                            else "equal metric extent without enough independent topology"
                        ),
                        "exact_equation_signatures": exact,
                        "evidence_refs": [cluster["id"], *cluster["constraint_ids"]],
                    }
                )
    return relations


def _scope_components(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accepted = [item for item in relations if item["status"] == "accepted"]
    adjacency: dict[str, set[str]] = defaultdict(set)
    by_view: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for relation in accepted:
        left, right = relation["left_view_id"], relation["right_view_id"]
        adjacency[left].add(right)
        adjacency[right].add(left)
        by_view[left].append(relation)
        by_view[right].append(relation)
    components = []
    visited: set[str] = set()
    for seed in sorted(adjacency):
        if seed in visited:
            continue
        stack = [seed]
        view_ids = set()
        relation_by_id = {}
        while stack:
            view_id = stack.pop()
            if view_id in visited:
                continue
            visited.add(view_id)
            view_ids.add(view_id)
            for relation in by_view[view_id]:
                relation_by_id[relation["id"]] = relation
            stack.extend(sorted(adjacency[view_id] - visited, reverse=True))
        components.append({"view_ids": sorted(view_ids), "relations": [relation_by_id[key] for key in sorted(relation_by_id)]})
    return components


def build_metric_equation_graph(
    page: fitz.Page,
    views: Iterable[dict[str, Any]],
    dimensions: Iterable[DimensionAttachment],
) -> dict[str, Any]:
    """Build fail-closed metric scopes from owned arithmetic chains."""

    views = list(views)
    dimensions = tuple(dimensions)
    constraints = [
        item
        for item in extract_spacing_constraints(page)
        if item.get("type") == "distribution_segment" and item.get("extent_mm") is not None
    ]
    _resolve_chains(page, constraints, views, dimensions)
    clusters = _extent_clusters(constraints)
    relations = _shared_relations(clusters, constraints)
    scopes = []
    candidates = []
    for component in _scope_components(relations):
        component_relations = component["relations"]
        cluster_ids = sorted({item["extent_cluster_id"] for item in component_relations})
        accepted = len(component["view_ids"]) >= 3 and len(cluster_ids) >= 2 and len(component_relations) >= 2
        record = {
            "id": f"metric_equation_scope.{len(scopes) + len(candidates) + 1:03d}",
            "status": "accepted" if accepted else "candidate",
            "state": "derived" if accepted else "unknown",
            "view_ids": component["view_ids"],
            "extent_cluster_ids": cluster_ids,
            "relation_refs": [item["id"] for item in component_relations],
            "physical_object_identity_state": "unresolved",
            "quantity_aggregation_eligible": False,
            "basis": "connected views share at least two independently closed arithmetic metric extents",
            "reason": None if accepted else "metric component lacks three views or two independent shared extents",
            "evidence_refs": sorted(
                {
                    ref
                    for relation in component_relations
                    for ref in (relation["id"], *relation.get("evidence_refs", []))
                }
            ),
        }
        (scopes if accepted else candidates).append(record)
    return {
        "schema_version": "0.1.0",
        "layer": "metric_equation_graph",
        "status": "resolved_subset" if scopes else "observations_only" if constraints else "unresolved",
        "constraints": constraints,
        "extent_clusters": clusters,
        "relations": relations,
        "scopes": scopes,
        "scope_candidates": candidates,
        "summary": {
            "equation_count": len(constraints),
            "accepted_chain_count": sum(item.get("status") == "accepted" for item in constraints),
            "arithmetic_conflict_count": sum(item["arithmetic"]["status"] == "conflict" for item in constraints),
            "shared_extent_relation_count": sum(item["status"] == "accepted" for item in relations),
            "accepted_scope_count": len(scopes),
            "candidate_scope_count": len(candidates),
        },
        "contract": {
            "schedule_values_used": False,
            "filename_dispatch_used": False,
            "formula_without_native_chain_creates_scope": False,
            "equal_extent_alone_creates_scope": False,
            "physical_object_identity_claimed": False,
        },
    }
