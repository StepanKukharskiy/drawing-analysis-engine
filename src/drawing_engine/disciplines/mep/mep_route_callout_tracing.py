"""Replayable projected traces to callout anchors; never an M4 acceptance.

Consumes current recovery certificates, not raw colour or legacy membership.
Geometry is frozen before anchors are consulted.  Fragment lengths are retained
individually: this layer neither deduplicates them nor creates new quantities.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import hashlib
import json
import math


def _id(kind, value):
    body = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return kind + "." + hashlib.sha256(body.encode()).hexdigest()[:20]


def _point_gap(point, a, b):
    vector = [b[i] - a[i] for i in range(2)]
    squared = sum(x * x for x in vector)
    if not squared:
        return math.dist(point, a), 0.0
    t = sum((point[i] - a[i]) * vector[i] for i in range(2)) / squared
    return math.dist(point, [a[i] + max(0, min(1, t)) * vector[i]
                             for i in range(2)]), t


def build_route_callout_traces(recovery):
    """Return bounded straight scopes, every stop, and paths to every anchor.

    Near joins reuse the frozen per-style precision bound, capped at .12 pt.
    No precision certificate means exact-only.  Symbols/annotation strokes and
    all current corridor competitors are checked at each prospective seam.
    This is explicitly recovery-inventory coverage, not complete source search.
    """
    all_rows = sorted([*recovery.get("outlined_corridor_components", []),
                       *recovery.get("single_centreline_components", [])],
                      key=lambda row: row["id"])
    allowed = {"identified_mep_route", "supported_unidentified_mep_candidate"}
    rows = {row["id"]: row for row in all_rows if row["state"] in allowed}
    precision = recovery.get("precision_characterization", {})
    annotations = {row["id"]: row for row in recovery.get("route_annotations", [])}
    endpoints, nearby = {}, defaultdict(list)
    segments, segment_cells, endpoint_cells = [], defaultdict(list), defaultdict(list)
    cell_size = 32.0

    def cell(point):
        return tuple(math.floor(value / cell_size) for value in point)

    def cells(box):
        start, end = cell(box[:2]), cell(box[2:])
        for x in range(start[0], end[0] + 1):
            for y in range(start[1], end[1] + 1):
                yield x, y

    for row in all_rows:
        points = row.get("polyline_display", [])
        for a, b in zip(points, points[1:]):
            index = len(segments)
            segments.append((row["id"], a, b))
            for key in cells([min(a[0], b[0]) - .12, min(a[1], b[1]) - .12,
                              max(a[0], b[0]) + .12, max(a[1], b[1]) + .12]):
                segment_cells[key].append(index)
        if len(points) < 2:
            continue
        for end, (point, inward) in enumerate(((points[0], points[1]),
                                               (points[-1], points[-2]))):
            key = (row["id"], end)
            length = math.dist(point, inward)
            endpoints[key] = {
                "point": point,
                "direction": [(inward[i] - point[i]) / length for i in range(2)]
                if length else [0, 0],
            }
            endpoint_cells[cell(point)].append(key)

    bodies = recovery.get("fitting_or_equipment_symbol_candidates", [])
    body_cells = defaultdict(list)
    for body in bodies:
        box = body.get("bbox_display")
        if box:
            for key in cells([box[0] - .12, box[1] - .12,
                              box[2] + .12, box[3] + .12]):
                body_cells[key].append(body)
    annotation_segments = defaultdict(list)
    for leader in recovery.get("excluded_text_leader_candidates", []):
        points = leader.get("polyline_display", [])
        for a, b in zip(points, points[1:]):
            for key in cells([min(a[0], b[0]) - .12, min(a[1], b[1]) - .12,
                              max(a[0], b[0]) + .12, max(a[1], b[1]) + .12]):
                annotation_segments[key].append((leader["id"], a, b))

    def tolerance(row):
        certificate = precision.get(str(row.get("style_id")), {})
        value = certificate.get("maximum_join_tolerance_display_points", 0)
        return min(.12, max(0.0, float(value)))

    for key, endpoint in endpoints.items():
        point = endpoint["point"]
        for grid in cells([point[0] - .120001, point[1] - .120001,
                           point[0] + .120001, point[1] + .120001]):
            for other in endpoint_cells[grid]:
                if other[0] != key[0] and math.dist(point, endpoints[other]["point"]) <= .120001:
                    nearby[key].append(other)

    # An endpoint landing inside a longer row is a branch alternative, even
    # when it is far from the seam being considered.  Do not trace through that
    # unsplit row; it needs interval splitting and interface adjudication first.
    interior_branches = defaultdict(set)
    for key, endpoint in endpoints.items():
        point = endpoint["point"]
        for index in segment_cells[cell(point)]:
            ref, a, b = segments[index]
            if ref == key[0] or ref not in rows:
                continue
            gap, t = _point_gap(point, a, b)
            if gap <= .120001 and 1e-6 < t < 1 - 1e-6:
                interior_branches[ref].add(key[0])

    proposals, stops = {}, {}
    for key, endpoint in endpoints.items():
        if key[0] not in rows:
            continue
        row, point = rows[key[0]], endpoint["point"]
        competitors = sorted(set(nearby[key]))
        reasons, refs = [], set()
        if interior_branches[key[0]]:
            reasons.append("interior_branch_scope_requires_split")
            refs.update(interior_branches[key[0]])
        if len(competitors) > 1:
            reasons.append("ambiguous_endpoint_or_branch")
        for index in segment_cells[cell(point)]:
            ref, a, b = segments[index]
            if ref == key[0]:
                continue
            gap, t = _point_gap(point, a, b)
            if gap <= .120001 and 1e-6 < t < 1 - 1e-6:
                reasons.append("interior_contact_or_crossing_not_joined")
                refs.add(ref)
        for body in body_cells[cell(point)]:
            box = body["bbox_display"]
            if box[0] - .12 <= point[0] <= box[2] + .12 and box[1] - .12 <= point[1] <= box[3] + .12:
                reasons.append("fitting_or_equipment_boundary_unresolved")
                refs.add(body["id"])
        for ref, a, b in annotation_segments[cell(point)]:
            if _point_gap(point, a, b)[0] <= .120001:
                reasons.append("annotation_stroke_at_seam")
                refs.add(ref)
        if len(competitors) == 1:
            other = competitors[0]
            target = rows.get(other[0])
            if target is None:
                reasons.append("unclassified_geometry_boundary")
            else:
                if row.get("page_ref") != target.get("page_ref"):
                    reasons.append("view_or_page_boundary")
                if row.get("style_id") is None or row.get("style_id") != target.get("style_id"):
                    reasons.append("native_style_change")
                if row.get("channel") != target.get("channel"):
                    reasons.append("representation_change_requires_interface")
                for field in ("route_system", "route_size", "route_elevation"):
                    values = [item.get("attributes", {}).get(field, {}) for item in (row, target)]
                    if all(value.get("state") == "accepted" for value in values) and values[0].get("value") != values[1].get("value"):
                        reasons.append(field + "_change")
                widths = [float(value.get("corridor_width_display_points") or 0)
                          for value in (row, target)]
                if row.get("channel") == "parallel_outline_corridor" and (
                        min(widths) <= 0 or abs(widths[0] - widths[1]) > max(.02, .01 * min(widths))):
                    reasons.append("corridor_width_change")
                dot = sum(endpoint["direction"][i] * endpoints[other]["direction"][i]
                          for i in range(2))
                if dot > -.999:
                    reasons.append("turn_or_overlap_requires_fitting_review")
                residual = math.dist(point, endpoints[other]["point"])
                bound = min(tolerance(row), tolerance(target))
                if residual > max(1e-6, bound) + 1e-6:
                    reasons.append("gap_exceeds_frozen_precision_bound")
                if not reasons:
                    proposals[key] = other
        elif not competitors and not reasons:
            reasons.append("open_end_or_missing_continuation")
        stops[key] = {
            "component_ref": key[0], "end_index": key[1], "point_display": point,
            "reasons": sorted(set(reasons)),
            "competing_endpoint_refs": [[ref, end] for ref, end in competitors],
            "blocking_evidence_refs": sorted(refs),
        }

    joins, adjacency, joined = [], defaultdict(list), set()
    for key, other in sorted(proposals.items()):
        if proposals.get(other) != key:
            stops[key]["reasons"].append("transition_not_mutually_unique")
            continue
        if key > other:
            continue
        residual = math.dist(endpoints[key]["point"], endpoints[other]["point"])
        join = {
            "id": _id("mep_callout_trace_join", [key, other]),
            "endpoint_refs": [list(key), list(other)],
            "kind": "exact" if residual <= 1e-6 else "bounded_near",
            "residual_display_points": residual,
            "maximum_residual_display_points": min(tolerance(rows[key[0]]), tolerance(rows[other[0]])),
            "geometry_scope": "current_recovery_inventory_only",
            "physical_continuity_established": False,
        }
        joins.append(join)
        joined.update((key, other))
        adjacency[key[0]].append((other[0], join["id"]))
        adjacency[other[0]].append((key[0], join["id"]))

    groups, traces, visited = [], [], set()
    for start in sorted(rows):
        if start in visited:
            continue
        queue, members = deque([start]), set([start])
        while queue:
            current = queue.popleft()
            for other, _ in adjacency[current]:
                if other not in members:
                    members.add(other)
                    queue.append(other)
        visited.update(members)
        anchors = []
        for ref in sorted(members):
            row = rows[ref]
            attrs = row.get("attributes", {})
            system = attrs.get("route_system", {})
            if system.get("state") == "accepted" and system.get("value") and system.get("relation_refs"):
                anchors.append({
                    "component_ref": ref, "kind": "accepted_M4_system_anchor",
                    "system": system["value"], "evidence_refs": system["relation_refs"],
                })
            for annotation_ref in row.get("route_candidate_support", {}).get("annotation_refs", []):
                annotation = annotations.get(annotation_ref)
                if annotation and annotation.get("system_candidate"):
                    anchors.append({
                        "component_ref": ref, "kind": "annotation_target_candidate",
                        "system": annotation["system_candidate"],
                        "evidence_refs": [annotation_ref],
                    })
        systems = sorted({anchor["system"] for anchor in anchors})
        group_id = _id("mep_callout_trace_scope", sorted(members))
        group_stops = [value for key, value in sorted(stops.items())
                       if key[0] in members and key not in joined]
        groups.append({
            "id": group_id, "member_component_refs": sorted(members),
            "anchors": anchors, "candidate_systems": systems,
            "state": ("conflicting_callout_anchors" if len(systems) > 1 else
                      "callout_reached_binding_review_required" if systems else "no_callout_reached"),
            "terminals": group_stops,
            "closed_projected_cycle": not group_stops,
            "system_binding_accepted": False,
            "complete_native_obstruction_search": False,
            "installed_length": None, "purchase_length": None,
        })
        for ref in sorted(members):
            paths = []
            parent = {ref: None}
            queue = deque([ref])
            while queue:
                current = queue.popleft()
                for other, join_ref in sorted(adjacency[current]):
                    if other not in parent:
                        parent[other] = (current, join_ref)
                        queue.append(other)
            for anchor in anchors:
                current = anchor["component_ref"]
                route, link_refs = [current], []
                while parent[current] is not None:
                    current, link_ref = parent[current]
                    route.append(current)
                    link_refs.append(link_ref)
                paths.append({"anchor": anchor,
                              "component_path": route[::-1], "join_refs": link_refs[::-1]})
            traces.append({"component_ref": ref, "scope_ref": group_id,
                           "paths_to_anchors": paths, "candidate_systems": systems,
                           "system_binding_accepted": False})
    return {
        "schema_version": "mep_route_callout_tracing.v1",
        "geometry_input_hash": _id("recovery", {
            "rows": all_rows, "precision": precision, "annotations": list(annotations.values()),
            "bodies": bodies, "leaders": recovery.get("excluded_text_leader_candidates", []),
        }),
        "joins": joins, "scopes": groups, "component_traces": traces,
        "summary": {
            "input_route_components": len(rows), "accounted_route_components": len(traces),
            "straight_projected_join_count": len(joins),
            "joined_scope_count": sum(len(row["member_component_refs"]) > 1 for row in groups),
            "components_reaching_accepted_M4_anchor": sum(any(
                path["anchor"]["kind"] == "accepted_M4_system_anchor"
                for path in row["paths_to_anchors"]) for row in traces),
            "previously_unidentified_components_reaching_M4_anchor": sum(
                rows[row["component_ref"]]["state"] != "identified_mep_route" and any(
                    path["anchor"]["kind"] == "accepted_M4_system_anchor"
                    for path in row["paths_to_anchors"]) for row in traces),
            "components_reaching_any_callout_candidate": sum(bool(row["paths_to_anchors"]) for row in traces),
            "stop_reason_counts": dict(sorted(Counter(
                reason for group in groups for stop in group["terminals"]
                for reason in stop["reasons"]).items())),
        },
        "authority": {"system_binding_accepted": False, "quantity_eligible": False,
                      "physical_continuity_established": False,
                      "complete_source_coverage_established": False},
    }


def validate_route_callout_traces(recovery):
    """Replay both geometry and anchors; catches stale joins and forged names."""
    actual = recovery.get("route_callout_tracing")
    return ([] if actual == build_route_callout_traces(recovery) else
            ["route callout tracing differs from current geometry/anchor replay"])
