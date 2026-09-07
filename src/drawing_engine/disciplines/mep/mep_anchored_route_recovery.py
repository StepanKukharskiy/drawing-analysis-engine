"""Geometry-first, bounded recovery of projected MEP route components.

The recovery graph is deliberately built before M4 attributes are inspected.
Its nodes are accepted M3.5 outlined-route composites.  Two nodes may join
only when both outline sides have independent exact or mutually-unique near
join certificates and their page ownership, non-colour style, and corridor
width agree.  Branches, one-sided contacts, crossings, and ambiguous near
joins remain explicit terminal evidence.

The resulting records are projected geometry candidates.  Applying M4 after
the geometry payload is frozen may name a component; it cannot change its
membership or connectivity.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "0.1.0"
LAYER = "mep_anchored_projected_route_recovery"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _stable_id(kind: str, *parts: Any) -> str:
    return f"{kind}.{_sha(parts)[:20]}"


def _non_colour_style(style: Mapping[str, Any]) -> tuple[Any, ...]:
    width = style.get("width_display_points")
    return (
        None if width is None else round(float(width), 6),
        style.get("dash_pattern"),
        style.get("fill") is not None,
    )


def _style_compatible(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_width = left.get("width_display_points")
    right_width = right.get("width_display_points")
    if left_width is None or right_width is None:
        return False
    tolerance = max(.05, .1 * max(float(left_width), float(right_width)))
    return (
        abs(float(left_width) - float(right_width)) <= tolerance
        and left.get("dash_pattern") == right.get("dash_pattern")
        and (left.get("fill") is None) == (right.get("fill") is None)
    )


def _corridor_compatible(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_width = float(left["derived_geometry"]["corridor_width_display_points"])
    right_width = float(right["derived_geometry"]["corridor_width_display_points"])
    return abs(left_width - right_width) <= max(.25, .15 * max(left_width, right_width))


def _endpoint_rows(composite: Mapping[str, Any]) -> list[dict[str, Any]]:
    points = composite["derived_geometry"]["centreline_points_display"]
    return [
        {"composite_ref": composite["id"], "endpoint_index": 0,
         "point_display": list(points[0])},
        {"composite_ref": composite["id"], "endpoint_index": 1,
         "point_display": list(points[-1])},
    ]


def _join_pair_index(near_joins: Mapping[str, Any]) -> dict[tuple[str, str], list[dict]]:
    index: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in near_joins.get("exact_joins", []):
        refs = sorted(set(row.get("fragment_refs", [])))
        for position, left in enumerate(refs):
            for right in refs[position + 1:]:
                index[(left, right)].append({
                    "kind": "exact_join", "certificate_ref": row["id"],
                    "incident_fragment_count": len(refs),
                })
    for row in near_joins.get("uniquely_certified_near_joins", []):
        if row.get("state") != "accepted" or not row.get("mutual_unique"):
            continue
        left, right = sorted(row["fragment_refs"])
        index[(left, right)].append({
            "kind": "certified_near_join", "certificate_ref": row["id"],
            "incident_fragment_count": 2,
        })
    return index


def _ambiguous_pair_index(near_joins: Mapping[str, Any]) -> set[tuple[str, str]]:
    return {
        tuple(sorted(row["fragment_refs"]))
        for row in near_joins.get("ambiguous_near_join_candidates", [])
        if len(row.get("fragment_refs", [])) == 2
    }


def _perfect_outline_matching(
    left_members: Sequence[str], right_members: Sequence[str],
    join_pairs: Mapping[tuple[str, str], list[dict]],
) -> list[dict[str, Any]] | None:
    if len(left_members) != 2 or len(right_members) != 2:
        return None
    candidates = []
    for left in left_members:
        for right in right_members:
            rows = join_pairs.get(tuple(sorted((left, right))), [])
            for row in rows:
                candidates.append({"left_fragment_ref": left, "right_fragment_ref": right, **row})
    for first in candidates:
        for second in candidates:
            if first is second:
                continue
            if ({first["left_fragment_ref"], second["left_fragment_ref"]} == set(left_members)
                    and {first["right_fragment_ref"], second["right_fragment_ref"]} == set(right_members)):
                return sorted((first, second), key=lambda row: (
                    row["left_fragment_ref"], row["right_fragment_ref"], row["certificate_ref"]))
    return None


def _component_groups(nodes: Iterable[str], accepted: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    neighbors: dict[str, set[str]] = defaultdict(set)
    for edge in accepted:
        left, right = edge["composite_refs"]
        neighbors[left].add(right)
        neighbors[right].add(left)
    unseen = set(nodes)
    output = []
    while unseen:
        start = min(unseen)
        queue = deque([start])
        unseen.remove(start)
        group = []
        while queue:
            node = queue.popleft()
            group.append(node)
            for other in sorted(neighbors[node]):
                if other in unseen:
                    unseen.remove(other)
                    queue.append(other)
        output.append(sorted(group))
    return sorted(output, key=lambda rows: rows)


def _relation_value(relation: Mapping[str, Any]) -> Any:
    candidate = relation.get("candidate", {})
    if relation.get("relation_type") == "route_system":
        return candidate.get("kind")
    return candidate.get("value") or candidate.get("raw_text")


def build_anchored_route_recovery(
    *, page_ref: str, route_page: Mapping[str, Any],
    accepted_composites: Sequence[Mapping[str, Any]],
    near_joins: Mapping[str, Any], denominator_manifest: Mapping[str, Any],
    m4_relations: Sequence[Mapping[str, Any]] = (),
    interface_boundaries: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build geometry, freeze it, then independently project M4 attributes."""
    fragments = {row["id"]: row for row in route_page.get("fragments", [])}
    composites = {
        row["id"]: row for row in accepted_composites
        if row.get("page_ref") == page_ref and row.get("state") == "accepted"
    }
    join_pairs = _join_pair_index(near_joins)
    ambiguous_pairs = _ambiguous_pair_index(near_joins)
    endpoint_candidates: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    excluded = []
    rows = list(composites.values())
    for position, left in enumerate(rows):
        for right in rows[position + 1:]:
            if left.get("page_ref") != right.get("page_ref"):
                continue
            left_endpoints, right_endpoints = _endpoint_rows(left), _endpoint_rows(right)
            distance, left_end, right_end = min(
                [(math.dist(a["point_display"], b["point_display"]), a, b)
                 for a in left_endpoints for b in right_endpoints],
                key=lambda row: (row[0], row[1]["endpoint_index"], row[2]["endpoint_index"]))
            maximum = max(.5, .05 * max(
                float(left["derived_geometry"]["corridor_width_display_points"]),
                float(right["derived_geometry"]["corridor_width_display_points"])))
            if distance > maximum:
                continue
            left_members = list(left.get("member_fragment_refs", []))
            right_members = list(right.get("member_fragment_refs", []))
            pair_key = {
                tuple(sorted((a, b))) for a in left_members for b in right_members
            }
            reason = None
            if any(key in ambiguous_pairs for key in pair_key):
                reason = "ambiguous_near_join"
            elif not _corridor_compatible(left, right):
                reason = "corridor_width_change"
            else:
                left_styles = [fragments.get(ref, {}).get("style", {}) for ref in left_members]
                right_styles = [fragments.get(ref, {}).get("style", {}) for ref in right_members]
                if not left_styles or not right_styles or not all(
                    any(_style_compatible(a, b) for b in right_styles) for a in left_styles
                ):
                    reason = "incompatible_non_colour_style"
            matching = None if reason else _perfect_outline_matching(
                left_members, right_members, join_pairs)
            if reason is None and matching is None:
                reason = "two_sided_join_certificate_missing"
            elif reason is None and any(row["incident_fragment_count"] > 2 for row in matching):
                reason = "branch_vertex"
            elif reason is None and any(
                not _style_compatible(
                    fragments[row["left_fragment_ref"]].get("style", {}),
                    fragments[row["right_fragment_ref"]].get("style", {}),
                ) for row in matching
            ):
                reason = "incompatible_non_colour_style"
            record = {
                "id": _stable_id("mep_route_transition", page_ref, left["id"], left_end["endpoint_index"],
                                 right["id"], right_end["endpoint_index"]),
                "page_ref": page_ref,
                "composite_refs": sorted((left["id"], right["id"])),
                "endpoint_refs": [
                    f"{left['id']}:{left_end['endpoint_index']}",
                    f"{right['id']}:{right_end['endpoint_index']}",
                ],
                "centreline_endpoint_residual_display_points": round(distance, 6),
                "outline_join_certificates": matching or [],
                "state": "candidate" if reason is None else "excluded",
                "reason": reason,
            }
            if reason is None:
                endpoint_candidates[(left["id"], left_end["endpoint_index"])].append(record)
                endpoint_candidates[(right["id"], right_end["endpoint_index"])].append(record)
            else:
                excluded.append(record)

    accepted = []
    for candidates in endpoint_candidates.values():
        if len(candidates) != 1:
            for row in candidates:
                row["state"] = "excluded"
                row["reason"] = "branch_or_non_unique_transition"
                excluded.append(row)
    seen = set()
    for key, candidates in endpoint_candidates.items():
        if len(candidates) != 1:
            continue
        row = candidates[0]
        other_ref = row["endpoint_refs"][0] if row["endpoint_refs"][1] == f"{key[0]}:{key[1]}" else row["endpoint_refs"][1]
        other_composite, other_endpoint = other_ref.rsplit(":", 1)
        if len(endpoint_candidates[(other_composite, int(other_endpoint))]) != 1:
            continue
        if row["id"] not in seen:
            row = dict(row)
            row["state"] = "accepted"
            row["reason"] = None
            accepted.append(row)
            seen.add(row["id"])

    boundary_points = []
    for boundary in interface_boundaries:
        point = boundary.get("point_display")
        if point and len(point) == 2:
            boundary_points.append((list(point), boundary))
    accepted_endpoint_refs = {ref for row in accepted for ref in row["endpoint_refs"]}
    excluded_by_endpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in excluded:
        for ref in row["endpoint_refs"]:
            excluded_by_endpoint[ref].append(row)

    geometry_components = []
    for group in _component_groups(composites, accepted):
        member_set = set(group)
        centreline_source_refs = sorted({
            ref for composite_ref in group
            for ref in composites[composite_ref].get("member_source_primitive_refs", [])
        })
        closure_source_refs = sorted({
            ref for composite_ref in group
            for ref in composites[composite_ref].get(
                "denominator_eligible_supporting_closure_source_refs", [])
        })
        source_refs = sorted(set(centreline_source_refs) | set(closure_source_refs))
        closure_fragment_refs = sorted({
            ref for composite_ref in group
            for ref in composites[composite_ref].get("supporting_closure_fragment_refs", [])
        })
        transition_rows = [row for row in accepted
                           if set(row["composite_refs"]).issubset(member_set)]
        terminals = []
        for composite_ref in group:
            composite = composites[composite_ref]
            for endpoint in _endpoint_rows(composite):
                endpoint_ref = f"{composite_ref}:{endpoint['endpoint_index']}"
                if endpoint_ref in accepted_endpoint_refs:
                    continue
                reasons = sorted({row["reason"] for row in excluded_by_endpoint[endpoint_ref]
                                  if row.get("reason")})
                nearby = [boundary for point, boundary in boundary_points
                          if math.dist(point, endpoint["point_display"]) <= float(
                              composite["derived_geometry"]["corridor_width_display_points"]) + 2.0]
                if nearby:
                    reasons.append("fitting_or_equipment_boundary")
                if not reasons:
                    reasons = ["no_compatible_certified_transition"]
                terminals.append({
                    "endpoint_ref": endpoint_ref,
                    "point_display": endpoint["point_display"],
                    "terminal_reasons": sorted(set(reasons)),
                    "interface_boundary_refs": sorted(row["id"] for row in nearby),
                })
        geometry_components.append({
            "id": _stable_id("mep_bounded_projected_route_component", page_ref, group),
            "record_type": "mep_bounded_projected_route_component_candidate",
            "state": "bounded_projected_geometry",
            "page_ref": page_ref,
            "composite_refs": group,
            "source_segment_refs": source_refs,
            "centreline_source_segment_refs": centreline_source_refs,
            "supporting_closure_fragment_refs": closure_fragment_refs,
            "supporting_closure_source_segment_refs": closure_source_refs,
            "transition_refs": sorted(row["id"] for row in transition_rows),
            "exact_join_certificate_refs": sorted({
                cert["certificate_ref"] for row in transition_rows
                for cert in row["outline_join_certificates"] if cert["kind"] == "exact_join"}),
            "near_join_certificate_refs": sorted({
                cert["certificate_ref"] for row in transition_rows
                for cert in row["outline_join_certificates"] if cert["kind"] == "certified_near_join"}),
            "excluded_competing_transition_refs": sorted({
                row["id"] for composite_ref in group for endpoint_index in (0, 1)
                for row in excluded_by_endpoint[f"{composite_ref}:{endpoint_index}"]}),
            "terminals": terminals,
            "recovered_denominator_rows": len(source_refs),
            "completeness": {
                "source_query_complete": True,
                "maximal_unbranched_component": True,
                "both_outline_sides_required_at_every_join": True,
                "physical_continuity_established": False,
            },
            "projected_path_display_points": round(sum(
                float(composites[ref]["derived_geometry"]["projected_path_display_points"])
                for ref in group), 6),
            "quantity_eligible": False,
        })

    geometry_hash = _sha(geometry_components)
    component_by_composite = {
        composite_ref: component["id"]
        for component in geometry_components for composite_ref in component["composite_refs"]
    }
    relations_by_component: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for relation in m4_relations:
        if relation.get("page_ref") != page_ref or relation.get("state") != "accepted":
            continue
        if relation.get("relation_type") not in {"route_system", "route_size", "route_elevation"}:
            continue
        targets = {component_by_composite[ref] for ref in relation.get("target_refs", [])
                   if ref in component_by_composite}
        if len(targets) == 1:
            relations_by_component[next(iter(targets))].append(relation)

    bindings = []
    for component in geometry_components:
        by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for relation in relations_by_component.get(component["id"], []):
            by_type[relation["relation_type"]].append(relation)
        attributes = {}
        for relation_type in ("route_system", "route_size", "route_elevation"):
            relations = by_type.get(relation_type, [])
            values = {_canonical(_relation_value(row)): _relation_value(row) for row in relations}
            if len(values) == 1:
                value = next(iter(values.values()))
                attributes[relation_type] = {
                    "state": "accepted", "value": value,
                    "relation_refs": sorted(row["id"] for row in relations),
                    "raw_text": sorted({str(row.get("candidate", {}).get("raw_text"))
                                        for row in relations if row.get("candidate", {}).get("raw_text")}),
                }
            else:
                attributes[relation_type] = {
                    "state": "unknown", "value": None,
                    "relation_refs": sorted(row["id"] for row in relations),
                    "reason": "no_unique_accepted_M4_value" if relations else "no_accepted_M4_relation",
                }
        bindings.append({
            "id": _stable_id("mep_post_geometry_attribute_binding", component["id"], attributes),
            "record_type": "mep_post_geometry_component_attribute_binding",
            "state": "identified" if attributes["route_system"]["state"] == "accepted" else "unknown",
            "page_ref": page_ref, "component_ref": component["id"],
            "attributes": attributes,
            "geometry_membership_sha256": geometry_hash,
            "geometry_frozen_before_M4": True,
            "colour_used_to_select_geometry": False,
            "quantity_eligible": False,
        })

    disposition_counts = denominator_manifest["source_disposition_pack"]["disposition_counts"]
    recovered_source_refs = {
        ref for row in geometry_components for ref in row["source_segment_refs"]
    }
    excluded_reason_counts = Counter(row["reason"] for row in excluded)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "layer": LAYER,
        "page_ref": page_ref,
        "geometry_input": {
            "accepted_outlined_composite_count": len(composites),
            "route_evidence_segment_count": disposition_counts.get("route_evidence", 0),
            "denominator_record_count": denominator_manifest["native_descriptor_pack"]["record_count"],
            "near_join_payload_sha256": _sha(near_joins),
            "colour_used_to_build_graph": False,
        },
        "accepted_transitions": sorted(accepted, key=lambda row: row["id"]),
        "excluded_competing_transitions": sorted(excluded, key=lambda row: row["id"]),
        "geometry_components": geometry_components,
        "geometry_components_sha256": geometry_hash,
        "post_geometry_M4_bindings": bindings,
        "coverage": {
            "original_route_evidence_segment_count": disposition_counts.get("route_evidence", 0),
            "new_bounded_route_component_candidate_count": len(geometry_components),
            "recovered_denominator_route_row_count": len(recovered_source_refs),
            "route_evidence_segment_not_recovered_count": max(
                0, disposition_counts.get("route_evidence", 0) - len(recovered_source_refs)),
            "M4_identified_component_count": sum(row["state"] == "identified" for row in bindings),
            "unresolved_drawing_view_geometry_remaining": disposition_counts.get(
                "unresolved_drawing_view_geometry", 0),
            "ambiguous_transition_count": excluded_reason_counts["ambiguous_near_join"]
                + excluded_reason_counts["branch_or_non_unique_transition"],
            "fitting_equipment_boundary_count": sum(
                "fitting_or_equipment_boundary" in terminal["terminal_reasons"]
                for row in geometry_components for terminal in row["terminals"]),
        },
        "negative_gate_measurement": {
            "state": "deferred",
            "owner": "mep_projected_component_evidence_classification",
            "reason": "geometry recovery cannot classify architectural content or assign literal zero counts",
        },
        "authority": {
            "projected_geometry_candidate_established": True,
            "system_identity_requires_post_geometry_M4": True,
            "colour_correlation_certifies_system": False,
            "physical_continuity_established": False,
            "installed_length_established": False,
            "quantity_eligible": False,
        },
    }
    return payload


def validate_anchored_route_recovery(payload: Mapping[str, Any]) -> list[str]:
    errors = []
    if payload.get("layer") != LAYER:
        errors.append("unexpected layer")
    if payload.get("geometry_input", {}).get("colour_used_to_build_graph") is not False:
        errors.append("colour influenced the recovery graph")
    if payload.get("geometry_components_sha256") != _sha(payload.get("geometry_components", [])):
        errors.append("geometry component hash mismatch")
    components = {row["id"]: row for row in payload.get("geometry_components", [])}
    memberships = [ref for row in components.values() for ref in row.get("composite_refs", [])]
    if len(memberships) != len(set(memberships)):
        errors.append("outlined composite belongs to multiple recovered components")
    for row in payload.get("accepted_transitions", []):
        certificates = row.get("outline_join_certificates", [])
        if len(certificates) != 2:
            errors.append(f"transition lacks two-sided evidence: {row.get('id')}")
        if row.get("reason") is not None or row.get("state") != "accepted":
            errors.append(f"accepted transition has invalid state: {row.get('id')}")
    for row in payload.get("post_geometry_M4_bindings", []):
        if row.get("component_ref") not in components:
            errors.append(f"M4 binding references unknown component: {row.get('id')}")
        if row.get("geometry_frozen_before_M4") is not True:
            errors.append(f"M4 binding did not preserve frozen geometry: {row.get('id')}")
    authority = payload.get("authority", {})
    for key in ("physical_continuity_established", "installed_length_established", "quantity_eligible"):
        if authority.get(key) is not False:
            errors.append(f"forbidden authority enabled: {key}")
    return errors
