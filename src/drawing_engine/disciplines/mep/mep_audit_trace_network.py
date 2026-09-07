"""Build page-local MEP audit traces without granting physical-run authority.

The trace layer is deliberately downstream of the region-owned takeoff.  It
renders every retained route observation and uses a one-to-one native colour
correlation only as a review aid when that colour is independently anchored by
an accepted system relation.  Colour-correlated rows remain review candidates;
they never become semantic quantity, physical continuity, or installed length.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from typing import Any, Mapping


SCHEMA_VERSION = "0.2.0"
SYSTEM_ABBREVIATIONS = {
    "heating_hot_water_supply": "HHWS",
    "heating_hot_water_return": "HHWR",
    "chilled_water_supply": "CHWS",
    "chilled_water_return": "CHWR",
}


def _stable_id(kind: str, *parts: Any) -> str:
    body = json.dumps([kind, *parts], ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode()
    return kind + "." + hashlib.sha256(body).hexdigest()[:20]


def _stroke(fragment: Mapping[str, Any]) -> tuple[float, ...] | None:
    value = fragment.get("style", {}).get("stroke")
    return tuple(round(float(item), 6) for item in value) if value else None


def _accepted_text(field: Mapping[str, Any] | None) -> str | None:
    if not field or field.get("state") != "accepted":
        return None
    values = field.get("observed_values", [])
    if len(values) != 1:
        return None
    value = values[0]
    raw = value.get("raw_text")
    if raw:
        return str(raw)
    numeric = value.get("value")
    unit = value.get("unit")
    return f"{numeric:g} {unit}" if isinstance(numeric, (int, float)) and unit else None


def _rgb_hex(colour: tuple[float, ...] | None) -> str | None:
    if not colour or len(colour) < 3:
        return None
    return "#" + "".join(f"{max(0, min(255, round(channel * 255))):02X}"
                           for channel in colour[:3])


def _fragments(route_graph: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {row["id"]: row for page in route_graph.get("pages", [])
            for row in page.get("fragments", [])}


def _topology_summary(assignments: list[dict], fragment_by_id: Mapping[str, Mapping]) -> dict:
    members = {row["occurrence_ref"]: row for row in assignments}
    parent = {ref: ref for ref in members}

    def find(ref):
        while parent[ref] != ref:
            parent[ref] = parent[parent[ref]]
            ref = parent[ref]
        return ref

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    by_vertex = defaultdict(set)
    for row in assignments:
        for fragment_ref in row["source_fragment_refs"]:
            fragment = fragment_by_id.get(fragment_ref)
            if fragment:
                for vertex_ref in fragment.get("endpoint_vertex_refs", []):
                    by_vertex[vertex_ref].add(row["occurrence_ref"])
    edge_count = 0
    for refs in by_vertex.values():
        refs = sorted(refs)
        if len(refs) > 1:
            for ref in refs[1:]:
                union(refs[0], ref)
                edge_count += 1
    component_count = len({find(ref) for ref in members}) if members else 0
    return {
        "exact_shared_endpoint_edge_count": edge_count,
        "exact_endpoint_component_count": component_count,
        "state": "one_exact_endpoint_component" if component_count == 1
                 else "multiple_exact_endpoint_components",
        "physical_continuity_established": False,
    }


def build_mep_audit_trace_network(*, observed: Mapping[str, Any], semantic: Mapping[str, Any],
                                  route_graph: Mapping[str, Any],
                                  source_pdf_sha256: str) -> dict[str, Any]:
    fragment_by_id = _fragments(route_graph)
    route_by_id = {row["id"]: row for row in observed["route_segment_ledger"]}
    page_number = {row["page_ref"]: row["page_number"] for row in observed["sheet_coverage"]}
    direct_system = {}
    semantic_length = Counter()
    for group in semantic["semantic_route_groups"]:
        system = group["semantic_signature"]["system"]["value"]["kind"]
        for ref in group["source_route_ledger_refs"]:
            direct_system[ref] = system
        owner = min(group["page_refs"], key=lambda ref: page_number[ref])
        semantic_length[(owner, system)] += (
            group["length_channels"].get("canonicalized_projected_length_m") or 0.0)

    colours_by_system = defaultdict(set)
    anchor_fragment_count = Counter()
    for route_ref, system in direct_system.items():
        for occurrence in route_by_id[route_ref].get("occurrences", []):
            colours = {_stroke(fragment_by_id[ref]) for ref in occurrence.get("source_fragment_refs", [])
                       if ref in fragment_by_id}
            colours.discard(None)
            if len(colours) == 1:
                colours_by_system[system].update(colours)
                anchor_fragment_count[system] += len(occurrence.get("source_fragment_refs", []))
    candidate_mapping = {next(iter(colours)): system for system, colours in colours_by_system.items()
                         if len(colours) == 1}
    colour_owners = Counter(next(iter(colours)) for colours in colours_by_system.values()
                            if len(colours) == 1)
    colour_mapping = {colour: system for colour, system in candidate_mapping.items()
                      if colour_owners[colour] == 1}
    colour_certificates = [{
        "id": _stable_id("mep_audit_colour_mapping", colour, system),
        "state": "accepted_one_to_one_review_correlation", "native_stroke_rgb": list(colour),
        "system": system, "accepted_anchor_fragment_count": anchor_fragment_count[system],
        "colour_alone_establishes_route_identity": False,
        "quantity_eligible": False,
    } for colour, system in sorted(colour_mapping.items())]

    assignments = []
    for route in observed["route_segment_ledger"]:
        overlay = route.get("semantic_overlay", {})
        nominal_size = _accepted_text(overlay.get("size"))
        elevation = _accepted_text(overlay.get("elevation"))
        for occurrence in route.get("occurrences", []):
            colours = {_stroke(fragment_by_id[ref]) for ref in occurrence.get("source_fragment_refs", [])
                       if ref in fragment_by_id}
            colours.discard(None)
            colour = next(iter(colours)) if len(colours) == 1 else None
            if route["id"] in direct_system:
                system = direct_system[route["id"]]
                state = "certified_system"
                basis = "accepted_M4_system_relation_and_region_owned_semantic_route"
            elif colour in colour_mapping:
                system = colour_mapping[colour]
                state = "colour_correlated_review"
                basis = "one_to_one_native_colour_correlation_to_accepted_system_anchor"
            else:
                system = None
                state = "unknown_engineer_review"
                basis = "no_accepted_system_relation_or_unique_anchored_colour_correlation"
            system_code = SYSTEM_ABBREVIATIONS.get(system, "UNKNOWN")
            system_label = (system_code if state == "certified_system" else
                            f"{system_code}? COLOR-REVIEW" if state == "colour_correlated_review"
                            else "SYSTEM UNKNOWN")
            length = occurrence.get("projected_2d_length_m")
            assignments.append({
                "id": _stable_id("mep_audit_route_assignment", occurrence["occurrence_ref"]),
                "route_ledger_ref": route["id"], "occurrence_ref": occurrence["occurrence_ref"],
                "page_ref": occurrence["page_ref"], "system": system,
                "display_state": state, "basis": basis,
                "native_stroke_rgb": list(colour) if colour else None,
                "native_stroke_hex": _rgb_hex(colour),
                "points_display": occurrence.get("points_display", []),
                "projected_2d_length_m": length,
                "system_display": system_label,
                "nominal_diameter_display": nominal_size or "UNKNOWN",
                "elevation_display": elevation or "UNKNOWN",
                "segment_kind_display": ("PIPE SEGMENT" if state == "certified_system"
                                         else "PIPE CANDIDATE" if state == "colour_correlated_review"
                                         else "ROUTE CANDIDATE"),
                "local_engineering_label": (
                    f"{system_label} | DIA {nominal_size or 'UNKNOWN'} | "
                    f"L2D {length:.3f} m" if length is not None else
                    f"{system_label} | DIA {nominal_size or 'UNKNOWN'} | L2D UNKNOWN"),
                "source_fragment_refs": occurrence.get("source_fragment_refs", []),
                "semantic_quantity_eligible": state == "certified_system",
                "physical_continuity_established": False,
                "installed_length_m": None,
            })

    by_page_assignments = defaultdict(list)
    for row in assignments:
        by_page_assignments[row["page_ref"]].append(row)
    for page_ref, members in by_page_assignments.items():
        def position(row):
            points = row.get("points_display", [])
            if not points:
                return (float("inf"), float("inf"), row["occurrence_ref"])
            centre = points[len(points) // 2]
            return (centre[1], centre[0], row["occurrence_ref"])
        for index, row in enumerate(sorted(members, key=position), 1):
            row["segment_display_id"] = f"P{page_number[page_ref]:02d}-S{index:04d}"

    trace_members = defaultdict(list)
    for row in assignments:
        unknown_colour = row.get("native_stroke_hex") or "NO-COLOR"
        trace_members[(row["page_ref"], row["system"],
                       unknown_colour if row["system"] is None else None)].append(row)
    traces = []
    trace_ref_by_key = {}
    for (page_ref, system, unknown_colour), members in sorted(
            trace_members.items(), key=lambda item: (
                page_number[item[0][0]], item[0][1] or "~", item[0][2] or "")):
        number = page_number[page_ref]
        display_id = (f"P{number:02d}-{SYSTEM_ABBREVIATIONS[system]}" if system else
                      f"P{number:02d}-C{unknown_colour.removeprefix('#')}")
        trace_id = _stable_id("mep_audit_page_trace", page_ref, system or "unknown",
                              unknown_colour or "system")
        trace_ref_by_key[(page_ref, system, unknown_colour)] = trace_id
        counts = Counter(row["display_state"] for row in members)
        topology = _topology_summary(members, fragment_by_id)
        traces.append({
            "id": trace_id, "display_id": display_id, "page_ref": page_ref,
            "page_number": number, "system": system,
            "native_colour_review_group": unknown_colour if system is None else None,
            "display_state": "identified_and_review_members" if system else "unknown_engineer_review",
            "member_occurrence_refs": sorted(row["occurrence_ref"] for row in members),
            "member_route_ledger_refs": sorted({row["route_ledger_ref"] for row in members}),
            "counts": {
                "certified_system_segment_count": counts["certified_system"],
                "colour_correlated_review_segment_count": counts["colour_correlated_review"],
                "unknown_engineer_review_segment_count": counts["unknown_engineer_review"],
                "rendered_segment_count": len(members),
            },
            "visible_projected_length_m": round(sum(
                row.get("projected_2d_length_m") or 0.0 for row in members), 8),
            "certified_semantic_length_m": round(semantic_length[(page_ref, system)], 8)
                if system else None,
            "topology": topology,
            "installed_length_m": None,
            "quantity_eligible": False,
        })
    for row in assignments:
        unknown_colour = row.get("native_stroke_hex") or "NO-COLOR"
        row["trace_ref"] = trace_ref_by_key[(row["page_ref"], row["system"],
                                             unknown_colour if row["system"] is None else None)]

    payload = {
        "schema_version": SCHEMA_VERSION, "layer": "mep_audit_trace_network",
        "source_pdf_sha256": source_pdf_sha256,
        "upstream": {
            "observed_layer": observed.get("layer"), "semantic_layer": semantic.get("layer"),
            "route_graph_layer": route_graph.get("layer"),
        },
        "colour_mapping_certificates": colour_certificates,
        "route_occurrence_assignments": sorted(assignments, key=lambda row: row["occurrence_ref"]),
        "page_system_traces": traces,
        "summary": {
            "rendered_route_occurrence_count": len(assignments),
            "certified_system_occurrence_count": sum(
                row["display_state"] == "certified_system" for row in assignments),
            "colour_correlated_review_occurrence_count": sum(
                row["display_state"] == "colour_correlated_review" for row in assignments),
            "unknown_engineer_review_occurrence_count": sum(
                row["display_state"] == "unknown_engineer_review" for row in assignments),
            "page_system_trace_count": sum(row["system"] is not None for row in traces),
            "unknown_native_colour_trace_count": sum(row["system"] is None for row in traces),
        },
        "authority": {
            "all_retained_route_occurrences_rendered": True,
            "colour_is_review_correlation_not_identity": True,
            "physical_continuity_established": False,
            "installed_length_established": False,
            "quantity_eligible": False,
        },
    }
    errors = validate_mep_audit_trace_network(payload, expected_occurrence_count=len(assignments))
    if errors:
        raise ValueError("invalid audit trace network: " + "; ".join(errors))
    return payload


def validate_mep_audit_trace_network(payload: Mapping[str, Any],
                                     expected_occurrence_count: int | None = None) -> list[str]:
    errors = []
    assignments = payload.get("route_occurrence_assignments", [])
    traces = {row.get("id"): row for row in payload.get("page_system_traces", [])}
    refs = [row.get("occurrence_ref") for row in assignments]
    if len(refs) != len(set(refs)):
        errors.append("route occurrences are not assigned exactly once")
    if expected_occurrence_count is not None and len(refs) != expected_occurrence_count:
        errors.append("route occurrence coverage differs from expected")
    for row in assignments:
        if row.get("trace_ref") not in traces:
            errors.append("assignment lacks page trace")
        if row.get("display_state") == "colour_correlated_review" and row.get("semantic_quantity_eligible"):
            errors.append("colour correlation gained semantic quantity authority")
        if row.get("installed_length_m") is not None or row.get("physical_continuity_established"):
            errors.append("audit assignment gained physical authority")
    if payload.get("authority", {}).get("quantity_eligible") is not False:
        errors.append("audit trace network must remain quantity ineligible")
    return sorted(set(errors))
