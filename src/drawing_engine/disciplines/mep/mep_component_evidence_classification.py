"""Evidence classification for bounded projected MEP geometry candidates.

Outlined-pair geometry is not itself MEP evidence.  This layer classifies a
frozen geometry component only after inspecting independent M4 annotations,
accepted fitting/equipment relations, certified connection evidence, and
package-level anchored route-style correlations.  Colour correlation can
support an unidentified MEP candidate but never supplies its system identity.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "0.1.0"
LAYER = "mep_projected_component_evidence_classification"
STATES = {
    "identified_mep_route",
    "supported_unidentified_mep_candidate",
    "unclassified_view_geometry",
    "non_route_drawing_content",
}
NON_VIEW_ROLES = {
    "title_block_revision_stamp", "border", "legend", "schedule_table",
    "notes_specifications",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _stable_id(kind: str, *parts: Any) -> str:
    return f"{kind}.{hashlib.sha256(_canonical(parts).encode()).hexdigest()[:20]}"


def _colour_signature(value: Any) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    return tuple(round(float(channel), 6) for channel in value)


def anchored_style_hypotheses(*, candidates, local_bindings, profiles):
    """Audit-only style correlation over already supported route candidates.

    Profiles come from native paths and page-local region ownership. Neither a
    matching colour nor this function can nominate new route geometry or M4.
    Two distinct locally bound targets are required; extended identities cannot
    amplify their own colour evidence. Conflicting proposals remain visible.
    """
    anchors = defaultdict(list)
    for relation in local_bindings:
        if relation.get('state') != 'accepted' or relation.get('relation_type') != 'route_system':
            continue
        for ref in relation['target_refs']:
            profile = profiles.get(ref, {})
            if profile.get('key') is not None:
                anchors[tuple(profile['key'])].append((ref, relation['id'], relation['candidate']['kind']))
    mappings, by_key = [], {}
    for key, rows in sorted(anchors.items()):
        rows = sorted(set(rows))
        systems = sorted({r[2] for r in rows})
        targets = sorted({r[0] for r in rows})
        state = ('conflicting_anchors' if len(systems) != 1 else
                 'unique_anchored_style_hypothesis' if len(targets) >= 2 else 'insufficient_local_anchors')
        record = {'id': _stable_id('mep_current_anchored_style_hypothesis', key, rows),
                  'scope_and_native_style': key, 'state': state, 'system_candidates': systems,
                  'anchor_target_refs': targets, 'anchor_relation_refs': sorted({r[1] for r in rows}),
                  'anchor_source_profiles': {ref: profiles[ref] for ref in targets},
                  'system_identity_established': False, 'quantity_eligible': False}
        mappings.append(record); by_key[key] = record
    hypotheses = []
    for row in candidates:
        profile = profiles.get(row['id'], {})
        mapping = by_key.get(tuple(profile['key'])) if profile.get('key') is not None else None
        eligible = (row.get('state') == 'supported_unidentified_mep_candidate'
                    and profile.get('route_role_checks_passed') is True
                    and not row.get('route_candidate_support', {}).get('transverse_attachment_negative'))
        existing = set(row.get('candidate_systems') or [])
        systems = sorted(existing | set(mapping['system_candidates'] if mapping else []))
        supported = (eligible and mapping is not None
                     and mapping['state'] == 'unique_anchored_style_hypothesis' and len(systems) == 1)
        hypotheses.append({'candidate_ref': row['id'],
            'state': 'colour_style_supported' if supported else 'unresolved',
            'system_hypothesis': systems[0] if supported else None,
            'mapping_ref': mapping['id'] if mapping else None,
            'anchor_relation_refs': mapping['anchor_relation_refs'] if mapping else [],
            'conflicting_systems': systems if len(systems) > 1 else [],
            'reason': ('unique_current_local_callout_style_correlation' if supported else
                       'candidate_role_checks_not_closed' if not eligible else
                       'conflicting_system_hypotheses' if len(systems) > 1 else
                       mapping['state'] if mapping else 'unmapped_native_style'),
            'source_profile': profile, 'system_identity_established': False,
            'size_elevation_connectivity_established': False, 'quantity_eligible': False})
    return {'mappings': mappings, 'hypotheses': hypotheses}


def _measured_gate(name: str, refs: set[str], basis: str) -> dict[str, Any]:
    return {
        "gate": name, "state": "measured",
        "promoted_source_reference_count": len(refs),
        "source_primitive_refs": sorted(refs),
        "measurement_basis": basis,
    }


def _unresolved_gate(name: str, reason: str) -> dict[str, Any]:
    return {
        "gate": name, "state": "unresolved",
        "promoted_source_reference_count": None,
        "source_primitive_refs": [], "measurement_basis": None,
        "reason": reason,
    }


def classify_projected_components(
    *, recovery: Mapping[str, Any], route_page: Mapping[str, Any],
    composites: Sequence[Mapping[str, Any]], m4_relations: Sequence[Mapping[str, Any]],
    colour_mapping_certificates: Sequence[Mapping[str, Any]],
    source_dispositions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Classify frozen components without changing geometry membership."""
    page_ref = recovery["page_ref"]
    fragments = {row["id"]: row for row in route_page.get("fragments", [])}
    composite_by_id = {row["id"]: row for row in composites
                       if row.get("page_ref") == page_ref}
    binding_by_component = {row["component_ref"]: row
                            for row in recovery.get("post_geometry_M4_bindings", [])}
    anchored_colours = {
        signature: row
        for row in colour_mapping_certificates
        if row.get("state") == "accepted_one_to_one_review_correlation"
        and (signature := _colour_signature(row.get("native_stroke_rgb"))) is not None
    }
    accepted_interfaces = [
        row for row in m4_relations
        if row.get("page_ref") == page_ref and row.get("state") == "accepted"
        and row.get("relation_type") in {"fitting", "valve", "damper", "equipment_endpoint"}
    ]
    interface_by_fragment: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for relation in accepted_interfaces:
        for fragment_ref in relation.get("target_fragment_refs", []):
            interface_by_fragment[fragment_ref].append(relation)

    records = []
    for component in recovery.get("geometry_components", []):
        if component.get("page_ref") != page_ref:
            continue
        binding = binding_by_component[component["id"]]
        composite_refs = component["composite_refs"]
        member_fragment_refs = sorted({
            fragment_ref for composite_ref in composite_refs
            for fragment_ref in composite_by_id[composite_ref].get("member_fragment_refs", [])
        })
        source_refs = set(component.get("source_segment_refs", []))
        disposition_rows = [source_dispositions[ref] for ref in source_refs
                            if ref in source_dispositions]
        explicit_non_route_refs = {
            ref for ref in source_refs
            if ref in source_dispositions and (
                source_dispositions[ref].get("region_role") in NON_VIEW_ROLES
                or source_dispositions[ref].get("primary_disposition")
                in {"drawing_furniture", "excluded_non_view_content"}
                or "drawing_furniture" in source_dispositions[ref].get("candidate_roles", []))
        }
        source_colours = sorted({
            signature for fragment_ref in member_fragment_refs
            if fragment_ref in fragments
            and (signature := _colour_signature(
                fragments[fragment_ref].get("style", {}).get("stroke"))) is not None
        })
        anchored = [anchored_colours[colour] for colour in source_colours
                    if colour in anchored_colours]
        system = binding["attributes"]["route_system"]
        applicable_annotations = [
            binding["attributes"][kind] for kind in ("route_size", "route_elevation")
            if binding["attributes"][kind].get("state") == "accepted"
        ]
        interface_relations = sorted({
            relation["id"] for fragment_ref in member_fragment_refs
            for relation in interface_by_fragment.get(fragment_ref, [])
        })
        support = []
        if applicable_annotations:
            support.append({
                "kind": "applicable_M4_annotation",
                "evidence_refs": sorted({ref for row in applicable_annotations
                                         for ref in row.get("relation_refs", [])}),
            })
        if interface_relations:
            support.append({
                "kind": "accepted_fitting_or_equipment_port_relation",
                "evidence_refs": interface_relations,
            })
        if anchored:
            support.append({
                "kind": "anchored_route_style_correlation",
                "evidence_refs": sorted(row["id"] for row in anchored),
                "native_stroke_rgb": [list(row["native_stroke_rgb"]) for row in anchored],
                "system_identity_established": False,
            })

        if explicit_non_route_refs:
            state = "non_route_drawing_content"
            reason = "explicit_source_disposition_or_non_view_region"
        elif system.get("state") == "accepted":
            state = "identified_mep_route"
            reason = "unique_post_geometry_M4_system_binding"
        elif support:
            state = "supported_unidentified_mep_candidate"
            reason = "independent_MEP_support_without_system_identity"
        else:
            state = "unclassified_view_geometry"
            reason = "outlined_pair_geometry_without_independent_MEP_evidence"
        records.append({
            "id": _stable_id("mep_component_evidence_classification", component["id"], state, support),
            "record_type": "mep_projected_component_evidence_classification",
            "state": state, "page_ref": page_ref,
            "component_ref": component["id"],
            "composite_refs": list(composite_refs),
            "source_segment_refs": sorted(source_refs),
            "member_fragment_refs": member_fragment_refs,
            "component_reason": reason,
            "independent_mep_support": support,
            "identified_system": system.get("value") if state == "identified_mep_route" else None,
            "identified_system_relation_refs": (
                system.get("relation_refs", []) if state == "identified_mep_route" else []),
            "source_stroke_rgb": [list(colour) for colour in source_colours],
            "explicit_non_route_source_refs": sorted(explicit_non_route_refs),
            "outlined_pair_alone_establishes_mep_route": False,
            "physical_continuity_established": False,
            "quantity_eligible": False,
        })

    all_promoted_refs = {
        ref for row in records
        if row["state"] in {"identified_mep_route", "supported_unidentified_mep_candidate"}
        for ref in row["source_segment_refs"]
    }
    promoted_composite_refs = {
        ref for row in records
        if row["state"] in {"identified_mep_route", "supported_unidentified_mep_candidate"}
        for ref in row["composite_refs"]
    }
    crossing_pairs = {
        frozenset(row.get("fragment_refs", []))
        for row in route_page.get("crossings", [])
        if len(row.get("fragment_refs", [])) == 2
    }
    crossing_join_refs: set[str] = set()
    incompatible_join_refs: set[str] = set()
    for transition in recovery.get("accepted_transitions", []):
        composite_refs = transition.get("composite_refs", [])
        if (len(composite_refs) != 2
                or any(ref not in promoted_composite_refs for ref in composite_refs)
                or any(ref not in composite_by_id for ref in composite_refs)):
            continue
        left, right = (composite_by_id[ref] for ref in composite_refs)
        left_fragments = set(left.get("member_fragment_refs", []))
        right_fragments = set(right.get("member_fragment_refs", []))
        if any(frozenset((a, b)) in crossing_pairs
               for a in left_fragments for b in right_fragments):
            crossing_join_refs.update(left.get("member_source_primitive_refs", []))
            crossing_join_refs.update(right.get("member_source_primitive_refs", []))
        left_styles = {
            (round(float(fragments[ref].get("style", {}).get("width_display_points", 0)), 6),
             fragments[ref].get("style", {}).get("dash_pattern"))
            for ref in left_fragments if ref in fragments
        }
        right_styles = {
            (round(float(fragments[ref].get("style", {}).get("width_display_points", 0)), 6),
             fragments[ref].get("style", {}).get("dash_pattern"))
            for ref in right_fragments if ref in fragments
        }
        if left_styles and right_styles and left_styles != right_styles:
            incompatible_join_refs.update(left.get("member_source_primitive_refs", []))
            incompatible_join_refs.update(right.get("member_source_primitive_refs", []))
    region_refs = defaultdict(set)
    disposition_refs = defaultdict(set)
    candidate_role_refs = defaultdict(set)
    for ref in all_promoted_refs:
        row = source_dispositions.get(ref, {})
        region_refs[row.get("region_role")].add(ref)
        disposition_refs[row.get("primary_disposition")].add(ref)
        for role in row.get("candidate_roles", []):
            candidate_role_refs[role].add(ref)
    negative_gates = {
        "title_block_revision_stamp": _measured_gate(
            "title_block_revision_stamp", region_refs["title_block_revision_stamp"],
            "component source refs intersected with denominator region ownership"),
        "border": _measured_gate(
            "border", region_refs["border"],
            "component source refs intersected with denominator region ownership"),
        "schedule_or_legend": _measured_gate(
            "schedule_or_legend", region_refs["schedule_table"] | region_refs["legend"],
            "component source refs intersected with denominator region ownership"),
        "dimension_or_annotation": _measured_gate(
            "dimension_or_annotation", candidate_role_refs["annotation_dimension"]
            | disposition_refs["annotation_dimension"],
            "component source refs intersected with denominator candidate and primary roles"),
        "drawing_furniture": _measured_gate(
            "drawing_furniture", candidate_role_refs["drawing_furniture"]
            | disposition_refs["drawing_furniture"],
            "component source refs intersected with denominator candidate and primary roles"),
        "hatching": _unresolved_gate(
            "hatching", "no independent hatch-ownership classifier covers every promoted source ref"),
        "room_or_building_outline": _unresolved_gate(
            "room_or_building_outline",
            "no independent architectural-boundary classifier covers every promoted source ref"),
        "crossing_only_contact": _measured_gate(
            "crossing_only_contact", crossing_join_refs,
            "accepted promoted transitions intersected with M3 unconnected crossing fragment pairs"),
        "incompatible_style_near_join": _measured_gate(
            "incompatible_style_near_join", incompatible_join_refs,
            "accepted promoted transitions compared by member stroke width and dash style"),
    }
    counts = Counter(row["state"] for row in records)
    payload = {
        "schema_version": SCHEMA_VERSION, "layer": LAYER,
        "development_status": "diagnostic_component_reclassification",
        "page_ref": page_ref,
        "geometry_components_sha256": recovery["geometry_components_sha256"],
        "classifications": records,
        "summary": {
            "component_count": len(records),
            "state_counts": {state: counts[state] for state in sorted(STATES)},
            "isolated_single_composite_count": sum(len(row["composite_refs"]) == 1 for row in records),
            "unidentified_isolated_single_composite_count": sum(
                row["state"] != "identified_mep_route" and len(row["composite_refs"]) == 1
                for row in records),
            "neutral_only_unidentified_component_count": sum(
                row["state"] != "identified_mep_route"
                and all(max(colour) - min(colour) < .18 for colour in row["source_stroke_rgb"])
                for row in records),
            "anchored_colour_supported_unidentified_count": sum(
                row["state"] == "supported_unidentified_mep_candidate"
                and any(item["kind"] == "anchored_route_style_correlation"
                        for item in row["independent_mep_support"])
                for row in records),
        },
        "negative_promotion_gates": negative_gates,
        "acceptance_gate": {
            "every_component_classified_once": len(records) == len(recovery.get("geometry_components", [])),
            "outlined_pair_alone_never_supports_MEP": all(
                row["state"] == "unclassified_view_geometry"
                for row in records if not row["independent_mep_support"]
                and row["identified_system"] is None and not row["explicit_non_route_source_refs"]),
            "all_negative_gates_measured": all(
                row["state"] == "measured" for row in negative_gates.values()),
            "engineer_facing_route_audit_eligible": False,
            "status": "diagnostic_only_negative_ownership_incomplete",
        },
        "authority": {
            "geometry_membership_changed": False,
            "colour_establishes_system_identity": False,
            "unclassified_geometry_is_MEP": False,
            "installed_length_established": False,
            "purchase_length_established": False,
            "quantity_eligible": False,
        },
    }
    return payload


def validate_component_evidence_classification(payload: Mapping[str, Any]) -> list[str]:
    errors = []
    rows = payload.get("classifications", [])
    if payload.get("layer") != LAYER:
        errors.append("unexpected layer")
    if len({row.get("component_ref") for row in rows}) != len(rows):
        errors.append("components are missing or multiply classified")
    for row in rows:
        if row.get("state") not in STATES:
            errors.append(f"invalid classification state: {row.get('component_ref')}")
        if row.get("state") == "identified_mep_route" and not row.get("identified_system"):
            errors.append(f"identified route lacks M4 system: {row.get('component_ref')}")
        if (row.get("state") == "supported_unidentified_mep_candidate"
                and not row.get("independent_mep_support")):
            errors.append(f"supported candidate lacks independent evidence: {row.get('component_ref')}")
        if (row.get("state") == "unclassified_view_geometry"
                and row.get("independent_mep_support")):
            errors.append(f"unclassified geometry has unused MEP support: {row.get('component_ref')}")
        if row.get("outlined_pair_alone_establishes_mep_route") is not False:
            errors.append(f"outlined geometry granted route authority: {row.get('component_ref')}")
    for gate in payload.get("negative_promotion_gates", {}).values():
        if gate.get("state") == "unresolved" and gate.get("promoted_source_reference_count") is not None:
            errors.append(f"unresolved negative gate has fabricated count: {gate.get('gate')}")
        if gate.get("state") == "measured" and (
                gate.get("promoted_source_reference_count")
                != len(gate.get("source_primitive_refs", []))):
            errors.append(f"measured negative gate count mismatch: {gate.get('gate')}")
    return errors
