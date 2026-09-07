"""M5C projected run/network hierarchy over frozen M3/M4 and replayed M5B.

An exact native endpoint junction can join a *projected trace*, not certify a
physical fitting. Branches split runs. M5B alone owns cross-sheet certificates.
Composite outline endpoints cannot masquerade as centreline endpoints.
Geometric eligibility and semantic compatibility are independent results.
"""

from collections import Counter, defaultdict
from copy import deepcopy
from itertools import combinations
import math

from src.drawing_engine.disciplines.mep.mep_cross_sheet_runs import (
    _canonical_sha256, _stable_id, _relation_indexes, _compatible_attributes,
    _style_compatible, build_mep_cross_sheet_runs,
)


VERSION = "0.2.0"
LAYER = "mep_projected_network_hierarchy"


def _semantic_overlay(fragment_refs, relations):
    fields = {}
    for field in ('system', 'size', 'elevation'):
        rows = [r for r in relations if r['relation_type'] == 'route_' + field
                and set(r['target_fragment_refs']).intersection(fragment_refs)]
        accepted = sorted((r for r in rows if r['state'] == 'accepted'), key=lambda r: r['id'])
        values = {_canonical_sha256({k: v for k, v in r['candidate'].items()
                                    if k not in {'raw_text', 'terminology_entry_ref'}}) for r in accepted}
        conflicts = sorted({ref for r in rows for ref in r.get('conflicts', [])})
        fields[field] = {'state': 'conflicted' if len(values) > 1 or conflicts else 'accepted' if values else 'unknown',
                         'accepted_relation_refs': [r['id'] for r in accepted],
                         'observed_values': [deepcopy(r['candidate']) for r in accepted],
                         'conflicting_relation_refs': conflicts,
                         'unresolved_relation_refs': sorted(r['id'] for r in rows if r['state'] != 'accepted')}
    return fields


def _groups(refs, connections):
    parent = {ref: ref for ref in refs}

    def find(ref):
        while parent[ref] != ref:
            parent[ref] = parent[parent[ref]]
            ref = parent[ref]
        return ref

    for group in connections:
        roots = sorted({find(ref) for ref in group})
        for root in roots[1:]:
            parent[root] = roots[0]
    output = defaultdict(list)
    for ref in sorted(parent):
        output[find(ref)].append(ref)
    return sorted(output.values())


def build_mep_networks(*, sheet_registry, route_graph, attribute_bindings,
                       cross_sheet_runs=None, boundary_connections=None,
                       projected_identity_bindings=None, fitting_replay=None,
                       trace_completion=None, trace_replay=None):
    """Build quantity-free hierarchy; verify inputs against their exact hashes."""
    if attribute_bindings.get("m3_contract_ref", {}).get("payload_sha256") != _canonical_sha256(route_graph):
        raise ValueError("M4 does not reference this frozen M3 graph")
    # Replay, rather than trusting an accepted flag or a hash-shaped string.
    replay = build_mep_cross_sheet_runs(sheet_registry=sheet_registry,
        route_graph=route_graph, attribute_bindings=attribute_bindings)
    if cross_sheet_runs is not None and cross_sheet_runs != replay:
        raise ValueError("M5B does not replay on the supplied M1/M3/M4 inputs")
    m5 = replay
    # M4 embeds geometric certificates independently of its semantic relations.
    # Rebuild them from the complete supplied M3 graph, including competitors;
    # removing an optional annotation cannot remove a certified outline.
    composite_by_id = {r['id']: r for r in attribute_bindings.get('outlined_route_composites', [])}
    replay_by_id = {}
    if composite_by_id or (projected_identity_bindings or {}).get('fitting_bindings'):
        from src.drawing_engine.disciplines.mep.mep_outlined_route_composites import build_mep_outlined_route_composites
        geometric_replay = build_mep_outlined_route_composites(route_graph=route_graph)
        replay_by_id = {r['id']: r for r in geometric_replay['accepted_composites']}
        if any(replay_by_id.get(ref) != row for ref, row in composite_by_id.items()):
            raise ValueError('M4 outlined geometry does not replay from frozen M3')
    by_fragment, _ = _relation_indexes(attribute_bindings)
    fragments = {row["id"]: row for page in route_graph["pages"] for row in page["fragments"]}
    accepted = [r for r in attribute_bindings["relations"] if r["state"] == "accepted"]
    attributed = {ref for r in accepted if r["relation_type"] in
                  {"route_system", "route_size", "route_elevation"}
                  for ref in r["target_fragment_refs"]}
    occurrences = [r for r in m5["projected_route_occurrences"]
                   if r['route_target_ref'] in composite_by_id or attributed.intersection(r["source_fragment_refs"])]
    occurrence_by_target = {r["route_target_ref"]: r for r in occurrences}
    # Deduplicate only M5B-certified intervals, never equal labels or spans.
    duplicate_groups = [[ref for ref in r["source_route_target_refs"]
                         if ref in occurrence_by_target]
                        for r in m5["canonical_projected_segments"]]
    segments, segment_by_target = [], {}
    for group in _groups(occurrence_by_target, duplicate_groups):
        rows = [occurrence_by_target[ref] for ref in group]
        child_refs = sorted(r["id"] for r in rows)
        segment_id = _stable_id("mep_network_segment", route_graph["document"], child_refs)
        fragment_refs = sorted({ref for r in rows for ref in r["source_fragment_refs"]})
        segment = {"id": segment_id, "record_type": "mep_network_segment", "record_version": VERSION,
            "source_occurrence_refs": child_refs, "route_target_refs": group,
            "source_fragment_refs": fragment_refs,
            "page_refs": sorted({r["page_ref"] for r in rows}),
            "attribute_relation_refs": sorted({r["id"] for r in accepted
                if set(r["target_fragment_refs"]).intersection(fragment_refs)}),
            "canonical_segment_refs": sorted(r["id"] for r in m5["canonical_projected_segments"]
                if set(r["source_route_target_refs"]).issubset(group)),
            "state": "derived", "quantity_eligible": False}
        overlay = _semantic_overlay(fragment_refs, attribute_bindings['relations'])
        composite_refs = sorted(ref for ref in group if ref in composite_by_id)
        segment.update(geometric_eligibility={'state': 'accepted',
            'basis': 'replayed_m3_5_outline' if composite_refs else 'm4_bound_single_stroke',
            'certificate_refs': composite_refs or segment['attribute_relation_refs'],
            'physical_identity_established': False}, semantic_overlay=overlay,
            geometry_only=not bool(attributed.intersection(fragment_refs)),
            unobserved_attributes=[field for field, value in overlay.items() if value['state'] == 'unknown'],
            physical_continuation_established=False)
        segments.append(segment)
        for ref in group:
            segment_by_target[ref] = segment_id

    # Only unconsumed single-stroke occurrences can use M3 endpoint topology.
    ordinary = {r["fragment_ref"]: segment_by_target[r["route_target_ref"]]
                for r in occurrences if r.get("fragment_ref")}
    attachments = []
    boundaries = set()
    for relation in accepted:
        if relation["relation_type"] in {"route_system", "route_size", "route_elevation"}:
            continue
        member_ids = sorted({segment_by_target[r["route_target_ref"]] for r in occurrences
            if set(r["source_fragment_refs"]).intersection(relation["target_fragment_refs"])})
        if not member_ids:
            continue
        boundaries.update(member_ids)
        attachments.append({"id": _stable_id("mep_network_attachment", relation["id"]),
            "record_type": "mep_network_attachment", "record_version": VERSION,
            "page_ref": relation["page_ref"], "relation_type": relation["relation_type"],
            "source_relation_ref": relation["id"], "segment_refs": member_ids,
            "target_refs": deepcopy(relation["target_refs"]),
            "state": "derived", "quantity_eligible": False})

    junctions, joined_endpoints = [], set()
    for page in route_graph["pages"]:
        endpoint_by_id = {r["id"]: r for r in page["endpoints"]}
        for vertex in page["vertices"]:
            refs = vertex["fragment_refs"]
            selected = [ref for ref in refs if ref in ordinary]
            if len(refs) < 2 or not selected:
                continue
            reasons, evidence = [], {vertex["id"]}
            endpoints = [endpoint_by_id[ref] for ref in vertex["endpoint_refs"]]
            if set(selected) != set(refs):
                reasons.append("unresolved_or_composite_incident_target")
            if any(fragments[ref]["source_kind"] != "native_pdf_vector" for ref in refs):
                reasons.append("native_endpoint_trace_required")
            # M3 clustering tolerates gaps. Do not silently promote that proximity.
            if not endpoints or any(math.dist(a["point_display"], b["point_display"]) > 1e-6
                                    for a, b in combinations(endpoints, 2)):
                reasons.append("clustered_endpoints_are_not_exactly_coincident")
            directions = []
            for endpoint in endpoints:
                points = fragments[endpoint["fragment_ref"]]["geometry"]["points_display"]
                near = points[1] if endpoint["role"] == "start" else points[-2]
                vector = [near[i] - endpoint["point_display"][i] for i in (0, 1)]
                length = math.hypot(*vector)
                if length:
                    directions.append([v / length for v in vector])
            if len(directions) != len(endpoints) or any(
                    sum(a[i] * b[i] for i in (0, 1)) > 1 - 1e-8
                    for a, b in combinations(directions, 2)):
                reasons.append("overlapping_or_degenerate_incident_paths")
            semantic_failures = ['conflicting_bound_' + field for field, value in
                _semantic_overlay(refs, attribute_bindings['relations']).items() if value['state'] == 'conflicted']
            for left, right in combinations(refs, 2):
                _, failures, _, attr_refs = _compatible_attributes(left, right, by_fragment)
                semantic_failures.extend(failures)
                evidence.update(attr_refs)
                if not _style_compatible(fragments[left]["style"], fragments[right]["style"]):
                    reasons.append("incompatible_non_color_style")
            members = sorted({ordinary[ref] for ref in selected})
            evidence.update(vertex["endpoint_refs"])
            is_accepted = not reasons
            junctions.append({"id": _stable_id("mep_projected_junction", vertex["id"], members),
                "record_type": "mep_network_junction", "record_version": VERSION,
                "page_ref": page["page_ref"], "point_display": vertex["point_display"],
                "relation_type": "projected_branch" if len(refs) > 2 else "projected_endpoint_join",
                "segment_refs": members, "endpoint_refs": vertex["endpoint_refs"],
                "evidence_refs": sorted(evidence), "state": "accepted" if is_accepted else "abstained",
                "reasons": sorted(set(reasons)), "physical_continuation_established": False,
                "semantic_conflicts": sorted({reason for reason in semantic_failures if 'missing' not in reason}),
                "unresolved_physical_reasons": sorted(set(semantic_failures)),
                "run_pass_through": is_accepted and len(refs) == 2 and not boundaries.intersection(members),
                "quantity_eligible": False})
            if is_accepted:
                joined_endpoints.update(vertex["endpoint_refs"])

    for relation in m5["continuation_candidates"]:
        refs = [relation["source_fragment_ref"], relation["target_fragment_ref"]]
        members = sorted({ordinary[ref] for ref in refs if ref in ordinary})
        reasons = list(relation["reasons"])
        if any(ref not in ordinary for ref in refs) or len(members) != 2:
            reasons.append("continuation_requires_two_distinct_unconsumed_segments")
        is_accepted = relation["state"] == "accepted" and not reasons
        endpoint_refs = [relation["source_endpoint_ref"], relation["target_endpoint_ref"]]
        junctions.append({"id": _stable_id("mep_network_continuation", relation["id"]),
            "record_type": "mep_network_junction", "record_version": VERSION,
            "relation_type": "cross_sheet_continuation", "segment_refs": members,
            "endpoint_refs": endpoint_refs, "evidence_refs": [relation["id"]],
            "state": "accepted" if is_accepted else "abstained", "reasons": sorted(set(reasons)),
            "physical_continuation_established": is_accepted,
            "run_pass_through": is_accepted,
            "quantity_eligible": False})
        if is_accepted:
            joined_endpoints.update(endpoint_refs)

    if boundary_connections is not None:
        from src.drawing_engine.disciplines.mep.mep_boundary_connection_replay import replay_boundary_connections
        from src.drawing_engine.disciplines.mep.mep_outlined_route_composites import _samples
        records = replay_boundary_connections(boundary_connections, route_graph, attribute_bindings)
        for connection in records:
            targets = connection['composite_refs']
            members = sorted({segment_by_target[ref] for ref in targets if ref in segment_by_target})
            reasons = list(connection['reasons'])
            evidence = {connection['id'], *connection['source_primitive_refs']}
            if len(members) != len(targets):
                reasons.append('distinct_geometrically_eligible_projected_segments_required')
            # Missing or conflicting semantics never erase replayed geometry.
            # They remain explicit blockers to a compatible physical route.
            target_fragments = {ref for target in targets for ref in composite_by_id[target]['member_fragment_refs']}
            unresolved_physical = ['conflicting_bound_' + field for field, value in
                _semantic_overlay(target_fragments, attribute_bindings['relations']).items() if value['state'] == 'conflicted']
            for left, right in combinations([composite_by_id[ref]['member_fragment_refs'][0] for ref in targets], 2):
                _, failures, _, attr_refs = _compatible_attributes(left, right, by_fragment)
                unresolved_physical.extend(failures)
                evidence.update(attr_refs)
            accepted_connection = connection['state'] == 'accepted' and not reasons
            centers = [r['center'] for r in connection['ports']]
            # Three candidate ports are not an ordered centreline polyline.
            points = centers if len(centers) == 2 else []
            if connection['relation_type'] == 'projected_native_bend' and len(connection['boundary_paths']) == 2:
                paths = [_samples(r['points_display'], 17) for r in connection['boundary_paths']]
                points = [[(a + b) / 2 for a, b in zip(p, q)] for p, q in zip(*paths)]
            junctions.append({'id': _stable_id('mep_network_boundary_junction', connection['id']),
                'record_type': 'mep_network_junction', 'record_version': VERSION,
                'page_ref': connection['page_ref'], 'relation_type': connection['relation_type'],
                'segment_refs': members, 'endpoint_refs': connection['port_refs'],
                'port_segment_bindings': [{'port_ref': port['id'], 'composite_ref': port['composite_ref'],
                    'segment_ref': segment_by_target.get(port['composite_ref']),
                    'composite_endpoint_index': port.get('end'),
                    'point_display': port['center'], 'outward': deepcopy(port.get('outward')),
                    'side_points_display': deepcopy(port.get('sides', []))}
                    for port in connection['ports']],
                'point_display': [sum(values) / len(centers) for values in zip(*centers)],
                **({'port_points_display': centers} if len(centers) > 2 else {}),
                'centreline_points_display': points, 'source_boundary_connection_ref': connection['id'],
                'source_search': deepcopy(connection.get('search', {})),
                'evidence_refs': sorted(evidence), 'state': 'accepted' if accepted_connection else 'abstained',
                'reasons': sorted(set(reasons)), 'unresolved_physical_reasons': sorted(set(unresolved_physical)),
                'semantic_conflicts': sorted({reason for reason in unresolved_physical if 'missing' not in reason}),
                'geometric_certificate_state': connection['state'], 'epistemic_state': 'derived',
                'physical_continuation_established': False,
                'run_pass_through': accepted_connection and len(members) == 2 and not boundaries.intersection(members),
                'quantity_eligible': False})

    breaks = []
    for page in route_graph["pages"]:
        for endpoint in page["endpoints"]:
            if endpoint["fragment_ref"] not in ordinary or endpoint["id"] in joined_endpoints:
                continue
            breaks.append({"id": _stable_id("mep_network_break", endpoint["id"]),
                "record_type": "mep_network_unresolved_boundary", "record_version": VERSION,
                "page_ref": page["page_ref"], "segment_ref": ordinary[endpoint["fragment_ref"]],
                "endpoint_ref": endpoint["id"], "point_display": endpoint["point_display"],
                "reason": "terminal_or_connection_not_certified", "state": "unknown", "quantity_eligible": False})
    for occurrence in occurrences:
        if occurrence.get("route_composite_ref"):
            breaks.append({"id": _stable_id("mep_network_break", occurrence["id"]),
                "record_type": "mep_network_unresolved_boundary", "record_version": VERSION,
                "page_ref": occurrence["page_ref"], "segment_ref": segment_by_target[occurrence["route_target_ref"]],
                "source_occurrence_ref": occurrence["id"],
                "reason": "complete_composite_terminal_and_physical_continuity_not_certified",
                "state": "unknown", "quantity_eligible": False})

    if projected_identity_bindings is not None:
        from src.drawing_engine.disciplines.mep.mep_projected_identity_bindings import projected_fitting_bindings
        from src.drawing_engine.disciplines.mep.mep_fitting_hypotheses import replay_fitting_hypotheses
        companion = projected_identity_bindings
        if (companion.get('layer') != 'mep_projected_identity_bindings' or
                companion.get('document') != route_graph['document'] or
                companion.get('input_payload_sha256', {}).get('attribute-bindings') != _canonical_sha256(attribute_bindings)):
            raise ValueError('projected identity companion does not match M4')
        if companion.get('fitting_bindings'):
            if not fitting_replay:
                raise ValueError('projected fitting network requires independent native replay')
            payload = fitting_replay['payload']
            if (payload['m3_payload_sha256'] != _canonical_sha256(route_graph) or
                    companion['input_payload_sha256'].get('fitting-hypotheses') != _canonical_sha256(payload)):
                raise ValueError('projected fitting network differs from source evidence')
            replayed = replay_fitting_hypotheses(payload, graph=route_graph,
                composites=fitting_replay['composites'], source_queries=fitting_replay['source_queries'])
            if projected_fitting_bindings(replayed) != companion['fitting_bindings']:
                raise ValueError('projected fitting bindings do not replay')
        for fitting in companion.get('fitting_bindings', []):
            if not fitting['accepted_projected_connection']:
                continue
            members = []
            for composite in fitting['input_composites']:
                ref = composite['id']
                if ref not in segment_by_target:
                    if replay_by_id.get(ref) != composite:
                        raise ValueError('fitting geometry is outside replayed M3.5 eligibility')
                    sid = _stable_id('mep_network_geometry_segment', route_graph['document'], ref)
                    fragment_refs = composite['member_fragment_refs']
                    overlay = _semantic_overlay(fragment_refs, attribute_bindings['relations'])
                    segments.append({'id': sid, 'record_type': 'mep_network_segment', 'record_version': VERSION,
                        'route_target_refs': [ref], 'source_fragment_refs': fragment_refs,
                        'source_occurrence_refs': [], 'canonical_segment_refs': [],
                        'page_refs': [composite['page_ref']],
                        'points_display': deepcopy(composite['derived_geometry']['centreline_points_display']),
                        'source_primitive_refs': composite['member_source_primitive_refs'],
                        'attribute_relation_refs': sorted(r['id'] for r in accepted if
                            set(r['target_fragment_refs']).intersection(fragment_refs)),
                        'source_projected_identity_binding_ref': fitting['id'],
                        'geometric_eligibility': {'state': 'accepted', 'basis': 'replayed_m3_5_outline',
                            'certificate_refs': [ref], 'physical_identity_established': False},
                        'semantic_overlay': overlay, 'geometry_only': not bool(attributed.intersection(fragment_refs)),
                        'unobserved_attributes': [field for field,value in overlay.items() if value['state']=='unknown'],
                        'state': 'derived', 'physical_continuation_established': False, 'quantity_eligible': False})
                    segment_by_target[ref] = sid
                    breaks.append({'id': _stable_id('mep_network_break', sid),
                        'record_type': 'mep_network_unresolved_boundary', 'record_version': VERSION,
                        'page_ref': composite['page_ref'], 'segment_ref': sid,
                        'reason': 'complete_composite_terminal_and_physical_continuity_not_certified',
                        'state': 'unknown', 'quantity_eligible': False})
                members.append(segment_by_target[ref])
            semantic_failures, semantic_evidence = [], set()
            for left, right in combinations([c['member_fragment_refs'][0] for c in fitting['input_composites']], 2):
                _, failures, _, refs = _compatible_attributes(left, right, by_fragment)
                semantic_failures.extend(failures)
                semantic_evidence.update(refs)
            junctions.append({'id': _stable_id('mep_network_fitting_junction', fitting['id']),
                'record_type': 'mep_network_junction', 'record_version': VERSION,
                'page_ref': fitting['page_ref'], 'relation_type': fitting['relation_type'],
                'segment_refs': sorted(members), 'endpoint_refs': [p['id'] for p in fitting['ports']],
                'port_points_display': [p['center'] for p in fitting['ports']],
                'port_segment_bindings': [{'port_ref': p['id'], 'composite_ref': p['composite_ref'],
                    'segment_ref': segment_by_target[p['composite_ref']], 'point_display': p['center']} for p in fitting['ports']],
                'point_display': fitting['centreline_junction_display'], 'centreline_points_display': [],
                'source_projected_identity_binding_ref': fitting['id'],
                'source_fitting_hypothesis_ref': fitting['source_certificate_ref'],
                'body_paths': [path for candidate in fitting['body_candidates']
                    if candidate['body']['id'] == fitting['selected_body_ref']
                    for path in candidate['body']['native_boundary_paths']],
                'evidence_refs': sorted({fitting['id'], *fitting['source_primitive_refs'], *semantic_evidence}),
                'state': 'accepted', 'epistemic_state': 'inferred', 'reasons': [],
                'classification_alternatives': fitting['classification_alternatives'],
                'semantic_conflicts': sorted({reason for reason in semantic_failures if 'missing' not in reason}),
                'unresolved_physical_reasons': sorted(set(semantic_failures) | {'physical_fitting_continuity_unresolved'}),
                'physical_continuation_established': False, 'run_pass_through': False,
                'automatic_attribute_propagation': False, 'quantity_eligible': False})

    scoped_traces = []
    connected_completion = None
    if trace_completion is not None:
        from src.drawing_engine.disciplines.mep.mep_projected_trace_completion import replay_projected_trace_completion
        if (not trace_replay or boundary_connections is None
                or _canonical_sha256(trace_replay['graph']) != _canonical_sha256(route_graph)
                or _canonical_sha256(trace_replay['boundary_connections']) != _canonical_sha256(boundary_connections)
                or _canonical_sha256(trace_replay['composites']) != attribute_bindings['m3_5_contract_ref']['payload_sha256']):
            raise ValueError('trace completion does not reference this frozen geometric scope')
        replayed_completion = replay_projected_trace_completion(trace_completion, **trace_replay)
        scoped_traces = replayed_completion['scoped_traces']
        connected_completion = replayed_completion.get('connected_trace_completion')
    covered_targets = {ref for trace in scoped_traces if trace['projected_scope_complete'] for ref in trace['composite_refs']}
    for segment in segments:
        traces = [t for t in scoped_traces if set(t['composite_refs']).intersection(segment['route_target_refs'])]
        segment['scoped_trace_refs'] = sorted(t['id'] for t in traces)
        segment['projected_scope_complete'] = bool(segment['route_target_refs']) and set(segment['route_target_refs']).issubset(covered_targets)
        segment['unresolved_trace_reasons'] = sorted({reason for t in traces for reason in t['reasons']})
        if not segment['projected_scope_complete'] and not segment['unresolved_trace_reasons']:
            segment['unresolved_trace_reasons'] = ['source_corridor_search_and_endpoint_interfaces_not_certified']
    ids = {s["id"] for s in segments}
    segments_by_id = {s['id']: s for s in segments}
    joined = [j for j in junctions if j["state"] == "accepted"]
    runs = []
    for group in _groups(ids, [j["segment_refs"] for j in joined if j["run_pass_through"]]):
        local_joins = [j for j in joined if set(j['segment_refs']).issubset(group)]
        endpoint_interfaces = [j for j in junctions if any(
            binding.get('segment_ref') in group
            for binding in j.get('port_segment_bindings', [])
        )]
        projected_endpoint_occurrences = []
        for segment_ref in group:
            for composite_ref in segments_by_id[segment_ref].get('route_target_refs', []):
                composite = composite_by_id.get(composite_ref)
                points = (composite or {}).get('derived_geometry', {}).get(
                    'centreline_points_display', [])
                if len(points) < 2:
                    continue
                for endpoint_index, point in ((0, points[0]), (1, points[-1])):
                    candidates = sorted({
                        junction['id']
                        for junction in endpoint_interfaces
                        for binding in junction.get('port_segment_bindings', [])
                        if binding.get('segment_ref') == segment_ref
                        and binding.get('composite_ref') == composite_ref
                        and binding.get('composite_endpoint_index') == endpoint_index
                    })
                    searches = [
                        junction.get('source_search', {})
                        for junction in endpoint_interfaces
                        if junction['id'] in candidates
                    ]
                    projected_endpoint_occurrences.append({
                        'segment_ref': segment_ref, 'composite_ref': composite_ref,
                        'page_ref': composite.get('page_ref'),
                        'composite_endpoint_index': endpoint_index,
                        'point_display': deepcopy(point),
                        'interface_candidate_refs': candidates,
                        'complete_candidate_search_refs': sorted({
                            ref for search in searches if search.get('complete') is True
                            for ref in search.get('region_refs', [])
                        }),
                        'search_state': ('ambiguous' if candidates else 'not_searched'),
                        'physical_terminal_established': False,
                        'quantity_eligible': False,
                    })
        uncovered = [ref for ref in group if not segments_by_id[ref]['projected_scope_complete']]
        unsupported_joins = [j['id'] for j in local_joins if not j.get('source_boundary_connection_ref')]
        trace_reasons = sorted({reason for ref in uncovered for reason in segments_by_id[ref]['unresolved_trace_reasons']})
        if unsupported_joins:
            trace_reasons.append('junction_interior_search_not_certified')
        complete = bool(group) and not uncovered and not unsupported_joins
        runs.append({"id": _stable_id("mep_projected_run", group),
            "record_type": "mep_projected_run", "record_version": VERSION,
            "segment_refs": group,
            "junction_refs": sorted(j["id"] for j in joined if set(j["segment_refs"]).issubset(group)),
            "unresolved_boundary_refs": sorted(b["id"] for b in breaks if b["segment_ref"] in group),
            "trace_state": "complete_scoped_projected_trace" if complete else "partial_trace" if len(group) > 1 else "segment_only",
            "complete_trace_established": complete, "physical_continuation_established": False,
            'completion_scope': 'listed_segments_between_certified_native_junction_interfaces',
            'scoped_trace_refs': sorted({ref for member in group for ref in segments_by_id[member]['scoped_trace_refs']}),
            'projected_endpoint_occurrences': projected_endpoint_occurrences,
            'endpoint_interface_candidate_refs': sorted(j['id'] for j in endpoint_interfaces),
            'unresolved_endpoint_interface_refs': sorted(
                j['id'] for j in endpoint_interfaces
                if j.get('state') != 'accepted'
                or j.get('physical_continuation_established') is not True
            ),
            'endpoint_interface_search_complete': bool(projected_endpoint_occurrences) and all(
                row['interface_candidate_refs'] for row in projected_endpoint_occurrences
            ),
            'uncovered_segment_refs': uncovered, 'unsearched_junction_refs': unsupported_joins,
            'unresolved_trace_reasons': trace_reasons,
            'unresolved_physical_reasons': ['physical_fitting_and_vertical_continuity_not_certified'],
            "unresolved_reasons": trace_reasons + ["physical_fitting_and_vertical_continuity_not_certified"],
            "state": "derived", "quantity_eligible": False})
    networks = []
    for group in _groups(ids, [j["segment_refs"] for j in joined]):
        networks.append({"id": _stable_id("mep_projected_network", group),
            "record_type": "mep_projected_network", "record_version": VERSION,
            "segment_refs": group, "run_refs": sorted(r["id"] for r in runs if set(r["segment_refs"]).issubset(group)),
            "junction_refs": sorted(j["id"] for j in joined if set(j["segment_refs"]).issubset(group)),
            "connectivity_state": "connected_projected" if len(group) > 1 else "isolated_segment",
            "physical_network_established": False, "state": "derived", "quantity_eligible": False})
    return {"schema_version": VERSION, "layer": LAYER, "document": deepcopy(route_graph["document"]),
        "input_payload_sha256": {"m1": _canonical_sha256(sheet_registry), "m3": _canonical_sha256(route_graph),
            "m4": _canonical_sha256(attribute_bindings), "m5b": _canonical_sha256(m5),
            **({'native_boundary_connections': _canonical_sha256(boundary_connections)} if boundary_connections is not None else {}),
            **({'trace_completion': _canonical_sha256(trace_completion)} if trace_completion is not None else {}),
            **({'projected_identity_bindings': _canonical_sha256(projected_identity_bindings)} if projected_identity_bindings is not None else {})},
        "segments": sorted(segments, key=lambda r: r["id"]),
        "junctions": sorted(junctions, key=lambda r: r["id"]), "runs": runs, "networks": networks,
        "attachments": sorted(attachments, key=lambda r: r["id"]), 'scoped_traces': deepcopy(scoped_traces),
        **({'connected_trace_completion': deepcopy(connected_completion)} if connected_completion is not None else {}),
        "unresolved_boundaries": sorted(breaks, key=lambda r: r["id"]),
        "coverage": {"source_occurrence_count": len(m5["projected_route_occurrences"]),
            "attributed_occurrence_count": sum(bool(attributed.intersection(r['source_fragment_refs'])) for r in occurrences),
            "geometrically_eligible_occurrence_count": len(occurrences),
            'complete_search_established': bool(segments) and all(s['projected_scope_complete'] for s in segments),
            'search_scope': 'eligible_projected_segments_only', 'complete_page_inventory_established': False,
            "eligibility_contract": {'version': '0.2.0', 'replayed_outlines_eligible_without_attributes': True,
                'unclassified_single_strokes_eligible': False, 'semantic_conflicts_preserved': True,
                'm5b_duplicate_and_physical_continuation_gates_unchanged': True},
            "excluded_observations": [{'source_occurrence_ref': r['id'], 'route_target_ref': r['route_target_ref'],
                'page_ref': r['page_ref'], 'reason': 'no_replayed_outline_or_independent_single_stroke_binding'}
                for r in m5['projected_route_occurrences'] if r['route_target_ref'] not in occurrence_by_target],
            "excluded_occurrence_refs": sorted(r["id"] for r in m5["projected_route_occurrences"]
                if r["route_target_ref"] not in occurrence_by_target)},
        "summary": {"segment_count": len(segments), "run_count": len(runs),
            "multi_segment_run_count": sum(len(r["segment_refs"]) > 1 for r in runs),
            "connected_projected_network_count": sum(len(r["segment_refs"]) > 1 for r in networks),
            "junction_states": dict(Counter(j["state"] for j in junctions)),
            "complete_run_count": sum(r['complete_trace_established'] for r in runs),
            'complete_projected_scope_count': sum(t['projected_scope_complete'] for t in scoped_traces),
            'geometry_only_segment_count': sum(s.get('geometry_only', False) for s in segments),
            'projected_fitting_branch_count': sum(j['relation_type'] == 'projected_fitting_body_branch' and j['state'] == 'accepted' for j in junctions)},
        "quantity_eligible": False}
