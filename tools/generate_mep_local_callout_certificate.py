#!/usr/bin/env python3
"""Close one native callout through existing M4, then extend system over a bend.

Frozen discovery nominates candidates only. Live native queries must reproduce
the leader, retain all contact competitors and replay both bend boundaries.
No reviewed coordinates or colour-based identity selection are used.
"""
from copy import deepcopy
import argparse
import gzip
import json
from pathlib import Path
import sys

import fitz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.drawing_engine.pipelines.generate_mep_automatic_items import _write
from src.drawing_engine.disciplines.mep.mep_automatic_target_binding import (
    _leader_paths, automatic_binding_evidence, apply_geometric_text_applicability,
    binding_ownership_context,
)
from src.drawing_engine.disciplines.mep.mep_attribute_binding import build_mep_attribute_bindings, validate_mep_attribute_bindings
from src.drawing_engine.disciplines.mep.mep_boundary_connection_replay import replay_boundary_connections
from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256, _file_sha256
from src.drawing_engine.disciplines.mep.mep_native_boundary_connections import propagate_collinear_applicability
from src.drawing_engine.disciplines.mep.mep_native_boundary_queries import NativeBoundaryQueries
from src.drawing_engine.disciplines.mep.mep_native_bend_connections import trace_bend_boundaries
from src.drawing_engine.disciplines.mep.mep_native_target_discovery import _intersects
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import build_mep_terminology_proposals, _stable_id
from src.drawing_engine.disciplines.mep.mep_text_observations import extract_mep_text_observations, build_mep_document_text_proposals


def bind_local(terms, targets, ownership_queries=None):
    evidence, unresolved = automatic_binding_evidence(
        targets=targets, terminology=terms, stroke_ownership_queries=ownership_queries)
    scoped, evidence = apply_geometric_text_applicability(
        terminology=terms, binding_evidence=evidence)
    bindings = build_mep_attribute_bindings(
        terminology_proposals=scoped, route_graph=targets['route_graph'],
        outlined_route_composites=targets['composites'], binding_evidence=evidence)
    return scoped, evidence, bindings, unresolved


class RecordingNativeQueries:
    def __init__(self, native):
        self.native, self.queries = native, []
        self._exact_queries = {}

    def query(self, box):
        key = tuple(box)
        if key in self._exact_queries:
            return self._exact_queries[key]
        rows, complete, refs = self.native.query(box)
        self.queries.append({
            'bbox_display': list(box), 'complete': complete, 'region_refs': refs,
            'source_rows': rows, 'source_rows_sha256': _sha256(rows),
            'all_source_refs_sha256': _sha256(sorted(r['source_primitive_ref'] for r in rows)),
        })
        self._exact_queries[key] = (rows, complete, refs)
        return rows, complete, refs


def replay_certificate(payload):
    """Source-free replay checks captured queries and both M4 stages."""
    observation = payload['observation']
    queries = payload['leader_native_queries']
    for query in queries:
        if not query['complete'] or query['source_rows_sha256'] != _sha256(query['source_rows']):
            raise ValueError('incomplete or changed native leader query')

    class ReplayQueries:
        def query(self, box):
            matches = [q for q in queries if q['bbox_display'] == list(box)]
            if not matches or any(q != matches[0] for q in matches):
                raise ValueError('missing or conflicting native leader query')
            row = matches[0]
            return row['source_rows'], row['complete'], row['region_refs']

    paths, complete = _leader_paths(ReplayQueries(), observation)
    leader = {'observation_ref': observation['id'], 'page_ref': observation['page_ref'],
              'paths': paths, 'search_complete': complete}
    if leader != payload['leader']:
        raise ValueError('native leader does not replay')
    targets = payload['targets']
    terms, evidence, direct, unresolved = bind_local(
        payload['local_terminology'], targets,
        {observation['id']: payload['ownership_query']})
    if direct != payload['local_M4_bindings'] or unresolved != payload['nearby_unresolved_proposals']:
        raise ValueError('local callout M4 does not replay')
    bend = payload['bend']
    replay_boundary_connections({
        'm3_payload_sha256': _sha256(targets['route_graph']),
        'm35_payload_sha256': _sha256(targets['composites']),
        'connections': [bend]}, targets['route_graph'], direct)
    expanded_terms, expanded_evidence = propagate_collinear_applicability(
        terminology=terms, binding_evidence=evidence, connections=[bend],
        direct_bindings=direct, proposal_types=('system',))
    extended = build_mep_attribute_bindings(
        terminology_proposals=expanded_terms, binding_evidence=expanded_evidence,
        route_graph=targets['route_graph'], outlined_route_composites=targets['composites'])
    if extended != payload['extended_M4_bindings']:
        raise ValueError('system-only bend applicability does not replay')
    return []


def generate(source, registry_path, targets_path, bends_path, output):
    if output.exists():
        raise ValueError('certificate output must be new')
    registry = json.loads(registry_path.read_text())
    targets = json.loads(targets_path.read_text())
    bends = json.loads(gzip.decompress(bends_path.read_bytes()))
    digest = _file_sha256(source)
    if any(doc['source_pdf_sha256'] != digest for doc in (
            registry['document'], targets['route_graph']['document'], bends['document'])):
        raise ValueError('source snapshots differ')
    terms = build_mep_document_text_proposals(extract_mep_text_observations(
        pdf_path=source, sheet_registry=registry))
    initial_terms, initial_evidence, nominated, _ = bind_local(terms, targets)
    proposals = {r['id']: r for r in initial_terms['proposals']}
    observations = {r['id']: r for r in initial_terms['source_observations']}
    candidates = []
    for bend in bends['connections']:
        if bend['state'] != 'accepted' or bend['relation_type'] != 'projected_native_bend':
            continue
        for relation in nominated['relations']:
            if (relation['state'] == 'accepted' and relation['relation_type'] == 'route_system'
                    and len(relation['target_refs']) == 1
                    and relation['target_refs'][0] in bend['composite_refs']):
                original = observations[proposals[relation['proposal_ref']]['anchor_ref']]
                while original.get('source_observation_ref') in observations:
                    original = observations[original['source_observation_ref']]
                candidates.append((bend, relation, original))
    if not candidates:
        raise ValueError('no native-callout candidate at a replayable bend')
    # A fixture is selected from the complete nominated list, not used to
    # resolve competing targets for any one annotation.
    bend, nomination, observation = min(candidates, key=lambda item: (item[0]['id'], item[1]['id']))
    scope = next(p for p in registry['pages'] if p['page_ref'] == observation['page_ref'])
    print(json.dumps({'phase': 'native_query_capture', 'source_page': scope['page_number'],
                      'callout': observation['text'], 'nominated_fixture_count': len(candidates)}), flush=True)
    with fitz.open(source) as pdf:
        native = NativeBoundaryQueries(pdf[scope['page_number'] - 1], scope['page_ref'])
        recording = RecordingNativeQueries(native)
        paths, complete = _leader_paths(recording, observation)
        leader = {'observation_ref': observation['id'], 'page_ref': scope['page_ref'],
                  'paths': paths, 'search_complete': complete}
        if not complete or len(paths) != 1:
            raise ValueError('live native leader is incomplete or ambiguous')
        targets = deepcopy(targets)
        targets['leader_observations'] = [leader]
        points = [p for q in recording.queries for p in (q['bbox_display'][:2], q['bbox_display'][2:])]
        box = [min(p[0] for p in points), min(p[1] for p in points),
               max(p[0] for p in points), max(p[1] for p in points)]
        sources, closed, refs = native.query(box)
        ownership = {
            'id': _stable_id('mep_local_callout_ownership_query', observation['id'], box),
            'page_ref': scope['page_ref'], 'source_rows': sources,
            'binding_context': binding_ownership_context(targets, observation, leader),
            'source_observations': [row for row in terms['source_observations']
                                    if row['page_ref'] == scope['page_ref']
                                    and _intersects(box, row['bbox_display'])],
            'search': {'bbox_display': box, 'complete': closed, 'region_refs': refs,
                       'budget_exhausted': False, 'row_budget': None,
                       'source_rows_sha256': _sha256(sources),
                       'all_source_refs_sha256': _sha256(sorted(r['source_primitive_ref'] for r in sources))},
        }
        local_terms = build_mep_terminology_proposals(document=terms['document'],
                                                     observations=ownership['source_observations'])
        # All nearby native text stays in M2 as competitors; only this leader
        # can supply binding evidence. Other unresolved text is not erased.
        direct_terms, evidence, direct, unresolved = bind_local(
            local_terms, targets, {observation['id']: ownership})
        selected_proposal_ids = {r['id'] for r in local_terms['proposals'] if r['anchor_ref'] == observation['id']}
        selected_unresolved = [r for r in unresolved if r['id'] in selected_proposal_ids]
        accepted = [r for r in direct['relations'] if r['state'] == 'accepted' and r['relation_type'] == 'route_system']
        if selected_unresolved or len(accepted) != 1:
            raise ValueError({'local_callout_failed': selected_unresolved, 'accepted_systems': len(accepted)})
        if accepted[0]['target_refs'] != nomination['target_refs']:
            raise ValueError('live callout changed the nominated bounded target')
        bend_sources, bend_complete, bend_refs = native.query(bend['search']['bbox_display'])
        boundary_paths, reasons = trace_bend_boundaries(bend['ports'], bend_sources)
        if not bend_complete or reasons or boundary_paths != bend['boundary_paths']:
            raise ValueError({'live_bend_failed': reasons})
        bend = deepcopy(bend)
        bend['source_rows'] = bend_sources
        bend['search'].update(complete=bend_complete, region_refs=bend_refs,
                             all_source_refs_sha256=_sha256(sorted(r['source_primitive_ref'] for r in bend_sources)))
    replay_boundary_connections({'m3_payload_sha256': _sha256(targets['route_graph']),
        'm35_payload_sha256': _sha256(targets['composites']), 'connections': [bend]}, targets['route_graph'], direct)
    expanded_terms, expanded_evidence = propagate_collinear_applicability(
        terminology=direct_terms, binding_evidence=evidence, connections=[bend],
        direct_bindings=direct, proposal_types=('system',))
    extended = build_mep_attribute_bindings(terminology_proposals=expanded_terms,
        binding_evidence=expanded_evidence, route_graph=targets['route_graph'],
        outlined_route_composites=targets['composites'])
    source_target = accepted[0]['target_refs'][0]
    destination = next(ref for ref in bend['composite_refs'] if ref != source_target)
    extensions = [r for r in extended['relations'] if r['state'] == 'accepted' and destination in r['target_refs']]
    if len(extensions) != 1 or extensions[0]['relation_type'] != 'route_system':
        raise ValueError('bend did not close exactly one system-only extension')
    if validate_mep_attribute_bindings(direct) or validate_mep_attribute_bindings(extended):
        raise ValueError('M4 validation failed')
    payload = {
        'schema_version': 'mep_local_callout_certificate.v1',
        'source': {'path': str(source.resolve()), 'sha256': digest, 'page_number': scope['page_number']},
        'inputs': {str(path): _file_sha256(path) for path in (registry_path, targets_path, bends_path)},
        'candidate_fixture_count': len(candidates), 'observation': observation, 'leader': leader,
        'leader_native_queries': recording.queries, 'ownership_query': ownership,
        'local_terminology': local_terms, 'targets': targets,
        'local_M4_bindings': direct, 'extended_M4_bindings': extended, 'bend': bend,
        'nearby_unresolved_proposals': unresolved,
        'summary': {'system': accepted[0]['candidate']['kind'], 'local_target': source_target,
                    'extended_target': destination, 'local_system_binding_accepted': True,
                    'bend_system_binding_accepted': True, 'size_propagated': False,
                    'elevation_propagated': False, 'complete_local_native_queries': True,
                    'source_page_coverage_established': False,
                    'installed_length': None, 'purchase_length': None},
    }
    # Replay keeps unrelated text abstentions, not a false requirement that
    # every nearby annotation must bind to this one pipe.
    replay_certificate(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write(output, payload)
    print(json.dumps(payload['summary'], indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'M&P mark-up against shop systems piping.pdf')
    parser.add_argument('--registry', type=Path, default=ROOT/'fixtures/mep/m_and_p_coordination/m_and_p_coordination.sheet-registry.json')
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--bends', type=Path, default=ROOT/'fixtures/mep/m_and_p_coordination/native-boundary-replay.json.gz')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    generate(args.source, args.registry, args.targets, args.bends, args.output)


if __name__ == '__main__':
    main()
