#!/usr/bin/env python3
"""Fresh native callouts, local M4, and system-only straight/bend applicability.

Page selection controls execution only. No legacy route graph selects text or
native geometry. Unsupported terminals, targets and connections abstain.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile

import fitz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.drawing_engine.pipelines.generate_mep_stroke_ownership import read, write
from tools.generate_mep_local_callout_certificate import RecordingNativeQueries
from src.drawing_engine.disciplines.mep.mep_automatic_target_binding import (
    RegionIndex, build_automatic_targets, certify_automatic_targets, _leader_paths, _contact,
    automatic_binding_evidence, apply_geometric_text_applicability, binding_ownership_context,
)
from src.drawing_engine.disciplines.mep.mep_attribute_binding import build_mep_attribute_bindings, validate_mep_attribute_bindings
from src.drawing_engine.disciplines.mep.mep_boundary_connection_replay import replay_boundary_connections
from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256, _sha256
from src.drawing_engine.disciplines.mep.mep_native_boundary_queries import NativeBoundaryQueries
from src.drawing_engine.disciplines.mep.mep_native_boundary_connections import discover_native_boundary_connections, propagate_collinear_applicability, _ports
from src.drawing_engine.disciplines.mep.mep_native_bend_connections import discover_native_bends
from src.drawing_engine.disciplines.mep.mep_projected_trace_completion import certify_junction_interior
from src.drawing_engine.disciplines.mep.mep_native_path_pack import NativePathPack
from src.drawing_engine.disciplines.mep.mep_native_descriptor_pack import NativeDescriptorPack
from src.drawing_engine.disciplines.mep.mep_route_observations import build_mep_route_graph
from src.drawing_engine.disciplines.mep.mep_native_target_discovery import _intersects
from src.drawing_engine.disciplines.mep.mep_source_primitive_denominator import VIEW_ROLES
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import build_mep_terminology_proposals
from src.drawing_engine.disciplines.mep.mep_text_observations import extract_mep_text_observations, build_mep_document_text_proposals


class LiveQueries(NativeBoundaryQueries):
    def with_initial_search_refs(self, rows):
        return rows


def m4(terms, evidence, targets):
    return build_mep_attribute_bindings(terminology_proposals=terms, binding_evidence=evidence,
        route_graph=targets['route_graph'], outlined_route_composites=targets['composites'])


def attach_native_leaders(targets,registry,denominator):
    """Register observed annotation primitives without granting them route status."""
    refs={r['source_primitive_ref'] for p in targets['route_graph']['pages'] for r in p['fragments']}
    refs.update(ref for leader in targets['leader_observations'] for path in leader['paths'] for ref in path['source_primitive_refs'])
    pack_info=denominator['native_descriptor_pack']
    pack=NativeDescriptorPack(pack_info['path'],pack_info)
    if not pack.verify_hashes():
        raise ValueError('native descriptor inventory changed')
    natives=[row['source_native_segment'] for row in pack.rows(lambda d:d['source_primitive_ref'] in refs)]
    if {r['id'] for r in natives}!=refs:
        raise ValueError('native evidence references missing from exhaustive denominator')
    graph=build_mep_route_graph(sheet_registry=registry,page_inputs={denominator['page_ref']:{
        'native_topology':{'segments':natives},'vertex_tolerance_display_points':.05,
        'gap_tolerance_display_points':.05}})
    rebound=certify_automatic_targets(graph=graph,witnesses=targets['envelope_searches'],
        leader_records=targets['leader_observations'])
    # Ownership exclusions cannot be undone by annotation registration.
    owned={c['id'] for c in targets['composites']['accepted_composites']}
    rebound['composites']['accepted_composites']=[c for c in rebound['composites']['accepted_composites'] if c['id'] in owned]
    for c in rebound['composites']['candidates']:
        if c['state']=='accepted' and c['id'] not in owned:
            c['state']='abstained';c['reasons'].append('accepted_drawing_view_ownership_missing')
    counts=Counter(c['state'] for c in rebound['composites']['candidates'])
    rebound['composites']['summary'].update(accepted_composite_count=counts['accepted'],
        abstained_candidate_count=counts['abstained'],state_counts=dict(counts))
    rebound['boundary_connections']=targets.get('boundary_connections',[])
    if 'junction_interior_version' in targets:
        rebound['junction_interior_version']=targets['junction_interior_version']
    return rebound


def connection_applicability(targets):
    """Boundary paths do not excuse hidden interior ink or multi-port branches."""
    connections = deepcopy(targets['boundary_connections'])
    for row in connections:
        if row['relation_type'] not in {'projected_collinear_boundary_join','projected_native_bend'}:
            row['state'] = 'abstained'
            row['reasons'] = sorted(set(row['reasons']+['interface_applicability_not_in_this_slice']))
        elif row['state'] == 'accepted':
            # Frozen v1 fixtures retain their historical interpretation. New
            # captures explicitly select v2; no existing artifact is rewritten.
            coverage = certify_junction_interior(row,version=targets.get('junction_interior_version',1))
            if not coverage['projected_scope_complete']:
                row['state'] = 'abstained'
                row['reasons'] = sorted(set(row['reasons']+coverage['reasons']))
    return connections


def bind_page(terms, targets, ownership_queries):
    """Existing M4 remains the sole local and extended attribute owner."""
    evidence, unresolved = automatic_binding_evidence(targets=targets, terminology=terms,
        stroke_ownership_queries=ownership_queries)
    scoped, evidence = apply_geometric_text_applicability(terminology=terms, binding_evidence=evidence)
    direct = m4(scoped, evidence, targets)
    replay_boundary_connections({'m3_payload_sha256': _sha256(targets['route_graph']),
        'm35_payload_sha256': _sha256(targets['composites']),
        'connections': targets['boundary_connections']}, targets['route_graph'], direct)
    extended_terms, extended_evidence = propagate_collinear_applicability(
        terminology=scoped, binding_evidence=evidence, direct_bindings=direct,
        connections=connection_applicability(targets), proposal_types=('system',))
    extended = m4(extended_terms, extended_evidence, targets)
    if validate_mep_attribute_bindings(direct) or validate_mep_attribute_bindings(extended):
        raise ValueError('M4 validation failed')
    return direct, extended, unresolved


def generate(args):
    out = args.output
    if out.exists() and not args.resume:
        raise ValueError('use a new output directory')
    out.mkdir(parents=True,exist_ok=args.resume)
    registry, denominator = read(args.registry), read(args.denominator)
    digest = _file_sha256(args.source)
    if digest != registry['document']['source_pdf_sha256'] or args.page != denominator['page_number']:
        raise ValueError('source or denominator page mismatch')
    if digest != denominator['source']['pdf_sha256']:
        raise ValueError('denominator source changed')
    scope = next(p for p in registry['pages'] if p['page_number'] == args.page)
    # All native text is extracted before any target or legacy graph is read.
    all_terms = build_mep_document_text_proposals(extract_mep_text_observations(
        pdf_path=args.source, sheet_registry=registry))
    observations = [r for r in all_terms['source_observations'] if r['page_ref'] == scope['page_ref']]
    terms = build_mep_terminology_proposals(document=all_terms['document'], observations=observations)
    eligible = {p['anchor_ref'] for p in terms['proposals'] if p['proposal_type'] in {'system','inline_size','elevation'}}
    seeds = [o for o in observations if o['id'] in eligible]
    write(out/'terminology.json.gz', terms)
    print(json.dumps({'phase':'native_text', 'page':args.page, 'text_rows':len(observations), 'eligible_callouts':len(seeds)}), flush=True)
    with fitz.open(args.source) as pdf:
        profile = {}
        fresh_path=out/'fresh-targets.json.gz'
        if args.resume and fresh_path.exists():
            targets=read(fresh_path)
            if targets['route_graph']['document']!=registry['document']:
                raise ValueError('resumed discovery source differs')
            profile={'resumed_fresh_geometry_sha256':_file_sha256(fresh_path)}
        else:
            with tempfile.TemporaryDirectory(dir=ROOT/'tmp', prefix='mep-callouts-') as scratch:
                with RegionIndex.from_native_page(page=pdf[args.page-1], page_ref=scope['page_ref'],
                        page_size=scope['page_size_display'], sqlite_path=Path(scratch)/'native.sqlite',
                        performance_profile=profile, observations=seeds) as index:
                    print(json.dumps({'phase':'fresh_native_index','native_segments':profile['native_descriptor_count']}), flush=True)
                    targets = build_automatic_targets(sheet_registry=registry,
                        page_indexes={scope['page_ref']:index}, observations=[], envelope_engine='page_topology')
        print(json.dumps({'phase':'fresh_targets','composites':len(targets['composites']['accepted_composites'])}), flush=True)
        write(out/'fresh-targets.json.gz',targets)
        # Ownership is independent of geometry/M4 membership. Use the compact
        # authored-path region certificates, never legacy candidate-role bits.
        pack_info = denominator['native_authored_path_pack']
        pack = NativePathPack(pack_info['path'], pack_info)
        if not pack.verify_hash():
            raise ValueError('path pack hash mismatch')
        import re
        required = {int(re.match(r'drawing\[(\d+)\]', ref)[1])
            for c in targets['composites']['accepted_composites'] for ref in c['member_source_primitive_refs']}
        roles = {r['drawing_ordinal']:r['region_role'] for r in pack.records() if r['drawing_ordinal'] in required}
        owned, excluded = [], []
        for c in targets['composites']['accepted_composites']:
            codes = {roles.get(int(re.match(r'drawing\[(\d+)\]', ref)[1])) for ref in c['member_source_primitive_refs']}
            (owned if codes and codes <= VIEW_ROLES else excluded).append(c)
        targets['composites']['accepted_composites'] = owned
        excluded_ids = {c['id'] for c in excluded}
        for c in targets['composites']['candidates']:
            if c['id'] in excluded_ids:
                c['state'] = 'abstained'
                c['reasons'] = sorted(set(c['reasons']+['accepted_drawing_view_ownership_missing']))
        summary = targets['composites']['summary']
        summary['accepted_composite_count'] = len(owned)
        summary['abstained_candidate_count'] = len(targets['composites']['candidates'])-len(owned)
        summary['state_counts'] = dict(Counter(c['state'] for c in targets['composites']['candidates']))
        write(out/'region-exclusions.json.gz', excluded)
        native = LiveQueries(pdf[args.page-1], scope['page_ref'])
        leaders, captures = [], []
        for number, observation in enumerate(seeds):
            path = out/'native-callouts'/f'{number:04d}.json.gz'
            if args.resume and path.exists():
                capture=read(path)
                if capture['observation']!=observation:
                    raise ValueError('resumed native callout text differs')
                leader=capture['leader']
            else:
                recording = RecordingNativeQueries(native)
                paths, complete = _leader_paths(recording, observation)
                leader = {'observation_ref':observation['id'], 'page_ref':scope['page_ref'],
                          'paths':paths, 'search_complete':complete}
                capture = {'observation':observation, 'leader':leader, 'queries':recording.queries}
                write(path, capture)
            leaders.append(leader)
            captures.append({'path':str(path.resolve()), 'sha256':_file_sha256(path), 'observation_ref':observation['id']})
            if number % 25 == 0:
                print(json.dumps({'phase':'live_leaders','processed':number+1,'total':len(seeds)}), flush=True)
        targets['leader_observations'] = leaders
        targets=attach_native_leaders(targets,registry,denominator)
        owned=targets['composites']['accepted_composites']
        # Capture ownership only for complete unique contacts; all other chains
        # still pass through M4 and retain their explicit failure reasons.
        ownership_paths = {}
        for number, (observation, leader, capture_ref) in enumerate(zip(seeds, leaders, captures)):
            if not leader['search_complete'] or len(leader['paths']) != 1:
                continue
            path = leader['paths'][0]
            if sum(_contact(path['contact_point_display'], c) for c in owned) != 1:
                continue
            capture = read(Path(capture_ref['path']))
            points = [pt for q in capture['queries'] for pt in (q['bbox_display'][:2], q['bbox_display'][2:])]
            box = [min(p[0] for p in points),min(p[1] for p in points),max(p[0] for p in points),max(p[1] for p in points)]
            rows, complete, refs = native.query(box)
            query = {'id':observation['id']+'.ownership', 'page_ref':scope['page_ref'],
                'source_rows':rows, 'binding_context':binding_ownership_context(targets, observation, leader),
                'source_observations':[o for o in observations if _intersects(box,o['bbox_display'])],
                'search':{'bbox_display':box,'complete':complete,'region_refs':refs,
                    'budget_exhausted':False,'row_budget':None,'source_rows_sha256':_sha256(rows),
                    'all_source_refs_sha256':_sha256(sorted(r['source_primitive_ref'] for r in rows))}}
            path = out/'ownership'/f'{number:04d}.json.gz'
            write(path,query)
            ownership_paths[observation['id']] = {'path':str(path.resolve()),'sha256':_file_sha256(path)}
            print(json.dumps({'phase':'ownership','callout_index':number,'source_rows':len(rows)}),flush=True)
        arguments = dict(index=native, composites=targets['composites'], graph=targets['route_graph'], page_ref=scope['page_ref'])
        connections = [*discover_native_boundary_connections(**arguments), *discover_native_bends(**arguments)]
        incidence = defaultdict(list)
        for row in connections:
            for port in row['port_refs']:
                incidence[port].append(row)
        for row in connections:
            if any(len(incidence[port]) != 1 for port in row['port_refs']):
                row['state'] = 'abstained'
                row['reasons'] = sorted(set(row['reasons']+['non_unique_projected_port_destination']))
        targets['boundary_connections'] = connections
        targets['junction_interior_version'] = 2
    write(out/'targets.json.gz', targets)
    write(out/'capture-index.json', {'callouts':captures,'ownership':ownership_paths})
    write(out/'source.json', {'source_pdf_sha256':digest,'page':args.page,'page_ref':scope['page_ref'],
        'registry_path':str(args.registry.resolve()),'denominator_sha256':_file_sha256(args.denominator),
        'denominator_path':str(args.denominator.resolve()),'native_index_profile':profile,
        'legacy_M3_used_as_input':False,'discovery_minimum_straight_member_points':24})
    evaluate(out)


class OwnershipQueries:
    """Load one captured scope at a time rather than expanding a page of JSON."""
    def __init__(self, index):
        self.index = index

    def get(self, ref):
        info = self.index.get(ref)
        if info is None:
            return None
        path = Path(info['path'])
        if _file_sha256(path) != info['sha256']:
            raise ValueError('ownership file changed')
        return read(path)


def replay_page(out):
    terms, targets, index = (read(out/name) for name in (
        'terminology.json.gz','targets.json.gz','capture-index.json'))
    observations = {r['id']:r for r in terms['source_observations']}
    expected = {p['anchor_ref'] for p in terms['proposals'] if p['proposal_type'] in {'system','inline_size','elevation'}}
    if len(index['callouts']) != len(expected) or {r['observation_ref'] for r in index['callouts']} != expected:
        raise ValueError('eligible native callout coverage changed')
    leaders = []
    for info in index['callouts']:
        path = Path(info['path'])
        if _file_sha256(path)!=info['sha256']:
            raise ValueError('native callout capture changed')
        capture = read(path)
        observation = observations[info['observation_ref']]
        if observation != capture['observation']:
            raise ValueError('native text differs from captured callout')
        queries = {}
        for q in capture['queries']:
            if q['source_rows_sha256']!=_sha256(q['source_rows']):
                raise ValueError('native query geometry changed')
            key=tuple(q['bbox_display'])
            if key in queries and queries[key]!=q:
                raise ValueError('conflicting native queries')
            queries[key]=q
        class ReplayQueries:
            def query(self,box):
                q=queries.get(tuple(box))
                if q is None:
                    raise ValueError('uncaptured native query')
                return q['source_rows'],q['complete'],q['region_refs']
        paths,complete=_leader_paths(ReplayQueries(),observation)
        leader={'observation_ref':observation['id'],'page_ref':observation['page_ref'],
            'paths':paths,'search_complete':complete}
        if leader!=capture['leader']:
            raise ValueError('native chain does not replay')
        leaders.append(leader)
    if leaders!=targets['leader_observations']:
        raise ValueError('local binding leader inventory changed')
    direct,extended,unresolved=bind_page(terms,targets,OwnershipQueries(index['ownership']))
    for actual,name in ((direct,'local-M4.json.gz'),(extended,'extended-M4.json.gz'),(unresolved,'abstentions.json.gz')):
        # Native geometry contains tuples in memory and arrays after JSON
        # serialization. Compare canonical payloads, preserving every value.
        if _sha256(actual)!=_sha256(read(out/name)):
            raise ValueError('M4 replay differs: '+name)
    return []


def integrate_captured_leaders(out):
    """Resume the serial M3-to-M4 handoff from already captured native queries."""
    source=read(out/'source.json')
    targets=read(out/'targets.json.gz')
    targets=attach_native_leaders(targets,read(Path(source['registry_path'])),read(Path(source['denominator_path'])))
    terms=read(out/'terminology.json.gz')
    observations={r['id']:r for r in terms['source_observations']}
    leaders={r['observation_ref']:r for r in targets['leader_observations']}
    index=read(out/'capture-index.json')
    for ref,info in index['ownership'].items():
        path=Path(info['path'])
        if _file_sha256(path)!=info['sha256']:
            raise ValueError('ownership input changed')
        query=read(path)
        query['binding_context']=binding_ownership_context(targets,observations[ref],leaders[ref])
        write(path,query)
        info['sha256']=_file_sha256(path)
    write(out/'targets.json.gz',targets)
    write(out/'capture-index.json',index)
    evaluate(out)


def evaluate(out):
    targets, terms = read(out/'targets.json.gz'), read(out/'terminology.json.gz')
    index = read(out/'capture-index.json')
    direct, extended, unresolved = bind_page(terms, targets, OwnershipQueries(index['ownership']))
    accepted = lambda payload: [r for r in payload['relations'] if r['state']=='accepted' and r['relation_type']=='route_system']
    local = accepted(direct)
    local_refs = {ref for r in local for ref in r['target_refs']}
    extended_refs = {ref for r in accepted(extended) for ref in r['target_refs']} - local_refs
    fragments = {r['id']:r for p in targets['route_graph']['pages'] for r in p['fragments']}
    ports = list(_ports(targets['composites']['accepted_composites'], fragments))
    incidents = defaultdict(list)
    applicability = connection_applicability(targets)
    for row in applicability:
        for ref in row['port_refs']:
            incidents[ref].append(row)
    stops = []
    for port in ports:
        if any(r['state']=='accepted' for r in incidents[port['id']]):
            continue
        reasons = sorted({reason for r in incidents[port['id']] for reason in r['reasons']})
        stops.append({'port':port, 'reasons':reasons or ['no_certified_straight_or_bend_interface'],
            'candidate_refs':[r['id'] for r in incidents[port['id']]]})
    summary = {'eligible_native_callouts':len(index['callouts']),
        'complete_unique_terminal_chains':sum(l['search_complete'] and len(l['paths'])==1 for l in targets['leader_observations']),
        'accepted_local_system_bindings':len(local), 'locally_identified_segments':len(local_refs),
        'system_extended_segments':len(extended_refs),
        'accepted_connection_types':dict(Counter(r['relation_type'] for r in applicability if r['state']=='accepted')),
        'unresolved_proposals':len(unresolved), 'unresolved_interfaces':len(stops),
        'stop_reasons':dict(Counter(reason for s in stops for reason in s['reasons'])),
        'wrong_accepts':None,'wrong_accept_measurement_state':'independent_review_required',
        'size_propagated':False,'elevation_propagated':False,'installed_length':None,'purchase_length':None,
        'complete_page_route_coverage':False}
    write(out/'local-M4.json.gz',direct)
    write(out/'extended-M4.json.gz',extended)
    write(out/'abstentions.json.gz',unresolved)
    write(out/'connection-stops.json.gz',stops)
    write(out/'connection-applicability.json.gz',applicability)
    write(out/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'M&P mark-up against shop systems piping.pdf')
    parser.add_argument('--registry',type=Path,default=ROOT/'fixtures/mep/m_and_p_coordination/m_and_p_coordination.sheet-registry.json')
    parser.add_argument('--page',type=int,required=True)
    parser.add_argument('--denominator',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    generate(args)
