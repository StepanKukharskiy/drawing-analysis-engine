#!/usr/bin/env python3
"""Capture/replay scoped annotation ownership for all nominated page contacts.

The input is frozen automatic discovery, never reviewed target selectors. This
bounded experiment extends M4 evidence; it does not rewrite the source snapshot.
"""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.drawing_engine.project.mep_json_field import read_field
from src.drawing_engine.disciplines.mep.mep_automatic_target_binding import _contact, binding_ownership_context, binding_stroke_roles
from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256, _sha256
from src.drawing_engine.disciplines.mep.mep_native_boundary_queries import NativeBoundaryQueries
from src.drawing_engine.disciplines.mep.mep_native_target_discovery import _intersects
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import _stable_id
from src.drawing_engine.operations.artifact_disk_usage import artifact_disk_usage


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)


def write(path, payload):
    data = (json.dumps(payload, sort_keys=True, separators=(',', ':')) + '\n').encode()
    data = gzip.compress(data, mtime=0) if path.suffix == '.gz' else data
    with artifact_disk_usage('artifact:' + path.name, [path], reserve_bytes=len(data)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def contact_scopes(targets, observations):
    """All unique complete leaders with a certified target and envelope rivals."""
    scopes = []
    for leader in targets['leader_observations']:
        obs = observations.get(leader['observation_ref'])
        paths = leader['paths']
        if not obs or not leader['search_complete'] or len(paths) != 1 or not paths[0]['search_complete']:
            continue
        path = paths[0]
        contacted = [c for c in targets['composites']['accepted_composites']
                     if c['page_ref'] == obs['page_ref'] and _contact(path['contact_point_display'], c)]
        rivals = [c for c in targets['envelope_searches'] if c['page_ref'] == obs['page_ref']
                  and c['state'] not in {'closed_geometric_proposal', 'rejected_no_native_cap_path'}
                  and _contact(path['contact_point_display'], c)]
        if len(contacted) != 1 or not rivals:
            continue
        box = obs['bbox_display']
        points = [[box[0], box[1]], [box[2], box[3]], path['contact_point_display']]
        points.extend(p for c in rivals for p in c['derived_geometry']['centreline_points_display'])
        margin = 2 * max(path['contact_search_radius'], *(c['geometry_metrics']['mean_separation_display_points'] for c in rivals))
        box = [min(p[i] for p in points) - margin for i in (0, 1)] + [max(p[i] for p in points) + margin for i in (0, 1)]
        scopes.append({'observation': obs, 'leader': leader, 'bbox_display': box,
            'competing_envelope_refs': sorted(c['id'] for c in rivals),
            'target_refs': [contacted[0]['id']], 'binding_context': binding_ownership_context(targets, obs, leader)})
    return scopes


def capture_trace_grid_contexts(args):
    """All named-axis nominations from the existing corridor blocker inventory."""
    import fitz
    if not args.registry or not args.source:
        raise ValueError('trace grid contexts require --registry and --source')
    registry = read(args.registry)
    terms = read(args.run / 'terminology-proposals.json')
    traces = read(args.run / 'trace-completion.json')
    inputs = read(args.run / 'trace-source-queries.json.gz')
    if (registry['document'] != terms['document'] or traces['document'] != terms['document']
            or _file_sha256(args.source) != terms['document']['source_pdf_sha256']):
        raise ValueError('trace grid source lineage changed')
    pages = {p['page_ref']: p for p in registry['pages']}
    queries = {q['scope_ref']: q for q in inputs['source_queries']}
    nominations = {}
    for trace in traces['scoped_traces']:
        page = pages[trace['page_ref']]
        if page['page_number'] not in args.pages:
            continue
        query = queries[trace['scope_ref']]
        if _sha256(query) != trace['source_query_sha256']:
            raise ValueError('trace grid nomination differs from frozen native evidence')
        rows = {r['source_primitive_ref']: r for r in query['source_rows']}
        for classification in trace['source_classifications']:
            if classification['classification'] != 'unresolved_incident_endpoint_or_interior_feature':
                continue
            row = rows[classification['source_primitive_ref']]
            if row['source_native_segment']['kind'] != 'line':
                continue
            for axis in page['grid_axes']:
                if axis['state'] != 'observed' or axis['method'] != 'opposing_equal_label_alignment_v1':
                    continue
                cross = 0 if axis['orientation_display'] == 'vertical' else 1
                # This is only nomination. Role acceptance independently
                # replays native bubble contacts, full cadence and competitors.
                tolerance = max(.2, row['source_native_segment']['style'].get('width') or 0)
                if any(abs(p[cross]-axis['coordinate_display']) > tolerance for p in row['points_display']):
                    continue
                key = page['page_ref'], axis['id']
                nom = nominations.setdefault(key, {'page': page, 'axis': axis, 'blockers': set()})
                nom['blockers'].add(row['source_primitive_ref'])
    captured = []
    with fitz.open(args.source) as pdf:
        for page_ref in sorted({key[0] for key in nominations}):
            page = pages[page_ref]
            index = NativeBoundaryQueries(pdf[page['page_number']-1], page_ref)
            for (ref, _), nom in sorted(nominations.items()):
                if ref != page_ref:
                    continue
                axis = nom['axis']
                cross = 0 if axis['orientation_display'] == 'vertical' else 1
                along = 1-cross
                # Label scale nominates a complete strip containing both
                # bubbles; no fixed page columns or reviewed boxes are used.
                tokens = [t for t in page['text_observations'] if t['id'] in axis['evidence_refs']]
                margin = 1.3 * max(min(t['bbox_display'][2]-t['bbox_display'][0],
                                      t['bbox_display'][3]-t['bbox_display'][1]) for t in tokens)
                # Full circular labels are wider than some narrow letters.
                margin = max(margin, 1.3*max(t['bbox_display'][3]-t['bbox_display'][1] for t in tokens))
                lo, hi = axis['opposing_label_span_display']
                coordinate = axis['coordinate_display']
                box = [0.]*4
                box[cross], box[cross+2] = coordinate-margin, coordinate+margin
                box[along], box[along+2] = lo-margin, hi+margin
                rows, complete, regions = index.query(box)
                texts = [o for o in terms['source_observations'] if o['page_ref'] == page_ref
                         and o['record_type'] == 'mep_native_text_line']
                captured.append({'id': _stable_id('mep_native_grid_context', page_ref, box), 'page_ref': page_ref,
                    'source_rows': rows, 'source_observations': [],
                    'grid_context': {'page_size_display': page['page_size_display'], 'source_observations': texts,
                        'source_pdf_sha256': registry['document']['source_pdf_sha256']},
                    'search': {'bbox_display': box, 'complete': complete, 'region_refs': regions,
                        'budget_exhausted': len(rows) > 50000, 'row_budget': 50000,
                        'source_rows_sha256': _sha256(rows),
                        'all_source_refs_sha256': _sha256(sorted(r['source_primitive_ref'] for r in rows))},
                    'nomination': {'axis_ref': axis['id'], 'axis_label': axis['label'],
                        'blocker_source_refs': sorted(nom['blockers'])}})
                print(f"page {page['page_number']} grid {axis['label']}: {len(rows)} native rows", flush=True)
            del index
    return {'schema_version': '0.1.0', 'layer': 'mep_native_trace_grid_contexts',
        'document': terms['document'], 'queries': captured,
        'source_trace_payload_sha256': _sha256(traces), 'source_registry_sha256': _sha256(registry),
        'source_implementation_sha256': _file_sha256(Path(__file__)), 'reviewed_selectors_used': False,
        'quantity_eligible': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--pages', nargs='+', type=int, required=True)
    parser.add_argument('--queries', type=Path, help='Exact frozen capture to replay without source PDF')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--trace-grid-contexts', action='store_true',
                        help='Capture all named-grid contexts nominated by the existing trace blockers')
    parser.add_argument('--registry', type=Path)
    args = parser.parse_args()
    if args.trace_grid_contexts:
        write(args.out, capture_trace_grid_contexts(args))
        return
    terms = read(args.run / 'terminology-proposals.json')
    observations = {r['id']: r for r in terms['source_observations'] if r['record_type'] == 'mep_native_text_line'}
    pages = []
    frozen = read(args.queries) if args.queries else None
    if frozen and (frozen['document'] != terms['document'] or frozen['pages_requested'] != sorted(set(args.pages))):
        raise ValueError('ownership capture belongs to a different source/page scope')
    import fitz
    pdf = None
    if not frozen:
        if not args.source or _file_sha256(args.source) != terms['document']['source_pdf_sha256']:
            raise ValueError('original PDF does not match frozen source')
        pdf = fitz.open(args.source)
    try:
        for number in sorted(set(args.pages)):
            path = args.run / f'page-{number:03d}.automatic-targets.json'
            targets = {key: read_field(path, key) for key in ('route_graph', 'composites', 'envelope_searches', 'leader_observations')}
            scopes = contact_scopes(targets, observations)
            prior = next(p for p in frozen['pages'] if p['page_number'] == number) if frozen else None
            if prior and prior['scope_candidates'] != scopes:
                raise ValueError('automatic ownership contact universe changed')
            print(f'page {number}: {len(scopes)} automatic contact scopes', flush=True)
            queries, outcomes = [], []
            index = NativeBoundaryQueries(pdf[number - 1], scopes[0]['observation']['page_ref']) if pdf and scopes else None
            for position, scope in enumerate(scopes):
                obs = scope['observation']
                if prior:
                    query = prior['queries'][position]
                    if query['search']['bbox_display'] != scope['bbox_display']:
                        raise ValueError('ownership query extent changed')
                else:
                    rows, complete, regions = index.query(scope['bbox_display'])
                    query = {'id': _stable_id('mep_native_ownership_query', obs['id'], scope['bbox_display']),
                        'page_ref': obs['page_ref'], 'source_rows': rows, 'binding_context': scope['binding_context'],
                        'source_observations': [o for o in observations.values() if o['page_ref'] == obs['page_ref']
                                                and _intersects(scope['bbox_display'], o['bbox_display'])],
                        'search': {'bbox_display': scope['bbox_display'], 'complete': complete, 'region_refs': regions,
                            'budget_exhausted': len(rows) > 20000, 'row_budget': 20000,
                            'source_rows_sha256': _sha256(rows), 'all_source_refs_sha256': _sha256(sorted(r['source_primitive_ref'] for r in rows))}}
                outcome = binding_stroke_roles(targets, obs, scope['leader'], query, observations)
                queries.append(query)
                outcomes.append(outcome)
                print(f'  {position + 1}/{len(scopes)}: {len(query["source_rows"])} rows; '
                      f'{dict(Counter((r["role"], r["state"]) for r in outcome["roles"]))}', flush=True)
            pages.append({'page_number': number, 'scope_candidates': scopes, 'queries': queries, 'outcomes': outcomes})
            del index
    finally:
        if pdf:
            pdf.close()
    payload = {'schema_version': '0.1.0', 'layer': 'mep_scoped_native_stroke_ownership',
        'document': terms['document'], 'pages_requested': sorted(set(args.pages)), 'pages': pages,
        'source_run': str(args.run.resolve()), 'source_terminology_sha256': _sha256(terms),
        'source_query_origin': {'path': str(args.queries.resolve()), 'sha256': _file_sha256(args.queries)} if args.queries else
                               {'source_pdf_sha256': terms['document']['source_pdf_sha256']},
        'implementation_sha256': {name: _file_sha256(ROOT / name) for name in
            ('src/drawing_engine/disciplines/mep/mep_automatic_target_binding.py', 'src/drawing_engine/disciplines/mep/mep_native_boundary_queries.py',
             'src/drawing_engine/disciplines/mep/mep_projected_trace_completion.py', 'src/drawing_engine/pipelines/generate_mep_stroke_ownership.py')},
        'quantity_eligible': False}
    write(args.out, payload)
    print(json.dumps({'output': str(args.out), 'page_count': len(pages),
        'contact_scope_count': sum(len(p['scope_candidates']) for p in pages)}))


if __name__ == '__main__':
    main()
