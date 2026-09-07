"""Fresh negative-only native cap certificates for legacy envelope witnesses.

The old witness is a selection input, never a proof that a cap is absent. Every
negative is rebuilt from complete native endpoint queries, with the existing
M3 endpoint clustering and M3.5 length-bounded cap search. Found, incomplete and
budget-limited searches retain the original ambiguity. No item is accepted here.
"""

from collections import Counter
from copy import deepcopy
import gzip
import json
import math
from pathlib import Path

from src.drawing_engine.disciplines.mep.mep_automatic_target_binding import (
    RegionIndex, _contact, _fragment, _point_box, _reachable_cap_fragments,
)
from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256, _sha256
from src.drawing_engine.disciplines.mep.mep_native_target_discovery import _intersects, _native_display_geometry
from src.drawing_engine.disciplines.mep.mep_outlined_route_composites import _parallel_metrics, _shortest_cap_path, _style_compatible
from src.drawing_engine.disciplines.mep.mep_route_observations import _native_candidates, _cluster_endpoints, _fragment_records
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import _stable_id
import fitz


VERSION = '0.1.0'
LAYER = 'mep_native_cap_rejections'
REACHABLE_SOURCE_BUDGET = 1200
ROOT = Path(__file__).resolve().parents[4]
IMPLEMENTATION_PATHS = (
    'src/drawing_engine/disciplines/mep/mep_native_cap_replay.py', 'tools/generate_mep_native_cap_replay.py',
    'src/drawing_engine/disciplines/mep/mep_automatic_target_binding.py', 'src/drawing_engine/disciplines/mep/mep_native_boundary_queries.py',
    'src/drawing_engine/disciplines/mep/mep_native_target_discovery.py', 'src/drawing_engine/core/vector_topology.py',
    'src/drawing_engine/disciplines/mep/mep_route_observations.py', 'src/drawing_engine/disciplines/mep/mep_outlined_route_composites.py',
    'tools/inspect_mep_binding_scope_delta.py',
)


def implementation_manifest():
    return {name: _file_sha256(ROOT / name) for name in IMPLEMENTATION_PATHS}


def select_native_cap_witnesses(terminology, unresolved_proposals, page_leaders, page_witnesses):
    """Select all unresolved contacts behind otherwise unique complete leaders."""
    observations = {r['id']: r for r in terminology['source_observations']}
    leaders = {r['observation_ref']: r for r in page_leaders}
    if len(leaders) != len(page_leaders) or len({r['id'] for r in page_witnesses}) != len(page_witnesses):
        raise ValueError('duplicate native cap selection inputs')
    blocked = {ref for row in unresolved_proposals
        if row.get('reason') == 'unresolved_competing_envelope_at_contact' and len(row.get('target_refs', [])) == 1
        for ref in row.get('evidence_refs', []) if ref in observations and ref in leaders}
    selected = {}
    for ref in sorted(blocked):
        leader = leaders[ref]
        paths = leader.get('paths', [])
        if not leader.get('search_complete') or len(paths) != 1 or not paths[0].get('search_complete'):
            continue
        for witness in page_witnesses:
            if (witness['page_ref'] != observations[ref]['page_ref'] or witness['state'] != 'unresolved'
                    or not _contact(paths[0]['contact_point_display'], witness)):
                continue
            item = selected.setdefault(witness['id'], {'witness_ref': witness['id'],
                'input_witness_sha256': _sha256(witness), 'source_observation_refs': [],
                'leader_payload_sha256': {}})
            item['source_observation_refs'].append(ref)
            item['leader_payload_sha256'][ref] = _sha256(leader)
    return [selected[ref] for ref in sorted(selected)]


def cap_query_plan(witness, source_rows):
    """Derive endpoint query boxes from the actual two native member strokes."""
    refs = witness['member_source_primitive_refs']
    if len(refs) != 2 or len(set(refs)) != 2:
        raise ValueError('cap witness requires two distinct native members')
    by_ref = {r['source_primitive_ref']: r for r in source_rows}
    if any(ref not in by_ref for ref in refs):
        raise ValueError('cap replay is missing native member geometry')
    left, right = [by_ref[ref] for ref in sorted(refs)]
    for row in (left, right):
        if row['page_ref'] != witness['page_ref'] or row['source_native_segment']['kind'] != 'line':
            raise ValueError('cap members must be same-page native lines')
    a, b = _fragment(left), _fragment(right)
    metrics = _parallel_metrics(a, b)
    mean, width = metrics['mean_separation_display_points'], metrics['member_width_display_points']
    length = min(metrics['left_length_display_points'], metrics['right_length_display_points'])
    if not (_style_compatible(a['style'], b['style']) and length >= max(10 * width, 5 * mean)
            and metrics['length_ratio'] >= .98
            and max(.1, .15 * width) < mean <= max(12 * width, .08 * length)
            and metrics['maximum_separation_deviation_display_points'] <= max(.35, .08 * mean)
            and metrics['maximum_tangent_difference_degrees'] <= 2
            and all(abs(value - mean) <= max(.35, .08 * mean)
                    for value in metrics['endpoint_separations_display_points'])):
        raise ValueError('native cap member geometry does not reproduce an eligible envelope')
    centreline = [[(x + y) / 2 for x, y in zip(p, q)]
                  for p, q in zip(metrics['left_samples'], metrics['right_samples'])]
    prior = witness['derived_geometry']['centreline_points_display']
    if (not math.isclose(mean, witness['geometry_metrics']['mean_separation_display_points'], abs_tol=1e-6)
            or len(centreline) != len(prior)
            or not any(all(math.dist(a, b) <= 1e-5 for a, b in zip(centreline, order))
                       for order in (prior, list(reversed(prior))))):
        raise ValueError('native cap geometry differs from the upstream witness')
    limit = max(2.5 * mean, 4 * width, 1)
    return left, right, metrics, limit, [_point_box(p, limit + .1) for p in left['points_display']]


def _validated_sources(native_page):
    by_id, primitive_ids = {}, set()
    for row in native_page['source_rows']:
        native = row['source_native_segment']
        ref = native['id']
        if (row['id'] in by_id or ref in primitive_ids or row['source_primitive_ref'] != ref
                or row['page_ref'] != native_page['page_ref']
                or row['id'] != _stable_id('mep_native_target_primitive', row['page_ref'], ref)):
            raise ValueError('conflicting native cap source IDs or page provenance')
        points, bounds, search = _native_display_geometry(native, fitz.Matrix(row['pdf_to_display_matrix']))
        if (points != row['points_display'] or bounds != row['bbox_display']
                or search != row['search_bbox_display'] or row.get('quantity_eligible') is not False):
            raise ValueError('native cap source display geometry does not replay')
        by_id[row['id']] = row
        primitive_ids.add(ref)
    return by_id


def build_native_cap_page(*, native_page, registry, upstream_witnesses, selection):
    """Recompute every selected outcome, including retained unresolved cases."""
    page_ref = native_page['page_ref']
    scopes = {p['page_ref']: p for p in registry['pages']}
    if (native_page['document'] != registry['document'] or native_page['m1_payload_sha256'] != _sha256(registry)
            or page_ref not in scopes or native_page['page_number'] != scopes[page_ref]['page_number']):
        raise ValueError('native cap page differs from frozen M1/source')
    scope = scopes[page_ref]
    witnesses = {r['id']: r for r in upstream_witnesses if r['page_ref'] == page_ref}
    selections = {r['witness_ref']: r for r in selection}
    cases = {r['witness_ref']: r for r in native_page['cases']}
    if len(cases) != len(native_page['cases']) or set(cases) != set(selections):
        raise ValueError('native cap capture does not cover the complete automatic selection')
    sources = _validated_sources(native_page)
    queries = native_page['queries']
    query_rows = {}
    for key, query in queries.items():
        rows = [sources[ref] for ref in query['source_row_refs']]
        if (key != _sha256(query['bbox_display']) or len({r['id'] for r in rows}) != len(rows)
                or _sha256(rows) != query['source_rows_sha256']
                or any(not _intersects(query['bbox_display'], r['search_bbox_display']) for r in rows)
                or type(query['complete']) is not bool):
            raise ValueError('native cap query does not reproduce its immutable sources')
        expected_ref = _stable_id('mep_full_native_boundary_query', page_ref, query['bbox_display'],
                                 sorted(r['source_primitive_ref'] for r in rows))
        if query['region_refs'] != [expected_ref]:
            raise ValueError('native cap query provenance differs from full native query')
        query_rows[key] = rows
    outcomes = []
    for ref in sorted(cases):
        case, selected = cases[ref], selections[ref]
        witness = witnesses.get(ref)
        if (witness is None or witness['state'] != 'unresolved' or _sha256(witness) != selected['input_witness_sha256']
                or case['selection'] != selected or case['input_witness'] != witness):
            raise ValueError('native cap replay input witness or automatic selection changed')
        members = [sources[r] for r in case['member_row_refs']]
        left, right, metrics, limit, boxes = cap_query_plan(witness, members)
        keys = [_sha256(box) for box in boxes]
        if case['query_keys'] != keys or any(key not in queries for key in keys):
            raise ValueError('native cap capture omits or changes an endpoint query')
        local = {r['source_primitive_ref']: r for r in members}
        for key in keys:
            local.update((r['source_primitive_ref'], r) for r in query_rows[key]
                         if r['source_native_segment']['length_points'] <= limit + .1)
        complete = all(queries[key]['complete'] for key in keys)
        proof = {'maximum_path_display_points': limit, 'endpoint_tolerance_display_points': .05,
            'input_source_count': len(local), 'input_source_refs_sha256': _sha256(sorted(local)),
            'reachable_source_budget': REACHABLE_SOURCE_BUDGET, 'reachable_source_count': None,
            'reachable_source_refs_sha256': None, 'endpoint_graph_sha256': None,
            'native_query_complete': complete, 'full_source_endpoint_clustering_preserved': False,
            'all_admissible_cap_paths_preserved': False, 'cap_result': 'incomplete_query',
            'closure_source_primitive_refs': []}
        state = 'unresolved'
        if complete:
            # RegionIndex.native preserves canonical primitives while applying
            # the recorded PDF-to-display transform; no source is clipped.
            class LocalRows:
                primitives = local
            natives = RegionIndex.native(LocalRows(), local)
            candidates = _native_candidates(scope, {'segments': natives})
            _, assignments = _cluster_endpoints(page_ref, candidates, .05)
            fragments, _ = _fragment_records(scope, None, candidates, assignments)
            by_ref = {r['source_primitive_ref']: r for r in fragments}
            reachable = _reachable_cap_fragments(fragments, by_ref[left['source_primitive_ref']],
                                                  by_ref[right['source_primitive_ref']], limit)
            proof.update({'reachable_source_count': len(reachable),
                'reachable_source_refs_sha256': _sha256(sorted(r['source_primitive_ref'] for r in reachable)),
                'endpoint_graph_sha256': _sha256(fragments),
                'full_source_endpoint_clustering_preserved': True,
                'all_admissible_cap_paths_preserved': True, 'cap_result': 'reachable_source_budget_exceeded'})
            if len(reachable) <= REACHABLE_SOURCE_BUDGET:
                closure, closure_refs = _shortest_cap_path({'fragments': reachable},
                    by_ref[left['source_primitive_ref']], by_ref[right['source_primitive_ref']], metrics)
                proof['cap_result'] = 'found_path' if closure else 'no_path'
                proof['closure_source_primitive_refs'] = sorted(r['source_primitive_ref'] for r in reachable
                                                               if r['id'] in closure_refs)
                if not closure:
                    state = 'rejected_no_native_cap_path'
        outcomes.append({'id': _stable_id('mep_native_cap_rejection', page_ref, ref, _sha256(witness),
                                          _sha256([queries[k] for k in keys]), proof),
            'record_type': 'mep_native_cap_replay_outcome', 'page_ref': page_ref, 'witness_ref': ref,
            'input_witness_sha256': _sha256(witness), 'state': state, 'evidence_state': 'derived',
            'source_observation_refs': selected['source_observation_refs'],
            'member_source_primitive_refs': witness['member_source_primitive_refs'],
            'source_query_refs': sorted({r for k in keys for r in queries[k]['region_refs']}),
            'query_payload_sha256': {k: _sha256(queries[k]) for k in sorted(set(keys))},
            'cap_proof': proof, 'quantity_eligible': False, 'physical_continuation_established': False,
            'physical_item_identity_established': False, 'positive_acceptance_authority': False})
    return {'schema_version': VERSION, 'layer': LAYER, 'document': deepcopy(registry['document']),
        'page_ref': page_ref, 'page_number': scope['page_number'], 'm1_payload_sha256': _sha256(registry),
        'native_page_payload_sha256': _sha256(native_page), 'selection_payload_sha256': _sha256(selection),
        'outcomes': outcomes, 'summary': dict(Counter(r['state'] for r in outcomes)),
        'authority': {'negative_cap_witness_only': True, 'positive_acceptance_authority': False,
                      'quantity_eligible': False}}


def replay_native_cap_rejections(payload, *, native_page, registry, upstream_witnesses=None, selection=None):
    # Portable replay uses the embedded original witnesses and selection inputs.
    # Import against a live upstream additionally supplies its exact originals.
    if upstream_witnesses is None:
        upstream_witnesses = [r['input_witness'] for r in native_page['cases']]
    if selection is None:
        inputs = native_page['selection_inputs']
        selection = select_native_cap_witnesses(inputs['terminology'], inputs['unresolved_proposals'],
            inputs['leader_observations'], upstream_witnesses)
    rebuilt = build_native_cap_page(native_page=native_page, registry=registry,
                                    upstream_witnesses=upstream_witnesses, selection=selection)
    if _sha256(rebuilt) != _sha256(payload):
        raise ValueError('native cap rejection payload does not independently replay')
    return {r['witness_ref']: r for r in rebuilt['outcomes'] if r['state'] == 'rejected_no_native_cap_path'}


def _artifact_path(directory, record):
    path = Path(directory) / record['path']
    if path.resolve().parent != Path(directory).resolve():
        raise ValueError('native cap artifact path escapes its frozen directory')
    if _file_sha256(path) != record['file_sha256']:
        raise ValueError('native cap artifact file hash differs')
    return path


def load_native_cap_manifest(directory, *, registry, terminology, upstream_context):
    manifest = json.loads((Path(directory) / 'native-cap-replay-manifest.json').read_text())
    if (manifest['document'] != registry['document'] or terminology['document'] != registry['document']
            or manifest['m1_payload_sha256'] != _sha256(registry)
            or manifest['terminology_payload_sha256'] != _sha256(terminology)
            or manifest['upstream_context_payload_sha256'] != _sha256(upstream_context)
            or manifest['implementation_sha256'] != implementation_manifest()
            or manifest['layer'] != 'mep_native_cap_replay_manifest' or manifest['schema_version'] != VERSION
            or len({p['page_ref'] for p in manifest['pages']}) != len(manifest['pages'])):
        raise ValueError('native cap manifest source, policy or frozen inputs differ')
    return manifest


def replay_native_cap_page(directory, manifest, *, page_ref, registry, terminology,
                           unresolved_proposals, page_leaders, upstream_witnesses):
    selection = select_native_cap_witnesses(terminology, unresolved_proposals, page_leaders, upstream_witnesses)
    entry = next((p for p in manifest['pages'] if p['page_ref'] == page_ref), None)
    if entry is None:
        if selection:
            raise ValueError('native cap manifest omits a selected page')
        return {}
    native = json.loads(gzip.decompress(_artifact_path(directory, entry['native_archive']).read_bytes()))
    payload = json.loads(_artifact_path(directory, entry['outcomes']).read_text())
    if (_sha256(native) != entry['native_archive']['payload_sha256']
            or _sha256(payload) != entry['outcomes']['payload_sha256']
            or native['page_ref'] != page_ref or native['implementation_sha256'] != manifest['implementation_sha256']):
        raise ValueError('native cap replay artifact payload differs')
    return replay_native_cap_rejections(payload, native_page=native, registry=registry,
                                       upstream_witnesses=upstream_witnesses, selection=selection)
