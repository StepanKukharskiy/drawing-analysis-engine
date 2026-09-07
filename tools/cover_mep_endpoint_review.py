#!/usr/bin/env python3
"""Repair frozen review-query coverage; replay geometry without changing authority."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.drawing_engine.pipelines.generate_mep_stroke_ownership import read, write
from tools.validate_mep_subpath_endpoints import checked, contains, intersects
from tools.replay_mep_native_subpaths import difference
from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256, _sha256
from src.drawing_engine.disciplines.mep.mep_native_descriptor_pack import NativeDescriptorPack
from src.drawing_engine.disciplines.mep.mep_native_path_pack import NativePathPack
from src.drawing_engine.disciplines.mep.mep_native_boundary_queries import FrozenNativeQueries
from src.drawing_engine.disciplines.mep.mep_route_body_partition import partition_corridor
from src.drawing_engine.disciplines.mep.mep_route_observations import _normalise_style
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import _stable_id
from tools.source_revision import verify_relocated_source


def union_boxes(boxes):
    return [min(b[k] for b in boxes) for k in (0, 1)] + [max(b[k] for b in boxes) for k in (2, 3)]


def plan_queries(protocol, originals):
    groups = defaultdict(list)
    for card in protocol['cards']:
        groups[card['candidate_ref']].append(card)
    plans = []
    for ref, cards in groups.items():
        old = originals[ref]
        q = checked(Path(old['query_ref']['path']), old['query_ref']['sha256'])
        missing = [c['case_id'] for c in cards if not contains(q['search']['bbox_display'], c['bbox_display'])]
        if missing:
            plans.append({'candidate_ref': ref, 'case_ids': [c['case_id'] for c in cards],
                'mismatched_case_ids': missing, 'old_query_ref': old['query_ref'],
                'bbox_display': union_boxes([q['search']['bbox_display'], *[c['bbox_display'] for c in cards]])})
    return plans


def validate_expansion(old, new):
    """A larger box must replay exactly to the old inventory, not just contain it."""
    if old['source_pdf_sha256'] != new['source_pdf_sha256'] or old['page_ref'] != new['page_ref']:
        raise ValueError('expanded query source/page differs')
    if not contains(new['search']['bbox_display'], old['search']['bbox_display']):
        raise ValueError('expanded query does not cover old extent')
    before = FrozenNativeQueries(old)
    after = FrozenNativeQueries(new)
    old_rows, old_complete, _ = before.query(old['search']['bbox_display'])
    rows, complete, _ = after.query(old['search']['bbox_display'])
    if not complete or not old_complete or rows != old_rows:
        raise ValueError('expanded inventory does not reproduce frozen old subquery')
    return len(new['source_rows']) - len(old_rows)


def proper_crossing(a, b, c, d):
    def cross(u, v): return u[0]*v[1] - u[1]*v[0]
    u = [b[i]-a[i] for i in (0, 1)]; v = [d[i]-c[i] for i in (0, 1)]
    delta = [c[i]-a[i] for i in (0, 1)]; det = cross(u, v)
    if det == 0: return None
    t, s = cross(delta, v)/det, cross(delta, u)/det
    if not (0 < t < 1 and 0 < s < 1): return None
    return [a[i]+t*u[i] for i in (0, 1)]


def boundary_witnesses(partition, query, cards, member_refs):
    """Test authored geometry, not physical caps or inferred pipe connections.

    Preserve explicit endpoint incidence and differently authored crossing paths.
    Source-first role labels remain separate from these representation checks.
    """
    sources = {r['source_primitive_ref']: r for r in query['source_rows']}
    inventory = partition['native_subpath_inventory']
    paths = inventory['subpaths']
    path_by_ref = {ref: p for p in paths for ref in p['source_primitive_refs']}
    singletons = dict(zip(inventory['singleton_source_refs'], inventory['singleton_endpoint_states']))
    transitions = {tuple(r['point_display']): r for r in inventory['endpoint_transitions']}
    join_pairs = {frozenset(j['source_refs_in_order']) for p in paths for j in p['joins']}
    incidence = defaultdict(list)
    for ref, row in sources.items():
        for end, point in enumerate((row['points_display'][0], row['points_display'][-1])):
            incidence[tuple(point)].append((ref, end))
    reports = []
    for card in cards:
        box = card['bbox_display']; ends = []; crossings = []; branches = []
        for ref in sorted(member_refs):
            row = sources[ref]; native = row['source_native_segment']
            for end, point in enumerate((row['points_display'][0], row['points_display'][-1])):
                if not contains(box, [*point, *point]): continue
                path = path_by_ref.get(ref)
                if path:
                    i = path['source_primitive_refs'].index(ref)
                    authored = path['authored_path_refs'][i] == native['drawing_ref']
                    outer = (end == 0 and i == 0) or (end == 1 and i == len(path['source_primitive_refs'])-1)
                    if outer:
                        witness = path['endpoints'][end]
                        preserved = witness['point_display'] == point
                        state = witness['state']
                    else:
                        witness = path['joins'][i-1 if end == 0 else i]
                        preserved = witness['point_display'] == point and ref in witness['source_refs_in_order']
                        state = witness['state']
                else:
                    authored = ref in singletons
                    preserved = authored; state = singletons[ref][end] if authored else 'missing'
                contacts = incidence[tuple(point)]
                transition = transitions.get(tuple(point))
                unresolved_branch = len(contacts) > 2
                branch_kept = (not unresolved_branch or transition is not None and
                    transition['state'] == 'branch_or_coincident_endpoint_alternatives' and
                    'source_refs_in_order' not in transition)
                ends.append({'source_ref': ref, 'authored_path_ref': native['drawing_ref'],
                    'endpoint_index': end, 'point_display': point, 'representation_state': state,
                    'endpoint_coordinate_and_authored_membership_preserved': preserved and authored,
                    'coincident_branch_alternatives_preserved': branch_kept,
                    'incident_source_refs': sorted({r for r, _ in contacts}),
                    'physical_termination_established': False})
                if unresolved_branch: branches.append(transition)
            if native['kind'] != 'line': continue
            a, b = row['points_display'][0], row['points_display'][-1]
            for other_ref, other in sources.items():
                other_native = other['source_native_segment']
                if (other_ref in member_refs or other_native['kind'] != 'line' or
                        other_native['drawing_ref'] == native['drawing_ref'] or
                        not intersects(other['search_bbox_display'], box)):
                    continue
                point = proper_crossing(a, b, other['points_display'][0], other['points_display'][-1])
                if point is not None and contains(box, [*point, *point]):
                    crossings.append({'source_refs': [ref, other_ref], 'point_display': point,
                        'different_authored_paths': True,
                        'not_merged_at_crossing': frozenset((ref, other_ref)) not in join_pairs,
                        'physical_connection_established': False})
        reports.append({'case_id': card['case_id'], 'authored_endpoint_witnesses': ends,
            'branch_alternative_witnesses': branches, 'independent_crossing_witnesses': crossings,
            'endpoint_witness_scope': 'native constituent endpoints within reviewed crop; not automatically the centreline analysis cut',
            'mechanism_passed': bool(ends or crossings) and all(
                e['endpoint_coordinate_and_authored_membership_preserved'] and e['coincident_branch_alternatives_preserved'] for e in ends)
                and all(c['not_merged_at_crossing'] for c in crossings),
            'physical_terminal_test': 'not asserted'})
    return reports


def protected(protocol):
    for key in ('M4', 'audit_pdf'):
        p = protocol['protected_artifacts']
        if _file_sha256(Path(p[key+'_path'])) != p[key+'_sha256']:
            raise ValueError('protected artifact changed')


def category_summary(witnesses, decisions):
    by_case = {r['case_id']: r for r in decisions}
    return {
        'authored_path_endpoint': [w['case_id'] for w in witnesses if w['authored_endpoint_witnesses']],
        'reviewed_fitting_body_contact_geometry': [w['case_id'] for w in witnesses
            if by_case[w['case_id']]['source_role'] == 'equipment_contact' and w['authored_endpoint_witnesses']],
        'nonunique_native_endpoint_boundary': [w['case_id'] for w in witnesses if w['branch_alternative_witnesses']],
        'independently_authored_crossing': [w['case_id'] for w in witnesses if w['independent_crossing_witnesses']],
        'authored_geometry_preservation_passed': bool(witnesses) and all(w['mechanism_passed'] for w in witnesses),
        'native_multi_incidence_proves_physical_branch': False,
        'typed_port_or_fitting_identity_established': False,
        'physical_terminal_categories': {
            'physical_cap': 'gap: separately qualified real-source fixture not supplied',
            'physical_termination': 'gap: separately qualified real-source fixture not supplied'},
        'physical_terminal_gaps_block_authored_geometry_test': False,
        'publication_gate_passed': False}


def verify(args):
    """Freeze the source-bound test set and replay against all compact descriptors."""
    contract = read(args.output/'protocol.json'); result = read(args.output/'results.json')
    if result['protocol_sha256'] != _file_sha256(args.output/'protocol.json'):
        raise ValueError('coverage capture protocol changed')
    previous = checked(Path(contract['previous_protocol_path']), contract['previous_protocol_sha256'])
    old_review_path = Path(contract['previous_protocol_path']).with_name('review-results.json')
    review = checked(old_review_path, contract['previous_review_sha256'])
    protected(previous)
    manifest = checked(args.denominator, contract['denominator_sha256'])
    for path, digest in contract['classifier_hashes'].items():
        verify_relocated_source(ROOT, path, digest)
    if _file_sha256(Path(manifest['source']['pdf_path'])) != contract['source_pdf_sha256']:
        raise ValueError('source PDF changed')
    pack_info = manifest['native_descriptor_pack']; pack = NativeDescriptorPack(pack_info['path'], pack_info)
    path_info = manifest['native_authored_path_pack']; path_pack = NativePathPack(path_info['path'], path_info)
    if not pack.verify_hashes() or not path_pack.verify_hash(): raise ValueError('source packs changed')
    plans = contract['plans']; expected_ordinals = [[] for _ in plans]
    for ordinal, raw in enumerate(pack.raw_records()):
        for i, plan in enumerate(plans):
            if intersects(raw[27:31], plan['bbox_display']): expected_ordinals[i].append(ordinal)
    needed = set()
    for row in result['expanded_query_replays']:
        q = checked(Path(row['query_ref']['path']), row['query_ref']['sha256'])
        needed.update(r['source_native_segment']['drawing_ref'] for r in q['source_rows'])
    context = {f"drawing[{p['drawing_ordinal']}]": {k: p[k] for k in ('drawing_ordinal', 'source_segment_count')}
               for p in path_pack.records() if f"drawing[{p['drawing_ordinal']}]" in needed}
    originals = checked(Path(previous['originals_path']), previous['originals_sha256'])
    originals = {r['candidate_ref']: r for r in originals['partitions']}
    checks = []
    for i, row in enumerate(result['expanded_query_replays']):
        if row['source_descriptor_ordinals'] != expected_ordinals[i]:
            raise ValueError('expanded query omitted or added a native descriptor')
        query = checked(Path(row['query_ref']['path']), row['query_ref']['sha256'])
        with pack.path.open('rb') as stream, pack.payload_path.open('rb') as payload:
            if len(query['source_rows']) != len(expected_ordinals[i]): raise ValueError('expanded row count changed')
            for ordinal, actual in zip(expected_ordinals[i], query['source_rows']):
                descriptor = pack.descriptor_at(ordinal, stream); expected = pack.row(descriptor, payload)
                if actual != {k: expected[k] for k in actual}: raise ValueError('expanded row differs from native pack')
        parent = originals[row['candidate_ref']]; old = checked(Path(parent['query_ref']['path']), parent['query_ref']['sha256'])
        validate_expansion(old, query)
        members = set(parent['native_member_source_refs'])
        native = next(r for r in query['source_rows'] if r['source_primitive_ref'] in members)
        local = {r['source_native_segment']['drawing_ref']: context[r['source_native_segment']['drawing_ref']] for r in query['source_rows']}
        replay = partition_corridor(parent['original_candidate'], query, members,
            _normalise_style(native['source_native_segment']['style']), authored_paths=local)
        expected = checked(Path(row['partition_path']), row['partition_sha256'])
        if replay != expected: raise ValueError('expanded partition replay changed')
        cards = [c for c in previous['cards'] if c['candidate_ref'] == row['candidate_ref']]
        witnesses = boundary_witnesses(replay, query, cards, members)
        if witnesses != [w for w in result['boundary_witnesses'] if w['case_id'] in row['case_ids']]:
            raise ValueError('native endpoint/interface witness replay changed')
        checks.append({'candidate_ref': row['candidate_ref'], 'expanded_query_sha256': row['query_ref']['sha256'],
            'source_descriptor_count': len(expected_ordinals[i]), 'native_capture_and_partition_replayed': True})
        print('verified full native capture and replay', i+1, '/', len(plans), flush=True)
    decisions = {r['case_id']: r for r in review['decisions']}
    coverage_by_case = {r['case_id']: r for r in result['coverage']}
    if set(coverage_by_case) != {c['case_id'] for c in previous['cards']}: raise ValueError('review cases lost')
    for c in result['coverage']:
        if not contains(c['native_query_bbox_display'], c['review_bbox_display']) or not c['complete_crop_coverage']:
            raise ValueError('unsupported review crop')
    tests = [{'case_id': w['case_id'], 'query_ref': coverage_by_case[w['case_id']]['query_ref'],
        'review_bbox_display': coverage_by_case[w['case_id']]['review_bbox_display'],
        'visual_source_role': decisions[w['case_id']]['source_role'], 'visual_review_note': decisions[w['case_id']]['note'],
        'visual_review_is_not_identity_authority': True, 'expectations': {
            'preserve_authored_endpoints_and_membership': True,
            'retain_nonunique_endpoint_contact_alternatives': True,
            'keep_independently_authored_crossings_disconnected': True},
        'native_witnesses': w} for w in result['boundary_witnesses']]
    dataset = {'coverage_results_sha256': _file_sha256(args.output/'results.json'),
        'source_pdf_sha256': contract['source_pdf_sha256'], 'source_page': contract['source_page'],
        'review_source_path': str(old_review_path), 'review_source_sha256': contract['previous_review_sha256'],
        'tests': tests, 'source_capture_replay_checks': checks,
        'source_review_crop_count': len(result['coverage']), 'coverage_gate_passed': result['coverage_gate_passed'],
        'apparently_continuous_intervals': result['apparently_continuous_intervals'],
        'ambiguous_intervals': result['ambiguous_intervals'],
        'category_results': category_summary(result['boundary_witnesses'], review['decisions']),
        'no_qualified_physical_branch_identity_claimed': True,
        'no_search_for_caps_performed': True, 'no_gemini_calls': True,
        'code_sha256': _file_sha256(Path(__file__)), 'new_identity_accepts': 0,
        'installed_length': None, 'purchase_length': None}
    write(args.output/'endpoint-test-set.json', dataset)
    protected(previous)
    print('source-bound endpoint test set replayed; physical terminal categories remain explicit gaps', flush=True)


def run(args):
    if args.output.exists(): raise ValueError('choose a new output directory')
    protocol = read(args.previous/'protocol.json'); review = read(args.previous/'review-results.json')
    if review['protocol_sha256'] != _file_sha256(args.previous/'protocol.json'):
        raise ValueError('review protocol changed')
    if any(_file_sha256(Path(i['path'])) != i['sha256'] for i in review['review_images']):
        raise ValueError('source review image changed')
    protected(protocol)
    frozen = checked(Path(protocol['replay_path']), protocol['replay_sha256'])
    for path, digest in frozen['code_hashes'].items():
        verify_relocated_source(ROOT, path, digest)
    originals = checked(Path(protocol['originals_path']), protocol['originals_sha256'])
    originals = {r['candidate_ref']: r for r in originals['partitions']}
    manifest = checked(args.denominator, frozen['denominator_sha256'])
    if _file_sha256(Path(manifest['source']['pdf_path'])) != protocol['source_pdf_sha256']:
        raise ValueError('source PDF changed')
    dm = manifest['native_descriptor_pack']; pack = NativeDescriptorPack(dm['path'], dm)
    pm = manifest['native_authored_path_pack']; path_pack = NativePathPack(pm['path'], pm)
    if not pack.verify_hashes() or not path_pack.verify_hash(): raise ValueError('source pack changed')
    if (dm.get('unsupported_native_item_kinds') or not manifest['acceptance_gate']['all_native_segments_accounted']
            or manifest['acceptance_gate']['unaccounted_segment_count'] != 0):
        raise ValueError('source inventory incomplete')
    plans = plan_queries(protocol, originals)
    args.output.mkdir(parents=True)
    contract = {'previous_protocol_sha256': _file_sha256(args.previous/'protocol.json'),
        'previous_review_sha256': _file_sha256(args.previous/'review-results.json'),
        'previous_protocol_path': str((args.previous/'protocol.json').resolve()),
        'source_pdf_sha256': protocol['source_pdf_sha256'], 'source_page': protocol['source_page'],
        'denominator_sha256': _file_sha256(args.denominator), 'plans': plans,
        'classifier_hashes': frozen['code_hashes'],
        'category_contract': {'representation': ['authored_path_endpoint', 'fitting_body_contact', 'branch_boundary', 'independent_crossing'],
            'physical_termination': ['qualified_real_source_cap', 'qualified_real_source_termination'],
            'physical_cap_not_required_for_authored_endpoint_test': True,
            'synthetic_examples_do_not_close_real_source_categories': True},
        'visual_continuity_is_not_publication_authority': True,
        'protected_artifacts': protocol['protected_artifacts']}
    write(args.output/'protocol.json', contract)
    selections = [[] for _ in plans]; ordinals = [[] for _ in plans]; needed = set()
    keys = ('id', 'page_ref', 'source_primitive_ref', 'source_native_segment', 'points_display',
            'bbox_display', 'search_bbox_display', 'pdf_to_display_matrix', 'state', 'quantity_eligible')
    with pack.payload_path.open('rb') as payload:
        for ordinal, raw in enumerate(pack.raw_records()):
            hits = [i for i, plan in enumerate(plans) if intersects(raw[27:31], plan['bbox_display'])]
            if not hits: continue
            descriptor = pack._decode_descriptor(raw)
            row = pack.row(descriptor, payload); row = {k: row[k] for k in keys}
            needed.add(descriptor['drawing_ordinal'])
            for i in hits: selections[i].append(row); ordinals[i].append(ordinal)
    context = {f"drawing[{p['drawing_ordinal']}]": {k: p[k] for k in ('drawing_ordinal', 'source_segment_count')}
               for p in path_pack.records() if p['drawing_ordinal'] in needed}
    indexed = {r['candidate_ref']: r for r in frozen['records']}
    expanded = {}; replays = []; witnesses = []
    for i, (plan, rows) in enumerate(zip(plans, selections)):
        ref = plan['candidate_ref']; old = checked(Path(plan['old_query_ref']['path']), plan['old_query_ref']['sha256'])
        search_ref = _stable_id('mep_full_native_boundary_query', old['page_ref'], plan['bbox_display'],
                               sorted(r['source_primitive_ref'] for r in rows))
        query = {**old, 'source_rows': rows, 'search': {'bbox_display': plan['bbox_display'], 'complete': True,
            'query_refs': [search_ref], 'source_rows_sha256': _sha256(rows),
            'all_source_refs_sha256': _sha256(sorted(r['source_primitive_ref'] for r in rows))}}
        added = validate_expansion(old, query)
        query_path = args.output/'queries'/f'{i:03d}.json.gz'; write(query_path, query)
        query_ref = {'path': str(query_path.resolve()), 'sha256': _file_sha256(query_path)}
        expanded[ref] = query_ref
        parent = originals[ref]; members = set(parent['native_member_source_refs'])
        native = next(r for r in old['source_rows'] if r['source_primitive_ref'] in members)
        style = _normalise_style(native['source_native_segment']['style'])
        def replay(q):
            local = {r['source_native_segment']['drawing_ref']: context[r['source_native_segment']['drawing_ref']] for r in q['source_rows']}
            return partition_corridor(parent['original_candidate'], q, members, style, authored_paths=local)
        before = replay(old)
        v3 = checked(Path(indexed[ref]['partition_path']), indexed[ref]['partition_sha256'])
        if before != v3: raise ValueError('old native replay no longer matches v3')
        after = replay(query)
        partition_path = args.output/'partitions'/f'{i:03d}.json.gz'; write(partition_path, after)
        before_intervals = [r['parameter_interval'] for r in before['retained_intervals']]
        after_intervals = [r['parameter_interval'] for r in after['retained_intervals']]
        cards = [c for c in protocol['cards'] if c['candidate_ref'] == ref]
        witnesses.extend(boundary_witnesses(after, query, cards, members))
        replays.append({'candidate_ref': ref, 'case_ids': plan['case_ids'], 'query_ref': query_ref,
            'partition_path': str(partition_path.resolve()), 'partition_sha256': _file_sha256(partition_path),
            'added_native_primitive_count': added, 'source_descriptor_ordinals': ordinals[i],
            'source_descriptor_ordinals_sha256': _sha256(ordinals[i]),
            'old_subquery_exactly_reproduced': True, 'old_partition_exactly_reproduced': True,
            'gained_parameter_intervals': difference(after_intervals, before_intervals),
            'lost_parameter_intervals': difference(before_intervals, after_intervals),
            'before_boundary_ports': [r['boundary_ports'] for r in before['body_intervals']],
            'after_boundary_ports': [r['boundary_ports'] for r in after['body_intervals']],
            'all_expanded_primitives_accounted': len(rows) == after['native_subpath_inventory']['accounted_source_primitive_count']})
        print('expanded/replayed', i+1, '/', len(plans), 'added primitives', added, flush=True)
    # Every case has one current, hash-bound coverage assertion. Unchanged captures
    # retain their original query refs; no stale judgment is silently promoted.
    coverage = []
    for card in protocol['cards']:
        query_ref = expanded.get(card['candidate_ref'], card['query_ref'])
        extent = next((p['bbox_display'] for p in plans if p['candidate_ref'] == card['candidate_ref']), None)
        if extent is None:
            q = checked(Path(query_ref['path']), query_ref['sha256']); extent = q['search']['bbox_display']
            complete = q['search']['complete']
        else: complete = True
        coverage.append({'case_id': card['case_id'], 'query_ref': query_ref, 'review_bbox_display': card['bbox_display'],
            'native_query_bbox_display': extent, 'complete_crop_coverage': complete and contains(extent, card['bbox_display']),
            'query_expanded_and_replayed': card['candidate_ref'] in expanded})
    delta_rows = [r for r in review['decisions'] if r['case_id'].startswith('delta-')]
    result = {'protocol_sha256': _file_sha256(args.output/'protocol.json'), 'code_sha256': _file_sha256(Path(__file__)),
        'coverage': coverage, 'expanded_query_replays': replays, 'boundary_witnesses': witnesses,
        'apparently_continuous_intervals': [r['case_id'] for r in delta_rows if r['source_role'] == 'pipe_continues'],
        'ambiguous_intervals': [r['case_id'] for r in delta_rows if r['source_role'] == 'unresolved'],
        'visual_decisions_sha256': _sha256(review['decisions']),
        'coverage_gate_passed': all(c['complete_crop_coverage'] for c in coverage),
        'physical_termination_category_gaps': ['qualified_real_source_cap', 'qualified_real_source_termination'],
        'publication_gate_passed': False, 'new_identity_accepts': 0, 'installed_length': None, 'purchase_length': None}
    write(args.output/'results.json', result)
    protected(protocol)
    print('coverage', sum(c['complete_crop_coverage'] for c in coverage), '/', len(coverage), 'publication unchanged', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('previous', 'denominator', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    verify(args) if args.verify else run(args)
