from copy import deepcopy
import gzip
import json
import math
from pathlib import Path
import unittest

from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256
from src.drawing_engine.disciplines.mep.mep_fitting_hypotheses import fitting_hypotheses_for_query, drafting_gap_for_port, build_fitting_hypotheses, replay_fitting_hypotheses
from src.drawing_engine.disciplines.mep.mep_native_boundary_connections import _ports


def fixture():
    style = {'width_display_points': .7, 'dash_pattern': '[] 0'}
    native_style = {'stroke': [0, 0, 0], 'width': .7, 'dash': '[] 0'}
    fragments, comps = [], []
    for name, sides, far in [('left', [[-2, -1], [-2, 1]], [-30, 0]),
                            ('right', [[2, -1], [2, 1]], [30, 0]),
                            ('down', [[-1, 3], [1, 3]], [0, 30])]:
        center = [(a + b) / 2 for a, b in zip(*sides)]
        members = []
        for side, point in enumerate(sides):
            ref = name + str(side)
            members.append(ref)
            fragments.append({'id': ref, 'style': style,
                'geometry': {'points_display': [point, [point[i] + far[i] - center[i] for i in (0, 1)]]}})
        comps.append({'id': name, 'page_ref': 'page', 'member_fragment_refs': members,
            'geometry_metrics': {'mean_separation_display_points': 2},
            'derived_geometry': {'centreline_points_display': [center, far],
                                 'projected_path_display_points': math.dist(center, far)}})
    graph = {'document': {'source_pdf_sha256': 'test-only'}, 'pages': [{'page_ref': 'page', 'fragments': fragments}]}
    composites = {'accepted_composites': comps}
    ports = [p for p in _ports(comps, {f['id']: f for f in fragments}) if p['end'] == 0]
    pairs = [([-2, -1], [2, -1]), ([-2, 1], [2, 1]), ([-2, -1.2], [-2, 1.2]),
             ([2, -1.2], [2, 1.2]), ([-1, 1], [-1, 3]), ([1, 1], [1, 3]), ([-1, 3], [1, 3])]
    rows = [{'source_primitive_ref': f'line{i}', 'points_display': [a, b],
             'source_native_segment': {'kind': 'line', 'style': native_style}} for i, (a, b) in enumerate(pairs)]
    for quadrant in range(4):
        rows.append({'source_primitive_ref': f'curve{quadrant}',
            'points_display': [[math.cos((quadrant + i / 16) * math.pi / 2),
                                math.sin((quadrant + i / 16) * math.pi / 2)] for i in range(17)],
            'source_native_segment': {'kind': 'cubic', 'style': native_style}})
    query = {'id': 'query', 'page_ref': 'page', 'ports': ports, 'source_rows': rows,
             'search': {'complete': True, 'bbox_display': [-10, -10, 10, 10],
                        'all_source_refs_sha256': _sha256(sorted(r['source_primitive_ref'] for r in rows))},
             'through_stroke_source_primitive_refs': ['line0', 'line1']}
    return query, composites, graph


def rehash(query):
    query['search']['all_source_refs_sha256'] = _sha256(sorted(r['source_primitive_ref'] for r in query['source_rows']))


class MepFittingHypothesesTest(unittest.TestCase):
    def test_body_collars_and_routes_establish_only_inferred_projected_connection(self):
        query, comps, graph = fixture()
        before = deepcopy((query, comps, graph))
        payload = build_fitting_hypotheses(source_queries=[query], composites=comps, graph=graph)
        self.assertEqual(payload['accepted_projected_connection_count'], 1)
        row = payload['fitting_hypotheses'][0]
        self.assertEqual(row['epistemic_state'], 'inferred')
        self.assertEqual(len(row['input_fragments']), 6)
        self.assertEqual(row['retained_through_stroke_refs'], ['line0', 'line1'])
        self.assertFalse(row['physical_continuation_established'])
        self.assertFalse(row['quantity_eligible'])
        self.assertEqual((query, comps, graph), before)

    def test_closed_ring_does_not_turn_crossing_or_continuing_elbow_into_tee(self):
        query, _, _ = fixture()
        for replace in ('crossing', 'continuing_elbow'):
            bad = deepcopy(query)
            bad['source_rows'] = [r for r in bad['source_rows'] if r['source_primitive_ref'] not in ('line4', 'line5')]
            if replace == 'crossing':
                for i, x in enumerate((-1, 1)):
                    line = deepcopy(query['source_rows'][4 + i])
                    line['points_display'] = [[x, -5], [x, 3]]
                    bad['source_rows'].append(line)
            else:
                bad['source_rows'].append(query['source_rows'][4])
                line = deepcopy(query['source_rows'][5])
                line['points_display'] = [[1, -5], [1, 3]]
                bad['source_rows'].append(line)
            rehash(bad)
            self.assertFalse(fitting_hypotheses_for_query(bad)['accepted_projected_connection'], replace)
        query['source_rows'] = [r for r in query['source_rows'] if r['source_primitive_ref'] != 'curve0']
        rehash(query)
        self.assertFalse(fitting_hypotheses_for_query(query)['accepted_projected_connection'])

    def test_paired_gap_needs_caps_ink_precision_and_matching_tangents(self):
        query, _, _ = fixture()
        port = deepcopy(query['ports'][2])
        port.update(center=[0, 0], sides=[[-1, 0], [1, 0]], outward=[0, 1])
        pairs = [([-1, 0], [1, 0]), ([-1, .12], [1, .12]), ([-1, .12], [-1, 4]), ([1, .12], [1, 4])]
        rows = [dict(source_primitive_ref=str(i), points_display=[a, b],
            source_native_segment=deepcopy(query['source_rows'][0]['source_native_segment'])) for i, (a, b) in enumerate(pairs)]
        precision = {'state': 'observed_grid', 'quantum_display_points': .12}
        gap = drafting_gap_for_port(port, rows, precision)
        self.assertEqual(gap['observed_gaps_display_points'], [.12, .12])
        self.assertFalse(gap['native_strokes_modified'])
        self.assertFalse(gap['body_applicability_established'])
        self.assertIsNone(drafting_gap_for_port(port, rows[1:], precision))
        self.assertIsNone(drafting_gap_for_port(port, rows, {'state': 'unknown'}))
        bad = deepcopy(rows)
        bad[-1]['points_display'][0][1] = .24
        self.assertIsNone(drafting_gap_for_port(port, bad, precision))
        overlap = deepcopy(rows[-1])
        overlap.update(source_primitive_ref='overlapping_native_rail', points_display=[[1, 0], [1, 4]])
        self.assertIsNone(drafting_gap_for_port(port, [*rows, overlap], precision))
        for row in rows:
            row['source_native_segment']['style']['width'] = .1
        port['style']['width_display_points'] = .1
        self.assertIsNone(drafting_gap_for_port(port, rows, precision))

    def test_full_replay_rejects_false_accepts_changed_sources_and_handle_sized_arms(self):
        query, comps, graph = fixture()
        payload = build_fitting_hypotheses(source_queries=[query], composites=comps, graph=graph)
        self.assertEqual(replay_fitting_hypotheses(payload, source_queries=[query], composites=comps, graph=graph), payload)
        altered = deepcopy(payload)
        altered['fitting_hypotheses'][0]['physical_continuation_established'] = True
        with self.assertRaisesRegex(ValueError, 'differs from full source replay'):
            replay_fitting_hypotheses(altered, source_queries=[query], composites=comps, graph=graph)
        bad = deepcopy(query)
        bad['ports'][0]['center'][0] += .1
        with self.assertRaisesRegex(ValueError, 'ports differ'):
            build_fitting_hypotheses(source_queries=[bad], composites=comps, graph=graph)
        changed = deepcopy(query)
        changed['source_rows'].pop()
        self.assertFalse(fitting_hypotheses_for_query(changed)['accepted_projected_connection'])
        comps['accepted_composites'][2]['derived_geometry']['projected_path_display_points'] = 4
        self.assertEqual(build_fitting_hypotheses(source_queries=[query], composites=comps, graph=graph)['accepted_projected_connection_count'], 0)

    def test_real_frozen_positive_gap_crossing_and_continuing_geometry(self):
        path = Path(__file__).resolve().parents[1] / 'fixtures/mep/m_and_p_coordination/fitting-body-replay.json.gz'
        frozen = json.loads(gzip.decompress(path.read_bytes()))
        results = {query['id']: fitting_hypotheses_for_query(query) for query in frozen['source_queries']}
        self.assertEqual(list(results.values()), frozen['expected_local_hypotheses'])
        positive = results['mep_native_branch_connection.22ee5adf53f453c3401f']
        self.assertTrue(positive['accepted_projected_connection'])
        self.assertEqual(positive['epistemic_state'], 'inferred')
        self.assertEqual(len(positive['body_candidates'][0]['port_interfaces']), 3)
        gap_row = results['mep_native_branch_connection.0ff5140a169d7883ae27']
        self.assertFalse(gap_row['accepted_projected_connection'])
        self.assertEqual(len(gap_row['drafting_gap_hypotheses']), 1)
        self.assertTrue(all(.119 < d < .121 for d in gap_row['drafting_gap_hypotheses'][0]['observed_gaps_display_points']))
        for suffix in ('424de08251bee6b19353', 'ebf89748ca9190cbafce'):
            ref = 'mep_native_branch_connection.' + suffix
            row = results[ref]
            self.assertFalse(row['accepted_projected_connection'])
            self.assertTrue(next(q for q in frozen['source_queries'] if q['id'] == ref)['through_stroke_source_primitive_refs'])

    def test_real_heldout_ring_does_not_force_a_missing_body_certificate(self):
        path = Path(__file__).resolve().parents[1] / 'fixtures/mep/m_and_p_coordination/fitting-heldout-replay.json.gz'
        frozen = json.loads(gzip.decompress(path.read_bytes()))
        rows = [fitting_hypotheses_for_query(q) for q in frozen['source_queries']]
        self.assertEqual(rows, frozen['expected_local_hypotheses'])
        self.assertTrue(all(q['search']['complete'] for q in frozen['source_queries']))
        self.assertEqual(len(rows[0]['native_ring_candidates']), 1)
        self.assertEqual(rows[0]['body_candidates'], [])
        self.assertFalse(rows[0]['accepted_projected_connection'])


if __name__ == '__main__':
    unittest.main()
