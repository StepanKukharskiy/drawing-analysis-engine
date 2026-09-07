from copy import deepcopy
import gzip
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import fitz

from src.drawing_engine.disciplines.mep.mep_automatic_target_binding import scoped_native_stroke_roles
from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256
from src.drawing_engine.disciplines.mep.mep_native_target_discovery import _native_display_geometry
from src.drawing_engine.disciplines.mep.mep_projected_trace_completion import native_dimension_role_candidates, certify_trace
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import _stable_id


def update_hashes(query):
    query['search']['source_rows_sha256'] = _sha256(query['source_rows'])
    query['search']['all_source_refs_sha256'] = _sha256(sorted(r['source_primitive_ref'] for r in query['source_rows']))
    query['search']['source_observations_sha256'] = _sha256(query['source_observations'])


class NativeDimensionRoleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'fixtures/mep/m_and_p_coordination/native-dimension-role-queries.json.gz'
        cls.queries = json.loads(gzip.decompress(path.read_bytes()))['queries']

    def scope(self, query, protected=()):
        return {'id': query['nomination']['scope_refs'][0], 'page_ref': query['page_ref'],
                'route_boundary_source_refs': list(protected)}

    def altered_line(self, source, ref, points):
        row = deepcopy(source)
        native = row['source_native_segment']
        native.update(id=ref, primitive_ref=ref, drawing_ref=ref,
            start_display=points[0], end_display=points[1], length_points=math.dist(*points),
            bbox_display=[min(p[i] for p in points) for i in (0, 1)] + [max(p[i] for p in points) for i in (0, 1)])
        rendered, bounds, search = _native_display_geometry(native, fitz.Matrix(row['pdf_to_display_matrix']))
        row.update(id=_stable_id('mep_native_target_primitive', row['page_ref'], ref), source_primitive_ref=ref,
                   points_display=rendered, bbox_display=bounds, search_bbox_display=search)
        return row

    def test_real_dimensions_preserve_native_provenance_without_exclusive_or_quantity_authority(self):
        for query in self.queries:
            before = deepcopy(query)
            records = native_dimension_role_candidates(self.scope(query), query)
            self.assertEqual(len(records), 2)
            self.assertEqual(sorted(len(r['source_primitive_refs']) for r in records), [16, 18])
            self.assertTrue(all(r['role_geometry_established'] and not r['reasons'] for r in records))
            self.assertTrue(all(not r['exclusive_role'] and not r['quantity_eligible'] for r in records))
            self.assertEqual(query, before)

    def test_missing_opposite_tick_and_extra_tick_arm_do_not_form_complete_dimensions(self):
        query = deepcopy(self.queries[0])
        original = native_dimension_role_candidates(self.scope(query), query)[0]
        target = original['baseline']['id']
        missing = original['terminal_source_refs_by_endpoint'][0][0]
        query['source_rows'] = [r for r in query['source_rows'] if r['source_primitive_ref'] != missing]
        update_hashes(query)
        self.assertFalse(any(r['baseline']['id'] == target for r in native_dimension_role_candidates(self.scope(query), query)))
        query = deepcopy(self.queries[0])
        source = next(r for r in query['source_rows'] if r['source_primitive_ref'] == missing)
        point = original['dimension_points_display'][0]
        query['source_rows'].append(self.altered_line(source, 'hostile.extra_tick_arm', [point, [point[0] - 3, point[1] + 4]]))
        update_hashes(query)
        self.assertFalse(any(r['baseline']['id'] == target for r in native_dimension_role_candidates(self.scope(query), query)))

    def test_connector_midpoint_branch_and_competing_native_text_remain_unresolved(self):
        query = deepcopy(self.queries[0])
        original = native_dimension_role_candidates(self.scope(query), query)[0]
        target = original['baseline']['id']
        ref = original['text_connectors'][0]['source_primitive_refs'][0]
        source = next(r for r in query['source_rows'] if r['source_primitive_ref'] == ref)
        center = [sum(p[i] for p in source['points_display']) / 2 for i in (0, 1)]
        query['source_rows'].append(self.altered_line(source, 'hostile.connector_midpoint_branch',
                                                    [center, [center[0] + 10, center[1]]]))
        update_hashes(query)
        self.assertFalse(any(r['baseline']['id'] == target for r in native_dimension_role_candidates(self.scope(query), query)))
        query = deepcopy(self.queries[0])
        text = deepcopy(next(t for t in query['source_observations'] if t['id'] == original['source_observation_refs'][0]))
        text.update(id='hostile.competing_native_text', text='EXHAUST')
        query['source_observations'].append(text)
        update_hashes(query)
        record = next(r for r in native_dimension_role_candidates(self.scope(query), query) if r['baseline']['id'] == target)
        self.assertFalse(record['role_geometry_established'])
        self.assertIn('competing_native_dimension_text_paths', record['reasons'])

    def test_shared_route_ink_keeps_dimension_ownership_ambiguous(self):
        query = self.queries[0]
        candidate = native_dimension_role_candidates(self.scope(query), query)[0]
        refs = candidate['baseline']['source_primitive_refs']
        result = scoped_native_stroke_roles(self.scope(query, refs), query)
        role = next(r for r in result['roles'] if set(refs).issubset(r['source_primitive_refs']))
        self.assertEqual(role['state'], 'ambiguous')
        self.assertTrue(role['dual_role_source_refs'])

    def test_incomplete_source_search_and_missing_text_do_not_supply_roles(self):
        query = deepcopy(self.queries[0])
        query['search']['complete'] = False
        self.assertEqual(native_dimension_role_candidates(self.scope(query), query), [])
        query = deepcopy(self.queries[0])
        query['source_observations'] = []
        update_hashes(query)
        with patch('src.drawing_engine.disciplines.mep.mep_projected_trace_completion.straight_supports', side_effect=AssertionError('unnecessary geometry work')):
            self.assertEqual(native_dimension_role_candidates(self.scope(query), query), [])


class NativeDimensionTraceConsumerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = Path(__file__).resolve().parents[1] / 'fixtures/mep/m_and_p_coordination'
        cls.roles = json.loads(gzip.decompress((folder / 'native-dimension-role-queries.json.gz').read_bytes()))['queries'][0]
        fixture = json.loads(gzip.decompress((folder / 'projected-trace-completion-replay.json.gz').read_bytes()))
        ref = cls.roles['nomination']['scope_refs'][0]
        cls.original = next(r for r in fixture['expected_completion']['scoped_traces'] if r['scope_ref'] == ref)
        cls.scope = cls.original['scope']
        cls.query = next(q for q in fixture['source_queries'] if q['scope_ref'] == ref)
        cls.protected = {source for r in fixture['expected_completion']['scoped_traces']
            if r['page_ref'] == cls.scope['page_ref'] for source in r['scope']['input_composite']['member_source_primitive_refs']}
        cls.document_hash = fixture['document']['source_pdf_sha256']

    def replay(self, query, protected=None):
        return certify_trace(self.scope, query, protected_route_source_refs=self.protected if protected is None else protected,
                             source_pdf_sha256=self.document_hash)

    def test_real_native_dimension_roles_close_the_long_trace_without_changing_source_geometry(self):
        query = deepcopy(self.query)
        query['stroke_role_queries'] = [deepcopy(self.roles)]
        before = deepcopy(query)
        result = self.replay(query)
        self.assertFalse(self.original['projected_scope_complete'])
        self.assertTrue(result['projected_scope_complete'])
        self.assertEqual(result['coverage']['source_classification_counts']['scope_certified_native_stroke_role'], 3)
        self.assertEqual(result['native_sidewall_coverage'], self.original['native_sidewall_coverage'])
        self.assertEqual(result['source_primitive_refs'], self.original['source_primitive_refs'])
        self.assertFalse(result['physical_continuation_established'])
        self.assertFalse(result['quantity_eligible'])
        self.assertEqual(result['epistemic_state'], 'inferred')
        self.assertEqual(query, before)

    def test_forged_roles_missing_native_tick_and_dual_use_cannot_close_trace(self):
        candidate = native_dimension_role_candidates({'id': self.scope['id'], 'page_ref': self.scope['page_ref']}, self.roles)[0]
        query = deepcopy(self.query)
        role_query = deepcopy(self.roles)
        missing = candidate['terminal_source_refs_by_endpoint'][0][0]
        role_query['source_rows'] = [r for r in role_query['source_rows'] if r['source_primitive_ref'] != missing]
        update_hashes(role_query)
        role_query['roles'] = [{'state': 'accepted', 'source_primitive_refs': self.original['source_primitive_refs']}]
        query['stroke_role_queries'] = [role_query]
        self.assertFalse(self.replay(query)['projected_scope_complete'])
        query['stroke_role_queries'] = [deepcopy(self.roles)]
        protected = self.protected | set(candidate['baseline']['source_primitive_refs'])
        self.assertFalse(self.replay(query, protected)['projected_scope_complete'])

    def test_role_context_cannot_change_page_document_or_native_corridor_rows(self):
        for kind in ('page', 'document', 'geometry'):
            query = deepcopy(self.query)
            role_query = deepcopy(self.roles)
            if kind == 'page':
                role_query['page_ref'] = 'another_page'
            elif kind == 'document':
                role_query['source_observations'][0]['source_pdf_sha256'] = 'another_document'
            else:
                ref = self.scope['input_composite']['member_source_primitive_refs'][0]
                row = next(r for r in role_query['source_rows'] if r['source_primitive_ref'] == ref)
                row['points_display'][0][0] += 1
            query['stroke_role_queries'] = [role_query]
            with self.assertRaises(ValueError, msg=kind):
                self.replay(query)


if __name__ == '__main__':
    unittest.main()
