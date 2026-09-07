"""Scope-bound drawing roles must never erase route or ambiguous geometry."""

from copy import deepcopy
import gzip
import json
import math
from pathlib import Path
import unittest

import fitz

from src.drawing_engine.disciplines.mep.mep_automatic_target_binding import scoped_native_stroke_roles, _shared_route_ink
from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256, _file_sha256
from src.drawing_engine.disciplines.mep.mep_native_boundary_queries import NativeBoundaryQueries, FrozenNativeQueries


def fixture(*, open_arm=False, missing_anchor=False, broken_body=False, interior_arm=False):
    with fitz.open() as pdf:
        page = pdf.new_page(width=200, height=200)
        for y in (90, 94):
            page.draw_line((20, y), (170, y), width=.7)
        edges = [((80, 70), (85, 70)), ((85, 70), (85, 115)),
                 ((85, 115), (80, 115)), ((80, 115), (80, 70))]
        for a, b in edges[:-1] if broken_body else edges:
            page.draw_line(a, b, width=.7)
        for cy in ([73] if missing_anchor else [73, 112]):
            points = [(82.5 + .6 * math.cos(i * math.tau / 16), cy + .6 * math.sin(i * math.tau / 16)) for i in range(16)]
            for a, b in zip(points, points[1:] + points[:1]):
                page.draw_line(a, b, width=.7)
        if open_arm:
            page.draw_line((82.5, 70), (82.5, 50), width=.7)
        if interior_arm:
            page.draw_line((82.5, 88), (84, 88), width=.7)
        index = NativeBoundaryQueries(page, 'page')
        rows, complete, refs = index.query([0, 0, 200, 200])
    protected = [r['source_primitive_ref'] for r in rows if r['points_display'][0][0] == 20]
    query = {'page_ref': 'page', 'source_rows': rows, 'source_observations': [],
        'search': {'bbox_display': [0, 0, 200, 200], 'complete': complete,
            'region_refs': refs, 'source_rows_sha256': _sha256(rows),
            'all_source_refs_sha256': _sha256(sorted(r['source_primitive_ref'] for r in rows))}}
    return {'id': 'scope', 'page_ref': 'page', 'route_boundary_source_refs': protected}, query


class MepStrokeOwnershipTest(unittest.TestCase):
    def test_two_anchor_body_explains_only_its_own_ink_and_preserves_routes(self):
        scope, query = fixture()
        before = deepcopy((scope, query))
        result = scoped_native_stroke_roles(scope, query)
        self.assertEqual((scope, query), before)
        accepted = [r for r in result['roles'] if r['state'] == 'accepted']
        self.assertEqual(len(accepted), 1)
        body = accepted[0]
        self.assertEqual(len(body['native_end_motifs']), 2)
        self.assertFalse(set(scope['route_boundary_source_refs']).intersection(body['source_primitive_refs']))
        self.assertTrue(set(scope['route_boundary_source_refs']).issubset(body['independent_transverse_source_refs']))
        self.assertFalse(body['physical_support_identity_established'])
        self.assertFalse(result['attribute_applicability_established'])

    def test_open_arm_missing_anchor_and_missing_body_edge_do_not_explain_route(self):
        for change in ('open_arm', 'missing_anchor', 'broken_body', 'interior_arm'):
            with self.subTest(change=change):
                scope, query = fixture(**{change: True})
                result = scoped_native_stroke_roles(scope, query)
                self.assertFalse(any(r['state'] == 'accepted' for r in result['roles']))

    def test_dual_role_protected_wall_is_retained(self):
        scope, query = fixture()
        candidate = scoped_native_stroke_roles(scope, query)['roles'][0]
        protected = candidate['native_perimeter']['source_primitive_refs'][0]
        scope['route_boundary_source_refs'].append(protected)
        body = scoped_native_stroke_roles(scope, query)['roles'][0]
        self.assertEqual(body['state'], 'ambiguous')
        self.assertIn(protected, body['dual_role_source_refs'])

    def test_different_source_ids_do_not_hide_overlapping_route_geometry(self):
        scope, query = fixture()
        wall = query['source_rows'][0]
        duplicate = deepcopy(wall)
        duplicate['source_primitive_ref'] = 'another-native-stroke'
        duplicate['points_display'] = [[40, 90], [60, 90]]
        dual = _shared_route_ink([duplicate['source_primitive_ref']],
                                [*query['source_rows'], duplicate], scope['route_boundary_source_refs'])
        self.assertEqual(dual, [duplicate['source_primitive_ref']])
        duplicate['points_display'] = [[170, 90], [180, 90]]
        self.assertEqual(_shared_route_ink([duplicate['source_primitive_ref']],
                         [*query['source_rows'], duplicate], scope['route_boundary_source_refs']), [])

    def test_incomplete_query_native_tamper_and_wrong_page_fail_closed(self):
        scope, query = fixture()
        incomplete = deepcopy(query)
        incomplete['search']['complete'] = False
        self.assertEqual(scoped_native_stroke_roles(scope, incomplete)['roles'], [])
        changed = deepcopy(query)
        changed['source_rows'].pop()
        with self.assertRaisesRegex(ValueError, 'inventory'):
            scoped_native_stroke_roles(scope, changed)
        changed = deepcopy(query)
        changed['source_rows'][0]['points_display'][0][0] += 1
        changed['search']['source_rows_sha256'] = _sha256(changed['source_rows'])
        with self.assertRaisesRegex(ValueError, 'geometry'):
            scoped_native_stroke_roles(scope, changed)
        with self.assertRaisesRegex(ValueError, 'page'):
            scoped_native_stroke_roles(dict(scope, page_ref='other'), query)
        index = FrozenNativeQueries(query)
        self.assertFalse(index.query([-1, 0, 200, 200])[1])


class MepStrokeOwnershipReplayTest(unittest.TestCase):
    def test_named_grid_dash_ownership_requires_both_labels_complete_cadence_and_no_dual_route(self):
        root = Path(__file__).resolve().parents[1]
        capture = json.loads(gzip.decompress((root/'fixtures/mep/m_and_p_coordination/native-grid-role-queries.json.gz').read_bytes()))
        for query in capture['queries']:
            scope = {'id': 'native-grid-test', 'page_ref': query['page_ref'], 'route_boundary_source_refs': []}
            result = scoped_native_stroke_roles(scope, query)
            accepted = {ref for r in result['roles'] if r['state'] == 'accepted' for ref in r['source_primitive_refs']}
            expected = set(query['nomination']['blocker_source_refs'])
            self.assertTrue(expected.issubset(accepted))
            dual = scoped_native_stroke_roles({**scope, 'route_boundary_source_refs': sorted(expected)}, query)
            self.assertFalse(expected.intersection(ref for r in dual['roles'] if r['state'] == 'accepted' for ref in r['source_primitive_refs']))
        query = capture['queries'][0]
        scope = {'id': 'negative', 'page_ref': query['page_ref'], 'route_boundary_source_refs': []}
        missing = deepcopy(query)
        refs = set(query['nomination']['blocker_source_refs'])
        missing['source_rows'] = [r for r in missing['source_rows'] if r['source_primitive_ref'] not in refs]
        missing['search']['source_rows_sha256'] = _sha256(missing['source_rows'])
        missing['search']['all_source_refs_sha256'] = _sha256(sorted(r['source_primitive_ref'] for r in missing['source_rows']))
        self.assertFalse(scoped_native_stroke_roles(scope, missing)['roles'])
        for change in ('incomplete_query', 'missing_labels'):
            bad = deepcopy(query)
            if change == 'incomplete_query':
                bad['search']['complete'] = False
            else:
                bad['grid_context']['source_observations'] = [t for t in bad['grid_context']['source_observations']
                    if t['text'].strip() != query['nomination']['axis_label']]
            self.assertFalse(scoped_native_stroke_roles(scope, bad)['roles'])

    def test_real_return_contacts_replay_and_incomplete_body_keeps_ambiguity(self):
        path = Path(__file__).resolve().parents[1] / 'fixtures/mep/m_and_p_coordination/stroke-ownership-replay.json.gz'
        capture = json.loads(gzip.decompress(path.read_bytes()))
        self.assertEqual(capture['pages_requested'], [2, 3])
        self.assertEqual(sum(len(p['queries']) for p in capture['pages']), 4)
        for page in capture['pages']:
            for query, frozen in zip(page['queries'], page['outcomes'], strict=True):
                result = scoped_native_stroke_roles(frozen['scope'], query)
                self.assertEqual(_sha256(result), _sha256(frozen))
                body = next(r for r in result['roles'] if r['role'] == 'transverse_two_anchor_body_detail')
                self.assertEqual(body['state'], 'accepted')
                # A cropped/missing end motif cannot be rescued by a label or
                # by the previously serialized accepted role.
                changed = deepcopy(query)
                missing = set(body['native_end_motifs'][0]['source_primitive_refs'])
                changed['source_rows'] = [r for r in changed['source_rows'] if r['source_primitive_ref'] not in missing]
                changed['search']['source_rows_sha256'] = _sha256(changed['source_rows'])
                changed['search']['all_source_refs_sha256'] = _sha256(sorted(r['source_primitive_ref'] for r in changed['source_rows']))
                replay = scoped_native_stroke_roles(frozen['scope'], changed)
                self.assertFalse(any(r['role'] == body['role'] and r['state'] == 'accepted' for r in replay['roles']))


class MepStrokeOwnershipLiveTest(unittest.TestCase):
    def test_live_source_reproduces_named_grid_contexts(self):
        root = Path(__file__).resolve().parents[1]
        capture = json.loads(gzip.decompress((root / 'fixtures/mep/m_and_p_coordination/native-grid-role-queries.json.gz').read_bytes()))
        source = root / 'M&P mark-up against shop systems piping.pdf'
        self.assertEqual(_file_sha256(source), capture['document']['source_pdf_sha256'])
        registry = json.loads((root / 'fixtures/mep/m_and_p_coordination/m_and_p_coordination.sheet-registry.json').read_text())
        numbers = {p['page_ref']: p['page_number'] for p in registry['pages']}
        with fitz.open(source) as pdf:
            by_page = {}
            for query in capture['queries']:
                by_page.setdefault(numbers[query['page_ref']], []).append(query)
            for number, queries in by_page.items():
                index = NativeBoundaryQueries(pdf[number - 1], queries[0]['page_ref'])
                for query in queries:
                    rows, complete, regions = index.query(query['search']['bbox_display'])
                    self.assertTrue(complete)
                    self.assertEqual(_sha256(rows), query['search']['source_rows_sha256'])
                    self.assertEqual(regions, query['search']['region_refs'])
                del index

    def test_live_source_reproduces_complete_contact_queries(self):
        root = Path(__file__).resolve().parents[1]
        capture = json.loads(gzip.decompress((root / 'fixtures/mep/m_and_p_coordination/stroke-ownership-replay.json.gz').read_bytes()))
        source = root / 'M&P mark-up against shop systems piping.pdf'
        self.assertEqual(_file_sha256(source), capture['document']['source_pdf_sha256'])
        with fitz.open(source) as pdf:
            for page in capture['pages']:
                index = NativeBoundaryQueries(pdf[page['page_number'] - 1], page['queries'][0]['page_ref'])
                for query in page['queries']:
                    rows, complete, regions = index.query(query['search']['bbox_display'])
                    self.assertTrue(complete)
                    self.assertEqual(_sha256(rows), query['search']['source_rows_sha256'])
                    self.assertEqual(regions, query['search']['region_refs'])
                del index


if __name__ == '__main__':
    unittest.main()
