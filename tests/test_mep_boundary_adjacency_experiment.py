from copy import deepcopy
import gzip
import json
import math
import os
from pathlib import Path
import random
import unittest
from unittest.mock import patch

from tools.mep_boundary_adjacency import (
    backend, indexed_point_hits, load_native, precomputed_styles, rust_point_hits,
)
import src.drawing_engine.disciplines.mep.mep_native_bend_connections as bend
from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256
from src.drawing_engine.core.vector_topology import point_distance_to_segment

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'fixtures/mep/m_and_p_coordination/native-boundary-replay.json.gz'


def available_backends():
    yield 'indexed', None
    path = os.environ.get('MEP_ADJACENCY_LIBRARY')
    if path:
        yield 'rust', load_native(path)


def source(name, a, b):
    return {'source_primitive_ref': name, 'points_display': [a, b],
            'source_native_segment': {'kind': 'line', 'style':
                {'stroke': [0, 0, 0], 'width': .7, 'dash': '[] 0'}}}


def simple_trace():
    style = {'width_display_points': .7, 'dash_pattern': '[] 0'}
    ports = [{'center': [x, 1], 'sides': [[x, 0], [x, 2]],
              'outward': direction, 'width': 2, 'style': style}
             for x, direction in ((0, [1, 0]), (10, [-1, 0]))]
    rows = [source('bottom', [0, 0], [10, 0]), source('top', [0, 2], [10, 2])]
    return ports, rows


class BoundaryAdjacencyExperimentTest(unittest.TestCase):
    def assert_hits_equal(self, points, edges, tolerance=.001):
        expected = [[i for i, v in enumerate(points)
                     if point_distance_to_segment(v, p, q) <= tolerance] for p, q in edges]
        self.assertEqual(expected, indexed_point_hits(points, edges, tolerance))
        for name, native in available_backends():
            if name == 'rust':
                self.assertEqual(expected, rust_point_hits(native, points, edges, tolerance))

    def test_threshold_edges_degenerate_reversed_duplicates_and_crossings(self):
        t = .001
        below, above = math.nextafter(t, 0), math.nextafter(t, math.inf)
        points = [(0., 0.), (0., 0.), (5., t), (5., below), (5., above),
                  (-t, 0.), (-above, 0.), (10+t, 0.), (5., 0.), (1e9, 1e9)]
        edges = [((0., 0.), (10., 0.)), ((10., 0.), (0., 0.)),
                 ((5., -5.), (5., 5.)), ((0., 0.), (0., 0.)),
                 ((0., 0.), (1e-7, 0.)), ((1e9, 1e9), (1e9+10, 1e9))]
        self.assert_hits_equal(points, edges)
        self.assert_hits_equal([], [])
        self.assert_hits_equal(points, edges, 0.)

    def test_seeded_rotated_and_translated_threshold_batches(self):
        rng = random.Random(92371)
        for offset in (0., -10000., 1e9):
            edges, points = [], []
            for _ in range(40):
                a = tuple(offset + rng.uniform(-20, 20) for _ in (0, 1))
                b = tuple(offset + rng.uniform(-20, 20) for _ in (0, 1))
                edges.append((a, b))
                dx, dy = b[0]-a[0], b[1]-a[1]
                norm = math.hypot(dx, dy)
                for t in (-.001, 0., .1, .5, 1., 1.001):
                    for distance in (.001, math.nextafter(.001, 0), math.nextafter(.001, math.inf)):
                        points.append((a[0]+t*dx-distance*dy/norm, a[1]+t*dy+distance*dx/norm))
            self.assert_hits_equal(points, edges)

    def test_trace_provenance_negatives_style_mutation_and_budget(self):
        ports, rows = simple_trace()
        duplicate = source('duplicate-bottom', [10, 0], [0, 0])
        curve = {'source_primitive_ref': 'curve', 'bbox_display': [4, -.1, 6, .1],
                 'source_native_segment': {'kind': 'cubic', 'style': rows[0]['source_native_segment']['style']}}
        cases = [rows, [*rows, duplicate], [*rows, source('branch', [5, 0], [5, -2])],
                 [*rows, curve], rows[:1],
                 [source('a', [0, 0], [5, 0]), source('b', [5.01, 0], [10, 0]), rows[1]],
                 [*rows, source('crossing', [5, -2], [5, 4]), source('vertex', [0, 0], [5, 0])]]
        with backend('exhaustive'):
            expected = [bend.trace_bend_boundaries(ports, case) for case in cases]
        self.assertEqual(expected[1][0][0]['source_primitive_refs'], ['bottom', 'duplicate-bottom'])
        self.assertEqual(expected[3][1], ['unsupported_curve_at_native_boundary'])
        self.assertEqual(expected[-1][1], ['native_boundary_branch_or_missing_trace'])
        for name, native in available_backends():
            before = _sha256([ports, cases])
            with backend(name, native):
                for case, result in zip(cases, expected):
                    self.assertEqual(result, bend.trace_bend_boundaries(ports, case))
                    self.assertEqual(result, bend.trace_bend_boundaries(ports, list(reversed(case))))
                    self.assertEqual(result, bend.trace_bend_boundaries(ports, iter(case)))
                self.assertEqual(([], ['native_boundary_graph_budget_exceeded']),
                                 bend.trace_bend_boundaries(ports, rows, max_edges=1))
            self.assertEqual(before, _sha256([ports, cases]))
            changed = deepcopy(rows)
            with backend(name, native):
                self.assertEqual([], bend.trace_bend_boundaries(ports, changed)[1])
                changed[0]['source_native_segment']['style']['width'] = 3
                self.assertTrue(bend.trace_bend_boundaries(ports, changed)[1])

    def test_style_precomputation_exact_and_no_cross_call_cache(self):
        ports, rows = simple_trace()
        for width in (None, .7, .85, math.nextafter(.85, math.inf), 3):
            for dash in ('[] 0', '[2 2] 0', None):
                rows[0]['source_native_segment']['style'].update(width=width, dash=dash)
                expected = list(bend._source_style_compatibility_exhaustive(ports[0]['style'], rows))
                self.assertEqual(expected, list(precomputed_styles(ports[0]['style'], rows)))
        rows[0]['source_native_segment']['style'].update(width=.7, dash=[2, 2])
        rows[1]['source_native_segment']['style'].update(width=.7, dash=(2, 2))
        ports[0]['style']['dash_pattern'] = [2, 2]
        self.assertEqual([True, False], [compatible for _, compatible in precomputed_styles(ports[0]['style'], rows)])

    def test_nested_reference_and_exception_restore_production_helpers(self):
        original_points, original_styles = bend._points_on_segments, bend._source_style_compatibility
        try:
            with backend('indexed'):
                with backend('exhaustive'):
                    self.assertIsNot(precomputed_styles, bend._source_style_compatibility)
                self.assertIs(precomputed_styles, bend._source_style_compatibility)
                raise RuntimeError('restore')
        except RuntimeError:
            pass
        self.assertIs(original_points, bend._points_on_segments)
        self.assertIs(original_styles, bend._source_style_compatibility)

    def test_ambiguous_coordinate_rounding_remains_an_abstention(self):
        ports, rows = simple_trace()
        rows.extend([source('a', [5, 0], [6, .5]), source('b', [5, .0018], [6, 1]),
                     source('c', [5.0001, .0009], [7, 1.5])])
        expected = ([], ['ambiguous_native_coordinate_rounding'])
        for name, native in [('exhaustive', None), *available_backends()]:
            with backend(name, native):
                self.assertEqual(expected, bend.trace_bend_boundaries(ports, rows))

    def test_explicit_rust_required_and_nonfinite_inputs_rejected(self):
        with self.assertRaises(ValueError), backend('rust'):
            pass
        for v in (math.nan, math.inf, 1e13):
            with self.assertRaises(ValueError):
                indexed_point_hits([(v, 0)], [], .001)
            for name, native in available_backends():
                if name == 'rust':
                    with self.assertRaises(ValueError):
                        native.point_hits([(v, 0)], [], .001, 1e-10)


class BoundaryAdjacencyProductionTest(unittest.TestCase):
    def test_index_does_not_hide_malformed_points_outside_its_bounds(self):
        for function in (bend._points_on_segments_exhaustive, bend._points_on_segments):
            with self.assertRaises(ValueError):
                list(function([(100., 100., 0.)], [((0., 0.), (10., 0.))]))

    def test_direct_production_uses_index_without_experimental_dispatch(self):
        ports, rows = simple_trace()
        with patch.object(bend, '_points_on_segments_exhaustive', side_effect=AssertionError('unexpected fallback')):
            self.assertEqual([], bend.trace_bend_boundaries(ports, rows)[1])
        self.assertEqual(bend.__name__, bend._points_on_segments.__module__)
        self.assertEqual(bend.__name__, bend._source_style_compatibility.__module__)

    def test_outside_index_domain_uses_exact_predicate_without_new_rejection(self):
        for offset in (1e13, math.inf, math.nan):
            points = [(0., 0.), (offset, 0.)]
            edges = [((0., 0.), (10., 0.))]
            expected = [[i for i, p in enumerate(points)
                         if point_distance_to_segment(p, *edge) <= bend.TOLERANCE] for edge in edges]
            stats = {}
            self.assertEqual(expected, bend._indexed_point_hits(points, edges, stats=stats))
            self.assertTrue(stats['exhaustive_fallback'])
            self.assertEqual(list(bend._points_on_segments_exhaustive(points, edges)),
                             list(bend._points_on_segments(points, edges)))

    def test_sparse_index_prunes_only_candidates_not_duplicates(self):
        points = [(i*10., 0.) for i in range(200)] + [(0., 0.)]
        edges = [((i*10., -1.), (i*10., 1.)) for i in range(200)]
        expected = [[i for i, p in enumerate(points) if point_distance_to_segment(p, *edge) <= bend.TOLERANCE]
                    for edge in edges]
        stats = {}
        self.assertEqual(expected, bend._indexed_point_hits(points, edges, stats=stats))
        self.assertEqual([0, 200], expected[0])
        self.assertLess(stats['distance_tests'], len(points)*len(edges)/100)


class BoundaryAdjacencyReplayTest(unittest.TestCase):
    def test_additional_frozen_package_pages_preserve_every_bend_outcome(self):
        root = ROOT / 'output/mep-package-native-verified-2026-08-31'
        count = 0
        for number in (4, 12, 16, 18):
            frozen = json.loads((root/f'page-{number:03d}.automatic-targets.json').read_bytes())
            rows = [r for r in frozen['boundary_connections'] if r['relation_type'] == 'projected_native_bend']
            before = _sha256(rows)
            with backend('exhaustive'):
                expected = [bend.trace_bend_boundaries(r['ports'], r['source_rows']) for r in rows]
            actual = [bend.trace_bend_boundaries(r['ports'], r['source_rows']) for r in rows]
            self.assertEqual(expected, actual, f'page {number}')
            self.assertEqual(before, _sha256(rows))
            for row, result in zip(rows, actual, strict=True):
                if row['state'] == 'accepted':
                    self.assertEqual((row['boundary_paths'], []), result)
            count += len(rows)
            del frozen, rows, expected, actual
        self.assertEqual(198, count)

    def test_current_stroke_ownership_queries_have_exact_raw_batch_coverage(self):
        fixture_root = ROOT / 'fixtures/mep/m_and_p_coordination'
        for name in ('stroke-ownership-replay.json.gz', 'native-grid-role-queries.json.gz'):
            capture = json.loads(gzip.decompress((fixture_root/name).read_bytes()))
            queries = capture.get('queries', [q for page in capture.get('pages', []) for q in page['queries']])
            for query in queries:
                before = _sha256(query)
                edges = [(tuple(r['points_display'][0]), tuple(r['points_display'][-1]))
                         for r in query['source_rows'] if r['source_native_segment']['kind'] == 'line']
                points = sorted({p for edge in edges for p in edge})
                expected = [sorted(hits) for hits in bend._points_on_segments_exhaustive(points, edges)]
                actual = [sorted(hits) for hits in bend._points_on_segments(points, edges)]
                self.assertEqual(expected, actual, query.get('id'))
                self.assertEqual(before, _sha256(query))

    def test_stale_bend_evidence_and_incomplete_queries_fail_closed(self):
        from src.drawing_engine.disciplines.mep.mep_boundary_connection_replay import replay_boundary_connections
        frozen = json.loads(gzip.decompress(FIXTURE.read_bytes()))
        row = next(r for r in frozen['connections'] if r['state'] == 'accepted'
                   and r['relation_type'] == 'projected_native_bend')
        # Synthetic exterior tails reproduce the frozen port planes. This is a
        # replay-negative harness, not new physical or native source evidence.
        fragments, composites = [], []
        for port in row['ports']:
            tail = lambda p: [p[a] - 1000*port['outward'][a] for a in (0, 1)]
            centers = [port['center'], tail(port['center'])]
            if port['end'] == 1:
                centers.reverse()
            refs = []
            for side in port['sides']:
                ref = f'fragment.{len(fragments)}'
                refs.append(ref)
                fragments.append({'id': ref, 'style': port['style'],
                                  'geometry': {'points_display': [side, tail(side)]}})
            composites.append({'id': port['composite_ref'], 'page_ref': row['page_ref'],
                'member_fragment_refs': refs, 'derived_geometry': {'centreline_points_display': centers},
                'geometry_metrics': {'mean_separation_display_points': port['width']}})
        graph = {'pages': [{'fragments': fragments}]}
        bindings = {'outlined_route_composites': composites, 'm3_5_contract_ref': {'payload_sha256': _sha256(composites)}}
        payload = {'connections': [row], 'm3_payload_sha256': _sha256(graph),
                   'm35_payload_sha256': _sha256(composites)}
        for name, native in [('exhaustive', None), *available_backends()]:
            with self.subTest(backend=name), backend(name, native):
                self.assertEqual([row], replay_boundary_connections(payload, graph, bindings))
                for mutation in ('source_membership', 'source_geometry', 'path_geometry', 'incomplete_query', 'upstream'):
                    bad = deepcopy(payload)
                    record = bad['connections'][0]
                    if mutation == 'source_membership':
                        record['source_rows'].pop()
                    elif mutation == 'source_geometry':
                        for source_row in record['source_rows']:
                            if source_row['source_primitive_ref'] in record['source_primitive_refs']:
                                source_row['points_display'] = [[-9999, -9999], [-9998, -9998]]
                    elif mutation == 'path_geometry':
                        record['boundary_paths'][0]['points_display'][0][0] += 1
                    elif mutation == 'incomplete_query':
                        record['search']['complete'] = False
                    else:
                        bad['m3_payload_sha256'] = 'stale'
                    with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                        replay_boundary_connections(bad, graph, bindings)

    def test_complete_frozen_bend_candidates_and_batches(self):
        frozen = json.loads(gzip.decompress(FIXTURE.read_bytes()))
        rows = [r for r in frozen['connections'] if r['relation_type'] == 'projected_native_bend']
        before = _sha256(frozen)
        batches = []
        with backend('exhaustive', batches=batches):
            expected = [bend.trace_bend_boundaries(r['ports'], r['source_rows']) for r in rows]
        self.assertEqual(len(rows), 61)
        self.assertEqual(sum(not reasons for _, reasons in expected), 7)
        for name, native in available_backends():
            with self.subTest(backend=name), backend(name, native):
                actual = [bend.trace_bend_boundaries(r['ports'], r['source_rows']) for r in rows]
            self.assertEqual(expected, actual)
            self.assertEqual(_sha256(expected), _sha256(actual))
            for points, edges in batches:
                reference = [[i for i, v in enumerate(points)
                              if point_distance_to_segment(v, p, q) <= bend.TOLERANCE] for p, q in edges]
                hits = (indexed_point_hits(points, edges, bend.TOLERANCE) if name == 'indexed'
                        else rust_point_hits(native, points, edges, bend.TOLERANCE))
                self.assertEqual(reference, hits)
        self.assertEqual(before, _sha256(frozen))
