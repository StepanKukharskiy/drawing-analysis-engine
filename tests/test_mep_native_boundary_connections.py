from copy import deepcopy
from pathlib import Path
import json
import gzip
import math
import tempfile
import unittest

import fitz

from test_mep_automatic_target_binding import automatic_fixture
from src.drawing_engine.disciplines.mep.mep_automatic_target_binding import RegionIndex, build_automatic_targets, automatic_binding_evidence, apply_geometric_text_applicability
from src.drawing_engine.disciplines.mep.mep_native_boundary_connections import boundary_coverage, propagate_collinear_applicability
from src.drawing_engine.disciplines.mep.mep_native_target_discovery import iter_bounded_native_page_regions
from src.drawing_engine.disciplines.mep.mep_attribute_binding import build_mep_attribute_bindings, elevation_applicability_outcomes
from src.drawing_engine.disciplines.mep.mep_native_bend_connections import trace_bend_boundaries
from src.drawing_engine.disciplines.mep.mep_boundary_connection_replay import replay_boundary_connections
from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256


class MepNativeBoundaryConnectionsTest(unittest.TestCase):
    def test_system_only_extension_keeps_size_and_elevation_local(self):
        with tempfile.TemporaryDirectory() as directory:
            targets, terminology, prior = self._connected(directory)
            evidence = [r for r in prior['binding_evidence']
                        if r['method']['name'] == 'complete_native_dot_leader_contact']
            direct = build_mep_attribute_bindings(terminology_proposals=terminology,
                route_graph=targets['route_graph'], outlined_route_composites=targets['composites'],
                binding_evidence=evidence)
            result, enriched = propagate_collinear_applicability(terminology=terminology,
                binding_evidence=evidence, connections=targets['boundary_connections'],
                direct_bindings=direct, proposal_types=('system',))
            proposals = {r['id']: r for r in result['proposals']}
            additions = [r for r in enriched if r['id'] not in {e['id'] for e in evidence}]
            self.assertTrue(additions)
            self.assertEqual({'system'}, {proposals[r['proposal_ref']]['proposal_type'] for r in additions})
            with self.assertRaises(ValueError):
                propagate_collinear_applicability(terminology=terminology, binding_evidence=evidence,
                    connections=[], direct_bindings=direct, proposal_types=('elevation',))

    def _connected(self, directory, missing=False, gap=.2, with_registry=False, label_text='2" HHWS'):
        registry, terminology, *_ = automatic_fixture(directory, label_text=label_text)
        source = Path(directory) / 'neutral.pdf'
        with fitz.open(source) as pdf:
            page = pdf[0]
            for y in (140, 144):
                page.draw_line((270 + gap, y), (370, y), width=.7)
                if not missing or y == 140:
                    page.draw_line((269, y), (272, y), width=.7)
            page.draw_line((370, 140), (370, 144), width=.7)
            packets = list(iter_bounded_native_page_regions(page, 'page.1',
                region_size_display_points=64, max_candidates_per_region=4000))
        with RegionIndex(primitives=[r for p in packets for r in p['primitive_candidates']],
                regions=[p['region'] for p in packets], page_size=[400, 300]) as index:
            targets = build_automatic_targets(sheet_registry=registry, page_indexes={'page.1': index},
                observations=terminology['source_observations'])
        evidence, _ = automatic_binding_evidence(targets=targets, terminology=terminology)
        terminology, evidence = apply_geometric_text_applicability(terminology=terminology, binding_evidence=evidence)
        before = deepcopy((terminology, evidence, targets))
        direct_bindings = build_mep_attribute_bindings(terminology_proposals=terminology, route_graph=targets['route_graph'],
            outlined_route_composites=targets['composites'], binding_evidence=evidence)
        enriched, bindings = propagate_collinear_applicability(terminology=terminology,
            binding_evidence=evidence, connections=targets['boundary_connections'], direct_bindings=direct_bindings)
        self.assertEqual(before, (terminology, evidence, targets))
        m4 = build_mep_attribute_bindings(terminology_proposals=enriched, route_graph=targets['route_graph'],
            outlined_route_composites=targets['composites'], binding_evidence=bindings)
        return (targets, enriched, m4, registry) if with_registry else (targets, enriched, m4)

    def test_both_native_sidewalls_required_before_annotation_scope_extends(self):
        for missing in (False, True):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                targets, terminology, m4 = self._connected(directory, missing=missing)
                joins = [r for r in targets['boundary_connections'] if r['state'] == 'accepted']
                self.assertEqual(len(joins), 0 if missing else 1)
                self.assertEqual(m4['summary']['accepted_relation_count'], 2 if missing else 4)
                if joins:
                    self.assertTrue(all(p['source_primitive_refs'] for p in joins[0]['boundary_paths']))
                    self.assertFalse(joins[0]['physical_continuation_established'])
                    self.assertFalse(joins[0]['quantity_eligible'])

    def test_elevation_extent_accepts_only_certified_straight_annotation_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            targets, terminology, m4 = self._connected(directory, label_text='BE=25\'-0"')
            boundary = {'m3_payload_sha256': _sha256(targets['route_graph']),
                'm35_payload_sha256': _sha256(targets['composites']), 'connections': targets['boundary_connections']}
            before = deepcopy(m4)
            result = elevation_applicability_outcomes(terminology=terminology, bindings=m4,
                graph=targets['route_graph'], boundary_connections=boundary)
            row = result['outcomes'][0]
            self.assertEqual(row['literal_text'], 'BE=25\'-0"')
            self.assertEqual(row['candidate']['basis'], 'bottom')
            self.assertEqual(len(row['direct_target_refs']), 1)
            self.assertEqual(len(row['candidate_extents']), 2)
            extension = next(t for t in row['candidate_extents']
                             if t['state'] == 'accepted_straight_annotation_extent')
            self.assertEqual(row['extension_accepted_count'], 1)
            self.assertTrue(row['constant_elevation_scope_established'])
            self.assertEqual(len(extension['straight_inline_extent_certificates']), 1)
            certificate = extension['straight_inline_extent_certificates'][0]
            self.assertEqual(certificate['method'], 'complete_native_straight_inline_elevation_extent')
            self.assertTrue(certificate['unique_two_port_join'])
            self.assertTrue(certificate['same_width_collinear_members'])
            self.assertFalse(certificate['bend_crossed'])
            self.assertFalse(certificate['branch_crossed'])
            self.assertFalse(extension['elevation_transfer_established'])
            self.assertFalse(row['physical_continuation_established'])
            self.assertEqual(m4, before)
            bad = deepcopy(terminology);bad['document']['source_pdf_sha256'] = 'changed'
            with self.assertRaises(ValueError):
                elevation_applicability_outcomes(terminology=bad, bindings=m4,
                    graph=targets['route_graph'], boundary_connections=boundary)

    def test_observed_attribute_change_stops_propagation_without_rewriting_direct_facts(self):
        with tempfile.TemporaryDirectory() as directory:
            targets, terminology, m4 = self._connected(directory)
            # Replay the direct channel only, then add the other endpoint's
            # independently observed system boundary to the read-only input.
            evidence = [r for r in m4['binding_evidence'] if r['method']['name'] == 'complete_native_dot_leader_contact']
            direct = deepcopy(m4)
            direct['relations'] = [r for r in m4['relations'] if set(r['binding_evidence_refs']).intersection(r['id'] for r in evidence)]
            original = next(r for r in direct['relations'] if r['relation_type'] == 'route_system')
            other = next(ref for r in targets['boundary_connections'] for ref in r['composite_refs'] if ref not in original['target_refs'])
            boundary = deepcopy(original)
            boundary.update(id='independent_system_boundary', target_refs=[other], binding_evidence_refs=[])
            boundary['candidate'] = {'kind': 'chilled_water_return'}
            direct['relations'].append(boundary)
            result, enriched = propagate_collinear_applicability(terminology=terminology,
                binding_evidence=evidence, connections=targets['boundary_connections'], direct_bindings=direct)
            by_id = {r['id']: r for r in result['proposals']}
            self.assertFalse(any(r['target_refs'] == [other] and by_id[r['proposal_ref']]['proposal_type'] == 'system'
                                 for r in enriched))
            self.assertEqual(original['candidate']['kind'], 'heating_hot_water_supply')

    def test_native_gap_is_not_repaired_by_display_tolerance(self):
        style = {'width_display_points': .7, 'dash_pattern': '[] 0'}
        def source(name, a, b):
            return {'source_primitive_ref': name, 'points_display': [a, b],
                'source_native_segment': {'kind': 'line', 'style': {'width': .7,
                    'stroke': [0, 0, 0], 'dash': '[] 0'}}}
        rows = [source('a', [0, 0], [4, 0]), source('b', [4.01, 0], [10, 0])]
        self.assertIsNone(boundary_coverage([0, 0], [10, 0], rows, style))
        rows[1]['points_display'][0][0] = 4
        self.assertEqual(boundary_coverage([0, 0], [10, 0], rows, style), ['a', 'b'])
        rows[1]['points_display'][1][1] = .1
        self.assertIsNone(boundary_coverage([0, 0], [10, 0], rows, style))

    def test_branch_geometry_never_supplies_implicit_attribute_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            targets, terminology, m4 = self._connected(directory)
            direct_evidence = [r for r in m4['binding_evidence']
                               if r['method']['name'] == 'complete_native_dot_leader_contact']
            branch = deepcopy(targets['boundary_connections'][0])
            branch['relation_type'] = 'projected_native_branch'
            for third in ([], ['independent_third_port_target']):
                connection = deepcopy(branch)
                connection['composite_refs'].extend(third)
                _, output = propagate_collinear_applicability(terminology=terminology,
                    binding_evidence=direct_evidence, connections=[connection], direct_bindings=m4)
                self.assertEqual(output, direct_evidence)
    def test_replay_checks_both_geometry_and_complete_query_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            targets, _, m4 = self._connected(directory)
            payload = {'m3_payload_sha256': _sha256(targets['route_graph']),
                'm35_payload_sha256': _sha256(targets['composites']),
                'connections': targets['boundary_connections']}
            self.assertEqual(replay_boundary_connections(payload, targets['route_graph'], m4), payload['connections'])
            bad = deepcopy(payload)
            bad['connections'][0]['source_rows'] = bad['connections'][0]['source_rows'][:-1]
            with self.assertRaisesRegex(ValueError, 'complete native query'):
                replay_boundary_connections(bad, targets['route_graph'], m4)
            bad = deepcopy(payload)
            bad['connections'][0]['boundary_paths'][0]['points_display'][0][1] += 1
            with self.assertRaisesRegex(ValueError, 'does not replay'):
                replay_boundary_connections(bad, targets['route_graph'], m4)

    def test_frozen_real_bends_replay_all_accepts_and_preserve_branch_negatives(self):
        path = Path(__file__).resolve().parents[1] / 'fixtures/mep/m_and_p_coordination/native-boundary-replay.json.gz'
        frozen = json.loads(gzip.decompress(path.read_bytes()))
        accepted = [r for r in frozen['connections'] if r['state'] == 'accepted' and r['relation_type'] == 'projected_native_bend']
        self.assertEqual(len(accepted), 7)
        for row in accepted:
            with self.subTest(connection=row['id']):
                paths, reasons = trace_bend_boundaries(row['ports'], row['source_rows'])
                self.assertEqual(reasons, [])
                self.assertEqual(paths, row['boundary_paths'])
                self.assertFalse(row['physical_continuation_established'])
                self.assertFalse(row['quantity_eligible'])
        row = accepted[0]
        a, b = max(((a, b) for path in row['boundary_paths']
                    for a, b in zip(path['points_display'], path['points_display'][1:])),
                   key=lambda pair: math.dist(*pair))
        middle = [(x + y) / 2 for x, y in zip(a, b)]
        radius = math.dist(a, b) / 10
        curve = {'source_primitive_ref': 'mid_edge_curve',
            'bbox_display': [middle[0]-radius, middle[1]-radius, middle[0]+radius, middle[1]+radius],
            'source_native_segment': {'kind': 'cubic', 'style': {'stroke': [0, 0, 0],
                'width': row['ports'][0]['style']['width_display_points'],
                'dash': row['ports'][0]['style']['dash_pattern']}}}
        paths, reasons = trace_bend_boundaries(row['ports'], [*row['source_rows'], curve])
        self.assertEqual(paths, [])
        self.assertIn('unsupported_curve_at_native_boundary', reasons)
        negatives = [r for r in frozen['connections'] if 'native_boundary_branch_or_missing_trace' in r['reasons']]
        self.assertGreater(len(negatives), 0)
        for row in negatives:
            paths, reasons = trace_bend_boundaries(row['ports'], row['source_rows'])
            self.assertEqual(paths, [])
            self.assertTrue(reasons)

    def test_live_native_boundary_queries_match_frozen_source(self):
        from src.drawing_engine.disciplines.mep.mep_native_boundary_queries import NativeBoundaryQueries
        from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256
        root = Path(__file__).resolve().parents[1]
        fixture = root / 'fixtures/mep/m_and_p_coordination'
        frozen = json.loads(gzip.decompress((fixture / 'native-boundary-replay.json.gz').read_bytes()))
        registry = json.loads((fixture / 'm_and_p_coordination.sheet-registry.json').read_text())
        source = root / 'M&P mark-up against shop systems piping.pdf'
        self.assertEqual(_file_sha256(source), frozen['document']['source_pdf_sha256'])
        with fitz.open(source) as pdf:
            for scope in registry['pages']:
                records = [r for r in frozen['connections'] if r['page_ref'] == scope['page_ref']]
                if not records:
                    continue
                index = NativeBoundaryQueries(pdf[scope['page_number'] - 1], scope['page_ref'])
                for record in records:
                    sources, complete, refs = index.query(record['search']['bbox_display'])
                    # Native PyMuPDF color tuples become JSON arrays. Compare
                    # the canonical serialized contract without rounding data.
                    self.assertEqual(_sha256(sources), _sha256(record['source_rows']))
                    self.assertEqual(complete, record['search']['complete'])
                    self.assertEqual(refs, record['search']['region_refs'])
                del index
