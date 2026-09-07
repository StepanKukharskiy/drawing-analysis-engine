from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from src.drawing_engine.disciplines.mep.mep_native_branch_connections import trace_branch_boundaries, port_attachment_observations
from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256


def tee_fixture():
    style = {'width_display_points': .7, 'dash_pattern': '[] 0'}
    ports = [dict(id='left', center=[-10, 0], sides=[[-10, -1], [-10, 1]], outward=[1, 0], style=style),
             dict(id='right', center=[10, 0], sides=[[10, -1], [10, 1]], outward=[-1, 0], style=style),
             dict(id='up', center=[0, -10], sides=[[-1, -10], [1, -10]], outward=[0, 1], style=style)]
    pairs = [([-10, 1], [10, 1]), ([-10, -1], [-1, -1]), ([-1, -1], [-1, -10]),
             ([1, -10], [1, -1]), ([1, -1], [10, -1])]
    rows = [dict(source_primitive_ref=str(i), points_display=[a, b],
                 source_native_segment={'kind': 'line', 'style': {'stroke': [0, 0, 0], 'width': .7, 'dash': '[] 0'}})
            for i, (a, b) in enumerate(pairs)]
    return ports, rows


class MepNativeBranchConnectionsTest(unittest.TestCase):
    def test_streamed_artifact_publish_preserves_existing_result_on_failure(self):
        from tools.discover_mep_native_branches import read_payload, write_payload
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'discovery.json.gz'
            write_payload(path, {'context': 'complete'}, iter([{'id': 'existing'}]))
            before = path.read_bytes()

            def interrupted_rows():
                yield {'id': 'partial'}
                raise ValueError('incomplete source query')

            with self.assertRaisesRegex(ValueError, 'incomplete source query'):
                write_payload(path, {'context': 'new'}, interrupted_rows())
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(read_payload(path), {'context': 'complete', 'connections': [{'id': 'existing'}]})

    def test_three_port_tee_requires_one_native_perimeter(self):
        ports, rows = tee_fixture()
        before = deepcopy((ports, rows))
        paths, reasons = trace_branch_boundaries(ports, rows)
        self.assertEqual(reasons, [])
        self.assertEqual(len(paths), 3)
        self.assertEqual({ref for path in paths for ref in path['source_primitive_refs']}, {'0', '1', '2', '3', '4'})
        self.assertEqual((ports, rows), before)
        self.assertEqual(trace_branch_boundaries(ports, rows[:-1])[0], [])

    def test_crossing_does_not_create_a_branch(self):
        ports, rows = tee_fixture()
        ports.append(dict(id='down', center=[0, 10], sides=[[-1, 10], [1, 10]], outward=[0, -1], style=ports[0]['style']))
        pairs = [([-10, -1], [10, -1]), ([-10, 1], [10, 1]),
                 ([-1, -10], [-1, 10]), ([1, -10], [1, 10])]
        rows = [dict(source_primitive_ref=str(i), points_display=[a, b],
                     source_native_segment=deepcopy(rows[0]['source_native_segment'])) for i, (a, b) in enumerate(pairs)]
        paths, reasons = trace_branch_boundaries(ports, rows)
        self.assertEqual(paths, [])
        self.assertIn('disconnected_crossing_perimeters', reasons)

    def test_continuing_geometry_is_retained_as_a_competing_destination(self):
        ports, rows = tee_fixture()
        row = deepcopy(rows[1])
        row.update(source_primitive_ref='continuing_native_stroke', points_display=[[-1, -1], [10, -1]])
        rows.append(row)
        diagnostics = {}
        paths, reasons = trace_branch_boundaries(ports, rows, diagnostics=diagnostics)
        self.assertEqual(paths, [])
        self.assertIn('native_boundary_branch_or_missing_trace', reasons)
        self.assertIn('continuing_native_stroke', diagnostics['incident_source_primitive_refs'])
        self.assertEqual(len(diagnostics['competing_next_points_display']), 2)

    def test_mid_edge_curve_is_retained_and_trace_cannot_leave_queried_scope(self):
        ports, rows = tee_fixture()
        curve = deepcopy(rows[0])
        curve.update(source_primitive_ref='native_cubic', points_display=[[0, 1], [1, 2], [2, 3], [3, 3]],
                     bbox_display=[0, 1, 3, 3], search_bbox_display=[0, 1, 3, 3])
        curve['source_native_segment']['kind'] = 'cubic'
        paths, reasons = trace_branch_boundaries(ports, [*rows, curve])
        self.assertEqual(paths, [])
        self.assertIn('unsupported_curve_at_native_boundary', reasons)
        paths, reasons = trace_branch_boundaries(ports, rows, search_bbox=[-5, -12, 12, 2])
        self.assertEqual(paths, [])
        self.assertIn('native_boundary_trace_leaves_complete_search_scope', reasons)

    def test_real_multi_port_queries_retain_every_unresolved_native_destination(self):
        root = Path(__file__).resolve().parents[1]
        path = root / 'fixtures/mep/m_and_p_coordination/native-branch-replay.json.gz'
        frozen = json.loads(gzip.decompress(path.read_bytes()))
        self.assertFalse(frozen['engineer_approved'])
        self.assertEqual(len(frozen['connections']), 21)
        self.assertEqual(sum(r['state'] == 'accepted' for r in frozen['connections']), 0)
        through = []
        literal_gaps = []
        for row in frozen['connections']:
            self.assertTrue(row['search']['complete'])
            self.assertEqual(_sha256(sorted(r['source_primitive_ref'] for r in row['source_rows'])),
                             row['search']['all_source_refs_sha256'])
            diagnostics = {}
            paths, reasons = trace_branch_boundaries(row['ports'], row['source_rows'], diagnostics=diagnostics)
            self.assertEqual(paths, [])
            self.assertTrue(set(reasons).issubset(row['reasons']))
            self.assertEqual(diagnostics, row['trace_diagnostics'])
            self.assertEqual(port_attachment_observations(row['ports'], row['source_rows']), row['port_attachment_observations'])
            literal_gaps.extend(r for r in row['port_attachment_observations'] if r['first_collinear_gap_display_points'] is not None)
            self.assertFalse(row['physical_continuation_established'])
            self.assertFalse(row['quantity_eligible'])
            if row['through_stroke_source_primitive_refs']:
                through.append(row)
                self.assertIn('opposing_sidewall_continuation_requires_fitting_body_applicability', row['reasons'])
        self.assertGreater(len(through), 0)
        self.assertTrue(any(.119 < r['first_collinear_gap_display_points'] < .121 for r in literal_gaps))
        self.assertTrue(all(not r['gap_repaired'] for r in literal_gaps))

    def test_live_multi_port_sources_match_complete_native_queries(self):
        import fitz
        from src.drawing_engine.disciplines.mep.mep_native_boundary_queries import NativeBoundaryQueries
        from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256
        root = Path(__file__).resolve().parents[1]
        folder = root / 'fixtures/mep/m_and_p_coordination'
        frozen = json.loads(gzip.decompress((folder / 'native-branch-replay.json.gz').read_bytes()))
        registry = json.loads((folder / 'm_and_p_coordination.sheet-registry.json').read_text())
        source = root / 'M&P mark-up against shop systems piping.pdf'
        self.assertEqual(_file_sha256(source), frozen['document']['source_pdf_sha256'])
        with fitz.open(source) as pdf:
            for page in registry['pages']:
                records = [r for r in frozen['connections'] if r['page_ref'] == page['page_ref']]
                if not records:
                    continue
                index = NativeBoundaryQueries(pdf[page['page_number'] - 1], page['page_ref'])
                for row in records:
                    sources, complete, refs = index.query(row['search']['bbox_display'])
                    self.assertEqual(_sha256(sources), _sha256(row['source_rows']))
                    self.assertEqual(complete, row['search']['complete'])
                    self.assertEqual(refs, row['search']['region_refs'])
                del index


if __name__ == '__main__':
    unittest.main()
