from copy import deepcopy
import unittest

import fitz

from tools.cover_mep_endpoint_review import (
    boundary_witnesses, category_summary, proper_crossing, union_boxes, validate_expansion,
)
from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256
from src.drawing_engine.disciplines.mep.mep_native_boundary_queries import NativeBoundaryQueries
from src.drawing_engine.disciplines.mep.mep_native_subpaths import reconstruct_native_subpaths


class EndpointReviewCoverageTest(unittest.TestCase):
    def fixture(self):
        doc = fitz.open(); page = doc.new_page(width=100, height=100)
        # Authored pipe-like line ending at a separately drawn body boundary,
        # plus an independently authored crossing. This is synthetic geometry,
        # deliberately not a qualified real-source termination fixture.
        page.draw_line((10, 20), (50, 20))
        page.draw_rect(fitz.Rect(50, 10, 70, 30))
        page.draw_line((30, 5), (30, 40))
        index = NativeBoundaryQueries(page, 'page')
        def query(box):
            rows, complete, refs = index.query(box)
            return {'page_ref':'page', 'source_pdf_sha256':'test-source', 'source_rows':rows,
                'search':{'bbox_display':box, 'complete':complete, 'query_refs':refs,
                    'source_rows_sha256':_sha256(rows),
                    'all_source_refs_sha256':_sha256(sorted(r['source_primitive_ref'] for r in rows))}}
        old, new = query([5, 15, 45, 25]), query([0, 0, 80, 50])
        doc.close()
        return old, new

    def test_expansion_reproduces_old_subquery(self):
        old, new = self.fixture()
        self.assertGreater(validate_expansion(old, new), 0)

    def test_larger_crop_alone_does_not_expand_inventory(self):
        old, new = self.fixture()
        new['search']['complete'] = False
        with self.assertRaises(ValueError): validate_expansion(old, new)

    def test_changed_source_or_page_rejected(self):
        for field in ('source_pdf_sha256', 'page_ref'):
            old, new = self.fixture(); new[field] = 'other'
            with self.assertRaises(ValueError): validate_expansion(old, new)

    def test_rehashed_missing_old_primitive_rejected(self):
        old, new = self.fixture(); missing = old['source_rows'][0]['source_primitive_ref']
        new['source_rows'] = [r for r in new['source_rows'] if r['source_primitive_ref'] != missing]
        new['search']['source_rows_sha256'] = _sha256(new['source_rows'])
        new['search']['all_source_refs_sha256'] = _sha256(sorted(r['source_primitive_ref'] for r in new['source_rows']))
        with self.assertRaisesRegex(ValueError, 'reproduce'): validate_expansion(old, new)

    def test_all_review_extents_are_in_union(self):
        self.assertEqual([0, -2, 30, 40], union_boxes([[0, 0, 10, 10], [5, -2, 30, 40]]))

    def test_endpoint_contact_not_misreported_as_interior_crossing(self):
        self.assertIsNone(proper_crossing([0, 0], [10, 0], [10, -2], [10, 2]))
        self.assertEqual([5, 0], proper_crossing([0, 0], [10, 0], [5, -2], [5, 2]))

    def test_authored_endpoint_and_body_contact_do_not_require_physical_cap(self):
        _, query = self.fixture()
        inventory = reconstruct_native_subpaths(query['source_rows'], query['search'])
        ref = query['source_rows'][0]['source_primitive_ref']
        card = {'case_id':'contact', 'bbox_display':[0, 0, 80, 50]}
        result = boundary_witnesses({'native_subpath_inventory':inventory}, query, [card], {ref})[0]
        self.assertTrue(result['mechanism_passed'])
        self.assertEqual(2, len(result['authored_endpoint_witnesses']))
        self.assertTrue(result['independent_crossing_witnesses'])
        self.assertTrue(all(w['not_merged_at_crossing'] for w in result['independent_crossing_witnesses']))
        self.assertTrue(all(not w['physical_termination_established'] for w in result['authored_endpoint_witnesses']))

    def test_authored_endpoint_mutation_fails_mechanism(self):
        _, query = self.fixture()
        inventory = reconstruct_native_subpaths(query['source_rows'], query['search'])
        path = inventory['subpaths'][0]
        # Test a real stored chain endpoint, not the line's singleton.
        ref = path['source_primitive_refs'][0]
        inventory = deepcopy(inventory); inventory['subpaths'][0]['endpoints'][0]['point_display'] = [999, 999]
        result = boundary_witnesses({'native_subpath_inventory':inventory}, query,
            [{'case_id':'body', 'bbox_display':[0, 0, 80, 50]}], {ref})[0]
        self.assertFalse(result['mechanism_passed'])

    def test_physical_categories_do_not_block_authored_preservation(self):
        witness = {'case_id':'contact', 'authored_endpoint_witnesses':[{}],
            'branch_alternative_witnesses':[{}], 'independent_crossing_witnesses':[{}], 'mechanism_passed':True}
        result = category_summary([witness], [{'case_id':'contact', 'source_role':'equipment_contact'}])
        self.assertTrue(result['authored_geometry_preservation_passed'])
        self.assertFalse(result['physical_terminal_gaps_block_authored_geometry_test'])
        self.assertFalse(result['publication_gate_passed'])
        self.assertFalse(result['native_multi_incidence_proves_physical_branch'])
        self.assertIn('gap:', result['physical_terminal_categories']['physical_cap'])

    def test_missing_boundary_witness_does_not_pass_empty_set(self):
        self.assertFalse(category_summary([], [])['authored_geometry_preservation_passed'])


class EndpointReviewCoverageReplayTest(unittest.TestCase):
    def test_frozen_source_bound_cases_and_separate_category_gaps(self):
        from pathlib import Path
        from src.drawing_engine.pipelines.generate_mep_stroke_ownership import read
        from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256
        root = Path(__file__).resolve().parents[1]
        folder = root/'output/mep-endpoint-review-coverage-2026-09-04'
        result = read(folder/'results.json'); dataset = read(folder/'endpoint-test-set.json')
        self.assertEqual(dataset['coverage_results_sha256'], _file_sha256(folder/'results.json'))
        self.assertEqual(65, dataset['source_review_crop_count'])
        self.assertTrue(dataset['coverage_gate_passed'])
        self.assertEqual(9, len(dataset['tests']))
        self.assertEqual(15, len(dataset['apparently_continuous_intervals']))
        self.assertEqual(6, len(dataset['ambiguous_intervals']))
        self.assertTrue(all(not r['gained_parameter_intervals'] and not r['lost_parameter_intervals'] for r in result['expanded_query_replays']))
        self.assertTrue(dataset['category_results']['authored_geometry_preservation_passed'])
        self.assertFalse(dataset['category_results']['publication_gate_passed'])
        self.assertEqual(0, dataset['new_identity_accepts'])
        self.assertIsNone(dataset['installed_length'])

    def test_expanded_native_capture_and_endpoint_test_set_replay(self):
        from pathlib import Path
        from shutil import copyfile
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from tools.cover_mep_endpoint_review import verify
        from src.drawing_engine.pipelines.generate_mep_stroke_ownership import read
        root = Path(__file__).resolve().parents[1]
        folder = root/'output/mep-endpoint-review-coverage-2026-09-04'
        expected = read(folder/'endpoint-test-set.json')
        with TemporaryDirectory() as directory:
            output = Path(directory)
            for name in ('protocol.json', 'results.json'): copyfile(folder/name, output/name)
            verify(SimpleNamespace(output=output, denominator=root/'output/mep-source-denominator-page5-v2-2026-09-03/manifest.json'))
            actual = read(output/'endpoint-test-set.json')
            from tools.source_revision import digest, frozen_source_path
            original_hash = expected.pop('code_sha256')
            original = frozen_source_path(root, 'experiments/cover_mep_endpoint_review.py', original_hash)
            self.assertEqual(original_hash, digest(original))
            self.assertEqual(actual.pop('code_sha256'), digest(root/'tools/cover_mep_endpoint_review.py'))
            self.assertEqual(expected, actual)
