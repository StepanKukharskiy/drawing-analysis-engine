"""Equipment shape/contact evidence cannot bypass body/tag/port authority."""

from copy import deepcopy
import gzip
import json
from pathlib import Path
import unittest

from src.drawing_engine.disciplines.mep.mep_native_equipment_observations import (
    boundary_contact_candidates, build_equipment_review_outcome, rectangular_outline_candidates,
)

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _ROOT / 'fixtures/mep/m_and_p_coordination/equipment-native/native-equipment-evidence.json.gz'


def line(ref, start, end):
    return {'id': ref, 'page_ref': 'page', 'source_primitive_ref': ref,
        'points_display': [start, end], 'bbox_display': [min(start[0], end[0]), min(start[1], end[1]),
                                                       max(start[0], end[0]), max(start[1], end[1])],
        'source_native_segment': {'kind': 'line', 'style': {'width': .5, 'dash': '[] 0',
                                                         'stroke': [0, 0, 0], 'fill': None}}}


def rectangle():
    points = [[0, 0], [20, 0], [20, 20], [0, 20], [0, 0]]
    return [line(str(i), a, b) for i, (a, b) in enumerate(zip(points, points[1:]))]


def outcome(rows):
    return build_equipment_review_outcome(proposal={'id': 'proposal', 'page_ref': 'page',
        'candidate': {'kind': 'equipment_tag', 'equipment_class_token': 'HUH', 'tag': 'HUH-9'}},
        observation={'id': 'text', 'page_ref': 'page', 'page_number': 1, 'text': 'HUH-9',
                     'bbox_display': [3, 4, 12, 8]},
        source_rows=rows, search_bbox_display=[0, 0, 30, 30], search_complete=True,
        region_refs=['query'], leader_paths=[], leader_search_complete=True,
        pdf_grouping={'form_xobject_count': 0, 'marked_content_count': 0})


class NativeEquipmentEvidenceTest(unittest.TestCase):
    def test_closed_rectangle_preserves_duplicate_native_edges_without_identity(self):
        rows = rectangle()
        rows.append(line('duplicate', [20, 0], [0, 0]))
        found = rectangular_outline_candidates(rows, 1)
        self.assertEqual(len(found), 1)
        self.assertEqual(len(found[0]['source_primitive_refs']), 5)
        self.assertFalse(found[0]['equipment_body_identity_established'])
        self.assertFalse(found[0]['quantity_eligible'])

    def test_gap_is_not_repaired_and_crossing_does_not_close_rectangle(self):
        rows = rectangle()
        rows[-1] = line('gap', [0, 20], [0, .0001])
        self.assertEqual(rectangular_outline_candidates(rows, 1), [])

    def test_native_drawing_indices_on_other_pages_do_not_merge_candidates(self):
        original = rectangular_outline_candidates(rectangle(), 1)[0]
        other_rows = rectangle()
        for row in other_rows:
            row['page_ref'] = 'other-page'
        other = rectangular_outline_candidates(other_rows, 1)[0]
        self.assertNotEqual(original['id'], other['id'])
        self.assertEqual(original['source_primitive_refs'], other['source_primitive_refs'])

    def test_through_strokes_remain_distinct_from_terminal_incidence(self):
        body = rectangular_outline_candidates(rectangle(), 1)[0]
        through = [line('a', [5, 10], [5, 30]), line('b', [6, 10], [6, 30])]
        result = boundary_contact_candidates(body, through)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['unresolved_reasons'], ['through_strokes_are_not_terminal_ports'])
        terminal = [line('a', [5, 20], [5, 30]), line('b', [6, 20], [6, 30])]
        result = boundary_contact_candidates(body, terminal)
        self.assertEqual(len(result), 1)
        self.assertIn('terminal_pair_requires_independent_port_opening_or_symbol_certificate', result[0]['unresolved_reasons'])
        self.assertFalse(result[0]['port_identity_established'])

    def test_contained_tag_and_observed_contacts_still_do_not_certify_equipment(self):
        rows = rectangle() + [line('a', [5, 20], [5, 30]), line('b', [6, 20], [6, 30])]
        result = outcome(rows)
        self.assertEqual(result['state'], 'abstained')
        self.assertTrue(result['tag_body_outcome']['containing_outline_candidate_refs'])
        self.assertTrue(result['port_outcome']['native_contact_geometry_observed'])
        self.assertFalse(result['tag_body_outcome']['relation_established'])
        self.assertFalse(result['port_outcome']['identity_established'])
        self.assertFalse(any(result['authority'].values()))

    def test_cross_page_native_evidence_cannot_enter_equipment_outcome(self):
        rows = deepcopy(rectangle())
        rows[0]['page_ref'] = 'other-page'
        with self.assertRaisesRegex(ValueError, 'one page'):
            outcome(rows)


class NativeEquipmentReplayTest(unittest.TestCase):
    def test_real_pilot_replays_separate_body_tag_and_port_abstentions(self):
        from tools.generate_mep_equipment_evidence import replay_outcomes
        from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256
        with gzip.open(_FIXTURE, 'rt') as stream:
            frozen = json.load(stream)
        results = replay_outcomes(frozen)
        self.assertEqual(_sha256(results), frozen['expected_outcomes_sha256'])
        self.assertEqual({(r['page_number'], r['tag']) for r in results}, {(2, 'HUH-9'), (2, 'HUH-13'), (3, 'HUH-14')})
        for row in results:
            self.assertFalse(any(row['authority'].values()))
            self.assertEqual(row['state'], 'abstained')
            self.assertTrue(row['body_outline_candidates'])
            self.assertTrue(row['port_outcome']['native_contact_geometry_observed'])
            self.assertFalse(row['tag_body_outcome']['relation_established'])
            self.assertFalse(row['port_outcome']['identity_established'])
            self.assertEqual(row['pdf_grouping']['form_xobject_count'], 0)
            self.assertEqual(row['pdf_grouping']['marked_content_count'], 0)
        # This source-reviewed body-shaped outline remains an observation. Its
        # actual pipe pair crosses a continuing boundary; it is not a port.
        heater = next(r for r in results if r['tag'] == 'HUH-9')
        body = next(r for r in heater['body_outline_candidates']
                    if r['bbox_display'] == [1370.760009765625, 1758.840087890625, 1408.3199462890625, 1811.4000244140625])
        contacts = [r for r in heater['port_geometry_candidates'] if r['body_outline_candidate_ref'] == body['id']
                    and r['contacts'][0]['side'] == 'bottom']
        self.assertEqual(len(contacts), 2)
        self.assertTrue(all(r['unresolved_reasons'] == ['through_strokes_are_not_terminal_ports'] for r in contacts))

    def test_live_native_queries_reproduce_every_pilot_equipment_source(self):
        import fitz
        from tools.generate_mep_equipment_evidence import grouping_evidence
        from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256, _sha256
        from src.drawing_engine.disciplines.mep.mep_native_boundary_queries import NativeBoundaryQueries
        with gzip.open(_FIXTURE, 'rt') as stream:
            frozen = json.load(stream)
        source = _ROOT / 'M&P mark-up against shop systems piping.pdf'
        self.assertEqual(_file_sha256(source), frozen['source_pdf_sha256'])
        with fitz.open(source) as document:
            for page in frozen['pages']:
                native_page = document[page['page_number'] - 1]
                index = NativeBoundaryQueries(native_page, page['page_ref'])
                self.assertEqual(grouping_evidence(document, native_page), page['items'][0]['inputs']['pdf_grouping'])
                for query in page['queries'].values():
                    rows, complete, regions = index.query(query['bbox_display'])
                    self.assertEqual(_sha256(rows), query['source_rows_sha256'])
                    self.assertEqual(complete, query['complete'])
                    self.assertEqual(regions, query['region_refs'])


if __name__ == '__main__':
    unittest.main()
