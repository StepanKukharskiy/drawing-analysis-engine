from copy import deepcopy
import unittest

from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256
from src.drawing_engine.disciplines.mep.mep_duct_evidence import (
    _dimension_note_basis,
    build_duct_size_applicability_certificates,
    duct_section_evidence,
)


def fixture(*, convention='outside', association='leader'):
    document = {'source_pdf_sha256': 'a' * 64}
    size = {
        'id': 'text-size', 'page_ref': 'page-plan', 'text': '24x18 SA',
        'region_role': 'drawing_inline', 'evidence_channels': ['native_pdf_text'],
        'source_native_ref': 'page[2].text[1]',
        'native_spans': [{'source_native_ref': 'page[2].text[1].span[0]'}],
    }
    note_text = {
        'outside': ('ALL DUCTWORK SIZES SHOWN ARE OUTSIDE DIMENSIONS, '
                    'UNLESS SPECIFICALLY NOTED ON PLANS.'),
        'inside': 'ALL DUCTWORK SIZES SHOWN ARE INSIDE DIMENSIONS.',
        'free_area': 'ALL DUCT SIZES SHOWN ARE FREE AREA SIZES.',
        'none': 'COORDINATE DUCTWORK WITH OTHER TRADES.',
    }[convention]
    note = {
        'id': 'text-convention', 'page_ref': 'page-notes', 'text': note_text,
        'region_role': 'note', 'evidence_channels': ['native_pdf_text'],
        'source_native_ref': 'page[1].text[4]',
        'native_spans': [{'source_native_ref': 'page[1].text[4].span[0]'}],
        'region_role_evidence': {'region_body_complete': True},
    }
    proposal = {
        'id': 'proposal-size', 'anchor_ref': size['id'], 'page_ref': size['page_ref'],
        'proposal_type': 'inline_size', 'candidate': {
            'kind': 'rectangular_duct_size', 'width': 24.0, 'height': 18.0,
            'unit': 'in', 'raw_text': size['text']},
    }
    terminology = {
        'document': document, 'source_observations': [size, note],
        'proposals': [proposal],
    }
    composite = {
        'id': 'duct-interval', 'page_ref': 'page-plan', 'state': 'accepted',
        'member_source_primitive_refs': ['rail-a', 'rail-b'],
        'derived_geometry': {'centreline_points_display': [[0, 5], [20, 5]]},
    }
    if association == 'leader':
        binding = {
            'id': 'binding', 'state': 'observed', 'target_kind': 'route_composite',
            'target_refs': [composite['id']],
            'geometric_evidence_refs': [composite['id'], 'rail-a', 'rail-b', 'leader'],
            'method': {'name': 'complete_native_dot_leader_contact', 'version': '1.0.0'},
            'automatic_search_certificate': {
                'leader_observation_ref': size['id'],
                'source_searches_sha256': 'b' * 64,
                'reviewed_selectors_used': False,
            },
        }
    else:
        binding = {
            'id': 'binding', 'state': 'observed', 'target_kind': 'route_composite',
            'target_refs': [composite['id']],
            'geometric_evidence_refs': [composite['id'], 'rail-a', 'rail-b'],
            'method': {
                'name': 'unique_native_corridor_inline_air_callout',
                'version': '1.0.0',
            },
            'duct_inline_certificate': {
                'state': 'accepted', 'source_observation_ref': size['id'],
                'measured_evidence': {
                    'query_complete': True,
                    'geometry_frozen_before_binding': True,
                    'color_used_for_identity': False,
                },
            },
        }
    relation = {
        'id': 'route-size', 'page_ref': 'page-plan',
        'proposal_ref': proposal['id'], 'relation_type': 'route_size',
        'candidate': deepcopy(proposal['candidate']), 'state': 'accepted',
        'target_kind': 'route_composite', 'target_refs': [composite['id']],
        'binding_evidence_refs': [binding['id']],
        'certificates': {
            'upstream_proposal_is_unconflicted': True,
            'unique_page_scope': True,
            'unique_geometric_target': True,
            'target_is_topologically_connected': True,
            'branch_coverage_is_explicit_or_not_required': True,
            'equipment_port_connectivity_is_explicit_or_not_applicable': True,
            'continuation_target_is_unique_terminal_endpoint_or_not_applicable': True,
        },
    }
    bindings = {
        'document': document,
        'm2_contract_ref': {'payload_sha256': _sha256(terminology)},
        'outlined_route_composites': [composite],
        'binding_evidence': [binding], 'relations': [relation],
    }
    return terminology, bindings


class DuctSizeApplicabilityCertificateTests(unittest.TestCase):
    def test_native_leader_binds_outside_size_to_one_interval(self):
        terminology, bindings = fixture()
        before = deepcopy((terminology, bindings))
        row = build_duct_size_applicability_certificates(
            terminology=terminology, bindings=bindings)[0]
        self.assertEqual(row['state'], 'accepted')
        self.assertEqual(row['association_kind'], 'leader')
        self.assertEqual(row['dimension_basis'], 'outside')
        self.assertTrue(all(row['certificates'].values()))
        self.assertEqual(row['route_composite_ref'], 'duct-interval')
        self.assertEqual(row['source_pdf_sha256'], 'a' * 64)
        self.assertEqual(row['evidence']['interval_source_primitive_refs'],
                         ['rail-a', 'rail-b'])
        self.assertTrue(row['authority']['outside_size_applicability_established'])
        self.assertFalse(row['authority']['physical_3d_envelope_established'])
        self.assertFalse(row['quantity_eligible'])
        self.assertEqual((terminology, bindings), before)

    def test_inline_association_replays_complete_native_corridor_query(self):
        terminology, bindings = fixture(association='inline')
        row = build_duct_size_applicability_certificates(
            terminology=terminology, bindings=bindings)[0]
        self.assertEqual(row['state'], 'accepted')
        self.assertEqual(row['association_kind'], 'inline')
        bad = deepcopy(bindings)
        bad['binding_evidence'][0]['duct_inline_certificate'][
            'measured_evidence']['query_complete'] = False
        bad['m2_contract_ref']['payload_sha256'] = _sha256(terminology)
        row = build_duct_size_applicability_certificates(
            terminology=terminology, bindings=bad)[0]
        self.assertEqual(row['state'], 'abstained')
        self.assertIn('unique_native_leader_or_inline_association_not_established',
                      row['reasons'])

    def test_ambiguous_native_interval_abstains(self):
        terminology, bindings = fixture()
        other = deepcopy(bindings['outlined_route_composites'][0])
        other['id'] = 'duct-interval-2'
        other['member_source_primitive_refs'] = ['rail-c', 'rail-d']
        bindings['outlined_route_composites'].append(other)
        relation = bindings['relations'][0]
        relation['target_refs'].append(other['id'])
        relation['certificates']['unique_geometric_target'] = False
        bindings['binding_evidence'][0]['target_refs'].append(other['id'])
        bindings['binding_evidence'][0]['geometric_evidence_refs'].append(other['id'])
        row = build_duct_size_applicability_certificates(
            terminology=terminology, bindings=bindings)[0]
        self.assertEqual(row['state'], 'abstained')
        self.assertFalse(row['certificates']['unique_native_duct_interval'])
        self.assertIn('unique_native_duct_interval_not_established', row['reasons'])
        self.assertIsNone(row['route_composite_ref'])

    def test_inside_free_area_missing_and_conflicting_conventions_fail_closed(self):
        expected = {
            'inside': 'inside_dimensions_do_not_establish_outside_dimensions',
            'free_area': 'free_area_dimensions_do_not_establish_outside_dimensions',
            'none': 'outside_dimension_convention_not_established',
        }
        for convention, reason in expected.items():
            with self.subTest(convention=convention):
                terminology, bindings = fixture(convention=convention)
                row = build_duct_size_applicability_certificates(
                    terminology=terminology, bindings=bindings)[0]
                self.assertEqual(row['state'], 'abstained')
                self.assertFalse(row['certificates']['outside_dimension_convention'])
                self.assertIn(reason, row['reasons'])

        terminology, bindings = fixture()
        local = deepcopy(terminology['source_observations'][1])
        local.update(id='local-inside', page_ref='page-plan',
                     text='DUCT SIZES SHOWN ON THE PLAN ARE INSIDE DIMENSIONS.')
        terminology['source_observations'].append(local)
        bindings['m2_contract_ref']['payload_sha256'] = _sha256(terminology)
        row = build_duct_size_applicability_certificates(
            terminology=terminology, bindings=bindings)[0]
        self.assertEqual(row['dimension_basis'], 'conflicting')
        self.assertIn('conflicting_duct_dimension_conventions', row['reasons'])

    def test_convention_parser_accepts_global_default_but_rejects_exceptions(self):
        observation = {
            'region_role': 'note',
            'region_role_evidence': {'region_body_complete': True},
        }
        observation['text'] = ('ALL DUCTWORK SIZES SHOWN ARE OUTSIDE DIMENSIONS, '
                               'UNLESS SPECIFICALLY NOTED ON PLANS.')
        self.assertEqual(_dimension_note_basis(observation), 'exterior')
        observation['text'] = 'ALL DUCTWORK SIZES SHOWN ARE INSIDE DIMENSIONS.'
        self.assertEqual(_dimension_note_basis(observation), 'interior')
        observation['text'] = 'DUCT SIZES ARE OUTSIDE DIMENSIONS EXCEPT LINED DUCTS.'
        self.assertIsNone(_dimension_note_basis(observation))

    def test_existing_cross_section_gate_does_not_treat_inside_note_as_exterior(self):
        terminology, bindings = fixture(convention='inside')
        terminology['source_observations'][1]['page_ref'] = 'page-plan'
        bindings['m2_contract_ref']['payload_sha256'] = _sha256(terminology)
        composite = bindings['outlined_route_composites'][0]
        composite['derived_geometry']['corridor_width_display_points'] = 18
        composite['geometry_metrics'] = {'member_width_display_points': .1}
        registry = {'pages': [{'page_ref': 'page-plan', 'fields': {'scale': {
            'state': 'observed', 'drawing_inches_per_paper_inch': 96,
            'evidence_refs': ['scale'],
        }}}]}
        row = duct_section_evidence(registry=registry, terminology=terminology,
                                    bindings=bindings)[0]
        self.assertEqual(row['state'], 'abstained')
        self.assertEqual(row['dimension_basis'], 'unknown')
        self.assertIn('duct_dimension_basis_not_explicitly_established',
                      row['reasons'])
        self.assertFalse(row['physical_outer_envelope_established'])

    def test_non_native_size_text_and_mismatched_frozen_m2_abstain_or_raise(self):
        terminology, bindings = fixture()
        terminology['source_observations'][0]['evidence_channels'] = ['ocr']
        bindings['m2_contract_ref']['payload_sha256'] = _sha256(terminology)
        row = build_duct_size_applicability_certificates(
            terminology=terminology, bindings=bindings)[0]
        self.assertEqual(row['state'], 'abstained')
        self.assertIn('native_width_height_text_not_established', row['reasons'])
        bindings['m2_contract_ref']['payload_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'matching frozen M2/M4'):
            build_duct_size_applicability_certificates(
                terminology=terminology, bindings=bindings)

    def test_unmarked_size_requires_native_rectangular_duct_unit_legend(self):
        terminology, bindings = fixture()
        terminology['proposals'][0]['candidate']['unit'] = None
        bindings['relations'][0]['candidate']['unit'] = None
        legend = {
            'id': 'rect-unit-legend', 'page_ref': 'page-notes',
            'text': 'RECT. DUCT SIZE (INCHES) (FACING SIDE LISTED FIRST)',
            'region_role': 'legend', 'evidence_channels': ['native_pdf_text'],
            'source_native_ref': 'page[1].text[9]',
            'native_spans': [{'source_native_ref': 'page[1].text[9].span[0]'}],
        }
        terminology['source_observations'].append(legend)
        bindings['m2_contract_ref']['payload_sha256'] = _sha256(terminology)
        row = build_duct_size_applicability_certificates(
            terminology=terminology, bindings=bindings)[0]
        self.assertEqual(row['state'], 'accepted')
        self.assertEqual(row['dimension_unit'], 'in')
        self.assertTrue(row['certificates']['rectangular_duct_dimension_unit'])

        terminology['source_observations'].remove(legend)
        bindings['m2_contract_ref']['payload_sha256'] = _sha256(terminology)
        row = build_duct_size_applicability_certificates(
            terminology=terminology, bindings=bindings)[0]
        self.assertEqual(row['state'], 'abstained')
        self.assertIn('rectangular_duct_dimension_unit_unresolved', row['reasons'])


if __name__ == '__main__':
    unittest.main()
