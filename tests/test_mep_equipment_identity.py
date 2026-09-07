from copy import deepcopy
import gzip
import json
from pathlib import Path
import unittest

from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256
from src.drawing_engine.disciplines.mep.mep_equipment_identity import build_equipment_2d_identity, discover_cabinet_motifs, replay_equipment_2d_identity

_ROOT = Path(__file__).resolve().parents[1]


def frozen_identity_scope(name):
    """Read the content-addressed producer artifacts, never claim-inline inputs."""
    from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256
    directory = _ROOT / f'output/mep-equipment-2d-{name}-2026-08-31'
    payload = json.loads((directory / 'equipment-2d-identities.json').read_text())
    manifest = json.loads((directory / 'equipment-2d-replay.json').read_text())
    native_path = Path(manifest['native_replay']['path'])
    assert _file_sha256(native_path) == manifest['native_replay']['file_sha256']
    assert _file_sha256(native_path) == payload['native_replay_sha256']
    assert _sha256(payload) == manifest['identity_payload_sha256']
    native = json.loads(gzip.decompress(native_path.read_bytes()))
    terminology = json.loads(Path(manifest['terminology']['path']).read_text())
    assert _sha256(terminology) == manifest['terminology']['payload_sha256']
    return payload, {'native_inputs': native, 'typography': manifest['typography'], 'terminology': terminology}


def cabinet_rows(page_ref, offset=0):
    boxes = [[0,0,70,14], [1.5,1,11.5,13], [12.5,1,57.5,13], [58.5,1,68.5,13],
             [2.5,4,10.5,10], [13,4,57,10], [59.5,4,67.5,10]]
    rows = []
    for i, (x0,y0,x1,y1) in enumerate(boxes):
        points = [[x0+offset,y0],[x1+offset,y0],[x1+offset,y1],[x0+offset,y1],[x0+offset,y0]]
        for j, (a,b) in enumerate(zip(points, points[1:])):
            ref = f'drawing[{offset+i}].item[{j}].segment[0]'
            rows.append({'id': f'{page_ref}.{ref}', 'page_ref': page_ref, 'source_primitive_ref': ref,
                'points_display': [a,b], 'bbox_display': [min(a[0],b[0]),min(a[1],b[1]),max(a[0],b[0]),max(a[1],b[1])],
                'source_native_segment': {'kind':'line','style':{'width':.2,'dash':'[] 0','stroke':[0,0,0],'fill':None}}})
    return rows


def identity_fixture():
    pages, typography = [], []
    for n in [1,2]:
        page = f'page.{n}'; rows = cabinet_rows(page); tag = f'HUH-{n}'
        obs = {'id':f'text.{n}','page_ref':page,'page_number':n,'source_pdf_sha256':'a'*64,
               'text':tag,'bbox_display':[20,-35,85,-10],'region_role':'unknown'}
        proposal = {'id':f'proposal.{n}','page_ref':page,'state':'abstained','reasons':['region_applicability_unresolved'],
            'candidate':{'kind':'equipment_tag','equipment_class_token':'HUH','tag':tag},'conflicts':[],'alternatives':[]}
        query = {'complete':True,'bbox_display':[-150,-185,235,140],
                 'source_row_refs':[r['id'] for r in rows],'source_rows_sha256':_sha256(rows)}
        pages.append({'page_number':n,'page_ref':page,'source_rows':rows,'queries':{'q':query},
                      'items':[{'query_key':'q','inputs':{'proposal':proposal,'observation':obs}}]})
        typography.append({'id':f'font.{n}','source_observation_ref':obs['id'],'page_ref':page,'bbox_display':obs['bbox_display'],
            'native_text':tag,'bold':True,'horizontal':True,'source_pdf_sha256':'a'*64,'font':'Fixture-Bold'})
    return {'pages':pages,'source_pdf_sha256':'a'*64,'document':{'source_pdf_sha256':'a'*64}}, typography


class EquipmentIdentityTest(unittest.TestCase):
    def test_repeated_cabinet_and_unique_native_tag_resolve_only_2d_context(self):
        native, typography = identity_fixture()
        result = build_equipment_2d_identity(native_inputs=native, typography=typography)
        self.assertEqual(result['summary']['accepted_projected_body_tag_identities'],2)
        for row in result['identities']:
            self.assertEqual(row['epistemic_state'],'inferred')
            self.assertEqual(row['source_m2_state'],'abstained')
            self.assertEqual(row['resolved_context_abstentions'],['region_applicability_unresolved'])
            self.assertTrue(row['authority']['projected_body_tag_identity_established'])
            self.assertFalse(any(value for key,value in row['authority'].items() if key!='projected_body_tag_identity_established'))
            self.assertEqual(row['port_outcome']['state'],'unresolved')

    def test_adjacent_rectangle_without_internal_motif_does_not_identify_equipment(self):
        self.assertEqual(discover_cabinet_motifs(cabinet_rows('page')[:4])[0],[])

    def test_two_bodies_in_same_tag_zone_abstain_instead_of_nearest_choice(self):
        native, typography = identity_fixture()
        for page, font in zip(native['pages'],typography):
            rows = page['source_rows'] + cabinet_rows(page['page_ref'],75)
            page['source_rows'] = rows
            page['queries']['q'].update(source_row_refs=[r['id'] for r in rows],source_rows_sha256=_sha256(rows))
            page['items'][0]['inputs']['observation']['bbox_display']=[35,-35,110,-10]
            font['bbox_display']=[35,-35,110,-10]
        result = build_equipment_2d_identity(native_inputs=native,typography=typography)
        self.assertTrue(all('body_tag_ownership_is_not_mutually_unique' in r['reasons'] for r in result['identities']))

    def test_region_only_exception_does_not_override_conflicts_or_legend(self):
        for mutation in ['conflict','legend','extra_reason']:
            native, typography = identity_fixture()
            for page in native['pages']:
                proposal = page['items'][0]['inputs']['proposal']
                if mutation=='conflict': proposal['conflicts']=['other class']
                if mutation=='legend': proposal['legend_membership']=True
                if mutation=='extra_reason': proposal['reasons'].append('semantic_ambiguity')
            result = build_equipment_2d_identity(native_inputs=native,typography=typography)
            self.assertEqual(result['summary']['accepted_projected_body_tag_identities'],0)

    def test_one_example_cannot_calibrate_its_own_equipment_class(self):
        native, typography = identity_fixture(); native['pages']=native['pages'][:1]
        result = build_equipment_2d_identity(native_inputs=native,typography=typography[:1])
        self.assertEqual(result['summary']['accepted_projected_body_tag_identities'],0)

    def test_incomplete_native_query_and_forged_typography_fail_closed(self):
        for mutation in ['incomplete','wrong_page','wrong_text']:
            native, typography = identity_fixture()
            if mutation=='incomplete':
                for page in native['pages']: page['queries']['q']['complete']=False
            if mutation=='wrong_page':
                for font in typography: font['page_ref']='other'
            if mutation=='wrong_text':
                for font in typography: font['native_text']='something else'
            result = build_equipment_2d_identity(native_inputs=native,typography=typography)
            self.assertEqual(result['summary']['accepted_projected_body_tag_identities'],0)


class EquipmentIdentityReplayTest(unittest.TestCase):
    def test_real_accepted_body_keeps_terminal_tabs_and_internal_endpoints_unresolved(self):
        pilot, inputs = frozen_identity_scope('pilot')
        result = replay_equipment_2d_identity(pilot, **inputs)
        accepted = {r['equipment_tag']: r for r in result['identities'] if r['state'] == 'accepted'}
        for tag, count, terminal_count in [('HUH-13', 13, 2), ('HUH-14', 8, 1)]:
            row = accepted[tag]
            candidates = row['port_outcome']['port_geometry_candidates']
            terminal = [c for c in candidates if all(p['contact_kind'] == 'native_endpoint_incidence'
                                                     for p in c['contacts'])]
            self.assertEqual(len(candidates), count)
            self.assertEqual(len(terminal), terminal_count)
            self.assertTrue(row['authority']['projected_body_tag_identity_established'])
            self.assertFalse(row['authority']['equipment_port_binding_established'])
            self.assertTrue(all(not c['port_identity_established'] for c in candidates))
            self.assertTrue(all(c['unresolved_reasons'] ==
                ['terminal_pair_requires_independent_port_opening_or_symbol_certificate'] for c in terminal))
        # These short source strokes end just inside the accepted body. Their
        # crossing of its continuous outside edge is not a terminal at a port.
        heater = accepted['HUH-13']
        pair_refs = {'drawing[131603].item[0].segment[0]', 'drawing[131604].item[0].segment[0]'}
        internal = next(c for c in heater['port_outcome']['port_geometry_candidates']
                        if set(c['source_primitive_refs']) == pair_refs)
        self.assertTrue(all(c['contact_kind'] == 'native_through_stroke' for c in internal['contacts']))
        self.assertEqual(internal['unresolved_reasons'], ['through_strokes_are_not_terminal_ports'])
        negative = next(r for r in result['identities'] if r['equipment_tag'] == 'HUH-9')
        self.assertEqual(negative['state'], 'abstained')
        self.assertEqual(negative['port_outcome']['state'], 'unresolved')
        forged = deepcopy(pilot)
        row = next(r for r in forged['identities'] if r['id'] == heater['id'])
        row['port_outcome']['state'] = 'accepted'
        row['port_outcome']['equipment_port_binding_established'] = True
        row['authority']['equipment_port_binding_established'] = True
        with self.assertRaisesRegex(ValueError, 'does not replay exactly'):
            replay_equipment_2d_identity(forged, **inputs)

    def test_real_projected_input_archives_replay_without_original_paths(self):
        from unittest.mock import patch
        from src.drawing_engine.disciplines.mep.mep_projected_identity_inputs import (load_equipment_identity_inputs,
            load_fitting_identity_inputs, replay_equipment_identity_artifacts, replay_fitting_identity_artifacts)
        cases = [(load_equipment_identity_inputs, replay_equipment_identity_artifacts,
                  _ROOT / 'output/mep-equipment-2d-pilot-2026-08-31/equipment-2d-identities.json'),
                 (load_fitting_identity_inputs, replay_fitting_identity_artifacts,
                  _ROOT / 'output/mep-fitting-hypotheses-2026-08-31/pilot.json.gz')]
        for loader, portable_replay, path in cases:
            payload, _, artifacts = loader(path)
            with patch.object(Path, 'read_bytes', side_effect=AssertionError('portable replay read a local file')):
                replayed, _ = portable_replay(artifacts)
            self.assertEqual(_sha256(payload), _sha256(replayed))
            archive = next(row for row in artifacts.values() if row['layer'] == 'mep_opaque_native_replay_archive')
            self.assertNotIn('source_rows', archive)
            archive['file_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'archive binding mismatch'):
                portable_replay(artifacts)

    def test_real_pilot_and_outside_tuning_page_replay_identity_without_ports(self):
        pilot, pilot_inputs = frozen_identity_scope('pilot')
        result = replay_equipment_2d_identity(pilot, **pilot_inputs)
        accepted = {row['equipment_tag']: row for row in result['identities'] if row['state'] == 'accepted'}
        self.assertEqual(set(accepted), {'HUH-13', 'HUH-14'})
        for row in accepted.values():
            self.assertEqual(row['epistemic_state'], 'inferred')
            self.assertEqual(len(row['source_primitive_refs']), 58)
            self.assertTrue(row['port_outcome']['port_geometry_candidates'])
            self.assertFalse(row['authority']['equipment_port_binding_established'])
            self.assertFalse(row['authority']['physical_item_identity_established'])
        unsupported = next(row for row in result['identities'] if row['equipment_tag'] == 'HUH-9')
        self.assertEqual(unsupported['state'], 'abstained')
        self.assertIn('no_supported_cabinet_motif_in_typographic_label_zone', unsupported['reasons'])

        heldout, heldout_inputs = frozen_identity_scope('holdout')
        self.assertEqual(heldout['coverage']['execution_page_numbers'], [5])
        self.assertFalse(heldout['coverage']['calibration_consumes_current_scope'])
        heldout_result = replay_equipment_2d_identity(heldout, **heldout_inputs,
            calibration_payload=pilot, calibration_native_inputs=pilot_inputs['native_inputs'],
            calibration_typography=pilot_inputs['typography'], calibration_terminology=pilot_inputs['terminology'])
        by_tag = {row['equipment_tag']: row for row in heldout_result['identities']}
        self.assertEqual(set(by_tag), {'HUH-14', 'HUH-12', 'HUH-11', 'HUH-8'})
        self.assertEqual({tag for tag, row in by_tag.items() if row['state'] == 'accepted'}, {'HUH-14'})
        self.assertTrue(by_tag['HUH-12']['body_motif_ref'])
        self.assertEqual(by_tag['HUH-12']['reasons'], ['independent_repeated_symbol_and_tag_class_convention_not_closed'])
        self.assertEqual(heldout_result['drawing_convention'], result['drawing_convention'])
        self.assertFalse(any(row['authority']['physical_item_identity_established'] for row in by_tag.values()))

        # Mutating the claim badge or authority cannot survive independent replay.
        forged = deepcopy(pilot)
        forged['identities'][0]['authority']['physical_item_identity_established'] = True
        with self.assertRaisesRegex(ValueError, 'does not replay exactly'):
            replay_equipment_2d_identity(forged, **pilot_inputs)
        forged = deepcopy(pilot)
        forged['input_payload_sha256']['terminology-proposals'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'different frozen M2'):
            replay_equipment_2d_identity(forged, **pilot_inputs)
        with self.assertRaisesRegex(ValueError, 'complete calibration replay evidence'):
            replay_equipment_2d_identity(heldout, **heldout_inputs, calibration_payload=pilot)
        # A newly captured subset cannot omit competing M2 tags to claim uniqueness.
        incomplete = {**pilot_inputs['native_inputs'], 'pages': [dict(page) for page in pilot_inputs['native_inputs']['pages']]}
        incomplete['pages'][0]['items'] = incomplete['pages'][0]['items'][1:]
        with self.assertRaisesRegex(ValueError, 'tag competitor search is incomplete'):
            replay_equipment_2d_identity(pilot, **{**pilot_inputs, 'native_inputs': incomplete})

    def test_live_typography_and_every_heldout_body_query_match_original_source(self):
        import fitz
        from tools.generate_mep_equipment_identity import native_typography
        from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256
        from src.drawing_engine.disciplines.mep.mep_native_boundary_queries import NativeBoundaryQueries
        source = _ROOT / 'M&P mark-up against shop systems piping.pdf'
        source_hash = _file_sha256(source)
        with fitz.open(source) as document:
            for name in ['pilot', 'holdout']:
                payload, inputs = frozen_identity_scope(name)
                self.assertEqual(source_hash, payload['document']['source_pdf_sha256'])
                self.assertEqual(native_typography(document, inputs['native_inputs']), inputs['typography'])
                if name == 'pilot':
                    # All 346 pilot native queries have their own live replay test.
                    continue
                for page in inputs['native_inputs']['pages']:
                    index = NativeBoundaryQueries(document[page['page_number'] - 1], page['page_ref'])
                    for key in {item['query_key'] for item in page['items']}:
                        query = page['queries'][key]
                        rows, complete, regions = index.query(query['bbox_display'])
                        self.assertEqual(_sha256(rows), query['source_rows_sha256'])
                        self.assertEqual(complete, query['complete'])
                        self.assertEqual(regions, query['region_refs'])


if __name__=='__main__':
    unittest.main()
