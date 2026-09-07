"""Real projected identity reaches M4/HVAC/M5 without leaking into M7."""

from copy import deepcopy
import json
import gzip
from pathlib import Path
import unittest

from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256
from src.drawing_engine.disciplines.mep.mep_hvac_inventory import build_mep_hvac_inventory
from src.drawing_engine.disciplines.mep.mep_item_catalog import build_mep_item_catalog
from src.drawing_engine.disciplines.mep.mep_network_assembly import build_mep_networks
from src.drawing_engine.disciplines.mep.mep_projected_identity_bindings import build_projected_identity_bindings
from src.drawing_engine.disciplines.mep.mep_projected_identity_inputs import load_equipment_identity_inputs, load_fitting_identity_inputs


ROOT = Path(__file__).resolve().parents[1]


class MepProjectedIdentityBindingsTest(unittest.TestCase):
    def test_real_pilot_replays_two_units_and_one_branch_without_physical_or_m7_authority(self):
        run = ROOT / 'output/mep-drawing-interpretation-2026-08-31'
        read = lambda name: json.loads((run / (name + '.json')).read_text())
        m2, m4, graph = read('terminology-proposals'), read('attribute-bindings'), read('route-observations')
        registry = json.loads((ROOT / 'fixtures/mep/m_and_p_coordination/m_and_p_coordination.sheet-registry.json').read_text())
        frozen_m4_hash = _sha256(m4)
        equipment, equipment_replay, _ = load_equipment_identity_inputs(
            ROOT / 'output/mep-equipment-2d-pilot-2026-08-31/equipment-2d-identities.json', terminology=m2)
        fitting, fitting_replay, _ = load_fitting_identity_inputs(
            ROOT / 'output/mep-fitting-hypotheses-2026-08-31/pilot.json.gz')
        companion = build_projected_identity_bindings(attribute_bindings=m4, terminology=m2,
            equipment_identity=equipment, equipment_replay=equipment_replay,
            fitting_hypotheses=fitting, fitting_replay=fitting_replay)
        self.assertEqual(companion['summary']['projected_equipment_identity_count'], 2)
        self.assertEqual(companion['summary']['projected_fitting_connection_count'], 1)
        self.assertEqual(companion['summary']['equipment_port_binding_count'], 0)
        self.assertIsNone(companion['summary']['physical_item_count'])

        inventory = build_mep_hvac_inventory(terminology=m2, attribute_bindings=m4,
            projected_identity_bindings=companion)
        identified = [row for row in inventory['items'] if row['state'] == 'identified_2d']
        self.assertEqual({row['candidate']['tag'] for row in identified}, {'HUH-13', 'HUH-14'})
        self.assertIn(_sha256(companion), inventory['input_payload_sha256'].values())
        for row in identified:
            self.assertEqual(row['epistemic_state'], 'inferred')
            self.assertTrue(row['authority']['projected_body_tag_identity_established'])
            self.assertFalse(row['authority']['equipment_port_binding_established'])
            self.assertFalse(row['authority']['physical_item_identity_established'])
            self.assertFalse(row['authority']['physical_count_established'])
            self.assertFalse(row['quantity_eligible'])
            self.assertIn('equipment_ports_unresolved', row['unresolved_reasons'])

        trace_run = ROOT / 'output/mep-projected-trace-completion-2026-08-31/final'
        trace = json.loads((trace_run / 'trace-completion.json').read_text())
        queries = json.loads(gzip.decompress((trace_run / 'trace-source-queries.json.gz').read_bytes()))
        network_inputs = {'sheet_registry': registry, 'route_graph': graph,
            'attribute_bindings': m4, 'cross_sheet_runs': read('cross-sheet-runs'),
            'boundary_connections': read('outlined-route-connections'),
            'trace_completion': trace, 'trace_replay': {'graph': graph, 'composites': read('outlined-route-composites'),
                'boundary_connections': read('outlined-route-connections'), 'source_queries': queries['source_queries']},
            'fitting_replay': {'payload': fitting, **fitting_replay}}
        network = build_mep_networks(**network_inputs, projected_identity_bindings=companion)
        accepted_native = {r['source_boundary_connection_ref'] for r in network['junctions']
                           if r['state'] == 'accepted' and r.get('source_boundary_connection_ref')}
        self.assertEqual(accepted_native, {r['id'] for r in read('outlined-route-connections')['connections'] if r['state'] == 'accepted'})
        self.assertEqual(len(accepted_native), 23)
        previous = json.loads((ROOT / 'output/mep-drawing-interpretation-project-2026-08-31/network-hierarchy.json').read_text())
        suppressed = {r['source_boundary_connection_ref'] for r in previous['junctions']
                      if r['state'] != 'accepted' and r.get('source_boundary_connection_ref') in accepted_native}
        self.assertEqual(len(suppressed), 14)
        self.assertEqual(network['summary']['segment_count'], 283)
        self.assertEqual(network['summary']['complete_projected_scope_count'], 1)
        complete = [r for r in network['segments'] if r['projected_scope_complete']]
        self.assertEqual(len(complete), 1)
        parent = next(r for r in network['runs'] if complete[0]['id'] in r['segment_refs'])
        self.assertTrue(parent['uncovered_segment_refs'])
        self.assertFalse(parent['complete_trace_established'])
        self.assertEqual(network['summary']['projected_fitting_branch_count'], 1)
        geometry_only = [row for row in network['segments'] if row.get('geometry_only')]
        branch = next(row for row in network['junctions'] if row['relation_type'] == 'projected_fitting_body_branch')
        branch_segments = [row for row in geometry_only if row['id'] in branch['segment_refs']]
        self.assertEqual(len(branch_segments), 3)
        for row in branch_segments:
            self.assertEqual(row['unobserved_attributes'], ['system', 'size', 'elevation'])
            self.assertTrue(row['source_occurrence_refs'])
            self.assertEqual(row['geometric_eligibility']['basis'], 'replayed_m3_5_outline')
            self.assertFalse(row['physical_continuation_established'])
            self.assertFalse(row['quantity_eligible'])
        branch = next(row for row in network['junctions'] if row['relation_type'] == 'projected_fitting_body_branch')
        self.assertEqual(branch['state'], 'accepted')
        self.assertEqual(branch['epistemic_state'], 'inferred')
        self.assertEqual(len(branch['port_segment_bindings']), 3)
        self.assertEqual(set(branch['segment_refs']), {row['id'] for row in branch_segments})
        self.assertFalse(branch['automatic_attribute_propagation'])
        self.assertFalse(branch['physical_continuation_established'])
        self.assertFalse(branch['run_pass_through'])
        self.assertEqual(network['summary']['complete_run_count'], 0)
        self.assertTrue(any(set(row['segment_refs']) == set(branch['segment_refs']) for row in network['networks']))

        # Replaying M7 after the presentation change yields the exact old catalog.
        catalog = build_mep_item_catalog(sheet_registry=registry, attribute_bindings=m4,
                                        bounded_local_3d=read('bounded-local-3d'))
        self.assertEqual(_sha256(catalog), _sha256(read('item-catalog')))
        self.assertTrue(all(row['item_type'] == 'routed_material' for row in catalog['item_occurrences']))
        self.assertFalse(catalog['physical_items'])
        self.assertFalse(catalog['calculated_discrete_counts'])
        self.assertEqual(_sha256(m4), frozen_m4_hash)

        forged = deepcopy(companion)
        next(row for row in forged['fitting_bindings'] if row['accepted_projected_connection'])['centreline_junction_display'][0] += .25
        with self.assertRaisesRegex(ValueError, 'fitting bindings do not replay'):
            build_mep_networks(**network_inputs, projected_identity_bindings=forged)
        with self.assertRaisesRegex(ValueError, 'independent native replay inputs'):
            build_projected_identity_bindings(attribute_bindings=m4, terminology=m2, equipment_identity=equipment)
        for key in ['physical_item_identity_established', 'physical_placement_established']:
            forged = deepcopy(companion)
            next(row for row in forged['equipment_bindings'] if row['state'] == 'accepted')['authority'][key] = True
            with self.assertRaisesRegex(ValueError, 'invalid ownership or authority'):
                build_mep_hvac_inventory(terminology=m2, attribute_bindings=m4, projected_identity_bindings=forged)


if __name__ == '__main__':
    unittest.main()
