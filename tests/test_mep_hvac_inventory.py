import copy
import unittest

from test_mep_attribute_binding import (_route_graph, _segment, _m2, _observation,
    _proposal, _binding, _endpoint)
from src.drawing_engine.disciplines.mep.mep_attribute_binding import build_mep_attribute_bindings
from src.drawing_engine.disciplines.mep.mep_hvac_inventory import build_mep_hvac_inventory
from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256


class MepHvacInventoryTest(unittest.TestCase):
    def test_unresolved_body_search_is_inspectable_without_creating_identity_or_ports(self):
        graph = _route_graph(_segment(0, (0, 0), (20, 0)))
        m2 = _m2(_observation('equipment', 'HUH-9'))
        bindings = build_mep_attribute_bindings(terminology_proposals=m2, route_graph=graph, binding_evidence=[])
        companion = {'layer': 'mep_projected_identity_bindings', 'document': bindings['document'],
            'input_payload_sha256': {'attribute-bindings': _sha256(bindings), 'terminology-proposals': _sha256(m2)},
            'equipment_bindings': [{'id': 'unresolved-body', 'proposal_ref': m2['proposals'][0]['id'],
                'state': 'abstained', 'reasons': ['competing_native_bodies']}]}
        result = build_mep_hvac_inventory(terminology=m2, attribute_bindings=bindings, projected_identity_bindings=companion)
        row = result['items'][0]
        self.assertEqual(row['projected_identity_assessment_refs'], ['unresolved-body'])
        self.assertIn('competing_native_bodies', row['unresolved_reasons'])
        self.assertEqual(row['state'], 'unresolved')
        self.assertEqual(row['target_refs'], [])
        self.assertNotIn('projected_identity_relation_refs', row)
        self.assertFalse(row['authority']['equipment_port_binding_established'])
        self.assertFalse(row['authority']['physical_item_identity_established'])

    def test_tags_are_review_observations_not_equipment_and_duplicate_tags_stay_separate(self):
        graph = _route_graph(_segment(0, (0, 0), (20, 0)))
        m2 = _m2(*[_observation(str(i), tag, bbox_display=[i, 0, i + 1, 1],
            region_role="drawing", confidence=0.82, evidence_channels=["ocr_text"])
            for i, tag in enumerate(["AHU-1", "AHU-1", "FCU-2", "HUH-3", "VFD-1"])])
        bindings = build_mep_attribute_bindings(terminology_proposals=m2, route_graph=graph, binding_evidence=[])
        frozen = copy.deepcopy(m2)
        result = build_mep_hvac_inventory(terminology=m2, attribute_bindings=bindings)
        self.assertEqual(len(result["items"]), 5)
        self.assertEqual(len({r["id"] for r in result["items"]}), 5)
        self.assertTrue(all(r["state"] == "unresolved" for r in result["items"]))
        self.assertFalse(next(r for r in result["items"] if r["candidate"]["tag"] == "VFD-1")["class_supported"])
        self.assertEqual(result["source_observations"], sorted(m2["source_observations"], key=lambda r: r["id"]))
        self.assertEqual(m2, frozen)
        self.assertIsNone(result["summary"]["physical_item_count"])

    def test_existing_m4_port_duct_and_damper_bindings_are_projected_without_new_authority(self):
        graph = _route_graph(_segment(0, (0, 0), (20, 0)))
        fragment = graph["pages"][0]["fragments"][0]
        endpoint = _endpoint(graph["pages"][0], fragment["id"], "start")
        m2 = _m2(_observation("equipment", "AHU-1"), _observation("duct", "600 X 300 MM"),
            _observation("damper", symbol_kind="fire_damper", evidence_channels=["native_vector_geometry"]))
        bindings = build_mep_attribute_bindings(terminology_proposals=m2, route_graph=graph,
            binding_evidence=[_binding("port", _proposal(m2, "equipment"), "route_endpoint", [endpoint["id"]],
                explicit_port_connectivity=True, port_geometry_evidence_refs=[endpoint["id"]]),
                *[_binding(name, _proposal(m2, name), "route_fragment", [fragment["id"]]) for name in ("duct", "damper")]])
        result = build_mep_hvac_inventory(terminology=m2, attribute_bindings=bindings)
        self.assertEqual(result["summary"]["state_counts"], {"bound_to_m4_target": 3})
        self.assertTrue(all(not r["authority"]["physical_item_identity_established"] for r in result["items"]))
        duct = next(r for r in result["items"] if r["item_type"] == "duct")
        self.assertEqual((duct["candidate"]["width"], duct["candidate"]["height"]), (600, 300))

    def test_legend_and_unknown_applicability_cannot_promote_and_stale_m2_is_rejected(self):
        graph = _route_graph(_segment(0, (0, 0), (20, 0)))
        m2 = _m2(_observation("legend", "FCU-2", region_role="legend"),
                 _observation("unknown", "600 X 300 MM", region_role="unknown"),
                 _observation("schedule", "AHU-99", region_role="schedule"))
        bindings = build_mep_attribute_bindings(terminology_proposals=m2, route_graph=graph, binding_evidence=[])
        result = build_mep_hvac_inventory(terminology=m2, attribute_bindings=bindings)
        self.assertEqual(len(result["items"]), 2)
        self.assertTrue(all(r["state"] == "unresolved" for r in result["items"]))
        m2["source_observations"][0]["bbox_display"] = [0, 0, 10, 10]
        with self.assertRaisesRegex(ValueError, "frozen M2"):
            build_mep_hvac_inventory(terminology=m2, attribute_bindings=bindings)
