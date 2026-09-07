import copy
import unittest

from src.drawing_engine.disciplines.mep.mep_bounded_local_3d import (
    _canonical_sha256,
    assemble_typed_duct_route,
)


def _inputs():
    route = {
        "schema_version": "0.1.0",
        "layer": "mep_projected_route_observations",
        "document": {"document_key": "synthetic-duct", "page_count": 1},
        "pages": [{
            "page_ref": "page.duct",
            "fragments": [],
            "vertices": [
                {"id": "vertex.taper", "point_display": [0.5, 1.0],
                 "endpoint_refs": ["endpoint.taper"], "fragment_refs": ["f2"]},
                {"id": "vertex.equipment", "point_display": [0.7, 2.0],
                 "endpoint_refs": ["endpoint.equipment"], "fragment_refs": ["f3"]},
            ],
            "endpoints": [
                {"id": "endpoint.taper", "fragment_ref": "f2",
                 "vertex_ref": "vertex.taper", "point_display": [0.5, 1.0]},
                {"id": "endpoint.equipment", "fragment_ref": "f3",
                 "vertex_ref": "vertex.equipment", "point_display": [0.7, 2.0]},
            ],
        }],
    }

    def composite(ref, fragment, points, width):
        return {
            "id": ref,
            "page_ref": "page.duct",
            "member_fragment_refs": [fragment],
            "member_source_primitive_refs": ["drawing." + fragment],
            "derived_geometry": {
                "centreline_points_display": points,
                "corridor_width_display_points": width,
            },
            "state": "accepted",
            "native_strokes_preserved": True,
            "quantity_eligible": False,
        }

    composites = {
        "schema_version": "0.1.0",
        "layer": "mep_outlined_route_composites",
        "m3_contract_ref": {"payload_sha256": _canonical_sha256(route)},
        "accepted_composites": [
            composite("c1", "f1", [[0, 0], [1, 0]], 1.0),
            composite("c2", "f2", [[1, 0], [1, 1]], 1.0),
            composite("c3", "f3", [[1, 1], [1, 2]], 0.6),
        ],
    }
    bindings = {
        "schema_version": "0.1.0",
        "layer": "mep_page_local_attribute_bindings",
        "m3_contract_ref": {"payload_sha256": _canonical_sha256(route)},
        "m3_5_contract_ref": {"payload_sha256": _canonical_sha256(composites)},
        "relations": [
            {"id": "relation.taper", "relation_type": "fitting", "state": "accepted",
             "target_refs": ["vertex.taper"],
             "candidate": {"kind": "reducer", "category": "fitting"}},
            {"id": "relation.equipment", "relation_type": "equipment_endpoint",
             "state": "accepted", "target_refs": ["endpoint.equipment"],
             "candidate": {"kind": "ahu", "tag": "AHU-1"}},
        ],
    }
    interfaces = [
        {"id": "interface.bend", "relation_type": "projected_native_bend",
         "state": "accepted", "search": {"complete": True},
         "port_refs": ["port.c1.1", "port.c2.0"],
         "source_primitive_refs": ["drawing.bend"],
         "ports": [
             {"id": "port.c1.1", "composite_ref": "c1", "end": 1,
              "sides": [[1, -0.5], [1, 0.5]]},
             {"id": "port.c2.0", "composite_ref": "c2", "end": 0,
              "sides": [[0.5, 0], [1.5, 0]]},
         ]},
        {"id": "interface.taper", "relation_type": "projected_collinear_boundary_join",
         "state": "accepted", "search": {"complete": True},
         "port_refs": ["port.c2.1", "port.c3.0"],
         "source_primitive_refs": ["drawing.taper"],
         "ports": [
             {"id": "port.c2.1", "composite_ref": "c2", "end": 1,
              "sides": [[0.5, 1], [1.5, 1]]},
             {"id": "port.c3.0", "composite_ref": "c3", "end": 0,
              "sides": [[0.7, 1], [1.3, 1]]},
         ]},
    ]
    return route, composites, bindings, interfaces


def _build(values=None):
    route, composites, bindings, interfaces = values or _inputs()
    return assemble_typed_duct_route(
        route_observations=route,
        outlined_route_composites=composites,
        attribute_bindings=bindings,
        route_composite_refs=["c1", "c2", "c3"],
        projected_interfaces=interfaces,
    )


class TypedDuctRouteAssemblyTest(unittest.TestCase):
    def test_chain_types_panels_bend_taper_equipment_and_open_end(self):
        result = _build()
        self.assertEqual(result["state"], "accepted")
        self.assertTrue(result["chain_integrity"]["complete_within_analysis_boundaries"])
        self.assertEqual(result["chain_integrity"]["panel_count"], 3)
        self.assertEqual(result["chain_integrity"]["internal_interface_count"], 2)
        self.assertEqual(result["chain_integrity"]["analysis_boundary_count"], 2)
        self.assertEqual(result["chain_integrity"]["ambiguous_endpoint_count"], 0)
        self.assertEqual(result["chain_integrity"]["ordered_panel_refs"],
                         ["c1", "c2", "c3"])
        self.assertEqual(len(result["chain_integrity"]
                             ["ordered_internal_interface_refs"]), 2)
        kinds = [row["kind"] for row in result["elements"]]
        self.assertEqual(kinds.count("straight"), 3)
        self.assertEqual(kinds.count("bend"), 1)
        self.assertEqual(kinds.count("taper"), 1)
        self.assertEqual(kinds.count("equipment_port"), 1)
        self.assertEqual(kinds.count("unresolved_boundary"), 1)
        taper = next(row for row in result["elements"] if row["kind"] == "taper")
        self.assertEqual(taper["m4_relation_refs"], ["relation.taper"])
        self.assertFalse(taper["attribute_propagation_across_interface"])
        self.assertFalse(result["physical_continuation_established"])
        self.assertFalse(result["quantity_eligible"])

    def test_ambiguous_interface_abstains_and_preserves_boundaries(self):
        values = _inputs()
        competing = copy.deepcopy(values[3][0])
        competing["id"] = "interface.competing"
        competing["relation_type"] = "projected_collinear_boundary_join"
        values[3].append(competing)
        result = _build(values)
        self.assertEqual(result["state"], "abstained")
        self.assertEqual(result["chain_integrity"]["ambiguous_endpoint_count"], 2)
        self.assertIn("typed_route_is_not_one_unambiguous_bounded_chain",
                      result["reasons"])
        ambiguous = [row for row in result["elements"]
                     if row.get("reason") == "ambiguous_interface_or_terminal"]
        self.assertEqual(len(ambiguous), 2)
        self.assertTrue(all(row["kind"] == "unresolved_boundary" for row in ambiguous))

    def test_unresolved_route_ends_are_analysis_boundaries_not_caps(self):
        values = _inputs()
        values[2]["relations"] = [values[2]["relations"][0]]
        result = _build(values)
        self.assertEqual(result["state"], "accepted")
        boundaries = [row for row in result["elements"]
                      if row["element_role"] == "analysis_boundary"]
        self.assertEqual([row["kind"] for row in boundaries],
                         ["unresolved_boundary", "unresolved_boundary"])
        self.assertTrue(all(row["physical_cap_established"] is False
                            and row["physical_continuation_established"] is False
                            for row in boundaries))

    def test_contract_mismatch_and_nonaccepted_panel_fail_closed(self):
        values = _inputs()
        values[0]["pages"][0]["page_ref"] = "changed"
        with self.assertRaisesRegex(ValueError, "exact frozen M3"):
            _build(values)

        values = _inputs()
        values[1]["accepted_composites"][0]["state"] = "abstained"
        values[2]["m3_5_contract_ref"]["payload_sha256"] = _canonical_sha256(values[1])
        result = _build(values)
        self.assertEqual(result["state"], "abstained")
        self.assertIn("selected_panel_lacks_accepted_m3_5_geometry", result["reasons"])


if __name__ == "__main__":
    unittest.main()
