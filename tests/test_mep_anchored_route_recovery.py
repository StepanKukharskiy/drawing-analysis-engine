import copy
import unittest

from src.drawing_engine.disciplines.mep.mep_anchored_route_recovery import (
    build_anchored_route_recovery, validate_anchored_route_recovery,
)


PAGE = "page.5"


def fragment(ref, source, start, end, stroke):
    return {
        "id": ref, "source_primitive_ref": source,
        "geometry": {"points_display": [start, end]},
        "style": {"width_display_points": .48, "dash_pattern": "[] 0",
                  "fill": None, "stroke": stroke},
    }


def composite(ref, members, sources, start, end):
    return {
        "id": ref, "page_ref": PAGE, "state": "accepted",
        "member_fragment_refs": members,
        "member_source_primitive_refs": sources,
        "supporting_closure_fragment_refs": [],
        "derived_geometry": {
            "centreline_points_display": [start, end],
            "corridor_width_display_points": 2.0,
            "projected_path_display_points": 10.0,
        },
    }


class AnchoredRouteRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.route_page = {"page_ref": PAGE, "fragments": [
            fragment("a1", "s1", [0, -1], [10, -1], [1, 0, 0]),
            fragment("a2", "s2", [0, 1], [10, 1], [1, 0, 0]),
            fragment("b1", "s3", [10, -1], [20, -1], [0, 0, 1]),
            fragment("b2", "s4", [10, 1], [20, 1], [0, 0, 1]),
        ]}
        self.composites = [
            composite("ca", ["a1", "a2"], ["s1", "s2"], [0, 0], [10, 0]),
            composite("cb", ["b1", "b2"], ["s3", "s4"], [10, 0], [20, 0]),
        ]
        self.near = {"exact_joins": [
            {"id": "j1", "fragment_refs": ["a1", "b1"]},
            {"id": "j2", "fragment_refs": ["a2", "b2"]},
        ], "uniquely_certified_near_joins": [], "ambiguous_near_join_candidates": []}
        self.manifest = {
            "native_descriptor_pack": {"record_count": 100},
            "source_disposition_pack": {"disposition_counts": {
                "route_evidence": 4, "unresolved_drawing_view_geometry": 96,
            }},
        }

    def build(self, relations=()):
        return build_anchored_route_recovery(
            page_ref=PAGE, route_page=self.route_page,
            accepted_composites=self.composites, near_joins=self.near,
            denominator_manifest=self.manifest, m4_relations=relations)

    def test_two_sided_exact_join_recovers_one_component_before_m4(self):
        payload = self.build()
        self.assertEqual(1, len(payload["accepted_transitions"]))
        self.assertEqual(1, len(payload["geometry_components"]))
        self.assertEqual(4, payload["geometry_components"][0]["recovered_denominator_rows"])
        self.assertEqual("unknown", payload["post_geometry_M4_bindings"][0]["state"])
        self.assertEqual([], validate_anchored_route_recovery(payload))

    def test_colour_does_not_control_connectivity_and_m4_is_post_geometry(self):
        relation = {
            "id": "m4.system", "page_ref": PAGE, "state": "accepted",
            "relation_type": "route_system", "target_refs": ["ca"],
            "candidate": {"kind": "heating_hot_water_supply", "raw_text": "HHWS"},
        }
        without = self.build()
        with_m4 = self.build([relation])
        self.assertEqual(without["geometry_components_sha256"],
                         with_m4["geometry_components_sha256"])
        self.assertEqual("identified", with_m4["post_geometry_M4_bindings"][0]["state"])
        self.assertFalse(with_m4["geometry_input"]["colour_used_to_build_graph"])

    def test_branch_and_ambiguous_near_join_do_not_connect(self):
        branch = copy.deepcopy(self.near)
        branch["exact_joins"][0]["fragment_refs"].append("competitor")
        payload = build_anchored_route_recovery(
            page_ref=PAGE, route_page=self.route_page,
            accepted_composites=self.composites, near_joins=branch,
            denominator_manifest=self.manifest)
        self.assertEqual([], payload["accepted_transitions"])
        self.assertEqual(2, len(payload["geometry_components"]))
        self.assertIn("branch_vertex", {
            row["reason"] for row in payload["excluded_competing_transitions"]})


if __name__ == "__main__":
    unittest.main()
