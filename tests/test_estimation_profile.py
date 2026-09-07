import copy
import inspect
import unittest

from src.drawing_engine.disciplines.rebar.estimation_profile import estimate_rebar_quantities, load_estimation_profile


def symbolic_solution(sharp_length=1402.132):
    return {
        "status": "convention_dependent",
        "topology": "open_rectangular_loop_with_two_diagonal_hooks",
        "sharp_vertex_length_mm": sharp_length,
        "centerline_radius_coefficient": 6.232076,
        "length_expression": f"{sharp_length} - 6.232076 * R_cl_mm",
    }


class EstimationProfileTest(unittest.TestCase):
    def setUp(self):
        self.profile = load_estimation_profile()
        self.groups = [
            {
                "id": "group.longitudinal",
                "quantity": {"value": 4, "state": "derived"},
                "topology": {"family": {"value": "straight"}},
                "bar_spec": {"diameter_mm": {"value": 16, "state": "convention_dependent"}},
                "fabrication": {"status": "resolved", "cutting_length_each_mm": 1000.0},
            },
            {
                "id": "group.transverse",
                "quantity": {"value": 10, "state": "derived"},
                "topology": {"family": {"value": "closed_polyline"}},
                "bar_spec": {"diameter_mm": {"value": 8, "state": "convention_dependent"}},
                "fabrication": {"status": "partial_equation", "cutting_length_each_mm": None},
            },
        ]
        self.details = {
            "details": [
                {"id": "detail.2", "fabrication_geometry_solution": symbolic_solution(1492.132)},
                {"id": "detail.9", "fabrication_geometry_solution": symbolic_solution()},
                {"id": "detail.7", "fabrication_geometry_solution": {"status": "unresolved"}},
            ],
            "physical_families": [
                {"id": "family.1", "mark": "1", "source_group_id": "group.longitudinal", "constraint_status": "resolved", "detail_ids": []},
                {"id": "family.2", "mark": "2", "source_group_id": "group.transverse", "constraint_status": "resolved", "detail_ids": ["detail.2"]},
                {"id": "family.9", "mark": "9", "constraint_status": "resolved", "detail_ids": ["detail.9"], "multiplicity": {"value": 2, "state": "derived"}},
                {"id": "family.7", "mark": "7", "constraint_status": "ambiguous", "detail_ids": ["detail.7"], "multiplicity": {"value": 1, "state": "derived"}},
            ],
        }

    def test_profile_is_internally_consistent(self):
        self.assertEqual(self.profile["id"], "universal_estimate_v1")
        self.assertEqual(self.profile["stirrup_tie_inside_bend_diameter_factor"], 4.0)
        self.assertEqual(self.profile["stirrup_tie_centerline_radius_factor"], 2.5)
        self.assertEqual(self.profile["waste_allowance_percent"], 0.0)

    def test_estimate_closes_symbolic_lengths_without_mutating_strict_inputs(self):
        original_groups = copy.deepcopy(self.groups)
        original_details = copy.deepcopy(self.details)

        result = estimate_rebar_quantities(self.groups, self.details, self.profile)

        self.assertEqual(self.groups, original_groups)
        self.assertEqual(self.details, original_details)
        self.assertEqual(result["status"], "estimated_partial")
        self.assertFalse(result["schedule_values_used"])
        rows = {row["mark"]: row for row in result["families"]}
        self.assertEqual(rows["2"]["centerline_bend_radius_mm"], 20.0)
        self.assertEqual(rows["2"]["fabrication_length_each_mm"], 1365.0)
        self.assertEqual(rows["9"]["diameter_state"], "assumed_unique_role_inheritance")
        self.assertEqual(rows["9"]["fabrication_length_each_mm"], 1275.0)
        self.assertEqual(rows["9"]["sensitivity"]["centerline_bend_radius_mm"], 24.0)
        self.assertGreater(result["totals"]["mass_kg"], 0)
        self.assertEqual(result["coverage"]["unresolved_family_marks"], ["7"])

    def test_no_object_or_file_dispatch_is_present(self):
        source = inspect.getsource(__import__("src.drawing_engine.disciplines.rebar.estimation_profile", fromlist=["*"])).lower()
        for forbidden in ("k1", "k7", ".pdf", "column", "beam", "slab"):
            self.assertNotIn(forbidden, source)

    def test_multi_object_sheet_does_not_inherit_unscoped_diameter(self):
        result = estimate_rebar_quantities(
            self.groups,
            self.details,
            self.profile,
            {"instances": [{"id": "object.1"}, {"id": "object.2"}]},
        )
        self.assertNotIn("9", result["coverage"]["estimated_family_marks"])
        self.assertIn("9", result["coverage"]["unresolved_family_marks"])
        self.assertTrue(result["validation"]["multi_object_inheritance_blocked_without_scope"])

    def test_profile_estimates_native_compound_callout_with_explicit_multiplier(self):
        details = {
            "details": [],
            "physical_families": [
                {
                    "id": "family.callout",
                    "mark": "12/594",
                    "constraint_status": "partial",
                    "detail_ids": [],
                    "multiplicity": {"value": 14, "state": "derived"},
                    "compound_callout_observation": {
                        "diameter_mm": 12,
                        "length_code": 594,
                        "basis": "native compound rebar callout outside a table grid",
                        "evidence_refs": ["mark.1"],
                    },
                }
            ],
        }

        result = estimate_rebar_quantities([], details, self.profile)

        row = result["families"][0]
        self.assertEqual(row["fabrication_length_each_mm"], 5940.0)
        self.assertEqual(row["fabrication_length_total_m"], 83.16)
        self.assertEqual(row["diameter_mm"], 12.0)
        self.assertIn("compound_callout_length_code_multiplier_mm", row["assumptions_used"])
        self.assertFalse(result["schedule_values_used"])


if __name__ == "__main__":
    unittest.main()
