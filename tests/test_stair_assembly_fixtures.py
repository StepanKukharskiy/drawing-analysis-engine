import copy
import unittest

from src.drawing_engine.disciplines.concrete.multi_component_solid_kernel import solve_same_object_constructive_solid
from tests.stair_assembly_fixtures import same_object_stair_fixture


class SameObjectConstructiveSolidTest(unittest.TestCase):
    def _solve(self, **options):
        return solve_same_object_constructive_solid(**same_object_stair_fixture(**options))

    def test_three_regions_union_into_one_stair_object(self):
        record = self._solve()

        self.assertEqual(record["status"], "accepted")
        self.assertEqual(record["stage"], "generic_same_object_constructive_solid_kernel")
        self.assertEqual(len(record["construction_regions"]), 3)
        self.assertEqual(len(record["internal_seams"]), 2)
        self.assertEqual(record["external_boundary"]["validation"]["connected_solid_count"], 1)
        self.assertTrue(record["external_boundary"]["validation"]["watertight"])
        self.assertEqual(record["volume_validation"]["internal_overlap_deduction_mm3"], 0.0)
        self.assertTrue(record["contract"]["construction_regions_are_not_additive_components"])
        self.assertTrue(record["contract"]["internal_coincident_faces_removed_by_union"])
        self.assertFalse(record["contract"]["independent_physical_profile_closure_required"])
        self.assertTrue(all(item["temporary_bounded_region_for_union"] for item in record["construction_regions"]))
        self.assertTrue(all(not item["independent_physical_component_closure"] for item in record["construction_regions"]))
        self.assertEqual(
            {item["record_type"] for item in record["construction_regions"]},
            {"construction_region"},
        )
        self.assertEqual(
            {item["record_type"] for item in record["internal_seams"]},
            {"internal_seam"},
        )
        self.assertEqual(record["external_boundary"]["record_type"], "external_boundary")
        self.assertEqual(
            record["separate_object_interfaces"][0]["record_type"],
            "separate_object_interface",
        )

    def test_declared_overlapping_regions_use_union_not_additive_sum(self):
        record = self._solve(overlapping_regions=True)

        volume = record["volume_validation"]
        self.assertEqual(record["status"], "accepted")
        self.assertGreater(volume["internal_overlap_deduction_mm3"], 0.0)
        self.assertEqual(volume["analytic_union_mm3"], volume["mesh_union_mm3"])
        self.assertEqual({item["seam_kind"] for item in record["internal_seams"]}, {"overlap_union"})

    def test_unrecorded_same_object_seam_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unrecorded internal seam"):
            self._solve(omit_internal_seam=True)

    def test_wrong_additive_analytic_volume_fails_union_agreement(self):
        with self.assertRaisesRegex(ValueError, "analytic union and union mesh volumes disagree"):
            self._solve(overlapping_regions=True, wrong_analytic_union=True)

    def test_disconnected_regions_do_not_form_one_physical_object(self):
        with self.assertRaisesRegex(ValueError, "one connected physical object"):
            self._solve(disconnected=True)

    def test_final_union_reprojection_is_required(self):
        with self.assertRaisesRegex(ValueError, "supplied-view reprojection failed"):
            self._solve(wrong_plan_reprojection=True)

    def test_separate_object_interface_cannot_join_one_scope_to_itself(self):
        fixture = same_object_stair_fixture()
        bad = copy.deepcopy(fixture["separate_object_interfaces"][0])
        bad["object_scope_refs"] = [
            "physical_object_scope.synthetic_stair",
            "physical_object_scope.synthetic_stair",
        ]
        fixture["separate_object_interfaces"] = [bad]

        with self.assertRaisesRegex(ValueError, "two distinct physical scopes"):
            solve_same_object_constructive_solid(**fixture)


if __name__ == "__main__":
    unittest.main()
