import copy
import inspect
import json
from pathlib import Path
import unittest

from src.drawing_engine.disciplines.concrete.physical_object_quantity_promotion import (
    aggregate_calculated_concrete_quantities,
    promote_constructive_union_physical_object,
)
import src.drawing_engine.disciplines.concrete.physical_object_quantity_promotion as promotion_module


ROOT = Path(__file__).resolve().parents[1]
ENGINEERING = ROOT / "output" / "object_agnostic" / "candidate-08-staircase-page.engineering-graph.json"


def _inputs():
    page = json.loads(ENGINEERING.read_text())["pages"][0]
    assembly = copy.deepcopy(page["same_object_constructive_assembly"])
    assembly["physical_object_scope"]["state"] = (
        "proposed_monolithic_physical_object_scope"
    )
    return assembly, copy.deepcopy(page["view_frame_graph"]), copy.deepcopy(
        page["banded_plan_sweep_evidence"]
    )


class PhysicalObjectQuantityPromotionTest(unittest.TestCase):
    def test_promotion_module_has_no_drawing_or_object_class_dispatch(self):
        source = inspect.getsource(promotion_module).lower()
        for forbidden in ("candidate-08", "stair", "flight", "landing", "beam", "975", "275"):
            self.assertNotIn(forbidden, source)

    def test_invariant_union_promotes_one_relative_object_and_one_quantity(self):
        result = promote_constructive_union_physical_object(*_inputs(), page_number=1)

        self.assertEqual(result["status"], "accepted")
        self.assertTrue(result["quantity_eligible"])
        physical_object = result["accepted_physical_object"]
        self.assertEqual(physical_object["record_type"], "physical_object")
        self.assertTrue(
            physical_object["physical_object_scope"]["accepted_physical_object_scope"]
        )
        self.assertFalse(
            physical_object["physical_object_scope"]["absolute_orientation_resolved"]
        )
        self.assertEqual(len(result["calculated_concrete_quantities"]), 1)
        quantity = result["calculated_concrete_quantities"][0]
        self.assertAlmostEqual(quantity["net_concrete_m3"], 1.7726147823710516)
        self.assertFalse(quantity["construction_regions_aggregated_separately"])
        self.assertFalse(quantity["equivalence_members_aggregated_separately"])
        self.assertFalse(quantity["separate_objects_included"])
        self.assertEqual(
            result["canonical_3d_preview"]["label"],
            "absolute orientation unresolved",
        )
        self.assertEqual(
            result["canonical_3d_preview"]["mesh_validation"]["status"], "pass"
        )
        replay = promote_constructive_union_physical_object(*_inputs(), page_number=1)
        self.assertEqual(result["input_sha256"], replay["input_sha256"])
        self.assertEqual(
            result["calculated_concrete_quantities"],
            replay["calculated_concrete_quantities"],
        )

    def test_scope_view_pair_mismatch_blocks_promotion(self):
        assembly, graph, source = _inputs()
        source["search_physical_object_scope"]["section_view_id"] = "wrong.scope"

        result = promote_constructive_union_physical_object(
            assembly, graph, source, page_number=1
        )

        self.assertEqual(result["status"], "insufficient_constraints")
        self.assertEqual(result["reason_code"], "unique_relation_driven_scope_unresolved")
        self.assertEqual(result["calculated_concrete_quantities"], [])

    def test_missing_directed_replay_blocks_promotion(self):
        assembly, graph, source = _inputs()
        member = next(r for r in assembly["kernel_replays"] if r["status"] == "accepted")
        plan = next(v for v in member["kernel_result"]["supplied_view_reprojections"] if v["projection_role"] == "plan")
        plan["directed_surface_path_reprojections"] = []
        result = promote_constructive_union_physical_object(assembly, graph, source, page_number=1)
        self.assertEqual(result["reason_code"], "member_directed_projection_replay_unresolved")
        self.assertFalse(result["quantity_eligible"])

    def test_incomplete_competitor_replay_blocks_promotion(self):
        assembly, graph, source = _inputs()
        assembly["summary"]["unresolved_live_alternative_count"] = 1

        result = promote_constructive_union_physical_object(
            assembly, graph, source, page_number=1
        )

        self.assertEqual(
            result["reason_code"],
            "competing_alternatives_not_fully_replayed_or_eliminated",
        )

    def test_unequal_equivalence_member_volumes_block_promotion(self):
        assembly, graph, source = _inputs()
        member_ref = assembly["invariant_union_equivalence_certificate"][
            "member_hypothesis_refs"
        ][0]
        replay = next(
            item
            for item in assembly["kernel_replays"]
            if item["constructive_union_hypothesis_ref"] == member_ref
        )
        replay["kernel_result"]["volume_validation"]["analytic_union_mm3"] += 10_000.0

        result = promote_constructive_union_physical_object(
            assembly, graph, source, page_number=1
        )

        self.assertIn(
            result["reason_code"],
            {
                "member_analytic_mesh_volume_disagreement",
                "equivalence_member_volumes_unequal",
            },
        )

    def test_axis_permutation_is_forbidden_even_when_certificate_says_pass(self):
        assembly, graph, source = _inputs()
        assembly["invariant_union_equivalence_certificate"]["pairwise_checks"][0][
            "allowed_transform_kind"
        ] = "axis_permutation_and_translation"

        result = promote_constructive_union_physical_object(
            assembly, graph, source, page_number=1
        )

        self.assertEqual(result["reason_code"], "forbidden_equivalence_transform")

    def test_equivalence_members_cannot_be_double_counted(self):
        result = promote_constructive_union_physical_object(*_inputs(), page_number=1)
        quantity = result["calculated_concrete_quantities"][0]

        with self.assertRaisesRegex(ValueError, "double-count"):
            aggregate_calculated_concrete_quantities([quantity, copy.deepcopy(quantity)])

        member = copy.deepcopy(quantity)
        member["id"] = "calculated_concrete_quantity.member"
        member["physical_object_ref"] = "equivalence.member"
        member["source_kind"] = "equivalence_class_member"
        with self.assertRaisesRegex(ValueError, "cannot be aggregated"):
            aggregate_calculated_concrete_quantities([quantity, member])

    def test_landing_or_flight_region_cannot_enter_physical_object_aggregation(self):
        result = promote_constructive_union_physical_object(*_inputs(), page_number=1)
        quantity = result["calculated_concrete_quantities"][0]
        region = copy.deepcopy(quantity)
        region["id"] = "calculated_concrete_quantity.construction_region"
        region["physical_object_ref"] = "construction_region.landing"
        region["aggregation_scope"] = "construction_region"
        region["construction_regions_aggregated_separately"] = True

        with self.assertRaisesRegex(ValueError, "cannot be aggregated"):
            aggregate_calculated_concrete_quantities([quantity, region])


if __name__ == "__main__":
    unittest.main()
