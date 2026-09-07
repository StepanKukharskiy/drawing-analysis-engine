import copy
import json
from pathlib import Path
import unittest

from src.drawing_engine.disciplines.concrete.landing_component_reconstruction import reconstruct_landing_component


ROOT = Path(__file__).resolve().parents[1]
ENGINEERING = ROOT / "output" / "object_agnostic" / "candidate-08-staircase-page.engineering-graph.json"


def _page():
    return json.loads(ENGINEERING.read_text())["pages"][0]


def _reconstruct(page):
    return reconstruct_landing_component(
        page["dimension_ownership"],
        page["dimension_adjudication"],
        page["title_anchored_view_segmentation"],
        page["title_scope_local_dimension_reclosure"],
        page["open_structural_boundary_assembly"],
        page["flight_interface_closure"],
        page["flight_profile_pair_certification"],
        page["banded_plan_sweep_evidence"],
        page["clear_span_prism_reconstruction"],
        page_number=1,
    )


class LandingComponentReconstructionTest(unittest.TestCase):
    def test_candidate_08_closes_quantity_ineligible_landing_construction_region(self):
        result = _reconstruct(_page())

        self.assertEqual(result["status"], "resolved_construction_region_pending_same_object_union")
        self.assertEqual(result["plan_footprint_certificate"]["plan_envelope_mm"], [1150.0, 2150.0])
        self.assertEqual(result["plan_footprint_certificate"]["clear_core_footprint_mm"], [975.0, 2150.0])
        self.assertEqual(result["section_thickness_certificate"]["thickness_mm"], 150.0)
        region = result["landing_construction_region"]
        self.assertEqual(region["record_type"], "construction_region")
        self.assertEqual(region["extents_xyz_mm"], [975.0, 2150.0, 150.0])
        self.assertAlmostEqual(region["analytic_region_volume"]["value_m3"], 0.3144375)
        self.assertFalse(region["analytic_region_volume"]["additive_physical_quantity"])
        self.assertEqual(len(result["internal_seam_hypotheses"]), 2)
        self.assertTrue(all(not item["quantity_eligible"] for item in result["internal_seam_hypotheses"]))
        self.assertEqual(len(result["separate_object_interface_hypotheses"]), 1)
        beam_interface = result["separate_object_interface_hypotheses"][0]
        self.assertEqual(beam_interface["state"], "resolved_relative_contact")
        self.assertEqual(beam_interface["actual_contact_area_mm2"], 322500.0)
        self.assertEqual(beam_interface["volumetric_overlap_mm3"], 0.0)
        self.assertFalse(beam_interface["monolithic_identity_proven"])
        placement = result["beam_relative_placement_certificate"]
        self.assertEqual(placement["state"], "accepted_relative_orientation_unresolved")
        self.assertTrue(placement["relative_physical_placement_resolved"])
        self.assertEqual(
            placement["beam_transform"]["origin_xyz_mm"], [975.0, 2150.0, 0.0]
        )
        self.assertEqual(
            placement["beam_transform"]["local_axes_xyz"],
            {
                "x": [0.0, -1.0, 0.0],
                "y": [0.0, 0.0, -1.0],
                "z": [1.0, 0.0, 0.0],
            },
        )
        replay = result["landing_beam_step4_replay"]
        self.assertEqual(replay["status"], "accepted")
        self.assertEqual(replay["pair_validations"][0]["contact_area_mm2"], 322500.0)
        self.assertEqual(replay["pair_validations"][0]["overlap_volume_mm3"], 0.0)
        self.assertEqual(len(replay["supplied_view_reprojections"]), 2)
        self.assertTrue(
            all(view["status"] == "pass" for view in replay["supplied_view_reprojections"])
        )
        self.assertTrue(result["contract"]["landing_is_construction_region_of_stair"])
        self.assertFalse(result["contract"]["landing_is_separate_physical_component"])
        self.assertFalse(result["contract"]["landing_used_as_flight_closure_patch"])
        self.assertTrue(result["contract"]["beam_relative_placement_resolved"])
        self.assertTrue(result["contract"]["landing_beam_multi_component_step4_replayed"])
        self.assertFalse(result["contract"]["same_object_step4_replayed"])

    def test_unresolved_175_observation_is_required_only_as_interface_validation(self):
        page = _page()
        local = copy.deepcopy(page["title_scope_local_dimension_reclosure"])
        local["scope_results"][0]["unresolved_metric_observations"] = [
            item
            for item in local["scope_results"][0]["unresolved_metric_observations"]
            if item["value_mm"] != 175.0
        ]
        page["title_scope_local_dimension_reclosure"] = local

        result = _reconstruct(page)

        self.assertEqual(result["status"], "insufficient_constraints")
        self.assertEqual(result["reason_code"], "landing_interface_allowance_validation_unresolved")


if __name__ == "__main__":
    unittest.main()
