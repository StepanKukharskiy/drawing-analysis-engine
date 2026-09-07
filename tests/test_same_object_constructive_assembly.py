import json
from pathlib import Path
import unittest

from src.drawing_engine.disciplines.concrete.same_object_constructive_assembly import assemble_same_object_construction
from src.drawing_engine.disciplines.concrete.constructive_union_enumerator import enumerate_constructive_union_hypotheses


ROOT = Path(__file__).resolve().parents[1]
ENGINEERING = ROOT / "output" / "object_agnostic" / "candidate-08-staircase-page.engineering-graph.json"


def _page():
    return json.loads(ENGINEERING.read_text())["pages"][0]


class SameObjectConstructiveAssemblyTest(unittest.TestCase):
    def test_adapter_enumerates_arbitrary_proposal_band_bijections_without_roles(self):
        proposals = [
            {"id": f"proposal.{index}", "evidence_refs": [f"proposal.evidence.{index}"]}
            for index in range(3)
        ]
        assignments = [
            {"id": f"assignment.part.{index}", "proposal_ref": proposal["id"], "classified_interfaces": [], "interface_options": []}
            for index, proposal in enumerate(proposals)
        ]
        result = assemble_same_object_construction(
            {"scope_results": [{"scope_ref": "section.scope", "flight_boundary_proposals": proposals}]},
            {"scope_results": [{"global_assignment_candidates": [{"id": "assignment.global", "assignments": assignments}]}]},
            {
                "plan_band_certificate": {
                    "bands": [
                        {"id": f"band.{index}", "sweep_width_mm": 900.0, "evidence_refs": [f"band.evidence.{index}"]}
                        for index in range(3)
                    ]
                }
            },
            {
                "physical_object_scope": {
                    "id": "physical.scope",
                    "record_type": "physical_object_scope",
                    "state": "proposed",
                    "evidence_refs": ["scope.evidence"],
                },
                "landing_construction_region": {
                    "id": "region.core",
                    "record_type": "construction_region",
                    "construction_role": "core",
                    "state": "incomplete",
                    "quantity_eligible": False,
                    "evidence_refs": ["core.evidence"],
                },
                "plan_footprint_certificate": {"title_scope_ref": "plan.scope"},
                "separate_object_interface_hypotheses": [],
            },
            page_number=1,
        )

        self.assertEqual(result["summary"]["local_assignment_count"], 1)
        self.assertEqual(result["summary"]["placement_alternative_count"], 6)
        self.assertEqual(result["summary"]["construction_region_count"], 4)
        self.assertEqual(
            {item["construction_role"] for item in result["construction_regions"][:-1]},
            {"bounded_profile_sweep"},
        )

    def test_candidate_08_preserves_five_global_alternatives_without_additive_regions(self):
        page = _page()
        result = page["same_object_constructive_assembly"]

        self.assertEqual(result["status"], "accepted_invariant_equivalence_class")
        self.assertIsNone(result["reason_code"])
        self.assertTrue(result["quantity_eligible"])
        self.assertEqual(result["summary"]["local_assignment_count"], 5)
        self.assertEqual(result["summary"]["placement_alternative_count"], 256)
        self.assertEqual(result["summary"]["construction_region_count"], 3)
        self.assertEqual(result["summary"]["temporary_closed_boundary_hypothesis_count"], 10)
        self.assertEqual(result["summary"]["temporary_region_materialization_count"], 768)
        self.assertEqual(result["summary"]["materialized_union_hypothesis_count"], 72)
        self.assertEqual(result["summary"]["pre_kernel_rejection_count"], 184)
        self.assertEqual(result["summary"]["hard_pre_kernel_rejection_count"], 184)
        self.assertEqual(result["summary"]["certified_elimination_count"], 0)
        self.assertEqual(result["summary"]["unresolved_live_alternative_count"], 0)
        self.assertEqual(result["summary"]["kernel_replay_count"], 72)
        self.assertEqual(result["summary"]["step4_same_object_survivor_count"], 6)
        self.assertEqual(result["summary"]["invariant_equivalence_member_count"], 6)
        self.assertEqual(len(result["assembly_alternatives"]), 256)
        self.assertEqual(
            sum(bool(item["pre_kernel_rejection_reasons"]) for item in result["assembly_alternatives"]),
            184,
        )
        self.assertEqual(
            sum(item["step4_same_object_union_replayed"] for item in result["assembly_alternatives"]),
            72,
        )
        failures = [
            item
            for item in result["assembly_alternatives"]
            if item["step4_status"] == "reclosed_fail"
        ]
        self.assertEqual(len(failures), 66)
        self.assertEqual(
            len(
                {
                    round(
                        replay["kernel_result"]["volume_validation"][
                            "analytic_union_mm3"
                        ],
                        6,
                    )
                    for replay in result["kernel_replays"]
                    if replay["status"] == "accepted"
                }
            ),
            1,
        )
        self.assertTrue(all(item["step4_reasons"] for item in failures))
        self.assertTrue(
            all(
                item["step4_residual_diagnostics"]["supplied_view_reprojections"]
                for item in failures
            )
        )
        self.assertEqual(len(result["temporary_closed_boundary_hypotheses"]), 10)
        self.assertEqual(len(result["materialized_search_hypotheses"]), 256)
        self.assertEqual(len(result["profile_to_sweep_transform_alternatives"]), 160)
        self.assertEqual(len(result["profile_to_band_transform_alternatives"]), 160)
        self.assertEqual(len(result["assignment_specific_landing_seam_polygons"]), 144)
        self.assertEqual(len(result["native_projection_evidence"]), 32)
        self.assertEqual(len(result["analytic_union_certificates"]), 256)
        self.assertTrue(
            all(
                item["derived_from_generated_geometry"] is False
                and item["alternative_independent_of_transform"] is True
                for item in result["native_projection_evidence"]
            )
        )
        folded = result["folded_plan_topology_certificate"]
        self.assertEqual(folded["native_outline_drawing_ref"], "drawing[3]")
        self.assertTrue(folded["flight_bands_share_longitudinal_interval"])
        self.assertTrue(folded["landing_connects_same_band_endpoint"])
        self.assertEqual(
            {item["longitudinal_sign"] for item in result["profile_to_sweep_transform_alternatives"]},
            {-1, 1},
        )
        self.assertEqual(
            {item["shared_coordinate_reflection_sign"] for item in result["profile_to_sweep_transform_alternatives"]},
            {-1, 1},
        )
        self.assertEqual(result["physical_object_scope"]["record_type"], "physical_object_scope")
        self.assertEqual(
            {item["record_type"] for item in result["construction_regions"]},
            {"construction_region"},
        )
        self.assertTrue(all(not item["quantity_eligible"] for item in result["construction_regions"]))
        self.assertTrue(all(item["record_type"] == "internal_seam_hypothesis" for item in result["internal_seam_hypotheses"]))
        self.assertTrue(all(item["record_type"] == "external_boundary_hypothesis" for item in result["external_boundary_hypotheses"]))
        self.assertEqual(len(result["separate_object_interface_hypotheses"]), 1)
        self.assertIsNone(result["step4_same_object_union"])
        self.assertTrue(result["contract"]["final_union_volume_only"])
        self.assertTrue(result["contract"]["unevaluated_hypotheses_are_not_ambiguity"])
        self.assertTrue(result["contract"]["same_object_step4_union_invoked"])
        self.assertTrue(result["contract"]["temporary_boundaries_are_search_inputs_only"])
        self.assertTrue(
            result["contract"][
                "generated_mesh_projection_cannot_supply_expected_projection"
            ]
        )
        self.assertTrue(result["contract"]["analytic_union_is_independent_of_generated_mesh"])
        self.assertTrue(result["contract"]["construction_region_seam_graph_connected_before_step4"])
        self.assertTrue(result["contract"]["assigned_interface_caps_require_coincident_faces"])
        self.assertTrue(result["contract"]["overlap_union_requires_independent_authorization"])
        certificate = result["invariant_union_equivalence_certificate"]
        self.assertEqual(certificate["state"], "accepted")
        self.assertEqual(len(certificate["pairwise_checks"]), 15)
        self.assertTrue(all(item["status"] == "pass" for item in certificate["pairwise_checks"]))
        self.assertEqual(certificate["preserved_transform_alternative_count"], 6)
        self.assertAlmostEqual(
            result["invariant_concrete_volume_candidate"]["value_m3"],
            1.7726147823710516,
        )
        self.assertEqual(
            result["placement_unresolved_union_equivalence_class"][
                "physical_placement_state"
            ],
            "resolved_in_directed_plan_gauge",
        )
        self.assertEqual(result["plan_direction_evidence"]["state"], "accepted")
        self.assertEqual({tuple(c["reflection_signs_xyz"]) for c in certificate["pairwise_checks"]}, {(1, 1, 1)})
        bindings = [b for b in result["directed_profile_band_bindings"] if b["assignment_status"] == "accepted"]
        self.assertEqual(len(bindings), 12)
        self.assertTrue(all(b["direction_status"] == "pass" and b["one_to_one_region_binding"] for b in bindings))
        self.assertEqual(len({(b["band_ref"], tuple(b["section_profile_proposal_refs"])) for b in bindings}), 2)
        self.assertTrue(all(len(b["section_profile_proposal_refs"]) == 1 for b in bindings))

    def test_candidate_08_directed_bindings_replay_without_source_pdf(self):
        assembly = _page()["same_object_constructive_assembly"]
        materialized = assembly["materialized_search_hypotheses"]
        replay = enumerate_constructive_union_hypotheses(
            [assembly["search_physical_object_scope"]],
            [region for h in materialized for region in h["construction_regions"]],
            [h["placement_alternative"] for h in materialized],
            [seam for h in materialized for seam in h["internal_seams"]],
            assembly["native_projection_evidence"],
        )
        self.assertEqual(replay["status"], assembly["status"])
        self.assertEqual(replay["summary"]["kernel_replay_count"], 72)
        self.assertEqual(replay["summary"]["survivor_count"], 6)
        self.assertEqual(replay["invariant_union_equivalence_certificate"], assembly["invariant_union_equivalence_certificate"])


if __name__ == "__main__":
    unittest.main()
