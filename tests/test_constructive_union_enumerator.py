import copy
import inspect
import unittest

import src.drawing_engine.disciplines.concrete.constructive_union_enumerator as enumerator_module
from src.drawing_engine.disciplines.concrete.constructive_union_enumerator import enumerate_constructive_union_hypotheses
from src.drawing_engine.disciplines.concrete.invariant_union_equivalence import certify_invariant_union_equivalence
from tests.stair_assembly_fixtures import same_object_stair_fixture


def _inputs(**fixture_options):
    fixture = same_object_stair_fixture(**fixture_options)
    alternative = {
        "id": "constructive_placement_alternative.synthetic.1",
        "record_type": "constructive_placement_alternative",
        "physical_object_scope_ref": fixture["physical_object_scope"]["id"],
        "source_assignment_ref": "source_assignment.synthetic.1",
        "construction_region_refs": [item["id"] for item in fixture["construction_regions"]],
        "internal_seam_refs": [item["id"] for item in fixture["internal_seams"]],
        "supplied_projection_refs": [item["id"] for item in fixture["supplied_views"]],
        "separate_object_interface_refs": [item["id"] for item in fixture["separate_object_interfaces"]],
        "analytic_union_volume": fixture["analytic_union_volume"],
    }
    return {
        "physical_object_scopes": [fixture["physical_object_scope"]],
        "construction_region_hypotheses": fixture["construction_regions"],
        "placement_alternatives": [alternative],
        "internal_seam_hypotheses": fixture["internal_seams"],
        "supplied_projection_evidence": fixture["supplied_views"],
        "separate_object_interfaces": fixture["separate_object_interfaces"],
    }


class ConstructiveUnionEnumeratorTest(unittest.TestCase):
    def test_one_survivor_cannot_hide_competitor_with_invalid_direction_evidence(self):
        inputs = _inputs()
        view = copy.deepcopy(inputs["supplied_projection_evidence"][0])
        old_ref = view["id"]
        view["id"] += ".invalid_direction"
        view["required_direction_evidence"] = {"state": "accepted", "paths": [{
            "id": "path.invalid", "evidence_refs": ["candidate.mesh"],
            "independent_of_candidate_geometry": False,
        }]}
        alternative = copy.deepcopy(inputs["placement_alternatives"][0])
        alternative["id"] += ".invalid_direction"
        alternative["supplied_projection_refs"] = [view["id"] if ref == old_ref else ref for ref in alternative["supplied_projection_refs"]]
        inputs["supplied_projection_evidence"].append(view)
        inputs["placement_alternatives"].append(alternative)
        result = enumerate_constructive_union_hypotheses(**inputs)
        self.assertEqual(result["summary"]["survivor_count"], 1)
        self.assertEqual(result["summary"]["unresolved_live_alternative_count"], 1)
        self.assertFalse(result["quantity_eligible"])

    def test_unresolved_required_direction_is_not_a_hard_rejection(self):
        inputs = _inputs()
        inputs["supplied_projection_evidence"][0]["required_direction_evidence"] = {
            "state": "unknown", "paths": [],
        }
        result = enumerate_constructive_union_hypotheses(**inputs)
        self.assertEqual(result["summary"]["kernel_replay_count"], 0)
        self.assertEqual(result["summary"]["unresolved_live_alternative_count"], 1)
        self.assertEqual(result["summary"]["hard_pre_kernel_rejection_count"], 0)
        self.assertFalse(result["quantity_eligible"])

    def test_enumerator_has_no_stair_or_drawing_dispatch(self):
        source = inspect.getsource(enumerator_module).lower()
        for forbidden in ("candidate-08", "flight", "landing", "beam", "975", "275"):
            self.assertNotIn(forbidden, source)

    def test_complete_hypothesis_is_materialized_and_replayed(self):
        result = enumerate_constructive_union_hypotheses(**_inputs())

        self.assertEqual(result["status"], "accepted_unique_union")
        self.assertEqual(result["summary"]["assignment_count"], 1)
        self.assertEqual(result["summary"]["materialized_union_hypothesis_count"], 1)
        self.assertEqual(result["summary"]["pre_kernel_rejection_count"], 0)
        self.assertEqual(result["summary"]["kernel_replay_count"], 1)
        self.assertEqual(result["summary"]["survivor_count"], 1)
        self.assertEqual(result["kernel_replays"][0]["kernel_result"]["status"], "accepted")
        self.assertTrue(result["kernel_replays"][0]["quantity_eligible"])

    def test_unmaterialized_region_is_pending_not_ambiguity(self):
        inputs = _inputs()
        inputs["construction_region_hypotheses"][0]["state"] = "incomplete_open_boundary"
        inputs["construction_region_hypotheses"][0].pop("mesh")

        result = enumerate_constructive_union_hypotheses(**inputs)

        self.assertEqual(result["status"], "insufficient_constraints")
        self.assertEqual(result["reason_code"], "regions_unmaterialized")
        self.assertEqual(result["summary"]["materialized_union_hypothesis_count"], 0)
        self.assertEqual(result["summary"]["pre_kernel_rejection_count"], 1)
        self.assertEqual(result["summary"]["hard_pre_kernel_rejection_count"], 0)
        self.assertEqual(result["summary"]["kernel_replay_count"], 0)
        self.assertEqual(result["summary"]["survivor_count"], 0)
        self.assertNotEqual(result["status"], "ambiguous_survivors")

    def test_complete_kernel_failure_is_counted_as_replay(self):
        inputs = _inputs(overlapping_regions=True, wrong_analytic_union=True)

        result = enumerate_constructive_union_hypotheses(**inputs)

        self.assertEqual(result["status"], "reclosed_fail")
        self.assertEqual(result["summary"]["materialized_union_hypothesis_count"], 1)
        self.assertEqual(result["summary"]["pre_kernel_rejection_count"], 0)
        self.assertEqual(result["summary"]["kernel_replay_count"], 1)
        self.assertEqual(result["summary"]["survivor_count"], 0)
        self.assertEqual(result["kernel_replays"][0]["status"], "reclosed_fail")

    def test_overlap_requires_independent_authorization(self):
        inputs = _inputs(overlapping_regions=True)
        for seam in inputs["internal_seam_hypotheses"]:
            seam.pop("overlap_authorization_certificate")

        result = enumerate_constructive_union_hypotheses(**inputs)

        self.assertEqual(result["summary"]["materialized_union_hypothesis_count"], 0)
        self.assertEqual(result["summary"]["hard_pre_kernel_rejection_count"], 1)
        self.assertEqual(result["summary"]["kernel_replay_count"], 0)
        certificate = result["evaluations"][0]["internal_seam_semantics_certificate"]
        self.assertEqual(certificate["state"], "contradiction")
        self.assertTrue(
            all(
                "independent_overlap_authorization_missing" in check["errors"]
                for check in certificate["checks"]
            )
        )

    def test_assigned_interface_boundary_cannot_become_authorized_overlap(self):
        inputs = _inputs(overlapping_regions=True)
        for seam in inputs["internal_seam_hypotheses"]:
            seam["source_cap_classification"] = "interface_boundary"
            seam["required_contact_semantics"] = "coincident_face"

        result = enumerate_constructive_union_hypotheses(**inputs)

        self.assertEqual(result["summary"]["hard_pre_kernel_rejection_count"], 1)
        self.assertEqual(result["summary"]["kernel_replay_count"], 0)
        self.assertTrue(
            all(
                "assigned_coincident_face_materialized_as_overlap_union"
                in check["errors"]
                for check in result["evaluations"][0][
                    "internal_seam_semantics_certificate"
                ]["checks"]
            )
        )

    def test_disconnected_seam_graph_is_rejected_before_kernel(self):
        inputs = _inputs()
        omitted = inputs["internal_seam_hypotheses"].pop()
        inputs["placement_alternatives"][0]["internal_seam_refs"].remove(omitted["id"])

        result = enumerate_constructive_union_hypotheses(**inputs)

        self.assertEqual(result["summary"]["materialized_union_hypothesis_count"], 0)
        self.assertEqual(result["summary"]["pre_kernel_rejection_count"], 1)
        self.assertEqual(result["summary"]["hard_pre_kernel_rejection_count"], 1)
        self.assertEqual(result["summary"]["kernel_replay_count"], 0)
        self.assertEqual(result["status"], "reclosed_fail")
        self.assertIn(
            "construction_region_seam_graph_disconnected",
            result["evaluations"][0]["pre_kernel_rejection_reasons"],
        )
        self.assertTrue(
            result["contract"][
                "connected_construction_region_seam_graph_required_before_step4"
            ]
        )

    def test_two_reflected_equivalent_survivors_form_one_invariant_class(self):
        inputs = _inputs()
        second = copy.deepcopy(inputs["placement_alternatives"][0])
        second["id"] = "constructive_placement_alternative.synthetic.2"
        second["source_assignment_ref"] = "source_assignment.synthetic.2"
        inputs["placement_alternatives"].append(second)

        result = enumerate_constructive_union_hypotheses(**inputs)

        self.assertEqual(result["status"], "accepted_invariant_equivalence_class")
        self.assertEqual(result["summary"]["kernel_replay_count"], 2)
        self.assertEqual(result["summary"]["survivor_count"], 2)
        self.assertTrue(all(not item["quantity_eligible"] for item in result["kernel_replays"]))
        certificate = result["invariant_union_equivalence_certificate"]
        self.assertEqual(certificate["state"], "accepted")
        self.assertEqual(certificate["preserved_transform_alternative_count"], 2)
        self.assertTrue(all(item["status"] == "pass" for item in certificate["pairwise_checks"]))
        self.assertTrue(result["invariant_union_volume_candidate"]["quantity_eligible"])

    def test_one_survivor_cannot_hide_an_unmaterialized_competitor(self):
        inputs = _inputs()
        unresolved = copy.deepcopy(inputs["placement_alternatives"][0])
        unresolved["id"] = "constructive_placement_alternative.synthetic.unresolved"
        unresolved["source_assignment_ref"] = "source_assignment.synthetic.unresolved"
        unresolved["construction_region_refs"] = ["construction_region.missing"]
        inputs["placement_alternatives"].append(unresolved)

        result = enumerate_constructive_union_hypotheses(**inputs)

        self.assertEqual(result["status"], "insufficient_constraints")
        self.assertEqual(result["reason_code"], "unreplayed_competing_alternatives")
        self.assertEqual(result["summary"]["survivor_count"], 1)
        self.assertEqual(result["summary"]["unresolved_live_alternative_count"], 1)
        self.assertFalse(result["quantity_eligible"])
        self.assertTrue(all(not item["quantity_eligible"] for item in result["kernel_replays"]))

    def test_equal_volume_without_mesh_congruence_is_not_equivalence(self):
        inputs = _inputs()
        result = enumerate_constructive_union_hypotheses(**inputs)
        left = copy.deepcopy(result["survivors"][0])
        right = copy.deepcopy(left)
        right["constructive_union_hypothesis_ref"] = "synthetic.equal_volume_distorted"
        right["kernel_result"]["external_boundary"]["mesh"]["vertices_xyz_mm"][0][0] += 10.0

        certificate = certify_invariant_union_equivalence(
            [left, right],
            inputs["supplied_projection_evidence"],
            all_competing_alternatives_resolved=True,
        )

        self.assertEqual(certificate["state"], "unresolved")
        self.assertIn("pairwise_invariant_equivalence_failed", certificate["errors"])
        self.assertIsNone(certificate["invariant_union_volume_candidate"])

    def test_hard_contradiction_certificate_can_eliminate_competitor(self):
        inputs = _inputs()
        eliminated = copy.deepcopy(inputs["placement_alternatives"][0])
        eliminated["id"] = "constructive_placement_alternative.synthetic.eliminated"
        eliminated["source_assignment_ref"] = "source_assignment.synthetic.eliminated"
        eliminated["construction_region_refs"] = ["construction_region.missing"]
        eliminated["elimination_certificate"] = {
            "id": "alternative_elimination_certificate.synthetic",
            "record_type": "alternative_elimination_certificate",
            "status": "accepted",
            "alternative_ref": eliminated["id"],
            "basis": "hard_contradiction",
            "evidence_refs": ["synthetic.contradiction"],
        }
        inputs["placement_alternatives"].append(eliminated)

        result = enumerate_constructive_union_hypotheses(**inputs)

        self.assertEqual(result["status"], "accepted_unique_union")
        self.assertEqual(result["summary"]["survivor_count"], 1)
        self.assertEqual(result["summary"]["certified_elimination_count"], 1)
        self.assertEqual(result["summary"]["unresolved_live_alternative_count"], 0)
        self.assertTrue(result["quantity_eligible"])


if __name__ == "__main__":
    unittest.main()
