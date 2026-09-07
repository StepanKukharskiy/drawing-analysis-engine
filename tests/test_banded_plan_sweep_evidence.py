import unittest

from src.drawing_engine.disciplines.concrete.banded_plan_sweep_evidence import derive_banded_plan_sweep_evidence


def _fixture():
    term_refs = [f"dimension.term.{index}" for index in range(5)]
    all_refs = ["dimension.overall", *term_refs]
    ownership = {
        "attachments": [
            {
                "dimension_ref": ref,
                "status": "accepted",
                "geometry_anchor_refs": [f"vertex.{index}", f"vertex.{index + 1}"],
            }
            for index, ref in enumerate(term_refs)
        ]
    }
    adjudication = {
        "records": [
            {"dimension_ref": ref, "status": "accepted", "view_refs": ["plan.view"]}
            for ref in all_refs
        ],
        "terminal_less_redundancy_certificates": [
            {
                "kind": "arithmetic_chain",
                "overall_dimension_ref": "dimension.overall",
                "term_dimension_refs": term_refs,
                "term_values_mm": [200.0, 975.0, 200.0, 975.0, 200.0],
                "total_value_mm": 2550.0,
                "arithmetic_residual_mm": 0.0,
            }
        ],
    }
    frames = {
        "relations": [
            {
                "id": "cut.1",
                "type": "cut_at",
                "state": "accepted",
                "parent_view_id": "plan.view",
                "section_view_id": "section.scope",
            }
        ],
        "object_scopes": [
            {
                "id": "physical.1",
                "relation_refs": ["cut.1"],
                "shared_coordinate_scope_id": "coordinate.1",
            }
        ],
        "shared_coordinate_system": {
            "scopes": [
                {
                    "id": "coordinate.1",
                    "signed_orientation_certificate": {
                        "id": "orientation.1",
                        "status": "insufficient_constraints",
                    },
                }
            ]
        },
    }
    profiles = {
        "scopes": [
            {
                "scope_ref": "section.scope",
                "profiles": [],
                "abstentions": [
                    {
                        "id": "profile.abstention.1",
                        "reason_code": "insufficient_dimensional_redundancy",
                    }
                ],
            }
        ]
    }
    return ownership, adjudication, frames, profiles


class BandedPlanSweepEvidenceTest(unittest.TestCase):
    def test_two_equal_disjoint_bands_close_while_profile_split_remains_unresolved(self):
        ownership, adjudication, frames, profiles = _fixture()

        result = derive_banded_plan_sweep_evidence(
            ownership,
            adjudication,
            frames,
            profiles,
            page_number=1,
        )

        self.assertEqual(result["status"], "plan_bands_resolved_profile_split_pending")
        certificate = result["plan_band_certificate"]
        self.assertEqual(certificate["equal_sweep_width_mm"], 975.0)
        self.assertEqual(certificate["clear_gap_mm"], 200.0)
        self.assertEqual(
            [item["interval_mm"] for item in certificate["bands"]],
            [[200.0, 1175.0], [1375.0, 2350.0]],
        )
        self.assertEqual(certificate["signed_orientation_state"], "unresolved")
        self.assertTrue(result["contract"]["unresolved_sign_does_not_invalidate_plan_bands"])
        self.assertEqual(
            result["profile_split_assessment"]["required_next_certificate"],
            "unique_two_profile_section_split_and_landing_footprint",
        )

    def test_unequal_band_widths_abstain(self):
        ownership, adjudication, frames, profiles = _fixture()
        adjudication["terminal_less_redundancy_certificates"][0]["term_values_mm"][3] = 900.0

        result = derive_banded_plan_sweep_evidence(
            ownership,
            adjudication,
            frames,
            profiles,
            page_number=1,
        )

        self.assertEqual(result["status"], "insufficient_constraints")
        self.assertEqual(result["reason_code"], "unique_five_term_equal_band_chain_unresolved")

    def test_per_dimension_duplicates_of_one_chain_are_canonicalized(self):
        ownership, adjudication, frames, profiles = _fixture()
        duplicate = dict(adjudication["terminal_less_redundancy_certificates"][0])
        duplicate["dimension_ref"] = "dimension.term.1"
        adjudication["terminal_less_redundancy_certificates"].append(duplicate)

        result = derive_banded_plan_sweep_evidence(
            ownership,
            adjudication,
            frames,
            profiles,
            page_number=1,
        )

        self.assertEqual(result["status"], "plan_bands_resolved_profile_split_pending")
        self.assertEqual(result["plan_band_certificate"]["equal_sweep_width_mm"], 975.0)


if __name__ == "__main__":
    unittest.main()
