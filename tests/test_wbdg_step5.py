import unittest
from pathlib import Path

import fitz

from src.drawing_engine.core.object_agnostic_understanding import understand_page


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "portability" / "wbdg_office_vertical_duct"
TRUTH_VOLUME_M3 = 1.4326825519015183


def _understand(kind):
    path = FIXTURE / f"wbdg_office_vertical_duct_{kind}_orientation.pdf"
    with fitz.open(path) as document:
        return understand_page(document[0])["engineering_graph"]


def _coordinate(engineering):
    scopes = engineering["view_frame_graph"]["shared_coordinate_system"]["scopes"]
    assert len(scopes) == 1
    return scopes[0]


def _correspondence(coordinate):
    rows = coordinate["contour_correspondences"]
    assert len(rows) == 1
    return rows[0]


class WbdgStep5Test(unittest.TestCase):
    def test_negative_canonicalizes_representations_but_preserves_mirror_abstention(self):
        engineering = _understand("negative")
        coordinate = _coordinate(engineering)
        correspondence = _correspondence(coordinate)

        self.assertEqual(correspondence["representation_passing_candidate_count"], 2)
        self.assertEqual(correspondence["canonical_correspondence_count"], 1)
        canonicalization = correspondence["selected"]["canonicalization"]
        self.assertEqual(canonicalization["representation_count"], 2)
        self.assertEqual(
            canonicalization["representation_kinds"],
            ["closed_contour", "native_edge_pair"],
        )
        self.assertTrue(canonicalization["provenance_merged"])
        self.assertFalse(canonicalization["mirror_resolved_by_canonicalization"])
        self.assertEqual(len(canonicalization["signed_transform_alternatives"]), 2)

        orientation = coordinate["signed_orientation_certificate"]
        self.assertEqual(orientation["status"], "insufficient_constraints")
        signed = orientation["constraint_certificates"][0]["signed_shared_axis"]
        self.assertEqual(signed["status"], "ambiguous")
        self.assertEqual(
            [item["status"] for item in signed["candidate_evaluations"]],
            ["pass", "pass"],
        )
        reclosure = engineering["physical_component_reclosure"]
        self.assertEqual(reclosure["summary"]["slice3_input_count"], 0)
        step4 = engineering["physical_component_step4_replay"]
        self.assertEqual(step4["status"], "insufficient_constraints")
        self.assertEqual(step4["physical_components"], [])
        self.assertIsNone(step4["gross_envelope_volume"])
        self.assertEqual(engineering["solid_hypotheses"], [])
        self.assertEqual(engineering["quantities"], [])

    def test_positive_resolves_orientation_and_replays_one_watertight_prism(self):
        engineering = _understand("positive")
        coordinate = _coordinate(engineering)
        correspondence = _correspondence(coordinate)

        self.assertEqual(correspondence["representation_passing_candidate_count"], 2)
        self.assertEqual(correspondence["canonical_correspondence_count"], 1)
        orientation = coordinate["signed_orientation_certificate"]
        self.assertEqual(orientation["status"], "accepted")
        constraint = orientation["constraint_certificates"][0]
        self.assertEqual(constraint["oriented_cutting_plane"]["status"], "pass")
        self.assertEqual(constraint["signed_shared_axis"]["status"], "pass")
        self.assertEqual(
            [item["status"] for item in constraint["signed_shared_axis"]["candidate_evaluations"]],
            ["pass", "fail"],
        )
        self.assertEqual(coordinate["relative_orientation_state"], "resolved")
        self.assertEqual(coordinate["cut_plane_constraints"][0]["coordinate_candidates_mm"], [250.0])

        placement = engineering["physical_component_placement_evidence"]
        self.assertEqual(placement["status"], "resolved")
        self.assertFalse(
            placement["component_groupings"][0]["separately_drawn_profiles_are_additive"]
        )
        reclosure = engineering["physical_component_reclosure"]
        self.assertEqual(reclosure["summary"]["slice3_input_count"], 1)

        step4 = engineering["physical_component_step4_replay"]
        self.assertEqual(step4["status"], "reclosed_pass")
        self.assertEqual(len(step4["physical_components"]), 1)
        kernel = step4["kernel_result"]
        self.assertEqual(kernel["assembly_mesh"]["validation"]["component_count"], 1)
        self.assertTrue(kernel["assembly_mesh"]["validation"]["separately_watertight"])
        self.assertEqual(kernel["assembly_mesh"]["validation"]["boundary_edge_count"], 0)
        self.assertEqual(kernel["shared_interfaces"], [])
        self.assertEqual(kernel["pair_validations"], [])
        self.assertEqual(len(kernel["supplied_view_reprojections"]), 2)
        self.assertTrue(
            all(item["status"] == "pass" for item in kernel["supplied_view_reprojections"])
        )
        self.assertEqual(kernel["volume_validation"]["status"], "pass")
        self.assertAlmostEqual(kernel["volume_validation"]["residual_mm3"], 0.0)
        self.assertAlmostEqual(
            step4["gross_envelope_volume"]["value_m3"],
            TRUTH_VOLUME_M3,
            delta=0.0003,
        )
        self.assertEqual(engineering["quantities"], [])


if __name__ == "__main__":
    unittest.main()
