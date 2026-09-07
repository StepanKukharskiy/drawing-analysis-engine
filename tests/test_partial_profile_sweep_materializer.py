import unittest

from src.drawing_engine.disciplines.concrete.partial_profile_sweep_materializer import materialize_partial_profile_sweeps


def _hypothesis(*, closure_kind="analysis_capped_closed", transforms=None):
    evidence = ["native_edge.001", "dimension.001"]
    return {
        "id": "partial_sweep.001",
        "record_type": "partial_profile_sweep_hypothesis",
        "state": "resolved_search_hypothesis",
        "construction_role": "folded_region",
        "classification": "unclassified_folded_region",
        "profile_boundary": {
            "id": "partial_profile.001",
            "record_type": "partial_profile_boundary",
            "state": "resolved",
            "closure_kind": closure_kind,
            "ordered_points_uv_mm": [[0, 0], [200, 0], [200, 300], [0, 300]],
            "classified_caps": [
                {
                    "id": "partial_cap.001",
                    "boundary_edge_index": 1,
                    "classification": "analysis_cap" if closure_kind.startswith("analysis") else "external_boundary",
                    "evidence_refs": evidence,
                }
            ],
            "evidence_refs": evidence,
        },
        "sweep_bound": {
            "id": "partial_bound.001",
            "record_type": "partial_sweep_bound",
            "state": "resolved",
            "closure_kind": "physical_bounded",
            "depth_mm": 400,
            "evidence_refs": evidence,
        },
        "local_to_world_transform_alternatives": transforms
        or [
            {
                "id": "transform.001",
                "record_type": "local_to_world_transform",
                "state": "resolved",
                "origin_xyz_mm": [10, 20, 30],
                "local_axes_xyz": {
                    "x": [0, 0, 1],
                    "y": [1, 0, 0],
                    "z": [0, 1, 0],
                },
                "evidence_refs": evidence,
            }
        ],
        "quantity_eligible": False,
        "evidence_refs": evidence,
    }


def _projection(*, derived=False):
    return {
        "id": "projection_coverage.001",
        "record_type": "projection_coverage_evidence",
        "state": "resolved",
        "hypothesis_ref": "partial_sweep.001",
        "projection_role": "section",
        "coverage_state": "not_depicted",
        "derived_from_generated_geometry": derived,
        "evidence_refs": ["title_scope.001"],
    }


class PartialProfileSweepMaterializerTest(unittest.TestCase):
    def test_materializes_arbitrary_rigid_axes_without_promoting_temporary_cap(self):
        result = materialize_partial_profile_sweeps([_hypothesis()], [_projection()], page_number=1)

        self.assertEqual(result["status"], "materialized_partial_hypotheses")
        self.assertEqual(result["summary"]["materialized_candidate_count"], 1)
        candidate = result["candidate_previews"][0]
        self.assertEqual(candidate["mesh"]["vertices_xyz_mm"][0], [10.0, 20.0, 30.0])
        self.assertEqual(candidate["mesh"]["vertices_xyz_mm"][1], [10.0, 20.0, 230.0])
        self.assertEqual(candidate["mesh"]["vertices_xyz_mm"][4], [410.0, 20.0, 30.0])
        self.assertTrue(candidate["mesh"]["validation"]["watertight"])
        self.assertFalse(candidate["physical_boundary_complete"])
        self.assertFalse(candidate["analytic_preview_enclosure"]["is_physical_volume"])
        self.assertNotIn("volume_candidate_m3", candidate)
        self.assertFalse(candidate["quantity_eligible"])

    def test_preserves_every_transform_alternative(self):
        first = _hypothesis()["local_to_world_transform_alternatives"][0]
        second = {**first, "id": "transform.002", "origin_xyz_mm": [-10, -20, -30]}
        result = materialize_partial_profile_sweeps(
            [_hypothesis(transforms=[first, second])],
            [_projection()],
            page_number=1,
        )

        self.assertEqual(result["summary"]["transform_alternative_count"], 2)
        self.assertEqual(len(result["candidate_previews"]), 2)
        self.assertEqual(
            {item["local_to_world_transform"]["id"] for item in result["candidate_previews"]},
            {"transform.001", "transform.002"},
        )

    def test_physically_bounded_preview_still_does_not_promote_quantity(self):
        result = materialize_partial_profile_sweeps(
            [_hypothesis(closure_kind="physical_closed")],
            [_projection()],
            page_number=1,
        )
        candidate = result["candidate_previews"][0]
        self.assertTrue(candidate["physical_boundary_complete"])
        self.assertEqual(candidate["analytic_preview_enclosure"]["value_mm3"], 24_000_000.0)
        self.assertFalse(candidate["analytic_preview_enclosure"]["is_physical_volume"])
        self.assertFalse(candidate["quantity_eligible"])

    def test_rejects_projection_evidence_generated_from_candidate_mesh(self):
        with self.assertRaisesRegex(ValueError, "independent of generated geometry"):
            materialize_partial_profile_sweeps(
                [_hypothesis()],
                [_projection(derived=True)],
                page_number=1,
            )

    def test_rejects_quantity_eligible_search_hypothesis(self):
        hypothesis = _hypothesis()
        hypothesis["quantity_eligible"] = True
        with self.assertRaisesRegex(ValueError, "must be quantity-ineligible"):
            materialize_partial_profile_sweeps([hypothesis], [_projection()], page_number=1)


if __name__ == "__main__":
    unittest.main()
