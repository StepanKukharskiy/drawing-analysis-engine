import copy
import unittest

from src.drawing_engine.disciplines.concrete.bounded_profile_sweep_materializer import materialize_bounded_profile_sweep_union
from src.drawing_engine.disciplines.concrete.constructive_union_enumerator import enumerate_constructive_union_hypotheses


def _scope():
    return {
        "id": "physical_object_scope.synthetic_folded_slab",
        "record_type": "physical_object_scope",
        "state": "resolved",
        "evidence_refs": ["synthetic.scope"],
    }


def _sweep(identifier, origin_x, seam_edge, cap_id):
    return {
        "id": identifier,
        "record_type": "bounded_profile_sweep_hypothesis",
        "physical_object_scope_ref": _scope()["id"],
        "construction_region_id": f"construction_region.{identifier}",
        "construction_role": "folded_slab_region",
        "state": "resolved",
        "profile_boundary": {
            "id": f"profile.{identifier}",
            "record_type": "bounded_profile_boundary",
            "state": "resolved",
            "ordered_points_uv_mm": [[0, 0], [10, 0], [10, 5], [0, 5]],
            "classified_caps": [
                {
                    "id": cap_id,
                    "classification": "internal_seam",
                    "boundary_edge_index": seam_edge,
                    "evidence_refs": [f"synthetic.cap.{identifier}"],
                }
            ],
            "evidence_refs": [f"synthetic.profile.{identifier}"],
        },
        "sweep_bound": {
            "id": f"sweep.{identifier}",
            "record_type": "bounded_sweep",
            "state": "resolved",
            "depth_mm": 10.0,
            "evidence_refs": [f"synthetic.sweep.{identifier}"],
        },
        "local_to_world_transform": {
            "id": f"transform.{identifier}",
            "record_type": "local_to_world_transform",
            "state": "resolved",
            "origin_xyz_mm": [origin_x, 0.0, 0.0],
            "local_axes_xyz": {
                "x": [1.0, 0.0, 0.0],
                "y": [0.0, 1.0, 0.0],
                "z": [0.0, 0.0, 1.0],
            },
            "evidence_refs": [f"synthetic.transform.{identifier}"],
        },
        "evidence_refs": [f"synthetic.hypothesis.{identifier}"],
    }


def _inputs():
    views = [
        {
            "id": "view.synthetic.plan",
            "record_type": "native_projection_evidence",
            "state": "resolved",
            "source_geometry_kind": "synthetic_known_truth",
            "derived_from_generated_geometry": False,
            "origin_xyz_mm": [0.0, 0.0, 0.0],
            "u_axis_xyz": [1.0, 0.0, 0.0],
            "v_axis_xyz": [0.0, 1.0, 0.0],
            "polygons_uv_mm": [[[0, 0], [20, 0], [20, 10], [0, 10]]],
            "tolerance_mm": 1e-6,
            "evidence_refs": ["synthetic.plan"],
        },
        {
            "id": "view.synthetic.section",
            "record_type": "native_projection_evidence",
            "state": "resolved",
            "source_geometry_kind": "synthetic_known_truth",
            "derived_from_generated_geometry": False,
            "origin_xyz_mm": [0.0, 0.0, 0.0],
            "u_axis_xyz": [1.0, 0.0, 0.0],
            "v_axis_xyz": [0.0, 0.0, 1.0],
            "polygons_uv_mm": [[[0, 0], [20, 0], [20, 5], [0, 5]]],
            "tolerance_mm": 1e-6,
            "evidence_refs": ["synthetic.section"],
        },
    ]
    return {
        "physical_object_scope": _scope(),
        "profile_sweep_hypotheses": [
            _sweep("left", 0.0, 1, "cap.left.right"),
            _sweep("right", 10.0, 3, "cap.right.left"),
        ],
        "internal_seam_hypotheses": [
            {
                "id": "internal_seam.synthetic.middle",
                "record_type": "internal_seam_hypothesis",
                "state": "resolved",
                "cap_face_refs": ["cap.left.right", "cap.right.left"],
                "evidence_refs": ["synthetic.seam"],
            }
        ],
        "supplied_projection_evidence": views,
        "analytic_union_certificate": {
            "id": "analytic_union_certificate.synthetic",
            "record_type": "analytic_union_certificate",
            "state": "resolved",
            "value_mm3": 1000.0,
            "basis": "two_independent_profile_area_times_sweep_terms",
            "independent_of_generated_mesh": True,
            "calculation_terms": [
                {"profile_area_mm2": 50.0, "sweep_depth_mm": 10.0},
                {"profile_area_mm2": 50.0, "sweep_depth_mm": 10.0},
            ],
            "evidence_refs": ["synthetic.union"],
        },
        "alternative_id": "constructive_placement_alternative.synthetic.materialized",
    }


class BoundedProfileSweepMaterializerTest(unittest.TestCase):
    def test_materialized_regions_and_seam_replay_through_step4(self):
        materialized = materialize_bounded_profile_sweep_union(**_inputs())

        self.assertEqual(materialized["status"], "materialized")
        self.assertEqual(len(materialized["construction_regions"]), 2)
        self.assertEqual(len(materialized["internal_seams"]), 1)
        self.assertEqual(len(materialized["supplied_projections"]), 2)
        self.assertEqual(materialized["analytic_union_certificate"]["residual_mm3"], 0.0)
        result = enumerate_constructive_union_hypotheses(
            [materialized["physical_object_scope"]],
            materialized["construction_regions"],
            [materialized["placement_alternative"]],
            materialized["internal_seams"],
            materialized["supplied_projections"],
        )
        self.assertEqual(result["status"], "accepted_unique_union")
        self.assertEqual(result["summary"]["kernel_replay_count"], 1)
        self.assertEqual(result["summary"]["survivor_count"], 1)

    def test_unresolved_transform_fails_before_region_materialization(self):
        inputs = _inputs()
        inputs["profile_sweep_hypotheses"][0]["local_to_world_transform"]["state"] = "proposed"
        with self.assertRaisesRegex(ValueError, "local-to-world transform is unresolved"):
            materialize_bounded_profile_sweep_union(**inputs)

    def test_noncoincident_seam_caps_fail_closed(self):
        inputs = _inputs()
        inputs["profile_sweep_hypotheses"][1]["local_to_world_transform"]["origin_xyz_mm"][0] = 11.0
        with self.assertRaisesRegex(ValueError, "not congruent and coincident"):
            materialize_bounded_profile_sweep_union(**inputs)

    def test_wrong_analytic_union_certificate_fails_closed(self):
        inputs = _inputs()
        inputs["analytic_union_certificate"] = copy.deepcopy(inputs["analytic_union_certificate"])
        inputs["analytic_union_certificate"]["value_mm3"] = 2000.0
        with self.assertRaisesRegex(ValueError, "disagrees with materialized regions"):
            materialize_bounded_profile_sweep_union(**inputs)

    def test_generated_projection_cannot_be_reused_as_expected_evidence(self):
        inputs = _inputs()
        inputs["supplied_projection_evidence"][0][
            "derived_from_generated_geometry"
        ] = True
        with self.assertRaisesRegex(ValueError, "projection independence"):
            materialize_bounded_profile_sweep_union(**inputs)

    def test_mesh_derived_analytic_union_is_rejected(self):
        inputs = _inputs()
        inputs["analytic_union_certificate"][
            "independent_of_generated_mesh"
        ] = False
        with self.assertRaisesRegex(ValueError, "independence"):
            materialize_bounded_profile_sweep_union(**inputs)


if __name__ == "__main__":
    unittest.main()
