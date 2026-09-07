import copy
import json
import unittest

from src.drawing_engine.disciplines.concrete.physical_component_hypothesis import (
    generate_physical_component_hypotheses,
    validate_physical_component_hypotheses,
)


def profile(identifier, x0, *, width=10.0, height=10.0):
    points = [
        [x0, 0.0],
        [x0 + width, 0.0],
        [x0 + width, height],
        [x0, height],
    ]
    return {
        "id": identifier,
        "record_type": "assembled_profile_candidate",
        "record_version": "0.1.0",
        "state": "resolved",
        "scope_ref": "title.1",
        "quantity_eligible": False,
        "bbox_display": [x0, 0.0, x0 + width, height],
        "ordered_boundary_display": points,
        "source_edge_refs": [f"{identifier}.edge.{index}" for index in range(4)],
        "closure": {
            "closed": True,
            "branch_free": True,
            "unique_completion": True,
            "scale_bounded": True,
            "dimensionally_redundant": True,
        },
        "dimension_certificate": {
            "passed": True,
            "scale_points_per_mm": 0.1,
            "dimension_refs": ["dimension.1", "dimension.2"],
        },
        "evidence_refs": [
            f"{identifier}.edge.{index}" for index in range(4)
        ]
        + ["dimension.1", "dimension.2"],
    }


def binding(item):
    return {
        "id": f"binding.{item['id']}",
        "record_type": "profile_physical_scope_binding",
        "record_version": "0.1.0",
        "state": "accepted",
        "profile_ref": item["id"],
        "source_view_ref": "supporting.1",
        "physical_scope_ref": "physical.1",
        "membership_role": "supporting_projection",
        "step5_reconstruction_input_eligible": True,
        "step4_kernel_invocation_eligible": False,
        "additive_component_identity_established": False,
        "quantity_eligible": False,
        "evidence_refs": [item["id"], "physical.1"],
    }


def inputs(*items):
    return (
        {"profiles": list(items)},
        {"bindings": [binding(item) for item in items]},
    )


def frame_graph(*, mirrored=False):
    transform = {
        "state": "resolved",
        "child_coordinate_to_parent": {"sign": 1, "offset_mm": 200.0},
        "candidates": [{"sign": 1, "offset_mm": 200.0}],
    }
    if mirrored:
        transform = {
            "state": "unresolved",
            "child_coordinate_to_parent": {"sign": None, "offset_mm": None},
            "candidates": [
                {"sign": 1, "offset_mm": 200.0},
                {"sign": -1, "offset_mm": 1200.0},
            ],
            "reason": "forward and mirrored native contour alignments remain equivalent",
        }
    return {
        "object_scopes": [
            {
                "id": "physical.1",
                "state": "resolved_relative",
                "shared_coordinate_scope_id": "coordinate.1",
            }
        ],
        "shared_coordinate_system": {
            "scopes": [
                {
                    "id": "coordinate.1",
                    "state": "resolved_relative",
                    "contour_correspondences": [
                        {
                            "id": "correspondence.1",
                            "state": "accepted",
                            "selected": {
                                "signed_transform": transform,
                                "primitive_refs": ["edge.parent", "edge.child"],
                            },
                        }
                    ],
                }
            ]
        },
    }


class PhysicalComponentHypothesisTest(unittest.TestCase):
    def test_metric_topology_equivalent_disjoint_profiles_remain_alternatives(self):
        first = profile("profile.1", 0.0)
        second = profile("profile.2", 30.0)
        third = profile("profile.3", 60.0)
        assembly, bindings = inputs(first, second, third)

        result = generate_physical_component_hypotheses(
            assembly, bindings, frame_graph(), page_number=4
        )

        self.assertEqual(validate_physical_component_hypotheses(result), [])
        self.assertEqual(result["status"], "insufficient_constraints")
        self.assertEqual(result["generation_status"], "bounded_alternatives")
        self.assertEqual(result["summary"]["hypothesis_count"], 3)
        self.assertEqual(result["summary"]["accepted_component_count"], 0)
        self.assertEqual(len(result["alternative_sets"]), 1)
        self.assertEqual(
            {item["alternative_set_ref"] for item in result["hypotheses"]},
            {result["alternative_sets"][0]["id"]},
        )
        self.assertTrue(
            all(item["additive_count_interpretation"] == "unresolved" for item in result["hypotheses"])
        )
        self.assertFalse(result["contract"]["separate_profile_implies_additive_count"])

    def test_profiles_with_one_shared_boundary_form_one_candidate_group(self):
        first = profile("profile.1", 0.0)
        second = profile("profile.2", 10.0)
        assembly, bindings = inputs(first, second)

        result = generate_physical_component_hypotheses(
            assembly, bindings, frame_graph(), page_number=1
        )

        self.assertEqual(validate_physical_component_hypotheses(result), [])
        self.assertEqual(result["summary"]["hypothesis_count"], 1)
        self.assertEqual(result["summary"]["multi_profile_hypothesis_count"], 1)
        hypothesis = result["hypotheses"][0]
        self.assertEqual(hypothesis["profile_refs"], ["profile.1", "profile.2"])
        self.assertEqual(
            hypothesis["compatibility_certificate"]["grouping_basis"],
            "shared_boundary_edges",
        )
        self.assertEqual(
            hypothesis["compatibility_certificate"]["shared_edges"]["status"],
            "passed",
        )
        self.assertIsNone(hypothesis["physical_component_ref"])
        self.assertEqual(hypothesis["physical_transform_refs"], [])
        self.assertFalse(hypothesis["step4_kernel_invocation_eligible"])
        self.assertFalse(hypothesis["quantity_eligible"])

    def test_mirrored_contour_transform_is_the_precise_scope_abstention(self):
        item = profile("profile.1", 0.0)
        assembly, bindings = inputs(item)

        result = generate_physical_component_hypotheses(
            assembly, bindings, frame_graph(mirrored=True), page_number=8
        )

        abstention = result["abstentions"][0]
        self.assertEqual(abstention["reason_code"], "mirrored_contour_transform_unresolved")
        self.assertIn("mirrored", abstention["reason"])
        self.assertEqual(
            abstention["surviving_hypothesis_refs"],
            [result["hypotheses"][0]["id"]],
        )
        self.assertEqual(
            abstention["required_next_certificate"],
            "signed_orientation_or_unsigned_bounded_sweep",
        )

    def test_upstream_signed_orientation_certificate_is_the_precise_scope_abstention(self):
        item = profile("profile.1", 0.0)
        assembly, bindings = inputs(item)
        graph = frame_graph(mirrored=True)
        coordinate = graph["shared_coordinate_system"]["scopes"][0]
        coordinate["signed_orientation_certificate"] = {
            "id": "signed_orientation_certificate.0001",
            "status": "insufficient_constraints",
            "constraint_certificates": [
                {
                    "oriented_cutting_plane": {
                        "status": "insufficient_constraints",
                        "reason": "missing_unique_native_arrowhead",
                        "evidence_refs": ["stem.1", "stem.2"],
                    },
                    "signed_shared_axis": {
                        "status": "ambiguous",
                        "reason": "forward and mirrored transforms both pass",
                        "evidence_refs": ["edge.1", "edge.2"],
                    },
                }
            ],
        }

        result = generate_physical_component_hypotheses(
            assembly, bindings, graph, page_number=8
        )

        abstention = result["abstentions"][0]
        self.assertEqual(abstention["reason_code"], "signed_orientation_certificate_unresolved")
        self.assertIn("missing_unique_native_arrowhead", abstention["reason"])
        self.assertIn("forward and mirrored transforms both pass", abstention["reason"])
        self.assertIn("signed_orientation_certificate.0001", abstention["evidence_refs"])
        self.assertEqual(
            abstention["required_next_certificate"],
            "signed_orientation_or_unsigned_bounded_sweep",
        )

    def test_hypothesis_bound_fails_closed(self):
        first = profile("profile.1", 0.0)
        second = profile("profile.2", 30.0)
        assembly, bindings = inputs(first, second)

        result = generate_physical_component_hypotheses(
            assembly,
            bindings,
            frame_graph(),
            page_number=1,
            max_hypotheses=1,
        )

        self.assertEqual(result["hypotheses"], [])
        self.assertEqual(result["status"], "abstained")
        self.assertEqual(result["abstentions"][0]["reason_code"], "hypothesis_search_not_bounded")
        self.assertEqual(validate_physical_component_hypotheses(result), [])

    def test_input_reordering_is_byte_identical(self):
        first = profile("profile.1", 0.0)
        second = profile("profile.2", 30.0)
        assembly, bindings = inputs(first, second)
        original = generate_physical_component_hypotheses(
            assembly, bindings, frame_graph(mirrored=True), page_number=3
        )
        reordered = generate_physical_component_hypotheses(
            {"profiles": [copy.deepcopy(second), copy.deepcopy(first)]},
            {"bindings": list(reversed(copy.deepcopy(bindings["bindings"])))},
            copy.deepcopy(frame_graph(mirrored=True)),
            page_number=3,
        )

        encode = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
        self.assertEqual(encode(original), encode(reordered))

    def test_validator_rejects_premature_component_promotion(self):
        item = profile("profile.1", 0.0)
        assembly, bindings = inputs(item)
        result = generate_physical_component_hypotheses(
            assembly, bindings, frame_graph(), page_number=1
        )
        result["hypotheses"][0]["state"] = "accepted"
        result["hypotheses"][0]["physical_component_ref"] = "component.1"
        result["hypotheses"][0]["step4_kernel_invocation_eligible"] = True

        errors = validate_physical_component_hypotheses(result)
        self.assertTrue(any("must remain candidate" in item for item in errors))
        self.assertTrue(any("physical_component_ref must remain unresolved" in item for item in errors))
        self.assertTrue(any("cannot authorize Step 4" in item for item in errors))


if __name__ == "__main__":
    unittest.main()
