import copy
import json
import unittest

from src.drawing_engine.disciplines.concrete.physical_component_hypothesis import generate_physical_component_hypotheses
from src.drawing_engine.disciplines.concrete.physical_component_reclosure import (
    reclose_physical_component_hypotheses,
    validate_physical_component_reclosure,
)
from src.drawing_engine.disciplines.concrete.unsigned_bounded_sweep_certificate import certify_unsigned_bounded_sweep
from tests.test_physical_component_hypothesis import inputs, profile


def closed_frame_graph(*, mirrored=False):
    signed_transform = {
        "state": "resolved",
        "child_coordinate_to_parent": {"sign": 1, "offset_mm": 50.0},
        "candidates": [{"sign": 1, "offset_mm": 50.0}],
    }
    if mirrored:
        signed_transform = {
            "state": "unresolved",
            "child_coordinate_to_parent": {"sign": None, "offset_mm": None},
            "candidates": [
                {"sign": 1, "offset_mm": 50.0},
                {"sign": -1, "offset_mm": 950.0},
            ],
            "reason": "forward and mirrored native contour alignments remain equivalent",
        }
    return {
        "relations": [
            {
                "id": "cut.1",
                "type": "cut_at",
                "state": "accepted",
                "parent_view_id": "plan.1",
                "section_view_id": "section.1",
                "integration_certificate": {
                    "status": "passed",
                    "relation_scoped_metric_pair": {"status": "passed"},
                },
                "evidence_refs": [
                    "dimension.depth",
                    "dimension.position.x",
                    "dimension.position.z",
                ],
            }
        ],
        "object_scopes": [
            {
                "id": "physical.1",
                "state": "resolved_relative",
                "parent_view_ids": ["plan.1"],
                "section_view_ids": ["section.1"],
                "relation_refs": ["cut.1"],
                "shared_coordinate_scope_id": "coordinate.1",
                "reprojection_validation_refs": ["coordinate.1.reprojection.1"],
            }
        ],
        "shared_coordinate_system": {
            "scopes": [
                {
                    "id": "coordinate.1",
                    "state": "resolved_relative",
                    "view_axis_mappings": [
                        {
                            "view_id": "plan.1",
                            "u": "X",
                            "v": "Z",
                            "normal": "Y",
                            "state": "resolved_relative",
                        },
                        {
                            "view_id": "section.1",
                            "u": "X",
                            "v": "Y",
                            "normal": "Z",
                            "state": "resolved_relative",
                        },
                    ],
                    "reprojection_validations": [
                        {
                            "status": "pass",
                            "evidence_refs": ["dimension.position.x"],
                        }
                    ],
                    "cut_plane_constraints": [
                        {
                            "relation_id": "cut.1",
                            "coordinate_candidates_mm": [200.0],
                            "viewing_sign_state": "resolved",
                            "evidence_refs": ["dimension.position.z"],
                        }
                    ],
                    "contour_correspondences": [
                        {
                            "id": "correspondence.1",
                            "state": "accepted",
                            "selected": {
                                "signed_transform": signed_transform,
                                "primitive_refs": ["edge.plan", "edge.section"],
                            },
                        }
                    ],
                }
            ]
        },
    }


def resolved_evidence(hypotheses):
    selected = [item["id"] for item in hypotheses["hypotheses"]]
    group = {
        "id": "grouping.1",
        "record_type": "physical_component_grouping_evidence",
        "state": "resolved",
        "physical_scope_ref": "physical.1",
        "completeness": "complete",
        "hypothesis_refs": selected,
        "excluded_hypothesis_records": [],
        "separately_drawn_profiles_are_additive": False,
        "component_identity_records": [
            {
                "hypothesis_ref": item["id"],
                "state": "resolved",
                "separately_drawn_profile_basis": False,
                "identity_evidence_refs": ["cut.1"],
            }
            for item in hypotheses["hypotheses"]
        ],
        "interfaces": {
            "state": "resolved",
            "completeness": "complete",
            "no_external_interfaces_required": len(selected) == 1,
            "records": [],
        },
        "evidence_refs": ["cut.1"],
    }
    placements = []
    for item in hypotheses["hypotheses"]:
        placements.append(
            {
                "id": f"placement.{item['id']}",
                "record_type": "component_placement_evidence",
                "state": "resolved",
                "hypothesis_ref": item["id"],
                "physical_scope_ref": "physical.1",
                "profile_refs": list(item["profile_refs"]),
                "relation_ref": "cut.1",
                "parent_view_ref": "plan.1",
                "section_view_ref": "section.1",
                "depth": {
                    "state": "resolved",
                    "value_mm": 1000.0,
                    "dimension_refs": ["dimension.depth"],
                },
                "position": {
                    "state": "resolved",
                    "coordinates_mm": {"X": 300.0, "Z": 200.0},
                    "dimension_refs": [
                        "dimension.position.x",
                        "dimension.position.z",
                    ],
                },
                "relative_transform": {
                    "state": "resolved",
                    "axis_signs": {"X": 1, "Y": 1, "Z": 1},
                    "offsets_mm": {"X": 300.0, "Y": 1000.0, "Z": 200.0},
                    "section_to_parent": {"sign": 1, "offset_mm": 50.0},
                    "cut_plane_coordinate_mm": 200.0,
                },
                "metric_extent_refs": ["coordinate.1.reprojection.1"],
                "evidence_refs": [
                    "dimension.depth",
                    "dimension.position.x",
                    "dimension.position.z",
                    "cut.1",
                ],
            }
        )
    return {"component_groupings": [group], "component_placements": placements}


def hypothesis_fixture(*items):
    assembly, bindings = inputs(*items)
    frame = closed_frame_graph()
    hypotheses = generate_physical_component_hypotheses(
        assembly, bindings, frame, page_number=1
    )
    return hypotheses, frame


class PhysicalComponentReclosureTest(unittest.TestCase):
    def test_all_six_acceptance_requirements_reclose_without_slice3_promotion(self):
        hypotheses, frame = hypothesis_fixture(profile("profile.1", 0.0))
        evidence = resolved_evidence(hypotheses)

        result = reclose_physical_component_hypotheses(
            hypotheses,
            frame,
            page_number=1,
            placement_evidence=evidence,
        )

        self.assertEqual(validate_physical_component_reclosure(result), [])
        self.assertEqual(result["status"], "reclosed_pass")
        self.assertEqual(result["summary"]["slice3_input_count"], 1)
        certificate = result["certificates"][0]
        self.assertEqual(certificate["status"], "reclosed_pass")
        self.assertTrue(all(item["status"] == "pass" for item in certificate["gates"].values()))
        self.assertTrue(certificate["slice3_input_eligible"])
        self.assertIsNone(certificate["physical_component_ref"])
        self.assertIsNone(certificate["physical_transform_ref"])
        self.assertFalse(certificate["step4_kernel_invocation_eligible"])
        self.assertFalse(certificate["quantity_eligible"])

    def test_mirrored_contour_transform_is_primary_blocker(self):
        item = profile("profile.1", 0.0)
        assembly, bindings = inputs(item)
        frame = closed_frame_graph(mirrored=True)
        hypotheses = generate_physical_component_hypotheses(
            assembly, bindings, frame, page_number=1
        )

        result = reclose_physical_component_hypotheses(
            hypotheses,
            frame,
            page_number=1,
            placement_evidence=resolved_evidence(hypotheses),
        )

        certificate = result["certificates"][0]
        self.assertEqual(certificate["status"], "insufficient_constraints")
        self.assertEqual(certificate["primary_blocker_gate"], "relative_axis_signs_and_offsets")
        self.assertEqual(certificate["reason_code"], "mirrored_contour_transform_unresolved")
        self.assertEqual(len(certificate["gates"]["relative_axis_signs_and_offsets"]["signed_transform_candidates"]), 2)
        self.assertFalse(certificate["slice3_input_eligible"])
        self.assertEqual(validate_physical_component_reclosure(result), [])

    def test_scope_orientation_certificate_blocks_before_component_transform(self):
        item = profile("profile.1", 0.0)
        assembly, bindings = inputs(item)
        frame = closed_frame_graph(mirrored=True)
        coordinate = frame["shared_coordinate_system"]["scopes"][0]
        coordinate["signed_orientation_certificate"] = {
            "id": "signed_orientation_certificate.0001",
            "status": "insufficient_constraints",
            "constraint_certificates": [
                {
                    "oriented_cutting_plane": {"status": "insufficient_constraints"},
                    "signed_shared_axis": {"status": "ambiguous"},
                }
            ],
        }
        hypotheses = generate_physical_component_hypotheses(
            assembly, bindings, frame, page_number=1
        )

        result = reclose_physical_component_hypotheses(
            hypotheses,
            frame,
            page_number=1,
            placement_evidence=resolved_evidence(hypotheses),
        )

        gate = result["certificates"][0]["gates"]["relative_axis_signs_and_offsets"]
        self.assertEqual(gate["reason_code"], "signed_orientation_certificate_unresolved")
        self.assertEqual(gate["oriented_cutting_plane_statuses"], ["insufficient_constraints"])
        self.assertEqual(gate["signed_shared_axis_statuses"], ["ambiguous"])

    def test_unsigned_bounded_sweep_passes_without_resolving_scope_orientation(self):
        item = profile("profile.1", 0.0)
        assembly, bindings = inputs(item)
        frame = closed_frame_graph(mirrored=True)
        coordinate = frame["shared_coordinate_system"]["scopes"][0]
        coordinate["signed_orientation_certificate"] = {
            "id": "signed_orientation_certificate.0001",
            "status": "insufficient_constraints",
            "constraint_certificates": [],
        }
        hypotheses = generate_physical_component_hypotheses(
            assembly, bindings, frame, page_number=1
        )
        evidence = resolved_evidence(hypotheses)
        placement = evidence["component_placements"][0]
        certificate = certify_unsigned_bounded_sweep(
            page_number=1,
            hypothesis_ref=hypotheses["hypotheses"][0]["id"],
            physical_scope_ref="physical.1",
            orientation_certificate=coordinate["signed_orientation_certificate"],
            signed_transform=coordinate["contour_correspondences"][0]["selected"]["signed_transform"],
            extents_by_axis_mm={"X": 1000.0, "Y": 1000.0, "Z": 1000.0},
            plan_bounded_axes=["X", "Z"],
            dimension_refs_by_axis={
                "X": ["dimension.position.x"],
                "Y": ["dimension.depth"],
                "Z": ["dimension.position.z"],
            },
            external_interface_count=0,
            evidence_refs=["cut.1", "correspondence.1"],
        )
        self.assertIsNotNone(certificate)
        placement["unsigned_bounded_sweep_certificate"] = certificate
        placement["orientation_sensitivity"] = "sign_invariant"
        placement["relative_transform"].update(
            {
                "state": "resolved_up_to_reflection",
                "axis_signs": {"X": None, "Y": None, "Z": None},
                "canonical_axis_signs": {"X": 1, "Y": 1, "Z": 1},
                "section_to_parent": {
                    "state": "unresolved_reflection",
                    "sign": None,
                    "offset_mm": None,
                    "candidates": certificate["signed_transform_alternatives"],
                },
            }
        )

        result = reclose_physical_component_hypotheses(
            hypotheses, frame, page_number=1, placement_evidence=evidence
        )

        gate = result["certificates"][0]["gates"]["relative_axis_signs_and_offsets"]
        self.assertEqual(gate["status"], "pass")
        self.assertEqual(gate["orientation_sensitivity"], "sign_invariant")
        self.assertEqual(gate["signed_orientation_state"], "unresolved")
        self.assertEqual(result["summary"]["slice3_input_count"], 1)

    def test_missing_interfaces_abstain_after_other_gates_pass(self):
        hypotheses, frame = hypothesis_fixture(profile("profile.1", 0.0))
        evidence = resolved_evidence(hypotheses)
        evidence["component_groupings"][0]["interfaces"] = {
            "state": "unresolved",
            "completeness": "unknown",
            "records": [],
        }

        result = reclose_physical_component_hypotheses(
            hypotheses, frame, page_number=1, placement_evidence=evidence
        )

        certificate = result["certificates"][0]
        self.assertEqual(certificate["status"], "insufficient_constraints")
        self.assertEqual(certificate["primary_blocker_gate"], "explicit_interfaces")
        self.assertEqual(certificate["reason_code"], "explicit_interfaces_missing")

    def test_grouping_must_classify_every_surviving_alternative(self):
        hypotheses, frame = hypothesis_fixture(
            profile("profile.1", 0.0), profile("profile.2", 30.0)
        )
        evidence = resolved_evidence(hypotheses)
        evidence["component_groupings"][0]["hypothesis_refs"] = [
            hypotheses["hypotheses"][0]["id"]
        ]
        evidence["component_groupings"][0]["component_identity_records"] = [
            evidence["component_groupings"][0]["component_identity_records"][0]
        ]
        evidence["component_groupings"][0]["interfaces"] = {
            "state": "resolved",
            "completeness": "complete",
            "no_external_interfaces_required": True,
            "records": [],
        }

        result = reclose_physical_component_hypotheses(
            hypotheses, frame, page_number=1, placement_evidence=evidence
        )

        self.assertTrue(
            all(
                item["gates"]["unique_component_grouping"]["reason_code"]
                == "component_grouping_incomplete"
                for item in result["certificates"]
            )
        )
        self.assertEqual(result["accepted_hypothesis_refs"], [])

    def test_separately_drawn_profiles_cannot_be_selected_as_additive_count(self):
        hypotheses, frame = hypothesis_fixture(
            profile("profile.1", 0.0), profile("profile.2", 30.0)
        )
        evidence = resolved_evidence(hypotheses)
        evidence["component_groupings"][0]["separately_drawn_profiles_are_additive"] = True

        result = reclose_physical_component_hypotheses(
            hypotheses, frame, page_number=1, placement_evidence=evidence
        )

        self.assertTrue(all(item["status"] == "reclosed_fail" for item in result["certificates"]))
        self.assertTrue(
            all(
                item["gates"]["unique_component_grouping"]["reason_code"]
                == "multiple_alternatives_selected_as_additive_components"
                for item in result["certificates"]
            )
        )
        self.assertEqual(result["accepted_hypothesis_refs"], [])

    def test_transform_contradiction_reclosed_fails(self):
        hypotheses, frame = hypothesis_fixture(profile("profile.1", 0.0))
        evidence = resolved_evidence(hypotheses)
        evidence["component_placements"][0]["relative_transform"]["section_to_parent"][
            "offset_mm"
        ] = 75.0

        result = reclose_physical_component_hypotheses(
            hypotheses, frame, page_number=1, placement_evidence=evidence
        )

        certificate = result["certificates"][0]
        self.assertEqual(certificate["status"], "reclosed_fail")
        self.assertEqual(certificate["reason_code"], "relative_transform_contradiction")
        self.assertFalse(certificate["slice3_input_eligible"])

    def test_reordering_is_byte_identical(self):
        hypotheses, frame = hypothesis_fixture(profile("profile.1", 0.0))
        evidence = resolved_evidence(hypotheses)
        original = reclose_physical_component_hypotheses(
            hypotheses, frame, page_number=2, placement_evidence=evidence
        )
        reordered = reclose_physical_component_hypotheses(
            copy.deepcopy(hypotheses),
            copy.deepcopy(frame),
            page_number=2,
            placement_evidence={
                "component_placements": list(reversed(copy.deepcopy(evidence["component_placements"]))),
                "component_groupings": list(reversed(copy.deepcopy(evidence["component_groupings"]))),
            },
        )

        encode = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
        self.assertEqual(encode(original), encode(reordered))

    def test_validator_rejects_slice3_promotion(self):
        hypotheses, frame = hypothesis_fixture(profile("profile.1", 0.0))
        result = reclose_physical_component_hypotheses(
            hypotheses,
            frame,
            page_number=1,
            placement_evidence=resolved_evidence(hypotheses),
        )
        result["certificates"][0]["physical_component_ref"] = "component.1"
        result["certificates"][0]["physical_transform_ref"] = "transform.1"
        result["certificates"][0]["step4_kernel_invocation_eligible"] = True

        errors = validate_physical_component_reclosure(result)
        self.assertTrue(any("cannot assign Slice 3" in item for item in errors))
        self.assertTrue(any("cannot authorize Step 4" in item for item in errors))


if __name__ == "__main__":
    unittest.main()
