import copy
import json
from pathlib import Path
import unittest

from src.drawing_engine.disciplines.mep.mep_bounded_clash_verification import (
    build_mep_bounded_clash_verification,
    validate_mep_bounded_clash_verification,
)
from src.drawing_engine.disciplines.mep.mep_bounded_local_3d import validate_mep_bounded_local_3d_segments
from src.drawing_engine.disciplines.mep.mep_claim_grounding import validate_mep_claim_grounding


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "fixtures" / "mep" / "m_and_p_coordination" / "real_m2_m5_checkpoint"


def _load(name):
    return json.loads((CHECKPOINT / name).read_text(encoding="utf-8"))


def _inputs():
    return {
        "attribute_bindings": _load("pages_1a_1b.attribute-bindings.json"),
        "bounded_local_3d": _load("pages_1a_1b.bounded-local-3d.json"),
        "claim_grounding": _load("pages_1a_1b.claim-grounding.json"),
    }


def _two_target_claim(claim_grounding):
    payload = copy.deepcopy(claim_grounding)
    by_page = {}
    for target in payload["eligible_targets"]:
        by_page.setdefault(target["page_ref"], []).append(target)
    targets = next(rows for rows in by_page.values() if len(rows) == 2)
    base = copy.deepcopy(payload["grounding_records"][0])
    polygon_points = [point for target in targets for point in target["corridor_boundary_points_display"]]
    xs = [point[0] for point in polygon_points]
    ys = [point[1] for point in polygon_points]
    base["record_type"] = "2d_overlap_candidate"
    base["id"] = "mep_2d_overlap_candidate.synthetic"
    base["page_ref"] = targets[0]["page_ref"]
    base["target_ref"] = None
    base["m3_5_composite_ref"] = None
    base["m5_partial_2_5d_centreline_ref"] = None
    base["claim_provenance"]["claim_polygon_display"] = [
        [min(xs) - 1, min(ys) - 1],
        [max(xs) + 1, min(ys) - 1],
        [max(xs) + 1, max(ys) + 1],
        [min(xs) - 1, max(ys) + 1],
    ]
    base["claim_provenance"]["claim_geometry_basis"] = "synthetic_test_polygon"
    base["alternatives"] = [
        {
            "target_ref": target["id"],
            "m3_5_composite_ref": target["m3_5_composite_ref"],
            "m5_partial_2_5d_centreline_ref": target["m5_partial_2_5d_centreline_ref"],
            "relation": "claim_polygon_overlaps_route_corridor_in_2d",
            "coordinate_space": "page_display_points_top_left",
            "three_dimensional_authority": False,
        }
        for target in targets
    ]
    base["ambiguity"] = {"eligible_2d_overlap_count": 2, "resolved": False}
    base["state"] = "candidate"
    base["epistemic_state"] = "unknown"
    base["reasons"] = ["multiple_partial_2_5d_targets_overlap_claim_in_2d"]
    payload["grounding_records"] = [base]
    payload["summary"] = {
        "claim_count": 1,
        "eligible_partial_2_5d_target_count": len(payload["eligible_targets"]),
        "accepted_unique_grounding_count": 0,
        "ambiguous_2d_overlap_candidate_count": 1,
        "abstention_count": 0,
    }
    assert validate_mep_claim_grounding(payload) == []
    return payload, targets


def _set_geometry(segment, points, width=1.0):
    segment["centreline_points_xyz_m"] = points
    segment["local_envelope"]["centreline_points_xyz_m"] = copy.deepcopy(points)
    segment["local_envelope"]["outer_width_m"] = width
    segment["local_envelope"]["radius_m"] = width / 2
    segment["physical_envelope_dimension"]["representative_outer_width_m"] = width
    segment["physical_envelope_dimension"]["maximum_width_residual_m"] = 0.0
    for observation in segment["physical_envelope_dimension"]["occurrence_observations"]:
        observation["outer_width_m"] = width
        observation["measurement_resolution_m"] = 0.001
    segment["analysis_caps"][0]["point_xyz_m"] = copy.deepcopy(points[0])
    segment["analysis_caps"][1]["point_xyz_m"] = copy.deepcopy(points[-1])


def _synthetic_inputs(right_points):
    inputs = _inputs()
    claim, targets = _two_target_claim(inputs["claim_grounding"])
    inputs["claim_grounding"] = claim
    by_composite = {
        composite: segment
        for segment in inputs["bounded_local_3d"]["bounded_local_3d_segments"]
        for composite in segment["source_route_composite_refs"]
    }
    left = by_composite[targets[0]["m3_5_composite_ref"]]
    right = by_composite[targets[1]["m3_5_composite_ref"]]
    _set_geometry(left, [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    _set_geometry(right, right_points)
    assert validate_mep_bounded_local_3d_segments(inputs["bounded_local_3d"]) == []
    return inputs


class MepBoundedClashVerificationTest(unittest.TestCase):
    def test_crossing_envelopes_close_bounded_clash_away_from_caps(self):
        payload = build_mep_bounded_clash_verification(
            **_synthetic_inputs([[5.0, -5.0, 0.0], [5.0, 5.0, 0.0]])
        )
        self.assertEqual(validate_mep_bounded_clash_verification(payload), [])
        record = payload["verification_records"][0]
        self.assertEqual(record["clash_status"], "confirmed_bounded_clash")
        self.assertTrue(record["confirmed_clash"])
        self.assertTrue(record["geometry_observation"]["intersection_away_from_analysis_caps"])
        self.assertGreater(record["geometry_observation"]["minimum_certified_penetration_m"], 0)
        self.assertFalse(record["authority"]["physical_run_identity_established"])
        self.assertFalse(record["authority"]["quantity_eligible"])

    def test_separated_envelopes_close_only_bounded_clear_result(self):
        payload = build_mep_bounded_clash_verification(
            **_synthetic_inputs([[5.0, -5.0, 3.0], [5.0, 5.0, 3.0]])
        )
        record = payload["verification_records"][0]
        self.assertEqual(record["clash_status"], "clear_in_bounded_scope")
        self.assertFalse(record["confirmed_clash"])
        self.assertGreater(record["geometry_observation"]["minimum_certified_clearance_m"], 0)

    def test_intersection_at_unresolved_analysis_cap_abstains(self):
        payload = build_mep_bounded_clash_verification(
            **_synthetic_inputs([[0.0, -5.0, 0.0], [0.0, 5.0, 0.0]])
        )
        record = payload["verification_records"][0]
        self.assertEqual(record["state"], "abstained")
        self.assertIn("intersection_overlaps_unresolved_analysis_cap", record["reasons"])
        self.assertNotIn("confirmed_clash", record)

    def test_incompatible_metric_frames_abstain(self):
        inputs = _synthetic_inputs([[5.0, -5.0, 0.0], [5.0, 5.0, 0.0]])
        inputs["bounded_local_3d"]["bounded_local_3d_segments"][1]["m1_metric_frame"][
            "id"
        ] = "metric-frame.other"
        payload = build_mep_bounded_clash_verification(**inputs)
        self.assertIn(
            "bounded_envelope_frames_are_incompatible",
            payload["verification_records"][0]["reasons"],
        )

    def test_unique_m6a_target_cannot_invent_second_clash_target(self):
        inputs = _inputs()
        claim, _ = _two_target_claim(inputs["claim_grounding"])
        record = claim["grounding_records"][0]
        record["record_type"] = "mep_claim_target_grounding"
        record["alternatives"] = record["alternatives"][:1]
        alternative = record["alternatives"][0]
        record["target_ref"] = alternative["target_ref"]
        record["m3_5_composite_ref"] = alternative["m3_5_composite_ref"]
        record["m5_partial_2_5d_centreline_ref"] = alternative["m5_partial_2_5d_centreline_ref"]
        record["ambiguity"] = {"eligible_2d_overlap_count": 1, "resolved": True}
        record["state"] = "accepted"
        record["epistemic_state"] = "derived"
        record["reasons"] = []
        claim["summary"]["accepted_unique_grounding_count"] = 1
        claim["summary"]["ambiguous_2d_overlap_candidate_count"] = 0
        self.assertEqual(validate_mep_claim_grounding(claim), [])
        inputs["claim_grounding"] = claim
        result = build_mep_bounded_clash_verification(**inputs)
        self.assertIn(
            "second_independent_bounded_target_missing",
            result["verification_records"][0]["reasons"],
        )

    def test_clearance_or_access_without_explicit_zone_fails_validation(self):
        payload = build_mep_bounded_clash_verification(
            **_synthetic_inputs([[5.0, -5.0, 3.0], [5.0, 5.0, 3.0]])
        )
        changed = copy.deepcopy(payload)
        changed["verification_records"][0]["finding_kind"] = "clearance"
        self.assertTrue(validate_mep_bounded_clash_verification(changed))

    def test_frozen_real_checkpoint_replays_all_claim_abstentions_and_zero_runs(self):
        payload = build_mep_bounded_clash_verification(**_inputs())
        stored = _load("pages_1a_1b.bounded-clash-verification.json")
        self.assertEqual(validate_mep_bounded_clash_verification(payload), [])
        self.assertEqual(validate_mep_bounded_clash_verification(stored), [])
        self.assertEqual(payload, stored)
        self.assertEqual(payload["summary"]["claim_count"], 15)
        self.assertEqual(payload["summary"]["confirmed_bounded_clash_count"], 0)
        self.assertEqual(payload["summary"]["abstention_count"], 15)
        self.assertEqual(payload["summary"]["whole_run_conclusion_count"], 0)
        self.assertTrue(
            all(
                row["reasons"] == ["m6a_claim_target_not_grounded"]
                for row in payload["verification_records"]
            )
        )

    def test_validator_rejects_m5b_m7_and_severity_authority(self):
        payload = build_mep_bounded_clash_verification(**_inputs())
        for key, value in (
            ("installed_length", 12.0),
            ("calculated_severity", "high"),
            ("quantity", 1),
            ("physical_run_identity_established", True),
        ):
            changed = copy.deepcopy(payload)
            changed["verification_records"][0][key] = value
            self.assertTrue(validate_mep_bounded_clash_verification(changed), key)


if __name__ == "__main__":
    unittest.main()
