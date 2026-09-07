import copy
import json
import unittest
from pathlib import Path

from src.drawing_engine.disciplines.mep.mep_bounded_local_3d import (
    LABEL,
    build_mep_bounded_local_3d_segments,
    validate_mep_bounded_local_3d_segments,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "fixtures" / "mep" / "m_and_p_coordination"
CHECKPOINT_ROOT = FIXTURE_ROOT / "real_m2_m5_checkpoint"
FROZEN_M5A = CHECKPOINT_ROOT / "pages_1a_1b.bounded-local-3d.json"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs():
    return {
        "sheet_registry": _load(FIXTURE_ROOT / "m_and_p_coordination.sheet-registry.json"),
        "outlined_route_composites": _load(
            CHECKPOINT_ROOT / "pages_1a_1b.outlined-route-composites.json"
        ),
        "attribute_bindings": _load(
            CHECKPOINT_ROOT / "pages_1a_1b.attribute-bindings.json"
        ),
        "cross_sheet_runs": _load(
            CHECKPOINT_ROOT / "pages_1a_1b.cross-sheet-runs.json"
        ),
    }


def _build(inputs=None):
    return build_mep_bounded_local_3d_segments(**(inputs or _inputs()))


class MepBoundedLocal3dTest(unittest.TestCase):
    def test_frozen_real_checkpoint_validates_and_replays_exactly(self):
        frozen = _load(FROZEN_M5A)
        self.assertEqual(validate_mep_bounded_local_3d_segments(frozen), [])
        self.assertEqual(_build(), frozen)

    def test_real_1a_1b_closes_two_bounded_segments_without_run_or_quantity(self):
        payload = _build()
        self.assertEqual(validate_mep_bounded_local_3d_segments(payload), [])
        self.assertEqual(payload["summary"]["candidate_count"], 2)
        self.assertEqual(
            payload["summary"]["accepted_bounded_local_3d_segment_count"], 2
        )
        self.assertEqual(payload["summary"]["physical_run_hypothesis_count"], 0)
        self.assertEqual(payload["summary"]["unresolved_vertical_span_count"], 0)
        self.assertEqual(payload["unresolved_vertical_spans"], [])
        self.assertNotIn("physical_run_hypotheses", payload)
        self.assertTrue(all(row["label"] == LABEL for row in payload["bounded_local_3d_segments"]))
        self.assertTrue(
            all(len(row["source_page_occurrence_refs"]) == 2 for row in payload["bounded_local_3d_segments"])
        )
        self.assertTrue(
            all(len(row["analysis_caps"]) == 2 for row in payload["bounded_local_3d_segments"])
        )
        for row in payload["bounded_local_3d_segments"]:
            self.assertFalse(row["authority"]["physical_run_identity_established"])
            self.assertFalse(row["authority"]["installed_length_emitted"])
            self.assertFalse(row["authority"]["confirmed_clash_established"])
            self.assertFalse(row["quantity_eligible"])
            self.assertFalse(
                row["physical_envelope_dimension"]["nominal_size_used_as_physical_dimension"]
            )
            self.assertEqual(row["elevation"]["reference_basis"], "bottom")
            self.assertGreater(
                row["elevation"]["centreline_elevation_m"],
                row["elevation"]["reference_elevation_m"],
            )

        supply = next(
            row
            for row in payload["bounded_local_3d_segments"]
            if row["system"]["kind"] == "heating_hot_water_supply"
        )
        dimension = supply["physical_envelope_dimension"]
        self.assertAlmostEqual(dimension["maximum_width_residual_m"], 0.002032, places=6)
        self.assertAlmostEqual(dimension["agreement_tolerance_m"], 0.012192, places=6)
        self.assertIn("lineweight", dimension["agreement_tolerance_derivation"])

    def test_nominal_size_alone_cannot_supply_outer_width(self):
        inputs = _inputs()
        for row in inputs["outlined_route_composites"]["accepted_composites"]:
            row["derived_geometry"]["corridor_width_display_points"] = None
        with self.assertRaises(ValueError):
            _build(inputs)

    def test_conflicting_duplicate_width_abstains(self):
        inputs = _inputs()
        rows = inputs["outlined_route_composites"]["accepted_composites"]
        changed_id = rows[0]["id"]
        for row in [*rows, *inputs["outlined_route_composites"]["candidates"]]:
            if row["id"] == changed_id:
                row["derived_geometry"]["corridor_width_display_points"] = 20.0
        payload = _build(inputs)
        self.assertEqual(payload["summary"]["accepted_bounded_local_3d_segment_count"], 1)
        abstention = next(row for row in payload["segment_candidates"] if row["state"] == "abstained")
        self.assertIn("duplicate_occurrence_outer_width_disagreement", abstention["reasons"])

    def test_absent_scale_and_missing_registration_fail_closed(self):
        scale_inputs = _inputs()
        page_ref = scale_inputs["outlined_route_composites"]["accepted_composites"][0]["page_ref"]
        page = next(row for row in scale_inputs["sheet_registry"]["pages"] if row["page_ref"] == page_ref)
        page["fields"]["scale"]["state"] = "unknown"
        page["fields"]["scale"]["drawing_inches_per_paper_inch"] = None
        payload = _build(scale_inputs)
        self.assertEqual(payload["summary"]["accepted_bounded_local_3d_segment_count"], 0)
        self.assertTrue(
            all("accepted_plan_scale_missing" in row["reasons"] for row in payload["segment_candidates"])
        )

        frame_inputs = _inputs()
        pair = {
            row["page_ref"]
            for row in frame_inputs["outlined_route_composites"]["accepted_composites"]
        }
        registration = next(
            row
            for row in frame_inputs["sheet_registry"]["adjoining_sheet_transforms"]
            if {row["source_page_ref"], row["target_page_ref"]} == pair
        )
        registration["state"] = "abstained"
        registration["matrix_source_display_to_target_display"] = None
        payload = _build(frame_inputs)
        self.assertEqual(payload["summary"]["accepted_bounded_local_3d_segment_count"], 0)
        self.assertTrue(
            all("accepted_registered_plan_frame_missing" in row["reasons"] for row in payload["segment_candidates"])
        )

    def test_page_scale_must_agree_with_registration_similarity(self):
        inputs = _inputs()
        page_ref = inputs["outlined_route_composites"]["accepted_composites"][0]["page_ref"]
        page = next(row for row in inputs["sheet_registry"]["pages"] if row["page_ref"] == page_ref)
        page["fields"]["scale"]["drawing_inches_per_paper_inch"] = 47.0
        payload = _build(inputs)
        self.assertEqual(payload["summary"]["accepted_bounded_local_3d_segment_count"], 0)
        self.assertTrue(
            all(
                "page_scales_conflict_with_registration_similarity" in row["reasons"]
                for row in payload["segment_candidates"]
            )
        )

    def test_unknown_or_duct_shape_cannot_use_circular_pipe_kernel(self):
        inputs = _inputs()
        for relation in inputs["attribute_bindings"]["relations"]:
            if relation["relation_type"] == "route_size":
                relation["candidate"] = {
                    "kind": "duct_size",
                    "width": 24.0,
                    "depth": 12.0,
                    "unit": "in",
                }
        payload = _build(inputs)
        self.assertEqual(payload["summary"]["accepted_bounded_local_3d_segment_count"], 0)
        self.assertTrue(
            all(
                "unsupported_or_unknown_route_envelope_shape" in row["reasons"]
                for row in payload["segment_candidates"]
            )
        )

    def test_conflicting_elevation_and_invalid_cap_fail_closed(self):
        elevation_inputs = _inputs()
        relations = [
            row
            for row in elevation_inputs["attribute_bindings"]["relations"]
            if row["relation_type"] == "route_elevation"
        ]
        relations[0]["candidate"]["value"] += 1.0
        payload = _build(elevation_inputs)
        self.assertEqual(payload["summary"]["accepted_bounded_local_3d_segment_count"], 1)
        self.assertIn(
            "duplicate_occurrence_route_elevation_disagreement",
            next(row for row in payload["segment_candidates"] if row["state"] == "abstained")["reasons"],
        )

        cap_inputs = _inputs()
        changed_id = cap_inputs["outlined_route_composites"]["accepted_composites"][0]["id"]
        for row in [
            *cap_inputs["outlined_route_composites"]["accepted_composites"],
            *cap_inputs["outlined_route_composites"]["candidates"],
        ]:
            if row["id"] == changed_id:
                row["certificates"]["explicit_envelope_closing_feature"] = False
        with self.assertRaises(ValueError):
            _build(cap_inputs)

    def test_unresolved_riser_destination_is_preserved_without_finite_extent(self):
        inputs = _inputs()
        bindings = inputs["attribute_bindings"]
        relation = copy.deepcopy(bindings["relations"][0])
        relation["id"] = "mep_page_local_binding_relation.synthetic_riser"
        relation["relation_type"] = "riser_drop"
        relation["target_kind"] = "route_endpoint"
        relation["target_refs"] = ["mep_route_endpoint.synthetic"]
        relation["candidate"] = {"kind": "riser", "direction": "up"}
        bindings["relations"].append(relation)
        bindings["summary"]["accepted_relation_count"] += 1
        payload = _build(inputs)
        self.assertEqual(payload["summary"]["unresolved_vertical_span_count"], 1)
        span = payload["unresolved_vertical_spans"][0]
        self.assertEqual(span["state"], "unknown")
        self.assertNotIn("vertical_extent_m", span)
        self.assertNotIn("resolved_3d_length_m", span)

    def test_reordering_preserves_segment_identity_and_geometry(self):
        baseline = _build()
        inputs = _inputs()
        inputs["outlined_route_composites"]["accepted_composites"].reverse()
        inputs["attribute_bindings"]["relations"].reverse()
        inputs["cross_sheet_runs"]["canonical_projected_segments"].reverse()
        replay = _build(inputs)
        self.assertEqual(
            replay["bounded_local_3d_segments"], baseline["bounded_local_3d_segments"]
        )

    def test_validator_recursively_rejects_m5b_m6_m7_authority(self):
        payload = _build()
        for key, value in (
            ("physical_run_identity_established", True),
            ("confirmed_clash", True),
            ("calculated_severity", "high"),
            ("installed_length", 6.4),
            ("quantity_eligible", True),
        ):
            changed = copy.deepcopy(payload)
            changed["bounded_local_3d_segments"][0][key] = value
            self.assertTrue(validate_mep_bounded_local_3d_segments(changed), key)


if __name__ == "__main__":
    unittest.main()
