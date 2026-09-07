import hashlib
import json
import unittest

from src.drawing_engine.disciplines.mep.mep_observed_takeoff import build_observed_mep_takeoff
from src.drawing_engine.disciplines.mep.mep_semantic_takeoff import build_semantic_mep_takeoff, validate_semantic_mep_takeoff


class SemanticMepTakeoffTests(unittest.TestCase):
    def _payload(self):
        occurrences = [
            {"id": "occ.a", "page_ref": "page.1", "route_target_ref": "comp.a",
             "route_composite_ref": "comp.a", "points_display": [[0, 0], [10, 0]],
             "projected_2d_length_m": 2.0, "source_fragment_refs": ["f.a"],
             "source_primitive_refs": ["d.a1", "d.a2"]},
            {"id": "occ.b", "page_ref": "page.1", "route_target_ref": "comp.b",
             "route_composite_ref": "comp.b", "points_display": [[10, 0], [20, 0]],
             "projected_2d_length_m": 2.0, "source_fragment_refs": ["f.b"],
             "source_primitive_refs": ["d.b1", "d.b2"]},
            {"id": "occ.c", "page_ref": "page.2", "route_target_ref": "stroke.c",
             "route_composite_ref": None, "points_display": [[0, 0], [5, 0]],
             "projected_2d_length_m": 1.0, "source_fragment_refs": ["f.c"],
             "source_primitive_refs": ["d.c"]},
        ]
        semantic = {"state": "accepted", "observed_values": [{
            "kind": "heating_hot_water_supply", "category": None,
            "raw_text": "HHWS", "terminology_entry_ref": "system.hws"}],
            "accepted_relation_refs": ["rel.system"]}
        unknown = {"state": "unknown", "observed_values": [], "accepted_relation_refs": []}
        segments = [
            {"id": "seg.a", "source_occurrence_refs": ["occ.a"],
             "semantic_overlay": {"system": semantic, "size": unknown, "elevation": unknown},
             "geometric_eligibility": {"basis": "replayed_m3_5_outline"}},
            {"id": "seg.b", "source_occurrence_refs": ["occ.b"],
             "semantic_overlay": {"system": semantic, "size": unknown, "elevation": unknown},
             "geometric_eligibility": {"basis": "replayed_m3_5_outline"}},
        ]
        baseline = build_observed_mep_takeoff(
            document={"source_pdf_sha256": "source"}, projected_occurrences=occurrences,
            canonical_segments=[], duplicate_candidates=[], bounded_local_3d_segments=[],
            network_segments=segments,
            junctions=[{"id": "join.ab", "state": "accepted", "run_pass_through": True,
                        "relation_type": "projected_collinear_boundary_join",
                        "segment_refs": ["seg.a", "seg.b"], "page_ref": "page.1",
                        "point_display": [10, 0], "evidence_refs": [], "reasons": []}],
            unresolved_boundaries=[{"id": "break.c", "page_ref": "page.2",
                                    "segment_ref": None, "source_occurrence_ref": "occ.c"}],
            hvac_items=[{"id": "equip.1", "page_ref": "page.2", "item_type": "equipment",
                         "item_class": "unit_heater", "source_observation_refs": [],
                         "source_fragment_refs": [], "unresolved_reasons": []}],
            sheet_coverage=[
                {"page_ref": "page.1", "page_number": 1, "sheet_number": "M1",
                 "source_first_review_state": "complete", "m3_observation_counts": {"crossing_count": 2}},
                {"page_ref": "page.2", "page_number": 2, "sheet_number": "M2",
                 "source_first_review_state": "complete", "m3_observation_counts": {"crossing_count": 0}},
            ])
        digest = hashlib.sha256((json.dumps(baseline, sort_keys=True) + "\n").encode()).hexdigest()
        return build_semantic_mep_takeoff(
            baseline=baseline, baseline_sha256=digest, network_segments=segments,
            network_junctions=[{"id": "join.ab", "state": "accepted", "run_pass_through": True,
                                "relation_type": "projected_collinear_boundary_join",
                                "segment_refs": ["seg.a", "seg.b"]}],
            projected_runs=[{"id": "run.ab", "segment_refs": ["seg.a", "seg.b"]}],
            bounded_local_3d_segments=[])

    def test_certified_contiguous_intervals_merge_and_unoutlined_stays_unresolved(self):
        payload = self._payload()
        self.assertEqual(len(payload["semantic_route_groups"]), 1)
        group = payload["semantic_route_groups"][0]
        self.assertEqual(group["merge_certificate"]["merged_interval_count"], 2)
        self.assertEqual(group["length_channels"]["canonicalized_projected_length_m"], 4.0)
        self.assertTrue(group["representation"]["sidewalls_collapsed_once"])
        self.assertEqual(len(payload["unresolved_route_classes"]), 1)

    def test_discrete_counts_and_commercial_authority_remain_separate(self):
        payload = self._payload()
        self.assertEqual(payload["acceptance_gate"]["discrete_classification_total"], 2)
        self.assertEqual(payload["discrete_category_counts"]["unresolved_symbol"]["observed_occurrence_count"], 1)
        self.assertEqual(payload["discrete_category_counts"]["equipment"]["observed_occurrence_count"], 1)
        self.assertIsNone(payload["length_channels"]["installed_length_m"])
        self.assertIsNone(payload["length_channels"]["purchase_length_m"])
        self.assertTrue(all(item["counts"]["physical_instance_count"] is None
                            for item in payload["discrete_items"]))
        self.assertTrue(all(item["connection_standard"] is None and item["exact_sku"] is None
                            for item in payload["discrete_items"]))
        self.assertEqual(len(payload["observed_item_row_assignments"]), 2)

    def test_review_overlay_does_not_promote_evidence_and_validation_passes(self):
        payload = self._payload()
        self.assertTrue(payload["engineer_review_overlays"])
        self.assertTrue(all(row["promotes_to_direct_evidence"] is False
                            for row in payload["engineer_review_overlays"]))
        self.assertEqual(validate_semantic_mep_takeoff(payload), [])


if __name__ == "__main__":
    unittest.main()
