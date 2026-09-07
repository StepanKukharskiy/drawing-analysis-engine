import unittest

from src.drawing_engine.disciplines.mep.mep_observed_takeoff import build_observed_mep_takeoff, validate_observed_mep_takeoff


class ObservedMepTakeoffTests(unittest.TestCase):
    def _payload(self):
        occurrences = [
            {"id": "occ.a", "page_ref": "page.1", "route_target_ref": "route.a",
             "points_display": [[0, 0], [10, 0]], "projected_2d_length_m": 2.0,
             "source_fragment_refs": ["frag.a"], "source_primitive_refs": ["drawing[1]"]},
            {"id": "occ.b", "page_ref": "page.2", "route_target_ref": "route.b",
             "points_display": [[0, 0], [10, 0]], "projected_2d_length_m": 2.0,
             "source_fragment_refs": ["frag.b"], "source_primitive_refs": ["drawing[2]"]},
            {"id": "occ.c", "page_ref": "page.1", "route_target_ref": "route.c",
             "points_display": [[5, 5], [5, 8]], "projected_2d_length_m": 0.6,
             "source_fragment_refs": ["frag.c"], "source_primitive_refs": ["drawing[3]"]},
            {"id": "occ.d", "page_ref": "page.2", "route_target_ref": "route.d",
             "points_display": [[5, 5], [5, 8]], "projected_2d_length_m": 0.6,
             "source_fragment_refs": ["frag.d"], "source_primitive_refs": ["drawing[4]"]},
        ]
        return build_observed_mep_takeoff(
            document={"source_pdf_sha256": "x"},
            projected_occurrences=occurrences,
            canonical_segments=[{
                "id": "canonical.ab", "source_page_occurrence_refs": ["occ.a", "occ.b"],
                "representative_projected_2d_length_m": 2.0,
                "overlap_duplicate_relation_refs": ["duplicate.ab"]}],
            duplicate_candidates=[{
                "id": "duplicate.cd", "state": "abstained",
                "source_route_target_ref": "route.c", "target_route_target_ref": "route.d"}],
            bounded_local_3d_segments=[{
                "id": "bounded.ab", "state": "accepted",
                "canonical_projected_segment_ref": "canonical.ab",
                "centreline_points_xyz_m": [[0, 0, 1], [2, 0, 1]],
                "source_page_occurrence_refs": ["occ.a", "occ.b"]}],
            network_segments=[],
            junctions=[{
                "id": "junction.1", "state": "accepted", "page_ref": "page.1",
                "relation_type": "projected_native_bend", "point_display": [10, 0],
                "evidence_refs": ["connection.1"], "reasons": []}],
            unresolved_boundaries=[{
                "id": "boundary.1", "page_ref": "page.1", "segment_ref": "segment.1",
                "source_occurrence_ref": "occ.c",
                "reason": "complete_terminal_not_certified"}],
            hvac_items=[{
                "id": "hvac.1", "page_ref": "page.2", "item_type": "equipment",
                "item_class": "unit_heater", "candidate": {"tag": "HUH-1"},
                "source_observation_refs": ["text.1"], "source_fragment_refs": [],
                "unresolved_reasons": ["port_unresolved"]}],
            sheet_coverage=[
                {"page_ref": "page.1", "source_first_review_state": "complete"},
                {"page_ref": "page.2", "source_first_review_state": "complete"}],
        )

    def test_channels_survive_without_physical_run(self):
        payload = self._payload()
        self.assertEqual(payload["length_channels"]["visible_projected_length_m"], 5.2)
        self.assertEqual(payload["length_channels"]["certified_canonicalized_projected_length_m"], 2.0)
        self.assertEqual(payload["length_channels"]["duplicate_identity_unresolved_visible_length_m"], 1.2)
        self.assertEqual(payload["length_channels"]["bounded_local_3d_length_m"], 2.0)
        self.assertIsNone(payload["length_channels"]["installed_length_m"])
        self.assertIsNone(payload["length_channels"]["purchase_length_m"])

    def test_counts_and_endpoint_markers_remain_separate(self):
        payload = self._payload()
        canonical = next(row for row in payload["route_segment_ledger"]
                         if row["canonical_projected_segment_ref"] == "canonical.ab")
        self.assertEqual(canonical["counts"], {
            "observed_occurrence_count": 2,
            "deduplicated_projected_count": 1,
            "physical_instance_count": None})
        unresolved = [row for row in payload["route_segment_ledger"]
                      if row["duplicate_competitor_refs"]]
        self.assertEqual(len(unresolved), 2)
        self.assertTrue(all(row["counts"]["deduplicated_projected_count"] is None
                            for row in unresolved))
        self.assertEqual(len(payload["unresolved_route_ends"][0]["endpoint_markers"]), 2)
        self.assertIsNone(payload["item_occurrence_ledger"][0]["counts"]["physical_instance_count"])

    def test_validation_accepts_fail_closed_payload(self):
        self.assertEqual(validate_observed_mep_takeoff(self._payload()), [])


if __name__ == "__main__":
    unittest.main()
