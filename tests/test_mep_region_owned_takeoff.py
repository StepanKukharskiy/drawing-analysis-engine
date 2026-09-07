import json
import unittest
from pathlib import Path

from src.drawing_engine.disciplines.mep.mep_region_owned_takeoff import validate_region_owned_takeoff


ROOT = Path(__file__).resolve().parents[1]
OBSERVED = ROOT / "output/mep-observed-takeoff-region-owned-2026-09-03/observed-takeoff.json"
SEMANTIC = ROOT / "output/mep-semantic-takeoff-region-owned-2026-09-03/semantic-takeoff.json"
BENCHMARK = ROOT / "output/mep-partial-network-benchmark-region-owned-2026-09-03/benchmark.json"
REJECTION = ROOT / "output/mep-development-rejection-2026-09-03/rejection.json"


class MepRegionOwnedTakeoffTest(unittest.TestCase):
    def setUp(self):
        self.observed = json.loads(OBSERVED.read_text())
        self.semantic = json.loads(SEMANTIC.read_text())
        self.benchmark = json.loads(BENCHMARK.read_text())

    def test_region_filter_removes_non_route_geometry_before_length_channels(self):
        self.assertEqual(validate_region_owned_takeoff(self.observed), [])
        channels = self.observed["length_channels"]
        self.assertGreater(channels["excluded_non_route_projected_length_m"], 351.3)
        self.assertLess(channels["valid_view_visible_projected_length_m"], 11735.994)
        retained = {item["occurrence_ref"] for row in self.observed["route_segment_ledger"]
                    for item in row["occurrences"]}
        self.assertFalse(retained.intersection(
            row["source_occurrence_ref"] for row in json.loads(
                (ROOT / "output/mep-sheet-region-ownership-2026-09-03/sheet-region-ownership.json").read_text()
            )["non_route_drawing_content"]))

    def test_only_region_owned_known_system_routes_enter_semantic_total(self):
        self.assertEqual(validate_region_owned_takeoff(self.semantic), [])
        self.assertTrue(self.semantic["semantic_route_groups"])
        self.assertTrue(all(row["semantic_signature"]["system"]["state"] == "accepted"
                            and row["semantic_signature"]["system"]["value"].get("kind")
                            for row in self.semantic["semantic_route_groups"]))
        self.assertFalse(self.semantic["acceptance_gate"]["unknown_geometry_presented_as_identified_pipe"])
        self.assertAlmostEqual(
            sum(row["accepted_system_semantic_length_m"] for row in self.semantic["per_page_schedule"]),
            self.semantic["length_channels"]["accepted_system_semantic_centreline_length_m"])

    def test_independent_benchmark_and_commercial_nulls_close(self):
        self.assertEqual(validate_region_owned_takeoff(self.benchmark), [])
        self.assertEqual(self.benchmark["independent_length_validation"]["status"], "passed")
        self.assertEqual(self.benchmark["independent_length_validation"]["delta_m"], 0.0)
        self.assertIsNone(self.benchmark["length_channels"]["installed_length_m"])
        self.assertIsNone(self.benchmark["length_channels"]["purchase_length_m"])
        self.assertTrue(self.benchmark["source_first_region_validation"]["negative_sample"])

    def test_superseded_semantic_benchmark_and_audit_are_rejected(self):
        rejection = json.loads(REJECTION.read_text())
        old_semantic = json.loads((ROOT / "output/mep-semantic-takeoff-2026-09-02/semantic-takeoff.json").read_text())
        old_benchmark = json.loads((ROOT / "output/mep-partial-network-benchmark-2026-09-02/benchmark.json").read_text())
        old_audit = json.loads((ROOT / "output/pdf/mep_all_sheet_engineer_review_2026-09-02.manifest.json").read_text())
        self.assertEqual(rejection["status"], "development_rejected")
        self.assertEqual(rejection["reason"], "sheet_region_ownership_not_closed")
        self.assertEqual(old_semantic["development_status"], "development_rejected")
        self.assertEqual(old_benchmark["benchmark_status"], "development_rejected")
        self.assertEqual(old_audit["development_status"], "development_rejected")


if __name__ == "__main__":
    unittest.main()
