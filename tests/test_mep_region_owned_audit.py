import hashlib
import json
import unittest
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "output/pdf/mep_all_sheet_engineer_review_region_owned_2026-09-03.pdf"
MANIFEST = PDF.with_suffix(".manifest.json")


class MepRegionOwnedAuditTest(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(MANIFEST.read_text())

    def test_every_source_sheet_has_one_clean_sheet_schedule_and_appendix(self):
        self.assertEqual(self.manifest["status"], "accepted_region_owned_audit")
        self.assertEqual(len(self.manifest["pages"]), 18)
        self.assertEqual(self.manifest["pdf_page_count"], 55)
        for source_number, row in enumerate(self.manifest["pages"], 1):
            self.assertEqual(row["source_page_number"], source_number)
            self.assertEqual(row["schedule_pdf_page_number"], row["main_marked_pdf_page_number"] + 1)
            self.assertEqual(row["evidence_appendix_pdf_page_number"], row["main_marked_pdf_page_number"] + 2)
            self.assertEqual(row["main_non_route_mark_count"], 0)
            self.assertEqual(row["main_low_level_crossing_boundary_mark_count"], 0)
            self.assertIn("review_required_symbol_count", row)
            self.assertEqual(
                row["rendered_route_occurrence_count"],
                row["certified_system_route_occurrence_count"]
                + row["colour_correlated_review_route_occurrence_count"]
                + row["unresolved_valid_view_candidate_count"])

    def test_primary_layer_is_region_owned_and_commercial_lengths_stay_null(self):
        coverage = self.manifest["coverage"]
        self.assertTrue(coverage["zero_non_route_geometry_on_primary_system_layer"])
        self.assertTrue(coverage["zero_low_level_crossing_boundary_marks_on_primary_layer"])
        self.assertTrue(coverage["every_identified_route_has_accepted_system"])
        self.assertTrue(coverage["every_route_occurrence_marked"])
        self.assertTrue(coverage["every_route_mark_has_trace_schedule_link"])
        self.assertEqual(coverage["rendered_route_occurrence_mark_count"], 7779)
        self.assertEqual(coverage["colour_correlated_review_route_occurrence_mark_count"], 816)
        self.assertEqual(coverage["page_system_trace_count"], 52)
        self.assertTrue(coverage["identified_unknown_visual_separation"])
        self.assertFalse(coverage["purchase_field_rendered"])
        self.assertEqual(self.manifest["authority"]["primary_source_reference_opacity"], .32)
        self.assertTrue(self.manifest["authority"]["unfaded_source_available_in_evidence_appendix"])
        self.assertEqual(coverage["excluded_non_route_appendix_mark_count"], 104)
        self.assertIsNone(coverage["installed_length_m"])
        self.assertIsNone(coverage["purchase_length_m"])

    def test_pdf_hash_page_topology_and_reciprocal_links_reopen(self):
        self.assertEqual(hashlib.sha256(PDF.read_bytes()).hexdigest(), self.manifest["pdf_sha256"])
        with fitz.open(PDF) as document:
            self.assertEqual(len(document), 55)
            for row in self.manifest["pages"]:
                main = document[row["main_marked_pdf_page_number"] - 1]
                schedule = document[row["schedule_pdf_page_number"] - 1]
                appendix = document[row["evidence_appendix_pdf_page_number"] - 1]
                self.assertGreaterEqual(len(main.get_links()), 1 + row["route_schedule_reciprocal_link_count"])
                self.assertGreaterEqual(len(schedule.get_links()), 2)
                self.assertGreaterEqual(len(appendix.get_links()), 2)

    def test_dense_source_page_five_is_one_review_triplet(self):
        page_five = self.manifest["pages"][4]
        self.assertEqual(page_five["main_marked_pdf_page_number"], 14)
        self.assertEqual(page_five["schedule_pdf_page_number"], 15)
        self.assertEqual(page_five["evidence_appendix_pdf_page_number"], 16)
        self.assertEqual(page_five["accepted_system_route_group_count"], 17)
        self.assertGreater(page_five["colour_correlated_review_route_occurrence_count"], 0)
        self.assertGreater(page_five["rendered_route_occurrence_count"], 1000)
        self.assertEqual(page_five["main_low_level_crossing_boundary_mark_count"], 0)


if __name__ == "__main__":
    unittest.main()
