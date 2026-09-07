import hashlib
import json
import unittest
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "output/pdf/mep_all_sheet_engineer_review_2026-09-02.pdf"
MANIFEST = PDF.with_suffix(".manifest.json")


class MepAllSheetAuditTest(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_everything_is_accounted_for_and_commercial_channels_stay_closed(self):
        coverage = self.manifest["coverage"]
        self.assertTrue(coverage["all_source_pages_present"])
        self.assertTrue(coverage["all_expected_rows_accounted_for"])
        self.assertTrue(coverage["all_marks_have_schedule_rows"])
        self.assertTrue(coverage["all_schedule_rows_have_reciprocal_links"])
        self.assertTrue(coverage["installed_lengths_all_null"])
        self.assertTrue(coverage["purchase_lengths_all_null"])
        self.assertEqual(coverage["bounded_3d_segment_count"], 15)
        self.assertEqual(len(self.manifest["pages"]), 18)

    def test_pdf_hash_page_count_and_links_reopen(self):
        self.assertEqual(hashlib.sha256(PDF.read_bytes()).hexdigest(), self.manifest["pdf_sha256"])
        with fitz.open(PDF) as pdf:
            self.assertEqual(len(pdf), self.manifest["pdf_page_count"])
            for page in self.manifest["pages"]:
                marked = pdf[page["marked_pdf_page_number"] - 1]
                self.assertGreaterEqual(len(marked.get_links()), page["reciprocal_link_count"] + 1)
                for number in page["schedule_pdf_page_numbers"]:
                    self.assertGreater(len(pdf[number - 1].get_links()), 0)

    def test_source_and_snapshot_hashes_are_bound(self):
        source = ROOT / "M&P mark-up against shop systems piping.pdf"
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), self.manifest["hashes"]["source_pdf"])
        self.assertRegex(self.manifest["snapshot_id"], r"^[0-9a-f]{64}$")
        self.assertFalse(self.manifest["authority"]["installed_length_established"])
        self.assertFalse(self.manifest["authority"]["purchase_length_established"])


if __name__ == "__main__":
    unittest.main()
