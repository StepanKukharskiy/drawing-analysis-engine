import hashlib
import json
import unittest
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "output/pdf/mep_marked_element_atlas_2026-09-03.pdf"
MANIFEST = PDF.with_suffix(".manifest.json")


class MepElementLabelAtlasTest(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(MANIFEST.read_text())

    def test_all_primary_routes_and_discrete_items_have_local_labels(self):
        coverage = self.manifest["coverage"]
        self.assertEqual(coverage["expected_identified_or_native_colour_route_segment_count"], 1472)
        self.assertEqual(coverage["printed_identified_or_native_colour_route_segment_label_count"], 1472)
        self.assertEqual(coverage["expected_discrete_item_count"], 1366)
        self.assertEqual(coverage["printed_discrete_item_label_count"], 1366)
        self.assertTrue(coverage["every_identified_or_native_colour_route_has_system_diameter_length_label"])
        self.assertTrue(coverage["every_discrete_item_has_classification_or_unknown_label"])
        self.assertEqual(coverage["unclassified_background_candidate_count"], 6307)

    def test_atlas_is_sequential_and_requires_no_click_or_purchase_lookup(self):
        authority = self.manifest["authority"]
        self.assertFalse(authority["click_navigation_required"])
        self.assertFalse(authority["purchase_data_rendered"])
        self.assertIsNone(self.manifest["coverage"]["installed_length_m"])
        self.assertIsNone(self.manifest["coverage"]["purchase_length_m"])
        self.assertEqual(hashlib.sha256(PDF.read_bytes()).hexdigest(), self.manifest["pdf_sha256"])
        with fitz.open(PDF) as document:
            self.assertEqual(len(document), self.manifest["pdf_page_count"])
            detail_text = "\n".join(document[index].get_text() for index in range(19, 22))
            self.assertIn("DIA", detail_text)
            self.assertIn("L2D", detail_text)
            self.assertTrue("ELBOW" in detail_text or "UNKNOWN:" in detail_text)


if __name__ == "__main__":
    unittest.main()
