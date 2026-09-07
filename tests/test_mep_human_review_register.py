"""Human unresolved registers retain every observation without implying items."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import fitz

from src.drawing_engine.audit.render_mep_partial_audit import _human_review_entries, _item_presentation, render_items
from src.drawing_engine.project.takeoff_intelligence import canonical_sha256


def _manifest():
    pages = [{"page_ref": f"page.{number}", "pdf_page_number": number,
              "drawing_sheet_number": f"TEST-{number}",
              "discovery_record": {"discovery_state": "bounded_search_processed", "regions": []}}
             for number in (1, 2)]
    row = {"id": "mep_item_occurrence.accepted", "source": {"page_ref": "page.2", "pdf_page_number": 2,
            "drawing_sheet_number": "TEST-2"}, "centreline_points_display": [[350, 300], [350, 500]],
           "occurrence": {"system": {"kind": "heating_hot_water_supply"},
               "typed_dimensions": [{"dimension_type": "nominal_size", "value": 1.5, "unit": "in"}],
               "elevations": [], "counts": {"physical_instance_count": None}},
           "takeoff_line": {"value_channels": {}}}
    proposals = [{"id": f"mep_interpretation_proposal.text-{index}", "page_ref": "page.1",
                  "description": '3/4" HHWS' if index % 9 else "Observed callout " + "continued text " * 20,
                  "reason": "no_complete_native_dot_leader; no_unique_leader_contact_target; incomplete_relevant_competitor_search",
                  "source_geometry": {"bbox_display": [100, 100 + index * 3, 200, 112 + index * 3]},
                  "quantity_eligible": False} for index in range(84)]
    proposals.extend({"id": f"mep_outlined_route_composite.shape-{index}", "page_ref": "page.1",
                      "description": "Unlabelled closed envelope - geometric proposal only",
                      "reason": "No certified system, size or equipment identity; not an accepted item.",
                      "source_geometry": {"points_display": [[200, 200 + index], [300, 200 + index]]},
                      "quantity_eligible": False} for index in range(310))
    proposals.append({"id": "mep_native_text_line.room", "page_ref": "page.2", "description": "OM-138",
                      "reason": "Unclassified tag; equipment identity and explicit port connectivity are unresolved.",
                      "source_geometry": {"bbox_display": [500, 250, 550, 265]}, "quantity_eligible": False})
    return {"document": {}, "execution_mode": "automatic_frozen_replay", "pages": pages,
            "item_rows": [row], "unresolved_proposals": proposals}


class MepHumanReviewRegisterTest(unittest.TestCase):
    def test_aliases_preserve_repeated_text_and_do_not_change_accepted_items(self):
        manifest = _manifest()
        accepted = deepcopy(_item_presentation(manifest))
        entries = _human_review_entries(manifest)
        self.assertEqual(manifest["item_rows"], accepted)
        self.assertEqual(len(entries), len(manifest["unresolved_proposals"]))
        self.assertEqual(len({entry["label"] for entry in entries}), len(entries))
        repeated = [entry for entry in entries if entry["observed_text"] == '3/4" HHWS']
        self.assertGreater(len(repeated), 1)
        self.assertTrue(all(entry["label"].startswith("U") for entry in repeated))
        room = next(entry for entry in entries if entry["observed_text"] == "OM-138")
        self.assertEqual(room["reason"], "Unclassified text; Identity and port unresolved")
        self.assertFalse(any(entry["quantity_eligible"] for entry in entries))
        self.assertFalse(manifest["pages"][0]["needs_review"]["counts_are_parts"])
        manifest["unresolved_proposals"].reverse()
        self.assertEqual(entries, _human_review_entries(manifest))

    def test_register_requires_source_geometry_and_processed_source(self):
        manifest = _manifest()
        manifest["unresolved_proposals"][0]["source_geometry"] = {}
        with self.assertRaisesRegex(ValueError, "exact source geometry"):
            _human_review_entries(manifest)
        manifest = _manifest()
        manifest["pages"][0]["discovery_record"]["discovery_state"] = "not_processed"
        with self.assertRaisesRegex(ValueError, "unprocessed page"):
            _human_review_entries(manifest)

    def test_pdf_register_paginates_all_rows_and_shape_links_without_overflow(self):
        manifest = _manifest()
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.pdf", Path(directory) / "review.pdf"
            with fitz.open() as pdf:
                for number in (1, 2):
                    page = pdf.new_page(width=700, height=900)
                    page.insert_text((40, 40), f"Original source {number}")
                pdf.save(source)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest["document"] = {"source_pdf_sha256": digest, "page_count": 2}
            before = deepcopy(manifest)
            expected_items = deepcopy(_item_presentation(deepcopy(manifest)))
            result = render_items(source, output, manifest, source_rendering="vector")
            self.assertEqual(manifest, before)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)
            for key in ("id", "source", "occurrence", "presentation", "centreline_points_display", "takeoff_line"):
                self.assertEqual(result["item_rows"][0][key], expected_items[0][key])
            self.assertEqual(result["unresolved_proposals"], before["unresolved_proposals"])
            register = result["human_review_register"]
            self.assertEqual(register["source_proposals_sha256"], canonical_sha256(before["unresolved_proposals"]))
            entries = register["entries"]
            self.assertEqual({entry["proposal_ref"] for entry in entries},
                             {proposal["id"] for proposal in before["unresolved_proposals"]})
            self.assertGreater(len({entry["audit_page_number"] for entry in entries if entry["kind"] == "text"}), 2)
            self.assertGreater(len({entry["audit_page_number"] for entry in entries if entry["kind"] == "geometry"}), 1)
            self.assertEqual(result["presentation"]["needs_review_page_numbers"], register["audit_page_numbers"])
            with fitz.open(output) as pdf:
                self.assertEqual(json.loads(pdf.embfile_get("audit-manifest.json")), result)
                all_text = "\n".join(page.get_text() for page in pdf).replace("\u00a0", " ").replace("\u00ad", "-")
                self.assertNotIn("mep_interpretation_proposal", all_text)
                self.assertNotIn("mep_outlined_route_composite", all_text)
                self.assertNotIn("mep_native_text_line", all_text)
                self.assertIn("NOT FOUND ELEMENTS", all_text)
                self.assertIn("not parts or quantities", all_text)
                self.assertIn("OM-138", all_text)
                toc_pages = {row[2] for row in pdf.get_toc() if row[1].startswith("Needs review")}
                self.assertEqual(toc_pages, set(register["audit_page_numbers"]))
                links_by_page = {number: pdf[number - 1].get_links() for number in register["audit_page_numbers"]}
                text_by_page = {number: pdf[number - 1].get_text() for number in register["audit_page_numbers"]}
                page_rects = {}
                for entry in entries:
                    page = pdf[entry["audit_page_number"] - 1]
                    rect = fitz.Rect(entry["audit_rect_display"])
                    self.assertTrue(page.rect.contains(rect))
                    self.assertLessEqual(rect.y1, 765)
                    self.assertIn(entry["label"], text_by_page[entry["audit_page_number"]])
                    self.assertTrue(any(link["page"] == entry["pdf_page_number"]
                                        and abs(link["to"].x - entry["source_point_display"][0]) < .01
                                        and abs(link["to"].y - entry["source_point_display"][1]) < .01
                                        for link in links_by_page[entry["audit_page_number"]]))
                    others = page_rects.setdefault(page.number, [])
                    self.assertFalse(any(rect.intersects(other) for other in others))
                    others.append(rect)
                for record in result["pages"]:
                    review = record["needs_review"]
                    page = pdf[record["audit_source_page_number"] - 1]
                    self.assertTrue(any(link["page"] == review["audit_page_numbers"][0] - 1 for link in page.get_links()))
                # Inspect actual glyph bounds in every appendix, not just row boxes.
                for number in register["audit_page_numbers"]:
                    page = pdf[number - 1]
                    for word in page.get_text("words"):
                        self.assertTrue(page.rect.contains(fitz.Rect(word[:4])))


if __name__ == "__main__":
    unittest.main()
