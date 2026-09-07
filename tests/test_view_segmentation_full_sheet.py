import unittest
from pathlib import Path

import fitz

from src.drawing_engine.core.object_agnostic_understanding import _union_find_components
from src.drawing_engine.core.view_segmentation import segment_views_by_titles


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "output" / "rc_staircase_search" / "candidate-08-staircase-page.pdf"


class TitleAnchoredViewSegmentationFullSheetTest(unittest.TestCase):
    def test_candidate_08_native_titles_scales_and_whitespace_freeze_three_primary_scopes(self):
        document = fitz.open(FIXTURE)
        self.addCleanup(document.close)
        page = document[0]
        drawings = page.get_drawings()
        words = page.get_text("words")
        self.assertEqual(len(words), 744)
        self.assertEqual(len(drawings), 27_235)
        self.assertEqual(page.get_images(full=True), [])

        views = [
            {
                "id": f"view.{index:03d}",
                "role_hypothesis": "drawing_view_candidate",
                "bbox_display": list(component["bbox"]),
                "primitive_refs": [
                    f"drawing[{drawing_index}]"
                    for drawing_index in component["drawing_indices"]
                ],
            }
            for index, component in enumerate(_union_find_components(page), start=1)
        ]
        text_roles = [
            {
                "id": f"text.{index:05d}",
                "text": word[4],
                "resolved_role": "unclassified_text",
                "bbox_display": list(word[:4]),
                "primitive_refs": [f"word[{index}]"],
            }
            for index, word in enumerate(words)
        ]
        geometry = [
            {
                "id": f"drawing[{index}]",
                "bbox_display": list(drawing["rect"]),
            }
            for index, drawing in enumerate(drawings)
        ]
        result = segment_views_by_titles(
            page.rect,
            views,
            text_roles,
            geometry_primitives=geometry,
        )

        resolved = [item for item in result["segments"] if item["state"] == "resolved"]
        staircase_details = next(
            item for item in resolved if item["title"] == "STAIRCASE RC DETAILS"
        )
        layout_plan = next(
            item for item in resolved if item["title"] == "STAIRCASE LAYOUT PLAN"
        )
        staircase_section = next(
            item
            for item in resolved
            if item["title"] == "SECTION A-A"
            and item["source_view_id"] == staircase_details["source_view_id"]
        )

        self.assertEqual(staircase_details["scale_ratio"], "1:25")
        self.assertEqual(staircase_section["scale_ratio"], "1:25")
        self.assertEqual(layout_plan["scale_ratio"], "1:50")
        self.assertEqual(len(staircase_details["primitive_refs"]), 2_393)
        self.assertEqual(len(staircase_section["primitive_refs"]), 1_147)
        self.assertEqual(
            staircase_details["excluded_primitive_refs"], ["drawing[19108]"]
        )
        self.assertEqual(
            staircase_details["spatial_partition"]["axis"],
            "vertical_whitespace",
        )
        self.assertLess(
            staircase_details["bbox_display"][2],
            staircase_section["bbox_display"][0],
        )
        self.assertEqual(
            set(staircase_details["primitive_refs"])
            & set(staircase_section["primitive_refs"]),
            set(),
        )
        parent = next(
            item for item in views if item["id"] == staircase_details["source_view_id"]
        )
        self.assertEqual(
            set(staircase_details["primitive_refs"])
            | set(staircase_section["primitive_refs"])
            | set(staircase_details["excluded_primitive_refs"]),
            set(parent["primitive_refs"]),
        )
        self.assertNotIn(
            "COLUMNS, BASES & STAIRCASE RC DETAILS",
            result["summary"]["resolved_titles"],
        )
        self.assertTrue(result["contract"]["titles_seed_roles_not_physical_identity"])


if __name__ == "__main__":
    unittest.main()
