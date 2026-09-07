import unittest

import fitz

from src.drawing_engine.core.view_segmentation import segment_views_by_titles


class TitleAnchoredViewSegmentationTest(unittest.TestCase):
    def test_unique_titles_certify_distinct_geometry_segments(self):
        result = segment_views_by_titles(
            fitz.Rect(0, 0, 800, 600),
            [
                {"id": "view.plan", "role_hypothesis": "plan_view_candidate", "bbox_display": [20, 60, 360, 280]},
                {"id": "view.section", "role_hypothesis": "section_view_candidate", "bbox_display": [440, 60, 700, 280]},
            ],
            [
                {"id": "title.plan", "text": "STAIRCASE LAYOUT PLAN", "resolved_role": "plan_view_candidate", "bbox_display": [80, 35, 260, 55], "primitive_refs": ["word.plan"]},
                {"id": "title.section", "text": "SECTION A-A", "resolved_role": "section_label", "bbox_display": [500, 35, 620, 55], "primitive_refs": ["word.section"]},
            ],
        )

        self.assertEqual(result["summary"]["resolved_segment_count"], 2)
        self.assertEqual({item["view_id"] for item in result["segments"] if item["state"] == "resolved"}, {"view.plan", "view.section"})
        self.assertTrue(result["contract"]["geometry_without_unique_title_remains_uncertified"])

    def test_unanchored_geometry_remains_unresolved(self):
        result = segment_views_by_titles(
            fitz.Rect(0, 0, 400, 300),
            [{"id": "view.unknown", "role_hypothesis": "drawing_view_candidate", "bbox_display": [20, 20, 200, 200]}],
            [],
        )
        self.assertEqual(result["segments"][0]["state"], "unresolved")

    def test_duplicate_section_tokens_collapse_but_distinct_titles_abstain(self):
        view = {"id": "view.section", "role_hypothesis": "section_view_candidate", "bbox_display": [100, 50, 300, 250]}
        duplicate = segment_views_by_titles(
            fitz.Rect(0, 0, 400, 300),
            [view],
            [
                {"id": "title.word", "text": "SECTION", "resolved_role": "section_view_candidate", "bbox_display": [130, 30, 190, 45]},
                {"id": "title.label.native", "text": "A-A", "resolved_role": "section_label", "bbox_display": [195, 30, 225, 45], "confidence": 1.0},
                {"id": "title.label.line", "text": "A-A", "resolved_role": "section_label", "bbox_display": [190, 28, 230, 46], "confidence": 0.98},
            ],
        )
        self.assertEqual(duplicate["segments"][0]["state"], "resolved")
        self.assertEqual(duplicate["segments"][0]["normalised_title"], "A-A")

        ambiguous = segment_views_by_titles(
            fitz.Rect(0, 0, 400, 300),
            [view],
            [
                {"id": "title.a", "text": "A-A", "resolved_role": "section_label", "bbox_display": [130, 30, 160, 45]},
                {"id": "title.b", "text": "B-B", "resolved_role": "section_label", "bbox_display": [220, 30, 250, 45]},
            ],
        )
        self.assertEqual(ambiguous["segments"][0]["state"], "candidate")

    def test_scale_and_whitespace_split_one_connected_parent_without_splitting_primitives(self):
        geometry = [
            {"id": f"drawing[{index}]", "bbox_display": box}
            for index, box in enumerate(
                [
                    [20, 20, 80, 30],
                    [20, 45, 80, 55],
                    [25, 70, 75, 80],
                    [30, 25, 40, 75],
                    [220, 20, 280, 30],
                    [220, 45, 280, 55],
                    [225, 70, 275, 80],
                    [260, 25, 270, 75],
                    [75, 85, 225, 87],
                ]
            )
        ]
        result = segment_views_by_titles(
            fitz.Rect(0, 0, 400, 220),
            [
                {
                    "id": "view.connected",
                    "role_hypothesis": "drawing_view_candidate",
                    "bbox_display": [20, 20, 280, 87],
                    "primitive_refs": [item["id"] for item in geometry],
                },
                {
                    "id": "view.table",
                    "role_hypothesis": "table_or_grid_candidate",
                    "bbox_display": [310, 20, 390, 180],
                    "primitive_refs": [],
                },
            ],
            [
                {"id": "left.1", "text": "STAIRCASE", "resolved_role": "unclassified_text", "bbox_display": [20, 95, 75, 105], "primitive_refs": ["word[1]"]},
                {"id": "left.2", "text": "RC", "resolved_role": "unclassified_text", "bbox_display": [78, 95, 93, 105], "primitive_refs": ["word[2]"]},
                {"id": "left.3", "text": "DETAILS", "resolved_role": "unclassified_text", "bbox_display": [96, 95, 140, 105], "primitive_refs": ["word[3]"]},
                {"id": "left.scale.1", "text": "SCALE", "resolved_role": "unclassified_text", "bbox_display": [45, 110, 80, 120], "primitive_refs": ["word[4]"]},
                {"id": "left.scale.2", "text": "1:25", "resolved_role": "unclassified_text", "bbox_display": [83, 110, 110, 120], "primitive_refs": ["word[5]"]},
                {"id": "right.1", "text": "SECTION", "resolved_role": "section_view_candidate", "bbox_display": [220, 95, 265, 105], "primitive_refs": ["word[6]"]},
                {"id": "right.2", "text": "A-A", "resolved_role": "section_label", "bbox_display": [268, 95, 290, 105], "primitive_refs": ["word[7]"]},
                {"id": "right.scale.1", "text": "SCALE", "resolved_role": "unclassified_text", "bbox_display": [230, 110, 265, 120], "primitive_refs": ["word[8]"]},
                {"id": "right.scale.2", "text": "1:25", "resolved_role": "unclassified_text", "bbox_display": [268, 110, 295, 120], "primitive_refs": ["word[9]"]},
                {"id": "title.block", "text": "STAIRCASE RC DETAILS", "resolved_role": "unclassified_text", "bbox_display": [315, 185, 390, 195], "primitive_refs": ["word[10]"]},
            ],
            geometry_primitives=geometry,
        )

        scopes = [item for item in result["segments"] if item["view_id"] == "view.connected"]
        self.assertEqual(len(scopes), 2)
        self.assertTrue(all(item["state"] == "resolved" for item in scopes))
        self.assertEqual({item["scale_ratio"] for item in scopes}, {"1:25"})
        self.assertEqual(result["summary"]["split_parent_view_count"], 1)
        self.assertEqual(set(scopes[0]["primitive_refs"]) & set(scopes[1]["primitive_refs"]), set())
        excluded = set(scopes[0]["excluded_primitive_refs"])
        self.assertIn("drawing[8]", excluded)
        self.assertEqual(
            set(scopes[0]["primitive_refs"]) | set(scopes[1]["primitive_refs"]) | excluded,
            {item["id"] for item in geometry},
        )
        table = next(item for item in result["segments"] if item["view_id"] == "view.table")
        self.assertEqual(table["state"], "unresolved")
        self.assertTrue(result["contract"]["tables_and_title_blocks_are_not_view_scopes"])


if __name__ == "__main__":
    unittest.main()
