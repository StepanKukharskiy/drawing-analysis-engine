import inspect
import json
import unittest
from pathlib import Path

import fitz

from src.drawing_engine.core.native_vector_detail_linking import (
    _attach_terminals_to_paths,
    _isolated_mark_token,
    _mark_variant_styles,
    _normalize_mark,
    _parse_variant_scheme,
    _point_polyline_distance,
    _native_detail_table_candidates,
    _native_or_ocr_mark,
)


ROOT = Path(__file__).resolve().parents[1]


class NativeVectorDetailLinkingTest(unittest.TestCase):
    def test_geometry_gated_ocr_normalizes_bracket_artifacts(self):
        self.assertEqual(_normalize_mark("314](5)"), ("3", "4", "5"))
        self.assertEqual(_normalize_mark("718](9)"), ("7", "8", "9"))
        self.assertEqual(_normalize_mark("10"), ("10",))
        self.assertEqual(_mark_variant_styles("34](5)", ("3", "4", "5")), {"3": "bare", "4": "square", "5": "parentheses"})
        self.assertEqual(_parse_variant_scheme("member K1, (K2), [K3]"), {"bare": "K1", "parentheses": "K2", "square": "K3"})

    def test_module_contains_no_drawing_specific_dispatch(self):
        source = inspect.getsource(__import__("src.drawing_engine.core.native_vector_detail_linking", fromlist=["*"])).lower()
        for forbidden in (".pdf", "k1", "k7", "beam", "stair", "slab"):
            self.assertNotIn(forbidden, source)

    def test_leader_terminal_requires_unique_projected_path_attachment(self):
        graph = {
            "fragments": [
                {
                    "id": "path.1",
                    "primitive_ref": "drawing[1].item[0]",
                    "view_id": "view.1",
                    "candidate_score": 0.9,
                    "geometry": {"points_display": [[0, 0], [100, 0]], "length_points": 100, "angle_deg": 0},
                },
                {
                    "id": "path.2",
                    "primitive_ref": "drawing[2].item[0]",
                    "view_id": "view.1",
                    "candidate_score": 0.6,
                    "geometry": {"points_display": [[0, 20], [100, 20]], "length_points": 100, "angle_deg": 0},
                },
            ]
        }
        attached = _attach_terminals_to_paths({"terminals": [[50, 1]]}, "view.1", graph)
        self.assertEqual([item["fragment_id"] for item in attached], ["path.1"])
        self.assertAlmostEqual(_point_polyline_distance((50, 1), [[0, 0], [100, 0]]), 1.0)

    def test_ambiguous_terminal_does_not_claim_a_path(self):
        graph = {
            "fragments": [
                {
                    "id": "path.1",
                    "primitive_ref": "drawing[1].item[0]",
                    "view_id": "view.1",
                    "candidate_score": 0.8,
                    "geometry": {"points_display": [[0, 0], [100, 0]], "length_points": 100, "angle_deg": 0},
                },
                {
                    "id": "path.2",
                    "primitive_ref": "drawing[2].item[0]",
                    "view_id": "view.1",
                    "candidate_score": 0.8,
                    "geometry": {"points_display": [[50, -50], [50, 50]], "length_points": 100, "angle_deg": 90},
                },
            ]
        }
        self.assertEqual(_attach_terminals_to_paths({"terminals": [[50, 1]]}, "view.1", graph), [])

    def test_mark_token_must_not_be_a_substring_of_dimension_text(self):
        import numpy as np

        isolated = np.full((100, 160), 255, dtype=np.uint8)
        isolated[30:70, 55:65] = 0
        isolated[30:70, 75:90] = 0
        self.assertTrue(_isolated_mark_token(__import__("fitz").Rect(55, 30, 90, 70), __import__("fitz").Rect(0, 0, 160, 100), isolated, scale=1))
        continued = isolated.copy()
        continued[30:70, 96:110] = 0
        self.assertFalse(_isolated_mark_token(__import__("fitz").Rect(55, 30, 90, 70), __import__("fitz").Rect(0, 0, 160, 100), continued, scale=1))

    def test_fragmented_grid_with_raster_detail_rows_is_detected(self):
        raw = fitz.open(ROOT / "1.pdf")
        display = fitz.open()
        display.insert_pdf(raw)
        display[0].remove_rotation()
        candidates = _native_detail_table_candidates(display[0])
        self.assertTrue(candidates)
        json.dumps(candidates)
        selected = candidates[0]
        self.assertGreaterEqual(selected["geometry_row_count"], 10)
        marks = {
            mark
            for row in selected["rows"]
            if row["geometry"]
            for observation in [_native_or_ocr_mark(display[0], row["mark_cell"])]
            if observation is not None
            for mark in observation["marks"]
        }
        self.assertTrue({"2", "3", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"} <= marks)
        display.close()
        raw.close()


if __name__ == "__main__":
    unittest.main()
