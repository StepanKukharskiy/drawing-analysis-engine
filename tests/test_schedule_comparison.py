import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import fitz

from src.drawing_engine.disciplines.concrete.schedule_comparison import (
    _ocr_words_region,
    build_estimate_comparison,
    calculated_from_graph,
    extract_declared_schedules,
)


ROOT = Path(__file__).resolve().parents[1]
GRAPHS = ROOT / "output" / "object_agnostic"


class ScheduleComparisonTest(unittest.TestCase):
    def test_convention_dependent_mass_remains_separate_from_strict_mass(self):
        calculated = calculated_from_graph(
            {
                "pages": [
                    {
                        "page": 1,
                        "quantities": [],
                        "reinforcement_quantities": {
                            "mass_kg": None,
                            "totals": {
                                "fabrication_length_m": 187.985,
                                "mass_kg": None,
                                "convention_dependent_mass_kg": 107.575446,
                            },
                        },
                    }
                ]
            }
        )["pages"][0]
        self.assertIsNone(calculated["reinforcement_mass_kg"])
        self.assertEqual(calculated["reinforcement_state"], "convention_dependent")
        self.assertAlmostEqual(
            calculated["reinforcement_convention_dependent_mass_kg"],
            107.575446,
        )

    def test_source_hash_mismatch_rejects_stale_graph_before_schedule_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "drawing.pdf"
            graph = root / "drawing.engineering-graph.json"
            source.write_bytes(b"%PDF-1.4\n")
            graph.write_text(
                json.dumps(
                    {
                        "result_binding": {
                            "source_pdf_sha256": "0" * 64,
                            "canonical_engineering_graph_sha256": "1" * 64,
                        },
                        "pages": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "source PDF hash"):
                build_estimate_comparison(source, graph)

    def test_ocr_region_outside_page_returns_no_words(self):
        document = fitz.open()
        page = document.new_page(width=100, height=100)
        self.assertEqual(_ocr_words_region(page, fitz.Rect(200, 200, 220, 220)), [])

    def test_k1_freezes_graph_then_reports_concrete_discrepancy(self):
        graph = GRAPHS / "1.engineering-graph.json"
        result = build_estimate_comparison(ROOT / "1.pdf", graph)
        expected_hash = hashlib.sha256(graph.read_bytes()).hexdigest()
        self.assertEqual(result["frozen_calculation"]["sha256"], expected_hash)
        self.assertTrue(result["frozen_calculation"]["frozen_before_declaration_extraction"])
        self.assertGreater(result["timing"]["declared_schedule_extraction_seconds"], 0)
        self.assertGreater(result["timing"]["estimate_comparison_seconds"], 0)
        self.assertIn("drawing_understanding_seconds", result["timing"])
        page = result["pages"][0]
        self.assertEqual(page["comparison"]["concrete"]["calculated"], 1.8102)
        self.assertEqual(page["comparison"]["concrete"]["declared"], 1.79)
        self.assertEqual(page["comparison"]["concrete"]["delta"], 0.0202)
        self.assertEqual(page["comparison"]["concrete"]["severity"], "review")
        self.assertEqual(page["declared_by_designer"]["reinforcement"][0]["value"], 242.2)
        self.assertEqual(page["calculated_from_drawing"]["reinforcement_state"], "unknown")
        self.assertIsNone(page["calculated_from_drawing"]["reinforcement_centerline_m"])
        self.assertIsNone(page["comparison"]["reinforcement_mass"]["calculated"])

    def test_k7_declared_totals_are_independent(self):
        result = build_estimate_comparison(ROOT / "2.pdf", GRAPHS / "2.engineering-graph.json")
        page = result["pages"][0]
        self.assertEqual(page["comparison"]["concrete"]["calculated"], 1.6632)
        self.assertEqual(page["comparison"]["concrete"]["declared"], 1.65)
        self.assertEqual(page["declared_by_designer"]["reinforcement"][0]["value"], 232.5)

    def test_class_subtotals_remain_declared_when_drawing_closure_is_unavailable(self):
        declared = extract_declared_schedules(ROOT / "КЖ0-2-4.pdf")["pages"][0]
        self.assertEqual(declared["concrete"][0]["value"], 3.01)
        self.assertEqual(declared["reinforcement"][0]["value"], 221.6)
        result = build_estimate_comparison(
            ROOT / "КЖ0-2-4.pdf",
            GRAPHS / "КЖ0-2-4.engineering-graph.json",
        )
        page = result["pages"][0]
        comparison = page["comparison"]["concrete"]
        self.assertIsNone(comparison["calculated"])
        self.assertEqual(comparison["declared"], 3.01)
        self.assertIsNone(comparison["delta"])
        self.assertEqual(comparison["status"], "calculation_unavailable")
        self.assertEqual(comparison["severity"], "unknown")
        self.assertEqual(page["calculated_from_drawing"]["concrete_state"], "unknown")
        self.assertEqual(
            page["calculated_from_drawing"]["reason"],
            "no dimensioned cross-view profile extrusion closure",
        )

    def test_textless_specification_uses_exact_ocr_total_cells(self):
        declared = extract_declared_schedules(ROOT / "test.pdf")["pages"][0]
        self.assertFalse(declared["native_text_available"])
        self.assertTrue(declared["ocr_fallback_used"])
        self.assertEqual(declared["concrete"][0]["value"], 1.95)
        self.assertEqual(declared["reinforcement"][0]["value"], 425.5)
        for declaration in (declared["concrete"][0], declared["reinforcement"][0]):
            left, top, right, bottom = declaration["bbox_display"]
            self.assertLess(right - left, 100)
            self.assertLess(bottom - top, 30)

    def test_fragmented_outlined_grid_uses_semantic_row_and_cell_ocr(self):
        declared = extract_declared_schedules(ROOT / "test2.pdf")["pages"][0]
        self.assertTrue(declared["ocr_fallback_used"])
        self.assertEqual(declared["concrete"][0]["value"], 3.18)
        self.assertEqual(declared["reinforcement"][0]["value"], 719.0)
        self.assertEqual(declared["reinforcement"][0]["aggregation_state"], "direct")
        for declaration in (declared["concrete"][0], declared["reinforcement"][0]):
            left, top, right, bottom = declaration["bbox_display"]
            self.assertLess(right - left, 60)
            self.assertLess(bottom - top, 30)

    def test_native_specification_sums_only_arithmetic_closed_steel_rows(self):
        declared = extract_declared_schedules(ROOT / "v24.pdf")["pages"][0]
        steel = declared["reinforcement"][0]
        self.assertEqual(declared["concrete"][0]["value"], 2.83)
        self.assertEqual(steel["value"], 592.64)
        self.assertEqual(steel["aggregation_state"], "derived")
        self.assertIn("arithmetic_closed_row_total_sum", steel["basis"])
        self.assertEqual(len(steel["components"]), 19)
        self.assertTrue(all(component["arithmetic_closed"] for component in steel["components"]))
        self.assertEqual(len(steel["cell_bboxes_display"]), 19)

    def test_outlined_schedule_cells_are_ocrd_only_after_3179_graph_is_frozen(self):
        graph = GRAPHS / "3179 ЛС (2)-2.engineering-graph.json"
        result = build_estimate_comparison(ROOT / "3179 ЛС (2)-2.pdf", graph)
        page = result["pages"][0]
        self.assertTrue(page["declared_by_designer"]["ocr_fallback_used"])
        self.assertEqual(page["comparison"]["concrete"]["declared"], 1.06)
        self.assertEqual(page["comparison"]["reinforcement_mass"]["declared"], 104.5)
        self.assertIsNone(page["comparison"]["reinforcement_mass"]["calculated"])
        self.assertIsNone(page["comparison"]["reinforcement_mass"]["delta"])
        self.assertEqual(page["comparison"]["reinforcement_mass"]["severity"], "unknown")
        self.assertEqual(page["calculated_from_drawing"]["reinforcement_state"], "convention_dependent")
        self.assertAlmostEqual(
            page["calculated_from_drawing"]["reinforcement_convention_dependent_mass_kg"],
            107.575446,
            places=6,
        )
        self.assertTrue(result["frozen_calculation"]["frozen_before_declaration_extraction"])


if __name__ == "__main__":
    unittest.main()
