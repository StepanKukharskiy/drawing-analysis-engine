import json
import tempfile
import unittest
from pathlib import Path

import fitz

from tools.render_estimate_comparison_audit import _declaration_boxes, _exact_declaration_boxes, build_audit
from src.drawing_engine.disciplines.concrete.schedule_comparison import write_estimate_comparison


ROOT = Path(__file__).resolve().parents[1]


class EstimateComparisonAuditTest(unittest.TestCase):
    def test_exact_declaration_boxes_reject_broad_regions(self):
        self.assertEqual(_exact_declaration_boxes({"bbox_display": [10, 20, 300, 180]}), [])
        exact = _exact_declaration_boxes(
            {
                "bbox_display": [10, 20, 300, 180],
                "total_cell_bbox_display": [240, 120, 280, 140],
            }
        )
        self.assertEqual([list(box) for box in exact], [[240.0, 120.0, 280.0, 140.0]])
        component_cells = _exact_declaration_boxes(
            {
                "basis": "outlined_schedule_material_row_component_cell_ocr_sum",
                "bbox_display": [10, 20, 300, 180],
                "components": [
                    {"label": "declared_component_1", "bbox_display": [210, 120, 230, 140]},
                    {"label": "declared_component_2", "bbox_display": [235, 120, 255, 140]},
                ],
            }
        )
        self.assertEqual(
            [list(box) for box in component_cells],
            [[210.0, 120.0, 230.0, 140.0], [235.0, 120.0, 255.0, 140.0]],
        )
        labeled = _declaration_boxes(
            {
                "declared_by_designer": {
                    "concrete": [{"value": 1.06, "cell_bboxes_display": [[210, 120, 230, 140], [235, 120, 255, 140]]}],
                    "reinforcement": [],
                }
            },
            "ru",
        )
        self.assertEqual(sum(bool(label) for _, label, _ in labeled), 1)
        self.assertEqual(len(labeled), 2)

    def test_k1_audit_marks_declaration_and_discrepancy(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            comparison = temporary / "comparison.json"
            output = temporary / "audit.pdf"
            write_estimate_comparison(
                ROOT / "1.pdf",
                ROOT / "output/object_agnostic/1.engineering-graph.json",
                comparison,
            )
            build_audit(ROOT / "1.pdf", comparison, output)
            document = fitz.open(output)
            self.assertEqual(document.page_count, 3)
            self.assertEqual(
                {item["name"] for item in document.get_ocgs().values()},
                {"SOURCE", "DECLARED_SCHEDULE", "DISCREPANCIES"},
            )
            overlay_text = document[1].get_text().replace("\u00a0", " ")
            summary_text = document[2].get_text().replace("\u00a0", " ")
            self.assertIn("КОД ПРОВЕРКИ", overlay_text)
            self.assertIn("Время анализа чертежа:", overlay_text)
            self.assertIn("1.8102 м³", summary_text)
            self.assertIn("БЕТОН: РАСХОЖДЕНИЕ", overlay_text)
            self.assertIn("СТАЛЬ: РАСЧЕТ НЕДОСТУПЕН", overlay_text)
            manifest = json.loads(output.with_suffix(".manifest.json").read_text())
            self.assertEqual(manifest["presentation_language"], "ru")
            self.assertIn("ЧЕРТЕЖ И ВЕДОМОСТЬ ПРОЕКТИРОВЩИКА", summary_text)
            self.assertEqual(manifest["frozen_graph_sha256"], json.loads(comparison.read_text())["frozen_calculation"]["sha256"])


if __name__ == "__main__":
    unittest.main()
