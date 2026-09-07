from __future__ import annotations

from pathlib import Path
import unittest

import fitz

from src.drawing_engine.core.dimension_attachment import attach_dimensions
from src.drawing_engine.core.metric_equation_graph import build_metric_equation_graph


ROOT = Path(__file__).resolve().parents[1]


def _chain(page: fitz.Page, x0: float, x1: float, y: float, text: str) -> None:
    page.draw_line((x0, y), (x1, y), width=0.5)
    page.draw_line((x0, y - 1), (x0, y + 16), width=0.5)
    page.draw_line((x1, y - 1), (x1, y + 16), width=0.5)
    page.draw_line((x0 - 2, y - 2), (x0 + 2, y + 2), width=0.5)
    page.draw_line((x1 - 2, y - 2), (x1 + 2, y + 2), width=0.5)
    page.insert_text(((x0 + x1) / 2 - 25, y - 4), text, fontsize=7)


class MetricEquationGraphTest(unittest.TestCase):
    def test_three_views_and_two_native_equation_extents_create_one_scope(self):
        document = fitz.open()
        page = document.new_page(width=660, height=260)
        views = []
        for index, left in enumerate((20.0, 235.0, 450.0), start=1):
            _chain(page, left + 10, left + 166, 70, "100x52=5200")
            _chain(page, left + 10, left + 49, 135, "100x13=1300")
            views.append(
                {
                    "id": f"view.{index}",
                    "bbox_display": [left, 20, left + 190, 190],
                }
            )
        graph = build_metric_equation_graph(page, views, ())
        self.assertEqual(graph["summary"]["equation_count"], 6)
        self.assertEqual(graph["summary"]["accepted_chain_count"], 6)
        self.assertEqual(graph["summary"]["accepted_scope_count"], 1)
        self.assertEqual(graph["scopes"][0]["view_ids"], ["view.1", "view.2", "view.3"])
        self.assertFalse(graph["scopes"][0]["quantity_aggregation_eligible"])

    def test_formula_text_without_native_chain_never_creates_scope(self):
        document = fitz.open()
        page = document.new_page(width=300, height=200)
        page.insert_text((40, 60), "100x52=5200", fontsize=8)
        graph = build_metric_equation_graph(
            page,
            [{"id": "view.1", "bbox_display": [20, 20, 280, 180]}],
            (),
        )
        self.assertEqual(graph["summary"]["equation_count"], 1)
        self.assertEqual(graph["summary"]["accepted_chain_count"], 0)
        self.assertEqual(graph["scopes"], [])

    def test_v24_equations_close_metric_scope_without_filename_dispatch(self):
        document = fitz.open(ROOT / "v24.pdf")
        page = document[0]
        # Frozen view boxes are evidence records produced by the same generic
        # region segmenter; the equation solver receives no source filename.
        views = [
            {"id": "view.1", "bbox_display": [106.181, 45.9602, 830.261, 390.6]},
            {"id": "view.2", "bbox_display": [94.1813, 463.0802, 843.821, 764.28]},
            {"id": "view.3", "bbox_display": [91.9012, 848.4, 832.061, 1151.64]},
            {"id": "view.4", "bbox_display": [1259.38, 1089.84, 1298.26, 1181.16]},
            {"id": "view.5", "bbox_display": [924.941, 141.48, 1041.34, 457.44]},
            {"id": "view.6", "bbox_display": [941.021, 628.44, 1060.9, 913.8]},
        ]
        graph = build_metric_equation_graph(page, views, attach_dimensions(page))
        self.assertEqual(graph["summary"]["equation_count"], 11)
        self.assertEqual(graph["summary"]["accepted_chain_count"], 10)
        self.assertEqual(graph["summary"]["accepted_scope_count"], 1)
        self.assertEqual(len(graph["scopes"][0]["view_ids"]), 5)
        self.assertFalse(graph["contract"]["filename_dispatch_used"])
        self.assertFalse(graph["contract"]["schedule_values_used"])


if __name__ == "__main__":
    unittest.main()
