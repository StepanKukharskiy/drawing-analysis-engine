import inspect
import unittest
from pathlib import Path

import fitz

from src.drawing_engine.core.dimension_attachment import DimensionAttachment, LineSegment, attach_dimensions
from src.drawing_engine.disciplines.concrete.generic_prismatic_solver import solve_generic_prismatic


ROOT = Path(__file__).resolve().parents[1]


def display_page(path: Path) -> tuple[fitz.Document, fitz.Page]:
    raw = fitz.open(path)
    display = fitz.open()
    display.insert_pdf(raw)
    raw.close()
    display[0].remove_rotation()
    return display, display[0]


class GenericPrismaticSolverTest(unittest.TestCase):
    @staticmethod
    def _accepted_dimension(value, orientation, measured_points, attachment_id):
        baseline = LineSegment(measured_points[0], measured_points[1], 0.5, f"{attachment_id}.baseline")
        extension = LineSegment(measured_points[0], measured_points[0], 0.5, f"{attachment_id}.extension")
        return DimensionAttachment(
            attachment_id=attachment_id,
            value_mm=value,
            text=str(value),
            text_method="native_pdf_text",
            text_confidence=1.0,
            text_bbox=(0.0, 0.0, 1.0, 1.0),
            orientation=orientation,
            baseline=baseline,
            extension_lines=(extension, extension),
            dimension_points=measured_points,
            measured_points=measured_points,
            terminal_refs=(f"{attachment_id}.left", f"{attachment_id}.right"),
            scale_points_per_mm=0.1,
            scale_support=4,
            endpoint_alignment_residual=0.0,
            score=1.0,
            status="accepted",
            evidence=("test",),
        )

    def test_cross_view_profiles_close_watertight_prism(self):
        document, page = display_page(ROOT / "КЖ0-2-4.pdf")
        record = solve_generic_prismatic(page)
        document.close()

        self.assertEqual(record["pipeline_mode"], "generic_cross_view_prismatic")
        self.assertEqual(sorted(record["concrete_3d_input"]["dimensions_mm"]), [300.0, 2880.0, 3600.0])
        self.assertAlmostEqual(record["concrete_quantity_takeoff"]["net_concrete_m3"], 3.1104, places=6)
        self.assertEqual(record["constraint_validation"]["status"], "pass")
        mesh = record["procedural_evidence"]["mesh"]["validation"]
        self.assertTrue(mesh["watertight"])
        self.assertEqual(mesh["euler_characteristic"], 2)

    def test_solver_is_drawing_neutral_and_abstains_without_closure(self):
        source = inspect.getsource(__import__("src.drawing_engine.disciplines.concrete.generic_prismatic_solver", fromlist=["*"])).lower()
        self.assertNotIn("кж", source)
        self.assertNotIn("v24", source)
        self.assertNotIn(".pdf", source)

        document, page = display_page(ROOT / "v24.pdf")
        with self.assertRaisesRegex(ValueError, "closure"):
            solve_generic_prismatic(page)
        document.close()

    def test_geometry_gated_ocr_closes_textless_beam(self):
        document, page = display_page(ROOT / "test2.pdf")
        dimensions = tuple(attach_dimensions(page))
        accepted = [item for item in dimensions if item.status == "accepted"]
        self.assertTrue(all(item.text_method == "geometry_gated_ocr" for item in accepted))
        self.assertIn(11350.0, [item.value_mm for item in accepted])
        self.assertIn(400.0, [item.value_mm for item in accepted])
        self.assertIn(700.0, [item.value_mm for item in accepted])

        record = solve_generic_prismatic(page, dimensions)
        document.close()
        self.assertEqual(sorted(record["concrete_3d_input"]["dimensions_mm"]), [400.0, 700.0, 11350.0])
        self.assertAlmostEqual(record["concrete_quantity_takeoff"]["net_concrete_m3"], 3.178, places=6)

    def test_separate_native_paths_assemble_closed_profiles(self):
        document = fitz.open()
        page = document.new_page(width=500, height=300)
        for rect in (fitz.Rect(100, 100, 220, 130), fitz.Rect(300, 100, 330, 120)):
            shape = page.new_shape()
            shape.draw_line(rect.top_left, rect.top_right)
            shape.draw_line(rect.top_right, rect.bottom_right)
            shape.finish()
            shape.commit()
            shape = page.new_shape()
            shape.draw_line(rect.bottom_right, rect.bottom_left)
            shape.draw_line(rect.bottom_left, rect.top_left)
            shape.finish()
            shape.commit()
        dimensions = (
            self._accepted_dimension(1200.0, "horizontal", ((100.0, 100.0), (220.0, 100.0)), "dimension.1"),
            self._accepted_dimension(300.0, "vertical", ((100.0, 100.0), (100.0, 130.0)), "dimension.2"),
            self._accepted_dimension(300.0, "horizontal", ((300.0, 100.0), (330.0, 100.0)), "dimension.3"),
            self._accepted_dimension(200.0, "vertical", ((300.0, 100.0), (300.0, 120.0)), "dimension.4"),
        )

        record = solve_generic_prismatic(page, dimensions)
        self.assertEqual(sorted(record["concrete_3d_input"]["dimensions_mm"]), [200.0, 300.0, 1200.0])
        profiles = record["procedural_evidence"]["metric_profiles"]
        self.assertTrue(any(item["path_kind"] == "assembled_closed_rectangle" for item in profiles))


if __name__ == "__main__":
    unittest.main()
