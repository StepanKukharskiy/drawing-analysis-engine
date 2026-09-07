import inspect
import unittest
from unittest.mock import patch

import fitz

from src.drawing_engine.core.dimension_attachment import DimensionAttachment, LineSegment
from src.drawing_engine.disciplines.concrete.object_instance_extrusion_solver import solve_object_instance_extrusions


def dimension(identifier, value, measured, orientation, scale, text_box):
    baseline = LineSegment(measured[0], measured[1], 0.2, f"{identifier}.baseline")
    return DimensionAttachment(
        attachment_id=identifier,
        value_mm=float(value),
        text=str(value),
        text_method="native_pdf_text",
        text_confidence=1.0,
        text_bbox=text_box,
        orientation=orientation,
        baseline=baseline,
        extension_lines=(),
        dimension_points=measured,
        measured_points=measured,
        terminal_refs=(),
        scale_points_per_mm=scale,
        scale_support=2,
        endpoint_alignment_residual=0.0,
        score=1.0,
        status="accepted",
        evidence=(),
    )


class ObjectInstanceExtrusionSolverTest(unittest.TestCase):
    def test_closed_profile_and_dimensioned_open_depth_outline_close(self):
        document = fitz.open()
        page = document.new_page(width=600, height=300)
        shape = page.new_shape()
        shape.draw_polyline([(50, 50), (150, 50), (150, 100), (250, 100), (250, 150), (50, 50)])
        shape.finish(color=(0, 0, 0), width=1.4)
        shape.commit()
        shape = page.new_shape()
        shape.draw_line((350, 50), (350, 150))
        shape.draw_line((350, 150), (370, 150))
        shape.draw_line((370, 150), (370, 50))
        shape.finish(color=(0, 0, 0), width=1.4)
        shape.commit()
        views = [
            {"id": "view.primary", "bbox_display": [40, 40, 270, 170]},
            {"id": "view.section", "bbox_display": [340, 40, 380, 170]},
        ]
        object_graph = {
            "instances": [
                {
                    "id": "object_instance.synthetic",
                    "primary_view_id": "view.primary",
                    "section_view_id": "view.section",
                }
            ]
        }
        dimensions = (
            dimension("dimension.profile", 1000, ((50, 50), (150, 50)), "horizontal", 0.1, (80, 55, 110, 65)),
            dimension("dimension.depth", 200, ((350, 50), (370, 50)), "horizontal", 0.1, (346, 42, 374, 48)),
        )

        profile = {
            "id": "native_loop.synthetic",
            "drawing_ref": "drawing[0]",
            "bbox_display": [50, 50, 250, 150],
            "points_display": [[50, 50], [150, 50], [150, 100], [250, 100], [250, 150]],
            "area_points2": 5000.0,
            "fill_ratio": 0.25,
            "diagonal_count": 1,
            "stroke_width_points": 1.4,
            "primitive_refs": ["drawing[0]"],
        }
        outline = {
            "id": "open_depth_outline.synthetic",
            "drawing_ref": "drawing[1]",
            "bbox_display": [350, 50, 370, 150],
            "primitive_refs": ["drawing[1]"],
            "stroke_width_points": 1.4,
            "points_display": [[350, 50], [350, 150], [370, 150], [370, 50]],
        }
        with patch("src.drawing_engine.disciplines.concrete.object_instance_extrusion_solver._native_loops", return_value=[profile]), patch(
            "src.drawing_engine.disciplines.concrete.object_instance_extrusion_solver._three_sided_outlines", return_value=[outline]
        ):
            result = solve_object_instance_extrusions(page, dimensions, object_graph, views)
            certified_graph = {
                "instances": [
                    {
                        **object_graph["instances"][0],
                        "state": "certified",
                        "extrusion_depth_mm": 200.0,
                        "certificate": {
                            "status": "passed",
                            "contour_correspondence_ref": "contour_correspondence.synthetic",
                            "reprojection_validation_refs": ["reprojection.synthetic"],
                            "metric_span_refs": ["dimension.depth"],
                        },
                    }
                ]
            }
            certified = solve_object_instance_extrusions(
                page, (dimensions[1],), certified_graph, views
            )
        document.close()

        self.assertEqual(result["constraint_validation"]["status"], "pass")
        self.assertTrue(result["constraint_validation"]["combined_mesh"]["watertight"])
        self.assertEqual(len(result["concrete_quantity_takeoff"]["components"]), 1)
        self.assertGreater(result["concrete_quantity_takeoff"]["net_concrete_m3"], 0)
        self.assertEqual(
            certified["procedural_evidence"]["object_instance_solutions"][0]["profile_scale_certificate"]["method"],
            "relation_certified_native_silhouette_scale_transfer",
        )

    def test_module_contains_no_drawing_specific_dispatch(self):
        source = inspect.getsource(__import__("src.drawing_engine.disciplines.concrete.object_instance_extrusion_solver", fromlist=["*"])).lower()
        for forbidden in (".pdf", "k1", "k7", "column", "beam", "stair", "slab"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
