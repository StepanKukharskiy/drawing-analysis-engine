import unittest
from unittest.mock import patch

import fitz

from src.drawing_engine.core.raster_fallback import (
    assess_native_page_quality,
    augment_observation_graph_with_raster,
    ocr_raster_dimension_label_crops,
    reconstruct_raster_observations,
)


class RasterFallbackTest(unittest.TestCase):
    @staticmethod
    def _vector_scene(document: fitz.Document, *, with_text: bool) -> fitz.Page:
        page = document.new_page(width=300, height=200)
        for index in range(9):
            page.draw_line((20, 20 + index * 16), (280, 20 + index * 16), width=1.2)
        if with_text:
            page.insert_text((20, 190), "SECTION A-A", fontsize=9)
        return page

    def test_router_prefers_native_and_uses_hybrid_only_for_missing_text(self):
        document = fitz.open()
        native = self._vector_scene(document, with_text=True)
        self.assertEqual(assess_native_page_quality(native)["route"], "native")
        fallback = reconstruct_raster_observations(native)
        self.assertEqual(fallback["status"], "not_invoked")
        self.assertEqual(fallback["observations"], [])

        textless_document = fitz.open()
        textless = self._vector_scene(textless_document, with_text=False)
        quality = assess_native_page_quality(textless)
        self.assertEqual(quality["route"], "hybrid_text_ocr")
        fallback = reconstruct_raster_observations(textless, quality)
        self.assertNotEqual(fallback["status"], "not_invoked")
        self.assertTrue(fallback["contract"]["observations_are_not_accepted_entities"])
        self.assertEqual(fallback["coordinate_transform"]["pixel_to_display"], [0.5, 0.0, 0.0, 0.5, 0.0, 0.0])
        document.close()
        textless_document.close()

    def test_image_dominant_page_reconstructs_page_coordinate_lines(self):
        source = fitz.open()
        source_page = self._vector_scene(source, with_text=False)
        image = source_page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).tobytes("png")
        document = fitz.open()
        page = document.new_page(width=300, height=200)
        page.insert_image(page.rect, stream=image)

        quality = assess_native_page_quality(page)
        self.assertEqual(quality["route"], "raster_reconstruct")
        fallback = reconstruct_raster_observations(page, quality)
        lines = [item for item in fallback["observations"] if item["kind"] == "line_segment"]
        self.assertTrue(lines)
        self.assertTrue(all(item["method"] == "opencv_otsu_skeleton_hough_collinear_merge" for item in lines))
        self.assertTrue(all(0 <= coordinate <= 300 for item in lines for coordinate in (item["bbox_display"][0], item["bbox_display"][2])))
        self.assertTrue(all(0 <= coordinate <= 200 for item in lines for coordinate in (item["bbox_display"][1], item["bbox_display"][3])))
        self.assertTrue(fallback["vector_topology"]["vertices"])
        self.assertEqual(
            len(fallback["vector_topology"]["edges"]),
            len(lines),
        )
        self.assertTrue(fallback["contract"]["raster_crossings_do_not_imply_connectivity"])
        self.assertIn("generic_structures", fallback)

        graph = {"nodes": [], "edges": [], "provenance": {}}
        augmented = augment_observation_graph_with_raster(graph, fallback)
        self.assertEqual(len(augmented["nodes"]), len(fallback["observations"]))
        self.assertIn("quality_gated_raster_fallback", augmented["provenance"]["methods"])
        source.close()
        document.close()

    def test_raster_topology_emits_generic_candidate_schemas(self):
        source = fitz.open()
        source_page = source.new_page(width=300, height=220)
        source_page.draw_rect(fitz.Rect(45, 35, 255, 185), width=1.2)
        for x in (90, 150, 210):
            source_page.draw_line((x, 35), (x, 185), width=1.2)
        for y in (80, 130):
            source_page.draw_line((45, y), (255, y), width=1.2)
        image = source_page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).tobytes("png")
        document = fitz.open()
        page = document.new_page(width=300, height=220)
        page.insert_image(page.rect, stream=image)

        fallback = reconstruct_raster_observations(page)
        generic = fallback["generic_structures"]
        self.assertTrue(generic["contour_hypotheses"])
        self.assertTrue(generic["view_hypotheses"])
        self.assertTrue(generic["contract"]["raster_hypotheses_are_not_accepted_entities"])
        contour = generic["contour_hypotheses"][0]
        self.assertEqual(contour["source_modality"], "raster")
        self.assertTrue(contour["primitive_refs"])
        self.assertTrue(contour["segments_display"])
        view = generic["view_hypotheses"][0]
        self.assertEqual(view["source_modality"], "raster")
        self.assertEqual(view["role_hypothesis"], "drawing_view_candidate")
        self.assertEqual(view["dimension_refs"], [])
        self.assertEqual(
            view["features"]["crossing_connectivity"],
            "unknown_until_semantic_or_native_topology_support",
        )
        source.close()
        document.close()

    def test_raster_dimension_topology_requires_terminal_backed_complete_chain(self):
        source = fitz.open()
        source_page = source.new_page(width=300, height=220)
        source_page.draw_line((60, 70), (240, 70), width=1.0)
        source_page.draw_line((60, 70), (60, 155), width=1.0)
        source_page.draw_line((240, 70), (240, 155), width=1.0)
        source_page.draw_line((57, 73), (63, 67), width=1.0)
        source_page.draw_line((237, 73), (243, 67), width=1.0)
        source_page.draw_line((60, 155), (240, 155), width=1.0)
        image = source_page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).tobytes("png")
        document = fitz.open()
        page = document.new_page(width=300, height=220)
        page.insert_image(page.rect, stream=image)

        fallback = reconstruct_raster_observations(page)
        topology = fallback["dimension_topology"]
        self.assertTrue(topology["chains"])
        self.assertTrue(topology["label_crop_candidates"])
        self.assertFalse(topology["contract"]["ocr_invoked_by_dimension_topology"])
        self.assertFalse(topology["contract"]["numeric_value_assigned"])
        chain = topology["chains"][0]
        self.assertEqual(chain["status"], "complete_geometry_candidate")
        self.assertIsNone(chain["value_mm"])
        self.assertEqual(len(chain["extension_line_refs"]), 2)
        self.assertGreaterEqual(len(chain["terminal_refs"]), 2)
        self.assertTrue(chain["topology_edge_refs"])

        terminal_free_source = fitz.open()
        terminal_free_page = terminal_free_source.new_page(width=300, height=220)
        terminal_free_page.draw_line((60, 70), (240, 70), width=1.0)
        terminal_free_page.draw_line((60, 70), (60, 155), width=1.0)
        terminal_free_page.draw_line((240, 70), (240, 155), width=1.0)
        terminal_free_image = terminal_free_page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).tobytes("png")
        terminal_free_document = fitz.open()
        terminal_free = terminal_free_document.new_page(width=300, height=220)
        terminal_free.insert_image(terminal_free.rect, stream=terminal_free_image)
        terminal_free_fallback = reconstruct_raster_observations(terminal_free)
        self.assertEqual(terminal_free_fallback["dimension_topology"]["chains"], [])
        self.assertEqual(terminal_free_fallback["dimension_label_ocr"]["attempts"], [])
        source.close()
        document.close()
        terminal_free_source.close()
        terminal_free_document.close()

    def test_dimension_label_ocr_reads_only_complete_chain_crops(self):
        source = fitz.open()
        source_page = source.new_page(width=300, height=220)
        source_page.draw_line((60, 70), (240, 70), width=1.0)
        source_page.draw_line((60, 70), (60, 155), width=1.0)
        source_page.draw_line((240, 70), (240, 155), width=1.0)
        source_page.draw_line((57, 73), (63, 67), width=1.0)
        source_page.draw_line((237, 73), (243, 67), width=1.0)
        source_page.draw_line((60, 155), (240, 155), width=1.0)
        source_page.insert_text((132, 60), "1800", fontsize=14)
        source_page.insert_text((12, 210), "9999", fontsize=14)
        image = source_page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).tobytes("png")
        document = fitz.open()
        page = document.new_page(width=300, height=220)
        page.insert_image(page.rect, stream=image)

        fallback = reconstruct_raster_observations(page)
        label_ocr = fallback["dimension_label_ocr"]
        self.assertEqual([item["text"] for item in label_ocr["observations"]], ["1800"])
        self.assertEqual(label_ocr["chain_results"][0]["status"], "unique_numeric_observation")
        self.assertGreater(label_ocr["summary"]["rejected_attempt_count"], 0)
        self.assertTrue(label_ocr["contract"]["whole_page_numeric_text_does_not_bind_to_chains"])
        observation = label_ocr["observations"][0]
        self.assertEqual(observation["numeric_value_candidate"], 1800.0)
        self.assertEqual(observation["unit_interpretation"], "drawing_dimension_unit_unresolved")
        self.assertGreaterEqual(len(observation["primitive_refs"]), 5)
        self.assertTrue(observation["topology_edge_refs"])
        self.assertTrue(fitz.Rect(observation["source_crop_display"]).contains(fitz.Rect(observation["bbox_display"])))
        self.assertIsNone(fallback["dimension_topology"]["chains"][0]["value_mm"])
        ownership = fallback["dimension_ownership"]
        self.assertEqual(ownership["summary"]["dimension_candidate_count"], 1)
        self.assertEqual(ownership["summary"]["accepted_candidate_count"], 0)
        self.assertEqual(ownership["summary"]["single_scale_scope_count"], 1)
        self.assertEqual(ownership["dimension_candidates"][0]["status"], "unknown")
        self.assertFalse(ownership["dimension_candidates"][0]["solver_eligible"])
        self.assertTrue(fallback["contract"]["raster_dimension_ownership_is_solver_gated"])
        augmented = augment_observation_graph_with_raster(
            {"nodes": [], "edges": [], "provenance": {}},
            fallback,
        )
        node = next(item for item in augmented["nodes"] if item["id"] == observation["id"])
        self.assertEqual(node["dimension_chain_ref"], observation["dimension_chain_ref"])
        self.assertEqual(node["primitive_refs"], observation["primitive_refs"])
        source.close()
        document.close()

    def test_conflicting_crop_readings_remain_ambiguous(self):
        document = fitz.open()
        page = document.new_page(width=200, height=120)
        topology = {
            "chains": [{"id": "raster.dimension_chain.0001"}],
            "label_crop_candidates": [
                {
                    "id": "raster.dimension_chain.0001.label_crop.1",
                    "dimension_chain_ref": "raster.dimension_chain.0001",
                    "bbox_display": [40, 20, 160, 50],
                    "orientation": "horizontal",
                },
                {
                    "id": "raster.dimension_chain.0001.label_crop.2",
                    "dimension_chain_ref": "raster.dimension_chain.0001",
                    "bbox_display": [40, 50, 160, 80],
                    "orientation": "horizontal",
                },
            ],
        }

        def result(text):
            return {
                "selected": None,
                "attempts": [
                    {
                        "psm": 7,
                        "raw_text": text,
                        "normalized_text": text,
                        "confidence": 0.9,
                        "bbox_pixels": [20, 20, 80, 60],
                        "status": "numeric_candidate",
                        "reason": None,
                        "value": float(text),
                    }
                ],
            }

        with patch(
            "src.drawing_engine.core.raster_fallback.ocr_isolated_dimension_label",
            side_effect=[result("1800"), result("1900")],
        ):
            label_ocr = ocr_raster_dimension_label_crops(page, topology)
        self.assertEqual({item["text"] for item in label_ocr["observations"]}, {"1800", "1900"})
        chain = label_ocr["chain_results"][0]
        self.assertEqual(chain["status"], "ambiguous_conflicting_numeric_observations")
        self.assertIsNone(chain["selected_observation_ref"])
        self.assertTrue(label_ocr["contract"]["conflicting_numeric_observations_abstain"])
        document.close()

    def test_vertical_crop_ocr_maps_rotated_token_back_to_page_space(self):
        source = fitz.open()
        source_page = source.new_page(width=260, height=220)
        source_page.insert_text((58, 130), "1400", fontsize=14, rotate=90)
        image = source_page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).tobytes("png")
        document = fitz.open()
        page = document.new_page(width=260, height=220)
        page.insert_image(page.rect, stream=image)
        topology = {
            "chains": [
                {
                    "id": "raster.dimension_chain.0001",
                    "baseline_ref": "raster.line.baseline",
                    "extension_line_refs": ["raster.line.extension.1", "raster.line.extension.2"],
                    "terminal_refs": ["raster.line.terminal.1", "raster.line.terminal.2"],
                    "topology_edge_refs": [],
                }
            ],
            "label_crop_candidates": [
                {
                    "id": "raster.dimension_chain.0001.label_crop.1",
                    "dimension_chain_ref": "raster.dimension_chain.0001",
                    "bbox_display": [36, 75, 73, 145],
                    "orientation": "vertical",
                }
            ],
        }
        label_ocr = ocr_raster_dimension_label_crops(page, topology)
        self.assertEqual([item["text"] for item in label_ocr["observations"]], ["1400"])
        observation = label_ocr["observations"][0]
        self.assertEqual(observation["coordinate_transform"]["rotation_degrees"], -90)
        self.assertTrue(fitz.Rect([47, 98, 60, 130]).contains(fitz.Rect(observation["bbox_display"])))
        self.assertTrue(any(item["status"] == "rejected" for item in label_ocr["attempts"]))
        source.close()
        document.close()


if __name__ == "__main__":
    unittest.main()
