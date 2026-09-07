import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import fitz

from src.drawing_engine.audit.render_object_agnostic_audit import (
    _declared_cell_boxes,
    _detail_color_assignments,
    _primary_dimension_text,
    _quantity_comparisons,
    _rebar_3d_path_count,
    build_audit,
    insert_text,
    load_precomputed_audit_records,
)
from src.drawing_engine.disciplines.rebar.estimation_profile import load_estimation_profile


ROOT = Path(__file__).resolve().parents[1]
GRAPH_DIR = ROOT / "output" / "object_agnostic"
COMPARISON_DIR = ROOT / "output" / "estimates"


def render_from_frozen_records(source: Path, output: Path, *, language: str = "ru") -> Path:
    stem = source.stem
    return build_audit(
        source,
        output,
        language=language,
        precomputed_bundle_path=GRAPH_DIR / f"{stem}.object-agnostic-bundle.json",
        comparison_path=COMPARISON_DIR / f"{stem}.estimate-comparison.json",
    )


def page_text(page: fitz.Page) -> str:
    return page.get_text().replace("\u00a0", " ")


class ObjectAgnosticAuditTest(unittest.TestCase):
    def test_empty_overlay_text_box_is_ignored(self):
        document = fitz.open()
        page = document.new_page(width=100, height=100)
        insert_text(page, fitz.Rect(20, 20, 20, 30), "hidden", 7, (0, 0, 0))
        self.assertEqual(page.get_text(), "")

    def test_each_detected_detail_has_a_distinct_color_and_overlapping_duplicate_is_aliased(self):
        native = [
            {"id": "native.1", "marks": ["2"], "bbox_display": [0, 0, 100, 50]},
            {"id": "native.2", "marks": ["3"], "bbox_display": [0, 50, 100, 100]},
            {"id": "native.3", "marks": ["3"], "bbox_display": [0, 100, 100, 150]},
        ]
        fabrication = [
            {
                "id": "fabrication.1",
                "group_id": "group.1",
                "mark": "2",
                "source": {"bbox_display": [10, 5, 90, 45]},
            },
            {
                "id": "fabrication.2",
                "group_id": "group.2",
                "mark": "9",
                "source": {"bbox_display": [200, 0, 250, 50]},
            },
        ]
        colors = _detail_color_assignments(fabrication, native)
        self.assertEqual(len(set(colors["native"].values())), 3)
        self.assertEqual(colors["fabrication"]["fabrication.1"], colors["native"]["native.1"])
        self.assertNotIn(colors["fabrication"]["fabrication.2"], set(colors["native"].values()))

    def test_3d_summary_uses_rendered_paths_and_localized_dimensions(self):
        self.assertEqual(_rebar_3d_path_count({"solid_preview": {"rebar_paths": [{}, {}]}}), 2)
        dimensions = _primary_dimension_text(
            {
                "shape_type": "multi_object_extrusion_collection",
                "objects": [
                    {"object_instance_id": "object.1", "profile_area_mm2": 430000, "extrusion_depth_mm": 1000},
                ],
            },
            "ru",
        )
        self.assertEqual(dimensions, "объект 1: площадь 0.4300 м²; глубина 1000 мм")
        self.assertNotIn("m2", dimensions)
        self.assertNotIn("mm", dimensions)

    def test_exact_cells_fail_closed_and_steel_status_is_independent(self):
        self.assertEqual(_declared_cell_boxes({"bbox_display": [10, 20, 300, 180]}), [])
        exact = _declared_cell_boxes(
            {
                "bbox_display": [10, 20, 300, 180],
                "cell_bbox_display": [240, 120, 280, 140],
            }
        )
        self.assertEqual([list(box) for box in exact], [[240.0, 120.0, 280.0, 140.0]])
        component_cells = _declared_cell_boxes(
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

        comparisons = _quantity_comparisons(
            {
                "concrete": [{"value": 1.5}],
                "reinforcement": [{"value": 100.0}],
            },
            {
                "quantities": [{"net_concrete_m3": 1.5}],
                "rebar_program": {
                    "drawing_detail_takeoff": {
                        "status": "resolved_drawing_takeoff",
                        "totals": {"mass_kg": 105.0},
                    }
                },
            },
        )
        self.assertEqual(comparisons["concrete"]["status"], "match")
        self.assertEqual(comparisons["steel"]["status"], "discrepancy")
        self.assertEqual(comparisons["steel"]["delta"], 5.0)

    def test_textless_non_column_keeps_resolved_marked_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit.pdf"
            render_from_frozen_records(ROOT / "test.pdf", output)
            document = fitz.open(output)
            self.assertEqual(document.page_count, 3)
            self.assertEqual(
                {item["name"] for item in document.get_ocgs().values()},
            {"SOURCE", "OBJECT_INSTANCES", "DETECTED_VIEWS", "CUTTING_PLANES", "CROSS_VIEW_MATCHES", "OBJECT_CONTOURS", "REBAR_DETECTED", "REBAR_REJECTED", "DETAIL_LINKS", "DIMENSIONS", "DECLARED_SCHEDULE", "DISCREPANCIES"},
            )
            self.assertIn("ПОНИМАНИЕ ЧЕРТЕЖА", page_text(document[1]))
            self.assertIn("ПРОВЕРКА ОБЪЕМОВ", page_text(document[1]))
            self.assertIn("МОДЕЛЬ БЕТОНА ПОСТРОЕНА", page_text(document[2]))
            self.assertIn("1.992", page_text(document[2]))
            self.assertIn("Время анализа чертежа:", page_text(document[1]))
            for drawing in document[1].get_drawings(extended=True):
                dash = str(drawing.get("dashes") or "").replace(" ", "")
                self.assertIn(dash, {"", "[]0"}, f"synthetic dashed overlay in {drawing.get('layer')}")
            manifest = json.loads(output.with_suffix(".manifest.json").read_text())
            self.assertEqual(manifest["presentation_language"], "ru")
            self.assertGreater(manifest["timing"]["drawing_understanding_seconds"], 0)
            self.assertEqual(manifest["drawing_understanding_source"], "validated_precomputed_bundle")
            self.assertEqual(
                manifest["precomputed_bundle_validation"]["status"],
                "historical_superseded",
            )
            self.assertIn(
                "source_pdf_hash_missing_or_mismatched",
                manifest["precomputed_bundle_validation"]["binding_issues"],
            )
            self.assertGreater(manifest["page_summaries"][0]["view_hypotheses"], 0)
            self.assertGreater(manifest["page_summaries"][0]["rebar_path_fragments"], 0)

    def test_generic_solid_waits_for_dimension_ownership_adjudication(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit.pdf"
            render_from_frozen_records(ROOT / "КЖ0-2-4.pdf", output, language="en")
            document = fitz.open(output)
            self.assertEqual(document.page_count, 3)
            self.assertIn("AXONOMETRIC MODEL", page_text(document[2]))
            self.assertIn("NOT GENERATED", page_text(document[2]))
            self.assertNotIn("3.1104 m³", page_text(document[2]))
            self.assertIn("Drawing understanding time:", page_text(document[2]))
            self.assertEqual(json.loads(output.with_suffix(".manifest.json").read_text())["presentation_language"], "en")

    def test_candidate_08_presentation_scene_records_preserve_union_and_beam(self):
        engineering = json.loads(
            (
                GRAPH_DIR
                / "candidate-08-staircase-page.engineering-graph.json"
            ).read_text()
        )["pages"][0]
        preview = engineering["solid_preview"]
        candidate = preview["separate_object_candidate_previews"][0]

        geometry = engineering["solid_hypotheses"][0]["geometry"]
        self.assertEqual(geometry["shape_type"], "constructive_union")
        self.assertEqual(geometry["construction_region_count"], 3)
        self.assertTrue(geometry["external_boundary_ref"])
        self.assertFalse(geometry["absolute_orientation_resolved"])
        self.assertEqual(preview["label"], "absolute orientation unresolved")
        self.assertEqual(
            preview["rendering_contract"]["surface_mode"],
            "transparent_evidence_wireframe",
        )
        self.assertTrue(preview["rendering_contract"]["per_triangle_strokes"])
        camera = preview["rendering_contract"]["presentation_camera"]
        self.assertEqual(
            camera["projection_preset"], "canonical_fold_revealing_axonometric"
        )
        self.assertFalse(camera["physical_transform_applied"])
        self.assertFalse(camera["absolute_orientation_claimed"])
        vertices = preview["mesh"]["vertices_xyz_mm"]
        first_band_z = [vertex[2] for vertex in vertices if vertex[1] <= 975.001]
        second_band_z = [vertex[2] for vertex in vertices if vertex[1] >= 1174.999]
        band_extents = [(min(values), max(values)) for values in (first_band_z, second_band_z)]
        # Direction evidence assigns identities; "either band" masks a mirror.
        self.assertLess(band_extents[0][0], -1500.0)
        self.assertLessEqual(band_extents[0][1], 0.1)
        self.assertGreater(band_extents[1][1], 1500.0)
        self.assertEqual(preview["relative_placement_state"], "resolved_in_directed_plan_gauge")
        self.assertEqual(candidate["classification"], "beam")
        self.assertEqual(candidate["volume_candidate_m3"], 0.129)
        self.assertFalse(candidate["included_in_primary_quantity"])
        self.assertTrue(candidate["relative_physical_placement_resolved"])
        self.assertEqual(candidate["presentation_frame"]["kind"], "shared_relative_scene")
        self.assertEqual(
            candidate["presentation_frame"]["physical_transform"]["origin_xyz_mm"],
            [975.0, 2150.0, 0.0],
        )
        self.assertEqual(engineering["quantities"][0]["net_concrete_m3"], 1.7726147823710516)
        contexts = preview["context_candidate_previews"]
        self.assertEqual(len(contexts), 2)
        self.assertEqual(
            contexts[0]["classification"], "upper_landing_or_floor_slab"
        )
        self.assertEqual(
            contexts[1]["classification"], "terminal_support_beam_candidate"
        )
        self.assertFalse(contexts[0]["geometry_complete"])
        slab_y = [v[1] for v in contexts[0]["mesh"]["vertices_xyz_mm"]]
        self.assertAlmostEqual(min(slab_y), 1175.0, places=2)
        self.assertAlmostEqual(max(slab_y), 2150.0, places=2)
        self.assertEqual(contexts[0]["external_cap_role"], "analysis_cap")
        self.assertEqual(contexts[1]["volume_candidate_m3"], 0.129)
        self.assertTrue(
            all(not item["included_in_primary_quantity"] for item in contexts)
        )
        partial_materialization = engineering[
            "partial_profile_sweep_materialization"
        ]
        self.assertEqual(
            partial_materialization["summary"]["materialized_candidate_count"],
            2,
        )
        self.assertEqual(
            partial_materialization["summary"]["quantity_eligible_candidate_count"],
            0,
        )
        self.assertEqual(
            engineering["rebar_program"]["family_constrained_3d"]["reason"],
            "no physical rebar families closed",
        )

    def test_candidate_08_renders_union_exterior_and_separate_beam_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit.pdf"
            source = (
                ROOT
                / "output"
                / "rc_staircase_search"
                / "candidate-08-staircase-page.pdf"
            )
            comparison = (
                COMPARISON_DIR
                / "candidate-08-staircase-page.estimate-comparison.json"
            )
            with mock.patch(
                "src.drawing_engine.audit.render_object_agnostic_audit.extract_declared_schedules"
            ) as live_declarations, mock.patch(
                "src.drawing_engine.audit.render_object_agnostic_audit.understand_page"
            ) as live_analysis:
                build_audit(
                    source,
                    output,
                    language="en",
                    precomputed_bundle_path=(
                        GRAPH_DIR
                        / "candidate-08-staircase-page.object-agnostic-bundle.json"
                    ),
                    comparison_path=comparison,
                )
                live_declarations.assert_not_called()
                live_analysis.assert_not_called()

            document = fitz.open(output)
            self.assertEqual(document.page_count, 3)
            model_text = page_text(document[2])
            self.assertIn("CONCRETE MODEL AVAILABLE", model_text)
            self.assertIn("one constructive union", model_text)
            self.assertIn("SEPARATE BEAM", model_text)
            self.assertIn("PLACED", model_text)
            self.assertIn("0.129 m³ candidate", model_text)
            self.assertIn("not aggregated", model_text)
            self.assertIn("landing contact resolved", model_text)
            self.assertIn("1.7726 m³", model_text)
            self.assertIn("No physical rebar families closed", model_text)
            self.assertIn("transparent evidence wireframe", model_text)
            self.assertIn("camera is", model_text)
            self.assertIn("does not resolve absolute orientation", model_text)
            manifest = json.loads(output.with_suffix(".manifest.json").read_text())
            self.assertEqual(
                manifest["drawing_understanding_source"],
                "validated_precomputed_bundle",
            )
            self.assertTrue(
                manifest["declared_schedule_extracted_after_drawing_calculation"]
            )

    def test_unresolved_solid_gets_explicit_3d_status_page(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit.pdf"
            render_from_frozen_records(ROOT / "v24.pdf", output)
            document = fitz.open(output)
            self.assertEqual(document.page_count, 3)
            self.assertIn("АКСОНОМЕТРИЧЕСКАЯ МОДЕЛЬ НЕ ПОСТРОЕНА", page_text(document[2]))
            self.assertIn("ПОЧЕМУ НЕТ ОБЪЕМА БЕТОНА / 3D", page_text(document[2]))

    def test_drawing_takeoff_keeps_convention_mass_out_of_strict_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit.pdf"
            render_from_frozen_records(ROOT / "3179 ЛС (2)-2.pdf", output)
            document = fitz.open(output)
            overlay_text = page_text(document[1])
            model_text = page_text(document[2])
            self.assertIn("АРМАТУРА ПО ЧЕРТЕЖУ", overlay_text)
            self.assertIn("187.985 м", overlay_text)
            self.assertIn("≈ 107.6 кг", overlay_text)
            self.assertIn("Бетон по ведомости\n1.060 м³", overlay_text)
            self.assertIn("Масса по ведомости: 104.5 кг", overlay_text)
            self.assertIn("СТАЛЬ: РАСЧЕТ НЕДОСТУПЕН", overlay_text)
            self.assertIn("Арматура, показанная в 3D: 181 шт.", overlay_text)
            self.assertIn("МОДЕЛЬ БЕТОНА ПОСТРОЕНА", model_text)
            self.assertNotIn(" m2", model_text)
            self.assertNotIn(" mm", model_text)
            self.assertNotIn("мм мм", model_text)

    def test_precomputed_bundle_rejects_a_different_source_pdf(self):
        with self.assertRaisesRegex(ValueError, "source PDF SHA-256"):
            load_precomputed_audit_records(
                ROOT / "2.pdf",
                GRAPH_DIR / "test.object-agnostic-bundle.json",
                load_estimation_profile(),
            )

    def test_small_pdf_to_analysis_to_audit_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "small.pdf"
            output = Path(directory) / "audit.pdf"
            document = fitz.open()
            page = document.new_page(width=240, height=180)
            page.draw_rect(fitz.Rect(30, 40, 180, 130), width=1)
            page.insert_text((35, 25), "PLAN")
            page.insert_textbox(
                fitz.Rect(32, 45, 175, 125),
                " ".join(f"NOTE{index}" for index in range(40)),
                fontsize=5,
            )
            for index in range(40):
                page.insert_text(
                    (8 + (index % 10) * 22, 142 + (index // 10) * 8),
                    f"N{index}",
                    fontsize=4,
                )
            document.save(source)
            document.close()

            build_audit(source, output, language="en")

            rendered = fitz.open(output)
            self.addCleanup(rendered.close)
            self.assertEqual(rendered.page_count, 3)
            manifest = json.loads(output.with_suffix(".manifest.json").read_text())
            self.assertEqual(manifest["drawing_understanding_source"], "live_pdf_analysis")
            self.assertIsNone(manifest["precomputed_bundle_validation"])


if __name__ == "__main__":
    unittest.main()
