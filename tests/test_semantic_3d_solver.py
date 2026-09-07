import inspect
import unittest
from pathlib import Path

import fitz

from src.drawing_engine.disciplines.concrete.semantic_3d_solver import solve_semantic_3d


ROOT = Path(__file__).resolve().parents[1]


def display_page(path: Path) -> tuple[fitz.Document, fitz.Page]:
    raw = fitz.open(path)
    display = fitz.open()
    display.insert_pdf(raw)
    raw.close()
    display[0].remove_rotation()
    return display, display[0]


class Semantic3DSolverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents = []
        cls.records = {}
        for name in ("1.pdf", "2.pdf"):
            document, page = display_page(ROOT / name)
            cls.documents.append(document)
            cls.records[name] = solve_semantic_3d(page)

    @classmethod
    def tearDownClass(cls):
        for document in cls.documents:
            document.close()

    def test_concrete_host_is_solved_from_cross_view_constraints(self):
        first = self.records["1.pdf"]
        second = self.records["2.pdf"]
        for record in (first, second):
            self.assertEqual(record["pipeline_mode"], "strict_procedural")
            self.assertEqual(record["concrete_3d_input"]["shaft"], {
                "width_x_mm": 400.0,
                "depth_y_mm": 400.0,
                "height_z_mm": 9480.0,
            })
            self.assertEqual(record["constraint_validation"]["status"], "pass")
        self.assertEqual(len(first["concrete_3d_input"]["corbels"]), 2)
        self.assertEqual(len(second["concrete_3d_input"]["corbels"]), 1)
        self.assertAlmostEqual(first["concrete_3d_input"]["corbels"][0]["taper_bottom_z_mm"], 5415, delta=3)
        self.assertAlmostEqual(second["concrete_3d_input"]["corbels"][0]["taper_bottom_z_mm"], 5510, delta=3)

    def test_primary_rebar_topology_count_and_diameter_close(self):
        for record in self.records.values():
            groups = record["reinforcement_3d_input"]["resolved_groups"]
            longitudinal, transverse = groups
            self.assertEqual((longitudinal["count"], longitudinal["diameter_mm"]), (4, 28))
            self.assertEqual((transverse["count"], transverse["diameter_mm"]), (67, 8))
            scene = record["reinforcement_3d_input"]["centerline_scene"]
            self.assertEqual(scene["path_count"], 71)
            self.assertEqual(scene["counts_by_mark"][longitudinal["mark"]], 4)
            self.assertEqual(scene["counts_by_mark"][transverse["mark"]], 67)
            self.assertEqual(record["reinforcement_3d_input"]["full_3d_status"], "partial_only")
            observations = record["reinforcement_3d_input"]["section_rebar_observations"]
            self.assertEqual(observations["resolved_section_count"], 5)
            self.assertGreater(observations["candidate_count"], 100)

    def test_solver_exports_profile_and_all_final_section_candidates(self):
        record = self.records["1.pdf"]
        contour = record["procedural_evidence"]["profile"]["calculation_contour"]
        self.assertEqual(contour["closure_validation"]["status"], "pass")
        self.assertGreaterEqual(len(contour["segments_display"]), 4)
        self.assertEqual(contour["polygon_points_display"][0], contour["polygon_points_display"][-1])
        self.assertTrue(contour["dimension_refs"])
        final_section = record["reinforcement_3d_input"]["section_rebar_observations"]["sections"][-1]
        self.assertGreater(final_section["host_bbox_display"][3] - final_section["host_bbox_display"][1], 100)
        self.assertGreater(final_section["candidate_count"], 250)
        marks = {mark for candidate in final_section["candidates"] for mark in candidate["leader_marks"]}
        self.assertTrue({"2", *{str(value) for value in range(9, 17)}} <= marks)
        self.assertNotIn("65", marks)
        self.assertNotIn("75", marks)

    def test_concrete_quantity_takeoff_is_componentized(self):
        first = self.records["1.pdf"]["concrete_quantity_takeoff"]
        second = self.records["2.pdf"]["concrete_quantity_takeoff"]
        self.assertAlmostEqual(first["gross_concrete_m3"], 1.8108, places=7)
        self.assertAlmostEqual(first["net_concrete_m3"], 1.8102, places=7)
        self.assertAlmostEqual(second["gross_concrete_m3"], 1.6638, places=7)
        self.assertAlmostEqual(second["net_concrete_m3"], 1.6632, places=7)
        self.assertEqual(first["recess_deduction_m3"], second["recess_deduction_m3"])

    def test_no_sheet_identity_or_drawing_specific_branch(self):
        source = inspect.getsource(__import__("src.drawing_engine.disciplines.concrete.semantic_3d_solver", fromlist=["*"])).lower()
        self.assertNotIn("k1", source)
        self.assertNotIn("k7", source)
        self.assertNotIn("1.pdf", source)
        self.assertNotIn("2.pdf", source)


if __name__ == "__main__":
    unittest.main()
