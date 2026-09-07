import inspect
import json
import unittest
from pathlib import Path

import fitz

from src.drawing_engine.core.object_agnostic_understanding import understand_page
from src.drawing_engine.disciplines.rebar.rebar_fabrication import solve_cutting_length
from src.drawing_engine.disciplines.rebar.rebar_program import (
    build_rebar_program,
    compress_linear_distribution,
    expand_linear_distribution,
    extract_spacing_constraints,
)


ROOT = Path(__file__).resolve().parents[1]


def display_page(path: Path) -> tuple[fitz.Document, fitz.Page]:
    raw = fitz.open(path)
    display = fitz.open()
    display.insert_pdf(raw)
    raw.close()
    display[0].remove_rotation()
    return display, display[0]


class RebarProgramTest(unittest.TestCase):
    def test_mark_plus_step_annotation_is_preserved_without_inventing_count(self):
        document = fitz.open()
        page = document.new_page(width=300, height=200)
        page.insert_text((80, 70), "10")
        page.insert_text((80, 82), "Spacing 200")
        constraints = extract_spacing_constraints(page)
        document.close()
        annotation = next(item for item in constraints if item["type"] == "annotated_distribution_spacing")
        self.assertEqual(annotation["mark_token"], "10")
        self.assertEqual(annotation["spacing_mm"], 200)
        self.assertIsNone(annotation["physical_count_if_isolated"])
        self.assertEqual(annotation["arithmetic"]["status"], "not_applicable")

    def test_piecewise_distribution_expands_and_validates_count(self):
        positions = [0, 100, 200, 300, 500, 700]
        distribution = compress_linear_distribution(positions)
        self.assertEqual(distribution["resolved_count"], 6)
        self.assertEqual(len(distribution["zones"]), 2)
        self.assertEqual(expand_linear_distribution(distribution), positions)
        self.assertEqual(distribution["expansion_validation"]["status"], "pass")

    def test_k1_groups_survive_dimension_ownership_adjudication(self):
        document, page = display_page(ROOT / "1.pdf")
        record = understand_page(page)
        document.close()
        program = record["engineering_graph"]["rebar_program"]
        adjudication = record["engineering_graph"]["dimension_adjudication"]
        self.assertEqual(program["status"], "partial_groups_resolved")
        self.assertEqual(len(program["groups"]), 2)
        self.assertGreater(adjudication["summary"]["proposal_count"], adjudication["summary"]["accepted_count"])
        self.assertTrue(adjudication["contract"]["all_proposals_reach_ownership"])
        self.assertTrue(program["contract"]["generic_path_graph_precedes_object_solver"])
        self.assertTrue(program["contract"]["visible_projection_is_not_physical_length"])

    def test_k7_frozen_graph_preserves_partial_groups(self):
        engineering = json.loads(
            (ROOT / "output" / "object_agnostic" / "2.engineering-graph.json").read_text(
                encoding="utf-8"
            )
        )["pages"][0]
        self.assertEqual(engineering["rebar_program"]["status"], "partial_groups_resolved")
        self.assertEqual(len(engineering["rebar_program"]["groups"]), 2)

    def test_symbolic_observations_survive_without_concrete_solid(self):
        document, page = display_page(ROOT / "v24.pdf")
        program = build_rebar_program(page, [], None)
        document.close()
        self.assertEqual(program["status"], "observations_only")
        self.assertFalse(program["groups"])
        self.assertGreaterEqual(len(program["unassigned_spacing_constraints"]), 10)
        self.assertFalse(program["contract"]["requires_concrete_solid"])
        self.assertFalse(program["contract"]["schedule_values_used"])
        self.assertEqual(program["fabrication_status"], "unavailable")

    def test_deterministic_solver_resolves_only_closed_equations(self):
        complete_rectangle = {
            "geometry": {
                "topology_class": "polyline_candidate",
                "bends": [],
                "dimensions": [
                    {"value_mm": 400, "attachment": {"role": "overall_width"}},
                    {"value_mm": 300, "attachment": {"role": "overall_height"}},
                ],
            }
        }
        solved = solve_cutting_length(complete_rectangle, 5)
        self.assertEqual(solved["status"], "resolved")
        self.assertEqual(solved["cutting_length_each_mm"], 1400.0)
        self.assertEqual(solved["cutting_length_total_mm"], 7000.0)

        incomplete = {
            "geometry": {
                "topology_class": "closed_loop_with_hook_candidates",
                "bends": [{"arc": {"radius_mm": None}}],
                "dimensions": complete_rectangle["geometry"]["dimensions"],
            }
        }
        unresolved = solve_cutting_length(incomplete, 5)
        self.assertEqual(unresolved["status"], "partial_equation")
        self.assertIsNone(unresolved["cutting_length_each_mm"])
        self.assertIn("bend radii and centreline correction convention", unresolved["missing_constraints"])

    def test_module_is_drawing_neutral(self):
        source = inspect.getsource(__import__("src.drawing_engine.disciplines.rebar.rebar_program", fromlist=["*"])).lower()
        fabrication_source = inspect.getsource(__import__("src.drawing_engine.disciplines.rebar.rebar_fabrication", fromlist=["*"])).lower()
        metric_source = inspect.getsource(__import__("src.drawing_engine.disciplines.rebar.rebar_metric_solver", fromlist=["*"])).lower()
        for forbidden in (".pdf", "k1", "k7", "column", "beam", "stair"):
            self.assertNotIn(forbidden, source)
            self.assertNotIn(forbidden, fabrication_source)
            self.assertNotIn(forbidden, metric_source)


if __name__ == "__main__":
    unittest.main()
