import inspect
import unittest
from pathlib import Path

import fitz

from src.drawing_engine.disciplines.concrete.generic_profile_extrusion_solver import solve_generic_profile_extrusion


ROOT = Path(__file__).resolve().parents[1]


class GenericProfileExtrusionSolverTest(unittest.TestCase):
    def test_dimensioned_cross_view_profile_closes_watertight_solid(self):
        document = fitz.open(ROOT / "test.pdf")
        record = solve_generic_profile_extrusion(document[0])
        document.close()

        self.assertEqual(record["pipeline_mode"], "generic_dimensioned_profile_extrusion")
        self.assertEqual(record["concrete_3d_input"]["shape_type"], "extruded_profile")
        self.assertAlmostEqual(record["concrete_3d_input"]["dimensions_mm"]["extrusion_depth_mm"], 2100.0, places=3)
        self.assertAlmostEqual(record["concrete_quantity_takeoff"]["net_concrete_m3"], 1.99155, delta=0.01)
        validation = record["procedural_evidence"]["mesh"]["validation"]
        self.assertTrue(validation["watertight"])
        self.assertEqual(validation["euler_characteristic"], 2)
        self.assertAlmostEqual(
            validation["tetrahedral_volume_mm3"],
            record["concrete_quantity_takeoff"]["net_concrete_mm3"],
            delta=1.0,
        )
        self.assertEqual(len(record["procedural_evidence"]["selected_profile_ids"]), 2)

    def test_solver_is_drawing_neutral_and_abstains_without_unique_closure(self):
        source = inspect.getsource(__import__("src.drawing_engine.disciplines.concrete.generic_profile_extrusion_solver", fromlist=["*"])).lower()
        for forbidden in (".pdf", "stair", "beam", "column", "3179", "test"):
            self.assertNotIn(forbidden, source)

        document = fitz.open(ROOT / "v24.pdf")
        with self.assertRaisesRegex(ValueError, "profile extrusion"):
            solve_generic_profile_extrusion(document[0])
        document.close()


if __name__ == "__main__":
    unittest.main()
