import unittest

from src.drawing_engine.disciplines.concrete.concrete_reconstruction import (
    analytic_volume_mm3,
    engineering_mesh,
    reconstruction_payload,
    spec_from_semantics,
)


def semantic_record(corbel_directions=(-1, 1)):
    return {
        "object_name": "generic_column_A",
        "shaft": {
            "width_x_mm": 400,
            "depth_y_mm": 400,
            "height_z_mm": 9480,
        },
        "corbels": [
            {
                "direction_x": direction,
                "projection_mm": 700,
                "depth_mm": 400,
                "taper_bottom_z_mm": 5415,
                "rectangular_bottom_z_mm": 5765,
                "top_z_mm": 6115,
            }
            for direction in corbel_directions
        ],
        "recesses": [
            {
                "center_x_mm": 0,
                "center_z_mm": z,
                "width_mm": 200,
                "height_mm": 100,
                "depth_mm": 15,
                "face_y": 1,
            }
            for z in (2410, 7100)
        ],
        "declared_volume_m3": 1.79,
        "source_status": "synthetic semantic record",
        "unknowns": ["reinforcement"],
    }


class UniversalConcreteReconstructionTest(unittest.TestCase):
    def test_bilateral_and_unilateral_records_use_the_same_engine(self):
        expected = {(-1, 1): 1.8102, (1,): 1.6632}
        for directions, volume_m3 in expected.items():
            with self.subTest(directions=directions):
                spec = spec_from_semantics(semantic_record(directions))
                solid, mesh = engineering_mesh(spec)
                self.assertTrue(mesh.is_watertight)
                self.assertTrue(mesh.is_winding_consistent)
                self.assertEqual(solid.genus(), 0)
                self.assertAlmostEqual(analytic_volume_mm3(spec) / 1e9, volume_m3, places=7)
                self.assertAlmostEqual(mesh.volume / 1e9, volume_m3, places=7)

    def test_third_shape_without_drawing_key(self):
        record = {
            "object_name": "plain_rectangular_host",
            "shaft": {"width_x_mm": 300, "depth_y_mm": 500, "height_z_mm": 2000},
            "corbels": [],
            "recesses": [],
        }
        spec = spec_from_semantics(record)
        payload = reconstruction_payload(spec)
        self.assertEqual(payload["object"], "plain_rectangular_host")
        self.assertAlmostEqual(payload["volume_validation"]["drawing_model_m3"], 0.3)
        self.assertTrue(payload["mesh"]["watertight"])

    def test_invalid_placement_fails_closed(self):
        record = semantic_record((1,))
        record["corbels"][0]["direction_x"] = 0
        with self.assertRaisesRegex(ValueError, "direction_x"):
            spec_from_semantics(record)


if __name__ == "__main__":
    unittest.main()
