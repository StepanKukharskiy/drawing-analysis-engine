import unittest

from src.drawing_engine.disciplines.rebar.reinforcement_scene import reinforcement_from_semantics, reinforcement_payload


class ReinforcementSceneTest(unittest.TestCase):
    def test_generic_centerline_contract_has_no_drawing_key(self):
        paths = reinforcement_from_semantics(
            {
                "reinforcement": [
                    {
                        "mark": "A",
                        "role": "generic_longitudinal",
                        "instance": 1,
                        "diameter_mm": 20,
                        "placement_status": "drawing_constrained",
                        "points_xyz_mm": [[10, 20, 30], [10, 20, 2030]],
                    },
                    {
                        "mark": "B",
                        "role": "generic_loop",
                        "instance": 1,
                        "diameter_mm": None,
                        "placement_status": "unknown",
                        "points_xyz_mm": [[0, 0, 100], [100, 0, 100], [100, 100, 100]],
                        "closed": True,
                    },
                ]
            }
        )
        payload = reinforcement_payload(paths)
        self.assertEqual(payload["path_count"], 2)
        self.assertEqual(payload["marks"], ["A", "B"])
        self.assertEqual(payload["status_counts"]["drawing_constrained"], 1)
        self.assertEqual(payload["status_counts"]["unknown"], 1)

    def test_invalid_status_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "placement_status"):
            reinforcement_from_semantics(
                {
                    "reinforcement": [
                        {
                            "mark": "1",
                            "role": "bar",
                            "instance": 1,
                            "diameter_mm": 8,
                            "placement_status": "invented",
                            "points_xyz_mm": [[0, 0, 0], [0, 0, 1]],
                        }
                    ]
                }
            )


if __name__ == "__main__":
    unittest.main()
