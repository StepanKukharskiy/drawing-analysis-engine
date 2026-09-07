import unittest

from src.drawing_engine.core.leader_target_ranker import (
    FEATURE_NAMES,
    candidate_features,
    deterministic_acceptance_gate,
    load_default_model,
    rank_candidates,
    train_logistic_ranker,
    validate_model,
)


def candidate(identifier: str, points: list[list[float]]) -> dict:
    return {
        "id": identifier,
        "geometry": {"kind": "line", "points_display": points, "length_points": 100.0, "angle_deg": 0.0},
        "style": {"width_pt": 1.5},
        "candidate_score": 0.8,
        "repetition_group_ids": [],
        "observation_method": "native_pdf_vector",
    }


class LeaderTargetRankerTest(unittest.TestCase):
    def test_checked_in_model_is_valid_and_schedule_blind(self):
        model = load_default_model()
        self.assertIsNotNone(model)
        validate_model(model)
        self.assertTrue(model["acceptance_contract"]["ranking_only"])
        self.assertFalse(model["acceptance_contract"]["schedule_values_used"])
        self.assertFalse(any("schedule" in name for name in FEATURE_NAMES))

    def test_ranker_accepts_unique_geometry_but_abstains_on_twin(self):
        model = load_default_model()
        terminal = (10.0, 50.0)
        target = candidate("target", [[10.0, 50.0], [110.0, 50.0]])
        distant = candidate("distant", [[10.0, 90.0], [110.0, 90.0]])
        rows = rank_candidates(model, terminal, [distant, target], view_bbox=[0, 0, 120, 120], leader_angle_deg=45)
        gate = deterministic_acceptance_gate(rows, {"target": target, "distant": distant}, terminal)
        self.assertEqual(gate["status"], "accepted")
        self.assertEqual(gate["candidate_id"], "target")

        twin = candidate("twin", [[10.0, 0.0], [10.0, 100.0]])
        rows = rank_candidates(model, terminal, [target, twin, distant], view_bbox=[0, 0, 120, 120], leader_angle_deg=45)
        gate = deterministic_acceptance_gate(rows, {"target": target, "twin": twin, "distant": distant}, terminal)
        self.assertEqual(gate["status"], "abstained")
        self.assertIsNone(gate["candidate_id"])

    def test_training_is_deterministic(self):
        terminal = (0.0, 0.0)
        rows = []
        for label, row in (
            (1, candidate("positive", [[0.0, 0.0], [100.0, 0.0]])),
            (0, candidate("negative", [[0.0, 40.0], [100.0, 40.0]])),
        ):
            rows.append({"label": label, "features": candidate_features(terminal, row, view_bbox=[0, 0, 120, 120])})
        self.assertEqual(train_logistic_ranker(rows)["model_sha256"], train_logistic_ranker(rows)["model_sha256"])


if __name__ == "__main__":
    unittest.main()
