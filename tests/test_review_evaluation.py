import unittest

from src.drawing_engine.project.review_evaluation import evaluate_review_records
from src.drawing_engine.project.review_feedback import create_review_record
from tests.test_review_feedback import canonical_graph


class ReviewEvaluationTest(unittest.TestCase):
    def test_task_metrics_separate_wrong_accepts_and_unnecessary_abstentions(self):
        graph = canonical_graph()
        decisions = [
            {
                "decision_id": "leader.reject",
                "candidate_id": "canonical.relation.leader",
                "task_type": "leader_target",
                "decision": "rejected",
                "evidence_refs": ["drawing[4]"],
                "reason_code": "wrong_leader_target",
            },
            {
                "decision_id": "dimension.accept",
                "candidate_id": "canonical.relation.dimension",
                "task_type": "dimension_ownership",
                "decision": "accepted",
                "evidence_refs": ["drawing[5]"],
                "reason_code": "dimension_terminals_confirmed",
            },
            {
                "decision_id": "view.accept",
                "candidate_id": "canonical.view.1",
                "task_type": "view_role",
                "decision": "accepted",
                "evidence_refs": ["word[1]"],
                "reason_code": "view_role_confirmed",
            },
        ]
        review = create_review_record(
            graph,
            reviewer_id="engineer-1",
            created_at="2026-08-23T12:00:00+03:00",
            decisions=decisions,
        )
        result = evaluate_review_records(graph, [review])
        self.assertEqual(result["validation"]["status"], "pass")
        self.assertEqual(result["overall"]["wrong_auto_accept_count"], 1)
        self.assertEqual(result["overall"]["unnecessary_abstention_count"], 2)
        self.assertEqual(result["by_task"]["leader_target"]["candidate_precision"], 0.0)
        self.assertEqual(result["by_task"]["dimension_ownership"]["candidate_precision"], 1.0)
        self.assertEqual(result["review_effort"]["reason_counts"]["wrong_leader_target"], 1)
        self.assertTrue(result["contract"]["declared_schedule_is_not_a_label_source"])

    def test_conflicting_reviews_are_excluded_and_fail_closed(self):
        graph = canonical_graph()
        base = {
            "decision_id": "decision.1",
            "candidate_id": "canonical.relation.dimension",
            "task_type": "dimension_ownership",
            "evidence_refs": ["drawing[5]"],
            "reason_code": "reviewed",
        }
        accepted = create_review_record(
            graph,
            reviewer_id="engineer-1",
            created_at="2026-08-23T12:00:00+03:00",
            decisions=[{**base, "decision": "accepted"}],
        )
        rejected = create_review_record(
            graph,
            reviewer_id="engineer-2",
            created_at="2026-08-23T12:05:00+03:00",
            decisions=[{**base, "decision_id": "decision.2", "decision": "rejected"}],
        )
        result = evaluate_review_records(graph, [accepted, rejected])
        self.assertEqual(result["validation"]["status"], "review_required")
        self.assertEqual(result["review_effort"]["conflicting_candidate_count"], 1)
        self.assertEqual(result["overall"]["reviewed_candidate_count"], 0)


if __name__ == "__main__":
    unittest.main()
