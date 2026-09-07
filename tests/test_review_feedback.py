import copy
import tempfile
import unittest
from pathlib import Path

from src.drawing_engine.project.review_feedback import (
    apply_review_record,
    build_review_candidate_catalog,
    build_single_target_leader_review_catalog,
    canonical_graph_sha256,
    create_review_record,
    load_review_record,
    validate_review_record,
    write_review_record,
)


def canonical_graph():
    entities = [
        {
            "id": "canonical.view.1",
            "entity_type": "view",
            "state": "inferred",
            "attributes": {"role_hypothesis": "section_view_candidate", "confidence": 0.7},
            "provenance": {"page_key": "page:1", "evidence_refs": ["word[1]"]},
        },
        {
            "id": "canonical.path.1",
            "entity_type": "projected_path",
            "state": "observed",
            "attributes": {"projection_dimensionality": "axis_projection_1d"},
            "provenance": {"page_key": "page:1", "evidence_refs": ["drawing[1]"]},
        },
        {
            "id": "canonical.path.2",
            "entity_type": "projected_path",
            "state": "observed",
            "attributes": {"projection_dimensionality": "axis_projection_1d"},
            "provenance": {"page_key": "page:1", "evidence_refs": ["drawing[2]"]},
        },
        {
            "id": "canonical.contour.1",
            "entity_type": "contour",
            "state": "observed",
            "attributes": {},
            "provenance": {"page_key": "page:1", "evidence_refs": ["drawing[3]"]},
        },
    ]
    relations = [
        {
            "id": "canonical.relation.leader",
            "type": "callout_targets",
            "from": "canonical.view.1",
            "to": "canonical.path.1",
            "state": "derived",
            "attributes": {"score": 0.9},
            "provenance": {"page_key": "page:1", "evidence_refs": ["drawing[4]"]},
        },
        {
            "id": "canonical.relation.dimension",
            "type": "dimension_of",
            "from": "canonical.view.1",
            "to": "canonical.contour.1",
            "state": "unknown",
            "attributes": {"score": 0.6},
            "provenance": {"page_key": "page:1", "evidence_refs": ["drawing[5]"]},
        },
    ]
    return {
        "schema_version": "0.1.0",
        "document_key": "pdf-sha256:test",
        "entities": entities,
        "relations": relations,
        "unresolved": [{"kind": "cut_at_relation_candidate", "page_key": "page:1"}],
    }


def graph_with_vague_leader():
    graph = canonical_graph()
    graph["entities"].append(
        {
            "id": "canonical.mark.ambiguous",
            "entity_type": "mark_occurrence",
            "state": "unknown",
            "status": "ambiguous",
            "attributes": {
                "token": "7",
                "leader_trace": {
                    "terminals": [[10.0, 10.0], [50.0, 10.0]],
                    "segments": [{"start": [10.0, 10.0], "end": [50.0, 10.0]}],
                },
            },
            "provenance": {"page_key": "page:1", "evidence_refs": ["drawing[7]"]},
        }
    )
    graph["entities"][1]["attributes"].update(
        {
            "bbox_display": [49.0, 9.0, 60.0, 11.0],
            "source_fragments": [
                {"geometry": {"points_display": [[50.0, 10.0], [60.0, 10.0]]}}
            ],
        }
    )
    graph["entities"][2]["attributes"].update(
        {
            "bbox_display": [90.0, 9.0, 100.0, 11.0],
            "source_fragments": [
                {"geometry": {"points_display": [[90.0, 10.0], [100.0, 10.0]]}}
            ],
        }
    )
    return graph


def graph_with_single_target_leader():
    graph = graph_with_vague_leader()
    graph["relations"].append(
        {
            "id": "canonical.relation.single-leader",
            "type": "callout_targets",
            "from": "canonical.mark.ambiguous",
            "to": "canonical.path.1",
            "state": "derived",
            "provenance": {"page_key": "page:1", "evidence_refs": ["drawing[7]", "drawing[1]"]},
        }
    )
    return graph


class ReviewFeedbackTest(unittest.TestCase):
    def test_catalog_has_stable_task_candidates_without_schedule_values(self):
        graph = canonical_graph()
        first = build_review_candidate_catalog(graph)
        second = build_review_candidate_catalog(copy.deepcopy(graph))
        self.assertEqual(first, second)
        self.assertEqual(first["base_canonical_graph_sha256"], canonical_graph_sha256(graph))
        tasks = {item["candidate_id"]: item["task_type"] for item in first["candidates"]}
        self.assertEqual(tasks["canonical.view.1"], "view_role")
        self.assertEqual(tasks["canonical.relation.leader"], "leader_target")
        self.assertEqual(tasks["canonical.relation.dimension"], "dimension_ownership")
        self.assertFalse(first["contract"]["schedule_values_used"])

    def test_single_target_catalog_is_separate_and_validates_its_candidates(self):
        graph = graph_with_single_target_leader()
        base = build_review_candidate_catalog(graph)
        expanded = build_single_target_leader_review_catalog(graph)
        self.assertNotEqual(base["catalog_sha256"], expanded["catalog_sha256"])
        self.assertEqual(expanded["base_catalog_sha256"], base["catalog_sha256"])
        groups = [
            item
            for item in expanded["candidates"]
            if item.get("unresolved", {}).get("kind") == "single_target_leader_group"
        ]
        alternatives = [
            item
            for item in expanded["candidates"]
            if item.get("unresolved", {}).get("kind") == "single_target_leader_pair"
        ]
        self.assertEqual(len(groups), 1)
        self.assertTrue(alternatives)
        relation = next(
            item for item in expanded["candidates"] if item["candidate_id"] == "canonical.relation.single-leader"
        )
        record = create_review_record(
            graph,
            reviewer_id="engineer-1",
            created_at="2026-08-25T12:00:00+03:00",
            decisions=[
                {
                    "decision_id": "decision.accept.single-leader",
                    "candidate_id": relation["candidate_id"],
                    "task_type": "leader_target",
                    "decision": "accepted",
                    "evidence_refs": relation["evidence_refs"],
                    "reason_code": "engineer_confirmed_leader_target",
                }
            ],
        )
        record["catalog_sha256"] = expanded["catalog_sha256"]
        self.assertEqual(validate_review_record(graph, record)["status"], "pass")
        overlay = apply_review_record(graph, record)
        reviewed = next(
            item for item in overlay["effective_relations"] if item["id"] == relation["candidate_id"]
        )
        self.assertEqual(reviewed["review_status"], "accepted")

    def test_review_delta_validates_and_applies_without_mutating_base(self):
        graph = canonical_graph()
        original = copy.deepcopy(graph)
        decisions = [
            {
                "decision_id": "decision.accept.dimension",
                "candidate_id": "canonical.relation.dimension",
                "task_type": "dimension_ownership",
                "decision": "accepted",
                "evidence_refs": ["drawing[5]"],
                "reason_code": "dimension_terminals_confirmed",
            },
            {
                "decision_id": "decision.correct.leader",
                "candidate_id": "canonical.relation.leader",
                "task_type": "leader_target",
                "decision": "corrected",
                "evidence_refs": ["drawing[4]", "drawing[2]"],
                "reason_code": "wrong_leader_target",
                "correction": {
                    "kind": "relation",
                    "relation_type": "callout_targets",
                    "from": "canonical.view.1",
                    "to": "canonical.path.2",
                },
            },
            {
                "decision_id": "decision.correct.role",
                "candidate_id": "canonical.view.1",
                "task_type": "view_role",
                "decision": "corrected",
                "evidence_refs": ["word[1]"],
                "reason_code": "wrong_view_role",
                "correction": {
                    "kind": "attribute",
                    "entity_id": "canonical.view.1",
                    "field": "role_hypothesis",
                    "value": "reinforcement_view_candidate",
                },
            },
        ]
        record = create_review_record(
            graph,
            reviewer_id="engineer-1",
            created_at="2026-08-23T12:00:00+03:00",
            decisions=decisions,
        )
        self.assertEqual(validate_review_record(graph, record)["status"], "pass")
        overlay = apply_review_record(graph, record)
        self.assertEqual(graph, original)
        self.assertTrue(overlay["validation"]["base_graph_unchanged"])
        self.assertTrue(overlay["validation"]["relation_endpoints_exist"])
        relations = {item["id"]: item for item in overlay["effective_relations"]}
        self.assertNotIn("canonical.relation.leader", relations)
        corrected = next(item for item in relations.values() if item.get("review_status") == "corrected")
        self.assertEqual(corrected["to"], "canonical.path.2")
        entities = {item["id"]: item for item in overlay["effective_entities"]}
        self.assertEqual(
            entities["canonical.view.1"]["reviewed_attributes"]["role_hypothesis"],
            "reinforcement_view_candidate",
        )
        self.assertFalse(overlay["contract"]["schedule_values_used"])

    def test_vague_leader_candidates_can_create_a_reviewed_relation(self):
        graph = graph_with_vague_leader()
        catalog = build_review_candidate_catalog(graph)
        group = next(
            item
            for item in catalog["candidates"]
            if item.get("unresolved", {}).get("kind") == "vague_leader_target_group"
        )
        pairs = [
            item
            for item in catalog["candidates"]
            if item.get("unresolved", {}).get("kind") == "vague_leader_target_pair"
        ]
        pairs.sort(key=lambda item: item["unresolved"]["terminal_distance_points"])
        self.assertEqual(group["unresolved"]["mark_id"], "canonical.mark.ambiguous")
        self.assertEqual(pairs[0]["to"], "canonical.path.1")
        self.assertEqual(pairs[0]["unresolved"]["terminal_distance_points"], 0.0)
        record = create_review_record(
            graph,
            reviewer_id="engineer-1",
            created_at="2026-08-24T12:00:00+03:00",
            decisions=[
                {
                    "decision_id": "decision.resolve.vague-leader",
                    "candidate_id": pairs[0]["candidate_id"],
                    "task_type": "leader_target",
                    "decision": "corrected",
                    "evidence_refs": pairs[0]["evidence_refs"],
                    "reason_code": "engineer_selected_leader_target",
                    "correction": {
                        "kind": "relation",
                        "relation_type": "callout_targets",
                        "from": pairs[0]["from"],
                        "to": pairs[0]["to"],
                    },
                }
            ],
        )
        self.assertEqual(validate_review_record(graph, record)["status"], "pass")
        overlay = apply_review_record(graph, record)
        corrected = [
            item
            for item in overlay["effective_relations"]
            if item.get("review_status") == "corrected"
        ]
        self.assertEqual(corrected[0]["from"], "canonical.mark.ambiguous")
        self.assertEqual(corrected[0]["to"], "canonical.path.1")

    def test_vague_leader_candidates_exclude_the_leader_trace_itself(self):
        graph = graph_with_vague_leader()
        mark = next(item for item in graph["entities"] if item["id"] == "canonical.mark.ambiguous")
        mark["attributes"]["leader_trace"]["segments"][0]["drawing_ref"] = "drawing[leader].item[0]"
        graph["entities"].append(
            {
                "id": "canonical.path.leader-trace",
                "entity_type": "projected_path",
                "state": "observed",
                "attributes": {
                    "bbox_display": [10.0, 10.0, 50.0, 10.0],
                    "source_fragments": [
                        {
                            "primitive_ref": "drawing[leader].item[0]",
                            "geometry": {"points_display": [[10.0, 10.0], [50.0, 10.0]]},
                        }
                    ],
                },
                "provenance": {"page_key": "page:1", "evidence_refs": ["drawing[leader].item[0]"]},
            }
        )
        catalog = build_review_candidate_catalog(graph)
        targets = {
            item["to"]
            for item in catalog["candidates"]
            if item.get("unresolved", {}).get("kind") == "vague_leader_target_pair"
        }
        self.assertNotIn("canonical.path.leader-trace", targets)

    def test_stale_hash_and_noncanonical_correction_fail_closed(self):
        graph = canonical_graph()
        record = create_review_record(
            graph,
            reviewer_id="engineer-1",
            created_at="2026-08-23T12:00:00+03:00",
            decisions=[
                {
                    "decision_id": "decision.1",
                    "candidate_id": "canonical.relation.leader",
                    "task_type": "leader_target",
                    "decision": "corrected",
                    "evidence_refs": ["drawing[4]"],
                    "reason_code": "wrong_leader_target",
                    "correction": {
                        "kind": "relation",
                        "relation_type": "callout_targets",
                        "from": "canonical.view.1",
                        "to": "not-canonical",
                    },
                }
            ],
        )
        record["base_canonical_graph_sha256"] = "stale"
        validation = validate_review_record(graph, record)
        self.assertEqual(validation["status"], "fail")
        self.assertTrue(any("hash" in error for error in validation["errors"]))
        self.assertTrue(any("canonical entities" in error for error in validation["errors"]))
        with self.assertRaises(ValueError):
            apply_review_record(graph, record)

    def test_unknown_fields_and_schedule_evidence_are_rejected(self):
        graph = canonical_graph()
        record = create_review_record(
            graph,
            reviewer_id="engineer-1",
            created_at="2026-08-23T12:00:00+03:00",
            decisions=[
                {
                    "decision_id": "decision.1",
                    "candidate_id": "canonical.relation.dimension",
                    "task_type": "dimension_ownership",
                    "decision": "accepted",
                    "evidence_refs": ["declared_schedule.cell[1]"],
                    "reason_code": "not_allowed",
                    "schedule_value": 400,
                }
            ],
        )
        validation = validate_review_record(graph, record)
        self.assertEqual(validation["status"], "fail")
        self.assertTrue(any("unsupported fields" in error for error in validation["errors"]))
        self.assertTrue(any("schedule evidence" in error for error in validation["errors"]))

    def test_review_record_persistence_is_validated_and_atomic(self):
        graph = canonical_graph()
        record = create_review_record(
            graph,
            reviewer_id="engineer-1",
            created_at="2026-08-23T12:00:00+03:00",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.json"
            write_review_record(path, graph, record, allow_empty=True)
            self.assertEqual(load_review_record(path), record)


if __name__ == "__main__":
    unittest.main()
