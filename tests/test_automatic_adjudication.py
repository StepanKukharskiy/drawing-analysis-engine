import copy
import unittest

from src.drawing_engine.core.automatic_adjudication import adjudicate_canonical_graph, ruleset_sha256
from src.drawing_engine.project.review_feedback import apply_review_record, validate_review_record
from src.drawing_engine.disciplines.concrete.solver_replay import build_solver_replay


def _entity(entity_id, entity_type, state, status, attributes, evidence):
    return {
        "id": entity_id,
        "entity_type": entity_type,
        "state": state,
        "status": status,
        "attributes": attributes,
        "provenance": {"page_key": "page:1", "evidence_refs": evidence},
    }


def graph():
    entities = [
        _entity(
            "canonical.view.section",
            "view",
            "inferred",
            None,
            {
                "role": "section_view_candidate",
                "confidence": 0.91,
                "frame": {
                    "section_label": "1-1",
                    "scale": {"state": "resolved"},
                    "metric_spans": [{"status": "accepted"}],
                },
            },
            ["word[1]"],
        ),
        _entity("canonical.object.1", "object", "inferred", "inferred", {}, ["drawing[1]"]),
        _entity(
            "canonical.mark.1",
            "mark_occurrence",
            "derived",
            "accepted",
            {"token": "1"},
            ["word[2]", "drawing[2]"],
        ),
        _entity(
            "canonical.detail.1",
            "detail_shape",
            "observed",
            "observed",
            {"mark_display": "1"},
            ["drawing[3]"],
        ),
        _entity(
            "canonical.family.1",
            "bar_family",
            "inferred",
            "single_view_family_candidate",
            {"mark": "1"},
            ["drawing[4]"],
        ),
        _entity(
            "canonical.path.1",
            "projected_path",
            "observed",
            "candidate_nonbranching_projection",
            {"physical_path_state": "candidate_nonbranching_projection"},
            ["drawing[5]"],
        ),
        _entity(
            "canonical.dimension.1",
            "dimension_attachment",
            "derived",
            "accepted",
            {},
            ["word[6]"],
        ),
        _entity(
            "canonical.contour.1",
            "contour",
            "derived",
            None,
            {"closed": True, "closure_validation": {"status": "pass"}},
            ["drawing[7]"],
        ),
        _entity(
            "canonical.contour.2",
            "contour",
            "observed",
            None,
            {"closed": True, "closure_validation": {"status": "pass"}},
            ["drawing[8]"],
        ),
    ]
    relations = [
        {
            "id": "canonical.relation.callout",
            "type": "callout_targets",
            "from": "canonical.mark.1",
            "to": "canonical.path.1",
            "state": "derived",
            "attributes": {},
            "provenance": {"page_key": "page:1", "evidence_refs": ["drawing[2]", "drawing[5]"]},
        },
        {
            "id": "canonical.relation.family",
            "type": "same_bar_family",
            "from": "canonical.mark.1",
            "to": "canonical.family.1",
            "state": "inferred",
            "attributes": {},
            "provenance": {"page_key": "page:1", "evidence_refs": ["word[2]"]},
        },
        {
            "id": "canonical.relation.detail",
            "type": "detail_defines",
            "from": "canonical.detail.1",
            "to": "canonical.family.1",
            "state": "inferred",
            "attributes": {},
            "provenance": {"page_key": "page:1", "evidence_refs": ["drawing[3]"]},
        },
        {
            "id": "canonical.relation.projection",
            "type": "projects_to",
            "from": "canonical.family.1",
            "to": "canonical.path.1",
            "state": "inferred",
            "attributes": {},
            "provenance": {"page_key": "page:1", "evidence_refs": ["drawing[5]"]},
        },
        {
            "id": "canonical.relation.dimension.accepted",
            "type": "dimension_of",
            "from": "canonical.dimension.1",
            "to": "canonical.contour.1",
            "state": "derived",
            "attributes": {},
            "provenance": {"page_key": "page:1", "evidence_refs": ["word[6]", "drawing[7]"]},
        },
        {
            "id": "canonical.relation.dimension.displaced",
            "type": "dimension_of",
            "from": "canonical.dimension.1",
            "to": "canonical.contour.2",
            "state": "inferred",
            "attributes": {},
            "provenance": {"page_key": "page:1", "evidence_refs": ["word[6]", "drawing[8]"]},
        },
        {
            "id": "canonical.relation.section",
            "type": "section_of",
            "from": "canonical.view.section",
            "to": "canonical.object.1",
            "state": "inferred",
            "attributes": {"projection_role": "compact_orthogonal_projection"},
            "provenance": {
                "page_key": "page:1",
                "evidence_refs": ["drawing[1]"],
                "basis": ["strong_cross_axis_overlap", "compatible_relative_extent"],
            },
        },
    ]
    return {
        "schema_version": "0.1.0",
        "document_key": "pdf-sha256:automatic-test",
        "entities": entities,
        "relations": relations,
        "unresolved": [
            {
                "kind": "dimension_ownership_attachment",
                "status": "candidate",
                "page_key": "page:1",
                "reason": "two owners remain equally supported",
                "evidence_refs": ["word[6]"],
            }
        ],
    }


class AutomaticAdjudicationTest(unittest.TestCase):
    def test_cut_relation_requires_prerequisite_certificate(self):
        base = graph()
        relation = {
            "id": "canonical.relation.cut",
            "type": "cut_at",
            "from": "canonical.view.section",
            "to": "canonical.object.1",
            "state": "derived",
            "attributes": {},
            "provenance": {"page_key": "page:1", "evidence_refs": ["trace.1"]},
        }
        base["relations"].append(relation)
        outcome = next(
            item
            for item in adjudicate_canonical_graph(base)["report"]["outcomes"]
            if item["candidate_id"] == relation["id"]
        )
        self.assertEqual(outcome["automatic_outcome"], "abstained")

        relation["attributes"]["integration_certificate"] = {
            "status": "passed",
            "title_segmentation_closed": True,
            "dimension_adjudication_closed": True,
            "evidence_refs": ["title.segment", "dimension.adjudication"],
        }
        outcome = next(
            item
            for item in adjudicate_canonical_graph(base)["report"]["outcomes"]
            if item["candidate_id"] == relation["id"]
        )
        self.assertEqual(outcome["automatic_outcome"], "accepted")
        self.assertEqual(outcome["rule"], "section_binding_prerequisites_closed")

    def test_closed_semantic_triangles_are_accepted_and_ambiguity_abstains(self):
        base = graph()
        original = copy.deepcopy(base)
        result = adjudicate_canonical_graph(base)
        outcomes = {item["candidate_id"]: item for item in result["report"]["outcomes"]}
        self.assertEqual(outcomes["canonical.relation.detail"]["rule"], "unique_exact_detail_mark_identity")
        self.assertEqual(outcomes["canonical.relation.family"]["rule"], "unique_exact_mark_family_identity")
        self.assertEqual(outcomes["canonical.relation.projection"]["rule"], "closed_mark_family_path_triangle")
        self.assertEqual(outcomes["canonical.relation.section"]["rule"], "unique_object_projection_membership")
        self.assertEqual(outcomes["canonical.relation.dimension.displaced"]["automatic_outcome"], "rejected")
        unresolved = next(item for item in outcomes.values() if item["subject_kind"] == "unresolved")
        self.assertEqual(unresolved["automatic_outcome"], "abstained")
        self.assertEqual(base, original)

    def test_machine_delta_validates_and_replays_without_human_or_schedule(self):
        base = graph()
        result = adjudicate_canonical_graph(base)
        delta = result["delta"]
        self.assertEqual(delta["layer"], "automatic_adjudication_delta")
        self.assertEqual(delta["reviewer"]["role"], "system")
        self.assertEqual(validate_review_record(base, delta)["status"], "pass")
        self.assertTrue(result["overlay"]["validation"]["base_graph_unchanged"])
        self.assertFalse(result["report"]["contract"]["human_required_for_decided_subset"])
        self.assertFalse(result["report"]["contract"]["schedule_values_used"])
        replay = build_solver_replay(
            base,
            result["overlay"],
            [{"page": 1, "quantities": [], "reinforcement_quantities": None}],
        )
        self.assertEqual(replay["status"], "solver_inputs_materialized")
        self.assertEqual(replay["summary"]["bar_family_path_triangle_count"], 1)
        self.assertFalse(replay["quantity_replay"]["changed"])
        self.assertFalse(replay["contract"]["schedule_values_used"])

    def test_ruleset_and_outcomes_are_deterministic(self):
        first = adjudicate_canonical_graph(graph())
        second = adjudicate_canonical_graph(copy.deepcopy(graph()))
        self.assertEqual(first, second)
        self.assertEqual(len(ruleset_sha256()), 64)
        applied = apply_review_record(graph(), first["delta"])
        self.assertEqual(applied, first["overlay"])


if __name__ == "__main__":
    unittest.main()
