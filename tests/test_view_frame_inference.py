import inspect
import json
import re
import unittest
from pathlib import Path

import fitz

from src.drawing_engine.core.view_frame_inference import SECTION_LABEL_RE, infer_view_frames


ROOT = Path(__file__).resolve().parents[1]


def view(view_id, role, bbox, dimension_refs=()):
    return {
        "id": view_id,
        "role_hypothesis": role,
        "bbox_display": bbox,
        "confidence": 0.9,
        "primitive_refs": [f"primitive.{view_id}"],
        "dimension_refs": list(dimension_refs),
    }


def dimension(dimension_id, scale, orientation, measured_points, score=0.95):
    return {
        "attachment_id": dimension_id,
        "status": "accepted",
        "scale_points_per_mm": scale,
        "orientation": orientation,
        "measured_points": measured_points,
        "score": score,
        "primitive_refs": [f"native.{dimension_id}"],
    }


class ViewFrameInferenceTest(unittest.TestCase):
    def test_complete_metric_equation_component_creates_containment_scope_only(self):
        equation = lambda equation_id, view_id, value, orientation: {
            "id": equation_id,
            "status": "accepted",
            "view_id": view_id,
            "extent_mm": value,
            "arithmetic": {"equation": f"100 * 10 = {value}"},
            "chain_attachment": {
                "orientation": orientation,
                "scale_points_per_mm": 0.2,
                "score": 1.0,
                "measured_points_display": [[0, 0], [100, 0]],
                "primitive_refs": [f"native.{equation_id}"],
            },
        }
        engineering = {
            "view_hypotheses": [
                view("view.1", "drawing_view_candidate", [0, 0, 200, 100]),
                view("view.2", "drawing_view_candidate", [220, 0, 420, 100]),
                view("view.3", "section_view_candidate", [440, 0, 540, 100]),
            ],
            "object_instance_graph": {"view_membership": {}, "instances": []},
            "metric_equation_graph": {
                "constraints": [
                    equation("eq.1a", "view.1", 5200, "horizontal"),
                    equation("eq.1b", "view.1", 1300, "vertical"),
                    equation("eq.2a", "view.2", 5200, "horizontal"),
                    equation("eq.2b", "view.2", 1300, "vertical"),
                    equation("eq.3a", "view.3", 5200, "horizontal"),
                    equation("eq.3b", "view.3", 1300, "vertical"),
                ],
                "scopes": [
                    {
                        "id": "metric_equation_scope.001",
                        "status": "accepted",
                        "view_ids": ["view.1", "view.2", "view.3"],
                        "relation_refs": ["shared.1", "shared.2"],
                        "evidence_refs": ["eq.1a", "eq.1b", "eq.2a", "eq.2b", "eq.3a", "eq.3b"],
                    }
                ],
            },
        }
        result = infer_view_frames(engineering)

        self.assertEqual(result["summary"]["metric_equation_scope_count"], 1)
        self.assertEqual(result["summary"]["object_scope_resolved_count"], 3)
        self.assertEqual(result["summary"]["metric_scale_resolved_count"], 3)
        self.assertTrue(all(item["parent_object"]["state"] == "resolved" for item in result["frames"]))
        self.assertTrue(all(item["parent_object"]["object_instance_id"] is None for item in result["frames"]))
        self.assertTrue(all(not item["parent_object"]["quantity_aggregation_eligible"] for item in result["frames"]))
        self.assertEqual(result["summary"]["axis_mapping_resolved_view_count"], 0)

    def test_metric_scale_origin_and_unique_object_scope_resolve_independently(self):
        engineering = {
            "view_hypotheses": [view("view.main", "drawing_view_candidate", [0, 0, 200, 100], ["dim.h", "dim.v"])],
            "claims": [
                {"id": "claim.h", "kind": "metric_dimension", "subject": "dim.h", "evidence_ref": "evidence.h"},
                {"id": "claim.v", "kind": "metric_dimension", "subject": "dim.v", "evidence_ref": "evidence.v"},
            ],
            "object_instance_graph": {
                "instances": [{"id": "object.1", "view_ids": ["view.main"]}],
                "view_membership": {"view.main": ["object.1"]},
            },
        }
        result = infer_view_frames(
            engineering,
            dimensions=[
                dimension("dim.h", 0.2, "horizontal", [[10, 10], [110, 10]]),
                dimension("dim.v", 0.202, "vertical", [[10.5, 10.2], [10.5, 90]]),
            ],
        )

        frame = result["frames"][0]
        self.assertEqual(frame["scale"]["state"], "resolved")
        self.assertAlmostEqual(frame["scale"]["value_points_per_mm"], 0.201, places=3)
        self.assertEqual(frame["origin"]["state"], "resolved")
        self.assertAlmostEqual(frame["origin"]["object_origin_display"][0], 10.25)
        self.assertEqual(frame["parent_object"]["object_instance_id"], "object.1")
        self.assertEqual(frame["projection_direction"]["state"], "unresolved")
        self.assertIn("projection_direction.vector_object_xyz", frame["unresolved_fields"])
        self.assertFalse(result["validation"]["schedule_values_used"])

    def test_equal_conflicting_scales_and_multiple_object_memberships_fail_closed(self):
        engineering = {
            "view_hypotheses": [view("view.a", "drawing_view_candidate", [0, 0, 200, 100], ["dim.1", "dim.2"])],
            "object_instance_graph": {
                "view_membership": {"view.a": ["object.1", "object.2"]},
                "instances": [],
            },
        }
        result = infer_view_frames(
            engineering,
            dimensions=[
                dimension("dim.1", 0.20, "horizontal", [[0, 0], [100, 0]]),
                dimension("dim.2", 0.35, "horizontal", [[0, 90], [100, 90]]),
            ],
        )

        frame = result["frames"][0]
        self.assertEqual(frame["scale"]["state"], "unresolved")
        self.assertEqual(len(frame["scale"]["candidates"]), 2)
        self.assertIsNone(frame["parent_object"]["object_instance_id"])
        self.assertEqual(frame["parent_object"]["candidates"], ["object.1", "object.2"])

    def test_unique_repeated_label_and_native_trace_accept_cut_relation(self):
        engineering = {
            "enforce_section_binding_prerequisites": True,
            "view_hypotheses": [
                view("view.parent", "reinforcement_view_candidate", [0, 0, 220, 200]),
                view("view.section", "section_view_candidate", [300, 0, 420, 120]),
            ],
            "section_rebar_observations": {
                "sections": [
                    {
                        "section_id": "section.5-5",
                        "label": "section 5-5",
                        "host_bbox_display": [310, 10, 410, 110],
                    }
                ]
            },
            "object_instance_graph": {
                "view_membership": {"view.parent": ["object.1"]},
                "instances": [],
            },
            "title_anchored_view_segmentation": {
                "segments": [
                    {"id": "segment.parent", "view_id": "view.parent", "state": "resolved", "title_anchor_ref": "title.parent", "title": "REINFORCEMENT", "normalised_title": "REINFORCEMENT", "evidence_refs": ["title.parent"]},
                    {"id": "segment.section", "view_id": "view.section", "state": "resolved", "title_anchor_ref": "label.section", "title": "5-5", "normalised_title": "5-5", "evidence_refs": ["label.section"]},
                ]
            },
            "dimension_adjudication": {
                "records": [
                    {"id": "dimension_adjudication.parent", "ownership_ref": "ownership.parent", "status": "accepted", "view_refs": ["view.parent"], "evidence_refs": ["ownership.parent"]},
                    {"id": "dimension_adjudication.section", "ownership_ref": "ownership.section", "status": "accepted", "view_refs": ["view.section"], "evidence_refs": ["ownership.section"]},
                ]
            },
        }
        roles = [
            {"id": "label.section", "resolved_role": "section_label", "text": "5-5", "bbox_display": [330, 10, 350, 20], "confidence": 0.98},
            {"id": "label.cut.left", "resolved_role": "unclassified_number", "text": "5", "bbox_display": [2, 90, 14, 100], "confidence": 0.7},
            {"id": "label.cut.right", "resolved_role": "unclassified_number", "text": "5", "bbox_display": [206, 90, 218, 100], "confidence": 0.7},
        ]
        observations = {
            "nodes": [
                {
                    "id": "drawing[7].item[0]",
                    "kind": "line_segment",
                    "orientation": "horizontal",
                    "bbox_display": [13, 94.9, 82, 95.1],
                },
                {
                    "id": "drawing[8].item[0]",
                    "kind": "line_segment",
                    "orientation": "horizontal",
                    "bbox_display": [138, 94.9, 207, 95.1],
                },
            ]
        }
        result = infer_view_frames(engineering, text_roles=roles, observation_graph=observations)

        self.assertEqual(len(result["relations"]), 1)
        relation = result["relations"][0]
        self.assertEqual(relation["section_label"], "5-5")
        self.assertEqual(relation["section_view_id"], "view.section")
        self.assertEqual(relation["parent_view_id"], "view.parent")
        self.assertEqual(relation["parent_object"]["object_instance_id"], "object.1")
        self.assertEqual(relation["trace"]["semantic_support"], "paired_identical_endpoint_tokens")
        self.assertEqual(relation["integration_certificate"]["status"], "passed")
        self.assertTrue(relation["integration_certificate"]["dimension_adjudication_closed"])
        self.assertEqual(len(relation["trace"]["primitive_refs"]), 2)
        frame = next(item for item in result["frames"] if item["view_id"] == "view.section")
        self.assertEqual(frame["parent_view"]["view_id"], "view.parent")
        self.assertEqual(frame["parent_object"]["object_instance_id"], "object.1")
        self.assertEqual(len(frame["projection_direction"]["candidates"]), 2)
        self.assertEqual(frame["projection_direction"]["state"], "axis_resolved_sign_unresolved")
        self.assertEqual(frame["axes"]["object_axis_mapping"]["state"], "resolved_relative")
        self.assertEqual(result["shared_coordinate_system"]["summary"]["axis_mapping_resolved_view_count"], 2)

    def test_unique_cut_trace_without_prerequisite_convergence_remains_candidate(self):
        engineering = {
            "enforce_section_binding_prerequisites": True,
            "view_hypotheses": [
                view("view.parent", "reinforcement_view_candidate", [0, 0, 220, 200]),
                view("view.section", "section_view_candidate", [300, 0, 420, 120]),
            ],
            "section_rebar_observations": {"sections": [{"section_id": "section.a-a", "label": "A-A", "host_bbox_display": [310, 10, 410, 110]}]},
            "object_instance_graph": {"view_membership": {}, "instances": []},
            "title_anchored_view_segmentation": {"segments": []},
            "dimension_adjudication": {"records": []},
        }
        roles = [
            {"id": "title.section", "resolved_role": "section_label", "text": "A-A", "bbox_display": [330, 10, 350, 20]},
            {"id": "cut.left", "resolved_role": "section_label", "text": "A-A", "bbox_display": [2, 90, 14, 100]},
            {"id": "cut.right", "resolved_role": "section_label", "text": "A-A", "bbox_display": [206, 90, 218, 100]},
        ]
        observations = {"nodes": [{"id": "trace", "kind": "line_segment", "orientation": "horizontal", "bbox_display": [13, 94.9, 207, 95.1]}]}
        result = infer_view_frames(engineering, text_roles=roles, observation_graph=observations)

        self.assertEqual(result["relations"], [])
        self.assertEqual(result["relation_candidates"][0]["integration_certificate"]["status"], "unresolved")
        self.assertIn("dimension_adjudication.section_view_owned_dimension", result["relation_candidates"][0]["unresolved_fields"])

    def test_more_popular_dimension_corner_does_not_override_second_possible_datum(self):
        engineering = {
            "view_hypotheses": [
                view("view.main", "drawing_view_candidate", [0, 0, 200, 120], ["h.top", "h.bottom", "v.left", "v.left.near"])
            ],
            "object_instance_graph": {"view_membership": {}, "instances": []},
        }
        result = infer_view_frames(
            engineering,
            dimensions=[
                dimension("h.top", 0.2, "horizontal", [[0, 0], [100, 0]]),
                dimension("h.bottom", 0.2, "horizontal", [[0, 100], [100, 100]]),
                dimension("v.left", 0.2, "vertical", [[0, 0], [0, 100]]),
                dimension("v.left.near", 0.2, "vertical", [[0.5, 0.2], [0.5, 50]]),
            ],
        )

        origin = result["frames"][0]["origin"]
        self.assertEqual(origin["state"], "unresolved")
        self.assertIsNone(origin["object_origin_display"])
        self.assertEqual(len(origin["candidates"]), 2)
        self.assertEqual(origin["candidates"][0]["support"], 3)
        self.assertIn("multiple dimension-axis datum intersections", origin["reason"])

    def test_two_equally_supported_traces_remain_candidates(self):
        engineering = {
            "view_hypotheses": [
                view("view.parent", "drawing_view_candidate", [0, 0, 220, 200]),
                view("view.section", "section_view_candidate", [300, 0, 420, 120]),
            ],
            "section_rebar_observations": {
                "sections": [
                    {"section_id": "section.a-a", "label": "A-A", "host_bbox_display": [310, 10, 410, 110]}
                ]
            },
            "object_instance_graph": {"view_membership": {}, "instances": []},
        }
        roles = [
            {"id": "label.section", "resolved_role": "section_label", "text": "A-A", "bbox_display": [330, 10, 350, 20]},
            {"id": "label.left", "resolved_role": "section_label", "text": "A-A", "bbox_display": [2, 85, 14, 105]},
            {"id": "label.right", "resolved_role": "section_label", "text": "A-A", "bbox_display": [206, 85, 218, 105]},
        ]
        observations = {
            "nodes": [
                {"id": "line.upper", "kind": "line_segment", "orientation": "horizontal", "bbox_display": [13, 91.9, 207, 92.1]},
                {"id": "line.lower", "kind": "line_segment", "orientation": "horizontal", "bbox_display": [13, 97.9, 207, 98.1]},
            ]
        }
        result = infer_view_frames(engineering, text_roles=roles, observation_graph=observations)

        self.assertEqual(result["relations"], [])
        self.assertEqual(len(result["relation_candidates"]), 1)
        self.assertEqual(result["object_scopes"], [])
        self.assertTrue(all(item["parent_object"]["state"] == "unresolved" for item in result["frames"]))
        candidate = result["relation_candidates"][0]
        self.assertEqual(candidate["reason"], "multiple cutting traces remain equally supported")
        self.assertEqual(len(candidate["trace_candidates"]), 2)

    def test_repeated_section_name_is_disambiguated_by_object_scope_not_collapsed(self):
        engineering = {
            "view_hypotheses": [
                view("parent.1", "drawing_view_candidate", [0, 0, 200, 100]),
                view("section.1", "section_view_candidate", [220, 0, 320, 100]),
                view("parent.2", "drawing_view_candidate", [0, 150, 200, 250]),
                view("section.2", "section_view_candidate", [220, 150, 320, 250]),
            ],
            "section_rebar_observations": {
                "sections": [
                    {"section_id": "section-observation.1", "label": "2-2", "host_bbox_display": [230, 10, 310, 90]},
                    {"section_id": "section-observation.2", "label": "2-2", "host_bbox_display": [230, 160, 310, 240]},
                ]
            },
            "object_instance_graph": {
                "view_membership": {
                    "parent.1": ["object.1"], "section.1": ["object.1"],
                    "parent.2": ["object.2"], "section.2": ["object.2"],
                },
                "instances": [],
            },
            "title_anchored_view_segmentation": {"segments": [
                {"id": f"segment.{view_id}", "view_id": view_id, "state": "resolved", "title": "2-2" if view_id.startswith("section") else "REINFORCEMENT", "normalised_title": "2-2" if view_id.startswith("section") else "REINFORCEMENT", "evidence_refs": [f"title.{view_id}"]}
                for view_id in ("parent.1", "section.1", "parent.2", "section.2")
            ]},
            "dimension_adjudication": {"records": [
                {"id": f"adjudication.{view_id}", "ownership_ref": f"ownership.{view_id}", "status": "accepted", "view_refs": [view_id], "evidence_refs": [f"dimension.{view_id}"]}
                for view_id in ("parent.1", "section.1", "parent.2", "section.2")
            ]},
        }
        roles = [
            {"id": "section-title.1", "resolved_role": "section_label", "text": "2-2", "bbox_display": [240, 10, 252, 20]},
            {"id": "section-title.2", "resolved_role": "section_label", "text": "2-2", "bbox_display": [240, 160, 252, 170]},
            {"id": "cut.1.left", "resolved_role": "section_label", "text": "2-2", "bbox_display": [2, 40, 14, 50]},
            {"id": "cut.1.right", "resolved_role": "section_label", "text": "2-2", "bbox_display": [186, 40, 198, 50]},
            {"id": "cut.2.left", "resolved_role": "section_label", "text": "2-2", "bbox_display": [2, 190, 14, 200]},
            {"id": "cut.2.right", "resolved_role": "section_label", "text": "2-2", "bbox_display": [186, 190, 198, 200]},
        ]
        observations = {
            "nodes": [
                {"id": "trace.1", "kind": "line_segment", "orientation": "horizontal", "bbox_display": [13, 44.9, 187, 45.1]},
                {"id": "trace.2", "kind": "line_segment", "orientation": "horizontal", "bbox_display": [13, 194.9, 187, 195.1]},
            ]
        }
        result = infer_view_frames(engineering, text_roles=roles, observation_graph=observations)

        self.assertEqual(len(result["section_view_bindings"]), 2)
        self.assertEqual(len(result["relations"]), 2)
        self.assertEqual(
            {(item["section_view_id"], item["parent_view_id"]) for item in result["relations"]},
            {("section.1", "parent.1"), ("section.2", "parent.2")},
        )

    def test_one_section_with_two_equally_certified_parents_abstains(self):
        engineering = {
            "view_hypotheses": [
                view("parent.1", "drawing_view_candidate", [0, 0, 200, 100]),
                view("parent.2", "drawing_view_candidate", [0, 150, 200, 250]),
                view("section.1", "section_view_candidate", [300, 0, 400, 100]),
            ],
            "section_rebar_observations": {"sections": [
                {"section_id": "section-observation.1", "label": "A-A", "host_bbox_display": [305, 5, 395, 95]}
            ]},
            "object_instance_graph": {"view_membership": {}, "instances": []},
            "title_anchored_view_segmentation": {"segments": [
                {"id": f"segment.{view_id}", "view_id": view_id, "state": "resolved", "title": "A-A" if view_id == "section.1" else "PLAN", "normalised_title": "A-A" if view_id == "section.1" else "PLAN", "evidence_refs": [f"title.{view_id}"]}
                for view_id in ("parent.1", "parent.2", "section.1")
            ]},
            "dimension_adjudication": {"records": [
                {"id": f"adjudication.{view_id}", "status": "accepted", "view_refs": [view_id], "evidence_refs": [f"dimension.{view_id}"]}
                for view_id in ("parent.1", "parent.2", "section.1")
            ]},
        }
        roles = [
            {"id": "section.title", "resolved_role": "section_label", "text": "A-A", "bbox_display": [330, 10, 350, 20]},
            {"id": "parent.1.left", "resolved_role": "unclassified_text", "text": "A", "bbox_display": [2, 40, 12, 50]},
            {"id": "parent.1.right", "resolved_role": "unclassified_text", "text": "A", "bbox_display": [188, 40, 198, 50]},
            {"id": "parent.2.left", "resolved_role": "unclassified_text", "text": "A", "bbox_display": [2, 190, 12, 200]},
            {"id": "parent.2.right", "resolved_role": "unclassified_text", "text": "A", "bbox_display": [188, 190, 198, 200]},
        ]
        observations = {"nodes": [
            {"id": "trace.1", "kind": "line_segment", "orientation": "horizontal", "bbox_display": [11, 44.9, 189, 45.1]},
            {"id": "trace.2", "kind": "line_segment", "orientation": "horizontal", "bbox_display": [11, 194.9, 189, 195.1]},
        ]}

        result = infer_view_frames(engineering, text_roles=roles, observation_graph=observations)

        self.assertEqual(result["relations"], [])
        conflicts = [
            item for item in result["relation_candidates"]
            if "ambiguous parent" in item.get("reason", "")
        ]
        self.assertEqual(len(conflicts), 2)

    def test_duplicate_section_titles_for_one_parent_abstain(self):
        engineering = {
            "view_hypotheses": [
                view("parent.1", "drawing_view_candidate", [0, 0, 200, 100]),
                view("section.1", "section_view_candidate", [300, 0, 400, 100]),
                view("section.2", "section_view_candidate", [300, 150, 400, 250]),
            ],
            "section_rebar_observations": {"sections": [
                {"section_id": "section-observation.1", "label": "A-A", "host_bbox_display": [305, 5, 395, 95]},
                {"section_id": "section-observation.2", "label": "A-A", "host_bbox_display": [305, 155, 395, 245]},
            ]},
            "object_instance_graph": {"view_membership": {}, "instances": []},
            "title_anchored_view_segmentation": {"segments": [
                {"id": f"segment.{view_id}", "view_id": view_id, "state": "resolved", "title": "PLAN" if view_id == "parent.1" else "A-A", "normalised_title": "PLAN" if view_id == "parent.1" else "A-A", "evidence_refs": [f"title.{view_id}"]}
                for view_id in ("parent.1", "section.1", "section.2")
            ]},
            "dimension_adjudication": {"records": [
                {"id": f"adjudication.{view_id}", "status": "accepted", "view_refs": [view_id], "evidence_refs": [f"dimension.{view_id}"]}
                for view_id in ("parent.1", "section.1", "section.2")
            ]},
        }
        roles = [
            {"id": "section.title.1", "resolved_role": "section_label", "text": "A-A", "bbox_display": [330, 10, 350, 20]},
            {"id": "section.title.2", "resolved_role": "section_label", "text": "A-A", "bbox_display": [330, 160, 350, 170]},
            {"id": "parent.left", "resolved_role": "unclassified_text", "text": "A", "bbox_display": [2, 40, 12, 50]},
            {"id": "parent.right", "resolved_role": "unclassified_text", "text": "A", "bbox_display": [188, 40, 198, 50]},
        ]
        observations = {"nodes": [
            {"id": "trace.1", "kind": "line_segment", "orientation": "horizontal", "bbox_display": [11, 44.9, 189, 45.1]}
        ]}

        result = infer_view_frames(engineering, text_roles=roles, observation_graph=observations)

        self.assertEqual(result["relations"], [])
        conflicts = [
            item for item in result["relation_candidates"]
            if "duplicate section binding" in item.get("reason", "")
        ]
        self.assertEqual(len(conflicts), 2)

    def test_missing_label_text_is_reported_instead_of_inventing_parent(self):
        engineering = {
            "view_hypotheses": [view("view.section", "section_view_candidate", [0, 0, 100, 100])],
            "section_rebar_observations": {
                "sections": [
                    {"section_id": "section.1-1", "label": "1-1", "host_bbox_display": [5, 5, 95, 95]}
                ]
            },
            "object_instance_graph": {"view_membership": {}, "instances": []},
        }
        result = infer_view_frames(engineering)

        self.assertEqual(result["relations"], [])
        self.assertEqual(result["relation_candidates"][0]["reason"], "no repeated section-label occurrence is scoped to another view")
        frame = result["frames"][0]
        self.assertIsNone(frame["parent_view"]["view_id"])
        self.assertIn("parent_view.view_id", frame["unresolved_fields"])

    def test_rotated_native_sheet_links_section_5_5_through_real_path_nodes(self):
        engineering = json.loads((ROOT / "output/object_agnostic/1.engineering-graph.json").read_text())["pages"][0]
        observation = json.loads((ROOT / "output/object_agnostic/1.drawing-scene.json").read_text())["pages"][0]
        document = fitz.open(ROOT / "1.pdf")
        page = document[0]
        roles = []
        for index, word in enumerate(page.get_text("words")):
            text = str(word[4]).strip()
            if re.fullmatch(r"[1-9]\d*|[A-ZА-Я]", text.upper()):
                roles.append(
                    {
                        "id": f"native.word.{index}",
                        "text": text,
                        "bbox_display": list(word[:4]),
                        "resolved_role": "unclassified_number" if text.isdigit() else "unclassified_text",
                    }
                )
        for block_index, block in enumerate(page.get_text("dict").get("blocks", [])):
            for line_index, line in enumerate(block.get("lines", [])):
                text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
                if SECTION_LABEL_RE.search(text.upper()):
                    roles.append(
                        {
                            "id": f"native.line.{block_index}.{line_index}",
                            "text": text,
                            "bbox_display": list(line["bbox"]),
                            "resolved_role": "section_label",
                        }
                    )
        matrix = page.rotation_matrix
        result = infer_view_frames(
            engineering,
            text_roles=roles,
            observation_graph=observation,
            page_number=1,
            text_to_display_transform=(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e, matrix.f),
        )

        matches = [item for item in result["relation_candidates"] if item["section_label"] == "5-5"]
        self.assertGreaterEqual(len(matches), 1)
        self.assertEqual(result["relations"], [])
        self.assertTrue(all(item["integration_certificate"]["status"] == "unresolved" for item in matches))
        self.assertTrue(
            all(
                item["trace_candidates"][0]["semantic_support"]
                == "paired_identical_endpoint_tokens_with_unique_native_stems"
                for item in matches
            )
        )
        self.assertTrue(all(item["trace_candidates"][0]["primitive_refs"] for item in matches))

    def test_module_has_no_document_specific_dispatch(self):
        source = inspect.getsource(__import__("src.drawing_engine.core.view_frame_inference", fromlist=["*"])).lower()
        for forbidden in (".pdf", "k1", "k7", "3179"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
