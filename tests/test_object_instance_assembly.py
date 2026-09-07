import inspect
import unittest

import fitz

from src.drawing_engine.core.object_instance_assembly import (
    assemble_object_instances,
    assemble_relation_certified_object_instances,
    scope_rebar_path_graph,
)
from src.drawing_engine.disciplines.rebar.rebar_path_graph import enrich_projection_identities


def view(view_id, bbox, dimensions, orthogonality, primitives=40):
    return {
        "id": view_id,
        "bbox_display": bbox,
        "features": {
            "accepted_dimensions": dimensions,
            "orthogonal_axis_ratio": orthogonality,
            "primitive_count": primitives,
        },
    }


class ObjectInstanceAssemblyTest(unittest.TestCase):
    def test_certified_relation_creates_instance_without_layout_proximity(self):
        frame_graph = {
            "relations": [
                {
                    "id": "cut.1",
                    "state": "accepted",
                    "section_label": "2-2",
                    "parent_view_id": "primary.a",
                    "section_view_id": "section.a",
                    "confidence": 0.9,
                    "trace": {"primitive_refs": ["native.marker.left", "native.marker.right"]},
                    "integration_certificate": {
                        "status": "passed",
                        "parent_segment_id": "title.primary",
                        "section_segment_id": "title.section",
                        "relation_scoped_metric_pair": {
                            "status": "passed",
                            "transverse_extent_mm": 200.0,
                            "spans": [{"id": "span.primary"}, {"id": "span.section"}],
                        },
                    },
                    "profile_section_reprojection_certificate": {"status": "passed"},
                }
            ],
            "object_scopes": [
                {
                    "id": "scope.1",
                    "state": "resolved_relative",
                    "view_ids": ["primary.a", "section.a"],
                    "confidence": 0.88,
                    "contour_correspondence_ref": "contour.match.1",
                    "reprojection_validation_refs": ["reprojection.1"],
                }
            ],
        }
        views = [
            view("primary.a", [0, 0, 100, 100], 0, 0.1),
            view("section.a", [900, 700, 920, 800], 0, 0.7),
        ]
        result = assemble_relation_certified_object_instances(frame_graph, views)

        self.assertEqual(result["summary"]["object_instance_count"], 1)
        self.assertEqual(result["instances"][0]["extrusion_depth_mm"], 200.0)
        self.assertFalse(result["validation"]["spatial_proximity_used_as_certificate"])

    def test_repeated_projection_layout_yields_three_instances_and_one_shared_view(self):
        views = [
            view("primary.a", [50, 50, 300, 250], 2, 0.25),
            view("section.a", [330, 60, 370, 240], 1, 0.72),
            view("primary.b", [50, 320, 300, 520], 3, 0.25),
            view("section.b", [330, 330, 370, 510], 1, 0.72),
            view("primary.c", [500, 50, 750, 250], 2, 0.25),
            view("section.c", [780, 60, 820, 240], 2, 0.72),
            view("shared.metric", [500, 320, 750, 520], 2, 0.25),
            view("schedule", [850, 50, 980, 300], 0, 0.85, 500),
        ]
        result = assemble_object_instances(fitz.Rect(0, 0, 1000, 800), views)

        self.assertEqual(result["summary"]["object_instance_count"], 3)
        self.assertEqual(result["layout_consensus"]["selected_mode"], "right")
        self.assertEqual(
            {(item["primary_view_id"], item["section_view_id"]) for item in result["instances"]},
            {("primary.a", "section.a"), ("primary.b", "section.b"), ("primary.c", "section.c")},
        )
        self.assertEqual(result["shared_supporting_views"][0]["view_id"], "shared.metric")
        self.assertFalse(result["validation"]["schedule_values_used"])
        self.assertFalse(result["validation"]["object_class_template_used"])

    def test_projection_identity_candidates_cannot_cross_object_scopes(self):
        graph = {
            "fragments": [
                {"id": "fragment.1", "view_ids": ["view.a"], "resolved_group_ids": []},
                {"id": "fragment.2", "view_ids": ["view.b"], "resolved_group_ids": []},
            ],
            "components": [
                {
                    "id": "component.1", "fragment_ids": ["fragment.1"], "view_ids": ["view.a"],
                    "projection_dimensionality": {"value": "axis_projection_1d", "normalized_center_in_view": [0.5, 0.5]},
                    "physical_path_state": "candidate_nonbranching_projection", "mark_hypotheses": ["8"],
                    "repetition_group_ids": ["repeat.a"], "candidate_score": 0.9, "topology": "single_fragment",
                    "style_signature": {"median_width_pt": 1.0}, "projected_length": {"value_mm": 400.0},
                },
                {
                    "id": "component.2", "fragment_ids": ["fragment.2"], "view_ids": ["view.b"],
                    "projection_dimensionality": {"value": "axis_projection_1d", "normalized_center_in_view": [0.5, 0.5]},
                    "physical_path_state": "candidate_nonbranching_projection", "mark_hypotheses": ["8"],
                    "repetition_group_ids": ["repeat.b"], "candidate_score": 0.9, "topology": "single_fragment",
                    "style_signature": {"median_width_pt": 1.0}, "projected_length": {"value_mm": 400.0},
                },
            ],
            "summary": {}, "validation": {},
        }
        assembly = {
            "instances": [
                {"id": "object.1", "view_ids": ["view.a"]},
                {"id": "object.2", "view_ids": ["view.b"]},
            ],
            "view_membership": {"view.a": ["object.1"], "view.b": ["object.2"]},
        }
        scope_rebar_path_graph(graph, assembly)
        enrich_projection_identities(graph)

        self.assertEqual(graph["cross_view_projection_identities"], [])
        self.assertEqual(graph["summary"]["object_scoped_component_count"], 2)
        self.assertEqual(graph["validation"]["cross_object_candidate_identity_count"], 0)

    def test_path_components_inherit_shared_coordinate_scope(self):
        graph = {
            "fragments": [{"id": "fragment.1", "view_ids": ["view.a"], "resolved_group_ids": []}],
            "components": [{"id": "component.1", "fragment_ids": ["fragment.1"], "view_ids": ["view.a"]}],
            "summary": {},
            "validation": {},
        }
        frame_graph = {
            "shared_coordinate_system": {
                "view_membership": {"view.a": ["shared_coordinate_scope.001"]},
                "scopes": [
                    {"id": "shared_coordinate_scope.001", "view_ids": ["view.a"], "bar_identity_eligible": True}
                ],
            }
        }
        scope_rebar_path_graph(graph, {"instances": [], "view_membership": {}}, frame_graph)

        self.assertEqual(graph["fragments"][0]["coordinate_scope_ids"], ["shared_coordinate_scope.001"])
        self.assertEqual(graph["components"][0]["coordinate_scope_ids"], ["shared_coordinate_scope.001"])
        self.assertEqual(graph["summary"]["coordinate_scoped_component_count"], 1)

    def test_module_contains_no_drawing_specific_dispatch(self):
        source = inspect.getsource(__import__("src.drawing_engine.core.object_instance_assembly", fromlist=["*"])).lower()
        for forbidden in (".pdf", "k1", "k7", "column", "beam", "stair", "slab"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
