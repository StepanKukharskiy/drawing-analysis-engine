import inspect
import unittest

import fitz

from src.drawing_engine.core.dimension_attachment import DimensionAttachment, LineSegment
from src.drawing_engine.disciplines.rebar.rebar_path_graph import build_rebar_path_graph, enrich_projection_identities


def accepted_scale() -> DimensionAttachment:
    baseline = LineSegment((100.0, 370.0), (500.0, 370.0), 0.2, "dimension.baseline")
    left = LineSegment((100.0, 360.0), (100.0, 380.0), 0.2, "dimension.left")
    right = LineSegment((500.0, 360.0), (500.0, 380.0), 0.2, "dimension.right")
    return DimensionAttachment(
        attachment_id="dimension_attachment.synthetic",
        value_mm=400.0,
        text="400",
        text_method="native_pdf_text",
        text_confidence=1.0,
        text_bbox=(285.0, 360.0, 315.0, 375.0),
        orientation="horizontal",
        baseline=baseline,
        extension_lines=(left, right),
        dimension_points=((100.0, 370.0), (500.0, 370.0)),
        measured_points=((100.0, 350.0), (500.0, 350.0)),
        terminal_refs=(),
        scale_points_per_mm=1.0,
        scale_support=1,
        endpoint_alignment_residual=0.0,
        score=1.0,
        status="accepted",
        evidence=(),
    )


class RebarPathGraphTest(unittest.TestCase):
    def test_parallel_fragments_bridge_into_metric_projection_paths(self):
        document = fitz.open()
        page = document.new_page(width=600, height=400)
        shape = page.new_shape()
        shape.draw_rect(fitz.Rect(50, 40, 550, 350))
        shape.finish(color=(0, 0, 0), width=0.8)
        shape.commit()
        for y in (100.0, 180.0, 260.0):
            for start, end in (((100.0, y), (298.0, y)), ((302.0, y), (500.0, y))):
                shape = page.new_shape()
                shape.draw_line(start, end)
                shape.finish(color=(0, 0, 0), width=1.2)
                shape.commit()

        views = [
            {
                "id": "view_hypothesis.synthetic",
                "role_hypothesis": "reinforcement_view_candidate",
                "bbox_display": [40.0, 30.0, 560.0, 390.0],
            }
        ]
        contours = [
            {
                "id": "contour.synthetic",
                "closed": True,
                "bbox_display": [50.0, 40.0, 550.0, 350.0],
                "primitive_refs": ["drawing[0]"],
            }
        ]
        graph = build_rebar_path_graph(page, [accepted_scale()], views, contours, [])
        document.close()

        self.assertEqual(graph["layer"], "rebar_physical_path_graph")
        self.assertEqual(len(graph["fragments"]), 6)
        self.assertEqual(len(graph["repetition_groups"]), 1)
        self.assertEqual(graph["repetition_groups"][0]["count"], 6)
        bridges = [item for item in graph["connections"] if item["relation"] == "short_collinear_occlusion_bridge"]
        self.assertEqual(sum(item["state"] == "accepted" for item in bridges), 3)
        self.assertEqual(len(graph["components"]), 3)
        self.assertTrue(all(item["topology"] == "open_chain" for item in graph["components"]))
        self.assertTrue(all(item["projected_length"]["value_mm"] == 400.0 for item in graph["components"]))
        self.assertTrue(all(item["installed_centerline"]["value_mm"] is None for item in graph["components"]))
        self.assertTrue(graph["contract"]["visible_projection_is_not_physical_length"])
        self.assertFalse(graph["validation"]["schedule_values_used"])

    def test_module_contains_no_drawing_specific_dispatch(self):
        source = inspect.getsource(__import__("src.drawing_engine.disciplines.rebar.rebar_path_graph", fromlist=["*"])).lower()
        for forbidden in (".pdf", "k1", "k7", "column", "beam", "stair", "slab"):
            self.assertNotIn(forbidden, source)

    def test_cross_view_identity_accepts_exact_group_provenance_only(self):
        components = []
        fragments = []
        for index, (view_id, dimensionality) in enumerate(
            (("view.a", "axis_projection_1d"), ("view.b", "axis_end_projection_0d")),
            start=1,
        ):
            fragment_id = f"fragment.{index}"
            fragments.append({"id": fragment_id, "resolved_group_ids": ["group.1"]})
            components.append(
                {
                    "id": f"component.{index}",
                    "fragment_ids": [fragment_id],
                    "view_ids": [view_id],
                    "projection_dimensionality": {"value": dimensionality, "normalized_center_in_view": [0.5, 0.5]},
                    "physical_path_state": "candidate_nonbranching_projection",
                    "mark_hypotheses": ["8"],
                    "repetition_group_ids": [],
                    "candidate_score": 0.9,
                    "topology": "single_fragment",
                    "projected_length": {"value_mm": 400.0},
                }
            )
        graph = {"fragments": fragments, "components": components, "summary": {}}
        enrich_projection_identities(graph)

        identities = graph["cross_view_projection_identities"]
        self.assertEqual(len(identities), 1)
        self.assertEqual(identities[0]["state"], "accepted")
        self.assertEqual(identities[0]["basis"], "exact source primitive provenance to one independently resolved group")
        self.assertEqual(graph["summary"]["accepted_cross_view_identity_count"], 1)

    def test_similarity_candidates_must_share_reprojection_validated_scope(self):
        def component(component_id, view_id, dimensionality, scope_id):
            return {
                "id": component_id,
                "fragment_ids": [],
                "view_ids": [view_id],
                "coordinate_scope_ids": [scope_id],
                "projection_dimensionality": {"value": dimensionality, "normalized_center_in_view": [0.5, 0.5]},
                "physical_path_state": "candidate_nonbranching_projection",
                "mark_hypotheses": ["8"],
                "repetition_group_ids": [],
                "candidate_score": 0.9,
                "topology": "single_fragment",
                "style_signature": {"median_width_pt": 1.0},
                "projected_length": {"value_mm": 400.0},
            }

        graph = {
            "fragments": [],
            "components": [
                component("component.1", "view.a", "axis_projection_1d", "scope.1"),
                component("component.2", "view.b", "axis_end_projection_0d", "scope.1"),
                component("component.3", "view.c", "axis_end_projection_0d", "scope.2"),
            ],
            "coordinate_scopes": [{"coordinate_scope_id": "scope.1"}, {"coordinate_scope_id": "scope.2"}],
            "summary": {},
        }
        enrich_projection_identities(graph)

        candidates = [item for item in graph["cross_view_projection_identities"] if item["state"] == "review_candidate"]
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["coordinate_scope_ids"], ["scope.1"])
        self.assertIn("same reprojection-validated coordinate scope", candidates[0]["basis"])


if __name__ == "__main__":
    unittest.main()
