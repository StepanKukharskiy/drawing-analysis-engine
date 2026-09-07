import inspect
import unittest

from src.drawing_engine.core.contour_correspondence import solve_contour_correspondence


def contour(contour_id, coordinates, orientation="horizontal"):
    segments = []
    for index, value in enumerate(coordinates):
        if orientation == "horizontal":
            start, end = [value, 0], [value, 100]
        else:
            start, end = [0, value], [100, value]
        segments.append(
            {
                "id": f"{contour_id}.segment.{index}",
                "kind": "line",
                "start_display": start,
                "end_display": end,
                "primitive_ref": f"drawing.{contour_id}.{index}",
            }
        )
    return {
        "id": contour_id,
        "closed": True,
        "bbox_display": [min(coordinates), 0, max(coordinates), 100],
        "primitive_refs": [f"drawing.{contour_id}"],
        "segments_display": segments,
        "topology": {"segment_count": len(segments), "cycle_rank": 1},
    }


def frame(view_id):
    return {
        "view_id": view_id,
        "scale": {"state": "resolved", "value_points_per_mm": 1.0},
    }


def coordinates():
    return {
        "constraints": [
            {
                "id": "coordinate_constraint.1",
                "parent_view_id": "parent",
                "child_view_id": "child",
                "mode": "cut_horizontal",
            }
        ],
        "scopes": [
            {
                "id": "scope.1",
                "state": "resolved_relative",
                "constraint_ids": ["coordinate_constraint.1"],
            }
        ],
        "summary": {},
    }


class ContourCorrespondenceTest(unittest.TestCase):
    def test_asymmetric_native_levels_resolve_one_signed_transform(self):
        parent = contour("parent.contour", [0, 20, 100])
        child = contour("child.contour", [200, 220, 300])
        shared = coordinates()

        result = solve_contour_correspondence(
            [frame("parent"), frame("child")],
            shared,
            [parent, child],
            [
                {"id": "parent", "contour_refs": [parent["id"]]},
                {"id": "child", "contour_refs": [child["id"]]},
            ],
        )

        record = result["records"][0]
        self.assertEqual(record["state"], "accepted")
        self.assertEqual(record["selected"]["path_signature_residual_mm"], 0)
        self.assertEqual(record["selected"]["signed_transform"]["state"], "resolved")
        self.assertTrue(shared["scopes"][0]["path_reprojection_eligible"])

    def test_symmetric_path_keeps_mirrored_sign_unresolved(self):
        parent = contour("parent.contour", [0, 100, 100])
        child = contour("child.contour", [200, 300, 300])
        shared = coordinates()

        result = solve_contour_correspondence(
            [frame("parent"), frame("child")],
            shared,
            [parent, child],
            [
                {"id": "parent", "contour_refs": [parent["id"]]},
                {"id": "child", "contour_refs": [child["id"]]},
            ],
        )

        self.assertEqual(result["records"][0]["state"], "accepted")
        self.assertEqual(result["records"][0]["selected"]["signed_transform"]["state"], "unresolved")
        self.assertFalse(shared["scopes"][0]["path_reprojection_eligible"])

    def test_multiple_equivalent_contour_pairs_abstain(self):
        parent_a = contour("parent.a", [0, 20, 100])
        parent_b = contour("parent.b", [0, 20, 100])
        child = contour("child.a", [200, 220, 300])
        shared = coordinates()

        result = solve_contour_correspondence(
            [frame("parent"), frame("child")],
            shared,
            [parent_a, parent_b, child],
            [
                {"id": "parent", "contour_refs": [parent_a["id"], parent_b["id"]]},
                {"id": "child", "contour_refs": [child["id"]]},
            ],
        )

        self.assertEqual(result["records"][0]["state"], "candidate")
        self.assertEqual(result["records"][0]["passing_candidate_count"], 2)

    def test_cut_intersects_native_parent_edges_before_matching_section_path(self):
        child = contour("child.contour", [0, 100, 100])
        shared = coordinates()
        shared["constraints"][0]["trace"] = {
            "orientation": "horizontal",
            "bbox_display": [10, 50, 150, 50],
            "primitive_refs": ["drawing.cut"],
        }
        native_segments = [
            {
                "id": f"drawing.edge.{index}",
                "drawing_ref": f"drawing[{index}]",
                "primitive_ref": f"drawing[{index}].item[0]",
                "start_display": [x, 0],
                "end_display": [x, 200],
                "sample_points_display": [],
                "axis": "vertical",
                "bbox_display": [x, 0, x, 200],
                "style": {"stroke": [0, 0, 0], "width": 1, "dash": ""},
            }
            for index, x in enumerate((20, 120))
        ]

        result = solve_contour_correspondence(
            [frame("parent"), frame("child")],
            shared,
            [child],
            [
                {"id": "parent", "bbox_display": [0, 0, 200, 200], "contour_refs": []},
                {"id": "child", "bbox_display": [220, 0, 420, 200], "contour_refs": [child["id"]]},
            ],
            native_segments=native_segments,
        )

        selected = result["records"][0]["selected"]
        self.assertEqual(result["records"][0]["state"], "accepted")
        self.assertEqual(selected["parent_signature"]["basis"], "style-compatible native edge pair intersected by accepted cutting plane")
        self.assertEqual(selected["span_residual_mm"], 0)
        self.assertEqual(len(selected["parent_geometry_refs"]), 2)

    def test_external_vertical_marker_uses_unique_native_silhouette_and_hydrates_composite(self):
        parent_segments = [
            {
                "id": f"parent.segment.{index}",
                "kind": "line",
                "start_display": [0, value],
                "end_display": [100, value],
                "primitive_ref": f"drawing.parent.{index}",
            }
            for index, value in enumerate((0, 50, 100))
        ]
        child_segments = [
            {
                "id": f"child.segment.{index}",
                "kind": "line",
                "start_display": [200, value],
                "end_display": [220, value],
                "primitive_ref": f"drawing.child.{index}",
            }
            for index, value in enumerate((200, 250, 300))
        ]
        parent = {
            "id": "parent.contour",
            "closed": True,
            "bbox_display": [0, 0, 100, 100],
            "segments_display": parent_segments,
            "topology": {"segment_count": 3, "cycle_rank": 1},
        }
        child = {
            "id": "child.composite",
            "kind": "composite_closed_loop",
            "closed": True,
            "bbox_display": [200, 200, 220, 300],
            "segment_refs": [item["id"] for item in child_segments],
            "topology": {"segment_count": 3, "cycle_rank": 1},
        }
        shared = coordinates()
        shared["constraints"][0]["mode"] = "cut_vertical"
        shared["constraints"][0]["trace"] = {
            "orientation": "vertical",
            "bbox_display": [150, -10, 150, 110],
            "primitive_refs": ["drawing.cut"],
        }
        parent_frame = frame("parent")
        parent_frame["scale"]["value_points_per_mm"] = 0.5

        result = solve_contour_correspondence(
            [parent_frame, frame("child")],
            shared,
            [parent, child],
            [
                {"id": "parent", "bbox_display": [0, 0, 100, 100], "contour_refs": [parent["id"]]},
                {"id": "child", "bbox_display": [200, 200, 220, 300], "contour_refs": [child["id"]]},
            ],
            native_segments=child_segments,
        )

        selected = result["records"][0]["selected"]
        self.assertEqual(result["records"][0]["state"], "accepted")
        self.assertEqual(selected["projection_mode"], "orthogonal_silhouette")
        self.assertEqual(selected["scale_transfer_certificate"]["parent_scale_points_per_mm"], 1.0)
        self.assertEqual(shared["scopes"][0]["reprojection_validations"][0]["status"], "pass")

    def test_module_has_no_drawing_or_object_class_dispatch(self):
        source = inspect.getsource(__import__("src.drawing_engine.core.contour_correspondence", fromlist=["*"])).lower()
        for forbidden in (".pdf", "k1", "k7", "3179", "column", "beam", "stair", "slab"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
