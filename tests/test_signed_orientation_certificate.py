import copy
import inspect
import unittest

from src.drawing_engine.disciplines.concrete.signed_orientation_certificate import certify_signed_orientation


def segment(segment_id, start, end, axis, drawing_ref, *, style=None):
    return {
        "id": segment_id,
        "drawing_ref": drawing_ref,
        "primitive_ref": f"{drawing_ref}.item[0]",
        "start_display": list(start),
        "end_display": list(end),
        "axis": axis,
        "length_points": ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5,
        "style": style or {"stroke": [0, 0, 0], "width": 1, "dash": ""},
    }


def arrow_marker(prefix, x, direction=1):
    tip_y = 10 * direction
    tail_y = 5 * direction
    stem = segment(f"{prefix}.stem", [x, -10], [x, 10], "vertical", f"{prefix}.stem.drawing")
    left = segment(f"{prefix}.arrow.left", [x, tip_y], [x - 5, tail_y], "oblique", f"{prefix}.arrow.drawing")
    right = segment(f"{prefix}.arrow.right", [x, tip_y], [x + 5, tail_y], "oblique", f"{prefix}.arrow.drawing")
    return [stem, left, right]


def shared_coordinates(*, parent=(0, 20, 100), child=(200, 220, 300)):
    return {
        "constraints": [
            {
                "id": "constraint.cut.1",
                "mode": "cut_horizontal",
                "parent_view_id": "parent",
                "child_view_id": "child",
                "source_relation_id": "cut.1",
                "trace": {
                    "orientation": "horizontal",
                    "bbox_display": [0, 0, 100, 0],
                    "primitive_refs": ["left.stem.drawing", "right.stem.drawing"],
                },
            }
        ],
        "scopes": [
            {
                "id": "scope.1",
                "state": "resolved_relative",
                "gauge": {"state": "relative", "absolute_axis_sign_state": "unresolved"},
                "constraint_ids": ["constraint.cut.1"],
                "cut_plane_constraints": [
                    {
                        "relation_id": "cut.1",
                        "coordinate_candidates_mm": [-20.0, 20.0],
                        "signed_coordinate_in_parent_gauge_mm": -20.0,
                        "viewing_sign_state": "unresolved",
                    }
                ],
                "contour_correspondences": [
                    {
                        "id": "correspondence.1",
                        "constraint_id": "constraint.cut.1",
                        "parent_view_id": "parent",
                        "child_view_id": "child",
                        "parent_display_orientation": "horizontal",
                        "child_display_orientation": "horizontal",
                        "state": "accepted",
                        "selected": {
                            "parent_signature": {
                                "coordinates_mm": list(parent),
                                "basis": "native contour junctions at accepted cutting-plane intersection",
                            },
                            "child_signature": {
                                "coordinates_mm": list(child),
                                "basis": "native contour junctions",
                            },
                            "tolerance_mm": 2.0,
                            "parent_geometry_refs": ["edge.parent.1", "edge.parent.2"],
                            "child_geometry_refs": ["edge.child.1", "edge.child.2"],
                            "owned_dimension_refs": [],
                            "signed_transform": {
                                "state": "unresolved",
                                "child_coordinate_to_parent": {"sign": None, "offset_mm": None},
                                "candidates": [
                                    {"sign": 1, "offset_mm": -200.0},
                                    {"sign": -1, "offset_mm": 300.0},
                                ],
                            },
                        },
                    }
                ],
                "signed_contour_transform_count": 0,
                "path_reprojection_eligible": False,
            }
        ],
        "summary": {},
    }


def frames():
    return [
        {"view_id": "parent", "scale": {"state": "resolved", "value_points_per_mm": 1.0}},
        {"view_id": "child", "scale": {"state": "resolved", "value_points_per_mm": 1.0}},
    ]


class SignedOrientationCertificateTest(unittest.TestCase):
    def test_native_arrowheads_and_asymmetric_landmarks_publish_one_scope_transform(self):
        shared = shared_coordinates()
        native = [*arrow_marker("left", 0), *arrow_marker("right", 100)]

        result = certify_signed_orientation(frames(), shared, native_segments=native)

        self.assertEqual(result["summary"]["accepted_count"], 1)
        certificate = result["records"][0]
        self.assertEqual(certificate["status"], "accepted")
        self.assertEqual(certificate["selected_orientation"]["cut_plane_coordinates_mm"], [-20.0])
        scope = shared["scopes"][0]
        transform = scope["contour_correspondences"][0]["selected"]["signed_transform"]
        self.assertEqual(transform["state"], "resolved")
        self.assertEqual(transform["child_coordinate_to_parent"], {"sign": 1, "offset_mm": -200.0})
        self.assertEqual(scope["cut_plane_constraints"][0]["coordinate_candidates_mm"], [-20.0])
        self.assertEqual(scope["gauge"]["relative_axis_sign_state"], "resolved")
        self.assertEqual(scope["gauge"]["absolute_axis_sign_state"], "unresolved")

    def test_parallel_terminal_flags_are_not_arrowheads(self):
        shared = shared_coordinates()
        native = [
            segment("left.stem", [0, -10], [0, 10], "vertical", "left.stem.drawing"),
            segment("right.stem", [100, -10], [100, 10], "vertical", "right.stem.drawing"),
            segment("left.flag.1", [0, -5], [12, -5], "horizontal", "left.flag.drawing"),
            segment("left.flag.2", [0, 0], [8, 0], "horizontal", "left.flag.drawing"),
            segment("right.flag.1", [100, -5], [88, -5], "horizontal", "right.flag.drawing"),
            segment("right.flag.2", [100, 0], [92, 0], "horizontal", "right.flag.drawing"),
        ]

        result = certify_signed_orientation(frames(), shared, native_segments=native)

        gate = result["records"][0]["constraint_certificates"][0]["oriented_cutting_plane"]
        self.assertEqual(gate["status"], "insufficient_constraints")
        self.assertEqual(gate["reason"], "missing_unique_native_arrowhead")
        self.assertEqual(shared["scopes"][0]["cut_plane_constraints"][0]["coordinate_candidates_mm"], [-20.0, 20.0])

    def test_bidirectional_endpoint_arrowheads_abstain(self):
        shared = shared_coordinates()
        native = [*arrow_marker("left", 0, 1), *arrow_marker("right", 100, -1)]

        result = certify_signed_orientation(frames(), shared, native_segments=native)

        gate = result["records"][0]["constraint_certificates"][0]["oriented_cutting_plane"]
        self.assertEqual(gate["status"], "insufficient_constraints")
        self.assertEqual(gate["reason"], "bidirectional_or_inconsistent_native_markers")

    def test_two_landmark_passes_remain_ambiguous(self):
        shared = shared_coordinates(parent=(0, 100), child=(200, 300))
        native = [*arrow_marker("left", 0), *arrow_marker("right", 100)]

        result = certify_signed_orientation(frames(), shared, native_segments=native)

        gate = result["records"][0]["constraint_certificates"][0]["signed_shared_axis"]
        self.assertEqual(gate["status"], "ambiguous")
        self.assertEqual(sum(item["status"] == "pass" for item in gate["candidate_evaluations"]), 2)
        self.assertEqual(result["records"][0]["status"], "insufficient_constraints")

    def test_zero_landmark_passes_are_a_contradiction(self):
        shared = shared_coordinates()
        selected = shared["scopes"][0]["contour_correspondences"][0]["selected"]
        selected["signed_transform"]["candidates"] = [
            {"sign": 1, "offset_mm": 0.0},
            {"sign": -1, "offset_mm": 0.0},
        ]
        native = [*arrow_marker("left", 0), *arrow_marker("right", 100)]

        result = certify_signed_orientation(frames(), shared, native_segments=native)

        gate = result["records"][0]["constraint_certificates"][0]["signed_shared_axis"]
        self.assertEqual(gate["status"], "contradiction")
        self.assertEqual(result["records"][0]["status"], "contradiction")

    def test_reordering_native_segments_is_deterministic(self):
        shared = shared_coordinates()
        native = [*arrow_marker("left", 0), *arrow_marker("right", 100)]
        first = certify_signed_orientation(frames(), shared, native_segments=native)
        second_shared = shared_coordinates()
        second = certify_signed_orientation(frames(), second_shared, native_segments=list(reversed(copy.deepcopy(native))))
        self.assertEqual(first, second)

    def test_module_has_no_page_title_or_component_sign_input(self):
        source = inspect.getsource(__import__("src.drawing_engine.disciplines.concrete.signed_orientation_certificate", fromlist=["*"])).lower()
        self.assertNotIn("title_position", source)
        self.assertNotIn("page_position", source)
        self.assertNotIn("component_placement", inspect.signature(certify_signed_orientation).parameters)


if __name__ == "__main__":
    unittest.main()
