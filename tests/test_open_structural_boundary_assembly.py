import unittest

from src.drawing_engine.disciplines.concrete.open_structural_boundary_assembly import assemble_open_structural_boundaries


def _segment(ref, drawing_ref, start, end, start_vertex, end_vertex, width=1.0):
    dx, dy = abs(end[0] - start[0]), abs(end[1] - start[1])
    axis = "horizontal" if dy < 0.1 else "vertical" if dx < 0.1 else "oblique"
    return {
        "id": ref,
        "primitive_ref": ref.split(".segment[", 1)[0],
        "drawing_ref": drawing_ref,
        "kind": "line",
        "start_display": list(start),
        "end_display": list(end),
        "start_vertex_id": start_vertex,
        "end_vertex_id": end_vertex,
        "axis": axis,
        "length_points": (dx * dx + dy * dy) ** 0.5,
        "style": {"stroke": [0.0, 0.0, 0.0], "width": width, "dash": "[] 0"},
    }


def _fixture():
    points = [(0, 30), (0, 25), (10, 25), (10, 20), (20, 20), (20, 15), (30, 15)]
    segments = []
    for index, (start, end) in enumerate(zip(points, points[1:])):
        segments.append(
            _segment(
                f"drawing[1].item[{index}].segment[0]",
                "drawing[1]",
                start,
                end,
                f"v{index}",
                f"v{index + 1}",
            )
        )
    segments.append(
        _segment(
            "drawing[2].item[0].segment[0]",
            "drawing[2]",
            (0, 35),
            (30, 20),
            "w0",
            "w1",
        )
    )
    segments.append(
        _segment(
            "drawing[3].item[0].segment[0]",
            "drawing[3]",
            (100, 0),
            (110, 0),
            "x0",
            "x1",
            width=0.5,
        )
    )
    topology = {"segments": segments, "vertex_tolerance_points": 0.25}
    segmentation = {
        "segments": [
            {
                "id": "section.scope",
                "state": "resolved",
                "primitive_refs": ["drawing[1]", "drawing[2]", "drawing[3]"],
                "excluded_primitive_refs": [],
            }
        ]
    }
    local = {
        "scope_results": [
            {
                "scope_ref": "section.scope",
                "status": "resolved_subset",
                "local_ownership": {
                    "attachments": [
                        {
                            "dimension_ref": "dimension.scale.1",
                            "value_mm": 1000.0,
                            "measured_endpoints": [
                                {"point_display": [0.0, 0.0]},
                                {"point_display": [100.0, 0.0]},
                            ],
                        },
                        {
                            "dimension_ref": "dimension.scale.2",
                            "value_mm": 500.0,
                            "measured_endpoints": [
                                {"point_display": [0.0, 0.0]},
                                {"point_display": [0.0, 50.0]},
                            ],
                        },
                    ]
                },
                "repeated_projected_measurement_certificates": [
                    {"orientation": "horizontal", "value_mm": 100.0}
                ],
            }
        ]
    }
    frames = {
        "shared_coordinate_system": {
            "scopes": [
                {
                    "contour_correspondences": [
                        {
                            "id": "correspondence.1",
                            "child_view_id": "section.scope",
                            "candidates": [
                                {
                                    "state": "pass",
                                    "child_geometry_refs": [segments[0]["id"]],
                                    "child_signature": {
                                        "style_signature": [[0.0, 0.0, 0.0], 1.0, "[]0"]
                                    },
                                }
                            ],
                        }
                    ]
                }
            ]
        }
    }
    return topology, segmentation, local, frames


class OpenStructuralBoundaryAssemblyTest(unittest.TestCase):
    def test_open_flight_proposal_keeps_interface_ports_and_never_bridges(self):
        topology, segmentation, local, frames = _fixture()

        result = assemble_open_structural_boundaries(
            topology,
            segmentation,
            local,
            frames,
            [],
            {"fragments": []},
            page_number=1,
        )

        self.assertEqual(result["status"], "incomplete_proposals")
        self.assertEqual(result["summary"]["native_structural_support_count"], 7)
        self.assertEqual(result["summary"]["eligible_edge_count"], 7)
        self.assertEqual(result["summary"]["interface_bounded_flight_proposal_count"], 1)
        self.assertEqual(result["summary"]["closed_profile_count"], 0)
        proposal = result["scope_results"][0]["flight_boundary_proposals"][0]
        self.assertEqual(proposal["state"], "incomplete")
        self.assertEqual(proposal["closure"]["invented_bridge_count"], 0)
        self.assertEqual(len(proposal["interface_ports"]), 4)
        self.assertFalse(result["contract"]["closure_bridges_invented"])

    def test_existing_style_evidence_filters_without_a_width_constant(self):
        topology, segmentation, local, frames = _fixture()

        result = assemble_open_structural_boundaries(
            topology,
            segmentation,
            local,
            frames,
            [],
            {"fragments": []},
            page_number=1,
        )

        scope = result["scope_results"][0]
        self.assertEqual(scope["scope_native_edge_count"], 8)
        self.assertEqual(scope["native_structural_support_count"], 7)
        self.assertTrue(result["contract"]["hard_coded_line_width_forbidden"])

    def test_interior_crossing_is_split_into_a_stable_vertex(self):
        topology, segmentation, local, frames = _fixture()
        topology["segments"] = [
            _segment("drawing[1].item[0].segment[0]", "drawing[1]", (0, 0), (20, 0), "a", "b"),
            _segment("drawing[2].item[0].segment[0]", "drawing[2]", (10, -10), (10, 10), "c", "d"),
        ]
        segmentation["segments"][0]["primitive_refs"] = ["drawing[1]", "drawing[2]"]
        frames["shared_coordinate_system"]["scopes"][0]["contour_correspondences"][0]["candidates"][0]["child_geometry_refs"] = [topology["segments"][0]["id"]]

        result = assemble_open_structural_boundaries(
            topology,
            segmentation,
            local,
            frames,
            [],
            {"fragments": []},
            page_number=1,
        )

        scope = result["scope_results"][0]
        self.assertEqual(scope["derived_intersection_vertex_count"], 1)
        self.assertEqual(scope["split_edge_count"], 4)

    def test_exact_native_support_paths_can_close_without_bridges(self):
        topology, segmentation, local, frames = _fixture()
        topology["segments"].extend(
            [
                _segment(
                    "drawing[4].item[0].segment[0]",
                    "drawing[4]",
                    (0, 30),
                    (0, 35),
                    "v0",
                    "w0",
                ),
                _segment(
                    "drawing[5].item[0].segment[0]",
                    "drawing[5]",
                    (30, 15),
                    (30, 20),
                    "v6",
                    "w1",
                ),
            ]
        )
        segmentation["segments"][0]["primitive_refs"].extend(["drawing[4]", "drawing[5]"])

        result = assemble_open_structural_boundaries(
            topology,
            segmentation,
            local,
            frames,
            [],
            {"fragments": []},
            page_number=1,
        )

        self.assertEqual(result["status"], "closed_profiles_available")
        self.assertEqual(result["summary"]["closed_profile_count"], 1)
        profile = result["scope_results"][0]["closed_profiles"][0]
        self.assertTrue(profile["closure"]["unique_completion"])
        self.assertEqual(profile["closure"]["invented_bridge_count"], 0)
        self.assertEqual(profile["interface_ports"], [])
        self.assertGreaterEqual(len(profile["ordered_boundary_display"]), 4)


if __name__ == "__main__":
    unittest.main()
