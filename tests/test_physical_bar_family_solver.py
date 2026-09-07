import inspect
import unittest

from src.drawing_engine.disciplines.detail.detail_fabrication_solver import solve_detail_fabrication_geometry
from src.drawing_engine.core.native_vector_detail_linking import (
    _classify_repeated_raster_dimensions,
    _looks_like_leading_crossed_seven,
)
from src.drawing_engine.disciplines.rebar.physical_bar_family_solver import (
    build_family_constrained_scene,
    resolve_physical_bar_families,
    summarize_group_quantities,
)


class PhysicalBarFamilySolverTest(unittest.TestCase):
    def test_validated_mesh_with_no_closed_families_reports_identity_boundary(self):
        scene = build_family_constrained_scene(
            {"physical_families": []},
            {"components": []},
            {"vertices_xyz_mm": [[0, 0, 0], [100, 100, 100]]},
        )

        self.assertEqual(scene["status"], "unresolved")
        self.assertEqual(scene["reason"], "no physical rebar families closed")
        self.assertEqual(scene["concrete_mesh_status"], "available")
        self.assertEqual(scene["path_count"], 0)

    def test_accepted_projection_marks_bootstrap_family_without_detail_table(self):
        views = [
            {"id": "view.e", "role_hypothesis": "reinforcement_view_candidate"},
            {"id": "view.s", "role_hypothesis": "section_view_candidate"},
        ]
        components = [
            {
                "id": "component.e",
                "fragment_ids": ["fragment.e"],
                "view_ids": ["view.e"],
                "mark_hypotheses": ["12/594"],
                "physical_path_state": "candidate_nonbranching_projection",
                "projection_dimensionality": {"value": "axis_projection_1d"},
                "projected_length": {"value_mm": 5940.0},
            },
            *[
                {
                    "id": f"component.s{index}",
                    "fragment_ids": [f"fragment.s{index}"],
                    "view_ids": ["view.s"],
                    "mark_hypotheses": ["12/594"],
                    "physical_path_state": "candidate_nonbranching_projection",
                    "projection_dimensionality": {"value": "axis_end_projection_0d"},
                    "projected_length": {"value_mm": None},
                }
                for index in range(1, 3)
            ],
        ]
        graph = {
            "fragments": [
                {"id": "fragment.e"},
                *[
                    {
                        "id": f"fragment.s{index}",
                        "primitive_ref": f"drawing[{index}]",
                        "mark_assignment_basis": "exclusive section mark plus first connected heavy-geometry leader contact",
                    }
                    for index in range(1, 3)
                ],
            ],
            "components": components,
            "mark_hypotheses": [
                {
                    "id": "path_mark_hypothesis.00001",
                    "state": "accepted",
                    "component_id": "component.e",
                }
            ],
        }
        details = {"status": "unresolved", "details": [], "placement_associations": []}

        validation = resolve_physical_bar_families(details, graph, views)

        self.assertEqual(validation["family_count"], 1)
        family = details["physical_families"][0]
        self.assertEqual(family["family_origin"], "evidence_backed_placed_projection_mark")
        self.assertEqual(family["count"], 2)
        self.assertEqual(family["constraint_status"], "resolved")
        self.assertFalse(family["schedule_values_used"])

    def test_multiple_callout_counts_do_not_assume_additive_or_duplicate(self):
        views = [{"id": "view.e", "role_hypothesis": "reinforcement_view_candidate"}]
        graph = {
            "fragments": [{"id": "fragment.1"}],
            "components": [
                {
                    "id": "component.1",
                    "fragment_ids": ["fragment.1"],
                    "view_ids": ["view.e"],
                    "mark_hypotheses": ["12/594"],
                    "projection_dimensionality": {"value": "axis_projection_1d"},
                    "physical_path_state": "candidate_nonbranching_projection",
                    "projected_length": {"value_mm": 5940.0},
                }
            ],
            "mark_hypotheses": [
                {
                    "id": f"mark.{index}",
                    "token": "12/594",
                    "state": "accepted",
                    "component_id": "component.1",
                    "semantic_value": {"diameter_mm": 12, "length_code": 594},
                    "count_spacing_observation": {"count": count, "spacing_mm": 100, "text_role_id": f"text.{index}"},
                }
                for index, count in enumerate((14, 3), start=1)
            ],
        }
        details = {"details": [], "placement_associations": []}

        resolve_physical_bar_families(details, graph, views)

        family = details["physical_families"][0]
        self.assertEqual(family["count_state"], "unknown")
        self.assertIsNone(family["count"])
        self.assertIn("contradictory count", family["multiplicity"]["reason"])

    def test_crossed_seven_outline_is_not_treated_as_one(self):
        import cv2
        import numpy as np

        image = np.full((100, 90), 255, dtype=np.uint8)
        cv2.line(image, (5, 8), (35, 8), 0, 5)
        cv2.line(image, (35, 8), (14, 82), 0, 5)
        cv2.line(image, (8, 45), (32, 38), 0, 4)
        cv2.line(image, (55, 8), (82, 8), 0, 5)
        cv2.line(image, (55, 8), (55, 42), 0, 5)
        cv2.line(image, (55, 42), (80, 42), 0, 5)
        cv2.line(image, (80, 42), (80, 82), 0, 5)
        cv2.line(image, (55, 82), (80, 82), 0, 5)
        self.assertTrue(_looks_like_leading_crossed_seven(image, "15"))

    def test_repeated_raster_glyph_consensus_repairs_low_confidence_digit(self):
        geometry = [{
            "source_type": "embedded_raster_detail",
            "curve_candidates": [{}, {}],
            "segments": [
                *[
                    {"kind": "line_candidate", "start_display": [0, y], "end_display": [20, y]}
                    for y in range(4)
                ],
                *[
                    {"kind": "line_candidate", "start_display": [x, 0], "end_display": [x, 20]}
                    for x in range(4)
                ],
                {"kind": "line_candidate", "start_display": [0, 0], "end_display": [10, 10]},
                {"kind": "line_candidate", "start_display": [10, 0], "end_display": [0, 10]},
            ],
        }]
        signature = "00" * 32
        details = []
        for index, (text, confidence) in enumerate((("75", .93), ("75", .92), ("75", .91), ("15", .51)), start=1):
            details.append({
                "id": f"detail.{index}",
                "bbox_display": [0, 0, 100, 100],
                "geometry_paths": geometry,
                "raster_fabrication_dimension_observations": [{
                    "text": text,
                    "value_mm": float(text),
                    "confidence": confidence,
                    "orientation": "vertical",
                    "bbox_display": [50, 50, 60, 60],
                    "glyph_signature": signature,
                }],
            })

        _classify_repeated_raster_dimensions(details)

        repaired = details[-1]["raster_fabrication_dimension_observations"][0]
        self.assertEqual((repaired["text"], repaired["value_mm"]), ("75", 75.0))
        self.assertEqual(repaired["semantic_role"], "hook_projection_dimension")
        self.assertEqual(repaired["glyph_consensus_support"], 3)

    def test_families_close_before_straight_bar_lift(self):
        views = [
            {"id": "view.e", "role_hypothesis": "reinforcement_view_candidate"},
            {"id": "view.s", "role_hypothesis": "section_view_candidate"},
        ]
        components = [
            {
                "id": "component.e",
                "view_ids": ["view.e"],
                "object_instance_ids": ["object.1"],
                "mark_hypotheses": ["4"],
                "physical_path_state": "candidate_nonbranching_projection",
                "projection_dimensionality": {"value": "axis_projection_1d", "normalized_center_in_view": [0.5, 0.5]},
                "projected_length": {"value_mm": 900.0},
            },
            *[
                {
                    "id": f"component.s{index}",
                    "view_ids": ["view.s"],
                    "object_instance_ids": ["object.1"],
                    "mark_hypotheses": ["4"],
                    "physical_path_state": "candidate_nonbranching_projection",
                    "projection_dimensionality": {"value": "axis_end_projection_0d", "normalized_center_in_view": center},
                    "projected_length": {"value_mm": None},
                }
                for index, center in enumerate(([0.2, 0.2], [0.8, 0.8]), start=1)
            ],
        ]
        graph = {"components": components}
        details = {
            "physical_families": [
                {"id": "family.1", "mark": "4", "component_ids": [item["id"] for item in components]}
            ]
        }
        validation = resolve_physical_bar_families(details, graph, views)
        self.assertEqual(validation["resolved_family_count"], 1)
        self.assertEqual(details["physical_families"][0]["count"], 2)
        mesh = {"vertices": [[0, 0, 0], [400, 400, 1000]]}
        scene = build_family_constrained_scene(details, graph, mesh)
        self.assertEqual(scene["status"], "drawing_constrained")
        self.assertEqual(scene["path_count"], 2)
        self.assertTrue(all(path["reprojection_validation"]["status"] == "pass" for path in scene["paths"]))

    def test_detail_length_sums_dimensioned_native_legs(self):
        detail = {
            "geometry_paths": [
                {
                    "primitive_ref": "drawing[1]",
                    "segments": [
                        {"kind": "line", "start_display": [0, 0], "end_display": [10, 0]},
                        {"kind": "line", "start_display": [10, 0], "end_display": [10, 20]},
                    ],
                }
            ],
            "fabrication_dimensions": [
                {"dimension_id": "d.h", "value_mm": 100, "orientation": "horizontal", "status": "accepted"},
                {"dimension_id": "d.v", "value_mm": 200, "orientation": "vertical", "status": "accepted"},
            ],
        }
        result = solve_detail_fabrication_geometry(detail)
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["length_mm"], 300.0)
        self.assertFalse(result["closure_validation"]["schedule_values_used"])

    @staticmethod
    def _raster_hook_detail(radius=None):
        detail = {
            "geometry_paths": [
                {
                    "source_type": "embedded_raster_detail",
                    "curve_candidates": [{}, {}],
                    "segments": [
                        *[
                            {"kind": "line_candidate", "start_display": [0, y], "end_display": [20, y]}
                            for y in range(4)
                        ],
                        *[
                            {"kind": "line_candidate", "start_display": [x, 0], "end_display": [x, 20]}
                            for x in range(4)
                        ],
                        {"kind": "line_candidate", "start_display": [0, 0], "end_display": [10, 10]},
                        {"kind": "line_candidate", "start_display": [10, 0], "end_display": [0, 10]},
                    ],
                }
            ],
            "raster_fabrication_dimension_observations": [
                {"id": "d.w", "value_mm": 265, "semantic_role": "overall_width_dimension"},
                {"id": "d.h", "value_mm": 330, "semantic_role": "overall_height_dimension"},
                {"id": "d.k", "value_mm": 75, "semantic_role": "hook_projection_dimension"},
            ],
        }
        if radius is not None:
            detail["centerline_bend_radius_mm"] = radius
        return detail

    def test_raster_hook_topology_closes_symbolically_without_radius(self):
        result = solve_detail_fabrication_geometry(self._raster_hook_detail())
        self.assertEqual(result["status"], "convention_dependent")
        self.assertEqual(result["topology"], "open_rectangular_loop_with_two_diagonal_hooks")
        self.assertAlmostEqual(result["sharp_vertex_length_mm"], 1402.132, places=3)
        self.assertIsNone(result["length_mm"])
        self.assertFalse(result["closure_validation"]["schedule_values_used"])

    def test_raster_hook_cutting_length_closes_with_centerline_radius(self):
        result = solve_detail_fabrication_geometry(self._raster_hook_detail(radius=8))
        self.assertEqual(result["status"], "resolved")
        self.assertGreater(result["length_mm"], 0)
        self.assertLess(result["length_mm"], result["sharp_vertex_length_mm"])
        self.assertTrue(result["closure_validation"]["centerline_radius_assigned"])

    def test_conflicting_section_counts_abstain(self):
        views = [{"id": "view.s", "role_hypothesis": "section_view_candidate", "bbox_display": [0, 0, 100, 100]}]
        graph = {"components": []}
        details = {"physical_families": [{"id": "family.1", "mark": "1", "component_ids": []}]}
        sections = {"sections": []}
        for section_index, count in enumerate((2, 4), start=1):
            sections["sections"].append(
                {
                    "section_id": f"section.{section_index}",
                    "candidates": [
                        {
                            "candidate_id": f"candidate.{section_index}.{item}",
                            "leader_marks": ["1"],
                            "kind": "filled_end_projection",
                            "state": "resolved",
                            "bbox_display": [10 * item, 10, 10 * item + 2, 12],
                        }
                        for item in range(1, count + 1)
                    ],
                }
            )
        resolve_physical_bar_families(details, graph, views, sections)
        self.assertEqual(details["physical_families"][0]["count_state"], "unknown")
        self.assertIsNone(details["physical_families"][0]["count"])

    def test_closed_two_leg_detail_does_not_count_each_leg_as_a_bar(self):
        views = [
            {
                "id": "view.s",
                "role_hypothesis": "section_view_candidate",
                "bbox_display": [0, 0, 100, 100],
            }
        ]
        graph = {"components": []}
        details = {
            "details": [
                {
                    "id": "detail.12",
                    "fabrication_geometry_solution": {
                        "topology": "open_rectangular_loop_with_two_diagonal_hooks",
                        "closure_validation": {"topology_closed": True},
                    },
                }
            ],
            "physical_families": [
                {
                    "id": "family.12",
                    "mark": "12",
                    "component_ids": [],
                    "detail_ids": ["detail.12"],
                }
            ],
        }
        sections = {
            "sections": [
                {
                    "section_id": "section.1",
                    "host_bbox_display": [0, 0, 100, 100],
                    "candidates": [
                        {
                            "candidate_id": f"candidate.{index}",
                            "leader_marks": ["12"],
                            "kind": "outlined_projection",
                            "bbox_display": bbox,
                        }
                        for index, bbox in enumerate(
                            (
                                [20, 10, 20, 80],
                                [22, 10, 22, 80],
                                [78, 10, 78, 80],
                                [80, 10, 80, 80],
                            ),
                            start=1,
                        )
                    ],
                }
            ]
        }

        resolve_physical_bar_families(details, graph, views, sections)

        multiplicity = details["physical_families"][0]["multiplicity"]
        self.assertEqual(multiplicity["projected_leg_observation_count"], 2)
        self.assertEqual(multiplicity["projected_legs_per_physical_bar"], 2)
        self.assertEqual(multiplicity["value"], 1)

    def test_disjoint_coordinate_scopes_cannot_form_one_family(self):
        views = [
            {"id": "view.e", "role_hypothesis": "reinforcement_view_candidate"},
            {"id": "view.s", "role_hypothesis": "section_view_candidate"},
        ]
        graph = {
            "components": [
                {
                    "id": "component.e", "view_ids": ["view.e"], "coordinate_scope_ids": ["scope.a"],
                    "mark_hypotheses": ["4"], "physical_path_state": "candidate_nonbranching_projection",
                    "projection_dimensionality": {"value": "axis_projection_1d"}, "projected_length": {"value_mm": 900},
                },
                {
                    "id": "component.s", "view_ids": ["view.s"], "coordinate_scope_ids": ["scope.b"],
                    "mark_hypotheses": ["4"], "physical_path_state": "candidate_nonbranching_projection",
                    "projection_dimensionality": {"value": "axis_end_projection_0d", "normalized_center_in_view": [0.5, 0.5]},
                },
            ]
        }
        details = {"physical_families": [{"id": "family.1", "mark": "4", "component_ids": ["component.e", "component.s"]}]}

        resolve_physical_bar_families(details, graph, views)

        self.assertEqual(details["physical_families"][0]["constraint_status"], "unresolved")

    def test_reuses_closed_symbolic_group_certificate(self):
        views = [{"id": "view.e", "role_hypothesis": "reinforcement_view_candidate"}]
        graph = {
            "components": [
                {
                    "id": "component.e",
                    "view_ids": ["view.e"],
                    "mark_hypotheses": [],
                    "projection_dimensionality": {"value": "axis_projection_1d"},
                    "physical_path_state": "candidate_nonbranching_projection",
                }
            ],
            "cross_view_projection_identities": [],
        }
        groups = [
            {
                "id": "group.1",
                "identity": {"mark": {"value": "1", "state": "derived", "evidence_refs": ["mark.1"]}},
                "quantity": {"value": 4, "state": "derived"},
                "topology": {"family": {"value": "straight"}, "parameters": {}},
                "placement": {
                    "path_component_ids": ["component.e"],
                    "metric_solution": {
                        "status": "pass",
                        "solved_centerline": {"z_start_mm": 50.0, "z_end_mm": 950.0},
                        "section_validation": {"status": "pass"},
                        "reprojection": {"supporting_primitive_count": 4},
                        "validation": {"status": "pass"},
                    },
                },
                "distribution": {
                    "positions_xy_mm": [[-40, -40], [40, -40], [40, 40], [-40, 40]],
                    "expansion_validation": {"status": "pass"},
                },
                "fabrication": {
                    "status": "resolved",
                    "cutting_length_each_mm": 900.0,
                    "cutting_length_total_mm": 3600.0,
                },
                "bar_spec": {
                    "diameter_mm": {"value": 16, "state": "convention_dependent"},
                    "steel_grade": {"value": None, "state": "unknown"},
                },
            }
        ]
        details = {"physical_families": [{"id": "family.1", "mark": "1", "component_ids": []}]}

        validation = resolve_physical_bar_families(details, graph, views, groups=groups)

        family = details["physical_families"][0]
        self.assertEqual(validation["resolved_family_count"], 1)
        self.assertEqual(family["source_group_id"], "group.1")
        self.assertEqual(family["count"], 4)
        self.assertEqual(family["scene_status"], "projection_pair_not_materialised")
        takeoff = summarize_group_quantities(groups, details)
        self.assertEqual(takeoff["resolved_fabrication_length_m"], 3.6)
        self.assertEqual(takeoff["physical_bar_count"], 4)
        self.assertIsNone(takeoff["mass_kg"])
        self.assertFalse(takeoff["validation"]["mass_inputs_closed"])
        scene = build_family_constrained_scene(details, graph, {"vertices": [[-100, -100, 0], [100, 100, 1000]]}, groups)
        self.assertEqual(scene["path_count"], 4)
        self.assertTrue(all(path["diameter_mm"] is None for path in scene["paths"]))

    def test_modules_are_drawing_neutral(self):
        import src.drawing_engine.disciplines.rebar.physical_bar_family_solver as family_solver
        import src.drawing_engine.disciplines.detail.detail_fabrication_solver as fabrication_solver
        for module in (family_solver, fabrication_solver):
            source = inspect.getsource(module).lower()
            for forbidden in (".pdf", "column", "beam", "stair", "slab"):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
