import copy
import unittest

from src.drawing_engine.disciplines.concrete.generic_profile_extrusion_solver import _extruded_mesh
from src.drawing_engine.disciplines.concrete.multi_component_solid_kernel import solve_multi_component_solid


def component(identifier, width, depth, height, origin_x):
    mesh = _extruded_mesh([(0, 0), (width, 0), (width, height), (0, height)], depth)
    return {
        "id": identifier,
        "mesh": {
            "vertices_xyz_mm": mesh["vertices_xyz_mm"],
            "triangles": mesh["triangles"],
        },
        "transform": {
            "id": f"physical_component_transform.{identifier}",
            "record_type": "physical_component_transform",
            "record_version": "0.1.0",
            "component_ref": identifier,
            "state": "resolved",
            "placement_role": "physical",
            "origin_xyz_mm": [origin_x, 0, 0],
            "local_axes_xyz": {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]},
            "axis_signs": "resolved",
            "evidence_refs": [f"transform.{identifier}"],
        },
        "analytic_volume": {
            "value_mm3": width * depth * height,
            "basis": "synthetic_profile_area_times_depth",
            "evidence_refs": [f"analytic.{identifier}"],
        },
        "evidence_refs": [f"mesh.{identifier}"],
    }


def fixture():
    components = [
        component("component.a", 100, 50, 40, 0),
        component("component.b", 60, 50, 40, 100),
    ]
    interfaces = [
        {
            "id": "interface.a_b",
            "component_refs": ["component.a", "component.b"],
            "polygon_xyz_mm": [[100, 0, 0], [100, 50, 0], [100, 50, 40], [100, 0, 40]],
            "evidence_refs": ["interface.evidence.a_b"],
        }
    ]
    views = [
        {
            "id": "view.top",
            "origin_xyz_mm": [0, 0, 0],
            "u_axis_xyz": [1, 0, 0],
            "v_axis_xyz": [0, 1, 0],
            "tolerance_mm": 1e-6,
            "evidence_refs": ["view.evidence.top"],
            "component_projections": [
                {
                    "component_ref": "component.a",
                    "polygons_uv_mm": [[[0, 0], [100, 0], [100, 50], [0, 50]]],
                    "evidence_refs": ["projection.top.a"],
                },
                {
                    "component_ref": "component.b",
                    "polygons_uv_mm": [[[100, 0], [160, 0], [160, 50], [100, 50]]],
                    "evidence_refs": ["projection.top.b"],
                },
            ],
        },
        {
            "id": "view.front",
            "origin_xyz_mm": [0, 0, 0],
            "u_axis_xyz": [1, 0, 0],
            "v_axis_xyz": [0, 0, 1],
            "tolerance_mm": 1e-6,
            "evidence_refs": ["view.evidence.front"],
            "component_projections": [
                {
                    "component_ref": "component.a",
                    "polygons_uv_mm": [[[0, 0], [100, 0], [100, 40], [0, 40]]],
                    "evidence_refs": ["projection.front.a"],
                },
                {
                    "component_ref": "component.b",
                    "polygons_uv_mm": [[[100, 0], [160, 0], [160, 40], [100, 40]]],
                    "evidence_refs": ["projection.front.b"],
                },
            ],
        },
    ]
    return components, interfaces, views


class MultiComponentSolidKernelTest(unittest.TestCase):
    def test_single_component_closes_without_shared_interfaces(self):
        components, _, views = fixture()
        components = components[:1]
        for view in views:
            view["component_projections"] = view["component_projections"][:1]

        record = solve_multi_component_solid(components, [], views)

        self.assertEqual(record["status"], "accepted")
        self.assertEqual(record["assembly_mesh"]["validation"]["component_count"], 1)
        self.assertEqual(record["shared_interfaces"], [])
        self.assertEqual(record["pair_validations"], [])
        self.assertEqual(len(record["supplied_view_reprojections"]), 2)

    def test_positioned_components_close_with_interface_volume_and_reprojections(self):
        components, interfaces, views = fixture()

        record = solve_multi_component_solid(components, interfaces, views)

        self.assertEqual(record["status"], "accepted")
        self.assertEqual(record["assembly_mesh"]["validation"]["component_count"], 2)
        self.assertTrue(record["assembly_mesh"]["validation"]["separately_watertight"])
        self.assertEqual(record["assembly_mesh"]["validation"]["boundary_edge_count"], 0)
        self.assertEqual(record["pair_validations"][0]["overlap_volume_mm3"], 0)
        self.assertAlmostEqual(record["shared_interfaces"][0]["area_mm2"], 2000)
        self.assertAlmostEqual(record["volume_validation"]["analytic_component_sum_mm3"], 320000)
        self.assertEqual(len(record["supplied_view_reprojections"]), 2)
        self.assertTrue(record["contract"]["quantity_eligible"])
        self.assertFalse(record["contract"]["schedule_values_used"])

    def test_replay_is_deterministic_and_does_not_mutate_inputs(self):
        components, interfaces, views = fixture()
        original = copy.deepcopy((components, interfaces, views))

        first = solve_multi_component_solid(components, interfaces, views)
        second = solve_multi_component_solid(components, interfaces, views)

        self.assertEqual(first, second)
        self.assertEqual((components, interfaces, views), original)

    def test_volumetric_overlap_fails_closed(self):
        components, interfaces, views = fixture()
        components[1]["transform"]["origin_xyz_mm"][0] = 90
        interfaces.clear()

        with self.assertRaisesRegex(ValueError, "unintended volumetric overlap"):
            solve_multi_component_solid(components, interfaces, views)

    def test_unrecorded_finite_area_contact_fails_closed(self):
        components, _, views = fixture()

        with self.assertRaisesRegex(ValueError, "unrecorded shared interface"):
            solve_multi_component_solid(components, [], views)

    def test_presentation_only_transform_cannot_place_a_component(self):
        components, interfaces, views = fixture()
        components[1]["transform"]["placement_role"] = "presentation_only"

        with self.assertRaisesRegex(ValueError, "not resolved physical placement"):
            solve_multi_component_solid(components, interfaces, views)

    def test_analytic_volume_disagreement_fails_closed(self):
        components, interfaces, views = fixture()
        components[0]["analytic_volume"]["value_mm3"] += 100

        with self.assertRaisesRegex(ValueError, "analytic and mesh volumes disagree"):
            solve_multi_component_solid(components, interfaces, views)

    def test_supplied_view_reprojection_disagreement_fails_closed(self):
        components, interfaces, views = fixture()
        views[1]["component_projections"][1]["polygons_uv_mm"][0][1][0] = 155

        with self.assertRaisesRegex(ValueError, "supplied-view reprojection failed"):
            solve_multi_component_solid(components, interfaces, views)

    def test_parallel_duplicate_views_do_not_satisfy_reprojection_gate(self):
        components, interfaces, views = fixture()
        views[1]["u_axis_xyz"] = [1, 0, 0]
        views[1]["v_axis_xyz"] = [0, 1, 0]
        views[1]["component_projections"] = copy.deepcopy(views[0]["component_projections"])

        with self.assertRaisesRegex(ValueError, "two non-parallel"):
            solve_multi_component_solid(components, interfaces, views)


if __name__ == "__main__":
    unittest.main()
