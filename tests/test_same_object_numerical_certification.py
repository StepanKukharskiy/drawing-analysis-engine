import copy
import unittest

from shapely.geometry import Polygon

from src.drawing_engine.disciplines.concrete.multi_component_solid_kernel import (
    SolidKernelValidationError,
    _projection_residual,
    solve_same_object_constructive_solid,
)
from tests.stair_assembly_fixtures import same_object_stair_fixture


def _solve(fixture):
    return solve_same_object_constructive_solid(
        fixture["physical_object_scope"],
        fixture["construction_regions"],
        fixture["internal_seams"],
        fixture["analytic_union_volume"],
        fixture["supplied_views"],
        separate_object_interfaces=fixture["separate_object_interfaces"],
    )


class SameObjectNumericalCertificationTest(unittest.TestCase):
    def test_certified_plane_snaps_float_gap_but_not_millimetre_gap(self):
        near = same_object_stair_fixture()
        near["construction_regions"][0]["transform"]["origin_xyz_mm"][0] -= 1e-5
        accepted = _solve(near)

        seam = accepted["internal_seams"][0]
        certificate = seam["seam_plane_snap_certificate"]
        flight = max(
            certificate["construction_region_residuals"],
            key=lambda item: item["minimum_absolute_residual_mm"],
        )
        self.assertAlmostEqual(flight["minimum_absolute_residual_mm"], 1e-5, places=8)
        self.assertTrue(flight["snapped_vertex_indices"])
        self.assertLessEqual(
            certificate["canonical_plane"]["snap_tolerance_mm"],
            0.01,
        )

        far = same_object_stair_fixture()
        far["construction_regions"][0]["transform"]["origin_xyz_mm"][0] -= 1.0
        with self.assertRaises(SolidKernelValidationError) as caught:
            _solve(far)
        diagnostic = caught.exception.diagnostics["internal_seams"][0]
        flight = max(
            diagnostic["seam_plane_snap_certificate"][
                "construction_region_residuals"
            ],
            key=lambda item: item["minimum_absolute_residual_mm"],
        )
        self.assertAlmostEqual(flight["minimum_absolute_residual_mm"], 1.0)
        self.assertEqual(flight["snapped_vertex_indices"], [])
        self.assertEqual(diagnostic["status"], "fail")

    def test_sub_resolution_projection_sliver_is_removed_but_real_hole_remains(self):
        expected = Polygon([(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)])
        numerical = Polygon(
            [(0.0001, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0001, 100.0)]
        )
        residual = _projection_residual(expected, numerical, 1e-6)
        self.assertGreater(residual["raw_symmetric_difference_area_mm2"], 0.0)
        self.assertEqual(residual["normalized_symmetric_difference_area_mm2"], 0.0)
        self.assertGreater(residual["sub_resolution_sliver_count"], 0)
        self.assertEqual(residual["status"], "pass")

        real_hole = Polygon(
            expected.exterior.coords,
            holes=[[(49.5, 10.0), (50.5, 10.0), (50.5, 90.0), (49.5, 90.0)]],
        )
        residual = _projection_residual(expected, real_hole, 1e-6)
        self.assertGreater(residual["meaningful_component_count"], 0)
        self.assertEqual(residual["status"], "fail")

    def test_failure_reports_every_seam_and_projection_residual(self):
        fixture = same_object_stair_fixture()
        for region in fixture["construction_regions"][:2]:
            region["transform"]["origin_xyz_mm"][0] -= 1.0
        for view in fixture["supplied_views"]:
            view["polygons_uv_mm"] = copy.deepcopy(view["polygons_uv_mm"])
            view["polygons_uv_mm"][-1][-1][0] += 50.0

        with self.assertRaises(SolidKernelValidationError) as caught:
            _solve(fixture)

        diagnostics = caught.exception.diagnostics
        self.assertEqual(len(diagnostics["internal_seams"]), 2)
        self.assertEqual(len(diagnostics["supplied_view_reprojections"]), 2)
        self.assertTrue(
            all(
                item["seam_plane_snap_certificate"]["construction_region_residuals"]
                for item in diagnostics["internal_seams"]
            )
        )
        self.assertTrue(
            all(
                item["precision_aware_residual"]["component_residuals"]
                for item in diagnostics["supplied_view_reprojections"]
            )
        )
        self.assertGreaterEqual(len(caught.exception.errors), 4)


if __name__ == "__main__":
    unittest.main()
