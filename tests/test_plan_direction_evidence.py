import copy
import unittest

import fitz
import numpy as np
import trimesh

from src.drawing_engine.disciplines.concrete.plan_direction_evidence import derive_plan_direction_evidence
from src.drawing_engine.core.vector_topology import extract_page_topology
from src.drawing_engine.disciplines.concrete.multi_component_solid_kernel import _directed_surface_path_records
from src.drawing_engine.disciplines.concrete.invariant_union_equivalence import _projection_pair_check


def native_fixture(*, scale=1., reflected=False, missing=None, branch=False, conflicting=False):
    doc = fitz.open()
    page = doc.new_page(width=1800, height=1600)

    def point(x, y):
        return fitz.Point(50 + scale * x, 50 + scale * (220 - y if reflected else y))

    def line(a, b):
        page.draw_line(point(*a), point(*b), width=.6 * scale)

    if missing != "circle":
        page.draw_circle(point(0, 50), 5 * scale, width=.6 * scale)
    for a, b in [((0, 50), (450, 50)), ((450, 50), (450, 170)), ((450, 170), (0, 170))]:
        if missing != "shaft" or a != (450, 50):
            line(a, b)
    if missing != "arrow":
        line((0, 170), (20, 155))
        line((0, 170), (20, 185))
        # Duplicate CAD projection of the same wing merges provenance only.
        line((0, 170), (20, 185))
    if branch:
        line((450, 50), (490, 50))
    if conflicting:
        page.draw_circle(point(0, 170), 5 * scale, width=.6 * scale)
        line((0, 50), (20, 35))
        line((0, 50), (20, 65))
    if missing != "treads":
        for x in range(0, 401, 50):
            for low, high in [(0, 100), (120, 220)]:
                line((x, low), (x, high))
    topology = extract_page_topology(page, vertex_tolerance=.5 * scale)
    doc.close()
    scope = {"id": "scope.plan", "state": "resolved", "primitive_refs": list(topology["drawing_styles"])}
    folded = {
        "id": "folded.native", "title_scope_ref": scope["id"],
        "plan_to_relative_xy": {"origin_display": [450 * scale + 50 - 50 * scale, 50],
                                "u_axis_display": [1., 0.], "v_axis_display": [0., 1.],
                                "scale_points_per_mm": scale},
        "polygon_uv_mm": [[-400, 0], [100, 0], [100, 220], [-400, 220],
                          [-400, 120], [0, 120], [0, 100], [-400, 100]],
    }
    bands = [{"id": "band.a", "interval_mm": [0, 100]}, {"id": "band.b", "interval_mm": [120, 220]}]
    return topology, {"segments": [scope]}, folded, bands


class PlanDirectionEvidenceTest(unittest.TestCase):
    def test_native_symbol_binds_order_not_page_band_index(self):
        evidence = derive_plan_direction_evidence(*native_fixture())
        self.assertEqual(evidence["state"], "accepted")
        path = evidence["paths"][0]
        self.assertEqual([r["band_ref"] for r in path["band_runs"]], ["band.a", "band.b"])
        self.assertEqual([r["longitudinal_ascent_sign"] for r in path["band_runs"]], [1, -1])
        self.assertEqual(len(path["arrowhead_edge_refs"]), 3)
        self.assertEqual(path["interpretation_state"], "convention_dependent")
        self.assertFalse(evidence["quantity_eligible"])
        reflected = derive_plan_direction_evidence(*native_fixture(reflected=True))
        self.assertEqual(reflected["state"], "accepted")
        self.assertEqual([r["band_ref"] for r in reflected["paths"][0]["band_runs"]], ["band.b", "band.a"])

    def test_order_and_scale_invariance(self):
        inputs = native_fixture()
        first = derive_plan_direction_evidence(*inputs)
        inputs[0]["segments"].reverse()
        inputs[3].reverse()
        self.assertEqual(first, derive_plan_direction_evidence(*inputs))
        enlarged = derive_plan_direction_evidence(*native_fixture(scale=2.))
        self.assertEqual(enlarged["state"], "accepted")
        np.testing.assert_allclose([s["uv_mm"] for s in first["paths"][0]["samples"]],
                                   [s["uv_mm"] for s in enlarged["paths"][0]["samples"]], atol=.001)

    def test_missing_evidence_or_branched_path_cannot_bind(self):
        for missing in ("circle", "arrow", "shaft", "treads"):
            with self.subTest(missing=missing):
                result = derive_plan_direction_evidence(*native_fixture(missing=missing))
                self.assertEqual(result["state"], "unknown")
                self.assertFalse(result["paths"])
        self.assertEqual(derive_plan_direction_evidence(*native_fixture(branch=True))["state"], "unknown")

    def test_excluded_circle_and_filled_circle_are_not_accepted(self):
        inputs = native_fixture()
        circle_ref = inputs[0]["segments"][0]["drawing_ref"]
        inputs[1]["segments"][0]["primitive_refs"].remove(circle_ref)
        self.assertEqual(derive_plan_direction_evidence(*inputs)["state"], "unknown")
        inputs = native_fixture()
        for s in inputs[0]["segments"]:
            if s["kind"] == "cubic":
                s["style"]["fill"] = [0, 0, 0]
        self.assertEqual(derive_plan_direction_evidence(*inputs)["state"], "unknown")

    def test_mismatched_native_styles_cannot_join(self):
        inputs = native_fixture()
        for s in inputs[0]["segments"]:
            if s["kind"] == "cubic":
                s["style"]["width"] *= 2
        self.assertEqual(derive_plan_direction_evidence(*inputs)["state"], "unknown")

    def test_conflicting_terminal_symbols_do_not_select_a_direction(self):
        result = derive_plan_direction_evidence(*native_fixture(conflicting=True))
        self.assertEqual(result["state"], "unknown")
        self.assertFalse(result["paths"])


def surface_fixture():
    meshes = []
    for y, sign in ((50, 1), (170, -1)):
        mesh = trimesh.creation.box(extents=[400, 100, 10])
        mesh.apply_translation([-200, y, 0])
        mesh.vertices[:, 2] += sign * mesh.vertices[:, 0]
        meshes.append(mesh)
    evidence = derive_plan_direction_evidence(*native_fixture())
    path = evidence["paths"][0]
    # Restrict this unit fixture to the independently supplied band stations;
    # full connected landing replay is exercised by the live-sheet test.
    path["samples"] = [s for s in path["samples"] if s["band_ref"]]
    for run in path["band_runs"]:
        run["sample_indices"] = [i for i, s in enumerate(path["samples"]) if s["band_ref"] == run["band_ref"]]
    return trimesh.util.concatenate(meshes), {"required_direction_evidence": evidence}


class DirectedSurfaceReplayTest(unittest.TestCase):
    def test_swapped_bands_fail_even_with_identical_volume_and_footprint(self):
        mesh, view = surface_fixture()
        records, errors = _directed_surface_path_records(view, mesh, np.zeros(3), np.array([1, 0, 0]), np.array([0, 1, 0]))
        self.assertFalse(errors)
        self.assertEqual(records[0]["status"], "pass")
        reflected = mesh.copy()
        reflected.vertices[:, 1] = 220 - reflected.vertices[:, 1]
        self.assertAlmostEqual(abs(mesh.volume), abs(reflected.volume))
        records, errors = _directed_surface_path_records(view, reflected, np.zeros(3), np.array([1, 0, 0]), np.array([0, 1, 0]))
        self.assertTrue(errors)
        self.assertEqual([r["status"] for r in records[0]["band_runs"]], ["fail", "fail"])
        self.assertEqual(len(records[0]["samples"]), 16)
        self.assertEqual(len(records[0]["adjacent_residuals"]), 15)

    def test_generated_direction_evidence_cannot_validate(self):
        mesh, view = surface_fixture()
        view["required_direction_evidence"]["paths"][0]["independent_of_candidate_geometry"] = False
        _, errors = _directed_surface_path_records(view, mesh, np.zeros(3), np.array([1, 0, 0]), np.array([0, 1, 0]))
        self.assertTrue(any("not_independent" in e for e in errors))

    def test_uncovered_station_is_reported_without_hiding_other_residuals(self):
        mesh, view = surface_fixture()
        view["required_direction_evidence"]["paths"][0]["samples"][2]["uv_mm"] = [900, 900]
        records, errors = _directed_surface_path_records(view, mesh, np.zeros(3), np.array([1, 0, 0]), np.array([0, 1, 0]))
        self.assertTrue(errors)
        self.assertIsNone(records[0]["samples"][2]["surface_elevation_mm"])
        self.assertEqual(len(records[0]["adjacent_residuals"]), 15)

    def test_reflection_equivalence_must_preserve_directed_stations(self):
        mesh, view = surface_fixture()
        checks, _ = _directed_surface_path_records(view, mesh, np.zeros(3), np.array([1, 0, 0]), np.array([0, 1, 0]))
        view.update(projection_role="plan", u_axis_xyz=[1, 0, 0], v_axis_xyz=[0, 1, 0],
                    polygons_uv_mm=[[[-400, 0], [0, 0], [0, 220], [-400, 220]]])
        kernel = {"status": "pass", "directed_surface_path_reprojections": checks}
        self.assertEqual(_projection_pair_check(view, view, [1, 1, 1], kernel, kernel, .001)["status"], "pass")
        reflected = _projection_pair_check(view, view, [1, -1, 1], kernel, kernel, .001)
        self.assertEqual(reflected["status"], "fail")
        self.assertIn("directed_projection_binding_not_preserved", reflected["errors"])
        self.assertEqual(reflected["symmetric_difference_area_mm2"], 0)
        missing = copy.deepcopy(kernel)
        missing.pop("directed_surface_path_reprojections")
        self.assertEqual(_projection_pair_check(view, view, [1, 1, 1], kernel, missing, .001)["status"], "fail")


if __name__ == "__main__":
    unittest.main()
