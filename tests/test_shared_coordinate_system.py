import inspect
import unittest

from src.drawing_engine.core.shared_coordinate_system import solve_shared_coordinate_system


def frame(view_id, bbox, horizontal=None, vertical=None):
    spans = []
    for orientation, value in (("horizontal", horizontal), ("vertical", vertical)):
        if value is not None:
            spans.append(
                {
                    "id": f"dimension.{view_id}.{orientation}",
                    "status": "accepted",
                    "value_mm": value,
                    "orientation": orientation,
                    "primitive_refs": [f"drawing.{view_id}.{orientation}"],
                }
            )
    return {
        "id": f"frame.{view_id}",
        "view_id": view_id,
        "view_role_hypothesis": "section_view_candidate" if "section" in view_id else "drawing_view_candidate",
        "bbox_display": bbox,
        "scale": {"state": "resolved", "value_points_per_mm": 0.25},
        "origin": {"state": "unresolved", "object_origin_display": None},
        "metric_spans": spans,
        "axes": {"object_axis_mapping": {"state": "unresolved"}},
        "projection_direction": {"state": "unresolved"},
        "unresolved_fields": ["axes.object_axis_mapping", "projection_direction.vector_object_xyz"],
    }


def cut(relation_id, parent, section, orientation, coordinate):
    if orientation == "horizontal":
        box = [10, coordinate, 190, coordinate]
    else:
        box = [coordinate, 10, coordinate, 190]
    return {
        "id": relation_id,
        "type": "cut_at",
        "state": "accepted",
        "parent_view_id": parent,
        "section_view_id": section,
        "trace": {"orientation": orientation, "bbox_display": box, "primitive_refs": [f"drawing.{relation_id}"]},
    }


class SharedCoordinateSystemTest(unittest.TestCase):
    def test_horizontal_cut_resolves_relative_axes_and_metric_reprojection(self):
        frames = [
            frame("parent", [0, 0, 200, 200], horizontal=400, vertical=1000),
            frame("section", [250, 0, 350, 100], horizontal=400, vertical=300),
        ]
        result = solve_shared_coordinate_system(
            frames,
            [cut("cut.1", "parent", "section", "horizontal", 80)],
            [{"object_scope_id": "scope.1", "object_instance_id": None, "view_ids": ["parent", "section"]}],
        )

        scope = result["scopes"][0]
        mappings = {item["view_id"]: item for item in scope["view_axis_mappings"]}
        self.assertEqual(scope["state"], "resolved_relative")
        self.assertEqual((mappings["parent"]["u"], mappings["parent"]["v"], mappings["parent"]["normal"]), ("X", "Z", "Y"))
        self.assertEqual((mappings["section"]["u"], mappings["section"]["v"], mappings["section"]["normal"]), ("X", "Y", "Z"))
        self.assertEqual(scope["reprojection_validations"][0]["status"], "pass")
        self.assertEqual(scope["cut_plane_constraints"][0]["object_axis"], "Z")
        self.assertFalse(scope["quantity_aggregation_eligible"])
        self.assertEqual(frames[1]["projection_direction"]["candidates"], ["+Z", "-Z"])

    def test_side_projection_pair_preserves_shared_vertical_axis(self):
        frames = [
            frame("primary", [0, 0, 200, 200], horizontal=800, vertical=1200),
            frame("compact", [220, 0, 270, 200], horizontal=300, vertical=1200),
        ]
        assembly = {
            "instances": [
                {
                    "id": "object.1",
                    "primary_view_id": "primary",
                    "section_view_id": "compact",
                    "view_ids": ["primary", "compact"],
                    "layout_mode": "right",
                    "basis": ["native orthogonal layout"],
                }
            ]
        }
        result = solve_shared_coordinate_system(frames, [], [], assembly)

        scope = result["scopes"][0]
        mappings = {item["view_id"]: item for item in scope["view_axis_mappings"]}
        self.assertEqual((mappings["compact"]["u"], mappings["compact"]["v"], mappings["compact"]["normal"]), ("Y", "Z", "X"))
        self.assertEqual(scope["reprojection_validations"][0]["status"], "pass")
        self.assertTrue(scope["bar_identity_eligible"])

    def test_conflicting_cycle_abstains(self):
        frames = [
            frame("parent", [0, 0, 200, 200], horizontal=400, vertical=400),
            frame("section", [250, 0, 350, 100], horizontal=400, vertical=400),
        ]
        result = solve_shared_coordinate_system(
            frames,
            [
                cut("cut.horizontal", "parent", "section", "horizontal", 80),
                cut("cut.vertical", "parent", "section", "vertical", 90),
            ],
            [{"object_scope_id": "scope.1", "object_instance_id": None, "view_ids": ["parent", "section"]}],
        )

        self.assertEqual(result["scopes"][0]["state"], "ambiguous")
        self.assertTrue(result["scopes"][0]["conflicts"])
        self.assertFalse(result["scopes"][0]["bar_identity_eligible"])

    def test_multiple_unowned_spans_do_not_create_a_false_conflict(self):
        parent = frame("primary", [0, 0, 200, 200], vertical=500)
        parent["metric_spans"].append(
            {"id": "dimension.primary.other", "status": "accepted", "value_mm": 700, "orientation": "vertical", "primitive_refs": []}
        )
        compact = frame("compact", [220, 0, 270, 200], vertical=2500)
        assembly = {
            "instances": [
                {
                    "id": "object.1", "primary_view_id": "primary", "section_view_id": "compact",
                    "view_ids": ["primary", "compact"], "layout_mode": "right", "basis": [],
                }
            ]
        }

        result = solve_shared_coordinate_system([parent, compact], [], [], assembly)

        scope = result["scopes"][0]
        self.assertEqual(scope["state"], "partial")
        self.assertEqual(scope["reprojection_validations"][0]["status"], "unknown")
        self.assertFalse(scope["bar_identity_eligible"])

    def test_module_has_no_document_or_object_class_dispatch(self):
        source = inspect.getsource(__import__("src.drawing_engine.core.shared_coordinate_system", fromlist=["*"])).lower()
        for forbidden in (".pdf", "k1", "k7", "3179", "column", "beam", "stair", "slab"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
