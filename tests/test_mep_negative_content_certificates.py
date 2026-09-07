import unittest

from src.drawing_engine.disciplines.mep.mep_negative_content_certificates import (
    build_negative_content_certificates, validate_negative_content_certificates,
)


class MepNegativeContentCertificatesTest(unittest.TestCase):
    def test_grid_axis_and_regular_hatch_are_measured_from_geometry(self):
        entities = [{
            "id": "grid", "entity_kind": "component", "page_ref": "page.5",
            "source_primitive_refs": ["source.grid"],
            "polylines_display": [[[10, 40], [190, 40]]],
            "non_colour_style": {"width_display_points": .5, "dash_pattern": "[] 0"},
        }]
        for index in range(8):
            entities.append({
                "id": f"hatch.{index}", "entity_kind": "deferred", "page_ref": "page.5",
                "source_primitive_refs": [f"source.hatch.{index}"],
                "polylines_display": [[[60, 80 + index * 2], [100, 80 + index * 2]]],
                "non_colour_style": {"width_display_points": .25, "dash_pattern": "[] 0"},
            })
        payload = build_negative_content_certificates(
            entities=entities,
            grid_axes=[{
                "id": "axis.40", "state": "observed",
                "method": "opposing_equal_label_alignment_v1",
                "opposing_label_pair_count": 1,
                "axis_group_evidence_refs": ["axis.group"],
                "evidence_refs": ["label.left", "label.right"],
                "orientation_display": "horizontal", "coordinate_display": 40,
                "opposing_label_span_display": [0, 200], "label": "A",
            }],
            page_rect_display=[0, 0, 200, 200], precision_display_points=.05)
        by_id = {row["entity_ref"]: row for row in payload["evaluations"]}
        self.assertEqual("accepted", by_id["grid"]["architectural_boundary_certificate"]["state"])
        self.assertTrue(all(by_id[f"hatch.{index}"]["hatch_certificate"]["state"] == "accepted"
                            for index in range(8)))
        self.assertFalse(payload["measurement_contract"]["fixed_page_coordinates_used"])
        self.assertEqual([], validate_negative_content_certificates(payload))

    def test_nonmatching_route_is_measured_without_a_fabricated_negative(self):
        payload = build_negative_content_certificates(
            entities=[{
                "id": "route", "entity_kind": "component", "page_ref": "page.5",
                "source_primitive_refs": ["source.route"],
                "polylines_display": [[[20, 20], [40, 25]]],
                "non_colour_style": {"width_display_points": .7, "dash_pattern": "[] 0"},
            }],
            grid_axes=[], page_rect_display=[0, 0, 200, 200],
            precision_display_points=.05)
        row = payload["evaluations"][0]
        self.assertEqual("no_match", row["hatch_certificate"]["state"])
        self.assertEqual("no_match", row["architectural_boundary_certificate"]["state"])
        self.assertFalse(payload["acceptance_gate"]["literal_zero_counts_used"])
        self.assertEqual([], validate_negative_content_certificates(payload))


if __name__ == "__main__":
    unittest.main()
