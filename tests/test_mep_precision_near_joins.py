import unittest

from src.drawing_engine.disciplines.mep.mep_precision_near_joins import (
    ABSOLUTE_MAXIMUM, build_precision_near_join_certificates,
    validate_precision_near_joins,
)


def _fragment(index, start, end):
    return {
        "id": f"fragment.{index}",
        "geometry": {"points_display": [start, end]},
        "endpoint_vertex_refs": [f"vertex.{index}.start", f"vertex.{index}.end"],
        "style": {"stroke": [1, 0, 0], "width_display_points": .72,
                  "dash_pattern": "[] 0"},
    }


class PrecisionNearJoinTests(unittest.TestCase):
    def test_style_precision_certifies_only_mutually_unique_bounded_seams(self):
        fragments = []
        left = 0.0
        for index in range(6):
            fragments.append(_fragment(index, [left, 0], [left + 10, 0]))
            left += 10.12
        payload = build_precision_near_join_certificates(route_page={
            "page_ref": "page.5", "fragments": fragments,
            "vertices": [], "crossings": [{"id": "crossing.1"}],
        })
        self.assertEqual(validate_precision_near_joins(payload), [])
        self.assertEqual(payload["summary"]["uniquely_certified_near_join_count"], 5)
        self.assertEqual(payload["summary"]["ambiguous_near_join_candidate_count"], 0)
        self.assertEqual(payload["summary"]["disconnected_crossing_count"], 1)
        certificate = payload["view_style_precision_certificates"][0]
        self.assertEqual(certificate["native_coordinate_precision"]["state"], "observed_grid")
        self.assertAlmostEqual(certificate["derived_near_join_tolerance_display_points"], .12)
        self.assertLessEqual(certificate["strict_style_maximum_display_points"], ABSOLUTE_MAXIMUM)
        self.assertFalse(certificate["global_tolerance_increase_used"])

    def test_ambiguous_endpoint_competitors_are_not_accepted(self):
        fragments = [
            _fragment(1, [0, 0], [10, 0]),
            _fragment(2, [10.12, 0], [20, 0]),
            _fragment(3, [10.12, .01], [20, .01]),
            _fragment(4, [20.12, 0], [30, 0]),
            _fragment(5, [30.12, 0], [40, 0]),
            _fragment(6, [40.12, 0], [50, 0]),
            _fragment(7, [50.12, 0], [60, 0]),
        ]
        payload = build_precision_near_join_certificates(route_page={
            "page_ref": "page.5", "fragments": fragments,
            "vertices": [], "crossings": [],
        })
        self.assertEqual(validate_precision_near_joins(payload), [])
        self.assertGreater(payload["summary"]["ambiguous_near_join_candidate_count"], 0)
        ambiguous_refs = {ref for row in payload["ambiguous_near_join_candidates"]
                          for ref in row["endpoint_refs"]}
        accepted_refs = {ref for row in payload["uniquely_certified_near_joins"]
                         for ref in row["endpoint_refs"]}
        self.assertNotIn("fragment.1:end", accepted_refs)
        self.assertIn("fragment.1:end", ambiguous_refs)


if __name__ == "__main__":
    unittest.main()
