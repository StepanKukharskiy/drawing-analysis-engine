import inspect
import unittest

from src.drawing_engine.disciplines.detail.native_detail_takeoff import (
    _cluster_points,
    _leader_attachment_fanout,
    _mass_per_metre,
    _parse_spacing_variants,
    _scene_paths,
    _zone_count,
    parse_detail_annotations,
)


class NativeDetailTakeoffTest(unittest.TestCase):
    def test_parses_bracketed_lengths_and_radius_without_schedule(self):
        parsed = parse_detail_annotations(
            [
                {"psm": 6, "text": "300 30 RBH=42 L=5760[4680](3960)"},
                {"psm": 11, "text": "RBH=42 (=5760[4680](3960)"},
            ],
            ["3", "4", "5"],
            {"3": "bare", "4": "square", "5": "parentheses"},
        )
        self.assertEqual(parsed["status"], "resolved")
        self.assertEqual(parsed["length_by_mark_mm"], {"3": 5760, "4": 4680, "5": 3960})
        self.assertEqual((parsed["inner_bend_radius_mm"], parsed["diameter_mm"]), (42, 14))

    def test_spacing_arithmetic_and_vector_station_deduplication(self):
        variants = _parse_spacing_variants(
            [{"text": "49x100=4900 (31x100=3100) [38x100=3800]"}]
        )
        self.assertEqual(variants, {"bare": 49, "parentheses": 31, "square": 38})
        clusters = _cluster_points([(0, 0), (0.7, 0), (10, 0), (10.8, 0), (200, 0), (200.7, 0)])
        centers = [tuple(sum(value) / len(value) for value in zip(*cluster)) for cluster in clusters]
        self.assertEqual(len(centers), 3)
        self.assertEqual(_zone_count(centers), 2)

    def test_mass_uses_nominal_area_and_steel_density(self):
        self.assertAlmostEqual(_mass_per_metre(8), 0.394584, places=6)
        self.assertAlmostEqual(_mass_per_metre(14), 1.208414, places=6)

    def test_arrowhead_glyph_vertices_do_not_inflate_fanout(self):
        association = {
            "path_attachments": [
                {"terminal_display": point}
                for point in ([0, 0], [2, 1], [3, -1], [0, 50], [2, 51])
            ]
        }
        self.assertEqual(_leader_attachment_fanout(association), 4)

    def test_end_anchor_paths_preserve_distinct_section_offsets(self):
        paths = _scene_paths(
            [{
                "object_instance_id": "object.1", "mark": "6", "role": "end_anchor",
                "diameter_mm": 14, "count": 8, "length_each_mm": 510,
                "evidence_refs": ["detail.1"],
            }],
            {
                "shape_type": "multi_object_extrusion_collection",
                "objects": [{
                    "object_instance_id": "object.1", "profile_points_xz_mm": [[0, 2000], [3200, 0]],
                    "extrusion_depth_mm": 200,
                }],
            },
            {"components": [{"object_instance_id": "object.1", "display_offset_x_mm": 0}]},
        )
        self.assertEqual(len(paths), 8)
        self.assertEqual(len({tuple(path["points_xyz_mm"][1]) for path in paths[:4]}), 4)

    def test_module_has_no_drawing_or_filename_dispatch(self):
        source = inspect.getsource(__import__("src.drawing_engine.disciplines.detail.native_detail_takeoff", fromlist=["*"])).lower()
        for forbidden in (".pdf", "3179", "k1", "k2", "k3", "stair", "column", "beam"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
