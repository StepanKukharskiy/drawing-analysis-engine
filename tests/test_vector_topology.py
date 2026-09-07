import unittest

import fitz

from src.drawing_engine.core.object_agnostic_understanding import extract_contours
from src.drawing_engine.core.vector_topology import extract_page_topology


class VectorTopologyTest(unittest.TestCase):
    def test_endpoints_shared_across_native_drawings_receive_one_vertex_id(self):
        document = fitz.open()
        page = document.new_page(width=300, height=300)
        page.draw_line((50, 50), (150, 50), width=1)
        page.draw_line((150, 50), (150, 150), width=1)

        topology = extract_page_topology(page)
        first, second = topology["segments"]

        self.assertEqual(first["end_vertex_id"], second["start_vertex_id"])
        vertex = next(item for item in topology["vertices"] if item["id"] == first["end_vertex_id"])
        self.assertEqual(vertex["degree"], 2)

    def test_contour_record_preserves_ordered_segments_and_closure(self):
        document = fitz.open()
        page = document.new_page(width=300, height=300)
        page.draw_rect((50, 60, 180, 190), width=1)

        topology = extract_page_topology(page)
        contours = extract_contours(page, topology)

        self.assertEqual(len(contours), 1)
        self.assertTrue(contours[0]["closed"])
        self.assertEqual(contours[0]["topology"]["segment_count"], 4)
        self.assertEqual(contours[0]["topology"]["cycle_rank"], 1)
        self.assertEqual(contours[0]["closure_validation"]["status"], "pass")
        self.assertTrue(all(item["start_vertex_id"] for item in contours[0]["segments_display"]))


if __name__ == "__main__":
    unittest.main()
