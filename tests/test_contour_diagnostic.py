import unittest

from tools.render_contour_diagnostic import classify_contour


class ContourDiagnosticTest(unittest.TestCase):
    def test_roles_are_topology_and_scale_driven(self):
        view = [0, 0, 100, 100]
        closed = {"bbox_display": [10, 10, 90, 90], "closed": True, "topology": {"cycle_rank": 1}}
        open_profile = {
            "bbox_display": [5, 20, 95, 80], "closed": False,
            "topology": {"endpoint_vertex_count": 2, "branch_vertex_count": 0, "cycle_rank": 0},
        }
        network = {
            "bbox_display": [20, 20, 70, 70], "closed": False,
            "topology": {"endpoint_vertex_count": 2, "branch_vertex_count": 6, "cycle_rank": 8},
        }
        self.assertEqual(classify_contour(closed, view), "outer_closed")
        self.assertEqual(classify_contour(open_profile, view), "outer_open")
        self.assertEqual(classify_contour(network, view), "internal_network")


if __name__ == "__main__":
    unittest.main()
