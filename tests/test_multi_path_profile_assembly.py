import unittest
from copy import deepcopy

import fitz

from src.drawing_engine.disciplines.concrete.multi_path_profile_assembly import (
    assemble_multi_path_profiles,
    validate_multi_path_profile_assembly,
)
from src.drawing_engine.core.vector_topology import extract_page_topology


def dimensions(scale: float = 0.1):
    return [
        {"id": "dimension.1", "state": "accepted", "scale_points_per_mm": scale},
        {"id": "dimension.2", "state": "accepted", "scale_points_per_mm": scale},
    ]


class MultiPathProfileAssemblyTest(unittest.TestCase):
    def _topology(self, paths, *, widths=None):
        document = fitz.open()
        page = document.new_page(width=240, height=180)
        for index, (start, end) in enumerate(paths):
            page.draw_line(start, end, width=(widths or [1.0] * len(paths))[index])
        return document, extract_page_topology(page, vertex_tolerance=0.2)

    def test_unique_scale_bounded_completion_preserves_every_edge_and_bridge(self):
        document, topology = self._topology(
            [
                ((20, 20), (99, 20)),
                ((100, 21), (100, 99)),
                ((99, 100), (20, 100)),
                ((19, 99), (19, 21)),
            ]
        )

        result = assemble_multi_path_profiles(topology, dimensions(), scope_ref="view_scope.1")
        document.close()

        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["summary"]["resolved_count"], 1)
        profile = result["profiles"][0]
        self.assertEqual(profile["scope_ref"], "view_scope.1")
        self.assertEqual(len(profile["source_edge_refs"]), 4)
        self.assertEqual(len(profile["derived_bridge_refs"]), 4)
        self.assertEqual(set(profile["derived_bridge_refs"]), {item["id"] for item in profile["derived_bridges"]})
        self.assertTrue(all(ref in profile["evidence_refs"] for ref in profile["source_edge_refs"]))
        self.assertTrue(all(ref in profile["evidence_refs"] for ref in profile["derived_bridge_refs"]))
        self.assertEqual(
            profile["closure"],
            {
                "closed": True,
                "branch_free": True,
                "unique_completion": True,
                "scale_bounded": True,
                "dimensionally_redundant": True,
            },
        )
        self.assertFalse(profile["quantity_eligible"])
        self.assertFalse(result["contract"]["physical_object_identity_established"])
        self.assertFalse(result["contract"]["schedule_values_used"])
        self.assertEqual(validate_multi_path_profile_assembly(result), [])

        profile["closure"]["unique_completion"] = False
        self.assertTrue(
            any("unique_completion" in error for error in validate_multi_path_profile_assembly(result))
        )

    def test_one_dimension_chain_cannot_certify_a_profile(self):
        document, topology = self._topology(
            [
                ((20, 20), (100, 20)),
                ((100, 20), (100, 100)),
                ((100, 100), (20, 100)),
                ((20, 100), (20, 20)),
            ]
        )

        result = assemble_multi_path_profiles(topology, dimensions()[:1])
        document.close()

        self.assertEqual(result["status"], "abstained")
        self.assertEqual(result["profiles"], [])
        self.assertEqual(result["abstentions"][0]["reason_code"], "insufficient_dimensional_redundancy")

    def test_gap_beyond_metric_bound_abstains(self):
        document, topology = self._topology(
            [
                ((20, 20), (96, 20)),
                ((100, 24), (100, 96)),
                ((96, 100), (20, 100)),
                ((16, 96), (16, 24)),
            ]
        )

        result = assemble_multi_path_profiles(topology, dimensions())
        document.close()

        self.assertEqual(result["status"], "abstained")
        self.assertEqual(result["profiles"], [])
        self.assertEqual(result["abstentions"][0]["reason_code"], "incomplete_profile_completion")

    def test_multiple_simple_completions_abstain(self):
        document, topology = self._topology(
            [
                ((18, 62), (10, 82)),
                ((90, 66), (87, 57)),
                ((55, 65), (72, 65)),
            ]
        )

        result = assemble_multi_path_profiles(
            topology,
            dimensions(scale=1.0),
            max_bridge_mm=200.0,
            max_relative_gap=1.0,
        )
        document.close()

        self.assertEqual(result["profiles"], [])
        self.assertEqual(result["abstentions"][0]["reason_code"], "multiply_closable_profile")

    def test_branched_multi_path_boundary_abstains(self):
        document, topology = self._topology(
            [
                ((20, 20), (100, 20)),
                ((100, 20), (100, 100)),
                ((100, 100), (20, 100)),
                ((20, 100), (20, 20)),
                ((100, 20), (130, 20)),
            ]
        )

        result = assemble_multi_path_profiles(topology, dimensions())
        document.close()

        self.assertEqual(result["profiles"], [])
        self.assertEqual(result["abstentions"][0]["reason_code"], "branched_native_boundary")
        self.assertTrue(result["abstentions"][0]["branch_vertex_refs"])

    def test_incompatible_styles_do_not_form_a_profile(self):
        document, topology = self._topology(
            [
                ((20, 20), (99, 20)),
                ((100, 21), (100, 99)),
                ((99, 100), (20, 100)),
                ((19, 99), (19, 21)),
            ],
            widths=[1.0, 1.0, 2.0, 1.0],
        )

        result = assemble_multi_path_profiles(topology, dimensions())
        document.close()

        self.assertEqual(result["profiles"], [])
        self.assertEqual(result["status"], "abstained")

    def test_ids_are_scope_namespaced_and_topology_order_invariant(self):
        document, topology = self._topology(
            [
                ((20, 20), (100, 20)),
                ((100, 20), (100, 100)),
                ((100, 100), (20, 100)),
                ((20, 100), (20, 20)),
            ]
        )
        reordered = deepcopy(topology)
        reordered["segments"].reverse()
        reordered["vertices"].reverse()

        first = assemble_multi_path_profiles(
            topology,
            dimensions(),
            page_number=3,
            scope_ref="title_view_segment.001",
        )
        replay = assemble_multi_path_profiles(
            reordered,
            list(reversed(dimensions())),
            page_number=3,
            scope_ref="title_view_segment.001",
        )
        other_scope = assemble_multi_path_profiles(
            topology,
            dimensions(),
            page_number=3,
            scope_ref="title_view_segment.002",
        )
        document.close()

        self.assertEqual(first, replay)
        self.assertNotEqual(first["profiles"][0]["id"], other_scope["profiles"][0]["id"])
        self.assertIn("page_0003.scope_", first["profiles"][0]["id"])


if __name__ == "__main__":
    unittest.main()
