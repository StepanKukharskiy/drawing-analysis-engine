import unittest

import fitz

from src.drawing_engine.disciplines.concrete.scoped_profile_assembly import assemble_profiles_by_view_scope
from src.drawing_engine.core.vector_topology import extract_page_topology


def dimension(ref, scale=0.1):
    return {"id": ref, "status": "accepted", "scale_points_per_mm": scale}


def ownership(ref, view_ref, drawing_ref):
    return {
        "id": f"ownership.{ref}",
        "dimension_ref": ref,
        "status": "accepted",
        "view_refs": [view_ref],
        "primitive_refs": [f"{drawing_ref}.item[0]"],
    }


class ScopedProfileAssemblyTest(unittest.TestCase):
    def _two_profiles(self):
        document = fitz.open()
        page = document.new_page(width=320, height=160)
        for offset in (0, 140):
            page.draw_line((20 + offset, 20), (100 + offset, 20), width=1)
            page.draw_line((100 + offset, 20), (100 + offset, 100), width=1)
            page.draw_line((100 + offset, 100), (20 + offset, 100), width=1)
            page.draw_line((20 + offset, 100), (20 + offset, 20), width=1)
        return document, extract_page_topology(page, vertex_tolerance=0.2)

    def test_two_scopes_use_only_local_dimensions_and_emit_unique_profiles(self):
        document, topology = self._two_profiles()
        dimensions = [dimension(ref) for ref in ("d1", "d2", "d3", "d4")]
        result = assemble_profiles_by_view_scope(
            topology,
            dimensions,
            {
                "attachments": [
                    ownership("d1", "view.1", "drawing[0]"),
                    ownership("d2", "view.1", "drawing[1]"),
                    ownership("d3", "view.2", "drawing[4]"),
                    ownership("d4", "view.2", "drawing[5]"),
                ]
            },
            {
                "segments": [
                    {
                        "id": "scope.1",
                        "view_id": "view.1",
                        "state": "resolved",
                        "title_anchor_ref": "title.1",
                        "primitive_refs": [f"drawing[{index}]" for index in range(4)],
                        "excluded_primitive_refs": [],
                    },
                    {
                        "id": "scope.2",
                        "view_id": "view.2",
                        "state": "resolved",
                        "title_anchor_ref": "title.2",
                        "primitive_refs": [f"drawing[{index}]" for index in range(4, 8)],
                        "excluded_primitive_refs": [],
                    },
                ]
            },
            page_number=7,
        )
        document.close()

        self.assertEqual(result["summary"]["profile_count"], 2)
        self.assertEqual(result["scopes"][0]["accepted_dimension_refs"], ["d1", "d2"])
        self.assertEqual(result["scopes"][1]["accepted_dimension_refs"], ["d3", "d4"])
        profile_ids = [item["id"] for item in result["profiles"]]
        self.assertEqual(len(profile_ids), len(set(profile_ids)))
        self.assertTrue(all(item["quantity_eligible"] is False for item in result["profiles"]))
        self.assertFalse(result["contract"]["physical_object_identity_established"])

    def test_dimensions_from_another_scope_cannot_supply_redundancy(self):
        document, topology = self._two_profiles()
        result = assemble_profiles_by_view_scope(
            topology,
            [dimension("local"), dimension("foreign.1"), dimension("foreign.2")],
            {
                "attachments": [
                    ownership("local", "view.1", "drawing[0]"),
                    ownership("foreign.1", "view.2", "drawing[4]"),
                    ownership("foreign.2", "view.2", "drawing[5]"),
                ]
            },
            {
                "segments": [
                    {
                        "id": "scope.1",
                        "view_id": "view.1",
                        "state": "resolved",
                        "title_anchor_ref": "title.1",
                        "primitive_refs": [f"drawing[{index}]" for index in range(4)],
                        "excluded_primitive_refs": [],
                    }
                ]
            },
            page_number=1,
        )
        document.close()

        scope = result["scopes"][0]
        self.assertEqual(scope["accepted_dimension_refs"], ["local"])
        self.assertEqual(scope["profiles"], [])
        self.assertEqual(scope["abstentions"][0]["reason_code"], "insufficient_dimensional_redundancy")

    def test_excluded_boundary_path_never_enters_bridge_search(self):
        document, topology = self._two_profiles()
        result = assemble_profiles_by_view_scope(
            topology,
            [dimension("d1"), dimension("d2")],
            {
                "attachments": [
                    ownership("d1", "view.1", "drawing[0]"),
                    ownership("d2", "view.1", "drawing[1]"),
                ]
            },
            {
                "segments": [
                    {
                        "id": "scope.1",
                        "view_id": "view.1",
                        "state": "resolved",
                        "title_anchor_ref": "title.1",
                        "primitive_refs": ["drawing[0]", "drawing[1]", "drawing[2]"],
                        "excluded_primitive_refs": ["drawing[3]"],
                    }
                ]
            },
            page_number=1,
        )
        document.close()

        scope = result["scopes"][0]
        self.assertEqual(scope["profiles"], [])
        self.assertEqual(scope["excluded_primitive_refs"], ["drawing[3]"])
        self.assertNotIn("drawing[3]", scope["primitive_refs"])
        self.assertTrue(result["contract"]["excluded_primitives_never_enter_bridge_search"])


if __name__ == "__main__":
    unittest.main()
