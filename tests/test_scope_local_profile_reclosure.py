import unittest

import fitz

from src.drawing_engine.disciplines.concrete.scope_local_profile_reclosure import reclose_profiles_in_title_scopes
from src.drawing_engine.core.vector_topology import extract_page_topology


class ScopeLocalProfileReclosureTest(unittest.TestCase):
    def test_existing_assembler_reruns_without_emitting_pair_certificate(self):
        document = fitz.open()
        page = document.new_page(width=320, height=160)
        for offset in (0, 140):
            page.draw_line((20 + offset, 20), (100 + offset, 20), width=1)
            page.draw_line((100 + offset, 20), (100 + offset, 100), width=1)
            page.draw_line((100 + offset, 100), (20 + offset, 100), width=1)
            page.draw_line((20 + offset, 100), (20 + offset, 20), width=1)
        topology = extract_page_topology(page, vertex_tolerance=0.2)
        result = reclose_profiles_in_title_scopes(
            topology,
            [
                {"id": "dimension.1", "status": "ambiguous", "scale_points_per_mm": 0.1},
                {"id": "dimension.2", "status": "ambiguous", "scale_points_per_mm": 0.1},
            ],
            {
                "segments": [
                    {
                        "id": "section.scope",
                        "state": "resolved",
                        "primitive_refs": [f"drawing[{index}]" for index in range(8)],
                        "excluded_primitive_refs": [],
                    }
                ]
            },
            {
                "scope_results": [
                    {
                        "scope_ref": "section.scope",
                        "status": "resolved_subset",
                        "accepted_dimension_refs": ["dimension.1", "dimension.2"],
                        "independent_metric_check_count": 2,
                    }
                ]
            },
            page_number=1,
        )
        document.close()

        scope = result["scope_results"][0]
        self.assertEqual(scope["status"], "two_profile_candidate_set_available")
        self.assertEqual(scope["profile_candidate_count"], 2)
        self.assertIsNone(scope["pair_certificate"])
        self.assertEqual(
            scope["pair_selection_assessment"]["validation_status"],
            "not_run",
        )
        self.assertFalse(result["contract"]["profile_selector_used"])
        self.assertFalse(result["contract"]["quantity_eligible"])


if __name__ == "__main__":
    unittest.main()
