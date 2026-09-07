import json
from pathlib import Path
import unittest

from src.drawing_engine.disciplines.concrete.terminal_landing_support import (
    materialize_terminal_landing_context,
    reconstruct_terminal_landing_support,
)


ROOT = Path(__file__).resolve().parents[1]
ENGINEERING = ROOT / "output" / "object_agnostic" / "candidate-08-staircase-page.engineering-graph.json"


def _page():
    return json.loads(ENGINEERING.read_text())["pages"][0]


class TerminalLandingSupportTest(unittest.TestCase):
    def test_candidate_08_recovers_partial_landing_and_unclassified_support(self):
        page = _page()
        result = reconstruct_terminal_landing_support(
            page["open_structural_boundary_assembly"],
            page["landing_component_reconstruction"],
            page["banded_plan_sweep_evidence"],
            page["clear_span_prism_reconstruction"],
            page_number=1,
        )

        self.assertEqual(result["status"], "resolved_partial_context")
        landing = result["terminal_landing_partial"]
        self.assertEqual(landing["thickness_mm"], 150.0)
        self.assertAlmostEqual(landing["native_measured_thickness_mm"], 148.652, places=3)
        self.assertEqual(landing["sweep_width_mm"], 975.0)
        self.assertAlmostEqual(landing["visible_section_extent_mm"], 504.583, places=3)
        self.assertFalse(landing["longitudinal_extent_complete"])
        self.assertEqual(landing["external_cap_role"], "analysis_cap")
        self.assertFalse(landing["quantity_eligible"])

        support = result["terminal_support_candidate"]
        self.assertEqual(support["section_width_mm"], 200.0)
        self.assertAlmostEqual(support["native_measured_section_width_mm"], 197.855, places=3)
        self.assertEqual(support["drop_below_slab_mm"], 300.0)
        self.assertAlmostEqual(support["native_measured_drop_below_slab_mm"], 296.260, places=3)
        self.assertEqual(support["clear_span_mm"], 2150.0)
        self.assertEqual(support["volume_candidate_m3"], 0.129)
        self.assertFalse(support["member_identity_resolved"])
        self.assertFalse(support["quantity_eligible"])

        materialization = materialize_terminal_landing_context(
            result,
            page["solid_preview"]["mesh"],
            page_number=1,
        )
        self.assertEqual(materialization["status"], "materialized_partial_hypotheses")
        self.assertEqual(materialization["summary"]["hypothesis_count"], 2)
        contexts = materialization["candidate_previews"]
        self.assertEqual(len(contexts), 2)
        self.assertEqual(
            [item["classification"] for item in contexts],
            ["upper_landing_or_floor_slab", "terminal_support_beam_candidate"],
        )
        self.assertTrue(all(item["relative_physical_placement_resolved"] for item in contexts))
        self.assertTrue(all(item["mesh"]["validation"]["watertight"] for item in contexts))
        self.assertTrue(all(not item["included_in_primary_quantity"] for item in contexts))
        self.assertFalse(contexts[0]["physical_boundary_complete"])
        self.assertTrue(contexts[1]["physical_boundary_complete"])
        self.assertTrue(
            all(
                not coverage["derived_from_generated_geometry"]
                for item in contexts
                for coverage in item["projection_coverage_records"]
            )
        )

    def test_missing_native_terminal_chain_abstains(self):
        page = _page()
        assembly = json.loads(json.dumps(page["open_structural_boundary_assembly"]))
        for scope in assembly["scope_results"]:
            if scope["scope_ref"] == "title_view_segment.002":
                scope["branch_free_chains"] = [
                    item
                    for item in scope["branch_free_chains"]
                    if item["id"] != "structural_boundary_chain.page_0001.evidence_72faa5491aa7541c"
                ]
        result = reconstruct_terminal_landing_support(
            assembly,
            page["landing_component_reconstruction"],
            page["banded_plan_sweep_evidence"],
            page["clear_span_prism_reconstruction"],
            page_number=1,
        )
        self.assertEqual(result["status"], "insufficient_constraints")
        self.assertEqual(result["reason_code"], "unique_terminal_landing_profile_unresolved")


if __name__ == "__main__":
    unittest.main()
