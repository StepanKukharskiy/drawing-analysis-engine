import json
import unittest
from pathlib import Path

from src.drawing_engine.disciplines.mep.mep_audit_trace_network import (
    build_mep_audit_trace_network, validate_mep_audit_trace_network,
)


ROOT = Path(__file__).resolve().parents[1]


class MepAuditTraceNetworkUnitTest(unittest.TestCase):
    def test_unique_anchored_colour_correlates_review_rows_without_quantity_authority(self):
        def fragment(ref, x0, x1, colour):
            return {"id": ref, "page_ref": "page.1", "endpoint_vertex_refs": [f"v{x0}", f"v{x1}"],
                    "style": {"stroke": colour}}

        graph = {"layer": "mep_route_observations", "pages": [{"fragments": [
            fragment("f.anchor", 0, 1, [1, 0, 0]),
            fragment("f.review", 1, 2, [1, 0, 0]),
            fragment("f.unknown", 2, 3, [0, 0, 0]),
        ]}]}
        routes = []
        for name, fragment_ref, x0, x1 in (("anchor", "f.anchor", 0, 1),
                                           ("review", "f.review", 1, 2),
                                           ("unknown", "f.unknown", 2, 3)):
            routes.append({"id": f"route.{name}", "occurrences": [{
                "occurrence_ref": f"occ.{name}", "page_ref": "page.1",
                "source_fragment_refs": [fragment_ref], "points_display": [[x0, 0], [x1, 0]],
                "projected_2d_length_m": 1.0,
            }]})
        observed = {"layer": "observed", "sheet_coverage": [{"page_ref": "page.1", "page_number": 1}],
                    "route_segment_ledger": routes}
        semantic = {"layer": "semantic", "semantic_route_groups": [{
            "id": "group.1", "source_route_ledger_refs": ["route.anchor"], "page_refs": ["page.1"],
            "semantic_signature": {"system": {"value": {"kind": "heating_hot_water_supply"}}},
            "length_channels": {"canonicalized_projected_length_m": 1.0},
        }]}
        payload = build_mep_audit_trace_network(
            observed=observed, semantic=semantic, route_graph=graph, source_pdf_sha256="source")
        self.assertEqual(validate_mep_audit_trace_network(payload, expected_occurrence_count=3), [])
        states = {row["occurrence_ref"]: row["display_state"]
                  for row in payload["route_occurrence_assignments"]}
        self.assertEqual(states["occ.anchor"], "certified_system")
        self.assertEqual(states["occ.review"], "colour_correlated_review")
        self.assertEqual(states["occ.unknown"], "unknown_engineer_review")
        review = next(row for row in payload["route_occurrence_assignments"]
                      if row["occurrence_ref"] == "occ.review")
        self.assertFalse(review["semantic_quantity_eligible"])
        trace = next(row for row in payload["page_system_traces"] if row["system"])
        self.assertEqual(trace["topology"]["exact_endpoint_component_count"], 1)
        self.assertFalse(trace["topology"]["physical_continuity_established"])


class MepAuditTraceNetworkRealOutputTest(unittest.TestCase):
    def test_all_region_owned_occurrences_are_indexed_and_colour_is_review_only(self):
        payload = json.loads((ROOT / "output/mep-audit-trace-network-2026-09-03/trace-network.json").read_text())
        self.assertEqual(validate_mep_audit_trace_network(payload, expected_occurrence_count=7779), [])
        self.assertEqual(payload["summary"]["certified_system_occurrence_count"], 160)
        self.assertEqual(payload["summary"]["colour_correlated_review_occurrence_count"], 816)
        self.assertEqual(payload["summary"]["unknown_engineer_review_occurrence_count"], 6803)
        self.assertEqual(len(payload["colour_mapping_certificates"]), 4)
        self.assertTrue(all(not row["colour_alone_establishes_route_identity"]
                            for row in payload["colour_mapping_certificates"]))


if __name__ == "__main__":
    unittest.main()
