import unittest

from src.drawing_engine.disciplines.mep.mep_endpoint_coverage import build_endpoint_coverage_matrix


def search(run, occurrence, endpoint, candidates=(), complete=True):
    return {
        "id": f"search.{run}.{occurrence}.{endpoint}",
        "network_run_ref": run,
        "page_ref": f"page.{occurrence}",
        "route_composite_ref": f"composite.{occurrence}",
        "canonical_endpoint_index": endpoint,
        "native_search_complete": complete,
        "raster_search_state": "not_required_native_primary",
        "source_query_refs": [f"native.query.{occurrence}.{endpoint}"],
        "source_inventory_sha256": "a" * 64,
        "interface_candidates": list(candidates),
    }


def terminal(identity):
    return {
        "id": f"terminal.{identity}", "state": "accepted",
        "explicit_terminal_identity": True, "uniquely_bound_to_endpoint": True,
        "identity_key": identity,
    }


class MepEndpointCoverageTest(unittest.TestCase):
    def test_competing_outline_and_bend_freeze_endpoint_identity(self):
        candidate = {"priority_rank": 1, "network_run_ref": "run.a",
                     "segment_refs": ["segment.a"], "source_page_refs": ["page.1", "page.2"],
                     "duplicate_occurrence_count": 2, "bounded_local_3d_available": True,
                     "complete_projected_topology": False}
        competitors = [{"id": "outline.cap", "state": "unresolved",
                        "explicit_terminal_identity": False, "uniquely_bound_to_endpoint": False},
                       {"id": "bend.branch", "state": "abstained",
                        "explicit_terminal_identity": False, "uniquely_bound_to_endpoint": False}]
        searches = [search("run.a", occurrence, endpoint,
                           competitors if endpoint == 0 else ())
                    for occurrence in (1, 2) for endpoint in (0, 1)]
        payload = build_endpoint_coverage_matrix(
            document={"document_key": "pdf-sha256:test"}, candidate_runs=[candidate],
            endpoint_searches=searches,
            connection_standard_search={"state": "complete_no_match", "resolved_standard": None},
        )
        row = payload["matrix"][0]
        self.assertEqual(row["status"], "endpoint_identity_unresolved")
        self.assertEqual(row["canonical_endpoint_outcomes"]["0"]["state"], "ambiguous")
        self.assertEqual(row["canonical_endpoint_outcomes"]["1"]["state"], "apparently_absent")
        self.assertIsNone(payload["selected_physical_replay_run_ref"])
        self.assertIsNone(row["commercial_channels"]["connection_standard"])
        self.assertTrue(row["geometry_channels"]["bounded_local_3d_available"])

    def test_first_two_terminal_candidate_wins_over_geometry_rank(self):
        runs = [
            {"priority_rank": 1, "network_run_ref": "run.negative", "segment_refs": ["s1"],
             "source_page_refs": ["p1"], "duplicate_occurrence_count": 1,
             "bounded_local_3d_available": True, "complete_projected_topology": False},
            {"priority_rank": 7, "network_run_ref": "run.positive", "segment_refs": ["s2"],
             "source_page_refs": ["p2"], "duplicate_occurrence_count": 1,
             "bounded_local_3d_available": True, "complete_projected_topology": True},
        ]
        searches = [search("run.negative", 1, endpoint) for endpoint in (0, 1)]
        searches += [search("run.positive", 2, endpoint, [terminal(f"physical.{endpoint}")])
                     for endpoint in (0, 1)]
        payload = build_endpoint_coverage_matrix(
            document={"document_key": "pdf-sha256:test"}, candidate_runs=runs,
            endpoint_searches=searches,
            connection_standard_search={"state": "complete_no_match", "resolved_standard": None},
        )
        self.assertEqual(payload["selected_physical_replay_run_ref"], "run.positive")
        positive = payload["matrix"][0]
        self.assertEqual(positive["status"], "eligible_for_m5c_physical_replay")
        self.assertFalse(positive["commercial_channels"]["marketplace_assembly_eligible"])
        self.assertTrue(positive["geometry_channels"]["measured_centreline_length_eligible"])


if __name__ == "__main__":
    unittest.main()
