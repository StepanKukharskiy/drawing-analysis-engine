import copy
import json
from pathlib import Path
import unittest
import tempfile

from test_mep_cross_sheet_runs import _segment, _build_graph, _bindings, _registration
from src.drawing_engine.disciplines.mep.mep_cross_sheet_runs import build_mep_cross_sheet_runs
from src.drawing_engine.disciplines.mep.mep_network_assembly import build_mep_networks


def network(page_segments, registrations=(), overrides=None):
    registry, graph = _build_graph(page_segments, registrations)
    specs = [{"page_ref": page, "source_ref": segment["id"],
              **(overrides or {}).get((page, segment["id"]), {})}
             for page, segments in page_segments.items() for segment in segments]
    bindings = _bindings(graph, specs)
    return build_mep_networks(sheet_registry=registry, route_graph=graph, attribute_bindings=bindings)


class MepNetworkAssemblyTest(unittest.TestCase):
    def test_frozen_real_composite_overlap_stays_two_intervals_without_a_run(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures/mep/m_and_p_coordination"
        checkpoint = fixture / "real_m2_m5_checkpoint"
        registry = json.loads((fixture / "m_and_p_coordination.sheet-registry.json").read_text())
        values = {name: json.loads((checkpoint / ("pages_1a_1b." + name + ".json")).read_text())
                  for name in ("route-observations", "attribute-bindings", "cross-sheet-runs")}
        result = build_mep_networks(sheet_registry=registry, route_graph=values["route-observations"],
            attribute_bindings=values["attribute-bindings"], cross_sheet_runs=values["cross-sheet-runs"])
        self.assertEqual(result["summary"]["segment_count"], 2)
        self.assertEqual(result["summary"]["multi_segment_run_count"], 0)
        self.assertEqual(result["summary"]["complete_run_count"], 0)
        self.assertEqual(sum(len(r["source_occurrence_refs"]) for r in result["segments"]), 4)
        self.assertTrue(all(not row["quantity_eligible"] for row in result["segments"]))

    def test_native_elbow_trace_groups_segments_but_never_claims_complete_physical_run(self):
        segments = [_segment(0, (0, 0), (10, 0)), _segment(1, (10, 0), (10, 20))]
        result = network({"p": segments})
        self.assertEqual(result["summary"]["multi_segment_run_count"], 1)
        self.assertEqual(result["summary"]["connected_projected_network_count"], 1)
        self.assertEqual(len(result["unresolved_boundaries"]), 2)
        self.assertFalse(result["runs"][0]["physical_continuation_established"])
        self.assertFalse(result["runs"][0]["complete_trace_established"])
        self.assertEqual(result["runs"], network({"p": list(reversed(segments))})["runs"])

    def test_branches_create_network_junction_and_split_runs(self):
        result = network({"p": [_segment(0, (0, 0), (10, 0)),
            _segment(1, (10, 0), (20, 0)), _segment(2, (10, 0), (10, 10))]})
        self.assertEqual(len(result["runs"]), 3)
        self.assertEqual(len(result["networks"]), 1)
        self.assertEqual(result["junctions"][0]["relation_type"], "projected_branch")

    def test_crossing_proximity_and_overlapping_paths_do_not_join(self):
        for segments, overrides in [
            ([_segment(0, (0, 5), (10, 5)), _segment(1, (5, 0), (5, 10))], {}),
            ([_segment(0, (0, 0), (10, 0)), _segment(1, (10.2, 0), (20, 0))], {}),
            ([_segment(0, (0, 0), (10, 0)), _segment(1, (0, 0), (20, 0))], {}),
        ]:
            with self.subTest(segments=segments):
                result = network({"p": segments}, overrides=overrides)
                self.assertEqual(result["summary"]["connected_projected_network_count"], 0)
                self.assertEqual(result["summary"]["complete_run_count"], 0)

    def test_attribute_conflict_preserves_geometry_but_never_physical_compatibility(self):
        segments = [_segment(0, (0, 0), (10, 0)), _segment(1, (10, 0), (20, 0))]
        result = network({'p': segments}, overrides={('p', segments[1]['id']): {'system': 'CHWR'}})
        self.assertEqual(result['summary']['connected_projected_network_count'], 1)
        join = result['junctions'][0]
        self.assertEqual(join['state'], 'accepted')
        self.assertTrue(join['semantic_conflicts'])
        self.assertTrue(all(r['semantic_overlay']['system']['state'] == 'conflicted' for r in result['segments']))
        self.assertFalse(join['physical_continuation_established'])

    def test_certified_outlines_survive_attribute_removal_but_not_geometry_tampering(self):
        from test_mep_native_boundary_connections import MepNativeBoundaryConnectionsTest
        from src.drawing_engine.disciplines.mep.mep_attribute_binding import build_mep_attribute_bindings
        from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256
        with tempfile.TemporaryDirectory() as directory:
            targets, terminology, m4, registry = MepNativeBoundaryConnectionsTest()._connected(directory, with_registry=True)
            graph = targets['route_graph']
            boundary = {'m3_payload_sha256': _sha256(graph),
                        'm35_payload_sha256': _sha256(targets['composites']),
                        'connections': targets['boundary_connections']}
            inputs = dict(sheet_registry=registry, route_graph=graph, boundary_connections=boundary)
            before = build_mep_networks(**inputs, attribute_bindings=m4)
            interface = before['junctions'][0]
            self.assertTrue(interface['source_search']['complete'])
            self.assertTrue(all(
                binding['composite_endpoint_index'] in (0, 1)
                and len(binding['side_points_display']) == 2
                for binding in interface['port_segment_bindings']
            ))
            run = before['runs'][0]
            self.assertEqual(len(run['projected_endpoint_occurrences']), 4)
            self.assertEqual(len(run['endpoint_interface_candidate_refs']), 1)
            self.assertFalse(run['endpoint_interface_search_complete'])
            self.assertTrue(all(
                not row['physical_terminal_established']
                and not row['quantity_eligible']
                for row in run['projected_endpoint_occurrences']
            ))
            empty = build_mep_attribute_bindings(terminology_proposals=terminology, route_graph=graph,
                outlined_route_composites=targets['composites'], binding_evidence=[])
            after = build_mep_networks(**inputs, attribute_bindings=empty)
            self.assertEqual([r['id'] for r in before['segments']], [r['id'] for r in after['segments']])
            self.assertEqual(after['summary']['connected_projected_network_count'], 1)
            self.assertTrue(all(r['geometry_only'] for r in after['segments']))
            self.assertTrue(all(r['semantic_overlay']['elevation']['state'] == 'unknown' for r in after['segments']))
            reordered = copy.deepcopy(empty)
            reordered['outlined_route_composites'].reverse()
            self.assertEqual(after['segments'], build_mep_networks(**inputs, attribute_bindings=reordered)['segments'])
            forged = copy.deepcopy(empty)
            forged['outlined_route_composites'][0]['derived_geometry']['centreline_points_display'][0][0] += 1
            with self.assertRaisesRegex(ValueError, 'replay'):
                build_mep_networks(**inputs, attribute_bindings=forged)

    def test_unclassified_single_strokes_do_not_become_network_segments(self):
        from src.drawing_engine.disciplines.mep.mep_attribute_binding import build_mep_attribute_bindings
        from src.drawing_engine.disciplines.mep.mep_terminology_proposals import build_mep_terminology_proposals
        registry, graph = _build_graph({'p': [_segment(0, (0, 0), (10, 0))]}, [])
        m2 = build_mep_terminology_proposals(document=graph['document'], observations=[])
        m4 = build_mep_attribute_bindings(terminology_proposals=m2, route_graph=graph, binding_evidence=[])
        result = build_mep_networks(sheet_registry=registry, route_graph=graph, attribute_bindings=m4)
        self.assertEqual(result['segments'], [])
        self.assertTrue(result['coverage']['excluded_observations'])

    def test_registered_continuation_joins_local_trace_across_pages(self):
        left, elbow, right = (_segment(0, (0, 0), (10, 0)),
            _segment(1, (10, 0), (10, 10)), _segment(2, (100, 100), (110, 100)))
        result = network({"a": [left, elbow], "b": [right]},
            [_registration("ab", "a", "b", 90, 90)],
            {("a", elbow["id"]): {"continuation_role": "end"},
             ("b", right["id"]): {"continuation_role": "start"}})
        self.assertEqual(result["summary"]["connected_projected_network_count"], 1)
        self.assertTrue(any(r["relation_type"] == "cross_sheet_continuation" and r["state"] == "accepted"
                            for r in result["junctions"]))
        self.assertEqual(sum(len(r["segment_refs"]) for r in result["runs"]), 3)

    def test_duplicate_interval_does_not_become_a_run_and_requires_m5_replay(self):
        left, right = _segment(0, (0, 0), (10, 0)), _segment(1, (100, 0), (110, 0))
        registry, graph = _build_graph({"a": [left], "b": [right]}, [_registration("ab", "a", "b", 100)])
        bindings = _bindings(graph, [{"page_ref": "a", "source_ref": left["id"]},
                                     {"page_ref": "b", "source_ref": right["id"]}])
        inputs = dict(sheet_registry=registry, route_graph=graph, attribute_bindings=bindings)
        m5 = build_mep_cross_sheet_runs(**inputs)
        frozen = copy.deepcopy(inputs)
        result = build_mep_networks(**inputs, cross_sheet_runs=m5)
        self.assertEqual(result["summary"]["segment_count"], 1)
        self.assertEqual(result["summary"]["multi_segment_run_count"], 0)
        self.assertEqual(len(result["segments"][0]["source_occurrence_refs"]), 2)
        self.assertEqual(inputs, frozen)
        m5["overlap_duplicate_candidates"][0]["state"] = "abstained"
        with self.assertRaisesRegex(ValueError, "does not replay"):
            build_mep_networks(**inputs, cross_sheet_runs=m5)
        inputs["route_graph"]["document"]["revision"] = "changed"
        with self.assertRaisesRegex(ValueError, "frozen M3"):
            build_mep_networks(**inputs)
