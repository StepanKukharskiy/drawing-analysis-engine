import copy
import unittest

from src.drawing_engine.disciplines.mep.mep_marketplace_assembly import build_mep_marketplace_assemblies
from src.drawing_engine.disciplines.mep.mep_physical_run_evidence import (
    FIELDS,
    build_mep_physical_run_readiness,
    validate_mep_physical_run_readiness,
)
from src.drawing_engine.project.takeoff_intelligence import canonical_sha256


def inputs(*, complete=True):
    document = {"document_key": "pdf-sha256:physical-run-test", "page_count": 1}
    relations = []
    attachments = []
    for endpoint_index in (0, 1):
        relation = {
            "id": f"terminal.{endpoint_index}",
            "state": "accepted",
            "relation_type": "physical_terminal",
            "page_ref": "page.1",
            "target_refs": [f"endpoint.{endpoint_index}"],
            "binding_evidence_refs": [f"native.terminal.{endpoint_index}"],
            "candidate": {
                "terminal_kind": "package_boundary",
                "endpoint_index": endpoint_index,
                "connection_standard": "grooved",
            },
        }
        relations.append(relation)
        attachments.append({
            "id": f"attachment.{endpoint_index}",
            "relation_type": "physical_terminal",
            "source_relation_ref": relation["id"],
            "segment_refs": ["segment.a"],
        })
    m4 = {
        "schema_version": "0.1.0", "layer": "mep_page_local_attribute_bindings",
        "document": document, "relations": relations,
    }
    m5b = {
        "schema_version": "0.1.0", "layer": "mep_cross_sheet_run_hypotheses",
        "document": document, "partial_2_5d_centreline_segments": [{
            "id": "partial.a", "route_target_ref": "composite.a",
            "projected_2d_length_m": 5.0,
            "reason": "elevation_reference_does_not_resolve_centreline_offset",
        }],
        "unresolved_vertical_spans": [],
    }
    m5a = {
        "schema_version": "0.1.0", "layer": "mep_bounded_local_3d_segments",
        "document": document,
        "m4_contract_ref": {"payload_sha256": canonical_sha256(m4)},
        "m5_contract_ref": {"payload_sha256": canonical_sha256(m5b)},
        "bounded_local_3d_segments": [{
            "id": "m5a.segment.a", "state": "accepted",
            "authority": {"bounded_local_segment_identity_established": True},
            "source_page_occurrence_refs": ["occurrence.a"],
            "source_route_composite_refs": ["composite.a"],
            "physical_envelope_dimension": {
                "kind": "drawing_derived_outer_width", "shape": "round",
                "representative_outer_width_m": 0.073,
                "nominal_size_used_as_physical_dimension": False,
            },
            "m1_metric_frame": {"id": "metric.page.1", "state": "accepted"},
            "elevation": {"basis": "centreline", "centreline_elevation_m": 2.5},
            "centreline_points_xyz_m": [[0.0, 0.0, 2.5], [3.0, 4.0, 2.5]],
        }],
    }
    segment = {
        "id": "segment.a", "page_refs": ["page.1"],
        "route_target_refs": ["composite.a"],
        "source_occurrence_refs": ["occurrence.a"],
        "semantic_overlay": {
            "system": {"state": "accepted", "accepted_relation_refs": ["system.1"],
                       "unresolved_relation_refs": [], "conflicting_relation_refs": [],
                       "observed_values": [{"kind": "heating_hot_water_supply"}]},
            "size": {"state": "accepted", "accepted_relation_refs": ["size.1"],
                     "unresolved_relation_refs": [], "conflicting_relation_refs": [],
                     "observed_values": [{"nominal_size": "NPS_2_1_2"}]},
        },
    }
    run = {
        "id": "run.a", "segment_refs": ["segment.a"], "page_refs": ["page.1"],
        "junction_refs": [], "uncovered_segment_refs": [],
        "complete_trace_established": complete,
    }
    if not complete:
        run.update({
            "uncovered_segment_refs": ["segment.a"],
            "unresolved_trace_reasons": ["source_corridor_search_not_certified"],
        })
        attachments.clear()
        m5a["bounded_local_3d_segments"].clear()
        segment["semantic_overlay"] = {
            key: {"state": "unknown", "accepted_relation_refs": [],
                  "unresolved_relation_refs": [], "conflicting_relation_refs": [],
                  "observed_values": []}
            for key in ("system", "size")
        }
        run["readiness_source_search"] = {
            "connection_standard": {
                "state": "complete", "method": "native_notes_and_callouts_search",
                "evidence_refs": ["search.connection-standard.1"],
                "negative_reason": "connection_standard_apparently_absent_from_reviewed_scope",
            }
        }
    m5c = {
        "schema_version": "0.1.0", "layer": "mep_projected_network_hierarchy",
        "document": document,
        "input_payload_sha256": {"m4": canonical_sha256(m4), "m5b": canonical_sha256(m5b)},
        "segments": [segment], "runs": [run], "junctions": [], "attachments": attachments,
    }
    return m4, m5a, m5b, m5c


def build(values):
    m4, m5a, m5b, m5c = values
    return build_mep_physical_run_readiness(
        attribute_bindings=m4, bounded_local_3d=m5a,
        cross_sheet_runs=m5b, network_hierarchy=m5c,
    )


class MepPhysicalRunEvidenceTest(unittest.TestCase):
    def test_complete_single_segment_run_emits_automatic_evidence(self):
        source = inputs()
        payload = build(source)
        self.assertEqual(validate_mep_physical_run_readiness(payload), [])
        self.assertEqual(payload["summary"]["automatic_physical_run_evidence_count"], 1)
        self.assertEqual(payload["summary"]["runs_with_projected_length_observation_count"], 1)
        row = payload["runs"][0]
        self.assertEqual(row["projected_length_observations"][0]["projected_length_m"], 5.0)
        self.assertTrue(row["physical_evidence_ready"])
        self.assertTrue(all(row["field_outcomes"][field]["state"] == "closed"
                            for field in FIELDS if field != "material_specification"))
        evidence = payload["automatic_physical_run_evidence"][0]
        self.assertEqual(evidence["centreline_segments"][0]["points_xyz_m"],
                         [[0.0, 0.0, 2.5], [3.0, 4.0, 2.5]])
        self.assertEqual(evidence["accounted_network_segment_refs"], ["segment.a"])
        marketplace = build_mep_marketplace_assemblies(
            network_hierarchy=source[3], physical_run_evidence=[evidence]
        )
        self.assertEqual(marketplace["summary"]["physical_run_count"], 1)
        self.assertEqual(marketplace["summary"]["marketplace_ready_count"], 0)
        self.assertEqual(marketplace["summary"]["approved_for_quote_count"], 0)

    def test_missing_connection_standard_preserves_closed_geometry(self):
        source = inputs()
        for relation in source[0]["relations"]:
            relation["candidate"]["connection_standard"] = None
        source[3]["input_payload_sha256"]["m4"] = canonical_sha256(source[0])
        source[1]["m4_contract_ref"]["payload_sha256"] = canonical_sha256(source[0])
        source[3]["runs"][0]["readiness_source_search"] = {
            "connection_standard": {
                "state": "complete", "method": "registered_notes_details_search",
                "evidence_refs": ["search.connection-standard.none"],
                "negative_reason": "connection_standard_apparently_absent_from_reviewed_scope",
            }
        }
        payload = build(source)
        row = payload["runs"][0]
        self.assertTrue(row["physical_evidence_ready"])
        self.assertIn("connection_standard", row["marketplace_blocking_fields"])
        self.assertEqual(row["field_outcomes"]["connection_standard"]["state"],
                         "apparently_absent")
        evidence = payload["automatic_physical_run_evidence"][0]
        self.assertTrue(all(terminal["connection_standard"] is None
                            for terminal in evidence["terminals"]))
        marketplace = build_mep_marketplace_assemblies(
            network_hierarchy=source[3], physical_run_evidence=[evidence]
        )
        assembly = marketplace["assemblies"][0]
        self.assertTrue(assembly["physical_run_established"])
        self.assertEqual(assembly["length_channels"]["resolved_centreline_length"]
                         ["calculated"]["value"], 5.0)
        self.assertFalse(assembly["marketplace_ready"])
        self.assertIn("connection_standard_unresolved",
                      assembly["uncertainty"]["requirements"])

    def test_incomplete_run_reports_field_specific_abstention(self):
        payload = build(inputs(complete=False))
        row = payload["runs"][0]
        self.assertFalse(row["physical_evidence_ready"])
        self.assertEqual(payload["automatic_physical_run_evidence"], [])
        self.assertEqual(row["field_outcomes"]["complete_projected_topology"]["state"],
                         "not_searched")
        self.assertEqual(row["field_outcomes"]["system_service"]["state"], "not_searched")
        self.assertEqual(row["field_outcomes"]["terminal_identities"]["state"], "not_searched")
        self.assertEqual(row["field_outcomes"]["connection_standard"]["state"],
                         "apparently_absent")
        self.assertIn("complete_projected_topology", row["abstention"]["blocking_fields"])
        self.assertIn("terminal_identity_relations_not_available", row["abstention"]["reasons"])

    def test_unresolved_endpoint_competitors_are_ambiguous_not_lost(self):
        source = inputs(complete=False)
        candidate = {
            "id": "junction.candidate", "state": "abstained",
            "segment_refs": ["segment.a", "segment.other"],
            "physical_continuation_established": False,
            "semantic_conflicts": [],
            "reasons": ["native_boundary_branch_or_missing_trace"],
        }
        source[3]["junctions"] = [candidate]
        source[3]["runs"][0].update({
            "endpoint_interface_candidate_refs": [candidate["id"]],
            "unresolved_endpoint_interface_refs": [candidate["id"]],
            "endpoint_interface_search_complete": False,
            "projected_endpoint_occurrences": [
                {
                    "segment_ref": "segment.a", "composite_ref": "composite.a",
                    "page_ref": "page.1", "composite_endpoint_index": 0,
                    "interface_candidate_refs": [candidate["id"]],
                    "complete_candidate_search_refs": ["native.query.0"],
                    "search_state": "ambiguous",
                },
                {
                    "segment_ref": "segment.a", "composite_ref": "composite.a",
                    "page_ref": "page.1", "composite_endpoint_index": 1,
                    "interface_candidate_refs": [],
                    "complete_candidate_search_refs": [],
                    "search_state": "not_searched",
                },
            ],
        })
        payload = build(source)
        row = payload["runs"][0]
        terminal = row["field_outcomes"]["terminal_identities"]
        ports = row["field_outcomes"]["fitting_equipment_ports"]
        self.assertEqual(terminal["state"], "ambiguous")
        self.assertIn("endpoint_interface_search_incomplete", terminal["reasons"])
        self.assertIn("junction.candidate", terminal["evidence_refs"])
        self.assertEqual(ports["state"], "ambiguous")
        self.assertIn("native_boundary_branch_or_missing_trace", ports["reasons"])
        self.assertFalse(row["physical_evidence_ready"])

    def test_trace_without_complete_endpoint_search_does_not_close_topology(self):
        source = inputs()
        source[3]["runs"][0].update({
            "endpoint_interface_search_complete": False,
            "endpoint_interface_candidate_refs": ["junction.candidate"],
            "projected_endpoint_occurrences": [{
                "segment_ref": "segment.a",
                "composite_ref": "composite.a",
                "page_ref": "page.1",
                "composite_endpoint_index": 0,
                "interface_candidate_refs": ["junction.candidate"],
                "complete_candidate_search_refs": ["search.lower"],
                "search_state": "ambiguous",
            }, {
                "segment_ref": "segment.a",
                "composite_ref": "composite.a",
                "page_ref": "page.1",
                "composite_endpoint_index": 1,
                "interface_candidate_refs": [],
                "complete_candidate_search_refs": [],
                "search_state": "not_searched",
            }],
        })
        source[3]["junctions"] = [{
            "id": "junction.candidate",
            "state": "abstained",
            "segment_refs": ["segment.a", "segment.other"],
            "physical_continuation_established": False,
            "semantic_conflicts": [],
            "reasons": ["native_boundary_branch_or_missing_trace"],
        }]

        row = build(source)["runs"][0]

        self.assertEqual(
            row["field_outcomes"]["complete_projected_topology"]["state"],
            "not_searched",
        )
        self.assertIn(
            "endpoint_interface_search_incomplete",
            row["field_outcomes"]["complete_projected_topology"]["reasons"],
        )
        self.assertFalse(row["physical_evidence_ready"])

    def test_mismatched_frozen_input_hashes_fail_before_compilation(self):
        m4, m5a, m5b, m5c = inputs()
        m5c = copy.deepcopy(m5c)
        m5c["input_payload_sha256"]["m4"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "M5C does not bind"):
            build((m4, m5a, m5b, m5c))

    def test_validator_rejects_readiness_authority_escalation(self):
        payload = build(inputs())
        changed = copy.deepcopy(payload)
        changed["runs"][0]["field_outcomes"]["system_service"]["authority"][
            "physical_run_established"
        ] = True
        self.assertTrue(validate_mep_physical_run_readiness(changed))
        changed = copy.deepcopy(payload)
        changed["automatic_physical_run_evidence"][0][
            "accounted_network_segment_refs"
        ] = ["segment.other"]
        self.assertTrue(validate_mep_physical_run_readiness(changed))


if __name__ == "__main__":
    unittest.main()
