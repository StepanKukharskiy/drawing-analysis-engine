import copy
import json
from pathlib import Path
import unittest

from src.drawing_engine.disciplines.mep.mep_marketplace_assembly import (
    _port_graph,
    _takeout_total,
    build_mep_marketplace_assemblies,
    projected_length_observations_from_m5c,
    validate_mep_marketplace_assemblies,
)
from src.drawing_engine.project.takeoff_intelligence import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]


def network():
    return {
        "schema_version": "0.2.0",
        "layer": "mep_projected_network_hierarchy",
        "document": {"document_key": "pdf-sha256:test", "page_count": 2},
        "runs": [{
            "id": "run.hhws", "record_type": "mep_projected_run",
            "segment_refs": ["segment.a"], "page_refs": ["page.1"],
            "complete_trace_established": True, "quantity_eligible": False,
        }],
        "segments": [{"id": "segment.a", "page_refs": ["page.1"],
                      "source_occurrence_refs": ["occurrence.a"]}],
        "quantity_eligible": False,
    }


def evidence(source):
    return {
        "id": "physical-run-evidence.hhws", "state": "observed",
        "network_run_ref": "run.hhws",
        "network_payload_sha256": canonical_sha256(source),
        "accounted_network_segment_refs": ["segment.a"],
        "accounted_projected_occurrence_refs": ["occurrence.a"],
        "system": {"kind": "heating_hot_water_supply", "service": "HHWS"},
        "source": {"page_refs": ["page.1"], "detail_refs": ["detail.4"], "section_refs": []},
        "material": {"material": "carbon_steel", "specification": "ASTM_A53_schedule_40"},
        "cross_section": {
            "shape": "round", "nominal_size": "NPS_2_1_2",
            "physical_dimensions": {"outside_diameter_m": 0.073},
        },
        "terminals": [
            {"id": "terminal.start", "port_ref": "port.start",
             "terminal_kind": "package_boundary", "connection_standard": "grooved",
             "evidence_refs": ["detail.4"]},
            {"id": "terminal.end", "port_ref": "port.end",
             "terminal_kind": "equipment_port", "connection_standard": "grooved",
             "evidence_refs": ["equipment.port.1"]},
        ],
        "centreline_segments": [{
            "id": "centreline.a", "network_segment_ref": "segment.a",
            "port_refs": ["port.start", "port.end"],
            "points_xyz_m": [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]],
            "evidence_refs": ["segment.a", "elevation.1"],
        }],
        "components": [], "connections": [],
        "coverage": {
            "all_projected_occurrences_accounted_once": True,
            "reprojection_passed": True,
            "duplicate_occurrences_eliminated": True,
            "vertical_spans_resolved": True,
        },
        "evidence_refs": ["segment.a", "detail.4", "equipment.port.1", "elevation.1"],
    }


RULES = {
    "id": "hhw-assembly-rules", "version": "2026.1", "assembly_rules": [],
    "stock_material_rules": [{
        "id": "stock.pipe.6m",
        "match": {"material": "carbon_steel", "specification": "ASTM_A53_schedule_40",
                  "nominal_size": "NPS_2_1_2"},
        "stock_length_m": 6.0, "waste_fraction": 0.1,
    }],
}


CATALOG = {
    "id": "test-marketplace", "version": "2026.1",
    "products": [{
        "id": "pipe.product.1", "engineering_item_kind": "straight_material",
        "match": {"material": "carbon_steel", "specification": "ASTM_A53_schedule_40",
                  "nominal_size": "NPS_2_1_2", "connection_standard": "grooved"},
        "product_family": "grooved_schedule_40_pipe", "manufacturer": "Example",
        "model": "CS-250", "sku": "EX-CS-250-6M", "approved_substitute_refs": [],
    }],
}


class MepMarketplaceAssemblyTest(unittest.TestCase):
    def test_complete_physical_run_keeps_four_lengths_and_maps_exact_sku(self):
        source = network()
        result = build_mep_marketplace_assemblies(
            network_hierarchy=source, physical_run_evidence=[evidence(source)],
            projected_length_observations=[{
                "id": "projected.1", "network_run_ref": "run.hhws", "state": "observed",
                "value_m": 4.8, "measurement_scope_ref": "scope.plan.1",
                "evidence_refs": ["dimension.chain.1"],
            }], assembly_rule_pack=RULES, catalog_pack=CATALOG,
        )
        self.assertEqual(validate_mep_marketplace_assemblies(result), [])
        row = result["assemblies"][0]
        calculated = {key: value["calculated"]["value"]
                      for key, value in row["length_channels"].items()}
        self.assertEqual(calculated, {
            "projected_length": 4.8,
            "resolved_centreline_length": 5.0,
            "net_material_length": 5.0,
            "purchase_length": 6.0,
        })
        self.assertTrue(row["physical_run_established"])
        self.assertTrue(row["quantity_eligible"])
        self.assertTrue(row["marketplace_ready"])
        self.assertFalse(row["approved_for_quote"])
        self.assertEqual(row["material_items"][0]["catalog_mapping"]["sku"], "EX-CS-250-6M")
        self.assertFalse(source["quantity_eligible"])

    def test_projected_measurement_alone_never_becomes_installed_or_purchase_length(self):
        result = build_mep_marketplace_assemblies(
            network_hierarchy=network(),
            projected_length_observations=[{
                "id": "projected.long", "network_run_ref": "run.hhws", "state": "observed",
                "value_m": 12.202, "measurement_scope_ref": "bounded.trace",
                "evidence_refs": ["native.path.1"],
            }],
        )
        row = result["assemblies"][0]
        self.assertEqual(row["length_channels"]["projected_length"]["calculated"]["value"], 12.202)
        for key in ("resolved_centreline_length", "net_material_length", "purchase_length"):
            self.assertIsNone(row["length_channels"][key]["calculated"]["value"])
        self.assertIn("physical_run_evidence_missing", row["uncertainty"]["requirements"])
        self.assertFalse(row["quantity_eligible"])

    def test_m5c_subtrace_adapts_to_one_bounded_projected_channel(self):
        source = network()
        source["segments"][0]["route_target_refs"] = ["composite.a"]
        source["connected_trace_completion"] = {"complete_subtraces": [{
            "id": "trace.a", "state": "derived", "projected_scope_complete": True,
            "external_continuation_resolved": False,
            "composite_refs": ["composite.a"], "covered_junction_refs": ["junction.a"],
            "projected_length_m": 12.202,
        }]}
        observations = projected_length_observations_from_m5c(source)
        self.assertEqual(len(observations), 1)
        self.assertFalse(observations[0]["complete_run_coverage"])
        result = build_mep_marketplace_assemblies(
            network_hierarchy=source, projected_length_observations=observations
        )
        channel = result["assemblies"][0]["length_channels"]["projected_length"]["calculated"]
        self.assertEqual(channel["value"], 12.202)
        self.assertEqual(channel["measurement_scope"]["scope_ref"], "trace.a")
        self.assertFalse(channel["measurement_scope"]["complete_run_coverage"])

    def test_endpoint_coverage_and_catalog_gates_fail_closed(self):
        source = network()
        broken = evidence(source)
        broken["terminals"][1]["terminal_kind"] = "unresolved"
        broken["coverage"]["vertical_spans_resolved"] = False
        broken["accounted_projected_occurrence_refs"] = []
        result = build_mep_marketplace_assemblies(
            network_hierarchy=source, physical_run_evidence=[broken],
            assembly_rule_pack=RULES, catalog_pack={"id": "empty", "version": "1", "products": []},
        )
        row = result["assemblies"][0]
        self.assertIn("endpoint_classification_unresolved", row["uncertainty"]["requirements"])
        self.assertIn("coverage_vertical_spans_resolved_not_closed", row["uncertainty"]["requirements"])
        self.assertIn("projected_occurrences_not_accounted_exactly_once", row["uncertainty"]["requirements"])
        self.assertFalse(row["physical_run_established"])
        self.assertFalse(row["marketplace_ready"])

    def test_validator_rejects_channel_collapse_and_invented_quote_approval(self):
        source = network()
        payload = build_mep_marketplace_assemblies(
            network_hierarchy=source, physical_run_evidence=[evidence(source)],
            assembly_rule_pack=RULES, catalog_pack=CATALOG,
        )
        changed = copy.deepcopy(payload)
        changed["assemblies"][0]["length_channels"]["purchase_length"]["declared"] = {
            "value": 6.0, "unit": "m"
        }
        self.assertTrue(validate_mep_marketplace_assemblies(changed))
        changed = copy.deepcopy(payload)
        changed["assemblies"][0]["approved_for_quote"] = True
        self.assertTrue(validate_mep_marketplace_assemblies(changed))

    def test_port_cardinality_and_takeout_overlap_are_deterministic_blockers(self):
        graph_evidence = {
            "cross_section": {"nominal_size": "NPS_2_1_2"},
            "centreline_segments": [{"id": "line", "port_refs": ["a", "b"]}],
            "components": [{
                "id": "elbow", "component_type": "elbow",
                "ports": [{"id": "c", "nominal_size": "NPS_2_1_2"}],
            }],
            "connections": [],
            "terminals": [
                {"port_ref": "a", "terminal_kind": "package_boundary",
                 "connection_standard": "grooved", "evidence_refs": ["a"]},
                {"port_ref": "b", "terminal_kind": "package_boundary",
                 "connection_standard": "grooved", "evidence_refs": ["b"]},
                {"port_ref": "c", "terminal_kind": "fitting_port",
                 "connection_standard": "grooved", "evidence_refs": ["c"]},
            ],
        }
        _, reasons, _ = _port_graph(graph_evidence)
        self.assertIn("elbow_port_count_mismatch", reasons)
        total, reasons = _takeout_total([
            {"id": "left", "centreline_takeout_m": 2.0,
             "takeout_allocations": [{"centreline_segment_ref": "line", "start_m": 1.0, "end_m": 3.0}]},
            {"id": "right", "centreline_takeout_m": 2.0,
             "takeout_allocations": [{"centreline_segment_ref": "line", "start_m": 2.0, "end_m": 4.0}]},
        ], {"line": 5.0})
        self.assertIsNone(total)
        self.assertIn("component_takeout_allocations_overlap", reasons)

    def test_quote_approval_binds_the_exact_calculation(self):
        source = network()
        inputs = dict(
            network_hierarchy=source, physical_run_evidence=[evidence(source)],
            assembly_rule_pack=RULES, catalog_pack=CATALOG,
        )
        unreviewed = build_mep_marketplace_assemblies(**inputs)
        calculation_hash = unreviewed["assemblies"][0]["calculation_payload_sha256"]
        approved = build_mep_marketplace_assemblies(**inputs, review_decisions=[{
            "id": "review.1", "promotion_evidence_ref": "physical-run-evidence.hhws",
            "network_payload_sha256": canonical_sha256(source),
            "calculation_payload_sha256": calculation_hash,
            "decision": "approved", "engineer": "Engineer A",
            "reviewed_at": "2026-09-01T10:00:00+03:00", "reason": "Verified",
            "evidence_refs": ["review.sheet.1"],
        }])
        self.assertTrue(approved["assemblies"][0]["approved_for_quote"])
        stale = copy.deepcopy(approved)
        stale["assemblies"][0]["calculation_payload_sha256"] = "0" * 64
        # Direct output tampering does not change approval state, so consumers
        # must reject it through the validator.
        self.assertTrue(validate_mep_marketplace_assemblies(stale))

    def test_bounded_real_m5c_checkpoint_remains_quantity_negative(self):
        source = json.loads((ROOT / "output/mep-drawing-interpretation-project-2026-08-31/network-hierarchy.json").read_text())
        result = build_mep_marketplace_assemblies(network_hierarchy=source)
        self.assertEqual(result["summary"]["assembly_candidate_count"], len(source["runs"]))
        self.assertEqual(result["summary"]["physical_run_count"], 0)
        self.assertEqual(result["summary"]["quantity_eligible_count"], 0)
        self.assertEqual(result["summary"]["marketplace_ready_count"], 0)


if __name__ == "__main__":
    unittest.main()
