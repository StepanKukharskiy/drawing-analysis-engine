import copy
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from tools.generate_mep_m7b_fixture import generate
import src.drawing_engine.disciplines.mep.mep_bounded_discrete_counts as count_module
from src.drawing_engine.disciplines.mep.mep_attribute_binding import build_mep_attribute_bindings
from src.drawing_engine.disciplines.mep.mep_bounded_discrete_counts import (
    build_mep_bounded_discrete_counts,
    validate_mep_bounded_discrete_counts,
)
from src.drawing_engine.disciplines.mep.mep_item_catalog import build_mep_item_catalog, flatten_mep_item_catalog
from src.drawing_engine.disciplines.mep.mep_occurrence_coverage_audit import build_mep_occurrence_coverage_audit
from src.drawing_engine.disciplines.mep.mep_route_observations import build_mep_route_graph
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import build_mep_terminology_proposals


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "fixtures" / "mep" / "m_and_p_coordination"
CHECKPOINT_ROOT = FIXTURE_ROOT / "real_m2_m5_checkpoint"
M4_COVERAGE_ROOT = FIXTURE_ROOT / "real_m4_discrete_coverage"
FROZEN_AUDIT = FIXTURE_ROOT / "m_and_p_coordination.occurrence-coverage-audit.json"
FROZEN_COUNTS = FIXTURE_ROOT / "m_and_p_coordination.bounded-discrete-counts.json"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _registry():
    return _load(FIXTURE_ROOT / "m_and_p_coordination.sheet-registry.json")


def _coverage(bindings):
    return build_mep_occurrence_coverage_audit(
        sheet_registry=_registry(),
        attribute_bindings=bindings,
        annotation_observations=_load(FIXTURE_ROOT / "m_and_p_coordination.annotation-observations.json"),
        reviewed_coverage=_load(FIXTURE_ROOT / "m_and_p_coordination_m7b_coverage_truth.json"),
    )


def _segment(index, y=0):
    source_ref = f"drawing[{index}].item[0].segment[0]"
    return {
        "id": source_ref,
        "drawing_ref": f"drawing[{index}]",
        "primitive_ref": f"drawing[{index}].item[0]",
        "item_index": 0,
        "part_index": 0,
        "kind": "line",
        "start_display": [0, y],
        "end_display": [20, y],
        "control_points_display": [],
        "sample_points_display": [],
        "style": {"stroke": [0.0, 0.0, 0.0], "fill": None, "width": 1.0, "dash": None},
    }


def _discrete_bindings(page_refs):
    registry = _registry()
    page_inputs = {
        page_ref: {
            "native_topology": {
                "schema_version": "0.1.0",
                "segments": [_segment(9000 + index, index * 10)],
                "vertices": [],
            }
        }
        for index, page_ref in enumerate(page_refs)
    }
    graph = build_mep_route_graph(sheet_registry=registry, page_inputs=page_inputs)
    observations = [
        {
            "id": f"valve.observation.{index}",
            "page_ref": page_ref,
            "text": "BALL VALVE",
            "evidence_channels": ["native_pdf_text"],
            "interpretation_scope_ref": f"valve.scope.{index}",
        }
        for index, page_ref in enumerate(page_refs)
    ]
    terminology = build_mep_terminology_proposals(
        document=registry["document"], observations=observations
    )
    bindings = []
    for index, page_ref in enumerate(page_refs):
        proposal = next(
            row for row in terminology["proposals"]
            if f"valve.observation.{index}" in row["evidence_refs"]
        )
        page = next(row for row in graph["pages"] if row["page_ref"] == page_ref)
        fragment = page["fragments"][0]
        vertex_ref = fragment["endpoint_vertex_refs"][0]
        bindings.append({
            "id": f"valve.binding.{index}",
            "proposal_ref": proposal["id"],
            "page_ref": page_ref,
            "state": "observed",
            "method": {"name": "unique_native_symbol_contact", "version": "1.0.0"},
            "target_kind": "route_vertex",
            "target_refs": [vertex_ref],
            "geometric_evidence_refs": [vertex_ref, fragment["id"]],
        })
    return build_mep_attribute_bindings(
        terminology_proposals=terminology,
        route_graph=graph,
        binding_evidence=bindings,
    )


def _identity_evidence(bindings, registration_refs=()):
    relation_refs = [row["id"] for row in bindings["relations"] if row["state"] == "accepted"]
    pages = {row["page_ref"] for row in bindings["relations"] if row["state"] == "accepted"}
    return [{
        "id": "discrete.identity.evidence.001",
        "state": "observed",
        "m4_relation_refs": relation_refs,
        "canonicalization": {
            "duplicate_projection": len(relation_refs) > 1,
            "mutually_unique": True,
            "target_signatures_match": True,
            "no_alternative_pairing": True,
        },
        "physical_identity": {
            "unique_physical_target": True,
            "repetition_resolved": True,
            "cross_sheet_identity": "accepted_same_physical_item" if len(pages) > 1 else "not_applicable",
            "accepted_registration_refs": list(registration_refs),
        },
        "evidence_refs": ["reviewed.identity.signature.001"],
    }]


class MepBoundedDiscreteCountsTest(unittest.TestCase):
    def test_real_coverage_limited_fixture_emits_no_discrete_count(self):
        bindings = _load(M4_COVERAGE_ROOT / "full_package.attribute-bindings.json")
        payload = build_mep_bounded_discrete_counts(
            sheet_registry=_registry(),
            attribute_bindings=bindings,
            occurrence_coverage_audit=_coverage(bindings),
        )
        self.assertEqual(validate_mep_bounded_discrete_counts(payload), [])
        self.assertEqual(payload["summary"], {
            "observed_discrete_occurrence_count": 0,
            "accepted_identity_certificate_count": 0,
            "deduplicated_projected_item_count": 0,
            "physical_item_count": 0,
            "calculated_count_record_count": 0,
            "calculated_count_total": 0,
            "identity_abstention_count": 0,
            "coverage_abstention_count": 11,
        })

    def test_frozen_real_result_and_generator_replay_exactly(self):
        bindings = _load(M4_COVERAGE_ROOT / "full_package.attribute-bindings.json")
        frozen = _load(FROZEN_COUNTS)
        self.assertEqual(validate_mep_bounded_discrete_counts(frozen), [])
        self.assertEqual(
            build_mep_bounded_discrete_counts(
                sheet_registry=_registry(),
                attribute_bindings=bindings,
                occurrence_coverage_audit=_load(FROZEN_AUDIT),
            ),
            frozen,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit_path = root / "audit.json"
            counts_path = root / "counts.json"
            generate(
                m0_path=FIXTURE_ROOT / "m_and_p_coordination.annotation-observations.json",
                m1_path=FIXTURE_ROOT / "m_and_p_coordination.sheet-registry.json",
                m4_path=M4_COVERAGE_ROOT / "full_package.attribute-bindings.json",
                target_review_path=M4_COVERAGE_ROOT / "full_package.discrete-target-review.json",
                truth_path=FIXTURE_ROOT / "m_and_p_coordination_m7b_coverage_truth.json",
                audit_path=audit_path,
                counts_path=counts_path,
            )
            self.assertEqual(audit_path.read_bytes(), FROZEN_AUDIT.read_bytes())
            self.assertEqual(counts_path.read_bytes(), FROZEN_COUNTS.read_bytes())

    def test_unique_page_local_valve_identity_emits_one_calculated_count(self):
        page_ref = _registry()["pages"][1]["page_ref"]
        bindings = _discrete_bindings([page_ref])
        payload = build_mep_bounded_discrete_counts(
            sheet_registry=_registry(),
            attribute_bindings=bindings,
            occurrence_coverage_audit=_coverage(bindings),
            identity_evidence=_identity_evidence(bindings),
        )
        self.assertEqual(validate_mep_bounded_discrete_counts(payload), [])
        self.assertEqual(payload["summary"]["observed_discrete_occurrence_count"], 1)
        self.assertEqual(payload["summary"]["deduplicated_projected_item_count"], 1)
        self.assertEqual(payload["summary"]["physical_item_count"], 1)
        self.assertEqual(payload["summary"]["calculated_count_total"], 1)
        self.assertEqual(payload["calculated_count_records"][0]["unit"], "ea")
        catalog = build_mep_item_catalog(
            sheet_registry=_registry(),
            attribute_bindings=bindings,
            occurrence_coverage_audit=_coverage(bindings),
            bounded_discrete_counts=payload,
        )
        self.assertEqual(catalog["summary"]["physical_item_count"], 1)
        self.assertEqual(catalog["summary"]["calculated_value_count"], 1)
        self.assertEqual(catalog["calculated_discrete_counts"][0]["value"], 1)
        physical_rows = [
            row for row in flatten_mep_item_catalog(catalog)
            if row["row_type"] == "physical_item"
        ]
        self.assertEqual(len(physical_rows), 1)
        self.assertEqual(physical_rows[0]["calculated_value"], 1)
        self.assertEqual(physical_rows[0]["calculated_unit"], "ea")

    def test_mutually_unique_cross_sheet_duplicates_count_once(self):
        registry = _registry()
        registration = next(
            row for row in registry["adjoining_sheet_transforms"] if row["state"] == "accepted"
        )
        page_refs = [registration["source_page_ref"], registration["target_page_ref"]]
        bindings = _discrete_bindings(page_refs)
        payload = build_mep_bounded_discrete_counts(
            sheet_registry=registry,
            attribute_bindings=bindings,
            occurrence_coverage_audit=_coverage(bindings),
            identity_evidence=_identity_evidence(bindings, [registration["id"]]),
        )
        self.assertEqual(validate_mep_bounded_discrete_counts(payload), [])
        certificate = payload["identity_certificates"][0]
        self.assertEqual(certificate["counts"], {
            "observed_occurrence_count": 2,
            "deduplicated_projected_count": 1,
            "physical_instance_count": 1,
        })
        self.assertEqual(payload["summary"]["calculated_count_total"], 1)

    def test_missing_registration_or_repetition_closure_abstains(self):
        registry = _registry()
        registration = next(
            row for row in registry["adjoining_sheet_transforms"] if row["state"] == "accepted"
        )
        bindings = _discrete_bindings([registration["source_page_ref"], registration["target_page_ref"]])
        evidence = _identity_evidence(bindings)
        evidence[0]["physical_identity"]["repetition_resolved"] = False
        payload = build_mep_bounded_discrete_counts(
            sheet_registry=registry,
            attribute_bindings=bindings,
            occurrence_coverage_audit=_coverage(bindings),
            identity_evidence=evidence,
        )
        self.assertEqual(payload["physical_items"], [])
        self.assertEqual(payload["calculated_count_records"], [])
        reasons = payload["abstentions"][0]["reasons"]
        self.assertIn("ambiguous_repetition", reasons)
        self.assertIn("accepted_cross_sheet_registration_missing", reasons)

    def test_no_identity_evidence_keeps_count_channels_null(self):
        page_ref = _registry()["pages"][1]["page_ref"]
        bindings = _discrete_bindings([page_ref])
        payload = build_mep_bounded_discrete_counts(
            sheet_registry=_registry(),
            attribute_bindings=bindings,
            occurrence_coverage_audit=_coverage(bindings),
        )
        self.assertEqual(payload["summary"]["identity_abstention_count"], 1)
        abstention = payload["abstentions"][0]
        self.assertEqual(abstention["reasons"], ["physical_identity_evidence_missing"])
        self.assertIsNone(abstention["calculated_count"])

    def test_validator_rejects_length_or_unbacked_count_promotion(self):
        page_ref = _registry()["pages"][1]["page_ref"]
        bindings = _discrete_bindings([page_ref])
        payload = build_mep_bounded_discrete_counts(
            sheet_registry=_registry(),
            attribute_bindings=bindings,
            occurrence_coverage_audit=_coverage(bindings),
        )
        changed = copy.deepcopy(payload)
        changed["abstentions"][0]["calculated_count"] = 1
        self.assertTrue(validate_mep_bounded_discrete_counts(changed))
        changed = copy.deepcopy(payload)
        changed["installed_length_m"] = 12.0
        self.assertTrue(validate_mep_bounded_discrete_counts(changed))
        changed = copy.deepcopy(payload)
        changed["identity_evidence_sha256"] = "0" * 64
        self.assertTrue(validate_mep_bounded_discrete_counts(changed))
        changed = copy.deepcopy(payload)
        changed["coverage_abstentions"][0]["item_occurrence_established"] = True
        self.assertTrue(validate_mep_bounded_discrete_counts(changed))

    def test_production_source_has_no_fixture_or_item_name_dispatch(self):
        source = inspect.getsource(count_module)
        for forbidden in ("m_and_p_coordination", "L01-MP", "BALL VALVE", "AHU-1", "VFD"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
