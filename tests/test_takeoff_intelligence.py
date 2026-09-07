import copy
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from tools.generate_shared_takeoff_checkpoint import generate
import src.drawing_engine.disciplines.concrete.concrete_takeoff_adapter as concrete_module
import src.drawing_engine.disciplines.mep.mep_takeoff_adapter as mep_module
import src.drawing_engine.project.takeoff_intelligence as takeoff_module
from src.drawing_engine.disciplines.concrete.concrete_takeoff_adapter import build_concrete_takeoff_adapter
from src.drawing_engine.disciplines.mep.mep_takeoff_adapter import build_mep_takeoff_adapter
from src.drawing_engine.disciplines.rebar.rebar_takeoff_adapter import build_rebar_takeoff_adapter
from src.drawing_engine.project.takeoff_intelligence import (
    build_takeoff_intelligence_catalog,
    create_takeoff_approval,
    validate_takeoff_adapter,
    validate_takeoff_intelligence_catalog,
)


ROOT = Path(__file__).resolve().parents[1]
ENGINEERING = ROOT / "output" / "object_agnostic" / "candidate-08-staircase-page.engineering-graph.json"
MEP = ROOT / "fixtures" / "mep" / "m_and_p_coordination" / "m_and_p_coordination.mep-item-catalog.json"
REBAR_POSITIVE = ROOT / "output" / "object_agnostic" / "3179 ЛС (2)-2.engineering-graph.json"
FROZEN = ROOT / "fixtures" / "takeoff_intelligence" / "step1c_shared"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _synthetic_adapter(*, calculated=1.0, declared=1.0):
    document_key = "pdf-sha256:" + "a" * 64
    occurrence = {
        "record_type": "takeoff_occurrence",
        "record_version": "0.1.0",
        "id": "takeoff_occurrence.synthetic",
        "discipline": "concrete",
        "document_key": document_key,
        "item_type": "concrete",
        "item_subtype": "synthetic",
        "description": "Synthetic scoped concrete",
        "source": {"pdf_page_number": 1, "page_ref": "page.1", "drawing_sheet_number": None},
        "native_record_refs": ["native.occurrence"],
        "evidence_refs": ["drawing.geometry.1"],
        "epistemic_state": "derived",
        "unresolved_reasons": [],
    }
    physical = {
        "record_type": "takeoff_physical_item",
        "record_version": "0.1.0",
        "id": "takeoff_physical_item.synthetic",
        "discipline": "concrete",
        "document_key": document_key,
        "occurrence_refs": [occurrence["id"]],
        "native_record_refs": ["native.physical"],
        "evidence_refs": ["drawing.geometry.1"],
        "epistemic_state": "derived",
    }
    calculated_line = {
        "record_type": "takeoff_calculated_line",
        "record_version": "0.1.0",
        "id": "takeoff_calculated_line.synthetic",
        "discipline": "concrete",
        "document_key": document_key,
        "physical_item_ref": physical["id"],
        "item_type": "concrete",
        "metric_kind": "net_volume",
        "value": calculated,
        "unit": "m3",
        "scope_ref": "scope.synthetic",
        "native_record_refs": ["native.calculated"],
        "evidence_refs": ["drawing.geometry.1"],
        "epistemic_state": "calculated",
    }
    declared_line = {
        "record_type": "takeoff_declared_line",
        "record_version": "0.1.0",
        "id": "takeoff_declared_line.synthetic",
        "discipline": "concrete",
        "document_key": document_key,
        "occurrence_ref": occurrence["id"],
        "item_type": "concrete",
        "metric_kind": "net_volume",
        "value": declared,
        "unit": "m3",
        "scope_ref": "scope.synthetic",
        "native_record_refs": ["native.declared"],
        "evidence_refs": ["schedule.cell.1"],
        "epistemic_state": "declared",
    }
    return {
        "schema_version": "0.1.0",
        "layer": "takeoff_intelligence_adapter",
        "adapter": {"name": "synthetic", "version": "1", "discipline": "concrete"},
        "native_payload_ref": {"payload_sha256": "b" * 64},
        "native_records": [],
        "native_records_preserved_unchanged": True,
        "occurrences": [occurrence],
        "physical_items": [physical],
        "calculated_lines": [calculated_line],
        "declared_lines": [declared_line],
        "contract": {
            "calculated_and_declared_channels_separate": True,
            "missing_values_not_substituted_with_zero": True,
            "approval_not_inferred": True,
        },
    }


def _certificate():
    return {
        "id": "takeoff_match_certificate.synthetic",
        "state": "accepted",
        "calculated_line_ref": "takeoff_calculated_line.synthetic",
        "declared_line_ref": "takeoff_declared_line.synthetic",
        "absolute_tolerance": 0.001,
        "evidence_refs": ["reviewed.scope.identity.1"],
    }


class TakeoffIntelligenceTest(unittest.TestCase):
    def test_candidate_08_adapts_one_physical_object_and_calculated_volume(self):
        payload = build_concrete_takeoff_adapter(_load(ENGINEERING))
        self.assertEqual(validate_takeoff_adapter(payload), [])
        self.assertEqual(len(payload["occurrences"]), 1)
        self.assertEqual(len(payload["physical_items"]), 1)
        self.assertEqual(len(payload["calculated_lines"]), 1)
        line = payload["calculated_lines"][0]
        self.assertEqual(line["metric_kind"], "net_volume")
        self.assertEqual(line["unit"], "m3")
        self.assertAlmostEqual(line["value"], 1.7726147823710516)
        self.assertEqual(payload["declared_lines"], [])
        self.assertFalse(payload["contract"]["schedule_values_used"])

    def test_concrete_adapter_fails_on_quantity_or_object_authority_mutation(self):
        baseline = _load(ENGINEERING)
        changed = copy.deepcopy(baseline)
        changed["pages"][0]["physical_object_quantity_promotion"]["quantity_eligible"] = False
        with self.assertRaisesRegex(ValueError, "authority contract"):
            build_concrete_takeoff_adapter(changed)
        changed = copy.deepcopy(baseline)
        changed["pages"][0]["physical_object_quantity_promotion"]["calculated_concrete_quantities"][0]["construction_regions_aggregated_separately"] = True
        with self.assertRaisesRegex(ValueError, "construction regions"):
            build_concrete_takeoff_adapter(changed)

    def test_mep_adapter_preserves_native_m7a_records_and_zero_value_channels(self):
        source = _load(MEP)
        payload = build_mep_takeoff_adapter(source)
        self.assertEqual(validate_takeoff_adapter(payload), [])
        self.assertEqual(payload["native_records"]["item_occurrences"], source["item_occurrences"])
        self.assertEqual(payload["native_records"]["takeoff_lines"], source["takeoff_lines"])
        self.assertEqual(len(payload["occurrences"]), 4)
        self.assertEqual(payload["physical_items"], [])
        self.assertEqual(payload["calculated_lines"], [])
        self.assertEqual(payload["declared_lines"], [])

    def test_empty_mep_channels_do_not_activate_comparison(self):
        catalog = build_takeoff_intelligence_catalog(
            adapters=[build_mep_takeoff_adapter(_load(MEP))]
        )
        self.assertEqual(catalog["comparisons"], [])
        self.assertEqual(
            catalog["activation"]["state"],
            "inactive_no_scoped_calculated_or_declared_lines",
        )

    def test_shared_checkpoint_activates_only_scoped_calculated_lines(self):
        catalog = build_takeoff_intelligence_catalog(adapters=[
            build_concrete_takeoff_adapter(_load(ENGINEERING)),
            build_rebar_takeoff_adapter(_load(ENGINEERING)),
            build_rebar_takeoff_adapter(_load(REBAR_POSITIVE)),
            build_mep_takeoff_adapter(_load(MEP)),
        ])
        self.assertEqual(validate_takeoff_intelligence_catalog(catalog), [])
        self.assertEqual(catalog["summary"], {
            "discipline_counts": {"concrete": 1, "mep": 4, "rebar": 21},
            "occurrence_count": 26,
            "physical_item_count": 13,
            "calculated_line_count": 25,
            "declared_line_count": 0,
            "comparison_status_counts": {"calculated_unmatched": 25},
            "approval_count": 0,
        })
        self.assertTrue(
            all(row["status"] == "calculated_unmatched" for row in catalog["comparisons"])
        )
        self.assertTrue(all(row["delta"] is None for row in catalog["comparisons"]))

    def test_explicit_scope_certificate_enables_match_and_discrepancy(self):
        matched = build_takeoff_intelligence_catalog(
            adapters=[_synthetic_adapter(calculated=2.0, declared=2.0005)],
            match_certificates=[_certificate()],
        )
        self.assertEqual(matched["comparisons"][0]["status"], "match")
        contradictory = build_takeoff_intelligence_catalog(
            adapters=[_synthetic_adapter(calculated=2.0, declared=2.4)],
            match_certificates=[_certificate()],
        )
        self.assertEqual(contradictory["comparisons"][0]["status"], "discrepancy")
        self.assertAlmostEqual(contradictory["comparisons"][0]["delta"], -0.4)
        tampered = copy.deepcopy(contradictory)
        tampered["comparisons"][0]["calculated_value"] = 9.0
        self.assertTrue(any(
            "matched values mismatch" in error
            for error in validate_takeoff_intelligence_catalog(tampered)
        ))

    def test_no_scope_certificate_preserves_both_lines_as_unmatched(self):
        catalog = build_takeoff_intelligence_catalog(
            adapters=[_synthetic_adapter(calculated=2.0, declared=2.0)]
        )
        self.assertEqual(
            {row["status"] for row in catalog["comparisons"]},
            {"calculated_unmatched", "declared_unmatched"},
        )
        self.assertTrue(all(row["delta"] is None for row in catalog["comparisons"]))

    def test_match_pairing_must_be_mutually_unique(self):
        adapter = _synthetic_adapter()
        duplicate = copy.deepcopy(adapter["declared_lines"][0])
        duplicate["id"] = "takeoff_declared_line.second"
        adapter["declared_lines"].append(duplicate)
        certificate = _certificate()
        second = copy.deepcopy(certificate)
        second["id"] = "takeoff_match_certificate.second"
        second["declared_line_ref"] = duplicate["id"]
        with self.assertRaisesRegex(ValueError, "mutually unique"):
            build_takeoff_intelligence_catalog(
                adapters=[adapter], match_certificates=[certificate, second]
            )

    def test_approval_is_exact_overlay_and_never_inferred(self):
        base = build_takeoff_intelligence_catalog(adapters=[_synthetic_adapter()])
        approval = create_takeoff_approval(
            base,
            calculated_line_ref="takeoff_calculated_line.synthetic",
            engineer="engineer.1",
            approved_at="2026-08-29T12:00:00+03:00",
            reason="Reviewed drawing-derived physical scope",
            evidence_refs=["review.record.1"],
        )
        catalog = build_takeoff_intelligence_catalog(
            adapters=[_synthetic_adapter()], approval_records=[approval]
        )
        self.assertEqual(validate_takeoff_intelligence_catalog(catalog), [])
        changed = copy.deepcopy(catalog)
        changed["approval_records"][0]["value"] = 99
        self.assertTrue(validate_takeoff_intelligence_catalog(changed))
        changed = copy.deepcopy(catalog)
        changed["approval_records"][0]["comparison_ref"] = None
        self.assertTrue(any(
            "approval comparison ref is invalid" in error
            for error in validate_takeoff_intelligence_catalog(changed)
        ))

    def test_frozen_checkpoint_and_generator_replay_exactly(self):
        frozen = _load(FROZEN / "shared-takeoff-catalog.json")
        self.assertEqual(validate_takeoff_intelligence_catalog(frozen), [])
        with tempfile.TemporaryDirectory() as directory:
            paths = generate(output_dir=Path(directory))
            for name, path in paths.items():
                self.assertEqual(path.read_bytes(), (FROZEN / name).read_bytes())

    def test_production_adapters_have_no_fixture_dispatch(self):
        for module in (takeoff_module, concrete_module, mep_module):
            source = inspect.getsource(module)
            for forbidden in ("candidate-08", "m_and_p_coordination", "1A", "1B"):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
