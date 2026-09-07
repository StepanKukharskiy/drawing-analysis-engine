import copy
import inspect
import json
import unittest
from collections import Counter
from pathlib import Path

import src.drawing_engine.disciplines.mep.mep_occurrence_coverage_audit as coverage_module
from src.drawing_engine.disciplines.mep.mep_occurrence_coverage_audit import (
    build_mep_occurrence_coverage_audit,
    validate_mep_occurrence_coverage_audit,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "fixtures" / "mep" / "m_and_p_coordination"
CHECKPOINT_ROOT = FIXTURE_ROOT / "real_m2_m5_checkpoint"
M4_COVERAGE_ROOT = FIXTURE_ROOT / "real_m4_discrete_coverage"
FROZEN = FIXTURE_ROOT / "m_and_p_coordination.occurrence-coverage-audit.json"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _build(attribute_bindings=None):
    return build_mep_occurrence_coverage_audit(
        sheet_registry=_load(FIXTURE_ROOT / "m_and_p_coordination.sheet-registry.json"),
        attribute_bindings=attribute_bindings or _load(M4_COVERAGE_ROOT / "full_package.attribute-bindings.json"),
        annotation_observations=_load(FIXTURE_ROOT / "m_and_p_coordination.annotation-observations.json"),
        reviewed_coverage=_load(FIXTURE_ROOT / "m_and_p_coordination_m7b_coverage_truth.json"),
        discrete_target_review=_load(M4_COVERAGE_ROOT / "full_package.discrete-target-review.json"),
    )


class MepOccurrenceCoverageAuditTest(unittest.TestCase):
    def test_reviewed_audit_marks_four_routes_as_current_m4_subset(self):
        payload = _build()
        self.assertEqual(validate_mep_occurrence_coverage_audit(payload), [])
        self.assertEqual(payload["summary"], {
            "page_count": 18,
            "item_bearing_page_count": 16,
            "m4_covered_page_count": 2,
            "uncovered_item_bearing_page_count": 14,
            "accepted_m4_occurrence_count": 4,
            "accepted_routed_material_occurrence_count": 4,
            "accepted_discrete_occurrence_count": 0,
            "reviewed_unresolved_finding_count": 11,
            "reviewed_negative_target_page_count": 8,
            "candidate_inventory_not_established_page_count": 6,
        })
        self.assertEqual(
            Counter(
                row["m4_discrete_target_review_state"]
                for row in payload["pages"]
                if row["m4_discrete_target_review_state"] is not None
            ),
            Counter({
                "reviewed_no_acceptable_discrete_target": 8,
                "candidate_inventory_not_established": 6,
            }),
        )
        self.assertEqual(
            payload["coverage_conclusion"]["accepted_set_scope"],
            "complete_for_frozen_m4_payload_only",
        )
        self.assertEqual(
            payload["coverage_conclusion"]["document_occurrence_completeness"],
            "not_established",
        )
        self.assertEqual(
            [row["coverage_state"] for row in payload["pages"]].count("not_applicable_divider"),
            2,
        )

    def test_frozen_reviewed_audit_replays_exactly(self):
        frozen = _load(FROZEN)
        self.assertEqual(validate_mep_occurrence_coverage_audit(frozen), [])
        self.assertEqual(_build(), frozen)

    def test_reviewed_hints_cover_discrete_categories_without_promoting_them(self):
        payload = _build()
        categories = {row["category"]: row for row in payload["category_coverage"]}
        self.assertEqual(categories["equipment"]["reviewed_unresolved_hint_count"], 9)
        self.assertEqual(categories["valve"]["reviewed_unresolved_hint_count"], 1)
        self.assertEqual(categories["damper"]["reviewed_unresolved_hint_count"], 2)
        self.assertEqual(categories["fitting"]["reviewed_unresolved_hint_count"], 0)
        self.assertEqual(categories["fixture"]["reviewed_unresolved_hint_count"], 0)
        self.assertTrue(all(not row["source_absence_established"] for row in categories.values()))
        for row in payload["reviewed_unresolved_findings"]:
            self.assertIsNone(row["m4_target_ref"])
            self.assertIsNone(row["observed_occurrence_count"])
            self.assertIsNone(row["physical_instance_count"])
            self.assertIsNone(row["calculated_count"])
            self.assertFalse(any(row["authority"].values()))

    def test_validator_rejects_review_hint_promotion_and_completeness_claim(self):
        baseline = _build()
        mutations = []
        changed = copy.deepcopy(baseline)
        changed["reviewed_unresolved_findings"][0]["physical_instance_count"] = 1
        mutations.append(changed)
        changed = copy.deepcopy(baseline)
        changed["reviewed_unresolved_findings"][0]["m4_target_ref"] = "invented.target"
        mutations.append(changed)
        changed = copy.deepcopy(baseline)
        changed["coverage_conclusion"]["document_occurrence_completeness"] = "complete"
        mutations.append(changed)
        changed = copy.deepcopy(baseline)
        changed["category_coverage"][2]["source_absence_established"] = True
        mutations.append(changed)
        changed = copy.deepcopy(baseline)
        changed["pages"][1]["accepted_m4_occurrence_refs"] = []
        mutations.append(changed)
        changed = copy.deepcopy(baseline)
        changed["category_coverage"][0]["reviewed_unresolved_hint_count"] += 1
        mutations.append(changed)
        changed = copy.deepcopy(baseline)
        target_page = next(
            row
            for row in changed["pages"]
            if row["m4_discrete_target_review_state"]
            == "reviewed_no_acceptable_discrete_target"
        )
        target_page["m4_discrete_target_relation_refs"] = []
        mutations.append(changed)
        changed = copy.deepcopy(baseline)
        changed["m4_discrete_target_review_contract_ref"][
            "document_occurrence_completeness"
        ] = "complete"
        mutations.append(changed)
        for changed in mutations:
            self.assertTrue(validate_mep_occurrence_coverage_audit(changed))

    def test_production_source_has_no_fixture_dispatch(self):
        source = inspect.getsource(coverage_module)
        for forbidden in ("m_and_p_coordination", "L01-MP", "VFD", "HUH", "fire damper"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
