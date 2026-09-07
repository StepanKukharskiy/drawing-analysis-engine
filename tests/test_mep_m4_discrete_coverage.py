import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from tools.generate_mep_m4_discrete_coverage import generate
from src.drawing_engine.disciplines.mep.mep_attribute_binding import validate_mep_attribute_bindings
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import validate_mep_terminology_proposals
from src.drawing_engine.disciplines.mep.mep_occurrence_coverage_audit import validate_mep_discrete_target_review


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "mep" / "m_and_p_coordination" / "real_m4_discrete_coverage"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class MepM4DiscreteCoverageTest(unittest.TestCase):
    def test_frozen_full_package_preserves_reviewed_negative_targets(self):
        terminology = _load(FIXTURE / "full_package.terminology-proposals.json")
        bindings = _load(FIXTURE / "full_package.attribute-bindings.json")
        review = _load(FIXTURE / "full_package.discrete-target-review.json")
        registry = _load(
            ROOT
            / "fixtures"
            / "mep"
            / "m_and_p_coordination"
            / "m_and_p_coordination.sheet-registry.json"
        )

        self.assertEqual(validate_mep_terminology_proposals(terminology), [])
        self.assertEqual(validate_mep_attribute_bindings(bindings), [])
        self.assertEqual(
            validate_mep_discrete_target_review(
                review,
                sheet_registry=registry,
                attribute_bindings=bindings,
            ),
            [],
        )
        self.assertEqual(len(terminology["proposals"]), 24)
        self.assertEqual(bindings["summary"]["accepted_relation_count"], 12)
        self.assertEqual(bindings["summary"]["abstained_relation_count"], 12)

        discrete = [
            row
            for row in bindings["relations"]
            if row["relation_type"] in {"equipment_endpoint", "valve", "fitting", "damper"}
        ]
        self.assertEqual(
            Counter(row["relation_type"] for row in discrete),
            Counter({"equipment_endpoint": 9, "valve": 1, "damper": 2}),
        )
        self.assertTrue(all(row["state"] == "abstained" for row in discrete))
        self.assertTrue(all(row["target_refs"] == [] for row in discrete))
        self.assertTrue(all(row["abstention_evidence_refs"] for row in discrete))
        self.assertTrue(all(row["quantity_eligible"] is False for row in discrete))

        self.assertEqual(review["summary"], {
            "uncovered_item_bearing_page_count": 14,
            "review_state_counts": {
                "candidate_inventory_not_established": 6,
                "reviewed_no_acceptable_discrete_target": 8,
            },
            "reviewed_discrete_proposal_count": 12,
            "accepted_discrete_target_count": 0,
        })
        self.assertEqual(len(review["pages"]), 14)
        self.assertTrue(
            all(not row["document_occurrence_completeness_established"] for row in review["pages"])
        )
        self.assertFalse(review["authority"]["item_occurrence_established"])
        self.assertFalse(review["authority"]["calculated_count_established"])

    def test_generator_replays_all_frozen_outputs_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = generate(output_dir=Path(directory))
            self.assertEqual(set(paths), {
                "full_package.terminology-proposals.json",
                "full_package.attribute-bindings.json",
                "full_package.discrete-target-review.json",
            })
            for name, path in paths.items():
                self.assertEqual(_load(path), _load(FIXTURE / name))


if __name__ == "__main__":
    unittest.main()
