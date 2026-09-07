import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.drawing_engine.project.quote_approval import (
    build_quote_summary,
    create_quote_approval_record,
    estimate_comparison_sha256,
    validate_quote_approval_record,
    write_quote_approval_record,
)


TIMESTAMP = "2026-08-26T12:00:00+03:00"


def _comparison_bytes() -> bytes:
    payload = {
        "schema_version": "0.1.0",
        "frozen_calculation": {"sha256": "a" * 64},
        "pages": [
            {
                "page": 1,
                "element_id": "element.1",
                "calculated_from_drawing": {
                    "page": 1,
                    "element_id": "element.1",
                    "concrete_volume_m3": 1.25,
                    "concrete_state": "derived",
                    "reinforcement_centerline_m": 14.0,
                    "reinforcement_fabrication_length_m": None,
                    "reinforcement_mass_kg": None,
                    "reinforcement_state": "partial",
                },
                "declared_by_designer": {
                    "concrete": [{"value": 9.99, "unit": "m3"}],
                    "reinforcement": [{"value": 42.0, "unit": "kg"}],
                },
                "approved_for_quote": None,
            },
            {
                "page": 2,
                "element_id": "element.2",
                "calculated_from_drawing": {
                    "page": 2,
                    "element_id": "element.2",
                    "concrete_volume_m3": 2.75,
                    "concrete_state": "derived",
                    "reinforcement_centerline_m": None,
                    "reinforcement_fabrication_length_m": None,
                    "reinforcement_mass_kg": 20.5,
                    "reinforcement_state": "derived",
                },
                "declared_by_designer": {
                    "concrete": [{"value": 2.7, "unit": "m3"}],
                    "reinforcement": [{"value": 20.0, "unit": "kg"}],
                },
                "approved_for_quote": None,
            },
        ],
    }
    return (json.dumps(payload, indent=2) + "\n").encode()


def _decision(
    element_id="element.1",
    quantity_field="concrete_volume_m3",
    value=1.25,
    unit="m3",
):
    return {
        "element_id": element_id,
        "quantity_field": quantity_field,
        "value": value,
        "unit": unit,
        "reason": "Drawing geometry and attached dimensions reviewed",
        "drawing_evidence": ["drawing[12].item[0]", "dimension_attachment.004"],
    }


class QuoteApprovalTest(unittest.TestCase):
    def _record(self, *decisions):
        return create_quote_approval_record(
            _comparison_bytes(),
            engineer="engineer.7",
            approved_at=TIMESTAMP,
            decisions=decisions or (_decision(),),
        )

    def _rehash(self, record):
        content = {key: value for key, value in record.items() if key != "record_sha256"}
        encoded = json.dumps(
            content,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        record["record_sha256"] = hashlib.sha256(encoded).hexdigest()

    def assert_invalid(self, record, message):
        validation = validate_quote_approval_record(_comparison_bytes(), record)
        self.assertEqual(validation["status"], "fail")
        self.assertTrue(any(message in error for error in validation["errors"]), validation["errors"])

    def test_record_binds_exact_comparison_bytes_and_frozen_graph(self):
        raw = _comparison_bytes()
        record = self._record()
        self.assertEqual(record["estimate_comparison_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(record["estimate_comparison_byte_count"], len(raw))
        self.assertEqual(record["frozen_engineering_graph_sha256"], "a" * 64)
        self.assertEqual(validate_quote_approval_record(raw, record)["status"], "pass")
        self.assertEqual(estimate_comparison_sha256(raw), hashlib.sha256(raw).hexdigest())

        reformatted = json.dumps(json.loads(raw), separators=(",", ":")).encode()
        validation = validate_quote_approval_record(reformatted, record)
        self.assertEqual(validation["status"], "fail")
        self.assertTrue(any("exact artifact bytes" in error for error in validation["errors"]))

    def test_record_contains_required_audit_fields(self):
        decision = self._record()["decisions"][0]
        self.assertEqual(decision["element_id"], "element.1")
        self.assertEqual(decision["quantity_field"], "concrete_volume_m3")
        self.assertEqual(decision["value"], 1.25)
        self.assertEqual(decision["unit"], "m3")
        self.assertEqual(decision["engineer"], "engineer.7")
        self.assertEqual(decision["approved_at"], TIMESTAMP)
        self.assertTrue(decision["reason"])
        self.assertTrue(decision["drawing_evidence"])

    def test_rejects_declared_or_arbitrary_value(self):
        with self.assertRaisesRegex(ValueError, "exactly equal"):
            self._record(_decision(value=9.99))
        with self.assertRaisesRegex(ValueError, "exactly equal"):
            self._record(_decision(value=1.3))

    def test_rejects_unknown_calculated_value_and_does_not_substitute_zero(self):
        with self.assertRaisesRegex(ValueError, "unknown calculated_from_drawing"):
            self._record(
                _decision(
                    quantity_field="reinforcement_mass_kg",
                    value=0,
                    unit="kg",
                )
            )

    def test_rejects_unknown_field_element_and_unit_mismatch(self):
        with self.assertRaisesRegex(ValueError, "not approvable"):
            self._record(_decision(quantity_field="declared_by_designer", value=9.99))
        with self.assertRaisesRegex(ValueError, "not in the estimate comparison"):
            self._record(_decision(element_id="missing"))
        with self.assertRaisesRegex(ValueError, "unit must be m3"):
            self._record(_decision(unit="kg"))

    def test_rejects_duplicate_decisions_and_declared_evidence(self):
        with self.assertRaisesRegex(ValueError, "duplicate approval decision"):
            self._record(_decision(), _decision())
        declared = _decision()
        declared["drawing_evidence"] = ["declared_schedule.cell[1]"]
        with self.assertRaisesRegex(ValueError, "cannot cite declared schedule"):
            self._record(declared)

    def test_rejects_stale_graph_binding_and_mutated_record(self):
        record = self._record()
        stale = copy.deepcopy(record)
        stale["frozen_engineering_graph_sha256"] = "b" * 64
        self._rehash(stale)
        self.assert_invalid(stale, "does not match the comparison")

        mutated = copy.deepcopy(record)
        mutated["decisions"][0]["reason"] = "changed after signing"
        self.assert_invalid(mutated, "record SHA-256 does not match")

    def test_summary_contains_only_explicit_approvals(self):
        record = self._record(
            _decision(),
            _decision(element_id="element.2", value=2.75),
            _decision(
                element_id="element.2",
                quantity_field="reinforcement_mass_kg",
                value=20.5,
                unit="kg",
            ),
        )
        summary = build_quote_summary(_comparison_bytes(), record)
        self.assertEqual(len(summary["approved_values"]), 3)
        self.assertEqual(
            summary["totals"],
            [
                {
                    "quantity_field": "concrete_volume_m3",
                    "value": 4.0,
                    "unit": "m3",
                    "approved_element_count": 2,
                },
                {
                    "quantity_field": "reinforcement_mass_kg",
                    "value": 20.5,
                    "unit": "kg",
                    "approved_element_count": 1,
                },
            ],
        )
        encoded = json.dumps(summary)
        self.assertNotIn("declared_by_designer", encoded)
        self.assertNotIn("reinforcement_fabrication_length_m\": 0", encoded)

    def test_immutable_writer_refuses_duplicate_path(self):
        record = self._record()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "approval.json"
            write_quote_approval_record(path, _comparison_bytes(), record)
            written = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_quote_approval_record(path, _comparison_bytes(), record)
            self.assertEqual(path.read_bytes(), written)


if __name__ == "__main__":
    unittest.main()
