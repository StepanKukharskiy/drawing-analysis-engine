import copy
import csv
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from tools.generate_mep_m7a_catalog import generate
import src.drawing_engine.disciplines.mep.mep_item_catalog as mep_item_catalog
from src.drawing_engine.disciplines.mep.mep_item_catalog import (
    RECORD_CONTRACTS,
    build_mep_item_catalog,
    flatten_mep_item_catalog,
    validate_mep_item_catalog,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "fixtures" / "mep" / "m_and_p_coordination"
CHECKPOINT_ROOT = FIXTURE_ROOT / "real_m2_m5_checkpoint"
M4_COVERAGE_ROOT = FIXTURE_ROOT / "real_m4_discrete_coverage"
FROZEN = FIXTURE_ROOT / "m_and_p_coordination.mep-item-catalog.json"
FROZEN_CSV = FIXTURE_ROOT / "m_and_p_coordination.mep-item-catalog.review.csv"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs():
    return {
        "sheet_registry": _load(FIXTURE_ROOT / "m_and_p_coordination.sheet-registry.json"),
        "attribute_bindings": _load(M4_COVERAGE_ROOT / "full_package.attribute-bindings.json"),
        "bounded_local_3d": _load(CHECKPOINT_ROOT / "pages_1a_1b.bounded-local-3d.json"),
        "occurrence_coverage_audit": _load(
            FIXTURE_ROOT / "m_and_p_coordination.occurrence-coverage-audit.json"
        ),
        "bounded_discrete_counts": _load(
            FIXTURE_ROOT / "m_and_p_coordination.bounded-discrete-counts.json"
        ),
    }


def _build(inputs=None):
    return build_mep_item_catalog(**(inputs or _inputs()))


def _all_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


class MepItemCatalogTest(unittest.TestCase):
    def test_real_catalog_covers_all_pages_and_preserves_unknown_product_fields(self):
        payload = _build()
        self.assertEqual(validate_mep_item_catalog(payload), [])
        self.assertEqual(payload["record_contracts"], RECORD_CONTRACTS)
        self.assertEqual(payload["summary"], {
            "page_count": 18,
            "page_with_item_count": 2,
            "item_occurrence_count": 4,
            "routed_material_occurrence_count": 4,
            "physical_item_count": 0,
            "takeoff_line_count": 4,
            "calculated_value_count": 0,
            "declared_value_count": 0,
            "coverage_abstention_count": 11,
        })
        self.assertEqual(
            [row["pdf_page_number"] for row in payload["pages"]], list(range(1, 19))
        )
        self.assertEqual(
            {row["source"]["drawing_sheet_number"] for row in payload["item_occurrences"]},
            {"L01-MP-P.1A", "L01-MP-P.1B"},
        )
        for row in payload["item_occurrences"]:
            self.assertEqual(row["item_type"], "routed_material")
            self.assertEqual(row["item_subtype"], "pipe_route")
            self.assertIsNone(row["manufacturer"])
            self.assertIsNone(row["model_number"])
            self.assertIsNone(row["specification_reference"])
            self.assertIsNone(row["mounting_type"])
            self.assertEqual(row["counts"]["observed_occurrence_count"], 1)
            self.assertIsNone(row["counts"]["deduplicated_projected_count"])
            self.assertIsNone(row["counts"]["physical_instance_count"])
            self.assertTrue(row["evidence_refs"])
            self.assertIn("model_number_not_observed", row["unresolved_reasons"])
        self.assertNotIn("installed_length", set(_all_keys(payload)))
        self.assertNotIn("installed_length_m", set(_all_keys(payload)))

    def test_m5a_enriches_dimensions_and_elevation_but_never_counts_or_values(self):
        with_m5a = _build()
        inputs = _inputs()
        inputs["bounded_local_3d"] = None
        without_m5a = _build(inputs)
        self.assertTrue(
            all(
                any(dimension["physical_dimension"] for dimension in row["typed_dimensions"])
                for row in with_m5a["item_occurrences"]
            )
        )
        self.assertTrue(
            all(
                not any(dimension["physical_dimension"] for dimension in row["typed_dimensions"])
                for row in without_m5a["item_occurrences"]
            )
        )
        self.assertEqual(with_m5a["summary"]["item_occurrence_count"], without_m5a["summary"]["item_occurrence_count"])
        for line in with_m5a["takeoff_lines"]:
            self.assertTrue(
                all(channel["value"] is None and channel["unit"] is None for channel in line["value_channels"].values())
            )

    def test_flat_review_table_has_one_row_per_item_and_every_empty_page(self):
        rows = flatten_mep_item_catalog(_build())
        self.assertEqual(len(rows), 20)
        self.assertEqual({row["pdf_page_number"] for row in rows}, set(range(1, 19)))
        self.assertEqual(sum(row["row_type"] == "item_occurrence" for row in rows), 4)
        self.assertEqual(
            sum(row["row_type"] == "page_without_accepted_item_evidence" for row in rows),
            16,
        )
        self.assertTrue(all(row["calculated_value"] is None for row in rows))

    def test_reordering_upstream_records_preserves_semantic_catalog(self):
        inputs = _inputs()
        inputs["occurrence_coverage_audit"] = None
        inputs["bounded_discrete_counts"] = None
        baseline = _build(inputs)
        inputs = _inputs()
        inputs["occurrence_coverage_audit"] = None
        inputs["bounded_discrete_counts"] = None
        inputs["sheet_registry"]["pages"].reverse()
        inputs["attribute_bindings"]["relations"].reverse()
        inputs["attribute_bindings"]["outlined_route_composites"].reverse()
        inputs["bounded_local_3d"]["bounded_local_3d_segments"].reverse()
        reordered = _build(inputs)
        for key in ("pages", "item_occurrences", "physical_items", "takeoff_lines", "summary"):
            self.assertEqual(reordered[key], baseline[key])
        for key in ("m1_contract_ref", "m4_contract_ref", "m5a_contract_ref"):
            self.assertNotEqual(
                reordered[key]["payload_sha256"], baseline[key]["payload_sha256"]
            )

    def test_validator_rejects_count_value_metadata_and_authority_promotions(self):
        payload = _build()
        mutations = []
        changed = copy.deepcopy(payload)
        changed["item_occurrences"][0]["counts"]["physical_instance_count"] = 1
        mutations.append(changed)
        changed = copy.deepcopy(payload)
        changed["takeoff_lines"][0]["value_channels"]["calculated"] = {
            "value": 6.4,
            "unit": "m",
            "state": "calculated",
            "reason": None,
        }
        mutations.append(changed)
        changed = copy.deepcopy(payload)
        del changed["item_occurrences"][0]["model_number"]
        mutations.append(changed)
        changed = copy.deepcopy(payload)
        changed["exchange_contract"]["installed_length_emitted"] = True
        mutations.append(changed)
        changed = copy.deepcopy(payload)
        changed["item_occurrences"][0]["installed_length_m"] = 6.4
        mutations.append(changed)
        changed = copy.deepcopy(payload)
        changed["physical_items"] = [{"id": "invented-physical-item"}]
        changed["summary"]["physical_item_count"] = 1
        mutations.append(changed)
        for changed in mutations:
            self.assertTrue(validate_mep_item_catalog(changed))

    def test_document_mismatch_and_invalid_upstream_fail_before_catalog(self):
        inputs = _inputs()
        inputs["bounded_local_3d"]["document"]["document_key"] = "pdf-sha256:other"
        with self.assertRaisesRegex(ValueError, "document keys must match"):
            _build(inputs)

    def test_frozen_catalog_validates_and_replays_exactly(self):
        frozen = _load(FROZEN)
        self.assertEqual(validate_mep_item_catalog(frozen), [])
        self.assertEqual(_build(), frozen)

    def test_generator_replays_json_and_explicit_null_review_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "catalog.json"
            csv_path = root / "catalog.csv"
            generate(
                registry_path=FIXTURE_ROOT / "m_and_p_coordination.sheet-registry.json",
                bindings_path=M4_COVERAGE_ROOT / "full_package.attribute-bindings.json",
                m5a_path=CHECKPOINT_ROOT / "pages_1a_1b.bounded-local-3d.json",
                coverage_path=FIXTURE_ROOT / "m_and_p_coordination.occurrence-coverage-audit.json",
                m7b_path=FIXTURE_ROOT / "m_and_p_coordination.bounded-discrete-counts.json",
                json_path=json_path,
                csv_path=csv_path,
            )
            self.assertEqual(json_path.read_bytes(), FROZEN.read_bytes())
            self.assertEqual(csv_path.read_bytes(), FROZEN_CSV.read_bytes())
        with FROZEN_CSV.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 20)
        self.assertEqual({int(row["pdf_page_number"]) for row in rows}, set(range(1, 19)))
        self.assertTrue(
            all(
                row["manufacturer"] == "null"
                and row["model_number"] == "null"
                and row["specification_reference"] == "null"
                and row["mounting_type"] == "null"
                for row in rows
            )
        )

    def test_production_source_has_no_fixture_or_sheet_dispatch(self):
        source = inspect.getsource(mep_item_catalog)
        for forbidden in (
            "L01-MP-P.1A",
            "L01-MP-P.1B",
            "m_and_p_coordination",
            "pages_1a_1b",
            "candidate-08",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
