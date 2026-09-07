import copy
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from tools.generate_mep_m7c_fixture import generate
import src.drawing_engine.disciplines.mep.mep_declared_data as declared_module
from src.drawing_engine.disciplines.mep.mep_declared_data import (
    build_mep_declared_data,
    extract_mep_document_region_observations,
    validate_mep_declared_data,
    validate_mep_document_region_observations,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "fixtures" / "mep" / "m_and_p_coordination"
M1 = FIXTURE_ROOT / "m_and_p_coordination.sheet-registry.json"
REGIONS = FIXTURE_ROOT / "m_and_p_coordination.declaration-region-observations.json"
FROZEN = FIXTURE_ROOT / "m_and_p_coordination.declared-data.json"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _positive_inputs():
    registry = _load(M1)
    regions = _load(REGIONS)
    page = registry["pages"][1]
    lines = [
        "EQUIPMENT SCHEDULE",
        "MANUFACTURER: Acme Mechanical",
        "MODEL NO.: PX-400",
        "SPEC SECTION: 23 05 00",
        "MOUNTING TYPE: WALL",
        "QTY: 4",
    ]
    rows = []
    for index, text in enumerate(lines):
        row = {
            "record_type": "mep_document_region_observation",
            "record_version": "0.1.0",
            "id": f"synthetic.declaration.text.{index}",
            "page_ref": page["page_ref"],
            "page_number": 2,
            "text": text,
            "bbox_display": [100.0, 100.0 + index * 20, 400.0, 115.0 + index * 20],
            "role_candidates": ["schedule"] if index == 0 else [],
            "region_group_ref": "synthetic.schedule.region",
            "region_text_complete": False,
            "method": "native_pdf_text_block_heading_scan",
            "epistemic_state": "observed",
            "quantity_eligible": False,
        }
        rows.append(row)
    regions["observations"].extend(rows)
    region_page = regions["pages"][1]
    region_page["observation_refs"].extend(row["id"] for row in rows)
    region_page["observation_refs"].sort()
    region_page["observation_count"] = len(region_page["observation_refs"])
    role_counts = {}
    for row in regions["observations"]:
        for role in row["role_candidates"]:
            role_counts[role] = role_counts.get(role, 0) + 1
    regions["summary"] = {
        "page_count": 18,
        "observation_count": len(regions["observations"]),
        "role_candidate_counts": dict(sorted(role_counts.items())),
    }
    return registry, regions


class MepDeclaredDataTest(unittest.TestCase):
    def test_real_fixture_freezes_region_roles_but_no_scoped_declarations(self):
        payload = build_mep_declared_data(
            sheet_registry=_load(M1), region_observations=_load(REGIONS)
        )
        self.assertEqual(validate_mep_declared_data(payload), [])
        self.assertEqual(payload["summary"], {
            "page_count": 18,
            "recognized_region_count": 27,
            "accepted_role_count": 27,
            "complete_content_region_count": 0,
            "role_counts": {"detail": 1, "legend": 13, "note": 13},
            "declared_record_count": 0,
            "declared_field_counts": {},
        })
        self.assertTrue(
            payload["negative_fixture_result"]["no_scoped_declared_fields_observed"]
        )
        self.assertFalse(
            payload["negative_fixture_result"]["document_field_absence_established"]
        )
        self.assertTrue(
            all(
                row["content_scope"] == "heading_only_unresolved"
                for row in payload["document_regions"]
            )
        )

    def test_grouped_schedule_extracts_each_declared_field_independently(self):
        registry, regions = _positive_inputs()
        payload = build_mep_declared_data(
            sheet_registry=registry, region_observations=regions
        )
        records = {
            row["field_name"]: row for row in payload["declared_records"]
        }
        self.assertEqual(records["manufacturer"]["value"], "Acme Mechanical")
        self.assertEqual(records["model_number"]["value"], "PX-400")
        self.assertEqual(records["specification_reference"]["value"], "23 05 00")
        self.assertEqual(records["mounting_type"]["value"], "WALL")
        self.assertEqual(records["quantity"]["value"], 4)
        self.assertEqual(records["quantity"]["unit"], "ea")
        self.assertTrue(all(row["value_channel"] == "declared" for row in records.values()))
        self.assertTrue(all(row["calculated_value_used"] is False for row in records.values()))
        self.assertTrue(all(row["quantity_eligible"] is False for row in records.values()))

    def test_heading_only_region_never_establishes_field_absence(self):
        payload = build_mep_declared_data(
            sheet_registry=_load(M1), region_observations=_load(REGIONS)
        )
        self.assertEqual(payload["declared_records"], [])
        self.assertTrue(
            all(
                not row["document_declaration_completeness_established"]
                for row in payload["pages"]
            )
        )

    def test_frozen_result_and_generator_replay_exactly(self):
        frozen = _load(FROZEN)
        self.assertEqual(validate_mep_declared_data(frozen), [])
        self.assertEqual(
            build_mep_declared_data(
                sheet_registry=_load(M1), region_observations=_load(REGIONS)
            ),
            frozen,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "declared.json"
            generate(
                m1_path=M1,
                source_pdf_path=None,
                regions_path=REGIONS,
                output_path=output,
            )
            self.assertEqual(_load(output), frozen)

    def test_live_pdf_region_extraction_replays_frozen_observations(self):
        extracted = extract_mep_document_region_observations(
            pdf_path=ROOT / "M&P mark-up against shop systems piping.pdf",
            sheet_registry=_load(M1),
        )
        self.assertEqual(validate_mep_document_region_observations(extracted), [])
        self.assertEqual(extracted, _load(REGIONS))

    def test_validator_rejects_calculated_or_reconciled_authority(self):
        registry, regions = _positive_inputs()
        payload = build_mep_declared_data(
            sheet_registry=registry, region_observations=regions
        )
        changed = copy.deepcopy(payload)
        changed["declared_records"][0]["calculated_quantity"] = 1
        self.assertTrue(validate_mep_declared_data(changed))
        changed = copy.deepcopy(payload)
        changed["exchange_contract"]["reconciliation_emitted"] = True
        self.assertTrue(validate_mep_declared_data(changed))
        changed = copy.deepcopy(payload)
        changed["negative_fixture_result"]["document_field_absence_established"] = True
        self.assertTrue(validate_mep_declared_data(changed))

    def test_source_has_no_m4_m5_m7a_or_fixture_dispatch(self):
        source = inspect.getsource(declared_module)
        for forbidden in (
            "mep_attribute_binding",
            "mep_cross_sheet_runs",
            "mep_item_catalog",
            "mep_bounded_discrete_counts",
            "m_and_p_coordination",
            "L01-MP",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
