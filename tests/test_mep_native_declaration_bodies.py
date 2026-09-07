import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import fitz

from tools.generate_mep_native_declaration_bodies import generate
from src.drawing_engine.disciplines.mep.mep_native_declaration_bodies import extract_mep_native_declaration_bodies, validate_mep_native_declaration_bodies
from src.drawing_engine.disciplines.mep.mep_sheet_registry import build_mep_sheet_registry, build_sheet_page_record


def _source(directory, *, bottom=True, heading_only=False, continuation=False, rotation=0,
            cross_cell_text=False, quantity_text="4"):
    source = Path(directory) / "synthetic.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page(width=600, height=400)
        if heading_only:
            page.insert_text((40, 40), "EQUIPMENT SCHEDULE")
            page.insert_text((40, 80), "AHU-1 MANUFACTURER: EXAMPLE QTY: 4")
        else:
            for y in (30, 70, 110, *([190] if bottom else [])):
                page.draw_line((30, y), (570, y))
            for x in (30, 570):
                page.draw_line((x, 30), (x, 190))
            for x in (130, 280, 410):
                page.draw_line((x, 70), (x, 190))
            page.insert_text((160, 55), "EQUIPMENT SCHEDULE")
            for x, text in ((40, "MODEL"), (140, "QTY"), (290, "ID"), (420, "REMARKS")):
                page.insert_text((x, 95), text)
            for x, text in ((40, "Example-2"), (140, quantity_text), (290, "AHU-1 & 2"),
                            (420, "CONTINUED" if continuation else "WALL MOUNTED")):
                page.insert_text((x, 140), text)
            if cross_cell_text:
                page.insert_text((400, 170), "CROSSED")
        page.set_rotation(rotation)
        pdf.new_page(width=600, height=400)
        pdf.save(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    pages = [build_sheet_page_record(page_ref=f"page.{number}", page_number=number,
             page_width=600, page_height=400, native_tokens=[], quality={"route": "native"})
             for number in (1, 2)]
    registry = build_mep_sheet_registry(document={"document_key": f"pdf-sha256:{digest}",
               "source_pdf_sha256": digest, "source_bytes": source.stat().st_size, "page_count": 2}, pages=pages)
    return source, registry


class MepNativeDeclarationBodiesTest(unittest.TestCase):
    def test_internal_cell_ambiguity_stays_local_and_unknown_quantity_is_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            source, registry = _source(directory, cross_cell_text=True, quantity_text="-")
            payload = extract_mep_native_declaration_bodies(pdf_path=source, sheet_registry=registry)
            self.assertEqual(validate_mep_native_declaration_bodies(payload), [])
            self.assertEqual(payload["summary"]["accepted_bounded_body_count"], 1)
            self.assertEqual(payload["summary"]["resolved_declared_quantity_count"], 0)
            self.assertEqual(len(payload["regions"][0]["cell_ownership_ambiguities"]), 1)
            fields = {field["field_name"]: field for field in payload["declared_fields"]}
            self.assertIsNone(fields["quantity"]["value"])
            self.assertIsNone(fields["item_designation"]["value"])
            self.assertIsNone(fields["declared_table_cell"]["value"])
            self.assertEqual(fields["declared_table_cell"]["mounting_statements"], [])
            self.assertEqual(fields["model_number"]["value"], "Example-2")

    def test_closed_native_grid_header_mapping_and_explicit_quantity(self):
        with tempfile.TemporaryDirectory() as directory:
            source, registry = _source(directory, rotation=90)
            payload = extract_mep_native_declaration_bodies(pdf_path=source, sheet_registry=registry)
            self.assertEqual(validate_mep_native_declaration_bodies(payload), [])
            self.assertEqual(payload["summary"]["accepted_bounded_body_count"], 1)
            self.assertEqual(payload["summary"]["resolved_declared_quantity_count"], 1)
            fields = {field["field_name"]: field for field in payload["declared_fields"]}
            self.assertEqual(fields["quantity"]["value"], 4)
            self.assertEqual(fields["quantity"]["unit"], "ea")
            self.assertEqual(fields["model_number"]["value"], "Example-2")
            self.assertEqual(fields["item_designation"]["value"], "AHU-1 & 2")
            self.assertEqual(fields["declared_table_cell"]["mounting_statements"], ["WALL MOUNTED"])
            self.assertTrue(all(field["physical_item_ref"] is None and field["calculated_quantity"] is None
                                and not field["quantity_eligible"] for field in fields.values()))
            region = payload["regions"][0]
            self.assertNotEqual(region["bbox_pdf"], region["bbox_display"])
            self.assertTrue(all(side["native_edge_refs"] for side in region["outer_boundary_support"]))
            self.assertEqual(len(payload["pages"]), 2)
            self.assertFalse(payload["pages"][1]["document_field_absence_established"])

    def test_heading_open_bottom_and_continuation_never_establish_complete_body(self):
        for options in ({"heading_only": True}, {"bottom": False}, {"continuation": True}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                source, registry = _source(directory, **options)
                payload = extract_mep_native_declaration_bodies(pdf_path=source, sheet_registry=registry)
                self.assertEqual(validate_mep_native_declaration_bodies(payload), [])
                self.assertEqual(payload["summary"]["accepted_bounded_body_count"], 0)
                self.assertEqual(payload["declared_fields"], [])
                self.assertTrue(payload["pages"][0]["unresolved_headings"])

    def test_validator_replays_cells_borders_and_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            source, registry = _source(directory)
            payload = extract_mep_native_declaration_bodies(pdf_path=source, sheet_registry=registry)
            changed = copy.deepcopy(payload)
            changed["declared_fields"][0]["value"] = "invented"
            self.assertTrue(any("declared_fields" in error for error in validate_mep_native_declaration_bodies(changed)))
            changed = copy.deepcopy(payload)
            changed["source_pages"][0]["native_edges"] = []
            self.assertTrue(any("regions" in error for error in validate_mep_native_declaration_bodies(changed)))
            changed = copy.deepcopy(payload)
            changed["exchange_contract"]["quantity_eligible"] = True
            self.assertTrue(any("quantity_eligible" in error for error in validate_mep_native_declaration_bodies(changed)))
            source.write_bytes(b"wrong source")
            with self.assertRaisesRegex(ValueError, "frozen M1 registry"):
                extract_mep_native_declaration_bodies(pdf_path=source, sheet_registry=registry)

    def test_generator_deterministic_and_legacy_contract_is_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            source, registry = _source(directory)
            registry_path = Path(directory) / "m1.json"
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            output_dir = Path(directory) / "result"
            first = generate(source=source, registry=registry_path, output_dir=output_dir)
            self.assertEqual(first, generate(source=source, registry=registry_path, output_dir=output_dir))
            self.assertFalse(first["calculated_records_used"])
            self.assertFalse(first["document_declaration_completeness_established"])
            from src.drawing_engine.disciplines.mep.mep_declared_data import build_mep_declared_data, extract_mep_document_region_observations
            old = build_mep_declared_data(sheet_registry=registry,
                  region_observations=extract_mep_document_region_observations(pdf_path=source, sheet_registry=registry))
            self.assertEqual(old["declared_records"], [])
            self.assertTrue(all(not region["declaration_content_complete"] for region in old["document_regions"]))


if __name__ == "__main__":
    unittest.main()
