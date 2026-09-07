import copy
import inspect
import json
import unittest
from pathlib import Path

import src.drawing_engine.disciplines.mep.mep_sheet_registry as mep_sheet_registry
from src.drawing_engine.disciplines.mep.mep_sheet_registry import (
    build_mep_sheet_registry,
    build_sheet_page_record,
    propose_adjoining_sheet_transform,
    propose_grid_axes,
    validate_mep_sheet_registry,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PDF = ROOT / "M&P mark-up against shop systems piping.pdf"
TRUTH_PATH = ROOT / "fixtures" / "mep" / "m_and_p_coordination" / "m_and_p_coordination_truth.json"
REGISTRY_PATH = ROOT / "fixtures" / "mep" / "m_and_p_coordination" / "m_and_p_coordination.sheet-registry.json"


def _token(page_ref, index, text, x, y, *, confidence=1.0, method="native_pdf_text"):
    return {
        "id": f"text.{page_ref}.{index}",
        "text": text,
        "bbox_display": [x, y, x + max(8, 5 * len(text)), y + 10],
        "confidence": confidence,
        "method": method,
        "reading_order": index,
    }


def _quality(route="native", *, path_count=20, image_count=0):
    return {
        "schema_version": "0.6.0",
        "layer": "perception_quality_route",
        "route": route,
        "reason": "synthetic quality route",
        "metrics": {
            "native_text_chars": 100,
            "native_path_item_count": path_count,
            "embedded_image_count": image_count,
            "embedded_image_area_ratio": 0.0,
        },
        "thresholds": {},
        "contract": {"native_primary": True, "raster_output_is_observation_only": True},
    }


def _plan_tokens(page_ref, *, dx=0, dy=0, sheet="L01-MP-P.1A", area="1A"):
    words = (
        f"Level 01 Area {area} Mechanical Piping Plan {sheet} "
        'SCALE: 1/8" = 1\'-0" A ISSUED FOR REVIEW 03/01/2023 PLAN NORTH'
    ).split()
    tokens = [_token(page_ref, index, word, 600 + index * 10, 700) for index, word in enumerate(words)]
    index = len(tokens)
    for label, x in (("A", 100 + dx), ("B", 500 + dx)):
        tokens.append(_token(page_ref, index, label, x, 40 + dy)); index += 1
        tokens.append(_token(page_ref, index, label, x, 750 + dy)); index += 1
    for label, y in (("1", 120 + dy), ("2", 620 + dy)):
        tokens.append(_token(page_ref, index, label, 40 + dx, y)); index += 1
        tokens.append(_token(page_ref, index, label, 950 + dx, y)); index += 1
    return tokens


def _page(page_ref, number, *, dx=0, dy=0, sheet="L01-MP-P.1A", area="1A"):
    return build_sheet_page_record(
        page_ref=page_ref,
        page_number=number,
        page_width=1100,
        page_height=820,
        native_tokens=_plan_tokens(page_ref, dx=dx, dy=dy, sheet=sheet, area=area),
        quality=_quality(),
    )


class MepSheetRegistryTest(unittest.TestCase):
    def test_hvac_level_title_uses_unique_prominent_title_block_sheet_number(self):
        words = ('PENTHOUSE LEVEL - HVAC - NEW M-301 M-301 M-202 '
                 'SCALE: 1/4" = 1\'-0"').split()
        tokens = [_token('page.hvac', index, word, 20 + index * 25, 100)
                  for index, word in enumerate(words)]
        for token in tokens:
            if token['text'] == 'M-202':
                token['bbox_display'][3] = token['bbox_display'][1] + 30
        page = build_sheet_page_record(page_ref='page.hvac', page_number=9,
            page_width=2448, page_height=1584, native_tokens=tokens,
            quality=_quality())
        self.assertEqual(page['role'], 'mechanical_plan')
        self.assertEqual(page['fields']['sheet_number']['value'], 'M-202')
        self.assertEqual(page['fields']['level']['value'], 'PENTHOUSE LEVEL')
        self.assertEqual(page['fields']['discipline']['value'], 'mechanical')

        section_tokens = [_token('page.section', index, word,
                                  20 + index * 25, 100)
                          for index, word in enumerate(
                              'MECHANICAL SECTIONS M-202 M-301'.split())]
        section_tokens[-1]['bbox_display'][3] = (
            section_tokens[-1]['bbox_display'][1] + 30)
        section = build_sheet_page_record(page_ref='page.section', page_number=12,
            page_width=2448, page_height=1584, native_tokens=section_tokens,
            quality=_quality())
        self.assertEqual(section['role'], 'mechanical_section_sheet')
        self.assertEqual(section['fields']['sheet_number']['value'], 'M-301')

    def test_native_title_fields_and_grid_axes_are_evidence_backed(self):
        page = _page("page.1", 1)

        self.assertEqual(page["role"], "mechanical_piping_plan")
        self.assertEqual(page["fields"]["sheet_number"]["value"], "L01-MP-P.1A")
        self.assertEqual(page["fields"]["level"]["normalised_value"], "01")
        self.assertEqual(page["fields"]["area"]["value"], "1A")
        self.assertEqual(page["fields"]["discipline"]["value"], "mechanical")
        self.assertEqual(page["fields"]["scale"]["drawing_inches_per_paper_inch"], 96.0)
        self.assertEqual(page["fields"]["revision"]["value"]["code"], "A")
        self.assertEqual(len(page["grid_axes"]), 4)
        self.assertTrue(all(len(axis["evidence_refs"]) == 2 for axis in page["grid_axes"]))
        self.assertEqual(len(page["north_proposals"]), 1)
        self.assertIsNone(page["north_proposals"][0]["direction_display"])
        self.assertFalse(page["quantity_eligible"])

    def test_registry_specific_ocr_augments_missing_title_without_changing_base_route(self):
        native = [_token("page.1", 0, "Coordination", 20, 20)]
        ocr = _plan_tokens("page.1", sheet="BR7", area="2")
        ocr_words = "1ST FLOOR ABOVE GROUND PIPING - AREA 2 - LUBE SYSTEM BR7 SCALE: 1/8\" = 1'-0\"".split()
        ocr[: len(ocr_words)] = [
            _token("page.1", index, word, 500 + index * 12, 700, confidence=0.9, method="tesseract_page_ocr")
            for index, word in enumerate(ocr_words)
        ]
        page = build_sheet_page_record(
            page_ref="page.1",
            page_number=1,
            page_width=1100,
            page_height=820,
            native_tokens=native,
            quality=_quality("native", path_count=200),
            ocr_tokens=ocr,
            ocr_provenance={"method": "tesseract_page_ocr"},
        )

        self.assertEqual(page["role"], "lube_system_shop_plan")
        self.assertEqual(page["fields"]["sheet_number"]["value"], "BR7")
        self.assertEqual(page["quality_route"]["base_structural_quality_route"]["route"], "native")
        self.assertEqual(page["quality_route"]["mep_registry_route"], "hybrid_text_ocr")

    def test_sparse_content_page_is_a_package_divider_not_a_sheet(self):
        tokens = [_token("page.divider", 0, "Mechanical", 500, 400), _token("page.divider", 1, "Package", 550, 400)]
        page = build_sheet_page_record(
            page_ref="page.divider",
            page_number=1,
            page_width=1100,
            page_height=820,
            native_tokens=tokens,
            quality=_quality(path_count=0),
        )
        member = _page("page.member", 2)
        payload = build_mep_sheet_registry(document={"page_count": 2}, pages=[page, member])

        self.assertEqual(page["role"], "divider")
        self.assertEqual(page["fields"]["title"]["value"], "Mechanical Package")
        self.assertEqual(len(payload["packages"]), 1)
        self.assertEqual(payload["packages"][0]["member_page_refs"], ["page.member"])
        self.assertFalse(payload["packages"][0]["physical_system_identity_established"])

    def test_redundant_grid_and_scale_agreement_accepts_translation(self):
        source = _page("page.source", 1)
        target = _page("page.target", 2, dx=40, dy=-20, sheet="L01-MP-P.1B", area="1B")

        proposal = propose_adjoining_sheet_transform(source, target)

        self.assertIsNotNone(proposal)
        self.assertEqual(proposal["state"], "accepted")
        self.assertEqual(proposal["shared_vertical_labels"], ["A", "B"])
        self.assertEqual(proposal["shared_horizontal_labels"], ["1", "2"])
        self.assertAlmostEqual(proposal["matrix_source_display_to_target_display"][4], 40.0)
        self.assertAlmostEqual(proposal["matrix_source_display_to_target_display"][5], -20.0)
        self.assertFalse(proposal["physical_continuation_established"])

    def test_equal_area_or_one_axis_each_cannot_accept_transform(self):
        source = _page("page.source", 1)
        target = _page("page.target", 2, sheet="L01-MP-P.1B")
        target["grid_axes"] = [
            axis
            for axis in target["grid_axes"]
            if axis["label"] in {"A", "1"}
        ]

        proposal = propose_adjoining_sheet_transform(source, target)

        self.assertIsNotNone(proposal)
        self.assertEqual(proposal["state"], "abstained")
        self.assertIsNone(proposal["matrix_source_display_to_target_display"])
        self.assertIn("no_unique_redundant_grid_offset_consensus", proposal["reasons"])

    def test_conflicting_title_scale_rejects_otherwise_matching_grid(self):
        source = _page("page.source", 1)
        target = _page("page.target", 2, sheet="L01-MP-P.1B")
        target["fields"]["scale"]["drawing_inches_per_paper_inch"] = 48.0

        proposal = propose_adjoining_sheet_transform(source, target)

        self.assertEqual(proposal["state"], "rejected")
        self.assertIn("title_scale_conflicts_with_grid_scale", proposal["reasons"])

    def test_grid_axis_needs_opposing_aligned_labels(self):
        tokens = [
            _token("page.1", 0, "A", 100, 50),
            _token("page.1", 1, "A", 300, 300),
            _token("page.1", 2, "ROOM", 100, 700),
        ]
        self.assertEqual(
            propose_grid_axes(tokens, page_ref="page.1", page_width=1000, page_height=800),
            [],
        )

    def test_validator_rejects_quantity_or_under_evidenced_transform_authority(self):
        source = _page("page.source", 1)
        target = _page("page.target", 2, sheet="L01-MP-P.1B", area="1B")
        payload = build_mep_sheet_registry(document={"page_count": 2}, pages=[source, target])
        self.assertEqual(validate_mep_sheet_registry(payload), [])

        changed = copy.deepcopy(payload)
        changed["adjoining_sheet_transforms"][0]["physical_continuation_established"] = True
        self.assertTrue(validate_mep_sheet_registry(changed))

        changed = copy.deepcopy(payload)
        changed["adjoining_sheet_transforms"][0]["grid_fit"]["x"]["maximum_residual"] = 1000
        self.assertTrue(validate_mep_sheet_registry(changed))

    def test_production_module_contains_no_fixture_dispatch(self):
        source = inspect.getsource(mep_sheet_registry)
        self.assertNotIn("M&P mark-up", source)
        self.assertNotIn("L01-MP-P.1A", source)
        self.assertNotIn("BR7", source)
        self.assertNotIn("page_number ==", source)

    def test_frozen_registry_contract_replays_without_source_pdf(self):
        stored = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

        self.assertEqual(validate_mep_sheet_registry(stored), [])
        self.assertEqual(stored["summary"]["page_count"], 18)
        self.assertEqual(stored["summary"]["package_count"], 2)
        self.assertEqual(
            stored["summary"]["accepted_adjoining_sheet_transform_count"],
            15,
        )
        self.assertFalse(
            stored["exchange_contract"]["route_identity_or_physical_continuation_established"]
        )

    def test_live_fixture_matches_frozen_registry_and_reviewed_page_roles(self):
        truth = json.loads(TRUTH_PATH.read_text(encoding="utf-8"))
        stored = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        extracted = mep_sheet_registry.extract_mep_sheet_registry(SOURCE_PDF)

        self.assertEqual(extracted, stored)
        self.assertEqual(validate_mep_sheet_registry(stored, SOURCE_PDF), [])
        observed_roles = {
            int(page["page_number"]): page["role"] for page in stored["pages"]
        }
        expected_roles = {
            int(page["page_number"]): page["role"] for page in truth["page_truth"]
        }
        self.assertEqual(observed_roles, expected_roles)
        self.assertEqual(stored["summary"]["page_count"], truth["expected_counts"]["pages"])
        self.assertEqual(stored["summary"]["package_count"], 2)
        self.assertTrue(stored["summary"]["accepted_adjoining_sheet_transform_count"] > 0)
        expected_sheet_numbers = {
            2: "L01-MP-P.1A",
            3: "L01-MP-P.1B",
            4: "L01-MP-P.2A",
            5: "L01-MP-P.2B",
            6: "L01-MP-P.3A",
            7: "L01-MP-P.3B",
            8: "L01-MP-P.4A",
            9: "L01-MP-P.4B",
            10: "L01-MP-P.5A",
            11: "L01-MP-P.5B",
            12: "L01-MP-P.6A",
            13: "L01-MP-P.6B",
            14: "L01-MP-P.MER",
            16: "BR7",
            17: "BR6",
            18: "BR5",
        }
        for page in stored["pages"]:
            if page["role"] != "divider":
                self.assertEqual(
                    page["fields"]["sheet_number"]["value"],
                    expected_sheet_numbers[page["page_number"]],
                )
                self.assertIn(page["fields"]["sheet_number"]["state"], {"observed", "derived"})
                expected_scale = (
                    None if page["page_number"] == 14
                    else 48.0 if page["page_number"] < 15
                    else 96.0
                )
                self.assertEqual(
                    page["fields"]["scale"]["drawing_inches_per_paper_inch"],
                    expected_scale,
                )
                self.assertEqual(page["fields"]["level"]["normalised_value"], "01")
                self.assertGreaterEqual(len(page["grid_axes"]), 4)


if __name__ == "__main__":
    unittest.main()
