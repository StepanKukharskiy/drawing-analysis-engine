import copy
import hashlib
import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import fitz

from src.drawing_engine.disciplines.mep.mep_sheet_registry import build_mep_sheet_registry, build_sheet_page_record
from src.drawing_engine.disciplines.mep.mep_text_observations import extract_mep_text_observations, build_mep_document_text_proposals
import src.drawing_engine.disciplines.mep.mep_terminology_proposals as mep_terminology_proposals
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import (
    build_mep_terminology_proposals,
    interpret_mep_text,
    propose_mep_interpretations,
    terminology_pack,
    validate_mep_terminology_proposals,
)


ROOT = Path(__file__).resolve().parents[1]


def _observation(identifier, text="", **extra):
    return {
        "id": identifier,
        "page_ref": extra.pop("page_ref", "page.alpha"),
        "text": text,
        "evidence_channels": extra.pop("evidence_channels", ["native_pdf_text"]),
        **extra,
    }


class MepTerminologyProposalsTest(unittest.TestCase):
    def test_all_25_zero_inch_diagnostic_texts_are_not_pipe_sizes(self):
        # Exact text forms from the 2026-08-30 failures; filenames, coordinates,
        # sheet numbers and schedule values are not parser inputs.
        texts = ['1/4" = 1\'-0"'] * 12 + [
            f'{feet}\' - 0"' for feet in (7, 6, 2, 2, 8, 2, 2, 7, 7, 16, 5, 5)
        ] + ['Bottom of Unit 15\'-0"']
        self.assertEqual(len(texts), 25)
        for index, text in enumerate(texts):
            with self.subTest(text=text):
                row = _observation(f"failure.{index}", text)
                payload = build_mep_terminology_proposals(document={}, observations=[row])
                self.assertFalse(any(item["proposal_type"] == "inline_size" for item in payload["proposals"]))
                self.assertEqual(validate_mep_terminology_proposals(payload), [])
                self.assertEqual(payload["source_observations"][0]["text"], text)

    def test_positive_distance_tails_decimal_inches_and_fractional_elevations(self):
        row = _observation("mixed", '9\' - 4 3/8"; 1.5" CHWS; BOP -0\'-1/2"')
        proposals = propose_mep_interpretations(row)
        sizes = [item["candidate"] for item in proposals if item["proposal_type"] == "inline_size"]
        self.assertEqual([item["value"] for item in sizes], [1.5])
        elevation = next(item["candidate"] for item in proposals if item["proposal_type"] == "elevation")
        self.assertAlmostEqual(elevation["value"], -0.5 / 12)
        interpretation = interpret_mep_text(row)
        distance = next(item for item in interpretation["unit_interpretations"] if item["role"] == "architectural_distance")
        self.assertAlmostEqual(distance["value"], 9 + 4.375 / 12)
        self.assertEqual(distance["unit"], "ft")
        self.assertEqual(propose_mep_interpretations(_observation("not-inch", "42 INLET")), [])
        self.assertEqual(propose_mep_interpretations(_observation("scale", "1:100")), [])
        for text in ('BE=25\'-0 5/8"', 'C/L=12\'-3/8"', 'BOP 24 IN', 'BOD 300 MM'):
            with self.subTest(text=text):
                rows = propose_mep_interpretations(_observation("elev", text))
                self.assertEqual([item["proposal_type"] for item in rows], ["elevation"])

    def test_declarations_titles_and_notes_cannot_supply_m2_inputs(self):
        for role in ("schedule", "table", "cut_sheet", "note", "title"):
            with self.subTest(role=role):
                row = _observation("declared", 'AHU-12 DN 50 CHWS', region_role=role, symbol_kind="valve")
                self.assertEqual(propose_mep_interpretations(row), [])
                self.assertEqual(interpret_mep_text(row)["text_role"], role)
        unknown = _observation("unknown", '2" CHWS', region_role="unknown")
        payload = build_mep_terminology_proposals(document={}, observations=[unknown])
        self.assertTrue(all(item["state"] == "abstained" for item in payload["proposals"]))
        changed = copy.deepcopy(payload)
        changed["proposals"][0]["state"] = "proposed"
        self.assertTrue(any("region applicability" in error for error in validate_mep_terminology_proposals(changed)))
        changed = copy.deepcopy(payload)
        changed["source_observations"][0]["region_role"] = "schedule"
        self.assertTrue(any("non-drawing text role" in error for error in validate_mep_terminology_proposals(changed)))

    def test_full_document_adapter_preserves_rotation_coverage_and_quarantines_invalid_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pdf"
            with fitz.open() as pdf:
                page = pdf.new_page(width=400, height=300)
                page.insert_text((20, 30), '2" CHWS')
                page.insert_text((20, 70), '1/4" = 1\'-0"')
                page.insert_text((20, 110), 'DN 0')
                page.insert_text((20, 150), '1/0"')
                page.insert_text((20, 190), 'EQUIPMENT SCHEDULE\nAHU-1 DN 50')
                page.set_rotation(90)
                pdf.new_page(width=400, height=300)
                pdf.save(source)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            records = [build_sheet_page_record(
                page_ref=f"page.{number}", page_number=number,
                page_width=300 if number == 1 else 400, page_height=400 if number == 1 else 300,
                native_tokens=[], quality={"route": "native", "reason": "synthetic"},
            ) for number in (1, 2)]
            registry = build_mep_sheet_registry(document={
                "document_key": f"pdf-sha256:{digest}", "source_pdf_sha256": digest,
                "source_bytes": source.stat().st_size, "page_count": 2,
            }, pages=records)
            observations = extract_mep_text_observations(pdf_path=source, sheet_registry=registry)
            self.assertEqual(observations["summary"]["page_count"], 2)
            self.assertEqual(observations["pages"][1]["native_text_line_count"], 0)
            self.assertFalse(observations["pages"][1]["item_inventory_complete"])
            first = observations["observations"][0]
            self.assertNotEqual(first["bbox_pdf"], first["bbox_display"])
            self.assertTrue(first["native_spans"][0]["source_native_ref"].startswith(first["source_native_ref"]))
            payload = build_mep_document_text_proposals(observations)
            self.assertEqual(validate_mep_terminology_proposals(payload), [])
            self.assertEqual(len(payload["observation_diagnostics"]), 2)
            self.assertEqual(payload["source_observations"], observations["observations"])
            self.assertEqual({item["proposal_type"] for item in payload["proposals"]}, {"inline_size", "system"})
            self.assertEqual(len(payload["proposals"]), 2)
            bad = copy.deepcopy(payload)
            bad["proposals"].extend(bad["observation_diagnostics"][0]["diagnostic_proposals"])
            errors = validate_mep_terminology_proposals(bad)
            self.assertTrue(any("inline size must be finite and positive" in error for error in errors))
            self.assertTrue(any("quarantined observation supplied" in error for error in errors))
            changed = copy.deepcopy(observations)
            changed["observations"][0]["bbox_pdf"][0] = float("nan")
            self.assertEqual(len(build_mep_document_text_proposals(changed)["observation_diagnostics"]), 3)
            renamed = source.with_name("renamed.pdf")
            renamed.write_bytes(source.read_bytes())
            self.assertEqual(extract_mep_text_observations(pdf_path=renamed, sheet_registry=registry), observations)
            from tools.generate_mep_document_text import generate
            registry_path = Path(directory) / "registry.json"
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            output_dir = Path(directory) / "generated"
            manifest = generate(source=source, registry=registry_path, output_dir=output_dir)
            self.assertEqual(manifest["mode"], "automatic_native_text_only")
            self.assertFalse(manifest["assisted_replay_used"])
            self.assertFalse(manifest["item_inventory_complete"])
            self.assertEqual(manifest["quarantined_observation_count"], 2)
            self.assertEqual(manifest, generate(source=renamed, registry=registry_path, output_dir=output_dir))
            source.write_bytes(b"wrong source")
            with self.assertRaisesRegex(ValueError, "frozen M1 registry"):
                extract_mep_text_observations(pdf_path=source, sheet_registry=registry)

    def test_versioned_pack_and_system_proposal_preserve_provenance(self):
        pack = terminology_pack()
        self.assertEqual(pack["version"], "0.1.0")
        self.assertTrue(pack["contract"]["legend_membership_is_evidence_only"])

        proposals = propose_mep_interpretations(_observation("text.1", "CHWS"))
        self.assertEqual(len(proposals), 1)
        proposal = proposals[0]
        self.assertEqual(proposal["proposal_type"], "system")
        self.assertEqual(proposal["candidate"]["kind"], "chilled_water_supply")
        self.assertEqual(proposal["state"], "proposed")
        self.assertEqual(proposal["evidence_refs"], ["text.1"])
        self.assertEqual(proposal["method"]["terminology_pack_version"], "0.1.0")
        self.assertTrue(all(value is False for value in proposal["acceptance"].values()))

    def test_inline_pipe_and_duct_sizes_are_unit_explicit_proposals(self):
        proposals = propose_mep_interpretations(
            _observation("text.size", 'DN 50 / 24x12 IN / 1 1/2"')
        )
        sizes = [item["candidate"] for item in proposals if item["proposal_type"] == "inline_size"]
        self.assertIn(
            {"kind": "nominal_diameter", "value": 50.0, "unit": "mm", "designation": "DN", "raw_text": "DN 50"},
            sizes,
        )
        duct = next(item for item in sizes if item["kind"] == "rectangular_duct_size")
        self.assertEqual((duct["width"], duct["height"], duct["unit"]), (24.0, 12.0, "in"))
        inch = next(item for item in sizes if item.get("designation") == "inch_size")
        self.assertEqual(inch["value"], 1.5)
        self.assertTrue(all(item["state"] == "abstained" for item in proposals if item["proposal_type"] == "inline_size"))

    def test_bottom_and_centreline_elevations_remain_unbound(self):
        proposals = propose_mep_interpretations(
            _observation("text.elev", "BOP EL +12.450 M; C/L = 10'-0")
        )
        elevations = [item for item in proposals if item["proposal_type"] == "elevation"]
        self.assertEqual({item["candidate"]["basis"] for item in elevations}, {"bottom", "centreline"})
        self.assertTrue(all(item["acceptance"]["elevation_established"] is False for item in elevations))
        self.assertTrue(all(item["state"] == "abstained" for item in elevations))

    def test_architectural_bottom_elevation_is_parsed_without_decimal_loss(self):
        proposals = propose_mep_interpretations(
            _observation("text.be", 'BE= 25\' - 0 5/8"')
        )
        elevations = [item for item in proposals if item["proposal_type"] == "elevation"]
        self.assertEqual(len(elevations), 1)
        self.assertFalse(
            any(item["proposal_type"] == "inline_size" for item in proposals)
        )
        candidate = elevations[0]["candidate"]
        self.assertEqual((candidate["basis"], candidate["unit"]), ("bottom", "ft"))
        self.assertAlmostEqual(candidate["value"], 25 + 0.625 / 12)
        self.assertEqual(
            elevations[0]["method"]["name"],
            "bounded_architectural_elevation_parser",
        )

    def test_equipment_valve_fitting_damper_and_service_zone_coverage(self):
        observations = [
            _observation("equipment.1", "AHU-12"),
            _observation("valve.1", "BALL VALVE"),
            _observation("fitting.1", "REDUCER"),
            _observation("damper.1", "FIRE DAMPER"),
            _observation("zone.1", "SERVICE CLEARANCE"),
        ]
        payload = build_mep_terminology_proposals(document={"document_key": "synthetic"}, observations=observations)
        found = {(item["proposal_type"], item["candidate"]["kind"]) for item in payload["proposals"]}
        self.assertIn(("equipment", "equipment_tag"), found)
        self.assertIn(("component", "ball_valve"), found)
        self.assertIn(("component", "reducer"), found)
        self.assertIn(("component", "fire_damper"), found)
        self.assertIn(("service_zone", "service_access_zone"), found)
        self.assertEqual(validate_mep_terminology_proposals(payload), [])

    def test_symbol_continuation_riser_and_drop_are_neutral_proposals(self):
        observations = [
            _observation("symbol.cont", symbol_kind="continuation", evidence_channels=["native_vector_geometry"]),
            _observation("symbol.up", symbol_candidates=["riser"], evidence_channels=["raster_shape_proposal"]),
            _observation("symbol.down", symbol_kind="drop", evidence_channels=["native_vector_geometry"]),
            _observation("symbol.valve", symbol_kind="gate_valve", evidence_channels=["native_vector_geometry"]),
        ]
        payload = build_mep_terminology_proposals(document={}, observations=observations)
        kinds = {item["candidate"]["kind"] for item in payload["proposals"]}
        self.assertTrue({"continuation_symbol", "riser", "drop", "gate_valve"} <= kinds)
        for item in payload["proposals"]:
            self.assertFalse(item["acceptance"]["physical_connection_established"])
            self.assertFalse(item["acceptance"]["quantity_eligible"])

    def test_ambiguous_term_and_cross_observation_conflict_abstain(self):
        ambiguous = propose_mep_interpretations(_observation("text.fd", "FD"))
        self.assertEqual({item["candidate"]["kind"] for item in ambiguous}, {"fire_damper", "floor_drain"})
        self.assertTrue(all(item["state"] == "abstained" for item in ambiguous))
        self.assertTrue(all(item["alternatives"] for item in ambiguous))

        observations = [
            _observation("text.a", "CHWS", interpretation_scope_ref="scope.route-candidate"),
            _observation("text.b", "CHWR", interpretation_scope_ref="scope.route-candidate"),
        ]
        payload = build_mep_terminology_proposals(document={}, observations=observations)
        self.assertTrue(all(item["state"] == "abstained" for item in payload["proposals"]))
        self.assertTrue(all(item["conflicts"] for item in payload["proposals"]))
        self.assertTrue(all("conflicting_evidence_in_interpretation_scope" in item["reasons"] for item in payload["proposals"]))
        self.assertEqual(validate_mep_terminology_proposals(payload), [])

        mixed = build_mep_terminology_proposals(
            document={},
            observations=[
                _observation("text.fd", "FD", interpretation_scope_ref="scope.accessory"),
                _observation(
                    "text.fire-damper",
                    "FIRE DAMPER",
                    interpretation_scope_ref="scope.accessory",
                ),
            ],
        )
        floor_drain = next(
            item for item in mixed["proposals"] if item["candidate"]["kind"] == "floor_drain"
        )
        fire_dampers = [
            item for item in mixed["proposals"] if item["candidate"]["kind"] == "fire_damper"
        ]
        self.assertTrue(floor_drain["conflicts"])
        self.assertTrue(all(item["state"] == "abstained" for item in fire_dampers))
        self.assertTrue(
            all(
                any(conflict["candidate"]["kind"] == "floor_drain" for conflict in item["conflicts"])
                for item in fire_dampers
            )
        )

    def test_legend_membership_and_color_only_evidence_always_abstain(self):
        observations = [
            _observation("legend.chws", "CHWS", context="legend"),
            _observation(
                "colour.route",
                "CHWS",
                evidence_channels=["color"],
                color=[0.0, 0.5, 1.0],
            ),
        ]
        payload = build_mep_terminology_proposals(document={}, observations=observations)
        self.assertEqual(len(payload["proposals"]), 2)
        self.assertTrue(all(item["state"] == "abstained" for item in payload["proposals"]))
        self.assertIn("legend_membership_is_evidence_only", payload["proposals"][0]["reasons"])
        self.assertIn("color_only_evidence", payload["proposals"][1]["reasons"])
        self.assertFalse(payload["exchange_contract"]["route_or_system_identity_established"])
        self.assertFalse(payload["exchange_contract"]["physical_elevation_established"])
        self.assertFalse(payload["exchange_contract"]["confirmed_clash_established"])

    def test_validator_rejects_authority_and_fail_open_negatives(self):
        payload = build_mep_terminology_proposals(
            document={}, observations=[_observation("legend.1", "CHWS", context="legend")]
        )
        changed = copy.deepcopy(payload)
        changed["proposals"][0]["state"] = "proposed"
        self.assertTrue(any("legend proposal did not abstain" in item for item in validate_mep_terminology_proposals(changed)))

        changed = copy.deepcopy(payload)
        changed["proposals"][0]["acceptance"]["system_identity_established"] = True
        self.assertTrue(any("acquired engineering authority" in item for item in validate_mep_terminology_proposals(changed)))

        changed = copy.deepcopy(payload)
        changed["proposals"][0]["installed_length"] = 100
        self.assertTrue(any("forbidden engineering authority" in item for item in validate_mep_terminology_proposals(changed)))

        changed = copy.deepcopy(payload)
        changed["proposals"][0]["candidate"]["quantity"] = 1
        self.assertTrue(any("forbidden engineering authority" in item for item in validate_mep_terminology_proposals(changed)))

        changed = copy.deepcopy(payload)
        changed["proposals"][0]["candidate"]["system_identity_established"] = True
        self.assertTrue(any("authority flag(s) must remain false" in item for item in validate_mep_terminology_proposals(changed)))

        for key in ("physical_continuation_established", "installed_length_emitted"):
            changed = copy.deepcopy(payload)
            changed["proposals"][0]["candidate"][key] = True
            self.assertTrue(
                any(
                    "authority flag(s) must remain false" in item
                    for item in validate_mep_terminology_proposals(changed)
                ),
                key,
            )

    def test_same_semantic_candidate_in_one_scope_is_not_a_conflict_by_spelling(self):
        payload = build_mep_terminology_proposals(
            document={},
            observations=[
                _observation("text.long", "CHW SUPPLY", interpretation_scope_ref="scope.same"),
                _observation("text.short", "CHWS", interpretation_scope_ref="scope.same"),
            ],
        )
        systems = [item for item in payload["proposals"] if item["proposal_type"] == "system"]
        self.assertEqual(len(systems), 2)
        self.assertTrue(all(item["state"] == "proposed" for item in systems))
        self.assertTrue(all(item["conflicts"] == [] for item in systems))

    def test_stable_ids_are_filename_page_number_and_coordinate_neutral(self):
        first = _observation("native.text.shared", "DN 80", page_ref="page.by-content")
        first["bbox_display"] = [10, 20, 30, 40]
        first["source_filename"] = "first.pdf"
        second = copy.deepcopy(first)
        second["bbox_display"] = [910, 820, 930, 840]
        second["source_filename"] = "renamed.pdf"
        second["page_number"] = 77

        first_ids = [item["id"] for item in propose_mep_interpretations(first)]
        second_ids = [item["id"] for item in propose_mep_interpretations(second)]
        self.assertEqual(first_ids, second_ids)

    def test_production_module_contains_no_fixture_dispatch_or_coordinate_constants(self):
        source = inspect.getsource(mep_terminology_proposals)
        self.assertNotIn("M&P mark-up", source)
        self.assertNotIn("L01-MP-P.1A", source)
        self.assertNotIn("page_number ==", source)
        self.assertNotRegex(source, r"bbox_display\s*\[[^]]+\]\s*[<>]=?\s*\d")

    def test_generator_writes_a_valid_deterministic_artifact(self):
        input_payload = {
            "document": {"document_key": "pdf-sha256:example"},
            "observations": [_observation("text.1", "HUH-2 DN 40")],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.json"
            output = Path(directory) / "output.json"
            source.write_text(json.dumps(input_payload), encoding="utf-8")
            subprocess.run(
                [sys.executable, str(ROOT / 'tools/generate_mep_m2_fixture.py'), "--input", str(source), "--output", str(output)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            generated = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(validate_mep_terminology_proposals(generated), [])
        self.assertEqual(generated["summary"]["source_observation_count"], 1)


if __name__ == "__main__":
    unittest.main()
