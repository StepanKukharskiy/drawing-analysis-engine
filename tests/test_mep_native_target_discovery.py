from copy import deepcopy
import hashlib
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import fitz

from src.drawing_engine.disciplines.mep.mep_native_target_discovery import (
    discover_native_targets, iter_bounded_native_page_regions, validate_native_target_discovery,
)
from src.drawing_engine.disciplines.mep.mep_item_ocr_observations import extract_mep_item_ocr_observations
from src.drawing_engine.disciplines.mep.mep_sheet_registry import build_sheet_page_record, build_mep_sheet_registry
from src.drawing_engine.disciplines.mep.mep_text_observations import extract_mep_text_observations
from src.drawing_engine.core.vector_topology import extract_page_topology, iter_native_segments
from tools.generate_mep_extraction_coverage import build_coverage


class MepNativeTargetDiscoveryTest(unittest.TestCase):
    def fixture(self, directory, rotation=0, curve=False):
        path = Path(directory) / "neutral.pdf"
        with fitz.open() as pdf:
            page = pdf.new_page(width=400, height=300)
            page.draw_line((300, 200), (370, 200))
            page.insert_text((80, 80), '2" CHWS')
            page.draw_line((60, 86), (180, 86), color=(1, 0, 0))
            page.draw_line((60, 90), (180, 90), color=(0, 0, 1))
            page.insert_text((90, 170), 'BCP-4')
            page.draw_rect((80, 180, 120, 215))
            page.insert_text((250, 30), 'EQUIPMENT SCHEDULE\nAHU-12')
            if curve:
                t = (13200 - math.sqrt(13200 ** 2 - 4 * 10800 * 3000)) / (2 * 10800)
                height = lambda value: 3000 * value - 6600 * value ** 2 + 3600 * value ** 3
                offset = 128 - (height(t) + max(height(i / 16) for i in range(17))) / 2
                page.draw_bezier((10, offset), (110, 1000 + offset), (210, -200 + offset), (310, offset))
            page.set_rotation(rotation)
            pdf.new_page(width=400, height=300)
            pdf.save(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        registry = build_mep_sheet_registry(document={
            "document_key": f"pdf-sha256:{digest}", "source_pdf_sha256": digest,
            "source_bytes": path.stat().st_size, "page_count": 2,
        }, pages=[build_sheet_page_record(page_ref=f"page.{n}", page_number=n,
                    page_width=300 if rotation and n == 1 else 400,
                    page_height=400 if rotation and n == 1 else 300,
                    native_tokens=[], quality={"route": "native", "reason": "synthetic"}) for n in (1, 2)])
        observations = extract_mep_text_observations(pdf_path=path, sheet_registry=registry)
        return path, registry, observations

    def test_automatic_windows_keep_alternatives_and_shared_native_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path, registry, text = self.fixture(directory)
            result = discover_native_targets(pdf_path=path, sheet_registry=registry, text_observations=text)
            self.assertEqual(validate_native_target_discovery(result), [])
            self.assertEqual(len(result["pages"]), 2)
            self.assertFalse(result["pages"][1]["item_inventory_complete"])
            self.assertEqual(result["pages"][1]["stage_states"]["annotation_seeded_target_search"], "no_eligible_native_text_seeds")
            self.assertFalse(any("AHU-12" in row["text"] for row in result["searches"]))
            route = next(row for row in result["searches"] if "CHWS" in row["text"])
            self.assertGreaterEqual(len(route["primitive_candidate_refs"]), 2)
            self.assertFalse(route["unique_target_established"])
            self.assertTrue(any("BCP-4" in row["unclassified_tag_tokens"] for row in result["searches"]))
            with fitz.open(path) as pdf:
                canonical = {row["id"]: row for row in extract_page_topology(pdf[0])["segments"]}
                streamed = list(iter_native_segments(pdf[0].get_drawings()))
            for primitive in result["primitive_candidates"]:
                native = primitive["source_native_segment"]
                expected = canonical[native["id"]]
                self.assertEqual(native, {key: value for key, value in expected.items() if key not in {"start_vertex_id", "end_vertex_id"}})
            self.assertEqual(len(streamed), len(canonical))
            renamed = path.with_name("different-name.pdf")
            renamed.write_bytes(path.read_bytes())
            self.assertEqual(result, discover_native_targets(pdf_path=renamed, sheet_registry=registry, text_observations=text))

    def test_budget_cannot_manufacture_unique_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path, registry, text = self.fixture(directory)
            result = discover_native_targets(pdf_path=path, sheet_registry=registry, text_observations=text, max_candidates_per_page=1)
            self.assertEqual(len(result["primitive_candidates"]), 1)
            self.assertGreater(result["summary"]["budget_limited_search_count"], 0)
            self.assertFalse(result["pages"][0]["candidate_search_complete"])
            changed = deepcopy(result)
            row = next(row for row in changed["searches"] if row["unretained_candidate_count"])
            row["state"] = "searched_unbound"
            self.assertIn("truncated search must remain budget limited", validate_native_target_discovery(changed))
            changed = deepcopy(result)
            changed["searches"][0]["unique_target_established"] = True
            self.assertTrue(validate_native_target_discovery(changed))

    def test_rotation_and_source_hash_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path, registry, text = self.fixture(directory, rotation=90)
            result = discover_native_targets(pdf_path=path, sheet_registry=registry, text_observations=text)
            primitive = result["primitive_candidates"][0]
            native = primitive["source_native_segment"]
            with fitz.open(path) as pdf:
                self.assertEqual(primitive["points_display"][0], list(fitz.Point(native["start_display"]) * pdf[0].rotation_matrix))
            changed = deepcopy(result)
            changed["primitive_candidates"][0]["source_primitive_ref"] = "invented"
            self.assertIn("native primitive provenance mismatch", validate_native_target_discovery(changed))
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "source PDF hash"):
                discover_native_targets(pdf_path=path, sheet_registry=registry, text_observations=text)

    def test_coverage_rejects_other_document_and_never_closes_unrun_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            path, registry, text = self.fixture(directory)
            result = discover_native_targets(pdf_path=path, sheet_registry=registry, text_observations=text)
            coverage = build_coverage(registry=registry, native_text=text, native_targets=result)
            self.assertFalse(coverage["acceptance"]["full_marked_audit_ready"])
            self.assertIsNone(coverage["pages"][1]["item_recall"])
            self.assertIsNone(coverage["pages"][1]["certified_automatic_item_count"])
            self.assertEqual(coverage["pages"][0]["independent_declarations"]["state"], "not_processed")
            self.assertEqual(coverage["pages"][0]["item_text_ocr"]["state"], "not_processed")
            changed = deepcopy(result)
            changed["summary"]["primitive_candidate_count"] = 999999
            checked = build_coverage(registry=registry, native_text=text, native_targets=changed)
            self.assertEqual(checked["stage_summaries"]["native_targets"]["primitive_candidate_count"], len(result["primitive_candidates"]))
            partial = deepcopy(text)
            partial["pages"][0]["native_text_scan_complete"] = False
            checked = build_coverage(registry=registry, native_text=partial)
            self.assertEqual(checked["pages"][0]["native_text"]["state"], "partial_native_scan")
            changed = deepcopy(result)
            changed["document"]["source_pdf_sha256"] = "other document"
            with self.assertRaisesRegex(ValueError, "source document mismatch"):
                build_coverage(registry=registry, native_text=text, native_targets=changed)

    def test_orphan_and_disjoint_evidence_cannot_pass_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            path, registry, text = self.fixture(directory)
            result = discover_native_targets(pdf_path=path, sheet_registry=registry, text_observations=text)
            changed = deepcopy(result)
            orphan = deepcopy(changed["searches"][0])
            orphan.update(id="orphan", page_ref="unregistered", primitive_candidate_refs=[],
                          cross_boundary_primitive_refs=[], candidate_count=0)
            changed["searches"].append(orphan)
            self.assertIn("evidence references unregistered page", validate_native_target_discovery(changed))
            changed = deepcopy(result)
            changed["searches"][0]["search_rect_display"] = [-100000, -100000, -99999, -99999]
            self.assertIn("source search window replay mismatch", validate_native_target_discovery(changed))
            self.assertIn("candidate does not intersect search window", validate_native_target_discovery(changed))
            changed = deepcopy(result)
            changed["primitive_candidates"][0]["points_display"][0][0] += 5
            self.assertIn("native coordinate replay mismatch", validate_native_target_discovery(changed))

    def test_bounded_regions_ignore_page_cap_and_include_unlabelled_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            path, registry, text = self.fixture(directory)
            result = discover_native_targets(pdf_path=path, sheet_registry=registry, text_observations=text,
                region_size_display_points=64, max_candidates_per_region=20, max_candidates_per_page=1)
            self.assertEqual(validate_native_target_discovery(result), [])
            with fitz.open(path) as pdf:
                canonical = {row["id"] for row in iter_native_segments(pdf[0].get_drawings())}
            self.assertEqual({row["source_primitive_ref"] for row in result["primitive_candidates"]}, canonical)
            self.assertGreater(len(result["primitive_candidates"]), 1)
            far = next(row for row in result["primitive_candidates"] if row["bbox_display"] == [300, 200, 370, 200])
            linked = [row for row in result["searches"] if row["id"] in far["search_refs"]]
            self.assertTrue(all(row["seed_kind"] == "geometry_first_region" for row in linked))
            self.assertTrue(all(row["relevant_competitor_search_complete"] for row in result["regions"]))
            self.assertFalse(any(row["unique_target_established"] for row in result["searches"]))
            self.assertFalse(result["pages"][1]["item_inventory_complete"])

    def test_boundary_touching_strokes_are_alternatives_in_both_regions(self):
        with fitz.open() as pdf:
            page = pdf.new_page(width=200, height=100)
            page.draw_line((100, 10), (100, 90))
            page.draw_line((50, 30), (150, 30))
            page.draw_circle((90, 55), 2, color=None, fill=(0, 0, 0))
            packets = list(iter_bounded_native_page_regions(page, "page.neutral",
                region_size_display_points=100, max_candidates_per_region=50))
        self.assertEqual(len(packets), 2)
        left, right = packets
        refs = [{row["source_primitive_ref"] for row in packet["primitive_candidates"]} for packet in packets]
        self.assertTrue({"drawing[0].item[0].segment[0]", "drawing[1].item[0].segment[0]"} <= refs[0] & refs[1])
        crossing = next(row["id"] for row in left["primitive_candidates"] if row["source_primitive_ref"] == "drawing[1].item[0].segment[0]")
        self.assertIn(crossing, left["region"]["cross_boundary_primitive_refs"])
        self.assertIn(crossing, right["region"]["cross_boundary_primitive_refs"])
        self.assertTrue(any(row["source_native_segment"]["style"]["fill"] for row in left["primitive_candidates"]))

    def test_each_region_records_truncated_and_unsupported_competitors(self):
        with fitz.open() as pdf:
            page = pdf.new_page(width=200, height=100)
            page.draw_line((10, 10), (40, 10))
            page.draw_line((20, 20), (50, 20))
            page.draw_line((120, 20), (160, 20))
            drawings = page.get_drawings()
            drawings[2]["items"].append(("unknown-operator",))
            with patch.object(page, "get_drawings", return_value=drawings):
                packets = list(iter_bounded_native_page_regions(page, "page.neutral",
                    region_size_display_points=100, max_candidates_per_region=1))
        left, right = [packet["region"] for packet in packets]
        self.assertEqual(left["candidate_count"], 2)
        self.assertEqual(left["unretained_candidate_count"], 1)
        self.assertFalse(left["relevant_competitor_search_complete"])
        self.assertEqual(right["unretained_candidate_count"], 0)
        self.assertEqual(right["unsupported_native_item_kinds"], {"unknown-operator": 1})
        self.assertEqual(right["unsupported_native_item_refs"], ["drawing[2].item[1]"])
        self.assertFalse(right["relevant_competitor_search_complete"])

    def test_adaptive_refinement_splits_only_overloaded_regions_without_rescanning(self):
        with fitz.open() as pdf:
            page = pdf.new_page(width=128, height=64)
            for x, y in ((4, 4), (36, 4), (4, 36), (36, 36), (80, 10)):
                page.draw_line((x, y), (x + 8, y))
            drawings = page.get_drawings()
            canonical = {row["id"] for row in iter_native_segments(drawings)}
            with patch.object(page, "get_drawings", return_value=drawings) as extraction:
                packets = list(iter_bounded_native_page_regions(page, "page.neutral",
                    region_size_display_points=64, max_candidates_per_region=1,
                    minimum_region_size_display_points=8))
                self.assertEqual(extraction.call_count, 1)
        regions = [packet["region"] for packet in packets]
        self.assertEqual(len(regions), 5)
        self.assertEqual(sum((row["bbox_display"][2] - row["bbox_display"][0]) *
                             (row["bbox_display"][3] - row["bbox_display"][1]) for row in regions), 128 * 64)
        for i, left in enumerate(regions):
            for right in regions[i + 1:]:
                self.assertEqual((fitz.Rect(left["bbox_display"]) & fitz.Rect(right["bbox_display"])).get_area(), 0)
        cold = next(row for row in regions if row["bbox_display"] == [64, 0, 128, 64])
        self.assertNotIn("refinement_ancestry", cold)
        self.assertTrue(all(row["relevant_competitor_search_complete"] for row in regions))
        self.assertEqual({row["source_primitive_ref"] for packet in packets for row in packet["primitive_candidates"]}, canonical)
        self.assertTrue(all(len(packet["primitive_candidates"]) <= 1 for packet in packets))
        self.assertTrue(all(not key.startswith("_") for region in regions for key in region))
        region_ids = {row["id"] for row in regions}
        self.assertTrue(all(set(row["search_refs"]) <= region_ids for packet in packets for row in packet["primitive_candidates"]))

    def test_adaptive_minimum_does_not_hide_coincident_missing_competitors(self):
        with fitz.open() as pdf:
            page = pdf.new_page(width=64, height=64)
            for _ in range(10):
                page.draw_line((1, 2), (3, 2))
            packets = list(iter_bounded_native_page_regions(page, "page.neutral",
                region_size_display_points=64, max_candidates_per_region=1,
                minimum_region_size_display_points=8))
        blocked = [packet["region"] for packet in packets if not packet["region"]["relevant_competitor_search_complete"]]
        self.assertEqual(len(blocked), 1)
        self.assertEqual(blocked[0]["bbox_display"], [0, 0, 8, 8])
        self.assertEqual(blocked[0]["candidate_count"], 10)
        self.assertEqual(blocked[0]["unretained_candidate_count"], 9)
        self.assertEqual(blocked[0]["state"], "budget_limited")
        self.assertIn("minimum_region_size_reached_with_unretained_competitors", blocked[0]["unresolved_reasons"])
        self.assertFalse(blocked[0]["unique_target_established"])

    def test_adaptive_shared_boundaries_keep_identical_unclipped_sources(self):
        with fitz.open() as pdf:
            page = pdf.new_page(width=64, height=64)
            page.draw_line((32, 0), (32, 64))
            page.draw_line((4, 4), (12, 4))
            page.draw_line((52, 52), (60, 52))
            packets = list(iter_bounded_native_page_regions(page, "page.neutral",
                region_size_display_points=64, max_candidates_per_region=1,
                minimum_region_size_display_points=8))
        shared = [row for packet in packets for row in packet["primitive_candidates"]
                  if row["source_primitive_ref"] == "drawing[0].item[0].segment[0]"]
        self.assertGreater(len(shared), 1)
        self.assertTrue(all(row == shared[0] for row in shared))
        self.assertEqual(shared[0]["points_display"], [[32, 0], [32, 64]])
        self.assertTrue(all(packet["region"]["relevant_competitor_search_complete"] for packet in packets))

    def test_curve_control_hull_covers_between_sample_excursion_without_inventing_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, _ = self.fixture(directory, curve=True)
            with fitz.open(path) as pdf:
                packets = list(iter_bounded_native_page_regions(pdf[0], "page.1",
                    region_size_display_points=128, max_candidates_per_region=100))
                canonical = next(row for row in iter_native_segments(pdf[0].get_drawings()) if row["kind"] == "cubic")
            packet = next(packet for packet in packets if packet["region"]["bbox_display"] == [0, 128, 128, 256])
            curve = next(row for row in packet["primitive_candidates"] if row["source_native_segment"]["kind"] == "cubic")
            native = curve["source_native_segment"]
            t = (13200 - math.sqrt(13200 ** 2 - 4 * 10800 * 3000)) / (2 * 10800)
            controls = [native["start_display"], *native["control_points_display"], native["end_display"]]
            actual_y = sum(weight * p[1] for weight, p in zip(
                [(1-t)**3, 3*t*(1-t)**2, 3*t*t*(1-t), t**3], controls))
            self.assertLess(curve["bbox_display"][3], 128)
            self.assertGreater(actual_y, 128)
            self.assertGreater(curve["search_bbox_display"][3], 700)
            self.assertEqual(curve["source_primitive_ref"], canonical["id"])
            self.assertEqual(native["sample_points_display"], [list(point) for point in canonical["sample_points_display"]])
            self.assertTrue(all(point[1] < 128 for point in curve["points_display"]))
            self.assertFalse(curve["search_bbox_is_geometry"])
            self.assertFalse(curve["quantity_eligible"])
            overincluded = next(packet for packet in packets if packet["region"]["bbox_display"] == [0, 256, 128, 300])
            self.assertIn(curve["id"], overincluded["region"]["primitive_candidate_refs"])
            self.assertLess(actual_y, 256)

    def test_curve_search_bounds_replay_in_legacy_and_region_modes_and_rotation(self):
        for rotation in (0, 90):
            with self.subTest(rotation=rotation), tempfile.TemporaryDirectory() as directory:
                path, registry, text = self.fixture(directory, rotation=rotation, curve=True)
                for region_size in (None, 64):
                    with self.subTest(region_size=region_size):
                        result = discover_native_targets(pdf_path=path, sheet_registry=registry,
                            text_observations=text, search_radius_in_text_heights=.1,
                            region_size_display_points=region_size)
                        self.assertEqual(validate_native_target_discovery(result), [])
                        curve = next(row for row in result["primitive_candidates"] if row["source_native_segment"]["kind"] == "cubic")
                        seed = next(row for row in result["searches"] if row["text"] == "BCP-4")
                        self.assertIn(curve["id"], seed["primitive_candidate_refs"])
                        self.assertTrue((fitz.Rect(curve["bbox_display"]) & fitz.Rect(seed["search_rect_display"])).is_empty)
                        changed = deepcopy(result)
                        candidate = next(row for row in changed["primitive_candidates"] if row["id"] == curve["id"])
                        candidate["search_bbox_display"] = candidate["bbox_display"]
                        self.assertIn("native conservative search bounds replay mismatch", validate_native_target_discovery(changed))
                        changed = deepcopy(result)
                        changed["primitive_candidates"][0].pop("search_bbox_display")
                        self.assertIn("conservative search bounds missing", validate_native_target_discovery(changed))
                        changed = deepcopy(result)
                        changed["method"].pop("search_bounds")
                        self.assertIn("conservative search bounds method missing", validate_native_target_discovery(changed))

    def test_adaptive_curve_hull_alternatives_cannot_disappear_during_refinement(self):
        with fitz.open() as pdf:
            page = pdf.new_page(width=128, height=256)
            page.draw_bezier((10, 0), (40, 256), (80, -256), (110, 0))
            page.draw_line((20, 200), (25, 200))
            packets = list(iter_bounded_native_page_regions(page, "page.1", region_size_display_points=128,
                max_candidates_per_region=1, minimum_region_size_display_points=8))
        # The sampled curve is nowhere near y=200, but its conservative search
        # hull still competes there; subdivision must not invent completeness.
        blocked = [packet for packet in packets if packet["region"]["bbox_display"][1] >= 128
                   and packet["region"]["candidate_count"] == 2]
        self.assertTrue(blocked)
        self.assertTrue(all(not packet["region"]["relevant_competitor_search_complete"] for packet in blocked))
        self.assertTrue(all(packet["region"]["unretained_candidate_count"] == 1 for packet in blocked))

    def test_annotation_completeness_requires_all_relevant_regions(self):
        with tempfile.TemporaryDirectory() as directory:
            path, registry, text = self.fixture(directory)
            result = discover_native_targets(pdf_path=path, sheet_registry=registry, text_observations=text,
                region_size_display_points=64, max_candidates_per_region=1)
            self.assertEqual(validate_native_target_discovery(result), [])
            self.assertTrue(any(not row["relevant_competitor_search_complete"] for row in result["searches"]
                                if row.get("seed_kind") != "geometry_first_region"))
            changed = deepcopy(result)
            annotation = next(row for row in changed["searches"] if row.get("seed_kind") != "geometry_first_region"
                              and not row["relevant_competitor_search_complete"])
            annotation["relevant_competitor_search_complete"] = True
            self.assertIn("relevant competitor completeness mismatch", validate_native_target_discovery(changed))
            changed = deepcopy(result)
            changed["regions"].pop()
            self.assertIn("region evidence membership mismatch", validate_native_target_discovery(changed))

    def test_page_execution_scope_never_claims_unselected_pages_processed(self):
        with tempfile.TemporaryDirectory() as directory:
            path, registry, text = self.fixture(directory)
            result = discover_native_targets(pdf_path=path, sheet_registry=registry, text_observations=text,
                region_size_display_points=100, page_numbers=[1])
            self.assertEqual(validate_native_target_discovery(result), [])
            second = result["pages"][1]
            self.assertFalse(second["candidate_search_complete"])
            self.assertEqual(second["region_refs"], [])
            self.assertEqual(set(second["stage_states"].values()), {"not_processed"})

    @patch('src.drawing_engine.disciplines.mep.mep_item_ocr_observations.pytesseract.get_tesseract_version', return_value='test-engine')
    @patch('src.drawing_engine.disciplines.mep.mep_item_ocr_observations._run_ocr')
    def test_eligible_ocr_seeds_keep_overlap_alternatives_and_quarantine(self, ocr, version):
        ocr.return_value = {"line_num": [1, 1, 2], "text": ['2"', 'CHWS', 'VALVE'],
                            "left": [160, 190, 160], "top": [136, 136, 180],
                            "width": [25, 60, 60], "height": [24, 24, 24], "conf": [98, 96, 20],
                            "block_num": [1, 1, 1], "par_num": [1, 1, 1]}
        with tempfile.TemporaryDirectory() as directory:
            path, registry, text = self.fixture(directory)
            with patch('src.drawing_engine.disciplines.mep.mep_item_ocr_observations.page_ocr_selection', return_value={"selected": True}):
                recovered = extract_mep_item_ocr_observations(pdf_path=path, sheet_registry=registry,
                    native_text_observations=text, render_scale=2, tile_pixels=1024)
            result = discover_native_targets(pdf_path=path, sheet_registry=registry, text_observations=text,
                region_size_display_points=100, item_ocr_observations=recovered)
            self.assertEqual(validate_native_target_discovery(result), [])
            seeds = [row for row in result["searches"] if row.get("seed_kind") == "ocr_annotation"]
            self.assertEqual(len(seeds), 2)
            self.assertTrue(all(row["text"] == '2" CHWS' for row in seeds))
            self.assertTrue(any(row["native_overlap_alternatives"] for row in seeds))
            self.assertTrue(all(row["source_observation"]["confidence"] == .96 for row in seeds))
            self.assertTrue(any(row["page_ref"] == "page.2" for row in seeds))
            changed = deepcopy(recovered)
            changed["m1_payload_sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "OCR observations do not match"):
                discover_native_targets(pdf_path=path, sheet_registry=registry, text_observations=text,
                    region_size_display_points=100, item_ocr_observations=changed)
            changed = deepcopy(recovered)
            changed["observations"][0]["bbox_display"][0] += 3
            with self.assertRaisesRegex(ValueError, "invalid OCR evidence"):
                discover_native_targets(pdf_path=path, sheet_registry=registry, text_observations=text,
                    region_size_display_points=100, item_ocr_observations=changed)
