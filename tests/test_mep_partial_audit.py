"""The partial renderer preserves frozen authority, provenance and source pages."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import fitz

from src.drawing_engine.audit.render_mep_partial_audit import (
    DEFAULT_INPUTS, build_automatic_manifest, build_manifest, render, render_items, render_compact_items, render_overlay,
    _item_presentation, _human_measurement, _clip_line, _detail_window, _overlay_colour,
    _readable_line_parts, _readable_highlight, _readable_anchor, _source_text_boxes, _human_closeups,
    _cached_detail_raster, _cached_source_raster, _source_page_sha256,
)
from src.drawing_engine.project.takeoff_intelligence import canonical_sha256


class MepPartialAuditTest(unittest.TestCase):
    def setUp(self):
        self.inputs = {name: json.loads(path.read_text()) for name, path in DEFAULT_INPUTS.items()}

    def automatic_inputs(self):
        # Reuse only the frozen geometry as a renderer fixture, not as evidence
        # that this historical reviewed selection was automatically discovered.
        inputs = {name: deepcopy(value) for name, value in self.inputs.items() if name != "coverage"}
        targets = {}
        for occurrence in inputs["catalog"]["item_occurrences"]:
            ref = occurrence["target_refs"][0]
            target = targets.setdefault(ref, {"id": ref, "page_ref": occurrence["source"]["page_ref"], "item_occurrence_refs": []})
            target["item_occurrence_refs"].append(occurrence["id"])
        inputs["discovery"] = {
            "document": deepcopy(inputs["catalog"]["document"]),
            "execution_mode": "automatic_frozen_replay",
            "authority": {"reviewed_selectors_used": False},
            "input_payload_sha256": {name: canonical_sha256(value) for name, value in inputs.items()},
            "accepted_targets": list(targets.values()),
            "pages": [{"page_ref": row["page_ref"], "discovery_state": "bounded_search_processed",
                       "item_inventory_complete": False} for row in inputs["catalog"]["pages"]],
            "unresolved_proposals": [],
        }
        return inputs

    def test_source_page_raster_cache_is_content_addressed_and_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pdf"
            with fitz.open() as pdf:
                for text in ("unchanged", "sibling"):
                    page = pdf.new_page(width=200, height=120)
                    page.insert_text((20, 40), text)
                pdf.save(source)
            with fitz.open(source) as pdf:
                digest = _source_page_sha256(pdf, 1)
                first, receipt, reused = _cached_source_raster(
                    pdf[0], page_number=1, cache_dir=root / "cache", raster_dpi=120)
                self.assertFalse(reused)
                second, same_receipt, reused = _cached_source_raster(
                    pdf[0], page_number=1, cache_dir=root / "cache", raster_dpi=120)
            self.assertTrue(reused)
            self.assertEqual(first, second)
            self.assertEqual(receipt, same_receipt)
            self.assertEqual(receipt["source_page_sha256"], digest)
            self.assertFalse((root / "cache" / (receipt["cache_key"] + ".jpg.partial")).exists())
            with fitz.open(source) as pdf:
                clip = fitz.Rect(0, 0, 100, 80)
                first, detail_receipt, reused = _cached_detail_raster(
                    pdf[0], source_page_sha256=digest, clip=clip, cache_dir=root / "cache")
                self.assertFalse(reused)
                second, _, reused = _cached_detail_raster(
                    pdf[0], source_page_sha256=digest, clip=clip, cache_dir=root / "cache")
            self.assertTrue(reused)
            self.assertEqual(first, second)
            self.assertEqual(detail_receipt["source_page_sha256"], digest)

    def test_auto_render_reuses_hash_bound_size_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pdf"
            output = root / "audit.pdf"
            with fitz.open() as pdf:
                pdf.new_page(width=200, height=120)
                pdf.save(source)
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            output.write_bytes(b"prior")
            output.with_suffix(".manifest.json").write_text(json.dumps({
                "document": {"source_pdf_sha256": source_hash},
                "presentation": {"source_rendering": "raster"},
            }))
            output.with_suffix(".size-comparison.json").write_text(json.dumps({
                "bytes": {"vector": 200, "raster": 100},
                "selected": "raster", "raster_dpi": 120,
            }))

            def fake_render(_source, path, _manifest, **_options):
                path.write_bytes(b"new-raster")
                path.with_suffix(".manifest.json").write_text("{}")

            with patch("src.drawing_engine.audit.render_mep_partial_audit.render_items",
                       side_effect=fake_render) as renderer:
                comparison = render_compact_items(
                    source, output, {}, page_cache_dir=root / "cache")
            self.assertEqual(renderer.call_count, 1)
            self.assertEqual(renderer.call_args.kwargs["source_rendering"], "raster")
            self.assertTrue(comparison["selection_reused"])
            self.assertEqual(output.read_bytes(), b"new-raster")
            self.assertTrue((root / "cache/source-rendering-policy.json").exists())
            with patch("src.drawing_engine.audit.render_mep_partial_audit.render_items") as renderer:
                reused = render_compact_items(
                    source, output, {}, page_cache_dir=root / "cache")
            renderer.assert_not_called()
            self.assertTrue(reused["complete_output_reused"])

    def test_human_labels_alias_frozen_ids_and_share_rebar_colors(self):
        from src.drawing_engine.audit.audit_presentation import item_palette_color
        from src.drawing_engine.audit.render_object_agnostic_audit import _detail_palette_color
        manifest = build_automatic_manifest(self.automatic_inputs())
        frozen = {row["id"]: deepcopy(row["occurrence"]) for row in manifest["item_rows"]}
        rows = _item_presentation(manifest)
        self.assertEqual([row["presentation"]["label"] for row in rows], ["E01", "E02", "E03", "E04"])
        self.assertEqual(len({tuple(row["presentation"]["color_rgb"]) for row in rows}), 4)
        for index, row in enumerate(rows):
            self.assertEqual(row["occurrence"], frozen[row["id"]])
            self.assertEqual(tuple(row["presentation"]["color_rgb"]), item_palette_color(index))
            self.assertEqual(item_palette_color(index), _detail_palette_color(index))
            self.assertFalse(row["presentation"]["designer_part_mark"])
        manifest["item_rows"].reverse()
        self.assertEqual(rows, _item_presentation(manifest))
        self.assertEqual(_human_measurement(25.13541667, "ft"), '25\' - 1 5/8"')
        self.assertEqual(_human_measurement(2.5, "in"), "2 1/2 in")
        self.assertEqual(_human_measurement(None, "in"), "not identified")
        self.assertEqual(_human_measurement(1.23456, "ft"), "1.235 ft")

    def test_reading_manifest_compacts_only_region_index_after_hash_validation(self):
        inputs = self.automatic_inputs()
        regions = [{"relevant_competitor_search_complete": True, "primitive_candidate_refs": ["native.1"]},
                   {"relevant_competitor_search_complete": False, "primitive_candidate_refs": ["native.2"]}]
        inputs["discovery"]["pages"][0]["regions"] = regions
        before = deepcopy(inputs)
        full = build_automatic_manifest(inputs)
        compact = build_automatic_manifest(inputs, compact_regions=True)
        self.assertEqual(inputs, before)
        self.assertEqual(full["item_rows"], compact["item_rows"])
        self.assertEqual(full["unresolved_proposals"], compact["unresolved_proposals"])
        self.assertEqual(full["input_payload_sha256"], compact["input_payload_sha256"])
        record = compact["pages"][0]["discovery_record"]
        self.assertNotIn("regions", record)
        self.assertEqual(record["region_search_summary"], {
            "region_count": 2, "incomplete_region_count": 1,
            "full_region_evidence_sha256": canonical_sha256(regions),
            "full_evidence_contract": "input_payload_sha256.discovery"})
        self.assertEqual(full["pages"][0]["discovery_record"]["regions"], regions)

    def test_closeup_clipping_preserves_crossing_and_outside_segments(self):
        rect = fitz.Rect(0, 0, 10, 10)
        self.assertEqual(_clip_line([-5, 5], [15, 5], rect), [fitz.Point(0, 5), fitz.Point(10, 5)])
        self.assertIsNone(_clip_line([-5, -1], [15, -1], rect))
        self.assertEqual(_clip_line([5, -5], [5, 15], rect), [fitz.Point(5, 0), fitz.Point(5, 10)])

    def test_compact_edition_selects_actual_smaller_file_without_overwriting_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pdf"
            source.write_bytes(b"immutable source")
            for case, (sizes, winner) in enumerate((
                    ({"vector": 100, "raster": 80}, "raster"),
                    ({"vector": 80, "raster": 100}, "vector"),
                    ({"vector": 80, "raster": 80}, "vector"))):
                output = Path(directory) / f"audit-{case}.pdf"
                def fake_render(source, path, manifest, *, source_rendering, raster_dpi,
                                page_cache_dir=None):
                    path.write_bytes(source_rendering[0].encode() * sizes[source_rendering])
                    path.with_suffix(".manifest.json").write_text(json.dumps({"mode": source_rendering}))
                with patch("src.drawing_engine.audit.render_mep_partial_audit.render_items", side_effect=fake_render):
                    comparison = render_compact_items(source, output, {})
                self.assertEqual(comparison["selected"], winner)
                self.assertEqual(output.stat().st_size, min(sizes.values()))
                self.assertEqual(json.loads(output.with_suffix(".manifest.json").read_text()), {"mode": winner})
                self.assertEqual(source.read_bytes(), b"immutable source")
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                render_compact_items(source, source, {})
            self.assertEqual(source.read_bytes(), b"immutable source")

    def test_highlight_gaps_preserve_text_pixels_and_overlapping_note_boxes(self):
        boxes = [fitz.Rect(30, 20, 45, 40), fitz.Rect(40, 10, 60, 35)]
        parts = _readable_line_parts([0, 30], [100, 30], boxes, 2)
        self.assertEqual(len(parts), 2)
        for actual, expected in zip([value for pair in parts for point in pair for value in point],
                                    [0, 30, 28, 30, 62, 30, 100, 30]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(_readable_line_parts([50, 30], [50, 30], boxes, 2), [])
        with fitz.open() as pdf:
            page = pdf.new_page(width=300, height=200)
            page.insert_text((30, 60), '2 1/2 in HHWS', fontsize=16)
            page.add_freetext_annot(fitz.Rect(70, 90, 230, 130), 'Coordinate clearances around VFDs',
                                    fontsize=12, fill_color=(1, 1, 0)).update()
            protected = _source_text_boxes(page)
            # Outlined PDF lettering has no native text box to mask. Multiply
            # must keep its dark glyph pixels intact too, without a white halo.
            outlined_glyph = fitz.Rect(265, 42, 270, 58)
            page.draw_rect(outlined_glyph, fill=(0, 0, 0), color=(0, 0, 0))
            glyph_before = page.get_pixmap(clip=outlined_glyph).samples
            before = [page.get_pixmap(clip=box, matrix=fitz.Matrix(2, 2)).samples for box in protected]
            _readable_highlight(page, [[0, 50], [300, 50], [100, 110], [0, 110]], protected, (0, 0, 1), 8, 14)
            _readable_anchor(page, fitz.Point(100, 110), protected, (0, 0, 1), 4, 2)
            self.assertEqual(before, [page.get_pixmap(clip=box, matrix=fitz.Matrix(2, 2)).samples for box in protected])
            self.assertEqual(page.get_pixmap(clip=outlined_glyph).samples, glyph_before)

    def test_vector_closeups_include_original_note_appearances(self):
        with fitz.open() as original, fitz.open() as output:
            page = original.new_page(width=500, height=500)
            page.add_freetext_annot(fitz.Rect(160, 185, 260, 215), 'Original note', fontsize=12,
                                    fill_color=(1, 1, 0)).update()
            output.new_page(width=500, height=500)
            row = {"source": {"pdf_page_number": 1, "drawing_sheet_number": "Test"},
                   "presentation": {"label": "E01", "title": "Pipe segment", "color_rgb": [0, 0, 1], "fields": []},
                   "source_leader_anchor_display": [200, 200], "centreline_points_display": [[0, 200], [500, 200]]}
            _human_closeups(output, original, [row], {1: 0})
            self.assertIn("Original note", output[1].get_text())

    def test_highlights_preserve_inherited_font_resources(self):
        with fitz.open() as pdf:
            page = pdf.new_page(width=300, height=200)
            page.insert_text((30, 60), "Original source text", fontsize=16)
            resources = pdf.xref_get_key(page.xref, "Resources")[1]
            resource_xref = int(resources.split()[0])
            pdf.xref_set_key(resource_xref, "ExtGState/MepTextSafe", "<< /BM /Screen >>")
            parent = int(pdf.xref_get_key(page.xref, "Parent")[1].split()[0])
            pdf.xref_set_key(parent, "Resources", resources)
            pdf.xref_set_key(page.xref, "Resources", "null")
            page = pdf.reload_page(page)
            clip = fitz.Rect(20, 35, 200, 75)
            before = page.get_pixmap(clip=clip).samples
            _readable_highlight(page, [[0, 120], [300, 120]], [], (0, 0, 1), 8)
            self.assertEqual(before, page.get_pixmap(clip=clip).samples)
            self.assertIn("Original source text", page.get_text())
            self.assertIn("Screen", pdf.xref_get_key(resource_xref, "ExtGState/MepTextSafe")[1])
            self.assertIn("Multiply", pdf.xref_get_key(resource_xref, "ExtGState/MepTextSafe1")[1])

    def test_detail_crop_contains_accepted_label_and_contact_not_remote_endpoint(self):
        row = {"source_leader_anchor_display": [40, 40], "source_label_context": [
            {"observation_ref": "text.1", "bbox_display": [650, 570, 780, 590],
             "contact_points_display": [[610, 610]], "relation_refs": ["system", "size"]},
            {"observation_ref": "text.2", "bbox_display": [650, 594, 805, 616],
             "contact_points_display": [[610, 610]], "relation_refs": ["elevation"]},
        ]}
        clip, contact, refs = _detail_window(row, fitz.Rect(0, 0, 1000, 1000))
        self.assertEqual(contact, fitz.Point(610, 610))
        self.assertEqual(refs, ["text.1", "text.2"])
        self.assertFalse(clip.contains(fitz.Point(40, 40)))
        self.assertTrue(all(clip.contains(fitz.Rect(entry["bbox_display"])) for entry in row["source_label_context"]))

    def test_source_label_context_requires_hash_linked_bindings_and_terminology(self):
        from src.drawing_engine.audit.render_mep_partial_audit import FIXTURE
        inputs = self.automatic_inputs()
        inputs["bindings"] = json.loads((FIXTURE / "real_m4_discrete_coverage/full_package.attribute-bindings.json").read_text())
        inputs["terminology"] = json.loads((FIXTURE / "real_m4_discrete_coverage/full_package.terminology-proposals.json").read_text())
        result = build_automatic_manifest(inputs)
        self.assertTrue(all(row["source_label_context"] for row in result["item_rows"]))
        inputs["terminology"]["source_observations"][0]["text"] = "Unrelated modified text"
        with self.assertRaisesRegex(ValueError, "terminology: frozen binding hash mismatch"):
            build_automatic_manifest(inputs)
        del inputs["terminology"]
        with self.assertRaisesRegex(ValueError, "requires both"):
            build_automatic_manifest(inputs)

    def test_overlay_colours_preserve_native_paint_without_inventing_identity(self):
        row = {"source": {"page_ref": "page.1"}, "source_primitive_refs": ["left", "right"],
               "occurrence": {"system": None}}
        before = deepcopy(row)
        colours = {("page.1", "left"): {(1., .5, 0.)}, ("page.1", "right"): {(1., .5, 0.)}}
        self.assertEqual(_overlay_colour(row, colours), ([1., .5, 0.], "native_source_strokes"))
        self.assertEqual(row, before)
        colours[("page.1", "right")].add((0., 0., 1.))
        self.assertEqual(_overlay_colour(row, colours)[1], "unresolved_source_colour")
        other_page = {("page.2", "left"): {(1., .5, 0.)}, ("page.2", "right"): {(1., .5, 0.)}}
        self.assertEqual(_overlay_colour(row, other_page)[1], "unresolved_source_colour")
        row["occurrence"]["system"] = {"kind": "heating_hot_water_return"}
        self.assertEqual(_overlay_colour(row, colours)[1], "accepted_system_palette")

    def test_static_overlay_fades_source_and_keeps_opaque_geometry_without_links(self):
        manifest = build_automatic_manifest(self.automatic_inputs())
        manifest["pages"] = manifest["pages"][:3]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.pdf", root / "overlay.pdf"
            with fitz.open() as pdf:
                for _ in range(3):
                    page = pdf.new_page(width=3456, height=2592)
                    page.draw_rect(fitz.Rect(0, 0, 20, 20), fill=(0, 0, 0))
                pdf.save(source)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest["document"].update(source_pdf_sha256=digest, page_count=3)
            before = deepcopy(manifest)
            report = render_overlay(source, output, manifest, page_cache_dir=root / "cache", raster_dpi=96)
            self.assertEqual(manifest, before)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)
            with fitz.open(output) as pdf:
                self.assertEqual(len(pdf), 3)
                self.assertEqual(json.loads(pdf.embfile_get("audit-manifest.json")), report)
                strokes = []
                for page in pdf:
                    self.assertFalse(page.get_links())
                    self.assertFalse(list(page.annots() or []))
                    self.assertTrue(page.get_images())
                    self.assertTrue(120 < page.get_pixmap().pixel(5, 5)[0] < 150)
                    strokes.extend(r for r in page.get_drawings() if r["type"] == "s")
                self.assertEqual(len(strokes), len(manifest["item_rows"]))
                self.assertTrue(all(r["stroke_opacity"] == 1 for r in strokes))

    def test_human_vector_and_raster_editions_keep_items_and_readable_links(self):
        manifest = build_automatic_manifest(self.automatic_inputs())
        manifest["pages"] = manifest["pages"][:3]
        manifest["original_annotation_observations"] = []
        for record in manifest["pages"]:
            record["discovery_record"]["regions"] = [
                {"relevant_competitor_search_complete": True, "primitive_candidate_refs": ["native.1"]},
                {"relevant_competitor_search_complete": False, "primitive_candidate_refs": ["native.2"]},
            ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pdf"
            with fitz.open() as pdf:
                for index in range(3):
                    page = pdf.new_page(width=3456, height=2592)
                    page.insert_text((50, 50), f"Original source {index + 1}", fontsize=20)
                    page.add_text_annot((100, 100), "Original engineer note").update()
                pdf.save(source)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest["document"].update(source_pdf_sha256=digest, page_count=3)
            before = deepcopy(manifest)
            for mode in ("vector", "raster"):
                output = Path(directory) / (mode + ".pdf")
                result = render_items(source, output, manifest, source_rendering=mode, raster_dpi=96)
                self.assertEqual(manifest, before)
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)
                self.assertEqual(result["presentation"]["processed_source_page_numbers"], [1, 2, 3])
                self.assertEqual(result["presentation"]["source_rendering"], mode)
                with fitz.open(output) as pdf:
                    self.assertEqual(len(pdf), 5)
                    self.assertEqual(json.loads(pdf.embfile_get("audit-manifest.json")), result)
                    text = " ".join(" ".join(page.get_text().split()) for page in pdf)
                    self.assertIn("3 of 3 source pages processed", text)
                    self.assertIn("No elements identified yet", text)
                    self.assertNotIn("mep_item_occurrence.", text)
                    self.assertNotIn("mep_outlined_route_composite.", text)
                    for record in result["pages"]:
                        page = pdf[record["audit_source_page_number"] - 1]
                        compact = record["discovery_record"]
                        self.assertNotIn("regions", compact)
                        self.assertEqual(compact["region_search_summary"]["region_count"], 2)
                        self.assertEqual(compact["region_search_summary"]["incomplete_region_count"], 1)
                        self.assertEqual(compact["region_search_summary"]["full_region_evidence_sha256"],
                                         canonical_sha256(before["pages"][0]["discovery_record"]["regions"]))
                        self.assertEqual(record["original_annotation_count"], 1)
                        self.assertEqual(sum(1 for _ in page.annots() or []), 0 if mode == "raster" else 1)
                        if mode == "raster":
                            self.assertTrue(page.get_images())
                            self.assertTrue(record["source_raster"]["original_annotations_rendered"])
                    for row in result["item_rows"]:
                        source_index = row["source"]["pdf_page_number"]
                        detail_index = row["audit_detail_page_number"] - 1
                        self.assertIn(row["presentation"]["label"], pdf[source_index].get_text())
                        self.assertIn(row["presentation"]["title"], " ".join(pdf[detail_index].get_text().replace("\u00ad", "-").split()))
                        self.assertTrue(any(link["page"] == detail_index for link in pdf[source_index].get_links()))
                        self.assertTrue(any(link["page"] == source_index for link in pdf[detail_index].get_links()))
                        self.assertEqual(row["occurrence"]["counts"]["physical_instance_count"], None)

    def test_automatic_packet_preserves_ids_without_reviewed_coverage(self):
        inputs = self.automatic_inputs()
        before = deepcopy(inputs)
        manifest = build_automatic_manifest(inputs)
        self.assertEqual(inputs, before)
        self.assertEqual(manifest, build_manifest(inputs))
        self.assertEqual(manifest["execution_mode"], "automatic_frozen_replay")
        self.assertEqual(manifest["reviewed_unresolved_findings"], [])
        self.assertEqual([row["id"] for row in manifest["item_rows"]],
                         sorted(row["id"] for row in inputs["catalog"]["item_occurrences"]))
        self.assertFalse(manifest["coverage_complete"])
        self.assertTrue(all(row["coverage_record"] is None for row in manifest["pages"]))
        self.assertTrue(all(row["discovery_record"]["item_inventory_complete"] is False for row in manifest["pages"]))

    def test_automatic_packet_requires_hashes_selector_exclusion_and_exact_membership(self):
        cases = [
            (lambda packet: packet["authority"].update(reviewed_selectors_used=True), "exclude reviewed selectors"),
            (lambda packet: packet["input_payload_sha256"].update(annotations="0" * 64), "automatic discovery hash mismatch"),
            (lambda packet: packet["accepted_targets"][0]["item_occurrence_refs"].clear(), "membership mismatch"),
            (lambda packet: packet["accepted_targets"][0].update(page_ref="other-page"), "source page"),
            (lambda packet: packet["pages"].pop(), "page registry mismatch"),
            (lambda packet: packet["pages"][0].update(item_inventory_complete=True), "incomplete inventory"),
        ]
        for mutation, message in cases:
            with self.subTest(message=message):
                inputs = self.automatic_inputs()
                mutation(inputs["discovery"])
                with self.assertRaisesRegex(ValueError, message):
                    build_automatic_manifest(inputs)

    def test_automatic_unresolved_proposals_are_separate_and_quantity_ineligible(self):
        inputs = self.automatic_inputs()
        proposal = {"id": "proposal.unlabelled-equipment", "page_ref": inputs["catalog"]["pages"][0]["page_ref"],
                    "description": "Unlabelled symbol", "reason": "equipment port evidence missing",
                    "target_refs": [], "evidence_refs": ["primitive.17"],
                    "source_geometry": {"bbox_display": [100, 100, 200, 200]}, "quantity_eligible": False}
        inputs["discovery"]["unresolved_proposals"] = [proposal]
        manifest = build_automatic_manifest(inputs)
        self.assertEqual(manifest["unresolved_proposals"], [proposal])
        self.assertNotIn(proposal["id"], [row["id"] for row in manifest["item_rows"]])
        proposal["quantity_eligible"] = True
        with self.assertRaisesRegex(ValueError, "quantity authority"):
            build_automatic_manifest(inputs)

    def test_automatic_pdf_paginates_all_proposals_with_source_links(self):
        inputs = self.automatic_inputs()
        inputs["discovery"]["unresolved_proposals"] = [
            {"id": f"proposal.{index:03}", "page_ref": inputs["catalog"]["pages"][0]["page_ref"],
             "description": "Unlabelled symbol candidate", "reason": "ambiguous port contact",
             "target_refs": [], "evidence_refs": [f"primitive.{index}"],
             "source_geometry": {"bbox_display": [100 + index * 30, 500, 120 + index * 30, 520]},
             "quantity_eligible": False} for index in range(24)]
        manifest = build_automatic_manifest(inputs)
        manifest["pages"] = manifest["pages"][:3]
        manifest["original_annotation_observations"] = []
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.pdf", Path(directory) / "automatic.pdf"
            with fitz.open() as pdf:
                for index in range(3):
                    page = pdf.new_page(width=3456, height=2592)
                    page.insert_text((50, 50), f"Original sheet {index + 1}", fontsize=20)
                pdf.save(source)
            manifest["document"].update(source_pdf_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), page_count=3)
            result = render(source, output, manifest)
            with fitz.open(output) as pdf:
                self.assertGreater(len(pdf), 4)
                self.assertEqual(json.loads(pdf.embfile_get("audit-manifest.json")), result)
                for index in range(3):
                    text = pdf[index].get_text().replace("\u00a0", " ")
                    self.assertIn(f"Original sheet {index + 1}", text)
                    self.assertIn("AUTOMATIC TARGET DISCOVERY", text)
                    self.assertNotIn("ASSISTED REPLAY", text)
                    self.assertNotIn("Geometry uses reviewed", text)
                for proposal in result["unresolved_proposals"]:
                    index = proposal["audit_schedule_page_number"] - 1
                    self.assertGreaterEqual(index, 3)
                    self.assertIn(proposal["id"], pdf[index].get_text())
                    self.assertTrue(any(link["page"] == 0 for link in pdf[index].get_links()))
                    self.assertTrue(any(link["page"] == index for link in pdf[0].get_links()))
                for row in result["item_rows"]:
                    self.assertIn(row["id"], pdf[row["source"]["pdf_page_number"] - 1].get_text())

    def test_twenty_occurrences_continue_without_losing_exact_source_targets(self):
        manifest = build_automatic_manifest(self.automatic_inputs())
        template = manifest["item_rows"][0]
        manifest["pages"] = manifest["pages"][:1]
        record = manifest["pages"][0]
        manifest["item_rows"] = []
        for index in range(20):
            row = deepcopy(template)
            row["id"] = f"mep_item_occurrence.renderer_stress_{index:02}"
            row["source"].update(page_ref=record["page_ref"], pdf_page_number=1)
            row["occurrence"]["id"] = row["id"]
            row["occurrence"]["source"] = deepcopy(row["source"])
            row["duplicate_projection_links"] = []
            row["centreline_points_display"] = [[500, 100 + 100 * index], [700, 100 + 100 * index]]
            manifest["item_rows"].append(row)
        record["item_occurrence_refs"] = [row["id"] for row in manifest["item_rows"]]
        manifest["original_annotation_observations"] = []
        manifest["unresolved_proposals"] = [
            {"id": "proposal.overlap-tag", "page_ref": record["page_ref"], "description": "Symbol candidate",
             "reason": "identity unresolved", "evidence_refs": [],
             "source_geometry": {"bbox_display": [450, 20, 950, 2350]}, "quantity_eligible": False},
            *[{"id": f"proposal.same-frame-{index}", "page_ref": record["page_ref"], "description": "Symbol candidate",
               "reason": "identity unresolved", "evidence_refs": [],
               "source_geometry": {"bbox_display": [100, 300, 140, 340]}, "quantity_eligible": False}
              for index in range(2)],
        ]
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.pdf", Path(directory) / "many.pdf"
            with fitz.open() as pdf:
                page = pdf.new_page(width=3456, height=2592)
                page.insert_text((50, 50), "Original source sheet", fontsize=20)
                # Every default opaque tag would cover an adjacent native
                # callout or grid label unless original text is an obstacle.
                for index in range(20):
                    page.insert_text((540, 80 + 100 * index), "E" if index % 2 else f"ADJACENT CALLOUT {index}", fontsize=16)
                native_text_obstacles = [fitz.Rect(span["bbox"]) + (-3, -3, 3, 3)
                    for block in page.get_text("dict")["blocks"] for line in block.get("lines", []) for span in line["spans"]]
                pdf.save(source)
            manifest["document"].update(source_pdf_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), page_count=1)
            result = render(source, output, manifest)
            self.assertTrue(result["accepted_schedule_continuation_page_numbers"])
            with fitz.open(output) as pdf:
                text = pdf[0].get_text().replace("\u00a0", " ")
                self.assertIn("Original source sheet", text)
                self.assertIn("PARTIAL CERTIFIED OCCURRENCES", text)
                self.assertIn("BOUNDED SUBSET ONLY", text)
                self.assertIn("not a fully identified physical item", " ".join(text.split()))
                source_links = pdf[0].get_links()
                for row in result["item_rows"]:
                    self.assertIn(row["id"], text)
                    self.assertIn(row["source_leader_anchor_display"], row["centreline_points_display"])
                    schedule_index = row["audit_schedule_page_number"] - 1
                    self.assertIn(row["id"], pdf[schedule_index].get_text())
                    self.assertTrue(any(link["page"] == schedule_index and
                                        max(abs(a - b) for a, b in zip(link["from"], row["source_mark_rect_display"])) < .001
                                        for link in source_links))
                    self.assertTrue(any(link["page"] == 0 for link in pdf[schedule_index].get_links()))
                marks = [fitz.Rect(row["source_mark_rect_display"]) for row in result["item_rows"]]
                self.assertFalse(any(first.intersects(second) for index, first in enumerate(marks) for second in marks[index + 1:]))
                self.assertFalse(any(mark.intersects(obstacle) for mark in marks for obstacle in native_text_obstacles))
                self.assertTrue(all(not mark.contains(fitz.Point(row["source_leader_anchor_display"]))
                                    for mark, row in zip(marks, result["item_rows"])))
                self.assertEqual(result["pages"][0]["source_tag_obstacle_basis"], "original_native_text_spans_only_raster_text_unchecked")
                proposals = {row["id"]: row for row in result["unresolved_proposals"]}
                self.assertEqual(proposals["proposal.overlap-tag"]["source_mark_state"], "appendix_only_overlap_preserves_source_legibility")
                self.assertEqual(proposals["proposal.same-frame-0"]["source_mark_state"], "drawn_nonoverlapping_candidate_frame")
                self.assertEqual(proposals["proposal.same-frame-1"]["source_mark_state"], "appendix_only_overlap_preserves_source_legibility")
            human_output = Path(directory) / "human-many.pdf"
            human = render_items(source, human_output, manifest)
            self.assertEqual(len(human["item_rows"]), 20)
            self.assertTrue(any(row["audit_schedule_page_number"] > 2 for row in human["item_rows"]))
            with fitz.open(human_output) as pdf:
                for row in human["item_rows"]:
                    self.assertIn(row["presentation"]["label"], pdf[1].get_text())
                    self.assertIn(row["presentation"]["label"], pdf[row["audit_schedule_page_number"] - 1].get_text())
                    self.assertTrue(any(link["page"] == row["audit_detail_page_number"] - 1 for link in pdf[1].get_links()))
                self.assertEqual(json.loads(pdf.embfile_get("audit-manifest.json")), human)

    def test_frozen_occurrences_keep_app_ids_evidence_and_null_amounts(self):
        before = deepcopy(self.inputs)
        manifest = build_manifest(self.inputs)
        self.assertEqual(self.inputs, before)
        catalog = self.inputs["catalog"]
        self.assertEqual(len(manifest["pages"]), 18)
        self.assertEqual(len(manifest["item_rows"]), 4)
        self.assertFalse(manifest["coverage_complete"])
        self.assertEqual(manifest["reference_register"]["state"], "unresolved_discovery_not_run")
        original = {row["id"]: row for row in catalog["item_occurrences"]}
        for row in manifest["item_rows"]:
            self.assertEqual(row["occurrence"], original[row["id"]])
            self.assertEqual(row["evidence_refs"], original[row["id"]]["evidence_refs"])
            self.assertTrue(row["shared_takeoff_occurrence_id"].startswith("takeoff_occurrence."))
            self.assertEqual(len(row["duplicate_projection_links"]), 1)
            self.assertFalse(row["duplicate_projection_links"][0]["additive"])
            self.assertEqual(row["duplicate_projection_links"][0]["evidence_ref"], row["bounded_local_3d_segment_ref"])
            self.assertIsNone(row["occurrence"]["counts"]["physical_instance_count"])
            for channel in row["takeoff_line"]["value_channels"].values():
                self.assertIsNone(channel["value"])
                self.assertIsNone(channel["unit"])

    def test_mismatched_frozen_inputs_abstain_before_render(self):
        self.inputs["geometry"]["bounded_local_3d_segments"][0]["centreline_points_xyz_m"][0][2] += 1
        with self.assertRaisesRegex(ValueError, "frozen catalog hash mismatch"):
            build_manifest(self.inputs)

    def test_cross_page_target_cannot_supply_source_geometry(self):
        self.inputs["composites"]["accepted_composites"][0]["page_ref"] = "wrong-page"
        self.inputs["geometry"]["m3_5_contract_ref"]["payload_sha256"] = canonical_sha256(self.inputs["composites"])
        self.inputs["catalog"]["m5a_contract_ref"]["payload_sha256"] = canonical_sha256(self.inputs["geometry"])
        with self.assertRaisesRegex(ValueError, "no accepted target on its source page"):
            build_manifest(self.inputs)

    def test_pdf_retains_source_annotations_navigation_and_unresolved_schedule(self):
        manifest = build_manifest(self.inputs)
        manifest["pages"] = manifest["pages"][:3]
        manifest["reviewed_unresolved_findings"] = []
        manifest["original_annotation_observations"] = []
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pdf"
            output = Path(directory) / "audit.pdf"
            with fitz.open() as pdf:
                for index in range(3):
                    page = pdf.new_page(width=3456, height=2592)
                    page.insert_text((50, 50), f"Original sheet {index + 1}", fontsize=20)
                    note = page.add_text_annot((100, 100), "Original engineer claim")
                    note.update()
                    cloud = page.add_polygon_annot([(200, 200), (300, 200), (300, 300), (200, 300)])
                    cloud.update()
                    pdf.xref_set_key(cloud.xref, "IRT", f"{note.xref} 0 R")
                    pdf.xref_set_key(cloud.xref, "RT", "/Group")
                pdf.save(source)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest["document"]["source_pdf_sha256"] = digest
            manifest["document"]["page_count"] = 3
            result = render(source, output, manifest)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)
            with fitz.open(source) as original, fitz.open(output) as pdf:
                self.assertEqual(len(pdf), 3)
                self.assertEqual(json.loads(pdf.embfile_get("audit-manifest.json")), result)
                for index, page in enumerate(pdf):
                    self.assertEqual(page.rect.height, original[index].rect.height)
                    self.assertGreater(page.rect.width, original[index].rect.width)
                    self.assertEqual([annot.info["content"] for annot in page.annots()], ["Original engineer claim", ""])
                    self.assertIn(f"Original sheet {index + 1}", page.get_text())
                    self.assertIn("PARTIAL MARKED M&P AUDIT", page.get_text().replace("\u00a0", " "))
                    self.assertIn("unresolved", page.get_text())
                interior = original[0].rect + (0, 0, -5, 0)
                self.assertEqual(original[0].get_pixmap(matrix=fitz.Matrix(.2, .2), clip=interior).samples,
                                 pdf[0].get_pixmap(matrix=fitz.Matrix(.2, .2), clip=interior).samples)
                self.assertGreaterEqual(len(pdf[1].get_links()), 6)
                for row in result["item_rows"]:
                    page = pdf[row["source"]["pdf_page_number"] - 1]
                    self.assertIn(row["id"], page.get_text())
                    self.assertIn(row["shared_takeoff_occurrence_id"], page.get_text())
                self.assertIn("nonadditive", pdf[1].get_text())
            manifest["document"]["source_pdf_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "source PDF hash"):
                render(source, Path(directory) / "wrong.pdf", manifest)
