from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import fitz

from src.drawing_engine.exports.detail_assembly_export import export_detail
from src.drawing_engine.disciplines.detail.detail_declarations import extract_detail_declarations
from src.drawing_engine.cli import inspect_result
from src.drawing_engine.disciplines.detail.native_detail_assembly import NativeDetail, extract_assembly


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/mep-br5-duct/GCC-SEC-DDD-12100-81-1303-KJ2-DWG-41702_00_3_publication.pdf"


class NativeDetailDimensionTest(unittest.TestCase):
    def drawing(self, *, missing_tick=False, offset=False, conflict=False,
                outside=False, missing_extension=False, witness_gap=2):
        doc = fitz.open()
        self.addCleanup(doc.close)
        page = doc.new_page(width=400, height=300)
        end_x = 104 if outside else 160
        page.draw_rect(fitz.Rect(100, 100, end_x, 180), width=.3)
        for x in (100, end_x):
            page.draw_line((x, 100 - witness_gap), (x, 66), width=.3)
            if not (missing_tick and x == end_x):
                page.draw_polyline([(x-3, 72.5), (x-2.5, 73), (x+3, 67.5), (x+2.5, 67)],
                                   color=(0, 0, 0), fill=(0, 0, 0), closePath=True)
        page.draw_line((94, 70), (end_x + 6, 70), width=.3)
        if outside:
            if not missing_extension:
                page.draw_line((73, 70), (100, 70), width=.3)
            page.insert_text((75, 68), "15", fontsize=8)
        elif offset:
            page.draw_polyline([(130, 70), (170, 50), (195, 50)], width=.3)
            page.insert_text((172, 48), "200", fontsize=8)
        else:
            page.insert_text((120, 68), "200", fontsize=8)
        if conflict:
            page.insert_text((140, 68), "300", fontsize=8)
        return page

    def test_short_outside_label_needs_extension_both_ticks_and_profile_anchors(self):
        native = NativeDetail(self.drawing(outside=True))
        self.assertEqual({d['value'] for d in native.dimensions(native.rectangles[0], 0)}, {15})
        for kwargs in ({'missing_extension': True}, {'missing_tick': True}, {'witness_gap': 8}):
            native = NativeDetail(self.drawing(outside=True, **kwargs))
            self.assertEqual(native.dimensions(native.rectangles[0], 0), [])

    def test_witness_crossing_profile_edge_closes_but_disconnected_witness_does_not(self):
        for gap, missing_tick, expected in ((-8, False, {200}), (8, False, set()), (-8, True, set())):
            native = NativeDetail(self.drawing(witness_gap=gap, missing_tick=missing_tick))
            dimensions = native.dimensions(native.rectangles[0], 0)
            self.assertEqual({d['value'] for d in dimensions}, expected)
            for dimension in dimensions:
                self.assertTrue(all(ref.startswith(('word[', 'drawing[')) for ref in dimension['evidence_refs']))

    def test_branch_arrow_uses_unique_authored_endpoint_contact_not_crossing(self):
        for endpoint, expected in ((140, 2), (142, 1)):
            doc = fitz.open()
            self.addCleanup(doc.close)
            page = doc.new_page(width=300, height=200)
            page.draw_polyline([(126, 100), (140, 100), (140, 60), (165, 60)], width=.3)
            page.draw_line((126, 80), (endpoint, 80), width=.3)
            for y in (80, 100):
                page.draw_polyline([(120, y), (126, y-1), (126, y+1)],
                                   fill=(0, 0, 0), color=(0, 0, 0), closePath=True)
            page.insert_text((143, 58), '9', fontsize=8)
            leaders = [l for l in NativeDetail(page).leaders if l['text'] == '9' and l['kind'] == 'arrow']
            self.assertEqual(len(leaders), expected)
            self.assertTrue(all(l['evidence_refs'] for l in leaders))

    def test_direct_and_offset_labels_require_two_native_terminals(self):
        for offset in (False, True):
            native = NativeDetail(self.drawing(offset=offset))
            dims = native.dimensions(native.rectangles[0], 0)
            self.assertEqual({d["value"] for d in dims}, {200})
            self.assertTrue(all(any("word[" in ref for ref in d["evidence_refs"]) for d in dims))
        native = NativeDetail(self.drawing(missing_tick=True))
        self.assertEqual(native.dimensions(native.rectangles[0], 0), [])

    def test_competing_numeric_labels_are_preserved(self):
        native = NativeDetail(self.drawing(conflict=True))
        self.assertEqual({d["value"] for d in native.dimensions(native.rectangles[0], 0)}, {200, 300})

    def test_missing_native_title_abstains(self):
        page = self.drawing()
        result, _ = extract_assembly(page, "unseen")
        self.assertEqual(result["state"], "unknown")
        self.assertEqual(result["child_parts"], [])
        self.assertIsNone(result["approved"])
        declarations = extract_detail_declarations(page, "unseen")
        self.assertEqual(declarations["records"], [])
        self.assertTrue(declarations["unresolved"])


class NativeDetailLiveTest(unittest.TestCase):
    def setUp(self):
        self.doc = fitz.open(SOURCE)
        self.addCleanup(self.doc.close)
        self.page = self.doc[0]

    def extract(self):
        return extract_assembly(self.page, "EP14")[0]

    def test_ep13_overrunning_witness_recovers_face_height_without_quantity_promotion(self):
        result, _ = extract_assembly(self.page, 'EP13')
        plate, = result['child_parts']
        self.assertEqual(plate['mark'], '43')
        self.assertEqual({k: {d['value'] for d in v} for k, v in plate['dimensions'].items()},
                         {'width': {400}, 'height': {930}, 'thickness': {25}, 'section_height': set()})
        self.assertIsNone(plate['solid'])
        self.assertIsNone(plate['calculated']['volume_m3'])
        self.assertIsNone(plate['physical']['job_count'])
        self.assertIsNone(plate['approved'])
        self.assertEqual(plate['internal_round_symbols']['state'], 'unknown')

    def test_ep13_exports_outer_face_even_with_unresolved_section_height_and_holes(self):
        from copy import deepcopy
        from src.drawing_engine.exports.detail_dxf_export import export_plate_dxfs
        result, _ = extract_assembly(self.page, 'EP13')
        before = deepcopy(result)
        with tempfile.TemporaryDirectory() as temporary:
            record = export_plate_dxfs(result, temporary, units='mm')['parts'][0]
            self.assertTrue((Path(temporary)/record['path']).is_file())
            self.assertEqual(record['outline_xy'], [(0,0),(400,0),(400,930),(0,930)])
            self.assertEqual(record['unit_basis'], 'assumed_mm')
            self.assertFalse(record['internal_geometry_exported'])
            self.assertEqual(record['source_contour_ref'], result['child_parts'][0]['views'][0]['drawing_ref'])
        self.assertEqual(result, before)

    def test_ep16_plain_section_ring_and_branch_leaders_preserve_authority(self):
        result, native = extract_assembly(self.page, 'EP16')
        plate, bar = result['child_parts']
        self.assertEqual((plate['mark'], bar['mark']), ('45', '46'))
        self.assertEqual(plate['annotations'][1]['kind'], 'plain')
        self.assertEqual({key: {d['value'] for d in values} for key, values in plate['dimensions'].items()},
                         {'width': {200}, 'height': {200}, 'section_height': {200}, 'thickness': {15}})
        self.assertTrue(plate['solid']['validation']['watertight'])
        self.assertAlmostEqual(plate['conditional']['volume_m3'], .0006)
        self.assertIsNone(plate['calculated']['volume_m3'])
        self.assertEqual(bar['projected'], {'observed_end_count': 4, 'observed_shaft_count': 2})
        self.assertTrue(any(a.get('shared_label_stem') for a in bar['annotations']))
        self.assertEqual(len(plate['annotation_rings']['records']), 1)
        self.assertEqual(plate['annotation_rings']['records'][0]['qualifier_observations'][0]['text'], '(TYP)')
        self.assertTrue(plate['native_stroke_observations']['zero_length_refs'])
        self.assertTrue(all(v is None for v in bar['physical'].values()))
        self.assertIsNone(bar['calculated']['mass_kg'])

    def test_ep16_missing_section_or_ring_mark_cannot_close_plate(self):
        baseline, _ = extract_assembly(self.page, 'EP16')
        plate = baseline['child_parts'][0]
        for annotation in (plate['annotations'][1], plate['annotation_rings']['records'][0]):
            ref = annotation['evidence_refs'][0]
            with self.mutate_words({int(ref[5:-1]): '999'}):
                result, _ = extract_assembly(self.page, 'EP16')
            self.assertTrue(all(p.get('solid') is None for p in result['child_parts']))

    def test_ep16_tiny_separate_path_hole_is_not_discarded_as_snapped_microstrokes(self):
        corners = [(685, 901), (685.3, 901), (685.15, 901.3)]
        for a, b in zip(corners, corners[1:] + corners[:1]):
            self.page.draw_line(a, b, width=.3)
        result, _ = extract_assembly(self.page, 'EP16')
        self.assertIsNone(result['child_parts'][0]['solid'])

    def test_ep16_competing_concentric_profile_blocks_ring_role(self):
        result, native = extract_assembly(self.page, 'EP16')
        ring = result['child_parts'][0]['annotation_rings']['records'][0]['ring_target']['bbox_display']
        self.page.draw_circle(((ring[0]+ring[2])/2, (ring[1]+ring[3])/2), 1.5, width=.3)
        changed, _ = extract_assembly(self.page, 'EP16')
        self.assertIsNone(changed['child_parts'][0]['solid'])

    def test_native_ep14_parts_dimensions_and_quantity_boundaries(self):
        result = self.extract()
        plate, bar = result["child_parts"]
        self.assertEqual((plate["mark"], bar["mark"]), ("35", "34"))
        self.assertEqual({key: {d["value"] for d in dims} for key, dims in plate["dimensions"].items()},
                         {"width": {210}, "height": {240}, "thickness": {15}, "section_height": {240}})
        self.assertTrue(plate["solid"]["validation"]["watertight"])
        self.assertAlmostEqual(plate["conditional"]["volume_m3"], .000756)
        self.assertIsNone(plate["calculated"]["volume_m3"])
        self.assertEqual(plate["internal_round_symbols"]["role"], "bar_end_projections")
        self.assertEqual(bar["projected"], {"observed_end_count": 6, "observed_shaft_count": 3})
        self.assertEqual({d["value"] for d in bar["length_dimensions"]}, {250})
        self.assertTrue(all(value is None for value in bar["physical"].values()))
        self.assertIsNone(bar["calculated"]["mass_kg"])

    def mutate_words(self, replacements):
        original = fitz.Page.get_text
        def changed(page, option="text", *args, **kwargs):
            result = original(page, option, *args, **kwargs)
            if option == "words":
                result = [tuple(list(w[:4]) + [replacements.get(i, w[4])] + list(w[5:])) for i, w in enumerate(result)]
            return result
        return patch.object(fitz.Page, "get_text", changed)

    def test_specification_changes_cannot_supply_geometry_diameter_or_count(self):
        baseline = self.extract()
        before = extract_detail_declarations(self.page, "EP14")
        refs = [ref for row in before["records"] for cell in row["cells"].values() for ref in cell["evidence_refs"]]
        words = self.page.get_text("words")
        replacements = {int(ref[5:-1]): words[int(ref[5:-1])][4].replace("250", "999").replace("16-А500С", "99-А500С")
                        for ref in refs}
        for row in before["records"]:
            for ref in row["cells"]["count"]["evidence_refs"]:
                replacements[int(ref[5:-1])] = "999"
        with self.mutate_words(replacements):
            self.assertEqual(self.extract(), baseline)
            after = extract_detail_declarations(self.page, "EP14")
        self.assertNotEqual(before, after)
        self.assertEqual(after["records"][0]["declared"]["bar_diameter_mm"], 99)

    def test_missing_thickness_or_cross_view_mark_blocks_plate_solid(self):
        baseline = self.extract()
        plate = baseline["child_parts"][0]
        refs = [plate["dimensions"]["thickness"][0]["evidence_refs"][0], plate["annotations"][1]["evidence_refs"][0]]
        for ref in refs:
            with self.mutate_words({int(ref[5:-1]): "?"}):
                result = self.extract()
                self.assertTrue(all(p.get("solid") is None for p in result["child_parts"]))

    def test_declared_mass_requires_native_kg_header(self):
        before = extract_detail_declarations(self.page, "EP14")
        refs = before["records"][0]["column_bindings"]["unit_mass"]["evidence_refs"]
        words = self.page.get_text("words")
        replacements = {int(ref[5:-1]): "lb" for ref in refs
                        if words[int(ref[5:-1])][4].rstrip(".,/").lower() in {"kg", "кг"}}
        self.assertTrue(replacements)
        with self.mutate_words(replacements):
            after = extract_detail_declarations(self.page, "EP14")
        self.assertEqual(len(after["records"]), 2)
        self.assertTrue(all(r["declared"]["unit_mass_kg"] is None for r in after["records"]))

    def test_unexplained_circle_or_polygon_blocks_net_volume(self):
        for polygon in (False, True):
            with fitz.open(SOURCE) as doc:
                page = doc[0]
                if polygon:
                    page.draw_polyline([(1540, 272), (1546, 275), (1541, 279)], closePath=True, width=.3)
                else:
                    page.draw_circle((1543, 276), 2.25, width=.3)
                result, _ = extract_assembly(page, "EP14")
                self.assertIsNone(result["child_parts"][0]["solid"])

    def test_hole_split_across_native_paint_paths_blocks_net_volume(self):
        corners = [(1540, 272), (1546, 272), (1546, 278), (1540, 278)]
        for a, b in zip(corners, corners[1:] + corners[:1]):
            self.page.draw_line(a, b, width=.3)
        result = self.extract()
        self.assertIsNone(result["child_parts"][0]["solid"])

    def test_missing_bar_mark_does_not_turn_ends_into_holes_or_plate_volume(self):
        baseline = self.extract()
        ref = baseline["child_parts"][1]["annotations"][0]["evidence_refs"][0]
        with self.mutate_words({int(ref[5:-1]): "?"}):
            result = self.extract()
        self.assertIsNone(result["child_parts"][0]["solid"])
        self.assertEqual(result["child_parts"][0]["internal_round_symbols"]["role"], "unresolved")

    def test_translated_native_page_preserves_semantics(self):
        doc = fitz.open()
        self.addCleanup(doc.close)
        page = doc.new_page(width=self.page.rect.width + 100, height=self.page.rect.height + 80)
        page.show_pdf_page(fitz.Rect(50, 40, self.page.rect.width + 50, self.page.rect.height + 40), self.doc, 0)
        for mark, positions, volume in (('EP14', ['35', '34'], .000756), ('EP16', ['45', '46'], .0006)):
            result, _ = extract_assembly(page, mark)
            self.assertEqual([p["mark"] for p in result["child_parts"]], positions)
            self.assertAlmostEqual(result["child_parts"][0]["conditional"]["volume_m3"], volume)

    def test_explicit_global_unit_evidence_closes_volume_without_approving_it(self):
        self.page.insert_text((1500, 500), "All dimensions are in mm.", fontsize=10)
        result = self.extract()
        plate = result["child_parts"][0]
        self.assertEqual(result["linear_unit_evidence"]["state"], "direct")
        self.assertAlmostEqual(plate["calculated"]["volume_m3"], .000756)
        self.assertEqual(plate["conditional"], {})
        self.assertIsNone(plate["approved"])
        from src.drawing_engine.disciplines.detail.detail_declarations import compare_detail
        declarations = extract_detail_declarations(self.page, "EP14")
        comparison = compare_detail(result, declarations)
        self.assertEqual(comparison["comparisons"][0]["severity"], "pass")
        declarations["records"][1]["declared"]["rectangular_stock_volume_m3"] *= 2
        changed = compare_detail(result, declarations)
        self.assertEqual(changed["comparisons"][0]["severity"], "review")
        self.assertLess(changed["comparisons"][0]["delta"], 0)

    def test_export_freezes_calculation_and_cli_inspects_all_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = export_detail(SOURCE, Path(temporary) / "job", assembly_mark="EP14")
            assembly = inspect_result(manifest, "assembly")
            comparison = inspect_result(manifest, "comparison")
            self.assertEqual(comparison["frozen_calculation"]["sha256"], inspect_result(manifest)["frozen_assembly_sha256"])
            self.assertEqual(len(assembly["child_parts"]), 2)
            self.assertEqual(len(inspect_result(manifest, "declarations")["records"]), 2)
            self.assertTrue(all(c["delta"] is None and c["approved"] is None for c in comparison["comparisons"]))
            self.assertEqual(assembly['child_parts'][0]['internal_round_symbols']['role'], 'bar_end_projections')
            self.assertEqual([p.name for p in manifest.parent.rglob('*.json')], ['result.json'])
            self.assertFalse((manifest.parent / 'audit.html').exists())
            dxf = inspect_result(manifest, "dxf_report")
            self.assertEqual(dxf["frozen_assembly_sha256"], inspect_result(manifest)["frozen_assembly_sha256"])
            self.assertEqual(dxf["parts"][0]["outline_xy"], [[0, 0], [210, 0], [210, 240], [0, 240]])
            self.assertEqual(dxf["parts"][0]["unit_basis"], "unspecified_drawing_units")
            self.assertIsNone(dxf["parts"][1]["path"])
            with fitz.open(manifest.parent / "audit.pdf") as audit:
                self.assertIn("EP14", audit[0].get_text())
                self.assertTrue(list(audit[1].annots()))
                audit_text = " ".join("".join(p.get_text() for p in audit).split())
                self.assertIn("Conditional volume: 0.000756", audit_text)
                self.assertIn('INTERPRETATION', audit[1].get_text())
                titles = [a.info['title'] for a in audit[1].annots()]
                self.assertTrue(any('complete native dimension chain' in t for t in titles))
                self.assertTrue(any('leader' in t for t in titles))
                self.assertIn('UNRESOLVED', titles)
            from src.drawing_engine.exports.estimation_project_export import FrozenProject
            self.assertEqual(FrozenProject(manifest.parent, inspect_result(manifest)).load('assembly'), assembly)
