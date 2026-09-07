from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import fitz

from src.drawing_engine.disciplines.detail.native_cage_assembly import extract_cage, ladder
from src.drawing_engine.disciplines.detail.native_detail_assembly import NativeDetail
from src.drawing_engine.disciplines.detail.cage_document_links import tables, document_anchor, reference_rows, link_references


class CageCertificateTest(unittest.TestCase):
    def shape(self):
        result = []
        def line(a, b):
            result.append({"id": f"s{len(result)}", "kind": "line", "start": a, "end": b})
        for x in (10, 20, 30):
            points = [(x,0),(x+1,0),(x+1,15),(x,15)]
            for a, b in zip(points, points[1:]+points[:1]):
                line(a,b)
        for y in (3,4,11,12):
            for left, right in ((5,10),(11,20),(21,30),(31,36)):
                line((left,y),(right,y))
        for x in (5,36):
            for a,b in ((3,4),(11,12)):
                line((x,a),(x,b))
        return result

    def test_complete_source_coverage_and_occlusion_boundaries(self):
        rows = self.shape()
        value = ladder(rows)
        self.assertEqual([len(value[k]) for k in ("transverse","rails")], [3,2])
        refs = {ref for kind in ("transverse","rails") for r in value[kind] for ref in r["evidence_refs"]}
        self.assertEqual(refs, {s["id"] for s in rows})
        for index in (0, 15, len(rows)-1):
            with self.subTest(missing=index):
                self.assertIsNone(ladder(rows[:index]+rows[index+1:]))
        changed = deepcopy(rows)
        changed[15]["end"] = (35.5,3)
        self.assertIsNone(ladder(changed))
        self.assertIsNone(ladder(rows+[dict(rows[0], id="unexplained")]))

    def test_reference_identity_applicability_and_quantities_stay_separate(self):
        source = {"document_code":"РЧ 2030-777.001", "sheet":9,"revision":["2","Зам.","2030-777.2"]}
        target = dict(source, document_code="РЧ 2030-777.001.04 СБ", sheet=14)
        row = {"part_mark":"7","destination_sheet":14,"declared":{"count":999},"evidence_refs":["word[1]"],"header_evidence_refs":["word[2]"]}
        assembly = {"state":"derived", "child_parts":[{"mark":"7","id":"part:7","projected_count":32}]}
        link = link_references(source,target,[row],assembly)[0]
        self.assertEqual(link["detail_applicability_state"],"derived")
        self.assertFalse(link["quantity_eligible"])
        self.assertIsNone(link["count_comparison"]["delta"])
        for mutation in (dict(target,sheet=15),dict(target,revision=["1","Зам.","2030-777.1"]),dict(target,document_code="РЧ 2030-888.04 СБ"),None):
            self.assertEqual(link_references(source,mutation,[row],assembly)[0]["reference_identity_state"],"unknown")
        self.assertEqual(link_references(source,target,[row,row],assembly)[0]["detail_applicability_state"],"unknown")
        self.assertEqual(link_references(source,target,[dict(row,part_mark="8")],assembly)[0]["detail_applicability_state"],"unknown")


class CageLiveTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]/"Пример_чертежей_для_декомпозиции"
        self.doc = fitz.open(self.root/"РЧ 2024-040 часть 1-15.pdf")
        self.addCleanup(self.doc.close)

    def test_children_repetition_and_actual_cross_file_sheet(self):
        assembly, native = extract_cage(self.doc[0],"К1")
        self.assertEqual([(p["mark"],p["projected_count"]) for p in assembly["child_parts"]],[("7",32),("8",2)])
        self.assertEqual([p["leader_target_count"] for p in assembly["parent_occurrences"]],[6])
        for p in assembly["child_parts"]:
            self.assertIsNone(p["physical_count"])
            self.assertIsNone(p["cutting_length_mm"])
        with fitz.open(self.root/"РЧ 2024-040 часть 1-10.pdf") as document:
            reference = NativeDetail(document[0])
            observed = tables(reference)
            source = document_anchor(reference,observed)
            destination = document_anchor(native,tables(native))
            self.assertEqual((source["sheet"],destination["sheet"]),(9,14))
            rows = reference_rows(reference,observed,"К1")
            links = link_references(source,destination,rows,assembly)
            self.assertEqual([(r["part_mark"],r["detail_applicability_state"]) for r in links],[("7","derived"),("8","derived")])

    def test_translation_and_missing_or_competing_caption(self):
        with fitz.open() as doc:
            page=doc.new_page(width=self.doc[0].rect.width+90,height=self.doc[0].rect.height+60)
            page.show_pdf_page(self.doc[0].rect+(40,30,40,30),self.doc,0)
            assembly,_=extract_cage(page,"К1")
            self.assertEqual([(p["mark"],p["projected_count"]) for p in assembly["child_parts"]],[("7",32),("8",2)])
        self.assertEqual(extract_cage(self.doc[0],"К999")[0]["state"],"unknown")
        native = NativeDetail(self.doc[0])
        caption = next(t for t in native.text if t["text"] == "Схема каркаса К 1")
        native.text.append(deepcopy(caption))
        with patch("src.drawing_engine.disciplines.detail.native_cage_assembly.NativeDetail", return_value=native):
            self.assertEqual(extract_cage(self.doc[0],"К1")[0]["state"],"unknown")
        page = self.doc[0]
        area = page.search_for("Схема каркаса К")[0]
        page.add_redact_annot(area)
        page.apply_redactions(graphics=0)
        self.assertEqual(extract_cage(page,"К1")[0]["state"],"unknown")
