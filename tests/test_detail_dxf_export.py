from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import ezdxf

from src.drawing_engine.exports.detail_dxf_export import export_plate_dxfs


class DetailDxfExportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        dims = {key: [{"value": value, "evidence_refs": [f"word:{key}"], "profile_ref": "drawing[1]",
                      "measured_interval_display": [10, 70] if key == "width" else [20, 60]}]
                for key, value in {"width": 120, "height": 80, "thickness": 8, "section_height": 80}.items()}
        self.plate = {"id": "part:7", "mark": "7", "kind": "plate", "dimensions": dims,
                      "outer_contour": {"kind": "closed_native_rectangle", "drawing_ref": "drawing[1]",
                          "points_display": [[10,20],[70,20],[70,60],[10,60]],
                          "evidence_refs": [f"drawing[1].segment[{i}]" for i in range(4)]},
                      "solid": {"validation": {"watertight": True}, "vertices_xyz_drawing_units":
                                [[x, y, z] for x in (0, 120) for y in (0, 80) for z in (0, 8)]},
                      "reprojection": {"passed": True}, "internal_round_symbols": {"role": "bar_end_projections"},
                      "calculated": {"volume_m3": None}, "approved": None}
        self.assembly = {"mark": "EP13", "linear_unit_evidence": {"state": "unknown", "unit": None, "observations": []},
                         "child_parts": [self.plate, {"id": "part:8", "mark": "8", "kind": "welded_bar_family"}]}

    def test_units_are_explicit_and_export_never_changes_calculation(self):
        before = deepcopy(self.assembly)
        native = export_plate_dxfs(self.assembly, self.output)
        assumed = export_plate_dxfs(self.assembly, self.output, units="mm")
        self.assertEqual(self.assembly, before)
        self.assertIsNone(native["parts"][0]["units"])
        record = assumed["parts"][0]
        self.assertEqual(record["path"], "dxf/EP13-part-7-outline.mm-assumed.dxf")
        self.assertEqual(record["state"], "convention_dependent")
        self.assertFalse(record["fabrication_release"])
        self.assertIsNone(record["physical_count"])
        self.assertEqual(assumed["parts"][1]["state"], "abstained")
        lines = (self.output / record["path"]).read_text().splitlines()
        pairs = [(int(c), v) for c, v in zip(lines[::2], lines[1::2])]
        start = pairs.index((0, "LWPOLYLINE"))
        entity = pairs[start: pairs.index((0, "ENDSEC"), start)]
        self.assertEqual([v for c, v in entity if c == 0], ["LWPOLYLINE"])
        self.assertIn((70, "1"), entity)
        self.assertIn((8, "0"), entity)
        xs = [float(v) for c, v in entity if c == 10]
        ys = [float(v) for c, v in entity if c == 20]
        self.assertEqual(len(set(zip(xs, ys))), 4)
        area = abs(sum(xs[i] * ys[(i + 1) % 4] - xs[(i + 1) % 4] * ys[i] for i in range(4))) / 2
        self.assertEqual(area, 9600)
        self.assertEqual((max(xs) - min(xs), max(ys) - min(ys)), (120, 80))
        self.assertNotIn((0, "CIRCLE"), pairs)

    def test_outline_does_not_require_solid_section_or_internal_geometry(self):
        for mutate in (
                lambda p: p.update(solid=None),
                lambda p: p["solid"]["validation"].update(watertight=False),
                lambda p: p["reprojection"].update(passed=False),
                lambda p: p["internal_round_symbols"].update(role="unresolved"),
                lambda p: p["dimensions"].pop("thickness"),
                lambda p: p["dimensions"].pop("section_height"),
                lambda p: p["dimensions"]["section_height"][0].update(value=79),
                lambda p: p["solid"]["vertices_xyz_drawing_units"][0].__setitem__(0, 1),
        ):
            candidate = deepcopy(self.assembly)
            mutate(candidate["child_parts"][0])
            before = deepcopy(candidate)
            result = export_plate_dxfs(candidate, self.output, units="mm")
            self.assertIsNotNone(result["parts"][0]["path"])
            self.assertFalse(result["parts"][0]["internal_geometry_exported"])
            self.assertFalse(result["parts"][0]["complete_part_profile_established"])
            self.assertEqual(candidate, before)

    def test_cad_document_roundtrip_needs_no_repairs_and_resolves_owners(self):
        for units, code in (("native", 0), ("mm", 4)):
            with self.subTest(units=units):
                record = export_plate_dxfs(self.assembly, self.output, units=units)["parts"][0]
                # Readers may silently synthesize missing tables/owners. Check
                # the serialized document too: this is what OpenDesign receives.
                lines = (self.output / record["path"]).read_text().splitlines()
                pairs = [(int(c), v.strip()) for c, v in zip(lines[::2], lines[1::2])]
                for section in ("TABLES", "BLOCKS", "OBJECTS"):
                    self.assertIn((2, section), pairs)
                start = pairs.index((0, "LWPOLYLINE"))
                entity_tags = pairs[start:pairs.index((0, "ENDSEC"), start)]
                owner_handle, = [v for c, v in entity_tags if c == 330]
                self.assertNotEqual(owner_handle, "0")
                doc = ezdxf.readfile(self.output / record["path"])
                audit = doc.audit()
                self.assertEqual(audit.errors, [])
                self.assertEqual(audit.fixes, [])
                self.assertEqual(doc.dxfversion, "AC1015")
                self.assertEqual(doc.units, code)
                entity, = doc.modelspace()
                self.assertEqual(entity.dxftype(), "LWPOLYLINE")
                self.assertTrue(entity.closed)
                self.assertEqual(entity.dxf.layer, "0")
                self.assertEqual(list(entity.get_points("xy")), [(0, 0), (120, 0), (120, 80), (0, 80)])
                self.assertEqual(entity.dxf.owner, doc.modelspace().block_record_handle)
                self.assertEqual(entity.dxf.owner, owner_handle)
                owner = doc.entitydb[entity.dxf.owner]
                self.assertEqual(owner.dxftype(), "BLOCK_RECORD")
                self.assertEqual(doc.entitydb[owner.dxf.layout].dxftype(), "LAYOUT")

    def test_missing_or_conflicting_face_geometry_dimensions_and_provenance_abstain(self):
        for mutate in (
                lambda p: p.pop("outer_contour"),
                lambda p: p["outer_contour"].update(points_display=[[10,20],[70,60],[70,20],[10,60]]),
                lambda p: p["outer_contour"].update(evidence_refs=[]),
                lambda p: p["dimensions"]["width"].append({"value":999}),
                lambda p: p["dimensions"]["height"][0].update(value=0),
                lambda p: p["dimensions"]["height"][0].update(value=float('nan')),
                lambda p: p["dimensions"]["height"][0].update(value=200),
                lambda p: p["dimensions"]["height"][0].update(profile_ref="drawing[2]"),
                lambda p: p["dimensions"]["height"][0].update(measured_interval_display=[10,50]),
                lambda p: p["dimensions"]["width"][0].update(evidence_refs=[]),
        ):
            candidate = deepcopy(self.assembly)
            mutate(candidate["child_parts"][0])
            record = export_plate_dxfs(candidate, self.output, units="mm")["parts"][0]
            self.assertIsNone(record["path"])
            self.assertTrue(record["reasons"])
        self.assertEqual(list(self.output.rglob("*.dxf")), [])

    def test_native_units_close_only_export_units_and_unsafe_marks_cannot_escape(self):
        self.assembly["linear_unit_evidence"] = {"state": "direct", "unit": "mm", "observations": [{"text": "All dimensions in mm"}]}
        self.plate["mark"] = "../../outside"
        result = export_plate_dxfs(self.assembly, self.output)
        part = result["parts"][0]
        self.assertEqual(part["unit_basis"], "native_mm")
        self.assertTrue((self.output / part["path"]).resolve().is_relative_to(self.output.resolve()))
        self.assertIsNone(self.plate["calculated"]["volume_m3"])
        self.assembly["linear_unit_evidence"].update(state="unknown", unit=None)
        self.assertIsNone(export_plate_dxfs(self.assembly, self.output, units="mm")["parts"][0]["path"])

    def test_drawing_names_are_safe_bounded_and_do_not_overwrite_colliding_parts(self):
        self.assembly["mark"] = "../ЭП13/"
        self.assembly["child_parts"] = []
        for index, mark in enumerate(("A/B", "A:B", "a-b", None, "板" * 100)):
            part = deepcopy(self.plate)
            part.update(id=f"part:{index}", mark=mark)
            self.assembly["child_parts"].append(part)
        before = deepcopy(self.assembly)
        report = export_plate_dxfs(self.assembly, self.output, units="mm")
        names = [Path(r["path"]).name for r in report["parts"]]
        self.assertEqual(names[:4], ["ЭП13-part-A-B-outline.mm-assumed.dxf",
                                    "ЭП13-part-A-B-outline-2.mm-assumed.dxf",
                                    "ЭП13-part-a-b-outline-3.mm-assumed.dxf",
                                    "ЭП13-part-unmarked-4-outline.mm-assumed.dxf"])
        self.assertEqual(len(list(self.output.rglob("*.dxf"))), 5)
        for record in report["parts"]:
            path = self.output / record["path"]
            self.assertTrue(path.resolve().is_relative_to(self.output.resolve()))
            self.assertLess(len(path.name.encode("utf-8")), 255)
        self.assertEqual(self.assembly, before)
