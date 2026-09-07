import tempfile
from pathlib import Path
import unittest

from src.drawing_engine.disciplines.mep.mep_native_descriptor_pack import NativeDescriptorPack, NativeDescriptorPackWriter
from src.drawing_engine.disciplines.mep.mep_source_primitive_denominator import (
    SourceDispositionPack, validate_source_denominator,
    write_source_disposition_pack,
)
from src.drawing_engine.disciplines.mep.mep_route_observations import build_mep_route_page_from_denominator


def _row(index, points):
    source_ref = f"drawing[{index}].item[0].segment[0]"
    xs, ys = [point[0] for point in points], [point[1] for point in points]
    box = [min(xs), min(ys), max(xs), max(ys)]
    return {
        "id": f"mep_native_target_primitive.{index:020x}",
        "page_ref": "page.5",
        "source_primitive_ref": source_ref,
        "source_native_segment": {
            "id": source_ref, "drawing_ref": f"drawing[{index}]",
            "primitive_ref": f"drawing[{index}].item[0]",
            "item_index": 0, "part_index": 0, "kind": "line",
            "start_display": points[0], "end_display": points[-1],
            "control_points_display": [], "sample_points_display": [],
            "axis": "horizontal", "length_points": abs(points[-1][0] - points[0][0]),
            "bbox_display": box,
            "style": {"width": .5, "stroke": [0, 0, 0], "fill": None, "dash": "[] 0"},
        },
        "source_drawing_close_path": False,
        "points_display": points, "bbox_display": box,
        "search_bbox_display": box, "search_refs": [],
    }


class SourcePrimitiveDenominatorTests(unittest.TestCase):
    def test_every_descriptor_gets_one_primary_disposition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            descriptor_path = root / "native.pack"
            writer = NativeDescriptorPackWriter(
                descriptor_path, page_ref="page.5",
                pdf_to_display_matrix=[1, 0, 0, 1, 0, 0],
                minimum_member_length=1000)
            rows = [
                _row(1, [[10, 10], [20, 10]]),
                _row(2, [[10, 20], [20, 20]]),
                _row(3, [[10, 30], [20, 30]]),
                _row(4, [[10, 40], [20, 40]]),
                _row(5, [[10, 50], [20, 50]]),
                _row(6, [[82, 20], [90, 20]]),
            ]
            for row in rows:
                writer.append(row)
            descriptor_manifest = writer.finish()
            descriptors = NativeDescriptorPack(descriptor_path, descriptor_manifest)
            roles = {
                "route_evidence": {
                    rows[0]["source_primitive_ref"], rows[1]["source_primitive_ref"],
                    rows[2]["source_primitive_ref"], rows[5]["source_primitive_ref"]},
                "equipment_fitting_evidence": {rows[1]["source_primitive_ref"]},
                "annotation_dimension": {rows[2]["source_primitive_ref"]},
                "drawing_furniture": {rows[3]["source_primitive_ref"]},
            }
            disposition_path = root / "dispositions.pack"
            manifest = write_source_disposition_pack(
                descriptor_pack=descriptors, output_path=disposition_path,
                page={"page_rect_display": [0, 0, 100, 100],
                      "accepted_view_role": "main_plan_view"},
                regions=[{"id": "title", "state": "accepted",
                          "region_role": "title_block_revision_stamp",
                          "bbox_display": [80, 0, 100, 100]}],
                role_refs=roles)
            disposition = SourceDispositionPack(disposition_path, manifest)
            records = list(disposition.records())
            self.assertEqual([row["primary_disposition"] for row in records], [
                "route_evidence", "equipment_fitting_evidence", "annotation_dimension",
                "drawing_furniture", "unresolved_drawing_view_geometry",
                "excluded_non_view_content",
            ])
            self.assertIn("route_evidence", records[1]["candidate_roles"])
            self.assertIn("equipment_fitting_evidence", records[1]["candidate_roles"])
            self.assertIn("route_evidence", records[5]["candidate_roles"])
            self.assertIn("excluded_non_view_content", records[5]["candidate_roles"])
            self.assertEqual(manifest["record_count"], 6)
            self.assertEqual(sum(manifest["disposition_counts"].values()), 6)
            self.assertEqual(manifest["unaccounted_segment_count"], 0)
            self.assertTrue(manifest["exactly_one_primary_disposition"])
            self.assertEqual(validate_source_denominator(
                descriptor_pack=descriptors, disposition_pack=disposition), [])

    def test_m3_expands_routes_but_references_the_complete_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            descriptor_path = root / "native.pack"
            writer = NativeDescriptorPackWriter(
                descriptor_path, page_ref="page.5",
                pdf_to_display_matrix=[1, 0, 0, 1, 0, 0],
                minimum_member_length=1000)
            route = _row(1, [[10, 10], [20, 10]])
            unresolved = _row(2, [[30, 10], [40, 10]])
            writer.append(route)
            writer.append(unresolved)
            descriptors = NativeDescriptorPack(descriptor_path, writer.finish())
            disposition_path = root / "dispositions.pack"
            disposition_manifest = write_source_disposition_pack(
                descriptor_pack=descriptors, output_path=disposition_path,
                page={"page_rect_display": [0, 0, 100, 100],
                      "accepted_view_role": "main_plan_view"},
                regions=[], role_refs={"route_evidence": {route["source_primitive_ref"]}})
            dispositions = SourceDispositionPack(disposition_path, disposition_manifest)
            page = build_mep_route_page_from_denominator(
                page_scope={
                    "record_type": "mep_sheet_page_record", "id": "scope.5",
                    "page_ref": "page.5", "page_number": 5, "role": "plan",
                    "fields": {}},
                descriptor_pack=descriptors, disposition_pack=dispositions)
            self.assertEqual(page["summary"]["fragment_count"], 1)
            self.assertEqual(page["source_denominator"]["native_segment_count"], 2)
            self.assertEqual(page["source_denominator"]["expanded_route_evidence_count"], 1)
            self.assertEqual(page["source_denominator"]["unaccounted_segment_count"], 0)
            self.assertFalse(page["source_denominator"]
                             ["source_existence_selected_by_envelope_or_leader"])

    def test_legacy_m3_membership_is_candidate_only_not_primary_route_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            descriptor_path = root / "native.pack"
            writer = NativeDescriptorPackWriter(
                descriptor_path, page_ref="page.5",
                pdf_to_display_matrix=[1, 0, 0, 1, 0, 0],
                minimum_member_length=1000)
            candidate = _row(1, [[10, 10], [20, 10]])
            writer.append(candidate)
            descriptors = NativeDescriptorPack(descriptor_path, writer.finish())
            disposition_path = root / "dispositions.pack"
            manifest = write_source_disposition_pack(
                descriptor_pack=descriptors, output_path=disposition_path,
                page={"page_rect_display": [0, 0, 100, 100],
                      "accepted_view_role": "main_plan_view"},
                regions=[], role_refs={
                    "legacy_m3_route_candidate": {candidate["source_primitive_ref"]}})
            record = next(SourceDispositionPack(disposition_path, manifest).records())
            self.assertEqual("unresolved_drawing_view_geometry",
                             record["primary_disposition"])
            self.assertIn("legacy_m3_route_candidate", record["candidate_roles"])
            self.assertNotIn("route_evidence", record["candidate_roles"])


if __name__ == "__main__":
    unittest.main()
