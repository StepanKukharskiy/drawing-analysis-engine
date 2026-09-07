import tempfile
from pathlib import Path
import unittest

from src.drawing_engine.disciplines.mep.mep_native_descriptor_pack import NativeDescriptorPack, NativeDescriptorPackWriter
from src.drawing_engine.disciplines.mep.mep_native_path_pack import (
    NativePathPack, validate_native_path_pack, write_native_path_pack,
)
from src.drawing_engine.disciplines.mep.mep_source_primitive_denominator import (
    SourceDispositionPack, write_source_disposition_pack,
)


def _row(drawing, item, part, start, end, *, style=None):
    source_ref = f"drawing[{drawing}].item[{item}].segment[{part}]"
    box = [min(start[0], end[0]), min(start[1], end[1]),
           max(start[0], end[0]), max(start[1], end[1])]
    style = style or {"width": .5, "stroke": [1, 0, 0], "fill": None,
                      "dash": "[] 0"}
    return {
        "id": f"mep_native_target_primitive.{drawing:08x}{item:06x}{part:06x}",
        "page_ref": "page.5", "source_primitive_ref": source_ref,
        "source_native_segment": {
            "id": source_ref, "drawing_ref": f"drawing[{drawing}]",
            "primitive_ref": f"drawing[{drawing}].item[{item}]",
            "item_index": item, "part_index": part, "kind": "line",
            "start_display": start, "end_display": end,
            "control_points_display": [], "sample_points_display": [],
            "axis": "horizontal" if start[1] == end[1] else "vertical",
            "length_points": abs(end[0] - start[0]) + abs(end[1] - start[1]),
            "bbox_display": box, "style": style,
        },
        "source_drawing_close_path": False,
        "points_display": [start, end], "bbox_display": box,
        "search_bbox_display": box, "search_refs": [],
    }


class NativePathPackTests(unittest.TestCase):
    def test_authored_paths_partition_every_source_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            descriptor_path = root / "native.pack"
            writer = NativeDescriptorPackWriter(
                descriptor_path, page_ref="page.5",
                pdf_to_display_matrix=[1, 0, 0, 1, 0, 0],
                minimum_member_length=1000)
            rows = [
                _row(4, 0, 0, [10, 10], [20, 10]),
                _row(4, 1, 0, [20, 10], [20, 20]),
                _row(5, 0, 0, [30, 10], [40, 10]),
            ]
            for row in rows:
                writer.append(row)
            descriptor_manifest = writer.finish()
            descriptors = NativeDescriptorPack(descriptor_path, descriptor_manifest)
            disposition_path = root / "disposition.pack"
            disposition_manifest = write_source_disposition_pack(
                descriptor_pack=descriptors, output_path=disposition_path,
                page={"page_rect_display": [0, 0, 100, 100],
                      "accepted_view_role": "main_plan_view"}, regions=[],
                role_refs={"legacy_m3_route_candidate": {
                    rows[0]["source_primitive_ref"]}})
            dispositions = SourceDispositionPack(disposition_path, disposition_manifest)
            path = root / "paths.pack"
            path_manifest = write_native_path_pack(
                descriptor_pack=descriptors, disposition_pack=dispositions,
                output_path=path)
            paths = NativePathPack(path, path_manifest)
            records = list(paths.records())
            self.assertEqual(2, len(records))
            self.assertEqual(3, path_manifest["source_segment_count"])
            self.assertEqual(2, records[0]["source_segment_count"])
            self.assertEqual(2, records[0]["native_item_path_count"])
            self.assertIn("legacy_m3_route_candidate", records[0]["candidate_roles"])
            self.assertEqual([10, 10], records[0]["start_display"])
            self.assertEqual([20, 20], records[0]["end_display"])
            self.assertEqual([], validate_native_path_pack(
                descriptor_pack=descriptors, disposition_pack=dispositions,
                path_pack=paths))


if __name__ == "__main__":
    unittest.main()
