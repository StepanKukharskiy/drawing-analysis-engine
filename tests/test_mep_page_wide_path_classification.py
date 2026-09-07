import tempfile
from pathlib import Path
import unittest

from src.drawing_engine.disciplines.mep.mep_native_descriptor_pack import NativeDescriptorPack, NativeDescriptorPackWriter
from src.drawing_engine.disciplines.mep.mep_native_path_pack import NativePathPack, write_native_path_pack
from src.drawing_engine.disciplines.mep.mep_page_wide_path_classification import (
    PageWidePathRolePack, validate_page_wide_path_roles,
    write_page_wide_path_roles,
)
from src.drawing_engine.disciplines.mep.mep_source_primitive_denominator import (
    SourceDispositionPack, write_source_disposition_pack,
)


def _row(index, start, end, style):
    source = f"drawing[{index}].item[0].segment[0]"
    box = [min(start[0], end[0]), min(start[1], end[1]),
           max(start[0], end[0]), max(start[1], end[1])]
    dx, dy = end[0] - start[0], end[1] - start[1]
    axis = "horizontal" if dy == 0 else "vertical" if dx == 0 else "oblique"
    return {
        "id": f"mep_native_target_primitive.{index:020x}", "page_ref": "page.5",
        "source_primitive_ref": source,
        "source_native_segment": {
            "id": source, "drawing_ref": f"drawing[{index}]",
            "primitive_ref": f"drawing[{index}].item[0]", "item_index": 0,
            "part_index": 0, "kind": "line", "start_display": start,
            "end_display": end, "control_points_display": [],
            "sample_points_display": [], "axis": axis,
            "length_points": (dx * dx + dy * dy) ** .5,
            "bbox_display": box, "style": style,
        },
        "source_drawing_close_path": False, "points_display": [start, end],
        "bbox_display": box, "search_bbox_display": box, "search_refs": [],
    }


class PageWidePathClassificationTests(unittest.TestCase):
    def test_legacy_membership_never_grants_route_role(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchored = {"width": .72, "stroke": [1, 0, .247], "fill": None,
                        "dash": "[] 0"}
            neutral = {"width": .24, "stroke": [0, 0, 0], "fill": None,
                       "dash": "[] 0"}
            rows = [
                _row(1, [10, 10], [50, 10], neutral),
                _row(2, [10, 20], [50, 20], anchored),
                _row(3, [80, 10], [80, 90], neutral),
            ]
            descriptor_path = root / "descriptor.pack"
            writer = NativeDescriptorPackWriter(
                descriptor_path, page_ref="page.5",
                pdf_to_display_matrix=[1, 0, 0, 1, 0, 0],
                minimum_member_length=1000)
            for row in rows:
                writer.append(row)
            descriptor_manifest = writer.finish()
            descriptors = NativeDescriptorPack(descriptor_path, descriptor_manifest)
            disposition_path = root / "dispositions.pack"
            disposition_manifest = write_source_disposition_pack(
                descriptor_pack=descriptors, output_path=disposition_path,
                page={"page_rect_display": [0, 0, 100, 100],
                      "accepted_view_role": "main_plan_view"}, regions=[],
                role_refs={"legacy_m3_route_candidate": {
                    rows[0]["source_primitive_ref"]}})
            dispositions = SourceDispositionPack(disposition_path, disposition_manifest)
            native_path = root / "paths.pack"
            native_path_manifest = write_native_path_pack(
                descriptor_pack=descriptors, disposition_pack=dispositions,
                output_path=native_path)
            paths = NativePathPack(native_path, native_path_manifest)
            roles_path = root / "roles.pack"
            roles_manifest = write_page_wide_path_roles(
                path_pack=paths, styles=descriptor_manifest["styles"],
                output_path=roles_path, text_boxes_display=[],
                anchored_style_ids={1}, outlined_member_drawing_ordinals=set(),
                page_rect_display=[0, 0, 100, 100])
            roles = PageWidePathRolePack(roles_path, roles_manifest)
            records = list(roles.records())
            self.assertNotEqual("anchored_route_style_candidate", records[0]["role"])
            self.assertEqual("anchored_route_style_candidate", records[1]["role"])
            self.assertEqual("architectural_boundary_candidate", records[2]["role"])
            self.assertEqual([], validate_page_wide_path_roles(
                path_pack=paths, role_pack=roles))


if __name__ == "__main__":
    unittest.main()
