import tempfile
from pathlib import Path
import unittest

from tests.test_mep_page_wide_path_classification import _row
from src.drawing_engine.disciplines.mep.mep_native_descriptor_pack import NativeDescriptorPack, NativeDescriptorPackWriter
from src.drawing_engine.disciplines.mep.mep_native_path_pack import NativePathPack, write_native_path_pack
from src.drawing_engine.disciplines.mep.mep_page_wide_path_classification import (
    PageWidePathRolePack, write_page_wide_path_roles,
)
from src.drawing_engine.disciplines.mep.mep_page_wide_route_recovery import (
    _observed_connector_candidates, _transverse_attachment_indices,
    build_page_wide_route_recovery,
    discover_page_wide_outline_corridors,
    nominate_annotation_outline_styles,
    validate_page_wide_route_recovery,
)
from src.drawing_engine.disciplines.mep.mep_source_primitive_denominator import (
    SourceDispositionPack, write_source_disposition_pack,
)


class PageWideRouteRecoveryTests(unittest.TestCase):
    def test_explicit_width_keeps_wide_pipe_below_generic_slender_limit(self):
        class Records:
            def __init__(self, rows):
                self.rows = rows

            def records(self):
                return iter(self.rows)

        paths = [{
            "path_ordinal": index, "drawing_ordinal": index,
            "first_descriptor_ordinal": index, "source_segment_count": 1,
            "style_id": 0, "kind_mask": 1, "endpoint_closed": False,
            "start_display": [10, y], "end_display": [190, y],
            "source_segment_length_points": 180,
        } for index, y in enumerate((20, 36.08))]
        arguments = dict(
            page_ref="page.test", path_pack=Records(paths),
            role_pack=Records([{"flags": 2,
                                "role": "anchored_route_style_candidate"}
                               for _ in paths]),
            styles=[{"stroke": [1, 0, .25], "width": .72}])
        self.assertEqual([], discover_page_wide_outline_corridors(
            **arguments)["accepted_corridors"])
        recovered = discover_page_wide_outline_corridors(
            **arguments, annotated_widths_by_style={0: [15.0]})
        self.assertEqual(1, len(recovered["accepted_corridors"]))
        self.assertTrue(recovered["accepted_corridors"][0]["certificates"][
            "annotation_supported_width_search"])
        self.assertFalse(recovered["accepted_corridors"][0][
            "system_identity_established"])

    def test_ambiguous_coloured_family_uses_current_paths_without_legacy_bits(self):
        class Records:
            def __init__(self, rows):
                self.rows = rows

            def records(self):
                return iter(self.rows)

        paths = []
        for style_id, y in ((0, 10), (1, 30), (2, 50)):
            for offset in (0, 2):
                ordinal = len(paths)
                paths.append({
                    "path_ordinal": ordinal, "drawing_ordinal": ordinal,
                    "first_descriptor_ordinal": ordinal,
                    "source_segment_count": 1, "style_id": style_id,
                    "kind_mask": 1, "endpoint_closed": False,
                    "start_display": [10, y + offset],
                    "end_display": [60, y + offset],
                    "source_segment_length_points": 50,
                })
        styles = [
            {"stroke": [0, .8, .2], "width": .72},
            {"stroke": [.4, .8, .5], "width": .72},
            {"stroke": [.6, .6, .6], "width": .72},
        ]
        annotations = [{
            "id": "annotation." + direction,
            "bbox_display": [65, 20, 80, 26],
            "system_candidate": "condenser_water_" + direction,
            "nominal_size_inches": 1.0,
        } for direction in ("supply", "return")]
        result = nominate_annotation_outline_styles(
            path_pack=Records(paths),
            role_pack=Records([{"flags": 0,
                                "role": "unresolved_drawing_view_geometry"}
                               for _ in paths]),
            styles=styles, route_annotations=annotations,
            drawing_inches_per_paper_inch=36)
        self.assertEqual([0, 1], result["nominated_style_ids"])
        self.assertEqual(2, len(result["supported_style_correlations"]))
        for row in result["supported_style_correlations"]:
            self.assertEqual(["condenser_water_return", "condenser_water_supply"],
                             row["system_candidates"])
            self.assertFalse(row["system_identity_established"])
            self.assertFalse(row["style_selection_unique"])

    def test_short_interior_crossing_is_measured_attachment_negative(self):
        rows = [
            {"polyline_display": [[0, 0], [100, 0]],
             "corridor_width_display_points": 10},
            {"polyline_display": [[50, -10], [50, 10]],
             "corridor_width_display_points": 1},
            {"polyline_display": [[100, 0], [100, 30]],
             "corridor_width_display_points": 3},
        ]
        self.assertEqual({1}, _transverse_attachment_indices(rows))
        self.assertEqual(set(), _transverse_attachment_indices([
            {"polyline_display": [[0, 0], [360, 0]],
             "corridor_width_display_points": 1.32},
            {"polyline_display": [[50, -30], [50, 35]],
             "corridor_width_display_points": 16.08},
        ]))

    def test_projected_endpoint_geometry_classifies_L_and_T_candidates(self):
        def row(identifier, points):
            return {
                "id": identifier,
                "state": "supported_unidentified_mep_candidate",
                "polyline_display": points,
                "corridor_width_display_points": 3.0,
            }

        elbow = _observed_connector_candidates(
            page_ref="page.5", route_rows=[
                row("a", [[0, 0], [10, 0]]),
                row("b", [[10, 0], [10, 10]]),
            ])
        self.assertEqual(["elbow_L"], [item["generic_class"] for item in elbow])
        tee = _observed_connector_candidates(
            page_ref="page.5", route_rows=[
                row("a", [[0, 0], [10, 0]]),
                row("b", [[10, 0], [20, 0]]),
                row("c", [[10, 0], [10, 10]]),
            ])
        self.assertEqual(["tee_T"], [item["generic_class"] for item in tee])
        self.assertIsNone(tee[0]["physical_count"])

    def test_current_authored_paths_discover_corridor_without_legacy_m3(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            style = {"width": .72, "stroke": [1, 0, .247], "fill": None,
                     "dash": "[] 0"}
            rows = [
                _row(1, [10, 20], [90, 20], style),
                _row(2, [10, 23], [90, 23], style),
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
            disposition_path = root / "disposition.pack"
            disposition_manifest = write_source_disposition_pack(
                descriptor_pack=descriptors, output_path=disposition_path,
                page={"page_rect_display": [0, 0, 1000, 1000],
                      "accepted_view_role": "main_plan_view"}, regions=[], role_refs={})
            dispositions = SourceDispositionPack(disposition_path, disposition_manifest)
            native_path = root / "paths.pack"
            native_manifest = write_native_path_pack(
                descriptor_pack=descriptors, disposition_pack=dispositions,
                output_path=native_path)
            paths = NativePathPack(native_path, native_manifest)
            role_path = root / "roles.pack"
            role_manifest = write_page_wide_path_roles(
                path_pack=paths, styles=descriptor_manifest["styles"],
                output_path=role_path, text_boxes_display=[], anchored_style_ids={0},
                outlined_member_drawing_ordinals=set(),
                page_rect_display=[0, 0, 1000, 1000])
            roles = PageWidePathRolePack(role_path, role_manifest)
            discovery = discover_page_wide_outline_corridors(
                page_ref="page.5", path_pack=paths, role_pack=roles,
                styles=descriptor_manifest["styles"])
            self.assertEqual(1, discovery["summary"]["mutually_unique_corridor_count"])
            self.assertEqual([0, 1], discovery["accepted_corridors"][0][
                "member_path_ordinals"])
            self.assertFalse(discovery["authority"][
                "legacy_M3_controls_candidate_existence"])

    def test_annotation_nominated_neutral_style_recovers_current_corridor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            style = {"width": .72, "stroke": [0, 0, 0], "fill": None,
                     "dash": "[] 0"}
            rows = [
                _row(1, [30, 40], [60, 40], style),
                _row(2, [30, 42], [60, 42], style),
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
            disposition_path = root / "disposition.pack"
            disposition_manifest = write_source_disposition_pack(
                descriptor_pack=descriptors, output_path=disposition_path,
                page={"page_rect_display": [0, 0, 1000, 1000],
                      "accepted_view_role": "main_plan_view"}, regions=[],
                role_refs={"legacy_m3_route_candidate": {
                    row["source_primitive_ref"] for row in rows}})
            dispositions = SourceDispositionPack(disposition_path, disposition_manifest)
            native_path = root / "paths.pack"
            native_manifest = write_native_path_pack(
                descriptor_pack=descriptors, disposition_pack=dispositions,
                output_path=native_path)
            paths = NativePathPack(native_path, native_manifest)
            role_path = root / "roles.pack"
            role_manifest = write_page_wide_path_roles(
                path_pack=paths, styles=descriptor_manifest["styles"],
                output_path=role_path, text_boxes_display=[], anchored_style_ids=set(),
                outlined_member_drawing_ordinals={1, 2},
                page_rect_display=[0, 0, 1000, 1000])
            roles = PageWidePathRolePack(role_path, role_manifest)
            discovery = discover_page_wide_outline_corridors(
                page_ref="page.5", path_pack=paths, role_pack=roles,
                styles=descriptor_manifest["styles"], additional_style_ids={0})
            self.assertEqual(1, len(discovery["accepted_corridors"]))
            self.assertTrue(discovery["accepted_corridors"][0]["certificates"][
                "annotation_nominated_neutral_style"])
            annotation = {
                "id": "annotation.cd", "bbox_display": [38, 45, 52, 51],
                "system_candidate": "condensate_drain",
                "nominal_size_inches": 1.0,
            }
            payload = build_page_wide_route_recovery(
                page_ref="page.5", path_pack=paths, role_pack=roles,
                styles=descriptor_manifest["styles"], text_boxes_display=[],
                outlined_composites=discovery["accepted_corridors"],
                m4_relations=[], fragment_source_refs={},
                route_annotations=[annotation],
                drawing_inches_per_paper_inch=36,
                annotation_supported_style_ids={0})
            recovered = payload["outlined_corridor_components"][0]
            self.assertEqual("supported_unidentified_mep_candidate",
                             recovered["state"])
            self.assertEqual(["condensate_drain"], recovered["candidate_systems"])
            self.assertFalse(recovered["attributes"]["route_system"]["state"] ==
                             "accepted")

    def test_short_parallel_attachment_does_not_become_corridor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            style = {"width": .72, "stroke": [1, 0, .247], "fill": None,
                     "dash": "[] 0"}
            rows = [
                _row(1, [10, 20], [10, 26], style),
                _row(2, [16, 20], [16, 26], style),
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
            disposition_path = root / "disposition.pack"
            disposition_manifest = write_source_disposition_pack(
                descriptor_pack=descriptors, output_path=disposition_path,
                page={"page_rect_display": [0, 0, 100, 100],
                      "accepted_view_role": "main_plan_view"}, regions=[], role_refs={})
            dispositions = SourceDispositionPack(disposition_path, disposition_manifest)
            native_path = root / "paths.pack"
            native_manifest = write_native_path_pack(
                descriptor_pack=descriptors, disposition_pack=dispositions,
                output_path=native_path)
            paths = NativePathPack(native_path, native_manifest)
            role_path = root / "roles.pack"
            role_manifest = write_page_wide_path_roles(
                path_pack=paths, styles=descriptor_manifest["styles"],
                output_path=role_path, text_boxes_display=[], anchored_style_ids={0},
                outlined_member_drawing_ordinals=set(),
                page_rect_display=[0, 0, 100, 100])
            roles = PageWidePathRolePack(role_path, role_manifest)
            discovery = discover_page_wide_outline_corridors(
                page_ref="page.5", path_pack=paths, role_pack=roles,
                styles=descriptor_manifest["styles"])
            self.assertEqual([], discovery["accepted_corridors"])

    def test_long_slender_outlined_corridor_is_supported_without_naming_system(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            style = {"width": .72, "stroke": [0, .24707, 1], "fill": None,
                     "dash": "[] 0"}
            rows = [
                _row(1, [10, 20], [90, 20], style),
                _row(2, [10, 22], [90, 22], style),
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
            disposition_path = root / "disposition.pack"
            disposition_manifest = write_source_disposition_pack(
                descriptor_pack=descriptors, output_path=disposition_path,
                page={"page_rect_display": [0, 0, 100, 100],
                      "accepted_view_role": "main_plan_view"}, regions=[], role_refs={})
            dispositions = SourceDispositionPack(disposition_path, disposition_manifest)
            native_path = root / "paths.pack"
            native_manifest = write_native_path_pack(
                descriptor_pack=descriptors, disposition_pack=dispositions,
                output_path=native_path)
            paths = NativePathPack(native_path, native_manifest)
            role_path = root / "roles.pack"
            role_manifest = write_page_wide_path_roles(
                path_pack=paths, styles=descriptor_manifest["styles"],
                output_path=role_path, text_boxes_display=[], anchored_style_ids={0},
                outlined_member_drawing_ordinals={1, 2},
                page_rect_display=[0, 0, 100, 100])
            roles = PageWidePathRolePack(role_path, role_manifest)
            composite = {
                "id": "outline.1", "page_ref": "page.5", "state": "accepted",
                "member_source_primitive_refs": [
                    rows[0]["source_primitive_ref"], rows[1]["source_primitive_ref"]],
                "derived_geometry": {
                    "centreline_points_display": [[10, 21], [90, 21]],
                    "corridor_width_display_points": 2,
                    "projected_path_display_points": 80,
                },
                "method": {"name": "closed_parallel_route_envelope"},
                "certificates": {"mutual_unique_pairing": True},
            }
            payload = build_page_wide_route_recovery(
                page_ref="page.5", path_pack=paths, role_pack=roles,
                styles=descriptor_manifest["styles"], text_boxes_display=[],
                outlined_composites=[composite], m4_relations=[],
                fragment_source_refs={})
            self.assertEqual("supported_unidentified_mep_candidate",
                             payload["outlined_corridor_components"][0]["state"])
            self.assertIsNone(payload["outlined_corridor_components"][0][
                "attributes"]["route_system"]["value"])
            rebound = build_page_wide_route_recovery(
                page_ref="page.5", path_pack=paths, role_pack=roles,
                styles=descriptor_manifest["styles"], text_boxes_display=[],
                outlined_composites=[composite], m4_relations=[],
                fragment_source_refs={},
                route_annotations=[{
                    "id": "annotation.chws", "bbox_display": [40, 25, 60, 31],
                    "system_candidate": "chilled_water_supply",
                    "nominal_size_inches": 1.0,
                }], drawing_inches_per_paper_inch=36,
                annotation_supported_style_ids={99})
            self.assertEqual(["chilled_water_supply"], rebound[
                "outlined_corridor_components"][0]["candidate_systems"])
            self.assertIsNone(rebound["outlined_corridor_components"][0][
                "attributes"]["route_system"]["value"])

    def test_exact_microsegments_form_one_bounded_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            style = {"width": .72, "stroke": [1, 0, .247], "fill": None,
                     "dash": "[] 0"}
            rows = [
                _row(1, [10, 10], [20, 10], style),
                _row(2, [20, 10], [30, 10], style),
                _row(3, [30, 10], [40, 10], style),
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
            disposition_path = root / "disposition.pack"
            disposition_manifest = write_source_disposition_pack(
                descriptor_pack=descriptors, output_path=disposition_path,
                page={"page_rect_display": [0, 0, 100, 100],
                      "accepted_view_role": "main_plan_view"}, regions=[],
                role_refs={"legacy_m3_route_candidate": {
                    rows[0]["source_primitive_ref"]}})
            dispositions = SourceDispositionPack(disposition_path, disposition_manifest)
            native_path = root / "paths.pack"
            native_manifest = write_native_path_pack(
                descriptor_pack=descriptors, disposition_pack=dispositions,
                output_path=native_path)
            paths = NativePathPack(native_path, native_manifest)
            role_path = root / "roles.pack"
            role_manifest = write_page_wide_path_roles(
                path_pack=paths, styles=descriptor_manifest["styles"],
                output_path=role_path, text_boxes_display=[],
                anchored_style_ids={0}, outlined_member_drawing_ordinals=set(),
                page_rect_display=[0, 0, 100, 100])
            roles = PageWidePathRolePack(role_path, role_manifest)
            payload = build_page_wide_route_recovery(
                page_ref="page.5", path_pack=paths, role_pack=roles,
                styles=descriptor_manifest["styles"], text_boxes_display=[],
                outlined_composites=[], m4_relations=[], fragment_source_refs={})
            self.assertEqual(1, len(payload["single_centreline_components"]))
            self.assertEqual(3, payload["coverage"][
                "anchored_candidate_accounted_path_count"])
            self.assertEqual(2, payload["coverage"]["exact_join_count"])
            self.assertEqual([], validate_page_wide_route_recovery(payload))

    def test_three_way_endpoint_without_typed_interface_is_not_a_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            style = {"width": .72, "stroke": [1, 0, .247], "fill": None,
                     "dash": "[] 0"}
            rows = [
                _row(1, [10, 20], [20, 20], style),
                _row(2, [20, 20], [30, 20], style),
                _row(3, [20, 20], [20, 30], style),
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
            disposition_path = root / "disposition.pack"
            disposition_manifest = write_source_disposition_pack(
                descriptor_pack=descriptors, output_path=disposition_path,
                page={"page_rect_display": [0, 0, 100, 100],
                      "accepted_view_role": "main_plan_view"}, regions=[],
                role_refs={})
            dispositions = SourceDispositionPack(disposition_path, disposition_manifest)
            native_path = root / "paths.pack"
            native_manifest = write_native_path_pack(
                descriptor_pack=descriptors, disposition_pack=dispositions,
                output_path=native_path)
            paths = NativePathPack(native_path, native_manifest)
            role_path = root / "roles.pack"
            role_manifest = write_page_wide_path_roles(
                path_pack=paths, styles=descriptor_manifest["styles"],
                output_path=role_path, text_boxes_display=[], anchored_style_ids={0},
                outlined_member_drawing_ordinals=set(),
                page_rect_display=[0, 0, 100, 100])
            roles = PageWidePathRolePack(role_path, role_manifest)
            payload = build_page_wide_route_recovery(
                page_ref="page.5", path_pack=paths, role_pack=roles,
                styles=descriptor_manifest["styles"], text_boxes_display=[],
                outlined_composites=[], m4_relations=[], fragment_source_refs={})
            self.assertEqual(1, payload["coverage"][
                "unresolved_branch_or_attachment_candidate_count"])
            self.assertEqual(0, payload["coverage"][
                "accepted_projected_branch_count"])
            self.assertEqual([], payload["accepted_projected_branch_certificates"])
            self.assertEqual([], payload["single_centreline_components"])


if __name__ == "__main__":
    unittest.main()
