import copy
import hashlib
import json
from pathlib import Path
import unittest

from src.drawing_engine.disciplines.mep.mep_route_callout_tracing import build_route_callout_traces, validate_route_callout_traces


def route(ref, a, b, system=None, style=1, width=2):
    return {
        "id": ref, "state": "identified_mep_route" if system else "supported_unidentified_mep_candidate",
        "page_ref": "page", "style_id": style, "channel": "parallel_outline_corridor",
        "polyline_display": [a, b], "corridor_width_display_points": width,
        "source_path_ordinals": [ref],
        "attributes": {"route_system": {"state": "accepted" if system else "unknown",
            "value": system, "relation_refs": ["m4." + ref] if system else []}},
    }


def payload(rows):
    return {"outlined_corridor_components": rows, "precision_characterization": {
        "1": {"maximum_join_tolerance_display_points": .12},
        "2": {"maximum_join_tolerance_display_points": .12}}}


class RouteCalloutTracingTests(unittest.TestCase):
    def test_required_trace_label_moves_instead_of_disappearing(self):
        import fitz
        from tools.render_mep_page5_page_wide_recovery import _label
        with fitz.open() as pdf:
            page = pdf.new_page(width=600, height=600)
            occupied = [fitz.Rect(0, 270, 600, 330)]
            self.assertTrue(_label(page, [[250, 300], [350, 300]], "via R01",
                                   (1, 0, 0), page.rect, occupied, required=True))
            self.assertIn("via R01", page.get_text())

    def test_overlay_shows_anchor_reference_without_changing_quantities_or_identity(self):
        from tools.render_mep_page5_page_wide_recovery import _traced_candidate_rows
        a, b = route("a", [0, 0], [10, 0], "HHWS"), route("b", [10, 0], [20, 0])
        b["projected_path_display_points"] = 10
        original = copy.deepcopy(b)
        tracing = build_route_callout_traces(payload([a, b]))
        result = _traced_candidate_rows([b], [a], tracing)[0]
        self.assertEqual("via R01", result["trace_label"])
        self.assertEqual(["HHWS"], result["candidate_systems"])
        self.assertEqual(original, b)
        self.assertEqual(b["attributes"], result["attributes"])
        self.assertEqual(10, result["projected_path_display_points"])

    def test_chain_reaches_M4_anchor_without_promoting_identity(self):
        data = payload([route("a", [0, 0], [10, 0], "HHWS"),
                        route("b", [10, 0], [20, 0]), route("c", [20.1, 0], [30, 0])])
        original = copy.deepcopy(data)
        result = build_route_callout_traces(data)
        self.assertEqual(original, data)
        self.assertEqual(2, len(result["joins"]))
        trace = next(row for row in result["component_traces"] if row["component_ref"] == "c")
        self.assertEqual(["c", "b", "a"], trace["paths_to_anchors"][0]["component_path"])
        self.assertEqual(["HHWS"], trace["candidate_systems"])
        self.assertFalse(trace["system_binding_accepted"])
        self.assertEqual(2, result["summary"]["previously_unidentified_components_reaching_M4_anchor"])
        data["route_callout_tracing"] = result
        self.assertEqual([], validate_route_callout_traces(data))
        result["component_traces"][0]["candidate_systems"] = ["forged"]
        self.assertTrue(validate_route_callout_traces(data))

    def test_branches_crossings_and_unclassified_competitors_stop(self):
        base = [route("a", [0, 0], [10, 0], "HHWS"), route("b", [10, 0], [20, 0])]
        for competitor in (route("c", [10, 0], [10, 5]),
                           route("c", [10, -5], [10, 5]),
                           {**route("c", [10, 0], [20, 1]), "state": "unclassified_outlined_corridor"}):
            with self.subTest(competitor=competitor):
                result = build_route_callout_traces(payload(base + [competitor]))
                self.assertFalse(result["joins"])
                self.assertEqual(0, result["summary"]["previously_unidentified_components_reaching_M4_anchor"])

    def test_style_width_turn_overlap_and_precision_changes_stop(self):
        base = route("a", [0, 0], [10, 0], "HHWS")
        targets = [route("b", [10, 0], [20, 0], style=2),
                   route("b", [10, 0], [20, 0], width=3),
                   route("b", [10, 0], [10, 10]),
                   route("b", [10, 0], [0, 0]),
                   route("b", [10.13, 0], [20, 0])]
        for target in targets:
            with self.subTest(target=target):
                self.assertFalse(build_route_callout_traces(payload([base, target]))["joins"])
        data = payload([base, route("b", [10.1, 0], [20, 0])])
        data["precision_characterization"] = {}
        self.assertFalse(build_route_callout_traces(data)["joins"])

    def test_branch_inside_long_row_blocks_tracing_through_unsplit_scope(self):
        data = payload([route("a", [0, 0], [20, 0], "HHWS"),
                        route("b", [20, 0], [30, 0]),
                        route("branch", [10, 0], [10, 10])])
        result = build_route_callout_traces(data)
        self.assertFalse(result["joins"])
        self.assertIn("interior_branch_scope_requires_split", result["summary"]["stop_reason_counts"])

    def test_known_attribute_change_blocks_but_conflicting_remote_anchors_are_explicit(self):
        a, b = route("a", [0, 0], [10, 0], "HHWS"), route("b", [10, 0], [20, 0], "HHWR")
        self.assertFalse(build_route_callout_traces(payload([a, b]))["joins"])
        c = route("c", [20, 0], [30, 0], "HHWR")
        b = route("b", [10, 0], [20, 0])
        result = build_route_callout_traces(payload([a, b, c]))
        self.assertEqual("conflicting_callout_anchors", result["scopes"][0]["state"])
        self.assertFalse(result["scopes"][0]["system_binding_accepted"])

    def test_symbol_and_leader_seams_are_boundaries(self):
        data = payload([route("a", [0, 0], [10, 0], "HHWS"), route("b", [10, 0], [20, 0])])
        for field, row in (("fitting_or_equipment_symbol_candidates", {"id": "body", "bbox_display": [9, -1, 11, 1]}),
                           ("excluded_text_leader_candidates", {"id": "leader", "polyline_display": [[10, 0], [12, 3]]})):
            with self.subTest(field=field):
                test = {**data, field: [row]}
                result = build_route_callout_traces(test)
                self.assertFalse(result["joins"])
                self.assertIn(row["id"], result["scopes"][0]["terminals"][-1]["blocking_evidence_refs"])

    def test_candidate_annotations_remain_candidates_and_ignore_old_propagation(self):
        a, b = route("a", [0, 0], [10, 0]), route("b", [10, 0], [20, 0])
        a["route_candidate_support"] = {"annotation_refs": ["text"]}
        b["annotation_candidate_refs"] = ["wrong"]
        b["candidate_systems"] = ["wrong"]
        data = payload([a, b])
        data["route_annotations"] = [{"id": "text", "system_candidate": "HHWS"},
                                     {"id": "wrong", "system_candidate": "HHWR"}]
        result = build_route_callout_traces(data)
        self.assertEqual(["HHWS"], result["scopes"][0]["candidate_systems"])
        self.assertEqual(0, result["summary"]["components_reaching_accepted_M4_anchor"])
        reversed_data = {**data, "outlined_corridor_components": [b, a]}
        self.assertEqual(result, build_route_callout_traces(reversed_data))


class RouteCalloutTracingReplayTests(unittest.TestCase):
    def test_frozen_page_replays_every_component_without_quantity_changes(self):
        root = Path(__file__).resolve().parents[1]
        parent = json.loads((root / "output/mep-page5-page-wide-recovery-v18-2026-09-03/recovery.json").read_text())
        data = json.loads((root / "output/mep-page5-page-wide-recovery-v20-2026-09-03/recovery.json").read_text())
        self.assertEqual([], validate_route_callout_traces(data))
        for field in ("outlined_corridor_components", "single_centreline_components", "coverage"):
            self.assertEqual(parent[field], data[field])
        tracing = data["route_callout_tracing"]
        refs = [row["component_ref"] for row in tracing["component_traces"]]
        self.assertEqual(len(refs), len(set(refs)))
        self.assertEqual(398, len(refs))
        self.assertEqual(36, len(tracing["joins"]))
        self.assertEqual(2, tracing["summary"]["previously_unidentified_components_reaching_M4_anchor"])
        self.assertTrue(all(row["installed_length"] is None and row["purchase_length"] is None
                            for row in tracing["scopes"]))


class RouteCalloutTracingRenderTests(unittest.TestCase):
    def test_frozen_pdf_keeps_both_trace_labels_and_raster_base(self):
        import fitz
        root = Path(__file__).resolve().parents[1]
        path = root / "output/pdf/mep_page5_network_tracing_2026-09-03.pdf"
        manifest = json.loads(path.with_suffix(".manifest.json").read_text())
        self.assertEqual(manifest["output_pdf_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        with fitz.open(path) as pdf:
            self.assertEqual(1, len(pdf))
            self.assertEqual(1, len(pdf[0].get_images(full=True)))
            text = pdf[0].get_text()
            self.assertIn("3.75m via R13", text)
            self.assertIn("3.72m via R12", text)
            self.assertIn("393.22 m", text)
        self.assertIsNone(manifest["installed_length"])
        self.assertIsNone(manifest["purchase_length"])
