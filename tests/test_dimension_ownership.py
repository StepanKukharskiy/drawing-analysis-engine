import unittest
from dataclasses import replace

import fitz

from src.drawing_engine.core.dimension_attachment import attach_dimensions, propose_dimensions
from src.drawing_engine.core.dimension_ownership import (
    adjudicate_dimension_proposals,
    resolve_dimension_ownership,
    resolve_raster_dimension_ownership,
)


def add_dimension(page: fitz.Page, y: float) -> None:
    page.insert_text((140, y - 5), "100", fontsize=10)
    page.draw_line((100, y), (200, y), width=0.5)
    page.draw_line((100, y), (100, y + 30), width=0.5)
    page.draw_line((200, y), (200, y + 30), width=0.5)
    for x in (100, 200):
        page.draw_line((x, y), (x + 3, y + 3), width=0.5)
        page.draw_line((x, y), (x - 3, y - 3), width=0.5)
    page.draw_line((100, y + 30), (200, y + 30), width=1.2)


def add_terminal_less_dimension(
    page: fitz.Page,
    x0: float,
    x1: float,
    y: float,
    value: str,
    *,
    left_depth: float = 30.0,
    right_depth: float = 30.0,
) -> None:
    page.insert_text(((x0 + x1) / 2 - 10, y - 5), value, fontsize=10)
    page.draw_line((x0, y), (x1, y), width=0.5)
    page.draw_line((x0, y), (x0, y + left_depth), width=0.5)
    page.draw_line((x1, y), (x1, y + right_depth), width=0.5)
    # Separate native target primitives continue beyond each extension. Their
    # exact shared endpoints are the measured geometry, including projected
    # dimensions whose targets lie at different cross-axis coordinates.
    left_tail = 15 if left_depth >= 0 else -15
    right_tail = 15 if right_depth >= 0 else -15
    page.draw_line((x0, y + left_depth), (x0, y + left_depth + left_tail), width=1.2)
    page.draw_line((x1, y + right_depth), (x1, y + right_depth + right_tail), width=1.2)


def add_short_dimension(page: fitz.Page, x0: float, x1: float, y: float, value: str) -> None:
    page.insert_text(((x0 + x1) / 2 - 5, y - 2), value, fontsize=5)
    page.draw_line((x0, y), (x1, y), width=0.5)
    page.draw_line((x0, y - 3), (x0, y + 12), width=0.5)
    page.draw_line((x1, y - 3), (x1, y + 12), width=0.5)
    for x in (x0, x1):
        page.draw_line((x - 1.5, y - 1.5), (x + 1.5, y + 1.5), width=0.5)


class DimensionOwnershipTest(unittest.TestCase):
    def _terminal_less_result(self, page: fitz.Page):
        proposals = propose_dimensions(page)
        ownership = resolve_dimension_ownership(
            page,
            proposals,
            [{"id": "view.terminal-less", "bbox_display": [20, 20, 300, 180]}],
            [],
        )
        return proposals, adjudicate_dimension_proposals(proposals, ownership)

    def test_ambiguous_proposals_reach_ownership_before_adjudication(self):
        document = fitz.open()
        page = document.new_page(width=400, height=320)
        for y in (50.0, 140.0, 230.0):
            add_dimension(page, y)
        proposals = tuple(
            replace(item, status="proposed", proposal_status="ambiguous")
            for item in attach_dimensions(page)
        )
        ownership = resolve_dimension_ownership(
            page,
            proposals,
            [{"id": "view.001", "bbox_display": [80, 35, 220, 280]}],
            [],
        )

        self.assertEqual(ownership["summary"]["dimension_count"], 3)
        self.assertTrue(ownership["contract"]["all_dimension_proposals_are_evaluated"])
        self.assertTrue(all(len(item["measured_endpoints"]) == 2 for item in ownership["attachments"]))
        adjudicated, final_ownership, report = adjudicate_dimension_proposals(proposals, ownership)
        self.assertTrue(all(item.status == "accepted" for item in adjudicated))
        self.assertEqual(final_ownership["summary"]["accepted_count"], 3)
        self.assertEqual(report["summary"]["ownership_evaluated_count"], 3)
        self.assertTrue(report["contract"]["detector_score_alone_never_accepts"])

    def test_adjudication_abstains_when_owned_targets_are_ambiguous(self):
        document = fitz.open()
        page = document.new_page(width=400, height=320)
        for y in (50.0, 140.0, 230.0):
            add_dimension(page, y)
            page.draw_line((100, y + 30.4), (200, y + 30.4), width=1.2)
        proposals = tuple(replace(item, status="proposed") for item in attach_dimensions(page))
        ownership = resolve_dimension_ownership(
            page,
            proposals,
            [{"id": "view.001", "bbox_display": [80, 35, 220, 280]}],
            [],
        )
        adjudicated, final_ownership, report = adjudicate_dimension_proposals(proposals, ownership)

        self.assertTrue(all(item.status == "ambiguous" for item in adjudicated))
        self.assertEqual(final_ownership["relations"], [])
        self.assertEqual(report["summary"]["accepted_count"], 0)

    def test_dimension_endpoints_attach_to_exact_native_geometry(self):
        document = fitz.open()
        page = document.new_page(width=400, height=320)
        for y in (50.0, 140.0, 230.0):
            add_dimension(page, y)
        dimensions = attach_dimensions(page)
        views = [{"id": "view.001", "bbox_display": [80, 35, 220, 280]}]
        result = resolve_dimension_ownership(page, dimensions, views, [])

        self.assertEqual(result["summary"]["dimension_count"], 3)
        self.assertEqual(result["summary"]["accepted_count"], 3)
        self.assertEqual(len(result["relations"]), 3)
        for attachment in result["attachments"]:
            self.assertEqual(attachment["owner_entity_refs"], ["view.001"])
            self.assertEqual(len(attachment["primitive_refs"]), 1)
            self.assertEqual(len(attachment["topology_segment_refs"]), 1)
            self.assertTrue(all(row["selected_geometry_anchor_ref"] for row in attachment["measured_endpoints"]))
            self.assertTrue(all(row["selected_primitive_ref"] for row in attachment["measured_endpoints"]))

    def test_ambiguous_endpoint_targets_are_not_promoted(self):
        document = fitz.open()
        page = document.new_page(width=400, height=320)
        for y in (50.0, 140.0, 230.0):
            add_dimension(page, y)
            page.draw_line((100, y + 30.4), (200, y + 30.4), width=1.2)
        dimensions = attach_dimensions(page)
        result = resolve_dimension_ownership(
            page,
            dimensions,
            [{"id": "view.001", "bbox_display": [80, 35, 220, 280]}],
            [],
        )

        self.assertEqual(result["summary"]["accepted_count"], 0)
        self.assertEqual(result["summary"]["candidate_count"], 3)
        self.assertEqual(result["relations"], [])

    def test_detector_qualified_ticks_retain_legacy_page_scale_across_views(self):
        document = fitz.open()
        page = document.new_page(width=400, height=320)
        views = []
        for index, y in enumerate((50.0, 140.0, 230.0), start=1):
            add_dimension(page, y)
            views.append(
                {
                    "id": f"view.{index}",
                    "bbox_display": [80, y - 15, 220, y + 45],
                }
            )
        proposals = propose_dimensions(page)
        ownership = resolve_dimension_ownership(page, proposals, views, [])
        adjudicated, _, report = adjudicate_dimension_proposals(proposals, ownership)

        self.assertTrue(all(item.status == "accepted" for item in adjudicated))
        self.assertTrue(
            all(
                item["certificate"]["scale_certificate_basis"]
                == "legacy_explicit_terminal_page_consensus"
                and item["certificate"]["legacy_solver_eligible"]
                for item in report["records"]
            )
        )

    def test_terminal_less_repeated_pitch_with_projected_targets_is_accepted(self):
        document = fitz.open()
        page = document.new_page(width=340, height=200)
        add_terminal_less_dimension(page, 50, 149, 50, "100", right_depth=45)
        add_terminal_less_dimension(page, 150, 249, 50, "100", left_depth=45)

        proposals, (adjudicated, final_ownership, report) = self._terminal_less_result(page)

        self.assertEqual(len(proposals), 2)
        self.assertTrue(all(item.terminal_style == "terminal_less" for item in proposals))
        self.assertTrue(all(item.status == "accepted" for item in adjudicated))
        self.assertEqual(final_ownership["summary"]["accepted_count"], 2)
        self.assertEqual(report["summary"]["terminal_less_redundancy_count"], 2)
        for record in report["records"]:
            certificate = record["certificate"]
            self.assertEqual(certificate["terminal_less_redundancy"]["kind"], "repeated_pitch")
            self.assertTrue(certificate["view_local_scale"]["passed"])
            self.assertTrue(certificate["unique_preliminary_view_owner"])
            self.assertFalse(certificate["legacy_solver_eligible"])
        self.assertTrue(
            any(record["certificate"]["different_cross_axis_targets"] for record in report["records"])
        )

    def test_terminal_less_arithmetic_chain_is_recorded(self):
        document = fitz.open()
        page = document.new_page(width=340, height=220)
        add_terminal_less_dimension(page, 50, 249, 45, "200")
        add_terminal_less_dimension(page, 50, 149, 120, "100")
        add_terminal_less_dimension(page, 150, 249, 120, "100")

        _, (adjudicated, _, report) = self._terminal_less_result(page)

        self.assertTrue(all(item.status == "accepted" for item in adjudicated))
        overall = next(record for record in report["records"] if record["dimension_ref"] == adjudicated[0].attachment_id)
        certificate = overall["certificate"]["terminal_less_redundancy"]
        self.assertEqual(certificate["kind"], "arithmetic_chain")
        self.assertEqual(certificate["term_values_mm"], [100.0, 100.0])
        self.assertEqual(certificate["total_value_mm"], 200.0)
        self.assertEqual(certificate["arithmetic_residual_mm"], 0.0)

    def test_opposed_dimension_columns_do_not_cross_certify_arithmetic(self):
        document = fitz.open()
        page = document.new_page(width=340, height=260)
        add_terminal_less_dimension(page, 50, 249, 45, "200", left_depth=30, right_depth=30)
        add_terminal_less_dimension(page, 50, 149, 120, "100", left_depth=30, right_depth=30)
        add_terminal_less_dimension(page, 150, 249, 120, "100", left_depth=30, right_depth=30)
        # Same stations and arithmetic, but extensions face the other way.
        add_terminal_less_dimension(page, 50, 249, 220, "200", left_depth=-30, right_depth=-30)

        proposals, (_, _, report) = self._terminal_less_result(page)

        records = {
            item["dimension_ref"]: item for item in report["records"]
        }
        opposed = max(proposals, key=lambda item: item.text_bbox[1]).attachment_id
        self.assertFalse(records[opposed]["certificate"]["terminal_less_redundancy"])

    def test_terminal_less_geometry_without_redundancy_abstains(self):
        document = fitz.open()
        page = document.new_page(width=400, height=200)
        add_terminal_less_dimension(page, 40, 139, 50, "100")
        add_terminal_less_dimension(page, 220, 319, 50, "100")

        _, (adjudicated, final_ownership, report) = self._terminal_less_result(page)

        self.assertTrue(all(item.status == "ambiguous" for item in adjudicated))
        self.assertEqual(final_ownership["relations"], [])
        self.assertTrue(all("lacks arithmetic or repeated-pitch" in item["reason"] for item in report["records"]))

    def test_short_native_chain_interval_is_not_discarded(self):
        document = fitz.open()
        page = document.new_page(width=240, height=160)
        add_short_dimension(page, 80.0, 92.0, 55.0, "200")

        proposals = propose_dimensions(page)

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].value_mm, 200.0)
        self.assertAlmostEqual(
            abs(proposals[0].dimension_points[1][0] - proposals[0].dimension_points[0][0]),
            12.0,
        )

    def test_conflicting_view_local_scales_abstain(self):
        document = fitz.open()
        page = document.new_page(width=420, height=200)
        add_terminal_less_dimension(page, 40, 139, 50, "100")
        add_terminal_less_dimension(page, 180, 329, 50, "100")

        _, (adjudicated, _, report) = self._terminal_less_result(page)

        self.assertTrue(all(item.status == "ambiguous" for item in adjudicated))
        self.assertEqual(report["summary"]["view_local_scale_pass_count"], 0)
        self.assertTrue(all("view-local scale does not repeat" in item["reason"] for item in report["records"]))


def _raster_fixture(*, second_value=500.0, ambiguous_edge=False, split_owner=False, conflicting_ocr=False):
    chain_specs = [
        ("raster.dimension_chain.0001", 10.0, 110.0, 1000.0),
        ("raster.dimension_chain.0002", 10.0, 60.0, second_value),
    ]
    chains = []
    observations = []
    chain_results = []
    for index, (chain_ref, left, right, value) in enumerate(chain_specs, start=1):
        chains.append(
            {
                "id": chain_ref,
                "status": "complete_geometry_candidate",
                "orientation": "horizontal",
                "baseline_ref": f"raster.line.dimension.{index}.baseline",
                "extension_line_refs": [
                    f"raster.line.dimension.{index}.extension.1",
                    f"raster.line.dimension.{index}.extension.2",
                ],
                "terminal_refs": [
                    f"raster.line.dimension.{index}.terminal.1",
                    f"raster.line.dimension.{index}.terminal.2",
                ],
                "topology_edge_refs": [
                    f"raster.edge.dimension.{index}.baseline",
                    f"raster.edge.dimension.{index}.extension.1",
                    f"raster.edge.dimension.{index}.extension.2",
                ],
                "dimension_points_display": [[left, 20.0 + index * 8.0], [right, 20.0 + index * 8.0]],
                "measured_points_display": [[left, 50.0], [right, 50.0]],
            }
        )
        observation_ref = f"raster.dimension_label.{index:05d}"
        observations.append(
            {
                "id": observation_ref,
                "dimension_chain_ref": chain_ref,
                "numeric_value_candidate": value,
            }
        )
        chain_results.append(
            {
                "dimension_chain_ref": chain_ref,
                "status": (
                    "ambiguous_conflicting_numeric_observations"
                    if conflicting_ocr and index == 1
                    else "unique_numeric_observation"
                ),
                "observation_refs": (
                    [observation_ref, "raster.dimension_label.conflict"]
                    if conflicting_ocr and index == 1
                    else [observation_ref]
                ),
                "selected_observation_ref": None if conflicting_ocr and index == 1 else observation_ref,
                "reason": "conflicting crop readings" if conflicting_ocr and index == 1 else None,
            }
        )

    segments = [
        ("raster.edge.owned.left", "raster.line.owned.left", 10.0),
        ("raster.edge.owned.middle", "raster.line.owned.middle", 60.0),
        ("raster.edge.owned.right", "raster.line.owned.right", 110.0),
    ]
    if ambiguous_edge:
        segments.append(("raster.edge.owned.left_duplicate", "raster.line.owned.left_duplicate", 10.0))
    vertices = []
    contour_segments = []
    edges = []
    for index, (edge_ref, primitive_ref, x) in enumerate(segments, start=1):
        start_ref = f"raster.vertex.owned.{index}.start"
        end_ref = f"raster.vertex.owned.{index}.end"
        vertices.extend([{"id": start_ref}, {"id": end_ref}])
        edges.append({"id": edge_ref})
        contour_segments.append(
            {
                "id": edge_ref,
                "primitive_ref": primitive_ref,
                "start_vertex_id": start_ref,
                "end_vertex_id": end_ref,
                "start_display": [x, 40.0],
                "end_display": [x, 60.0],
                "axis": "vertical",
            }
        )
    contours = [
        {
            "id": "contour.raster.owner",
            "source_modality": "raster",
            "topology_component_ref": "raster.component.owner",
            "segments_display": contour_segments if not split_owner else contour_segments[:1],
        }
    ]
    if split_owner:
        contours.append(
            {
                "id": "contour.raster.other",
                "source_modality": "raster",
                "topology_component_ref": "raster.component.other",
                "segments_display": contour_segments[1:],
            }
        )
    annotation_edges = [
        {"id": edge_ref}
        for chain in chains
        for edge_ref in chain["topology_edge_refs"]
    ]
    return {
        "dimension_topology": {"chains": chains},
        "dimension_label_ocr": {"observations": observations, "chain_results": chain_results},
        "vector_topology": {
            "edges": [*edges, *annotation_edges],
            "vertices": vertices,
            "snap_tolerance_display_points": 0.75,
        },
        "views": [
            {
                "id": "view_hypothesis.raster.001",
                "source_modality": "raster",
                "bbox_display": [0.0, 0.0, 120.0, 80.0],
            }
        ],
        "contours": contours,
    }


class RasterDimensionOwnershipTest(unittest.TestCase):
    def _resolve(self, **kwargs):
        fixture = _raster_fixture(**kwargs)
        return resolve_raster_dimension_ownership(**fixture)

    def test_repeated_local_scale_and_common_contour_accept_candidates(self):
        result = self._resolve()

        self.assertEqual(result["summary"]["accepted_candidate_count"], 2)
        self.assertEqual(result["scale_scopes"][0]["status"], "accepted")
        self.assertAlmostEqual(result["scale_scopes"][0]["scale_points_per_mm"], 0.1)
        self.assertEqual(len(result["relations"]), 2)
        for candidate, attachment in zip(result["dimension_candidates"], result["attachments"]):
            self.assertEqual(candidate["status"], "accepted_dimension_ownership_candidate")
            self.assertFalse(candidate["solver_eligible"])
            self.assertEqual(attachment["owner_entity_refs"], ["contour.raster.owner"])
            self.assertTrue(all(row["selected_topology_edge_ref"] for row in attachment["measured_endpoints"]))
            self.assertTrue(all(row["selected_geometry_anchor_ref"] for row in attachment["measured_endpoints"]))
            self.assertFalse(attachment["solver_eligible"])

    def test_single_scale_observation_remains_unknown(self):
        fixture = _raster_fixture()
        fixture["dimension_topology"]["chains"] = fixture["dimension_topology"]["chains"][:1]
        fixture["dimension_label_ocr"]["observations"] = fixture["dimension_label_ocr"]["observations"][:1]
        fixture["dimension_label_ocr"]["chain_results"] = fixture["dimension_label_ocr"]["chain_results"][:1]
        result = resolve_raster_dimension_ownership(**fixture)

        self.assertEqual(result["summary"]["accepted_candidate_count"], 0)
        self.assertEqual(result["summary"]["single_scale_scope_count"], 1)
        self.assertEqual(result["dimension_candidates"][0]["metric_scale"]["status"], "unknown")
        self.assertIn("one local scale observation", result["dimension_candidates"][0]["reason"])

    def test_conflicting_scale_or_crop_readings_remain_unknown(self):
        scale_conflict = self._resolve(second_value=400.0)
        self.assertEqual(scale_conflict["summary"]["accepted_candidate_count"], 0)
        self.assertIn("scale observations conflict", scale_conflict["dimension_candidates"][0]["reason"])

        crop_conflict = self._resolve(conflicting_ocr=True)
        self.assertEqual(crop_conflict["summary"]["accepted_candidate_count"], 0)
        self.assertEqual(crop_conflict["summary"]["conflicting_ocr_candidate_count"], 1)
        self.assertEqual(crop_conflict["dimension_candidates"][0]["ocr_status"], "ambiguous_conflicting_numeric_observations")
        self.assertIsNone(crop_conflict["dimension_candidates"][0]["value_mm"])

    def test_ambiguous_edge_or_split_contour_owner_remains_unknown(self):
        ambiguous = self._resolve(ambiguous_edge=True)
        self.assertEqual(ambiguous["summary"]["accepted_candidate_count"], 0)
        self.assertEqual(ambiguous["attachments"][0]["measured_endpoints"][0]["state"], "candidate")

        split = self._resolve(split_owner=True)
        self.assertEqual(split["summary"]["accepted_candidate_count"], 0)
        self.assertEqual(split["attachments"][0]["owner_entity_refs"], [])
        self.assertEqual(split["relations"], [])


if __name__ == "__main__":
    unittest.main()
