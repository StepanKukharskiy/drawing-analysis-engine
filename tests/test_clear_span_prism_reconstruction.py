import unittest

import fitz

from src.drawing_engine.disciplines.concrete.clear_span_prism_reconstruction import reconstruct_clear_span_prism


def _fixture():
    document = fitz.open()
    page = document.new_page(width=600, height=300)
    page.insert_textbox(
        fitz.Rect(40, 40, 180, 85),
        "300x200mm\nbeam to details",
        fontsize=9,
    )
    page.insert_textbox(
        fitz.Rect(340, 40, 480, 85),
        "300x200mm\nbeam to details",
        fontsize=9,
    )
    segmentation = {
        "segments": [
            {"id": "detail.scope", "state": "resolved", "bbox_display": [20, 20, 250, 150]},
            {"id": "section.scope", "state": "resolved", "bbox_display": [300, 20, 540, 150]},
        ]
    }
    ownership = {
        "attachments": [
            {
                "dimension_ref": "dimension.overall",
                "status": "accepted",
                "orientation": "vertical",
            },
            {
                "dimension_ref": "dimension.support.1",
                "status": "accepted",
                "geometry_anchor_refs": ["vertex.outer.1", "vertex.inner.1"],
            },
            {
                "dimension_ref": "dimension.clear",
                "status": "accepted",
                "geometry_anchor_refs": ["vertex.inner.1", "vertex.inner.2"],
            },
            {
                "dimension_ref": "dimension.support.2",
                "status": "accepted",
                "geometry_anchor_refs": ["vertex.inner.2", "vertex.outer.2"],
            },
        ]
    }
    refs = [
        "dimension.overall",
        "dimension.support.1",
        "dimension.clear",
        "dimension.support.2",
    ]
    adjudication = {
        "records": [
            {"dimension_ref": ref, "status": "accepted", "view_refs": ["plan.view"]}
            for ref in refs
        ],
        "terminal_less_redundancy_certificates": [
            {
                "dimension_ref": "dimension.overall",
                "kind": "arithmetic_chain",
                "supporting_dimension_refs": refs,
                "overall_dimension_ref": "dimension.overall",
                "term_dimension_refs": [
                    "dimension.support.1",
                    "dimension.clear",
                    "dimension.support.2",
                ],
                "term_values_mm": [200.0, 2150.0, 200.0],
                "total_value_mm": 2550.0,
                "arithmetic_residual_mm": 0.0,
                "extension_direction": -1,
            }
        ],
    }
    frame = {
        "relations": [
            {
                "id": "cut.1",
                "type": "cut_at",
                "state": "accepted",
                "parent_view_id": "plan.view",
                "section_view_id": "section.scope",
            }
        ],
        "object_scopes": [
            {
                "id": "physical.1",
                "relation_refs": ["cut.1"],
                "parent_view_ids": ["plan.view"],
                "section_view_ids": ["section.scope"],
                "supporting_projection_view_ids": ["detail.scope"],
                "shared_coordinate_scope_id": "coordinate.1",
            }
        ],
        "shared_coordinate_system": {
            "scopes": [
                {
                    "id": "coordinate.1",
                    "signed_orientation_certificate": {
                        "id": "signed.orientation.1",
                        "status": "insufficient_constraints",
                    },
                    "view_axis_mappings": [
                        {
                            "view_id": "plan.view",
                            "u": "X",
                            "v": "Y",
                            "normal": "Z",
                            "state": "resolved_relative",
                        },
                        {
                            "view_id": "section.scope",
                            "u": "X",
                            "v": "Z",
                            "normal": "Y",
                            "state": "resolved_relative",
                        },
                    ],
                    "contour_correspondences": [
                        {
                            "id": "correspondence.1",
                            "state": "accepted",
                            "selected": {
                                "signed_transform": {
                                    "state": "unresolved",
                                    "candidates": [
                                        {"sign": 1, "offset_mm": 50.0},
                                        {"sign": -1, "offset_mm": 950.0},
                                    ],
                                }
                            },
                        }
                    ],
                }
            ]
        },
    }
    return document, page, ownership, adjudication, segmentation, frame


class ClearSpanPrismReconstructionTest(unittest.TestCase):
    def test_duplicate_referred_callouts_and_plan_chain_close_clear_span_prism(self):
        document, page, ownership, adjudication, segmentation, frame = _fixture()
        self.addCleanup(document.close)

        result = reconstruct_clear_span_prism(
            page,
            ownership,
            adjudication,
            segmentation,
            frame,
            page_number=1,
        )

        self.assertEqual(result["status"], "reclosed_pass")
        self.assertEqual(len(result["member_size_callouts"]), 2)
        self.assertEqual(
            result["duplicate_projection_certificate"]["physical_instance_count"], 1
        )
        self.assertFalse(
            result["duplicate_projection_certificate"]["separate_callouts_are_additive"]
        )
        self.assertEqual(result["clear_span_certificate"]["clear_span_mm"], 2150.0)
        self.assertEqual(
            result["clear_span_certificate"]["termination_convention"],
            "inner_faces_clear_span_only",
        )
        self.assertEqual(
            result["unsigned_bounded_sweep_certificate"]["signed_orientation_state"],
            "unresolved",
        )
        self.assertAlmostEqual(
            result["clear_span_volume_candidate"]["value_m3"], 0.129
        )
        self.assertFalse(result["clear_span_volume_candidate"]["support_overlap_included"])
        self.assertFalse(result["clear_span_volume_candidate"]["quantity_eligible"])
        self.assertTrue(
            all(
                item["status"] == "pass"
                for item in result["solid_preview"]["validation"]["reprojections"]
            )
        )

    def test_missing_duplicate_projection_abstains(self):
        document, page, ownership, adjudication, segmentation, frame = _fixture()
        self.addCleanup(document.close)
        segmentation["segments"] = segmentation["segments"][:1]

        result = reconstruct_clear_span_prism(
            page,
            ownership,
            adjudication,
            segmentation,
            frame,
            page_number=1,
        )

        self.assertEqual(result["status"], "insufficient_constraints")
        self.assertEqual(
            result["reason_code"],
            "unique_referred_member_duplicate_projection_unresolved",
        )

    def test_clear_span_prism_does_not_require_contour_correspondence(self):
        document, page, ownership, adjudication, segmentation, frame = _fixture()
        self.addCleanup(document.close)
        frame["shared_coordinate_system"]["scopes"][0]["contour_correspondences"] = []

        result = reconstruct_clear_span_prism(
            page,
            ownership,
            adjudication,
            segmentation,
            frame,
            page_number=1,
        )

        self.assertEqual(result["status"], "reclosed_pass")
        self.assertEqual(
            {
                item["sign"]
                for item in result["unsigned_bounded_sweep_certificate"][
                    "signed_transform_alternatives"
                ]
            },
            {-1, 1},
        )


if __name__ == "__main__":
    unittest.main()
