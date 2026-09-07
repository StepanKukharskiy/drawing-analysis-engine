import copy
import inspect
import unittest
from unittest.mock import patch

from src.drawing_engine.pipelines.generate_object_agnostic_bundle import _page_summaries
from src.drawing_engine.core.canonical_knowledge_graph import (
    RELATION_TYPES,
    build_canonical_knowledge_graph,
    canonical_entity_id,
)


def supported_page(page_number=0):
    return {
        "page": page_number,
        "view_hypotheses": [
            {
                "id": "view.elevation",
                "role_hypothesis": "reinforcement_view_candidate",
                "bbox_display": [0, 0, 100, 300],
                "state": "inferred",
            },
            {
                "id": "view.section",
                "role_hypothesis": "section_view_candidate",
                "bbox_display": [120, 0, 200, 80],
                "parent_view_id": "view.elevation",
                "cutting_plane": {"label": "A-A", "primitive_refs": ["drawing[20]"]},
                "integration_certificate": {
                    "status": "passed",
                    "title_segmentation_closed": True,
                    "dimension_adjudication_closed": True,
                    "evidence_refs": ["title.segment", "dimension.adjudication"],
                },
                "state": "inferred",
            },
        ],
        "object_instance_graph": {
            "instances": [
                {
                    "id": "object.1",
                    "state": "inferred",
                    "primary_view_id": "view.elevation",
                    "section_view_id": "view.section",
                    "view_ids": ["view.elevation", "view.section"],
                    "projection_roles": {
                        "view.elevation": "primary_metric_projection",
                        "view.section": "compact_orthogonal_projection",
                    },
                    "basis": ["explicit object/view membership"],
                }
            ]
        },
        "claims": [
            {
                "id": "claim.dimension.1",
                "kind": "metric_dimension",
                "subject": "dimension.1",
                "value": 400,
                "unit": "mm",
                "state": "direct",
                "target_id": "component.1",
                "evidence_ref": "evidence.dimension.1",
            }
        ],
        "relations": [],
        "section_rebar_observations": {"sections": []},
        "rebar_program": {
            "physical_path_graph": {
                "fragments": [
                    {
                        "id": "fragment.1",
                        "primitive_ref": "drawing[10].item[0]",
                        "source_path_ref": "drawing[10]",
                    }
                ],
                "components": [
                    {
                        "id": "component.1",
                        "fragment_ids": ["fragment.1"],
                        "view_ids": ["view.elevation"],
                        "object_instance_ids": ["object.1"],
                        "topology": "single_fragment",
                        "projection_dimensionality": {"value": "axis_projection_1d", "state": "inferred"},
                        "state": "observed",
                    }
                ],
                "mark_hypotheses": [
                    {
                        "id": "mark.occurrence.1",
                        "token": "8",
                        "fragment_id": "fragment.1",
                        "state": "accepted",
                        "text_role_id": "text_role.8",
                        "leader_trace": {
                            "terminal_method": "arrowhead",
                            "segments": [{"drawing_ref": "drawing[12].item[0]"}],
                        },
                    }
                ],
                "cross_view_projection_identities": [],
            },
            "groups": [
                {
                    "id": "group.8",
                    "state": "derived",
                    "identity": {"mark": {"value": "8", "state": "derived"}},
                    "placement": {"path_component_ids": ["component.1"]},
                }
            ],
            "spacing_constraints": [
                {
                    "id": "spacing.1",
                    "spacing_mm": 200,
                    "interval_count": 4,
                    "extent_mm": 800,
                    "state": "direct",
                }
            ],
            "spacing_associations": [
                {
                    "id": "spacing.association.1",
                    "constraint_id": "spacing.1",
                    "group_id": "group.8",
                    "view_id": "view.elevation",
                    "status": "accepted",
                }
            ],
            "native_vector_detail_linking": {
                "details": [
                    {
                        "id": "detail.8",
                        "marks": ["8"],
                        "state": "observed",
                        "primitive_refs": ["drawing[30]"],
                        "geometry_paths": [{"kind": "line", "points": [[0, 0], [1, 0]]}],
                        "fabrication_dimensions": [
                            {
                                "id": "detail.dimension.1",
                                "value_mm": 400,
                                "kind": "linear",
                                "state": "direct",
                                "evidence_refs": ["drawing[31]"],
                            }
                        ],
                    }
                ],
                "placement_associations": [
                    {
                        "id": "placement.8.1",
                        "detail_id": "detail.8",
                        "marks": ["8"],
                        "target_view_id": "view.elevation",
                        "target_mark_bbox_display": [10, 10, 15, 15],
                        "state": "accepted",
                        "path_attachments": [
                            {"fragment_id": "fragment.1", "state": "accepted", "terminal_display": [10, 10]}
                        ],
                    }
                ],
                "physical_families": [
                    {
                        "id": "family.8",
                        "mark": "8",
                        "state": "derived",
                        "detail_ids": ["detail.8"],
                        "placement_association_ids": ["placement.8.1"],
                        "component_ids": ["component.1"],
                        "view_ids": ["view.elevation"],
                        "count": 5,
                        "count_state": "derived",
                    }
                ],
            },
        },
    }


def relation_types(graph):
    return {item["type"] for item in graph["relations"]}


def canonical_for(graph, page_key, entity_type, source_id):
    return next(
        item["canonical_id"]
        for item in graph["source_index"]
        if item["page_key"] == page_key
        and item["entity_type"] == entity_type
        and item["source_id"] == source_id
    )


class CanonicalKnowledgeGraphTest(unittest.TestCase):
    def test_composite_projected_bar_preserves_member_path_ids_without_physical_identity(self):
        page = supported_page()
        page["rebar_program"]["physical_path_graph"]["composite_projected_bars"] = {
            "proposals": [
                {
                    "id": "composite.1",
                    "mark": "8",
                    "view_id": "view.elevation",
                    "component_ids": ["component.1"],
                    "fragment_ids": ["fragment.1"],
                    "geometry_metrics": {"perpendicular_separation_points": 2.0},
                    "certificate": "test_certificate",
                    "state": "accepted",
                    "epistemic_state": "derived",
                }
            ]
        }
        graph = build_canonical_knowledge_graph(page)
        composite_id = canonical_for(
            graph, "page:0", "projected_bar_composite", "composite.1"
        )
        composite = next(item for item in graph["entities"] if item["id"] == composite_id)
        path_id = canonical_for(graph, "page:0", "projected_path", "component.1")
        self.assertEqual(composite["attributes"]["member_projected_path_ids"], [path_id])
        self.assertFalse(composite["attributes"]["physical_placement_identity_inferred"])
        self.assertFalse(composite["attributes"]["quantities_changed"])

    def test_supported_evidence_yields_all_typed_relations(self):
        graph = build_canonical_knowledge_graph(supported_page())

        self.assertEqual(relation_types(graph), set(RELATION_TYPES))
        self.assertTrue(graph["validation"]["relation_endpoints_exist"])
        self.assertFalse(graph["validation"]["implicit_equal_mark_merge_used"])
        self.assertGreaterEqual(graph["summary"]["entities_by_type"]["bar_family"], 2)

        dimension = canonical_for(graph, "page:0", "dimension", "dimension.1")
        path = canonical_for(graph, "page:0", "projected_path", "component.1")
        self.assertTrue(
            any(
                item["type"] == "dimension_of" and item["from"] == dimension and item["to"] == path
                for item in graph["relations"]
            )
        )

        mark = canonical_for(graph, "page:0", "mark_occurrence", "mark.occurrence.1")
        mark_node = next(item for item in graph["entities"] if item["id"] == mark)
        self.assertIn("drawing[12].item[0]", mark_node["provenance"]["evidence_refs"])

    def test_ambiguous_or_merely_equal_marks_fail_closed(self):
        page = supported_page()
        page["object_instance_graph"] = {"instances": []}
        page["view_hypotheses"][1].pop("parent_view_id")
        page["view_hypotheses"][1].pop("cutting_plane")
        page["claims"][0].pop("target_id")
        page["rebar_program"]["physical_path_graph"]["mark_hypotheses"][0]["state"] = "ambiguous"
        page["rebar_program"]["groups"][0]["placement"] = {"path_component_ids": []}
        page["rebar_program"]["spacing_associations"][0]["status"] = "review_candidate"
        details = page["rebar_program"]["native_vector_detail_linking"]
        details["details"][0]["fabrication_dimensions"] = []
        details["placement_associations"][0]["state"] = "review_candidate"
        details["physical_families"][0]["detail_ids"] = []
        details["physical_families"][0]["placement_association_ids"] = []
        details["physical_families"][0]["component_ids"] = []
        page["relations"] = [
            {"type": "dimension_located_in_view", "from": "dimension.1", "to": "view.elevation"}
        ]

        graph = build_canonical_knowledge_graph(page)

        self.assertEqual(graph["relations"], [])
        for relation_type in RELATION_TYPES:
            self.assertEqual(graph["summary"]["relations_by_type"][relation_type], 0)
        self.assertGreater(graph["summary"]["entity_count"], 0)

    def test_ids_are_page_qualified_and_stable_when_pages_are_reordered(self):
        first = supported_page(1)
        second = supported_page(2)
        original = copy.deepcopy([first, second])

        graph = build_canonical_knowledge_graph({"pages": [first, second]})
        reversed_graph = build_canonical_knowledge_graph([second, first])

        first_view = canonical_for(graph, "page:1", "view", "view.elevation")
        second_view = canonical_for(graph, "page:2", "view", "view.elevation")
        self.assertNotEqual(first_view, second_view)
        self.assertEqual(
            {item["id"] for item in graph["entities"]},
            {item["id"] for item in reversed_graph["entities"]},
        )
        self.assertEqual([first, second], original)
        self.assertEqual(
            first_view,
            canonical_entity_id(
                "page:1",
                "view",
                "view.elevation",
                document_key=graph["document_key"],
            ),
        )

    def test_ids_are_document_qualified(self):
        first = build_canonical_knowledge_graph(
            {"document_key": "document:first", "pages": [supported_page(1)]}
        )
        second = build_canonical_knowledge_graph(
            {"document_key": "document:second", "pages": [supported_page(1)]}
        )

        self.assertNotEqual(
            canonical_for(first, "page:1", "view", "view.elevation"),
            canonical_for(second, "page:1", "view", "view.elevation"),
        )
        self.assertNotEqual(
            {item["id"] for item in first["relations"]},
            {item["id"] for item in second["relations"]},
        )
        self.assertTrue(first["contract"]["canonical_ids_are_document_qualified"])

    def test_existing_supported_relation_is_preserved_with_reference_nodes(self):
        page = {"page": 4, "relations": [{"type": "cut_at", "from": "cut.1", "to": "view.1", "state": "direct"}]}
        graph = build_canonical_knowledge_graph(page)

        self.assertEqual(graph["summary"]["relations_by_type"]["cut_at"], 1)
        self.assertEqual(graph["summary"]["entities_by_type"]["source_reference"], 2)
        self.assertTrue(graph["validation"]["relation_endpoints_exist"])

    def test_optional_dimension_ownership_keeps_candidates_unresolved(self):
        page = supported_page()
        page["claims"][0].pop("target_id")
        page["dimension_ownership"] = {
            "attachments": [
                {
                    "id": "ownership.accepted",
                    "dimension_ref": "dimension.1",
                    "value_mm": 400,
                    "orientation": "horizontal",
                    "status": "accepted",
                    "epistemic_state": "direct",
                    "role": "bar_leg_length",
                    "owner_entity_refs": ["component.1"],
                    "view_refs": ["view.elevation"],
                    "measured_endpoints": [[0, 0], [100, 0]],
                    "primitive_refs": ["drawing[40]"],
                },
                {
                    "id": "ownership.candidate",
                    "dimension_ref": "dimension.1",
                    "value_mm": 400,
                    "status": "candidate",
                    "epistemic_state": "unknown",
                    "owner_entity_refs": ["component.2", "component.3"],
                    "reason": "two equally supported owners",
                    "primitive_refs": ["drawing[41]"],
                },
            ],
            "relations": [
                {
                    "id": "ownership.relation.1",
                    "type": "dimension_of",
                    "from": "dimension.1",
                    "to": "component.1",
                    "state": "direct",
                    "basis": "complete dimension chain terminates on the path",
                    "evidence_refs": ["drawing[40]"],
                }
            ],
        }

        graph = build_canonical_knowledge_graph(page)

        self.assertEqual(graph["summary"]["entities_by_type"]["dimension_attachment"], 2)
        self.assertEqual(graph["summary"]["relations_by_type"]["dimension_of"], 2)
        unresolved = [item for item in graph["unresolved"] if item.get("kind") == "dimension_ownership_attachment"]
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(unresolved[0]["source_id"], "ownership.candidate")
        self.assertIn("drawing[41]", unresolved[0]["provenance"]["evidence_refs"])

    def test_contour_owner_is_a_traversable_geometry_entity(self):
        page = supported_page()
        page["claims"][0].pop("target_id")
        page["contour_hypotheses"] = [
            {
                "id": "contour.1",
                "kind": "closed_loop",
                "bbox_display": [5, 5, 95, 295],
                "closed": True,
                "epistemic_state": "observed",
                "basis": "native vector topology",
            }
        ]
        page["calculation_contours"] = [
            {
                "id": "calculation_contour.1",
                "role": "calculation_profile",
                "state": "derived",
                "bbox_display": [0, 0, 100, 300],
                "segments_display": [
                    {
                        "start_display": [0, 0],
                        "end_display": [100, 0],
                        "primitive_ref": "drawing[60].item[0]",
                    }
                ],
                "polygon_points_display": [[0, 0], [100, 0], [100, 300], [0, 0]],
                "primitive_refs": ["drawing[60].item[0]"],
            }
        ]
        page["dimension_ownership"] = {
            "attachments": [
                {
                    "id": "ownership.contour",
                    "dimension_ref": "dimension.1",
                    "value_mm": 400,
                    "status": "accepted",
                    "epistemic_state": "derived",
                    "role": "contour_edge_span",
                    "owner_entity_refs": ["contour.1"],
                    "primitive_refs": ["drawing[40].item[0]"],
                }
            ],
            "relations": [
                {
                    "id": "ownership.contour.relation",
                    "type": "dimension_of",
                    "from": "dimension.1",
                    "to": "contour.1",
                    "state": "derived",
                    "evidence_refs": ["drawing[40].item[0]"],
                }
            ],
        }

        graph = build_canonical_knowledge_graph(page)
        contour_id = canonical_for(graph, "page:0", "contour", "contour.1")
        contour = next(item for item in graph["entities"] if item["id"] == contour_id)
        self.assertEqual(contour["attributes"]["bbox_display"], [5, 5, 95, 295])
        self.assertEqual(contour["attributes"]["geometry"]["representation"], "native_vector_reference")
        self.assertIn("drawing[40].item[0]", contour["provenance"]["evidence_refs"])
        self.assertTrue(
            any(item["type"] == "dimension_of" and item["to"] == contour_id for item in graph["relations"])
        )
        calculation_id = canonical_for(graph, "page:0", "contour", "calculation_contour.1")
        calculation = next(item for item in graph["entities"] if item["id"] == calculation_id)
        self.assertEqual(calculation["attributes"]["geometry"]["representation"], "explicit_geometry")

    def test_source_states_and_path_and_detail_provenance_are_preserved(self):
        page = supported_page()
        component = page["rebar_program"]["physical_path_graph"]["components"][0]
        component.pop("state")
        component["physical_path_state"] = "candidate_nonbranching_projection"
        family = page["rebar_program"]["native_vector_detail_linking"]["physical_families"][0]
        family["state"] = "derived_cross_view_family"
        detail_dimension = page["rebar_program"]["native_vector_detail_linking"]["details"][0][
            "fabrication_dimensions"
        ][0]
        detail_dimension["method"] = "native_dimension_chain"

        graph = build_canonical_knowledge_graph(page)
        path_id = canonical_for(graph, "page:0", "projected_path", "component.1")
        path = next(item for item in graph["entities"] if item["id"] == path_id)
        self.assertEqual(path["state"], "observed")
        self.assertIn("drawing[10].item[0]", path["provenance"]["evidence_refs"])
        self.assertEqual(path["attributes"]["source_fragments"][0]["id"], "fragment.1")

        family_id = canonical_for(graph, "page:0", "bar_family", "family.8")
        family_node = next(item for item in graph["entities"] if item["id"] == family_id)
        self.assertEqual(family_node["state"], "derived")

        dimension_id = canonical_for(graph, "page:0", "dimension", "detail.dimension.1")
        dimension = next(item for item in graph["entities"] if item["id"] == dimension_id)
        self.assertEqual(dimension["attributes"]["method"], "native_dimension_chain")
        self.assertIn("drawing[31]", dimension["provenance"]["evidence_refs"])
        ownership = next(
            item
            for item in graph["relations"]
            if item["type"] == "dimension_of" and item["from"] == dimension_id
        )
        self.assertEqual(ownership["state"], "derived")

    def test_sibling_view_frame_graph_is_attached_and_controls_cut_relations(self):
        page = supported_page()
        page["view_hypotheses"][1].pop("parent_view_id")
        page["view_hypotheses"][1].pop("cutting_plane")
        page["view_frame_graph"] = {
            "frames": [
                {
                    "id": "frame.elevation",
                    "view_id": "view.elevation",
                    "state": "partial",
                    "scale": {"state": "resolved", "value_points_per_mm": 0.25},
                    "origin": {"state": "resolved", "object_origin_display": [0, 300]},
                    "axes": {"object_axis_mapping": {"state": "unresolved"}},
                    "projection_direction": {"state": "unresolved"},
                    "evidence_refs": ["dimension.1"],
                },
                {
                    "id": "frame.section",
                    "view_id": "view.section",
                    "state": "partial",
                    "scale": {"state": "resolved", "value_points_per_mm": 0.25},
                    "parent_view": {"state": "resolved_by_cutting_plane", "view_id": "view.elevation"},
                    "evidence_refs": ["cut.trace.1"],
                },
            ],
            "relations": [
                {
                    "id": "cut.relation.1",
                    "type": "cut_at",
                    "section_label": "A-A",
                    "section_id": "section.a-a",
                    "section_view_id": "view.section",
                    "parent_view_id": "view.elevation",
                    "state": "accepted",
                    "integration_certificate": {
                        "status": "passed",
                        "title_segmentation_closed": True,
                        "dimension_adjudication_closed": True,
                        "evidence_refs": ["title.segment", "dimension.adjudication"],
                    },
                    "trace": {"primitive_refs": ["drawing[50]"]},
                    "confidence": 0.94,
                    "evidence_refs": ["drawing[50]"],
                }
            ],
            "relation_candidates": [
                {
                    "id": "cut.candidate.2",
                    "type": "cut_at",
                    "section_view_id": "view.section",
                    "parent_view_id": None,
                    "state": "candidate",
                    "reason": "second parent remains equally supported",
                    "evidence_refs": ["drawing[51]"],
                }
            ],
        }

        graph = build_canonical_knowledge_graph(page)

        section_id = canonical_for(graph, "page:0", "view", "view.section")
        section = next(item for item in graph["entities"] if item["id"] == section_id)
        self.assertEqual(section["attributes"]["frame"]["id"], "frame.section")
        self.assertIn("cut.trace.1", section["provenance"]["evidence_refs"])
        self.assertEqual(graph["summary"]["relations_by_type"]["cut_at"], 1)
        self.assertEqual(graph["summary"]["cut_candidate_count"], 1)
        self.assertEqual(graph["summary"]["unresolved_count"], 1)
        self.assertTrue(graph["validation"]["accepted_cut_relation_section_parent_pairs_are_unique"])
        candidate = next(item for item in graph["unresolved"] if item.get("source_id") == "cut.candidate.2")
        self.assertEqual(candidate["kind"], "cut_at_relation_candidate")
        self.assertIn("drawing[51]", candidate["provenance"]["evidence_refs"])

    def test_bundle_page_summary_exposes_canonical_uncertainty(self):
        canonical = {
            "entities": [{"provenance": {"page_key": "page:1"}}],
            "relations": [{"provenance": {"page_key": "page:1"}}],
            "unresolved": [
                {"page_key": "page:1", "kind": "cut_at_relation_candidate"},
                {"page_key": "page:1", "kind": "dimension_ownership_attachment"},
            ],
        }
        record = {
            "page": 1,
            "engineering_graph": {
                "view_frame_graph": {"summary": {"cutting_plane_candidate_count": 3}}
            },
        }
        with patch(
            "src.drawing_engine.pipelines.generate_object_agnostic_bundle.page_summary",
            return_value={"page": 1},
        ):
            summary = _page_summaries([record], canonical)[0]

        self.assertEqual(summary["canonical_entities"], 1)
        self.assertEqual(summary["canonical_relations"], 1)
        self.assertEqual(summary["canonical_unresolved"], 2)
        self.assertEqual(summary["canonical_cut_candidates"], 1)
        self.assertEqual(summary["cutting_plane_candidates"], 3)

    def test_module_contains_no_drawing_specific_dispatch(self):
        source = inspect.getsource(__import__("src.drawing_engine.core.canonical_knowledge_graph", fromlist=["*"])).lower()
        for forbidden in (".pdf", "k1", "k7", "column", "beam", "stair", "slab"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
