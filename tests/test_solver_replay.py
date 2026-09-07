import copy
import unittest

from src.drawing_engine.project.review_feedback import canonical_graph_sha256
from src.drawing_engine.disciplines.concrete.solver_replay import build_solver_replay


def _frame(view_id, bbox, *, vertical=None, horizontal=None):
    spans = []
    for orientation, value in (("vertical", vertical), ("horizontal", horizontal)):
        if value is not None:
            spans.append(
                {
                    "id": f"dimension.{view_id}.{orientation}",
                    "status": "accepted",
                    "value_mm": value,
                    "orientation": orientation,
                    "semantic_role": "overall_shared_axis_span",
                    "primitive_refs": [f"drawing.{view_id}.{orientation}"],
                }
            )
    return {
        "id": f"frame.{view_id}",
        "view_id": view_id,
        "view_role_hypothesis": "drawing_view_candidate",
        "bbox_display": bbox,
        "scale": {"state": "resolved", "value_points_per_mm": 0.25},
        "origin": {"state": "unresolved", "object_origin_display": None},
        "metric_spans": spans,
        "axes": {"object_axis_mapping": {"state": "unresolved"}},
        "projection_direction": {"state": "unresolved"},
        "unresolved_fields": ["axes.object_axis_mapping"],
    }


def _view(canonical_id, native_id, bbox, *, vertical=None, horizontal=None):
    return {
        "id": canonical_id,
        "entity_type": "view",
        "state": "inferred",
        "status": None,
        "attributes": {
            "role": "drawing_view_candidate",
            "bbox_display": bbox,
            "frame": _frame(native_id, bbox, vertical=vertical, horizontal=horizontal),
        },
        "provenance": {
            "page_key": "page:1",
            "source_path": f"view_hypotheses[id={native_id}]",
            "source_ids": [f"frame.{native_id}", native_id],
            "evidence_refs": [f"drawing.{native_id}.outline"],
        },
    }


def _object():
    return {
        "id": "canonical.object.1",
        "entity_type": "object",
        "state": "inferred",
        "status": "inferred",
        "attributes": {
            "role": "physical_object_candidate",
            "view_ids": ["canonical.view.primary", "canonical.view.section"],
        },
        "provenance": {
            "page_key": "page:1",
            "source_path": "object_instance_graph.instances[id=object.native.1]",
            "source_ids": ["object.native.1"],
            "evidence_refs": ["drawing.object.1"],
        },
    }


def _relation(relation_id, relation_type, source, target, attributes, evidence, *, certified=True):
    row = {
        "id": relation_id,
        "type": relation_type,
        "from": source,
        "to": target,
        "state": "inferred",
        "attributes": attributes,
        "provenance": {
            "page_key": "page:1",
            "source_path": f"native_relations[id={relation_id}.native]",
            "source_ids": [f"{relation_id}.native"],
            "evidence_refs": evidence,
        },
    }
    if certified:
        row["review_status"] = "accepted"
        row["review_decision_id"] = f"decision.{relation_id}"
    return row


def _graph(*, section_vertical=1000, section_horizontal=None, object_constraints=True, cut=False):
    primary = _view(
        "canonical.view.primary",
        "view.native.primary",
        [0, 0, 200, 200],
        vertical=1000,
        horizontal=500,
    )
    section = _view(
        "canonical.view.section",
        "view.native.section",
        [220, 0, 300, 200],
        vertical=section_vertical,
        horizontal=section_horizontal,
    )
    entities = [primary, section]
    relations = []
    if object_constraints:
        entities.append(_object())
        relations.extend(
            [
                _relation(
                    "relation.projects.primary",
                    "projects_to",
                    "canonical.object.1",
                    "canonical.view.primary",
                    {"projection_role": "primary_metric_projection"},
                    ["drawing.primary.projection"],
                ),
                _relation(
                    "relation.projects.section",
                    "projects_to",
                    "canonical.object.1",
                    "canonical.view.section",
                    {"projection_role": "compact_orthogonal_projection"},
                    ["drawing.section.projection"],
                ),
                _relation(
                    "relation.section",
                    "section_of",
                    "canonical.view.section",
                    "canonical.object.1",
                    {"projection_role": "compact_orthogonal_projection"},
                    ["drawing.section.membership"],
                ),
            ]
        )
    if cut:
        relations.append(
            _relation(
                "relation.cut",
                "cut_at",
                "canonical.view.section",
                "canonical.view.primary",
                {
                    "trace": {
                        "orientation": "horizontal",
                        "bbox_display": [10, 80, 190, 80],
                        "primitive_refs": ["drawing.cut.native"],
                    }
                },
                ["drawing.cut.native"],
            )
        )
    return {
        "schema_version": "0.1.0",
        "document_key": "pdf-sha256:solver-replay-test",
        "entities": entities,
        "relations": relations,
        "unresolved": [],
    }


def _overlay(graph):
    return {
        "layer": "reviewed_canonical_overlay",
        "base_canonical_graph_sha256": canonical_graph_sha256(graph),
        "effective_entities": copy.deepcopy(graph["entities"]),
        "effective_relations": copy.deepcopy(graph["relations"]),
        "validation": {"status": "pass"},
    }


def _native_contour(contour_id, coordinates):
    segments = [
        {
            "id": f"{contour_id}.segment.{index}",
            "kind": "line",
            "start_display": [value, 0],
            "end_display": [value, 100],
            "primitive_ref": f"drawing.{contour_id}.{index}",
        }
        for index, value in enumerate(coordinates)
    ]
    return {
        "id": contour_id,
        "closed": True,
        "bbox_display": [min(coordinates), 0, max(coordinates), 100],
        "primitive_refs": [f"drawing.{contour_id}"],
        "segments_display": segments,
        "topology": {"segment_count": len(segments), "cycle_rank": 1},
    }


def _contour_case(*, duplicate_parent=False, omit_child=False, child_coordinates=None):
    graph = _graph(section_horizontal=500, cut=True)
    graph["entities"].extend(
        [
            {
                "id": "canonical.dimension.1",
                "entity_type": "dimension",
                "state": "derived",
                "status": "accepted",
                "attributes": {"value": 400.0, "unit": "mm"},
                "provenance": {
                    "page_key": "page:1",
                    "source_path": "claims[id=claim.1]",
                    "source_ids": ["claim.1", "dimension_attachment.1"],
                    "evidence_refs": ["drawing.dimension.1"],
                },
            },
            {
                "id": "canonical.contour.child",
                "entity_type": "contour",
                "state": "derived",
                "status": None,
                "attributes": {"closed": True},
                "provenance": {
                    "page_key": "page:1",
                    "source_path": "contour_hypotheses[id=contour.child]",
                    "source_ids": ["contour.child"],
                    "evidence_refs": ["drawing.contour.child"],
                },
            },
        ]
    )
    graph["relations"].append(
        _relation(
            "relation.dimension.child",
            "dimension_of",
            "canonical.dimension.1",
            "canonical.contour.child",
            {"source_payload": "dimension_ownership"},
            ["drawing.dimension.1", "drawing.contour.child"],
        )
    )
    parent_ids = ["contour.parent"]
    contours = [_native_contour("contour.parent", [0, 20, 100])]
    if duplicate_parent:
        parent_ids.append("contour.parent.duplicate")
        contours.append(_native_contour("contour.parent.duplicate", [0, 20, 100]))
    if not omit_child:
        contours.append(_native_contour("contour.child", child_coordinates or [220, 240, 320]))
    page = {
        "page": 1,
        "view_hypotheses": [
            {
                "id": "view.native.primary",
                "bbox_display": [0, 0, 200, 200],
                "contour_refs": parent_ids,
            },
            {
                "id": "view.native.section",
                "bbox_display": [220, 0, 420, 200],
                "contour_refs": [] if omit_child else ["contour.child"],
            },
        ],
        "view_frame_graph": {
            "frames": [
                _frame("view.native.primary", [0, 0, 200, 200], vertical=1000, horizontal=500),
                _frame("view.native.section", [220, 0, 420, 200], vertical=1000, horizontal=500),
            ],
            "relations": [
                {
                    "id": "relation.cut.native",
                    "type": "cut_at",
                    "state": "accepted",
                    "parent_view_id": "view.native.primary",
                    "section_view_id": "view.native.section",
                    "trace": {
                        "orientation": "horizontal",
                        "bbox_display": [10, 80, 190, 80],
                        "primitive_refs": ["drawing.cut.native"],
                    },
                }
            ],
        },
        "object_instance_graph": {
            "instances": [
                {
                    "id": "object.native.1",
                    "view_ids": ["view.native.primary", "view.native.section"],
                }
            ]
        },
        "contour_hypotheses": contours,
        "calculation_contours": [],
        "dimension_ownership": {
            "attachments": [
                {
                    "id": "dimension_ownership.1",
                    "dimension_ref": "dimension_attachment.1",
                    "value_mm": 400.0,
                    "orientation": "horizontal",
                    "status": "accepted",
                    "semantic_role": "overall_shared_axis_span",
                    "owner_entity_refs": ["contour.child", "unrelated.contour"],
                    "primitive_refs": ["drawing.dimension.1"],
                }
            ],
            "relations": [
                {
                    "id": "relation.dimension.child.native",
                    "type": "dimension_of",
                    "from": "dimension_attachment.1",
                    "to": "contour.child",
                    "state": "derived",
                }
            ],
        },
        "native_segments": [],
        "quantities": [{"net_concrete_m3": 1.25}],
        "reinforcement_quantities": None,
        "estimated_reinforcement_quantities": None,
    }
    return graph, [page]


class SolverReplayTest(unittest.TestCase):
    def test_projection_constraints_reclose_and_keep_quantities_byte_identical(self):
        graph = _graph()
        quantities = [
            {
                "page": 1,
                "quantities": [{"net_concrete_m3": 1.25}],
                "reinforcement_quantities": {"mass_kg": 20.5},
                "estimated_reinforcement_quantities": None,
            }
        ]
        original_graph = copy.deepcopy(graph)
        original_quantities = copy.deepcopy(quantities)

        replay = build_solver_replay(graph, _overlay(graph), quantities)

        coordinate_input = replay["solver_inputs"]["shared_coordinate_system"]
        self.assertEqual(
            {item["relation_type"] for item in coordinate_input["constraints"]},
            {"projects_to", "section_of"},
        )
        native = {item["canonical_id"]: item for item in coordinate_input["native_entities"]}
        self.assertEqual(native["canonical.view.primary"]["native_entity_id"], "view.native.primary")
        self.assertEqual(native["canonical.object.1"]["native_entity_id"], "object.native.1")
        primary_constraint = next(
            item for item in coordinate_input["constraints"] if item["relation_id"] == "relation.projects.primary"
        )
        self.assertIn("drawing.primary.projection", primary_constraint["native_evidence_refs"])
        self.assertEqual(coordinate_input["reclosure"]["status"], "reclosed_pass")
        self.assertEqual(coordinate_input["reclosure"]["gates"]["metric"]["status"], "pass")
        self.assertEqual(coordinate_input["reclosure"]["gates"]["uniqueness"]["status"], "pass")
        self.assertEqual(coordinate_input["reclosure"]["gates"]["axis"]["status"], "pass")
        self.assertEqual(coordinate_input["reclosure"]["gates"]["reprojection"]["status"], "pass")
        self.assertTrue(replay["quantity_replay"]["byte_identical"])
        self.assertEqual(replay["quantity_replay"]["before"], replay["quantity_replay"]["after"])
        self.assertEqual(graph, original_graph)
        self.assertEqual(quantities, original_quantities)

    def test_conflicting_metric_reprojection_fails_reclosure(self):
        graph = _graph(section_vertical=1300)
        replay = build_solver_replay(graph, _overlay(graph), [{"page": 1, "quantities": []}])
        reclosure = replay["solver_inputs"]["shared_coordinate_system"]["reclosure"]
        self.assertEqual(reclosure["status"], "reclosed_fail")
        self.assertEqual(reclosure["gates"]["reprojection"]["status"], "fail")
        self.assertFalse(replay["quantity_replay"]["changed"])

    def test_missing_metric_evidence_reports_insufficient_constraints(self):
        graph = _graph(section_vertical=None)
        replay = build_solver_replay(graph, _overlay(graph), [{"page": 1, "quantities": []}])
        reclosure = replay["solver_inputs"]["shared_coordinate_system"]["reclosure"]
        self.assertEqual(reclosure["status"], "insufficient_constraints")
        self.assertEqual(reclosure["gates"]["metric"]["status"], "insufficient")

    def test_non_unique_certified_primary_projection_fails_reclosure(self):
        graph = _graph()
        graph["entities"].append(
            _view(
                "canonical.view.other",
                "view.native.other",
                [0, 220, 200, 420],
                vertical=1000,
            )
        )
        graph["relations"].append(
            _relation(
                "relation.projects.other",
                "projects_to",
                "canonical.object.1",
                "canonical.view.other",
                {"projection_role": "primary_metric_projection"},
                ["drawing.other.projection"],
            )
        )
        replay = build_solver_replay(graph, _overlay(graph), [{"page": 1, "quantities": []}])
        reclosure = replay["solver_inputs"]["shared_coordinate_system"]["reclosure"]
        self.assertEqual(reclosure["status"], "reclosed_fail")
        self.assertEqual(reclosure["gates"]["uniqueness"]["status"], "fail")

    def test_certified_cut_translates_direction_trace_and_native_evidence(self):
        graph = _graph(
            object_constraints=False,
            cut=True,
            section_vertical=None,
            section_horizontal=500,
        )
        graph["relations"].append(
            _relation(
                "relation.cut.uncertified",
                "cut_at",
                "canonical.view.section",
                "canonical.view.primary",
                {"trace": {"orientation": "vertical", "bbox_display": [20, 0, 20, 100]}},
                ["drawing.cut.uncertified"],
                certified=False,
            )
        )
        replay = build_solver_replay(graph, _overlay(graph), [{"page": 1, "quantities": []}])
        coordinate_input = replay["solver_inputs"]["shared_coordinate_system"]
        self.assertEqual([item["relation_id"] for item in coordinate_input["constraints"]], ["relation.cut"])
        translated = coordinate_input["reclosure"]["coordinate_system"]["constraints"][0]
        self.assertEqual(translated["parent_view_id"], "view.native.primary")
        self.assertEqual(translated["child_view_id"], "view.native.section")
        self.assertIn("drawing.cut.native", translated["evidence_refs"])
        self.assertEqual(coordinate_input["reclosure"]["status"], "reclosed_pass")

    def test_contour_correspondence_recloses_from_exact_native_page_records(self):
        graph, pages = _contour_case()
        original_pages = copy.deepcopy(pages)
        replay = build_solver_replay(graph, _overlay(graph), pages)
        contour_input = replay["solver_inputs"]["contour_correspondence"]
        self.assertEqual(
            {item["relation_type"] for item in contour_input["constraints"]},
            {"cut_at", "dimension_of", "section_of"},
        )
        reclosure = contour_input["reclosure"]
        self.assertEqual(reclosure["status"], "reclosed_pass")
        page = reclosure["pages"][0]
        self.assertEqual(page["gates"]["native_path"]["status"], "pass")
        self.assertEqual(page["gates"]["uniqueness"]["status"], "pass")
        record = page["contour_correspondence"]["records"][0]
        self.assertEqual(record["state"], "accepted")
        self.assertEqual(record["selected"]["path_signature_residual_mm"], 0)
        self.assertEqual(record["selected"]["signed_transform"]["state"], "resolved")
        self.assertEqual(record["selected"]["owned_dimension_refs"], ["dimension_ownership.1"])
        self.assertTrue(replay["quantity_replay"]["byte_identical"])
        self.assertEqual(pages, original_pages)

    def test_multiple_native_contour_matches_fail_reclosure_uniqueness(self):
        graph, pages = _contour_case(duplicate_parent=True)
        replay = build_solver_replay(graph, _overlay(graph), pages)
        page = replay["solver_inputs"]["contour_correspondence"]["reclosure"]["pages"][0]
        self.assertEqual(page["status"], "reclosed_fail")
        self.assertEqual(page["gates"]["uniqueness"]["status"], "fail")
        self.assertGreater(page["gates"]["uniqueness"]["ambiguous_record_count"], 0)

    def test_conflicting_native_path_reprojection_fails_reclosure(self):
        graph, pages = _contour_case(child_coordinates=[220, 270, 420])
        replay = build_solver_replay(graph, _overlay(graph), pages)
        page = replay["solver_inputs"]["contour_correspondence"]["reclosure"]["pages"][0]
        self.assertEqual(page["status"], "reclosed_fail")
        self.assertEqual(page["gates"]["reprojection"]["status"], "fail")

    def test_missing_native_contour_path_is_insufficient(self):
        graph, pages = _contour_case(omit_child=True)
        replay = build_solver_replay(graph, _overlay(graph), pages)
        reclosure = replay["solver_inputs"]["contour_correspondence"]["reclosure"]
        self.assertEqual(reclosure["status"], "insufficient_constraints")
        self.assertEqual(reclosure["pages"][0]["gates"]["native_path"]["status"], "insufficient")

    def test_replay_contract_is_reusable_by_remaining_solvers(self):
        replay = build_solver_replay(_graph(), _overlay(_graph()), [{"page": 1, "quantities": []}])
        contract = replay["contract"]["reusable_solver_replay_contract"]
        self.assertTrue(contract["certified_relation_slice"])
        self.assertTrue(contract["canonical_to_native_translation"])
        self.assertEqual(
            contract["next_consumers"],
            ["physical_bar_family"],
        )
        self.assertEqual(contract["completed_consumers"], ["solid_hypothesis"])


if __name__ == "__main__":
    unittest.main()
