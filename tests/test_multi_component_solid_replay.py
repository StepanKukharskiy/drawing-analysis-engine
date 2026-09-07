import copy
import unittest

from src.drawing_engine.disciplines.concrete.generic_profile_extrusion_solver import _extruded_mesh
from src.drawing_engine.disciplines.concrete.multi_component_solid_replay import replay_multi_component_solid
from src.drawing_engine.project.review_feedback import canonical_graph_sha256
from src.drawing_engine.disciplines.concrete.solver_replay import build_solver_replay


GRAPH_HASH = "a" * 64


def _mesh(width, depth, height):
    mesh = _extruded_mesh([(0, 0), (width, 0), (width, height), (0, height)], depth)
    return {
        "vertices_xyz_mm": mesh["vertices_xyz_mm"],
        "triangles": mesh["triangles"],
    }


def _constraint(identifier, relation_type, source, target):
    return {
        "relation_id": identifier,
        "relation_type": relation_type,
        "from": source,
        "to": target,
        "page_keys": ["page:1"],
        "native_relation_ids": [f"native.{identifier}"],
        "native_evidence_refs": [f"evidence.{identifier}"],
        "decision_id": f"decision.{identifier}",
        "relation_attributes": {},
    }


def _native(canonical_id, entity_type, native_id):
    row = {
        "canonical_id": canonical_id,
        "entity_type": entity_type,
        "native_source_ids": [native_id],
        "evidence_refs": [f"evidence.{native_id}"],
    }
    if entity_type in {"view", "contour", "dimension", "object"}:
        row["native_entity_id"] = native_id
    return row


def fixture(graph_hash=GRAPH_HASH):
    coordinate_ids = ["relation.assembly.top", "relation.assembly.front", "relation.assembly.section"]
    constraints = [
        _constraint("relation.assembly.top", "projects_to", "canonical.assembly", "canonical.view.top"),
        _constraint("relation.assembly.front", "projects_to", "canonical.assembly", "canonical.view.front"),
        _constraint("relation.assembly.section", "section_of", "canonical.view.front", "canonical.assembly"),
        _constraint("relation.dimension.a", "dimension_of", "canonical.dimension.a", "canonical.profile.a"),
        _constraint("relation.dimension.b", "dimension_of", "canonical.dimension.b", "canonical.profile.b"),
    ]
    for component in "ab":
        for view in ("top", "front"):
            constraints.append(
                _constraint(
                    f"relation.projection.{component}.{view}",
                    "projects_to",
                    f"canonical.component.{component}",
                    f"canonical.view.{view}",
                )
            )
    native_entities = [
        _native("canonical.assembly", "object", "assembly.native"),
        _native("canonical.view.top", "view", "view.top"),
        _native("canonical.view.front", "view", "view.front"),
        _native("canonical.component.a", "solid_component", "component.a"),
        _native("canonical.component.b", "solid_component", "component.b"),
        _native("canonical.profile.a", "contour", "profile.a"),
        _native("canonical.profile.b", "contour", "profile.b"),
        _native("canonical.dimension.a", "dimension", "dimension.a"),
        _native("canonical.dimension.b", "dimension", "dimension.b"),
    ]
    components = []
    profiles = []
    transforms = []
    for component, width, origin in (("a", 100, 0), ("b", 60, 100)):
        components.append(
            {
                "id": f"component.{component}",
                "canonical_component_ref": f"canonical.component.{component}",
                "canonical_profile_ref": f"canonical.profile.{component}",
                "profile_ref": f"profile.{component}",
                "profile_relation_ref": f"relation.dimension.{component}",
                "transform_ref": f"transform.{component}",
                "mesh": _mesh(width, 50, 40),
                "analytic_volume": {
                    "value_mm3": width * 50 * 40,
                    "basis": "synthetic_profile_area_times_depth",
                    "evidence_refs": [f"analytic.{component}"],
                },
                "evidence_refs": [f"mesh.{component}"],
            }
        )
        profiles.append(
            {
                "id": f"profile.{component}",
                "record_type": "assembled_profile_candidate",
                "record_version": "0.1.0",
                "state": "resolved",
                "quantity_eligible": False,
                "closure": {
                    "closed": True,
                    "branch_free": True,
                    "unique_completion": True,
                    "scale_bounded": True,
                    "dimensionally_redundant": True,
                },
                "evidence_refs": [f"profile.evidence.{component}"],
            }
        )
        transforms.append(
            {
                "id": f"transform.{component}",
                "record_type": "physical_component_transform",
                "record_version": "0.1.0",
                "component_ref": f"component.{component}",
                "state": "resolved",
                "placement_role": "physical",
                "origin_xyz_mm": [origin, 0, 0],
                "local_axes_xyz": {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]},
                "axis_signs": "resolved",
                "certified_relation_ids": coordinate_ids,
                "evidence_refs": [f"transform.evidence.{component}"],
            }
        )
    interface = {
        "id": "interface.a_b",
        "state": "resolved",
        "component_refs": ["component.a", "component.b"],
        "polygon_xyz_mm": [[100, 0, 0], [100, 50, 0], [100, 50, 40], [100, 0, 40]],
        "certified_relation_ids": ["relation.projection.a.front", "relation.projection.b.front"],
        "evidence_refs": ["interface.evidence.a_b"],
    }
    views = [
        {
            "id": "view.top",
            "canonical_view_ref": "canonical.view.top",
            "origin_xyz_mm": [0, 0, 0],
            "u_axis_xyz": [1, 0, 0],
            "v_axis_xyz": [0, 1, 0],
            "tolerance_mm": 1e-6,
            "evidence_refs": ["view.evidence.top"],
            "component_projections": [
                {
                    "component_ref": "component.a",
                    "polygons_uv_mm": [[[0, 0], [100, 0], [100, 50], [0, 50]]],
                    "certified_relation_id": "relation.projection.a.top",
                    "evidence_refs": ["projection.top.a"],
                },
                {
                    "component_ref": "component.b",
                    "polygons_uv_mm": [[[100, 0], [160, 0], [160, 50], [100, 50]]],
                    "certified_relation_id": "relation.projection.b.top",
                    "evidence_refs": ["projection.top.b"],
                },
            ],
        },
        {
            "id": "view.front",
            "canonical_view_ref": "canonical.view.front",
            "origin_xyz_mm": [0, 0, 0],
            "u_axis_xyz": [1, 0, 0],
            "v_axis_xyz": [0, 0, 1],
            "tolerance_mm": 1e-6,
            "evidence_refs": ["view.evidence.front"],
            "component_projections": [
                {
                    "component_ref": "component.a",
                    "polygons_uv_mm": [[[0, 0], [100, 0], [100, 40], [0, 40]]],
                    "certified_relation_id": "relation.projection.a.front",
                    "evidence_refs": ["projection.front.a"],
                },
                {
                    "component_ref": "component.b",
                    "polygons_uv_mm": [[[100, 0], [160, 0], [160, 40], [100, 40]]],
                    "certified_relation_id": "relation.projection.b.front",
                    "evidence_refs": ["projection.front.b"],
                },
            ],
        },
    ]
    record = {
        "schema_version": "0.1.0",
        "id": "solid_hypothesis_replay.synthetic",
        "base_canonical_graph_sha256": graph_hash,
        "components": components,
        "profiles": profiles,
        "physical_component_transforms": transforms,
        "shared_interfaces": [interface],
        "supplied_views": views,
        "profile_reclosure": {
            "status": "reclosed_pass",
            "base_canonical_graph_sha256": graph_hash,
            "certified_constraint_ids": ["relation.dimension.a", "relation.dimension.b"],
            "profile_ids": ["profile.a", "profile.b"],
            "interface_ids": ["interface.a_b"],
            "gates": {
                "closure": {"status": "pass"},
                "uniqueness": {"status": "pass"},
                "evidence": {"status": "pass"},
            },
        },
        "evidence_refs": ["solid.synthetic"],
    }
    coordinate_reclosure = {
        "status": "reclosed_pass",
        "certified_constraint_ids": coordinate_ids,
        "physical_component_transform_ids": ["transform.a", "transform.b"],
        "projection_view_ids": ["view.top", "view.front"],
        "gates": {
            "metric": {"status": "pass"},
            "uniqueness": {"status": "pass"},
            "axis": {"status": "pass"},
            "reprojection": {"status": "pass"},
        },
    }
    pages = [{"page": 1, "solid_hypothesis_replay_records": [record], "quantities": []}]
    return constraints, native_entities, pages, coordinate_reclosure


def replay_fixture(graph_hash=GRAPH_HASH):
    constraints, native_entities, pages, coordinate = fixture(graph_hash)
    return replay_multi_component_solid(
        canonical_graph_sha256=graph_hash,
        constraints=constraints,
        native_entities=native_entities,
        engineering_pages=pages,
        shared_coordinate_reclosure=coordinate,
    )


def single_component_replay_fixture(graph_hash=GRAPH_HASH):
    constraints, native_entities, pages, coordinate = fixture(graph_hash)
    constraints[:] = [
        item
        for item in constraints
        if not any(token in item["relation_id"] for token in ("dimension.b", "projection.b"))
    ]
    native_entities[:] = [
        item
        for item in native_entities
        if item["canonical_id"]
        not in {"canonical.component.b", "canonical.profile.b", "canonical.dimension.b"}
    ]
    record = pages[0]["solid_hypothesis_replay_records"][0]
    record["components"] = record["components"][:1]
    record["profiles"] = record["profiles"][:1]
    record["physical_component_transforms"] = record["physical_component_transforms"][:1]
    record["shared_interfaces"] = []
    for view in record["supplied_views"]:
        view["component_projections"] = view["component_projections"][:1]
    profile_reclosure = record["profile_reclosure"]
    profile_reclosure["certified_constraint_ids"] = ["relation.dimension.a"]
    profile_reclosure["profile_ids"] = ["profile.a"]
    profile_reclosure["interface_ids"] = []
    profile_reclosure["no_external_interfaces_required"] = True
    coordinate["physical_component_transform_ids"] = ["transform.a"]
    return replay_multi_component_solid(
        canonical_graph_sha256=graph_hash,
        constraints=constraints,
        native_entities=native_entities,
        engineering_pages=pages,
        shared_coordinate_reclosure=coordinate,
    )


def _frame(view_id, bbox):
    return {
        "id": f"frame.{view_id}",
        "view_id": view_id,
        "bbox_display": bbox,
        "scale": {"state": "resolved", "value_points_per_mm": 0.25},
        "origin": {"state": "unresolved", "object_origin_display": None},
        "metric_spans": [
            {
                "id": f"dimension.{view_id}.vertical",
                "status": "accepted",
                "value_mm": 1000,
                "orientation": "vertical",
                "semantic_role": "overall_shared_axis_span",
                "primitive_refs": [f"drawing.{view_id}.vertical"],
            }
        ],
        "axes": {"object_axis_mapping": {"state": "unresolved"}},
        "projection_direction": {"state": "unresolved"},
        "unresolved_fields": ["axes.object_axis_mapping"],
    }


def _canonical_entity(identifier, entity_type, native_id, *, bbox=None):
    attributes = {}
    source_ids = [native_id]
    if entity_type == "view":
        attributes["frame"] = _frame(native_id, bbox)
        source_ids.insert(0, f"frame.{native_id}")
    return {
        "id": identifier,
        "entity_type": entity_type,
        "state": "derived",
        "attributes": attributes,
        "provenance": {
            "page_key": "page:1",
            "source_ids": source_ids,
            "evidence_refs": [f"evidence.{native_id}"],
        },
    }


def _canonical_relation(row):
    return {
        "id": row["relation_id"],
        "type": row["relation_type"],
        "from": row["from"],
        "to": row["to"],
        "state": "derived",
        "review_status": "accepted",
        "review_decision_id": f"decision.{row['relation_id']}",
        "attributes": {},
        "provenance": {
            "page_key": "page:1",
            "source_ids": row["native_relation_ids"],
            "evidence_refs": row["native_evidence_refs"],
        },
    }


def solver_fixture():
    constraints, _, pages, _ = fixture()
    entities = [
        _canonical_entity("canonical.assembly", "object", "assembly.native"),
        _canonical_entity("canonical.view.top", "view", "view.top", bbox=[0, 0, 200, 200]),
        _canonical_entity("canonical.view.front", "view", "view.front", bbox=[220, 0, 420, 200]),
        _canonical_entity("canonical.component.a", "solid_component", "component.a"),
        _canonical_entity("canonical.component.b", "solid_component", "component.b"),
        _canonical_entity("canonical.profile.a", "contour", "profile.a"),
        _canonical_entity("canonical.profile.b", "contour", "profile.b"),
        _canonical_entity("canonical.dimension.a", "dimension", "dimension.a"),
        _canonical_entity("canonical.dimension.b", "dimension", "dimension.b"),
    ]
    graph = {
        "schema_version": "0.1.0",
        "document_key": "pdf-sha256:multi-component-solid-replay",
        "entities": entities,
        "relations": [_canonical_relation(item) for item in constraints],
        "unresolved": [],
    }
    graph_hash = canonical_graph_sha256(graph)
    record = pages[0]["solid_hypothesis_replay_records"][0]
    record["base_canonical_graph_sha256"] = graph_hash
    record["profile_reclosure"]["base_canonical_graph_sha256"] = graph_hash
    pages[0].update(
        {
            "reinforcement_quantities": None,
            "estimated_reinforcement_quantities": None,
        }
    )
    overlay = {
        "layer": "reviewed_canonical_overlay",
        "base_canonical_graph_sha256": graph_hash,
        "effective_entities": copy.deepcopy(entities),
        "effective_relations": copy.deepcopy(graph["relations"]),
        "validation": {"status": "pass"},
    }
    return graph, overlay, pages


class MultiComponentSolidReplayTest(unittest.TestCase):
    def test_single_component_replays_with_explicit_no_interface_certificate(self):
        result = single_component_replay_fixture()

        self.assertEqual(result["status"], "reclosed_pass")
        kernel = result["kernel_result"]
        self.assertEqual(kernel["assembly_mesh"]["validation"]["component_count"], 1)
        self.assertEqual(kernel["shared_interfaces"], [])
        self.assertEqual(len(kernel["supplied_view_reprojections"]), 2)

    def test_complete_certified_record_replays_deterministically_into_kernel(self):
        first = replay_fixture()
        second = replay_fixture()

        self.assertEqual(first, second)
        self.assertEqual(first["status"], "reclosed_pass")
        self.assertIsNotNone(first["solid_preview"])
        self.assertEqual(first["solid_preview"]["mesh"]["validation"]["component_count"], 2)
        self.assertEqual(len(first["audit"]["validation_records"]["components"]), 2)
        self.assertEqual(len(first["audit"]["validation_records"]["interfaces"]), 1)
        self.assertEqual(len(first["audit"]["validation_records"]["overlaps"]), 1)
        self.assertEqual(len(first["audit"]["validation_records"]["reprojections"]), 2)
        self.assertEqual(first["audit"]["validation_records"]["volume"]["status"], "pass")
        self.assertTrue(first["contract"]["quantity_eligible_for_step5"])
        self.assertFalse(first["contract"]["quantity_writes_allowed"])

    def test_missing_input_classes_abstain_without_preview(self):
        for field, mutation in (
            ("profiles", lambda record: record.__setitem__("profiles", [])),
            ("physical transforms", lambda record: record.__setitem__("physical_component_transforms", [])),
            ("interfaces", lambda record: record.__setitem__("shared_interfaces", [])),
            ("two projections", lambda record: record.__setitem__("supplied_views", record["supplied_views"][:1])),
        ):
            with self.subTest(field=field):
                constraints, native_entities, pages, coordinate = fixture()
                mutation(pages[0]["solid_hypothesis_replay_records"][0])
                result = replay_multi_component_solid(
                    canonical_graph_sha256=GRAPH_HASH,
                    constraints=constraints,
                    native_entities=native_entities,
                    engineering_pages=pages,
                    shared_coordinate_reclosure=coordinate,
                )
                self.assertEqual(result["status"], "insufficient_constraints")
                self.assertIsNone(result["solid_preview"])
                self.assertIsNotNone(result["audit"]["abstention"])

    def test_stale_hash_presentation_transform_and_native_alias_conflict_fail(self):
        mutations = (
            lambda record, entities: record.__setitem__("base_canonical_graph_sha256", "stale"),
            lambda record, entities: record["physical_component_transforms"][0].__setitem__(
                "placement_role", "presentation_only"
            ),
            lambda record, entities: entities[3].__setitem__("native_source_ids", ["component.other"]),
        )
        for mutation in mutations:
            constraints, native_entities, pages, coordinate = fixture()
            mutation(pages[0]["solid_hypothesis_replay_records"][0], native_entities)
            result = replay_multi_component_solid(
                canonical_graph_sha256=GRAPH_HASH,
                constraints=constraints,
                native_entities=native_entities,
                engineering_pages=pages,
                shared_coordinate_reclosure=coordinate,
            )
            self.assertEqual(result["status"], "reclosed_fail")
            self.assertIsNone(result["solid_preview"])

    def test_unresolved_axis_missing_evidence_and_incomplete_projection_abstain(self):
        mutations = (
            lambda record: record["physical_component_transforms"][0].__setitem__("axis_signs", "unresolved"),
            lambda record: record["components"][0].__setitem__("evidence_refs", []),
            lambda record: record["supplied_views"][0]["component_projections"].pop(),
        )
        for mutation in mutations:
            constraints, native_entities, pages, coordinate = fixture()
            mutation(pages[0]["solid_hypothesis_replay_records"][0])
            result = replay_multi_component_solid(
                canonical_graph_sha256=GRAPH_HASH,
                constraints=constraints,
                native_entities=native_entities,
                engineering_pages=pages,
                shared_coordinate_reclosure=coordinate,
            )
            self.assertEqual(result["status"], "insufficient_constraints")
            self.assertIsNone(result["solid_preview"])
            self.assertFalse(result["contract"]["quantity_eligible_for_step5"])

    def test_solver_replay_call_site_abstains_until_step2_certifies_physical_transforms(self):
        graph, overlay, pages = solver_fixture()
        original_pages = copy.deepcopy(pages)

        replay = build_solver_replay(graph, overlay, pages)

        solid = replay["solver_reclosures"]["solid_hypothesis"]
        self.assertEqual(solid["status"], "insufficient_constraints")
        self.assertIn("physical component transforms", solid["reason"])
        self.assertIsNone(replay["solid_preview"])
        self.assertTrue(replay["quantity_replay"]["byte_identical"])
        self.assertEqual(pages, original_pages)


if __name__ == "__main__":
    unittest.main()
