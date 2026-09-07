import copy
import inspect
import unittest

import src.drawing_engine.disciplines.mep.mep_attribute_binding as mep_attribute_binding
from src.drawing_engine.disciplines.mep.mep_attribute_binding import (
    build_mep_attribute_bindings,
    build_page_local_route_scopes,
    validate_mep_attribute_bindings,
)
from src.drawing_engine.disciplines.mep.mep_route_observations import build_mep_route_graph
from src.drawing_engine.disciplines.mep.mep_outlined_route_composites import build_mep_outlined_route_composites
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import build_mep_terminology_proposals


PAGE_REF = "page.synthetic"
DOCUMENT_KEY = "pdf-sha256:m4-synthetic"


def _scope():
    return {
        "record_type": "mep_sheet_page_record",
        "record_version": "0.1.0",
        "id": "m1.scope.synthetic",
        "page_ref": PAGE_REF,
        "page_number": 1,
        "role": "mechanical_piping_plan",
        "fields": {
            "sheet_number": {"value": "SYN-M4", "evidence_refs": ["title.sheet"]},
            "scale": {"drawing_inches_per_paper_inch": 96.0},
        },
    }


def _segment(index, start, end):
    source_ref = f"drawing[{index}].item[0].segment[0]"
    return {
        "id": source_ref,
        "drawing_ref": f"drawing[{index}]",
        "primitive_ref": f"drawing[{index}].item[0]",
        "item_index": 0,
        "part_index": 0,
        "kind": "line",
        "start_display": list(start),
        "end_display": list(end),
        "control_points_display": [],
        "sample_points_display": [],
        "style": {
            "stroke": [0.0, 0.0, 0.0],
            "fill": None,
            "width": 1.0,
            "dash": None,
        },
    }


def _route_graph(*segments):
    registry = {
        "schema_version": "0.1.0",
        "layer": "mep_sheet_coordinate_registry",
        "document": {"document_key": DOCUMENT_KEY, "page_count": 1},
        "pages": [_scope()],
        "packages": [],
    }
    return build_mep_route_graph(
        sheet_registry=registry,
        page_inputs={
            PAGE_REF: {
                "native_topology": {
                    "schema_version": "0.1.0",
                    "segments": list(segments),
                    "vertices": [],
                }
            }
        },
    )


def _observation(identifier, text="", **extra):
    return {
        "id": identifier,
        "page_ref": PAGE_REF,
        "text": text,
        "evidence_channels": extra.pop("evidence_channels", ["native_pdf_text"]),
        "interpretation_scope_ref": extra.pop(
            "interpretation_scope_ref", f"proposal.scope.{identifier}"
        ),
        **extra,
    }


def _m2(*observations):
    return build_mep_terminology_proposals(
        document={"document_key": DOCUMENT_KEY}, observations=observations
    )


def _proposal(payload, observation_ref, *, candidate_kind=None):
    rows = [
        row
        for row in payload["proposals"]
        if observation_ref in row["evidence_refs"]
        and (candidate_kind is None or row["candidate"]["kind"] == candidate_kind)
    ]
    if len(rows) != 1:
        raise AssertionError((observation_ref, candidate_kind, rows))
    return rows[0]


def _fragment(page, source_ref):
    return next(
        row for row in page["fragments"] if row["source_primitive_ref"] == source_ref
    )


def _endpoint(page, fragment_ref, role):
    return next(
        row
        for row in page["endpoints"]
        if row["fragment_ref"] == fragment_ref and row["role"] == role
    )


def _binding(identifier, proposal, target_kind, target_refs, **extra):
    refs = list(target_refs)
    return {
        "id": identifier,
        "proposal_ref": proposal["id"],
        "page_ref": PAGE_REF,
        "state": "observed",
        "method": {
            "name": extra.pop("method", "unique_native_leader_or_symbol_contact"),
            "version": "1.0.0",
        },
        "target_kind": target_kind,
        "target_refs": refs,
        "geometric_evidence_refs": extra.pop("geometric_evidence_refs", refs),
        **extra,
    }


class MepAttributeBindingTest(unittest.TestCase):
    def test_repeated_composite_attributes_share_scope_without_losing_evidence(self):
        graph = _route_graph(_segment(0, (0, 0), (100, 0)),
                             _segment(1, (0, 4), (100, 4)), _segment(2, (0, 0), (0, 4)))
        composites = build_mep_outlined_route_composites(route_graph=graph)
        target = composites['accepted_composites'][0]['id']
        for text, expected in [('2"', 2), ('4"', 0)]:
            with self.subTest(text=text):
                terminology = _m2(_observation('size.a', '2"', region_role='drawing_inline'),
                                  _observation('size.b', text, region_role='drawing_inline'))
                evidence = [_binding('bind.' + ref, _proposal(terminology, ref), 'route_composite', [target])
                            for ref in ('size.a', 'size.b')]
                result = build_mep_attribute_bindings(terminology_proposals=terminology, route_graph=graph,
                    outlined_route_composites=composites, binding_evidence=evidence)
                accepted = [row for row in result['relations'] if row['state'] == 'accepted']
                self.assertEqual(len(accepted), expected)
                scopes = [row for row in result['attribute_applicability_scopes'] if row.get('route_composite_ref') == target]
                self.assertEqual(len(scopes), 1 if expected else 0)
                if expected:
                    self.assertEqual(scopes[0]['change_relation_refs'], sorted(row['id'] for row in accepted))
                    self.assertTrue(all(row['applicability_scope_refs'] == [scopes[0]['id']] for row in accepted))
                else:
                    self.assertTrue(all('conflicting_bound_annotations' in row['reasons'] for row in result['relations']))

    def test_closed_outline_composite_is_one_semantic_binding_target(self):
        graph = _route_graph(
            _segment(0, (0, 0), (100, 0)),
            _segment(1, (0, 4), (100, 4)),
            _segment(2, (0, 0), (0, 4)),
        )
        composites = build_mep_outlined_route_composites(route_graph=graph)
        composite = composites["accepted_composites"][0]
        terminology = _m2(_observation("system.composite", "CHWS"))
        proposal = _proposal(terminology, "system.composite")
        payload = build_mep_attribute_bindings(
            terminology_proposals=terminology,
            route_graph=graph,
            outlined_route_composites=composites,
            binding_evidence=[{
                "id": "bind.composite",
                "proposal_ref": proposal["id"],
                "page_ref": PAGE_REF,
                "state": "observed",
                "method": {"name": "synthetic_composite_target", "version": "1.0.0"},
                "target_kind": "route_composite",
                "target_refs": [composite["id"]],
                "geometric_evidence_refs": [
                    composite["id"],
                    *composite["member_fragment_refs"],
                    *composite["supporting_closure_fragment_refs"],
                ],
            }],
        )

        relation = payload["relations"][0]
        self.assertEqual(relation["state"], "accepted")
        self.assertEqual(relation["target_kind"], "route_composite")
        self.assertEqual(len(payload["route_scopes"]), 1)
        self.assertEqual(
            payload["route_scopes"][0]["route_composite_ref"], composite["id"]
        )

    def test_page_local_route_scopes_stop_at_branch_and_ignore_crossing(self):
        graph = _route_graph(
            _segment(0, (-20, 0), (0, 0)),
            _segment(1, (0, 0), (20, 0)),
            _segment(2, (0, 0), (0, 20)),
            _segment(3, (-10, -10), (-10, 10)),
        )
        page = graph["pages"][0]
        scopes = build_page_local_route_scopes(page)

        self.assertEqual(len(page["branches"]), 1)
        self.assertEqual(len(page["crossings"]), 1)
        self.assertEqual(len(scopes), 4)
        self.assertTrue(all(len(scope["fragment_refs"]) == 1 for scope in scopes))
        self.assertTrue(all(scope["page_local_only"] for scope in scopes))

    def test_all_required_relation_classes_certify_independently(self):
        graph = _route_graph(
            _segment(0, (0, 0), (10, 0)),
            _segment(1, (10, 0), (20, 0)),
        )
        page = graph["pages"][0]
        first = _fragment(page, "drawing[0].item[0].segment[0]")
        second = _fragment(page, "drawing[1].item[0].segment[0]")
        first_start = _endpoint(page, first["id"], "start")
        second_end = _endpoint(page, second["id"], "end")
        shared_vertex = first["endpoint_vertex_refs"][1]
        observations = [
            _observation("system", "CHWS"),
            _observation("size", "DN 50"),
            _observation("elevation", "BOP EL +2.500 M"),
            _observation("equipment", "AHU-1"),
            _observation("valve", "BALL VALVE"),
            _observation("fitting", "REDUCER"),
            _observation(
                "damper",
                symbol_kind="fire_damper",
                evidence_channels=["native_vector_geometry"],
            ),
            _observation(
                "riser",
                symbol_kind="riser",
                evidence_channels=["native_vector_geometry"],
            ),
            _observation("zone", "SERVICE CLEARANCE"),
            _observation(
                "continuation",
                symbol_kind="continuation",
                evidence_channels=["native_vector_geometry"],
            ),
            _observation(
                "cap",
                symbol_kind="pipe_cap",
                evidence_channels=["native_vector_geometry"],
            ),
            _observation(
                "boundary",
                symbol_kind="scope_boundary",
                evidence_channels=["native_vector_geometry"],
            ),
            _observation(
                "detail-interface",
                symbol_kind="detail_section_interface",
                evidence_channels=["native_vector_geometry"],
            ),
        ]
        terminology = _m2(*observations)
        bindings = [
            _binding("bind.system", _proposal(terminology, "system"), "route_fragment", [first["id"]]),
            _binding("bind.size", _proposal(terminology, "size"), "route_fragment", [first["id"]]),
            _binding("bind.elevation", _proposal(terminology, "elevation"), "route_fragment", [first["id"]]),
            _binding(
                "bind.equipment",
                _proposal(terminology, "equipment"),
                "route_endpoint",
                [first_start["id"]],
                explicit_port_connectivity=True,
                port_geometry_evidence_refs=[first_start["id"]],
            ),
            _binding("bind.valve", _proposal(terminology, "valve"), "route_vertex", [shared_vertex]),
            _binding("bind.fitting", _proposal(terminology, "fitting"), "route_vertex", [shared_vertex]),
            _binding("bind.damper", _proposal(terminology, "damper"), "route_fragment", [second["id"]]),
            _binding("bind.riser", _proposal(terminology, "riser"), "route_endpoint", [second_end["id"]]),
            _binding("bind.zone", _proposal(terminology, "zone"), "route_fragment", [second["id"]]),
            _binding("bind.continuation", _proposal(terminology, "continuation"), "route_endpoint", [second_end["id"]]),
            _binding(
                "bind.cap", _proposal(terminology, "cap"), "route_endpoint",
                [second_end["id"]], projected_endpoint_index=1,
            ),
            _binding(
                "bind.boundary", _proposal(terminology, "boundary"), "route_endpoint",
                [second_end["id"]], projected_endpoint_index=1,
            ),
            _binding(
                "bind.detail-interface", _proposal(terminology, "detail-interface"),
                "route_endpoint", [second_end["id"]], projected_endpoint_index=1,
            ),
        ]

        payload = build_mep_attribute_bindings(
            terminology_proposals=terminology,
            route_graph=graph,
            binding_evidence=bindings,
        )

        self.assertEqual(validate_mep_attribute_bindings(payload), [])
        accepted = {row["relation_type"] for row in payload["relations"] if row["state"] == "accepted"}
        self.assertEqual(
            accepted,
            {
                "route_system",
                "route_size",
                "route_elevation",
                "equipment_endpoint",
                "valve",
                "fitting",
                "damper",
                "riser_drop",
                "service_zone",
                "continuation_endpoint",
                "physical_terminal",
                "package_boundary",
                "detail_section_interface",
            },
        )
        physical_terminals = [
            row for row in payload["relations"]
            if row["relation_type"] in {
                "physical_terminal", "package_boundary", "detail_section_interface"
            }
        ]
        self.assertTrue(all(row["candidate"]["endpoint_index"] == 1
                            for row in physical_terminals))
        self.assertTrue(all(row["candidate"]["terminal_kind"] == row["relation_type"]
                            for row in physical_terminals))
        for relation in payload["relations"]:
            self.assertFalse(relation["authority"]["physical_continuation_established"])
            self.assertFalse(relation["authority"]["installed_length_emitted"])
            self.assertFalse(relation["authority"]["fitting_count_emitted"])
            self.assertFalse(relation["authority"]["confirmed_clash_established"])
            self.assertFalse(relation["quantity_eligible"])

    def test_physical_terminal_without_explicit_projected_endpoint_index_abstains(self):
        graph = _route_graph(_segment(0, (0, 0), (20, 0)))
        page = graph["pages"][0]
        fragment = page["fragments"][0]
        endpoint = _endpoint(page, fragment["id"], "end")
        terminology = _m2(_observation(
            "cap", symbol_kind="pipe_cap",
            evidence_channels=["native_vector_geometry"],
        ))

        payload = build_mep_attribute_bindings(
            terminology_proposals=terminology,
            route_graph=graph,
            binding_evidence=[_binding(
                "bind.cap", _proposal(terminology, "cap"), "route_endpoint",
                [endpoint["id"]],
            )],
        )

        relation = payload["relations"][0]
        self.assertEqual(relation["relation_type"], "physical_terminal")
        self.assertEqual(relation["state"], "abstained")
        self.assertIn(
            "physical_endpoint_index_is_not_explicit_and_unique",
            relation["reasons"],
        )

    def test_physical_terminal_cannot_bind_to_internal_degree_two_endpoint(self):
        graph = _route_graph(
            _segment(0, (0, 0), (10, 0)),
            _segment(1, (10, 0), (20, 0)),
        )
        page = graph["pages"][0]
        first = _fragment(page, "drawing[0].item[0].segment[0]")
        internal_endpoint = _endpoint(page, first["id"], "end")
        terminology = _m2(_observation(
            "cap", symbol_kind="pipe_cap",
            evidence_channels=["native_vector_geometry"],
        ))

        payload = build_mep_attribute_bindings(
            terminology_proposals=terminology,
            route_graph=graph,
            binding_evidence=[_binding(
                "bind.cap", _proposal(terminology, "cap"), "route_endpoint",
                [internal_endpoint["id"]], projected_endpoint_index=1,
            )],
        )

        relation = payload["relations"][0]
        self.assertEqual(relation["state"], "abstained")
        self.assertIn(
            "physical_terminal_target_is_not_one_degree_one_endpoint",
            relation["reasons"],
        )

    def test_parallel_route_ambiguity_abstains(self):
        graph = _route_graph(
            _segment(0, (0, 0), (20, 0)),
            _segment(1, (0, 3), (20, 3)),
        )
        page = graph["pages"][0]
        left, right = page["fragments"]
        terminology = _m2(_observation("system", "CHWS"))
        proposal = _proposal(terminology, "system")
        payload = build_mep_attribute_bindings(
            terminology_proposals=terminology,
            route_graph=graph,
            binding_evidence=[
                _binding("candidate.a", proposal, "route_fragment", [left["id"]]),
                _binding("candidate.b", proposal, "route_fragment", [right["id"]]),
            ],
        )

        relation = payload["relations"][0]
        self.assertEqual(relation["state"], "abstained")
        self.assertIn("multiple_geometric_targets", relation["reasons"])

    def test_malformed_competing_evidence_cannot_be_ignored(self):
        graph = _route_graph(_segment(0, (0, 0), (20, 0)))
        fragment = graph["pages"][0]["fragments"][0]
        terminology = _m2(_observation("system", "CHWS"))
        proposal = _proposal(terminology, "system")
        malformed = _binding(
            "candidate.malformed",
            proposal,
            "route_fragment",
            [fragment["id"]],
        )
        malformed["page_ref"] = "page.other"
        payload = build_mep_attribute_bindings(
            terminology_proposals=terminology,
            route_graph=graph,
            binding_evidence=[
                _binding("candidate.valid", proposal, "route_fragment", [fragment["id"]]),
                malformed,
            ],
        )

        relation = payload["relations"][0]
        self.assertEqual(relation["state"], "abstained")
        self.assertIn("binding_evidence_page_mismatch", relation["reasons"])

    def test_replay_is_deterministic_when_binding_evidence_order_changes(self):
        graph = _route_graph(_segment(0, (0, 0), (20, 0)))
        fragment = graph["pages"][0]["fragments"][0]
        terminology = _m2(
            _observation("system", "CHWS"),
            _observation("size", "DN 50"),
        )
        evidence = [
            _binding("bind.system", _proposal(terminology, "system"), "route_fragment", [fragment["id"]]),
            _binding("bind.size", _proposal(terminology, "size"), "route_fragment", [fragment["id"]]),
        ]

        first = build_mep_attribute_bindings(
            terminology_proposals=terminology,
            route_graph=graph,
            binding_evidence=evidence,
        )
        replay = build_mep_attribute_bindings(
            terminology_proposals=terminology,
            route_graph=graph,
            binding_evidence=reversed(evidence),
        )

        self.assertEqual(first, replay)

    def test_unconnected_crossing_cannot_be_one_bound_scope(self):
        graph = _route_graph(
            _segment(0, (-10, 0), (10, 0)),
            _segment(1, (0, -10), (0, 10)),
        )
        page = graph["pages"][0]
        terminology = _m2(_observation("zone", "SERVICE CLEARANCE"))
        proposal = _proposal(terminology, "zone")
        payload = build_mep_attribute_bindings(
            terminology_proposals=terminology,
            route_graph=graph,
            binding_evidence=[
                _binding(
                    "crossing.coverage",
                    proposal,
                    "route_fragment_set",
                    [row["id"] for row in page["fragments"]],
                    explicit_branch_coverage=True,
                )
            ],
        )

        relation = payload["relations"][0]
        self.assertEqual(relation["state"], "abstained")
        self.assertIn("target_fragments_are_not_topologically_connected", relation["reasons"])

    def test_conflicting_bound_annotations_abstain(self):
        graph = _route_graph(_segment(0, (0, 0), (20, 0)))
        fragment = graph["pages"][0]["fragments"][0]
        terminology = _m2(
            _observation("size.50", "DN 50"),
            _observation("size.80", "DN 80"),
        )
        payload = build_mep_attribute_bindings(
            terminology_proposals=terminology,
            route_graph=graph,
            binding_evidence=[
                _binding("bind.50", _proposal(terminology, "size.50"), "route_fragment", [fragment["id"]]),
                _binding("bind.80", _proposal(terminology, "size.80"), "route_fragment", [fragment["id"]]),
            ],
        )

        self.assertTrue(all(row["state"] == "abstained" for row in payload["relations"]))
        self.assertTrue(all("conflicting_bound_annotations" in row["reasons"] for row in payload["relations"]))
        self.assertTrue(all(row["conflicts"] for row in payload["relations"]))

    def test_legend_and_color_only_proposals_cannot_bind(self):
        graph = _route_graph(_segment(0, (0, 0), (20, 0)))
        fragment = graph["pages"][0]["fragments"][0]
        terminology = _m2(
            _observation("legend", "CHWS", context="legend"),
            _observation("color", "CHWR", evidence_channels=["color"], color=[0, 0, 1]),
        )
        bindings = [
            _binding(
                f"bind.{index}",
                proposal,
                "route_fragment",
                [fragment["id"]],
            )
            for index, proposal in enumerate(terminology["proposals"])
        ]
        payload = build_mep_attribute_bindings(
            terminology_proposals=terminology,
            route_graph=graph,
            binding_evidence=bindings,
        )

        self.assertTrue(all(row["state"] == "abstained" for row in payload["relations"]))
        self.assertTrue(
            all("upstream_proposal_abstained_or_conflicted" in row["reasons"] for row in payload["relations"])
        )

    def test_nearby_equipment_without_explicit_port_connectivity_abstains(self):
        graph = _route_graph(_segment(0, (0, 0), (20, 0)))
        page = graph["pages"][0]
        endpoint = page["endpoints"][0]
        terminology = _m2(_observation("equipment", "AHU-2"))
        payload = build_mep_attribute_bindings(
            terminology_proposals=terminology,
            route_graph=graph,
            binding_evidence=[
                _binding(
                    "nearby.only",
                    _proposal(terminology, "equipment"),
                    "route_endpoint",
                    [endpoint["id"]],
                    method="nearest_endpoint_proximity",
                    explicit_port_connectivity=False,
                )
            ],
        )

        relation = payload["relations"][0]
        self.assertEqual(relation["state"], "abstained")
        self.assertIn("equipment_port_connectivity_is_not_explicit", relation["reasons"])

    def test_reviewed_negative_target_evidence_is_preserved_without_a_target(self):
        graph = _route_graph()
        terminology = _m2(
            _observation(
                "equipment.review",
                symbol_kind="equipment",
                evidence_channels=["reviewed_region_classification"],
            )
        )
        proposal = _proposal(terminology, "equipment.review")
        payload = build_mep_attribute_bindings(
            terminology_proposals=terminology,
            route_graph=graph,
            binding_evidence=[{
                "id": "reviewed.negative.target",
                "proposal_ref": proposal["id"],
                "page_ref": PAGE_REF,
                "state": "observed",
                "method": {
                    "name": "reviewed_discrete_target_coverage",
                    "version": "1.0.0",
                },
                "disposition": "abstain",
                "target_kind": None,
                "target_refs": [],
                "geometric_evidence_refs": ["equipment.review"],
                "negative_gate_reasons": [
                    "nearby_equipment_without_port_connectivity",
                    "no_unique_geometric_target",
                ],
            }],
        )

        relation = payload["relations"][0]
        self.assertEqual(relation["state"], "abstained")
        self.assertIsNone(relation["target_kind"])
        self.assertEqual(relation["target_refs"], [])
        self.assertEqual(
            relation["abstention_evidence_refs"], ["reviewed.negative.target"]
        )
        self.assertIn(
            "nearby_equipment_without_port_connectivity", relation["reasons"]
        )
        self.assertFalse(relation["authority"]["page_local_binding_established"])

        changed = copy.deepcopy(payload)
        changed["relations"][0]["abstention_evidence_refs"] = ["unknown"]
        self.assertTrue(validate_mep_attribute_bindings(changed))

    def test_branch_inheritance_requires_explicit_coverage(self):
        graph = _route_graph(
            _segment(0, (-10, 0), (0, 0)),
            _segment(1, (0, 0), (10, 0)),
            _segment(2, (0, 0), (0, 10)),
        )
        page = graph["pages"][0]
        fragments = page["fragments"]
        terminology = _m2(_observation("size", "DN 50"))
        proposal = _proposal(terminology, "size")
        payload = build_mep_attribute_bindings(
            terminology_proposals=terminology,
            route_graph=graph,
            binding_evidence=[
                _binding(
                    "implicit.branch.coverage",
                    proposal,
                    "route_fragment_set",
                    [fragments[0]["id"], fragments[1]["id"]],
                    explicit_branch_coverage=False,
                )
            ],
        )
        relation = payload["relations"][0]
        self.assertEqual(relation["state"], "abstained")
        self.assertIn("branch_coverage_is_not_explicit", relation["reasons"])

        accepted = build_mep_attribute_bindings(
            terminology_proposals=terminology,
            route_graph=graph,
            binding_evidence=[
                _binding(
                    "explicit.branch.coverage",
                    proposal,
                    "route_fragment_set",
                    [fragments[0]["id"], fragments[1]["id"]],
                    explicit_branch_coverage=True,
                )
            ],
        )["relations"][0]
        self.assertEqual(accepted["state"], "accepted")
        self.assertEqual(len(accepted["route_scope_refs"]), 2)

    def test_size_and_elevation_change_points_split_applicability(self):
        graph = _route_graph(
            _segment(0, (0, 0), (10, 0)),
            _segment(1, (10, 0), (20, 0)),
            _segment(2, (20, 0), (30, 0)),
        )
        page = graph["pages"][0]
        first = _fragment(page, "drawing[0].item[0].segment[0]")
        last = _fragment(page, "drawing[2].item[0].segment[0]")
        terminology = _m2(
            _observation("size", "DN 50"),
            _observation("elevation", "BOP EL +2.000 M"),
        )
        payload = build_mep_attribute_bindings(
            terminology_proposals=terminology,
            route_graph=graph,
            binding_evidence=[
                _binding("size.change", _proposal(terminology, "size"), "route_fragment", [first["id"]]),
                _binding("elevation.change", _proposal(terminology, "elevation"), "route_fragment", [last["id"]]),
            ],
        )

        size_relation = next(row for row in payload["relations"] if row["relation_type"] == "route_size")
        elevation_relation = next(row for row in payload["relations"] if row["relation_type"] == "route_elevation")
        self.assertEqual(size_relation["state"], "accepted")
        self.assertEqual(elevation_relation["state"], "accepted")
        self.assertEqual(len(size_relation["applicability_scope_refs"]), 1)
        self.assertEqual(len(elevation_relation["applicability_scope_refs"]), 1)
        size_scope = next(
            row for row in payload["attribute_applicability_scopes"]
            if row["id"] == size_relation["applicability_scope_refs"][0]
        )
        elevation_scope = next(
            row for row in payload["attribute_applicability_scopes"]
            if row["id"] == elevation_relation["applicability_scope_refs"][0]
        )
        self.assertEqual(size_scope["fragment_refs"], [first["id"]])
        self.assertEqual(elevation_scope["fragment_refs"], [last["id"]])

    def test_continuation_requires_one_terminal_endpoint_and_never_joins_sheets(self):
        graph = _route_graph(
            _segment(0, (0, 0), (10, 0)),
            _segment(1, (10, 0), (20, 0)),
        )
        page = graph["pages"][0]
        first = _fragment(page, "drawing[0].item[0].segment[0]")
        shared_endpoint = _endpoint(page, first["id"], "end")
        terminology = _m2(
            _observation(
                "continuation",
                symbol_kind="continuation",
                evidence_channels=["native_vector_geometry"],
            )
        )
        proposal = _proposal(terminology, "continuation")

        nonterminal = build_mep_attribute_bindings(
            terminology_proposals=terminology,
            route_graph=graph,
            binding_evidence=[
                _binding("nonterminal", proposal, "route_endpoint", [shared_endpoint["id"]])
            ],
        )["relations"][0]
        self.assertEqual(nonterminal["state"], "abstained")
        self.assertIn("continuation_target_is_not_one_terminal_endpoint", nonterminal["reasons"])

        endpoints = [row for row in page["endpoints"] if row["vertex_ref"] != shared_endpoint["vertex_ref"]]
        ambiguous = build_mep_attribute_bindings(
            terminology_proposals=terminology,
            route_graph=graph,
            binding_evidence=[
                _binding(f"terminal.{index}", proposal, "route_endpoint", [row["id"]])
                for index, row in enumerate(endpoints)
            ],
        )["relations"][0]
        self.assertEqual(ambiguous["state"], "abstained")
        self.assertIn("multiple_geometric_targets", ambiguous["reasons"])
        self.assertFalse(ambiguous["authority"]["physical_continuation_established"])

    def test_validator_recursively_prohibits_m5_m6_m7_outputs(self):
        graph = _route_graph(_segment(0, (0, 0), (20, 0)))
        fragment = graph["pages"][0]["fragments"][0]
        terminology = _m2(_observation("system", "CHWS"))
        payload = build_mep_attribute_bindings(
            terminology_proposals=terminology,
            route_graph=graph,
            binding_evidence=[
                _binding("bind.system", _proposal(terminology, "system"), "route_fragment", [fragment["id"]])
            ],
        )
        forbidden = (
            "installed_length_m",
            "fitting_count",
            "physical_continuation",
            "confirmed_clash",
            "clash_status",
            "quantity",
            "takeoff_line",
        )
        for key in forbidden:
            changed = copy.deepcopy(payload)
            changed["relations"][0]["nested"] = {key: 1}
            self.assertTrue(validate_mep_attribute_bindings(changed), key)
        changed = copy.deepcopy(payload)
        changed["relations"][0]["authority"]["physical_continuation_established"] = True
        self.assertTrue(validate_mep_attribute_bindings(changed))
        changed = copy.deepcopy(payload)
        changed["route_scopes"][0]["quantity_eligible"] = True
        self.assertTrue(validate_mep_attribute_bindings(changed))
        changed = copy.deepcopy(payload)
        changed["binding_evidence"][0]["method"]["version"] = "tampered"
        self.assertTrue(
            any(
                "binding_evidence_sha256" in error
                for error in validate_mep_attribute_bindings(changed)
            )
        )

    def test_document_mismatch_and_fixture_dispatch_fail_closed(self):
        graph = _route_graph(_segment(0, (0, 0), (20, 0)))
        terminology = _m2(_observation("system", "CHWS"))
        changed = copy.deepcopy(terminology)
        changed["document"]["document_key"] = "pdf-sha256:other"
        with self.assertRaisesRegex(ValueError, "document keys"):
            build_mep_attribute_bindings(
                terminology_proposals=changed,
                route_graph=graph,
                binding_evidence=[],
            )

        source = inspect.getsource(mep_attribute_binding)
        self.assertNotIn("M&P mark-up", source)
        self.assertNotIn("SYN-M4", source)
        self.assertNotIn("page_number ==", source)
        self.assertNotIn("schedule_comparison", source)


if __name__ == "__main__":
    unittest.main()
