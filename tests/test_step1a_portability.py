import json
import unittest
from collections import Counter
from pathlib import Path

import fitz

from src.drawing_engine.core.object_agnostic_understanding import understand_page

ROOT = Path(__file__).resolve().parents[1]
ENGINEERING_GRAPH = ROOT / "output" / "object_agnostic" / "candidate-08-staircase-page.engineering-graph.json"


def load_frozen_engineering() -> dict:
    return json.loads(ENGINEERING_GRAPH.read_text(encoding="utf-8"))["pages"][0]


def candidate_08_signature(engineering: dict) -> dict:
    adjudication = engineering["dimension_adjudication"]
    attachments = {
        item["dimension_ref"]: item
        for item in engineering["dimension_ownership"]["attachments"]
    }
    accepted_values = sorted(
        attachments[item["dimension_ref"]]["value_mm"]
        for item in adjudication["records"]
        if item["status"] == "accepted"
    )
    frame_summary = engineering["view_frame_graph"]["summary"]
    return {
        "dimension_summary": adjudication["summary"],
        "accepted_values_mm": accepted_values,
        "frame_summary": {
            key: frame_summary[key]
            for key in (
                "cutting_plane_relation_count",
                "resolved_relative_physical_scope_count",
                "resolved_relative_scope_count",
                "reprojection_pass_count",
                "reprojection_fail_count",
            )
        },
        "specialised_solver_status": engineering["specialised_solver"]["status"],
        "quantity_count": len(engineering["quantities"]),
    }


class Step1APortabilityTest(unittest.TestCase):
    def _assert_candidate_08_closes_only_redundant_terminal_less_dimensions(self, engineering):
        record = {"engineering_graph": engineering}
        report = engineering["dimension_adjudication"]
        attachments = {
            item["dimension_ref"]: item
            for item in engineering["dimension_ownership"]["attachments"]
        }
        accepted = [
            attachments[item["dimension_ref"]]
            for item in report["records"]
            if item["status"] == "accepted"
        ]

        self.assertEqual(report["summary"]["proposal_count"], 111)
        self.assertEqual(report["summary"]["accepted_count"], 42)
        self.assertEqual(
            Counter(item["value_mm"] for item in accepted),
            Counter(
                {
                    150.0: 8,
                    3150.0: 8,
                    200.0: 5,
                    2999.0: 5,
                    3000.0: 3,
                    4100.0: 2,
                    300.0: 2,
                    975.0: 2,
                    2550.0: 2,
                    2925.0: 1,
                    1175.0: 1,
                    2350.0: 1,
                    1150.0: 1,
                    2150.0: 1,
                }
            ),
        )
        accepted_records = [item for item in report["records"] if item["status"] == "accepted"]
        self.assertTrue(
            all(
                item["certificate"]["terminal_style"] == "terminal_less"
                and item["certificate"]["terminal_less_redundancy"] is not None
                and item["certificate"]["exact_endpoint_geometry_resolved"]
                and item["certificate"]["unique_preliminary_view_owner"]
                and item["certificate"]["view_local_scale"]["passed"]
                for item in accepted_records
            )
        )
        self.assertTrue(
            any(item["certificate"]["different_cross_axis_targets"] for item in accepted_records)
        )
        # The final quantity is permitted only because the downstream
        # relation, constructive-union, equivalence, and scope gates close.
        self.assertEqual(
            record["engineering_graph"]["specialised_solver"]["status"],
            "resolved",
        )
        self.assertEqual(
            record["engineering_graph"]["specialised_solver"]["selected"],
            "same_object_invariant_union",
        )
        frame_graph = record["engineering_graph"]["view_frame_graph"]
        summary = frame_graph["summary"]
        self.assertEqual(summary["cutting_plane_relation_count"], 0)
        self.assertEqual(summary["resolved_relative_physical_scope_count"], 0)
        self.assertEqual(summary["resolved_relative_scope_count"], 1)
        self.assertEqual(summary["reprojection_pass_count"], 1)
        self.assertEqual(summary["reprojection_fail_count"], 0)
        relation = next(
            item
            for item in frame_graph["relation_candidates"]
            if item["section_view_id"] == "title_view_segment.002"
            and item["parent_view_id"] == "view_hypothesis.021"
        )
        self.assertEqual(relation["reason"], "profile/section reprojection has not closed uniquely")
        metric_pair = relation["integration_certificate"]["relation_scoped_metric_pair"]
        self.assertEqual(metric_pair["status"], "passed")
        self.assertEqual(metric_pair["candidate_pair_count"], 1)
        self.assertEqual(metric_pair["transverse_extent_mm"], 4100.0)
        self.assertEqual(
            {item["global_dimension_adjudication_status"] for item in metric_pair["spans"]},
            {"accepted", "ambiguous"},
        )
        self.assertEqual(frame_graph["object_scopes"], [])
        self.assertEqual(len(frame_graph["object_scope_candidates"]), 1)
        self.assertEqual(
            frame_graph["object_scope_candidates"][0]["reason"],
            "one section-to-parent relation is insufficient to create a new physical-object scope",
        )
        self.assertEqual(len(record["engineering_graph"]["solid_hypotheses"]), 1)
        self.assertEqual(
            record["engineering_graph"]["solid_hypotheses"][0]["record_type"],
            "physical_object",
        )
        self.assertIsNone(record["engineering_graph"].get("concrete_quantity"))
        self.assertEqual(len(record["engineering_graph"]["quantities"]), 1)
        self.assertAlmostEqual(
            record["engineering_graph"]["quantities"][0]["net_concrete_m3"],
            1.7726147823710516,
        )
        profile_assembly = record["engineering_graph"]["scoped_profile_assembly"]
        resolved_scope_refs = {
            item["id"]
            for item in record["engineering_graph"]["title_anchored_view_segmentation"]["segments"]
            if item["state"] == "resolved"
        }
        self.assertEqual(
            {item["scope_ref"] for item in profile_assembly["scopes"]},
            resolved_scope_refs,
        )
        self.assertTrue(
            all(item["profiles"] or item["abstentions"] for item in profile_assembly["scopes"])
        )
        self.assertTrue(
            all(
                profile["scope_ref"] in resolved_scope_refs
                and profile["page"] == 1
                and profile["quantity_eligible"] is False
                for profile in profile_assembly["profiles"]
            )
        )
        excluded = {
            ref
            for scope in profile_assembly["scopes"]
            for ref in scope["excluded_primitive_refs"]
        }
        self.assertFalse(
            excluded
            & {
                ref
                for profile in profile_assembly["profiles"]
                for ref in profile["drawing_refs"]
            }
        )
        self.assertTrue(
            profile_assembly["contract"]["excluded_primitives_never_enter_bridge_search"]
        )
        self.assertFalse(profile_assembly["contract"]["physical_object_identity_established"])
        self.assertFalse(profile_assembly["contract"]["quantity_eligible"])
        local_dimensions = record["engineering_graph"][
            "title_scope_local_dimension_reclosure"
        ]
        self.assertEqual(local_dimensions["summary"]["target_scope_count"], 12)
        self.assertEqual(local_dimensions["summary"]["accepted_scope_local_dimension_count"], 24)
        self.assertEqual(local_dimensions["summary"]["projected_station_transfer_count"], 3)
        self.assertEqual(local_dimensions["summary"]["independent_metric_check_count"], 8)
        section_local = local_dimensions["scope_results"][0]
        self.assertEqual(section_local["scope_ref"], "title_view_segment.002")
        self.assertEqual(
            sorted(
                (
                    item["total_value_mm"],
                    item["term_values_mm"],
                )
                for item in section_local["independent_arithmetic_chain_certificates"]
            ),
            [(3150.0, [150.0, 1425.0, 1575.0]), (4100.0, [2925.0, 1175.0])],
        )
        self.assertEqual(
            [
                item["value_mm"]
                for item in section_local["repeated_projected_measurement_certificates"]
            ],
            [275.0],
        )
        self.assertEqual(section_local["observed_value_multiplicity"]["158.0"], 2)
        self.assertEqual(section_local["observed_value_multiplicity"]["175.0"], 1)
        self.assertEqual(
            sorted(item["value_mm"] for item in section_local["unresolved_metric_observations"]),
            [158.0, 158.0, 175.0],
        )
        self.assertTrue(section_local["global_dimension_states_preserved"])
        self.assertFalse(section_local["quantity_eligible"])
        local_profiles = record["engineering_graph"]["scope_local_profile_reclosure"]
        self.assertEqual(local_profiles["status"], "pair_unresolved")
        self.assertEqual(local_profiles["summary"]["profile_candidate_count"], 310)
        self.assertEqual(
            local_profiles["scope_results"][0]["profile_vertex_count_histogram"],
            {"3": 82, "4": 14, "7": 21, "8": 11},
        )
        self.assertIsNone(local_profiles["scope_results"][0]["pair_certificate"])
        self.assertFalse(local_profiles["contract"]["profile_selector_used"])
        open_boundaries = record["engineering_graph"]["open_structural_boundary_assembly"]
        self.assertEqual(open_boundaries["status"], "incomplete_proposals")
        self.assertEqual(
            open_boundaries["scope_results"][0]["transition"],
            "96 native structural supports -> 96 eligible edges -> 33 branch-free chains -> "
            "2 interface-bounded flight proposals -> 0 closed profiles",
        )
        self.assertEqual(
            open_boundaries["summary"]["boundary_role_counts"],
            {
                "internal_annotation_line": 5,
                "stepped_upper_surface": 2,
                "support_or_landing_boundary": 24,
                "waist_underside": 2,
            },
        )
        self.assertEqual(
            sorted(
                item["closure"]["explicit_open_interface_count"]
                for item in open_boundaries["scope_results"][0]["flight_boundary_proposals"]
            ),
            [2, 4],
        )
        self.assertEqual(
            open_boundaries["scope_results"][0]["reprojection_edges_missing_from_eligible"],
            [],
        )
        self.assertEqual(
            open_boundaries["scope_results"][0]["dual_role_structural_override_count"],
            3,
        )
        self.assertFalse(open_boundaries["contract"]["closure_bridges_invented"])
        interface_closure = record["engineering_graph"]["flight_interface_closure"]
        self.assertEqual(interface_closure["status"], "ambiguous_assignments")
        self.assertEqual(
            interface_closure["scope_results"][0]["transition"],
            "2 open flight proposals -> 5 valid port assignments -> 0 closed profiles",
        )
        self.assertEqual(interface_closure["closed_profiles"], [])
        self.assertEqual(interface_closure["certificates"], [])
        self.assertEqual(
            interface_closure["summary"][
                "temporary_closed_boundary_hypothesis_count"
            ],
            10,
        )
        self.assertEqual(
            interface_closure["scope_results"][0]["reason_code"],
            "multiple_valid_port_assignments",
        )
        self.assertTrue(
            interface_closure["contract"][
                "every_matched_port_pair_requires_one_common_certified_station"
            ]
        )
        self.assertFalse(interface_closure["contract"]["landing_geometry_constructed"])
        pair_gate = record["engineering_graph"]["flight_profile_pair_certification"]
        self.assertEqual(pair_gate["status"], "insufficient_constraints")
        self.assertEqual(
            pair_gate["scope_results"][0]["transition"],
            "128 raw candidates -> 128 canonical contours -> 0 admissible pairs -> no admissible pair",
        )
        self.assertEqual(pair_gate["summary"]["collapsed_representation_count"], 0)
        self.assertEqual(pair_gate["summary"]["flight_role_eligible_contour_count"], 0)
        self.assertEqual(pair_gate["summary"]["generated_pair_count"], 0)
        self.assertEqual(pair_gate["summary"]["admissible_pair_count"], 0)
        self.assertEqual(
            pair_gate["scope_results"][0]["structural_boundary_transition"],
            "96 native structural supports -> 96 eligible edges -> 33 branch-free chains -> "
            "2 interface-bounded flight proposals -> 0 closed profiles -> 0 admissible pairs",
        )
        self.assertEqual(
            pair_gate["scope_results"][0]["interface_closure_transition"],
            "2 open flight proposals -> 5 valid port assignments -> 0 closed profiles -> "
            "0 admissible flight pairs",
        )
        self.assertEqual(pair_gate["summary"]["structural_reprojection_edge_count"], 5)
        self.assertEqual(pair_gate["summary"]["structural_reprojection_edge_missing_count"], 5)
        self.assertEqual(
            pair_gate["summary"]["contour_rejection_reason_counts"],
            {
                "projected_rise_station_agreement": 310,
                "repeated_tread_run_support": 310,
                "stepped_surface_and_waist_underside_topology": 310,
                "structural_native_style_compatible": 310,
            },
        )
        self.assertEqual(pair_gate["certificates"], [])
        self.assertEqual(
            {
                item["reason_code"]
                for item in pair_gate["scope_results"][0][
                    "structural_boundary_upstream_abstentions"
                ]
            },
            {"branched_native_boundary", "incomplete_profile_completion"},
        )
        self.assertFalse(pair_gate["contract"]["landing_or_mesh_construction_performed"])
        landing = record["engineering_graph"]["landing_component_reconstruction"]
        self.assertEqual(landing["status"], "resolved_construction_region_pending_same_object_union")
        self.assertEqual(landing["reason_code"], "flight_construction_regions_unresolved")
        self.assertEqual(
            landing["summary"]["transition"],
            "plan envelope 1150x2150 -> landing construction region 975x2150x150 -> "
            "2 internal seam hypotheses + 1 separate-object interface resolved -> "
            "same-object union pending",
        )
        self.assertEqual(landing["plan_footprint_certificate"]["plan_envelope_mm"], [1150.0, 2150.0])
        self.assertEqual(landing["plan_footprint_certificate"]["clear_core_footprint_mm"], [975.0, 2150.0])
        self.assertEqual(landing["section_thickness_certificate"]["thickness_mm"], 150.0)
        self.assertEqual(landing["landing_construction_region"]["extents_xyz_mm"], [975.0, 2150.0, 150.0])
        self.assertAlmostEqual(landing["landing_construction_region"]["analytic_region_volume"]["value_m3"], 0.3144375)
        self.assertFalse(landing["landing_construction_region"]["analytic_region_volume"]["additive_physical_quantity"])
        self.assertEqual(len(landing["internal_seam_hypotheses"]), 2)
        self.assertEqual(len(landing["separate_object_interface_hypotheses"]), 1)
        self.assertEqual(
            landing["separate_object_interface_hypotheses"][0]["state"],
            "resolved_relative_contact",
        )
        self.assertTrue(
            landing["beam_relative_placement_certificate"][
                "relative_physical_placement_resolved"
            ]
        )
        self.assertEqual(landing["landing_beam_step4_replay"]["status"], "accepted")
        self.assertFalse(landing["contract"]["landing_used_as_flight_closure_patch"])
        self.assertFalse(landing["contract"]["landing_is_separate_physical_component"])
        self.assertFalse(landing["contract"]["same_object_step4_replayed"])
        assembly = record["engineering_graph"]["same_object_constructive_assembly"]
        self.assertEqual(assembly["status"], "accepted_invariant_equivalence_class")
        self.assertIsNone(assembly["reason_code"])
        self.assertEqual(
            assembly["summary"]["transition"],
            "5 assignments -> 72 materialized union hypotheses -> "
            "184 pre-kernel rejections -> 72 kernel replays -> 6 survivors",
        )
        self.assertEqual(assembly["summary"]["placement_alternative_count"], 256)
        self.assertEqual(
            assembly["summary"]["temporary_closed_boundary_hypothesis_count"],
            10,
        )
        self.assertEqual(
            assembly["summary"]["temporary_region_materialization_count"], 768
        )
        self.assertEqual(assembly["summary"]["hard_pre_kernel_rejection_count"], 184)
        self.assertEqual(assembly["summary"]["certified_elimination_count"], 0)
        self.assertEqual(assembly["summary"]["unresolved_live_alternative_count"], 0)
        self.assertEqual(assembly["summary"]["kernel_replay_count"], 72)
        self.assertEqual(assembly["summary"]["step4_same_object_survivor_count"], 6)
        self.assertEqual(
            len(
                {
                    round(
                        replay["kernel_result"]["volume_validation"][
                            "analytic_union_mm3"
                        ],
                        6,
                    )
                    for replay in assembly["kernel_replays"]
                    if replay["status"] == "accepted"
                }
            ),
            1,
        )
        self.assertEqual(
            sum(item["step4_status"] == "reclosed_fail"
                for item in assembly["assembly_alternatives"]),
            66,
        )
        self.assertTrue(
            all(
                item["step4_residual_diagnostics"]["supplied_view_reprojections"]
                for item in assembly["assembly_alternatives"]
                if item["step4_status"] == "reclosed_fail"
            )
        )
        self.assertEqual(
            {item["longitudinal_sign"] for item in assembly["profile_to_sweep_transform_alternatives"]},
            {-1, 1},
        )
        self.assertEqual(
            {item["shared_coordinate_reflection_sign"] for item in assembly["profile_to_sweep_transform_alternatives"]},
            {-1, 1},
        )
        self.assertTrue(assembly["contract"]["unevaluated_hypotheses_are_not_ambiguity"])
        self.assertTrue(assembly["contract"]["unreplayed_live_alternatives_block_unique_acceptance"])
        self.assertTrue(assembly["contract"]["final_union_volume_only"])
        self.assertTrue(assembly["contract"]["same_object_step4_union_invoked"])
        self.assertTrue(assembly["contract"]["quantity_eligible"])
        self.assertEqual(
            assembly["invariant_union_equivalence_certificate"]["state"],
            "accepted",
        )
        self.assertAlmostEqual(
            assembly["invariant_concrete_volume_candidate"]["value_m3"],
            1.7726147823710516,
        )
        binding = record["engineering_graph"]["profile_physical_scope_binding"]
        self.assertEqual(binding["status"], "abstained")
        self.assertEqual(binding["summary"]["binding_count"], 0)
        self.assertEqual(binding["summary"]["abstention_count"], 264)
        self.assertEqual(binding["bindings"], [])
        self.assertTrue(
            all(
                item["binding_refs"] or item["upstream_abstention_refs"]
                for item in binding["scope_results"]
            )
        )
        self.assertFalse(binding["contract"]["physical_origin_established"])
        self.assertFalse(binding["contract"]["component_geometry_established"])
        hypotheses = record["engineering_graph"]["physical_component_hypothesis"]
        self.assertEqual(hypotheses["status"], "abstained")
        self.assertEqual(hypotheses["generation_status"], "no_eligible_inputs")
        self.assertEqual(hypotheses["summary"]["eligible_profile_count"], 0)
        self.assertEqual(hypotheses["summary"]["hypothesis_count"], 0)
        self.assertEqual(hypotheses["summary"]["accepted_component_count"], 0)
        self.assertEqual(
            {item["reason_code"] for item in hypotheses["abstentions"]},
            {"no_eligible_step5_inputs"},
        )
        self.assertEqual(hypotheses["hypotheses"], [])
        reclosure = record["engineering_graph"]["physical_component_reclosure"]
        signed_orientation = record["engineering_graph"]["view_frame_graph"][
            "shared_coordinate_system"
        ]["signed_orientation"]
        self.assertEqual(signed_orientation["summary"]["accepted_count"], 0)
        self.assertEqual(signed_orientation["summary"]["insufficient_constraints_count"], 1)
        orientation_certificate = signed_orientation["records"][0]
        self.assertEqual(orientation_certificate["status"], "insufficient_constraints")
        orientation_constraint = orientation_certificate["constraint_certificates"][0]
        self.assertEqual(
            orientation_constraint["oriented_cutting_plane"]["reason"],
            "missing_unique_native_arrowhead",
        )
        self.assertEqual(
            orientation_constraint["signed_shared_axis"]["status"],
            "insufficient_constraints",
        )
        self.assertEqual(
            sum(
                item["status"] == "pass"
                for item in orientation_constraint["signed_shared_axis"]["candidate_evaluations"]
            ),
            0,
        )
        self.assertEqual(reclosure["status"], "abstained")
        self.assertEqual(reclosure["summary"]["hypothesis_count"], 0)
        self.assertEqual(reclosure["summary"]["reclosed_pass_count"], 0)
        self.assertEqual(reclosure["summary"]["reclosed_fail_count"], 0)
        self.assertEqual(reclosure["summary"]["insufficient_constraints_count"], 0)
        self.assertEqual(reclosure["summary"]["slice3_input_count"], 0)
        self.assertEqual(reclosure["summary"]["primary_blockers"], {})
        self.assertEqual(reclosure["accepted_hypothesis_refs"], [])
        self.assertEqual(reclosure["certificates"], [])
        beam = record["engineering_graph"]["clear_span_prism_reconstruction"]
        self.assertEqual(beam["status"], "reclosed_pass")
        self.assertEqual(beam["duplicate_projection_certificate"]["physical_instance_count"], 1)
        self.assertEqual(beam["clear_span_certificate"]["clear_span_mm"], 2150.0)
        self.assertEqual(beam["clear_span_volume_candidate"]["value_m3"], 0.129)
        self.assertFalse(beam["clear_span_volume_candidate"]["quantity_eligible"])

        bands = record["engineering_graph"]["banded_plan_sweep_evidence"]
        self.assertEqual(bands["status"], "plan_bands_resolved_profile_split_pending")
        self.assertEqual(
            bands["plan_band_certificate"]["term_values_mm"],
            [200.0, 975.0, 200.0, 975.0, 200.0],
        )
        self.assertEqual(
            [item["interval_mm"] for item in bands["plan_band_certificate"]["bands"]],
            [[200.0, 1175.0], [1375.0, 2350.0]],
        )
        self.assertEqual(
            bands["profile_split_assessment"]["required_next_certificate"],
            "unique_two_profile_section_split_and_landing_footprint",
        )
        self.assertEqual(
            bands["profile_split_assessment"]["profile_evidence_basis"],
            "title_scope_local_metric_reclosure",
        )
        self.assertEqual(bands["profile_split_assessment"]["resolved_profile_count"], 128)
        promotion = record["engineering_graph"]["physical_object_quantity_promotion"]
        self.assertEqual(promotion["status"], "accepted")
        self.assertEqual(len(promotion["calculated_concrete_quantities"]), 1)
        self.assertEqual(
            promotion["canonical_3d_preview"]["label"],
            "absolute orientation unresolved",
        )
        preview = record["engineering_graph"]["solid_preview"]
        mesh_validation = preview["mesh"]["validation"]
        self.assertEqual(mesh_validation["vertex_count"], 121)
        self.assertEqual(mesh_validation["triangle_count"], 238)
        self.assertEqual(mesh_validation["connected_solid_count"], 1)
        self.assertEqual(mesh_validation["boundary_edge_count"], 0)
        self.assertTrue(preview["rendering_contract"]["per_triangle_strokes"])
        self.assertEqual(
            preview["rendering_contract"]["surface_mode"],
            "transparent_evidence_wireframe",
        )
        self.assertEqual(
            preview["rendering_contract"]["presentation_camera"],
            {
                "projection_preset": "canonical_fold_revealing_axonometric",
                "physical_transform_applied": False,
                "absolute_orientation_claimed": False,
            },
        )
        self.assertFalse(
            preview["rendering_contract"][
                "construction_region_seams_visible_by_default"
            ]
        )
        self.assertTrue(
            preview["rendering_contract"]["construction_region_overlay_available"]
        )
        self.assertFalse(
            preview["rendering_contract"]["construction_region_overlay_enabled"]
        )
        self.assertEqual(len(preview["construction_region_overlays"]), 3)
        terminal = record["engineering_graph"][
            "terminal_landing_support_reconstruction"
        ]
        self.assertEqual(terminal["status"], "resolved_partial_context")
        upper_landing = terminal["terminal_landing_partial"]
        self.assertEqual(upper_landing["thickness_mm"], 150.0)
        self.assertEqual(upper_landing["sweep_width_mm"], 975.0)
        self.assertFalse(upper_landing["longitudinal_extent_complete"])
        self.assertEqual(upper_landing["external_cap_role"], "analysis_cap")
        self.assertEqual(
            upper_landing["plan_projection_state"],
            "not_depicted_in_schematic_stair_plan",
        )
        self.assertFalse(upper_landing["quantity_eligible"])
        terminal_support = terminal["terminal_support_candidate"]
        self.assertEqual(
            [
                terminal_support["section_width_mm"],
                terminal_support["drop_below_slab_mm"],
                terminal_support["clear_span_mm"],
            ],
            [200.0, 300.0, 2150.0],
        )
        self.assertFalse(terminal_support["member_identity_resolved"])
        self.assertFalse(terminal_support["quantity_eligible"])
        partial_materialization = record["engineering_graph"][
            "partial_profile_sweep_materialization"
        ]
        self.assertEqual(
            partial_materialization["status"],
            "materialized_partial_hypotheses",
        )
        self.assertEqual(
            partial_materialization["summary"],
            {
                "hypothesis_count": 2,
                "transform_alternative_count": 2,
                "materialized_candidate_count": 2,
                "physically_complete_candidate_count": 1,
                "quantity_eligible_candidate_count": 0,
            },
        )
        self.assertFalse(
            partial_materialization["candidate_previews"][0][
                "physical_boundary_complete"
            ]
        )
        self.assertTrue(
            partial_materialization["candidate_previews"][1][
                "physical_boundary_complete"
            ]
        )
        contexts = preview["context_candidate_previews"]
        self.assertEqual(len(contexts), 2)
        self.assertEqual(
            [item["classification"] for item in contexts],
            ["upper_landing_or_floor_slab", "terminal_support_beam_candidate"],
        )
        self.assertTrue(
            all(not item["included_in_primary_quantity"] for item in contexts)
        )
        candidates = preview["separate_object_candidate_previews"]
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["classification"], "beam")
        self.assertEqual(candidates[0]["volume_candidate_m3"], 0.129)
        self.assertFalse(candidates[0]["quantity_eligible"])
        self.assertFalse(candidates[0]["included_in_primary_object"])
        self.assertFalse(candidates[0]["included_in_primary_quantity"])
        self.assertTrue(candidates[0]["relative_physical_placement_resolved"])
        self.assertEqual(
            candidates[0]["presentation_frame"]["kind"],
            "shared_relative_scene",
        )
        self.assertIsNone(
            candidates[0]["presentation_frame"]["presentation_only_transform"]
        )
        self.assertEqual(
            candidates[0]["presentation_frame"]["physical_transform"]["origin_xyz_mm"],
            [975.0, 2150.0, 0.0],
        )
        placed_beam = candidates[0]["mesh"]["vertices_xyz_mm"]
        self.assertEqual([min(point[0] for point in placed_beam), max(point[0] for point in placed_beam)], [975.0, 1175.0])
        self.assertEqual([min(point[1] for point in placed_beam), max(point[1] for point in placed_beam)], [0.0, 2150.0])
        self.assertEqual([min(point[2] for point in placed_beam), max(point[2] for point in placed_beam)], [-300.0, 0.0])
        family_scene = record["engineering_graph"]["rebar_program"][
            "family_constrained_3d"
        ]
        self.assertEqual(family_scene["status"], "unresolved")
        self.assertEqual(family_scene["reason"], "no physical rebar families closed")
        self.assertEqual(family_scene["concrete_mesh_status"], "available")
        self.assertEqual(family_scene["path_count"], 0)
        self.assertEqual(preview["rebar_paths"], [])
        reinforcement = record["engineering_graph"]["reinforcement_quantities"]
        self.assertIsNone(reinforcement["total_placed_centerline_m"])
        self.assertIsNone(reinforcement["mass_kg"])

    def test_candidate_08_frozen_semantic_snapshot(self):
        engineering = load_frozen_engineering()
        signature = candidate_08_signature(engineering)
        self.assertEqual(signature["dimension_summary"]["proposal_count"], 111)
        self.assertEqual(signature["dimension_summary"]["accepted_count"], 42)
        self.assertEqual(
            Counter(signature["accepted_values_mm"]),
            Counter(
                {
                    150.0: 8,
                    3150.0: 8,
                    200.0: 5,
                    2999.0: 5,
                    3000.0: 3,
                    4100.0: 2,
                    300.0: 2,
                    975.0: 2,
                    2550.0: 2,
                    2925.0: 1,
                    1175.0: 1,
                    2350.0: 1,
                    1150.0: 1,
                    2150.0: 1,
                }
            ),
        )
        self.assertEqual(signature["specialised_solver_status"], "resolved")
        self.assertEqual(signature["quantity_count"], 1)
        self.assertEqual(signature["frame_summary"]["cutting_plane_relation_count"], 0)
        self.assertEqual(signature["frame_summary"]["reprojection_pass_count"], 1)
        self.assertEqual(signature["frame_summary"]["reprojection_fail_count"], 0)

    def test_candidate_08_live_regeneration_matches_all_current_certificates(self):
        document = fitz.open(
            ROOT / "output" / "rc_staircase_search" / "candidate-08-staircase-page.pdf"
        )
        self.addCleanup(document.close)
        engineering = understand_page(document[0])["engineering_graph"]
        self.assertEqual(candidate_08_signature(engineering), candidate_08_signature(load_frozen_engineering()))
        self._assert_candidate_08_closes_only_redundant_terminal_less_dimensions(engineering)


if __name__ == "__main__":
    unittest.main()
