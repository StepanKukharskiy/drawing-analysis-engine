import copy
import unittest

from src.drawing_engine.disciplines.concrete.flight_profile_pair_certificate import certify_flight_profile_pair


def _profile(ref, points, drawing_start):
    edges = [
        f"drawing[{drawing_start + index}].item[0].segment[0]"
        for index in range(len(points))
    ]
    return {
        "id": ref,
        "source_edge_refs": edges,
        "primitive_refs": [edge.split(".segment[", 1)[0] for edge in edges],
        "ordered_boundary_display": [list(point) for point in points],
        "derived_bridges": [],
        "closure": {
            "closed": True,
            "branch_free": True,
            "unique_completion": True,
        },
        "dimension_certificate": {
            "scale_points_per_mm": 0.1,
        },
        "evidence_refs": edges,
    }


def _fixture():
    lower_points = [
        (0, 100),
        (20, 100),
        (20, 80),
        (40, 80),
        (40, 60),
        (60, 60),
        (60, 40),
        (80, 40),
        (80, 20),
        (100, 20),
        (100, 0),
    ]
    upper_points = [
        (100, 0),
        (80, 0),
        (80, -20),
        (60, -20),
        (60, -40),
        (40, -40),
        (40, -60),
        (20, -60),
        (20, -80),
        (0, -80),
        (0, -100),
    ]
    lower = _profile("profile.lower", lower_points, 0)
    upper = _profile("profile.upper", upper_points, 20)
    duplicate_lower = copy.deepcopy(lower)
    duplicate_lower["id"] = "profile.lower.representation.2"
    topology = {
        "drawing_styles": {
            f"drawing[{index}]": {
                "stroke": (0.0, 0.0, 0.0),
                "width": 0.96,
                "dash": "[] 0",
            }
            for index in range(60)
        }
    }
    local_profiles = {
        "scope_results": [
            {
                "scope_ref": "section.scope",
                "assembly": {"profiles": [lower, duplicate_lower, upper]},
            }
        ]
    }
    local_dimensions = {
        "scope_results": [
            {
                "scope_ref": "section.scope",
                "local_ownership": {
                    "attachments": [
                        {
                            "dimension_ref": "dimension.total",
                            "orientation": "vertical",
                        }
                    ]
                },
                "independent_arithmetic_chain_certificates": [
                    {
                        "overall_dimension_ref": "dimension.total",
                        "total_value_mm": 2000.0,
                        "term_values_mm": [1000.0, 1000.0],
                    }
                ],
                "repeated_projected_measurement_certificates": [
                    {"orientation": "horizontal", "value_mm": 200.0}
                ],
            }
        ]
    }
    bands = {
        "plan_band_certificate": {
            "equal_sweep_width_mm": 975.0,
            "bands_are_disjoint": True,
            "bands": [
                {"id": "band.1", "sweep_width_mm": 975.0},
                {"id": "band.2", "sweep_width_mm": 975.0},
            ],
            "evidence_refs": ["band.1", "band.2"],
        }
    }
    frames = {
        "shared_coordinate_system": {
            "scopes": [
                {
                    "contour_correspondences": [
                        {
                            "id": "correspondence.1",
                            "child_view_id": "section.scope",
                            "candidates": [
                                {
                                    "state": "pass",
                                    "child_geometry_refs": [lower["source_edge_refs"][0]],
                                    "child_signature": {
                                        "style_signature": [[0.0, 0.0, 0.0], 0.96, "[]0"]
                                    },
                                    "signed_transform": {
                                        "state": "unresolved",
                                        "candidates": [
                                            {"sign": 1, "offset_mm": 10.0},
                                            {"sign": -1, "offset_mm": 20.0},
                                        ],
                                    },
                                },
                                {
                                    "state": "pass",
                                    "child_geometry_refs": [upper["source_edge_refs"][0]],
                                    "child_signature": {
                                        "style_signature": [[0.0, 0.0, 0.0], 0.96, "[]0"]
                                    },
                                    "signed_transform": {
                                        "state": "unresolved",
                                        "candidates": [
                                            {"sign": 1, "offset_mm": 10.0},
                                            {"sign": -1, "offset_mm": 20.0},
                                        ],
                                    },
                                },
                            ],
                        }
                    ]
                }
            ]
        }
    }
    return local_profiles, local_dimensions, bands, frames, topology


class FlightProfilePairCertificateTest(unittest.TestCase):
    def test_exact_representations_merge_and_one_semantic_pair_passes(self):
        local_profiles, local_dimensions, bands, frames, topology = _fixture()

        result = certify_flight_profile_pair(
            local_profiles,
            local_dimensions,
            bands,
            frames,
            topology,
            [],
            {"fragments": []},
            page_number=1,
        )

        self.assertEqual(result["status"], "accepted_unique_pair")
        self.assertEqual(result["summary"]["raw_candidate_count"], 3)
        self.assertEqual(result["summary"]["canonical_contour_count"], 2)
        self.assertEqual(result["summary"]["collapsed_representation_count"], 1)
        self.assertEqual(result["summary"]["admissible_pair_count"], 1)
        self.assertEqual(len(result["certificates"]), 1)
        self.assertFalse(result["certificates"][0]["mesh_construction_eligible"])
        self.assertFalse(result["contract"]["landing_or_mesh_construction_performed"])

    def test_distinct_native_support_is_not_bbox_deduplicated_and_remains_ambiguous(self):
        local_profiles, local_dimensions, bands, frames, topology = _fixture()
        lower = local_profiles["scope_results"][0]["assembly"]["profiles"][0]
        third = _profile("profile.lower.distinct", lower["ordered_boundary_display"], 40)
        local_profiles["scope_results"][0]["assembly"]["profiles"].append(third)
        candidate = copy.deepcopy(
            frames["shared_coordinate_system"]["scopes"][0]["contour_correspondences"][0][
                "candidates"
            ][0]
        )
        candidate["child_geometry_refs"] = [third["source_edge_refs"][0]]
        frames["shared_coordinate_system"]["scopes"][0]["contour_correspondences"][0][
            "candidates"
        ].append(candidate)

        result = certify_flight_profile_pair(
            local_profiles,
            local_dimensions,
            bands,
            frames,
            topology,
            [],
            {"fragments": []},
            page_number=1,
        )

        self.assertEqual(result["status"], "ambiguous_multiple_pairs")
        self.assertEqual(result["summary"]["canonical_contour_count"], 3)
        self.assertEqual(result["summary"]["admissible_pair_count"], 2)
        self.assertEqual(result["certificates"], [])

    def test_rebar_overlap_rejects_a_contour_before_pairing(self):
        local_profiles, local_dimensions, bands, frames, topology = _fixture()
        lower = local_profiles["scope_results"][0]["assembly"]["profiles"][0]
        rebar = {
            "fragments": [
                {
                    "id": "fragment.1",
                    "primitive_ref": lower["primitive_refs"][0],
                    "mark_hypotheses": ["mark.1"],
                }
            ]
        }

        result = certify_flight_profile_pair(
            local_profiles,
            local_dimensions,
            bands,
            frames,
            topology,
            [],
            rebar,
            page_number=1,
        )

        self.assertEqual(result["status"], "insufficient_constraints")
        self.assertEqual(result["summary"]["flight_role_eligible_contour_count"], 1)
        self.assertEqual(result["summary"]["admissible_pair_count"], 0)
        self.assertEqual(
            result["summary"]["contour_rejection_reason_counts"]["no_reinforcement_path_overlap"],
            1,
        )

    def test_interface_closed_profiles_are_replayed_by_existing_pair_gate(self):
        local_profiles, local_dimensions, bands, frames, topology = _fixture()
        profiles = local_profiles["scope_results"][0]["assembly"]["profiles"]
        local_profiles["scope_results"][0]["assembly"]["profiles"] = []
        interface_closure = {
            "scope_results": [
                {
                    "scope_ref": "section.scope",
                    "open_flight_proposal_count": 2,
                    "valid_port_assignment_count": 1,
                    "closed_profiles": [profiles[0], profiles[2]],
                }
            ]
        }

        result = certify_flight_profile_pair(
            local_profiles,
            local_dimensions,
            bands,
            frames,
            topology,
            [],
            {"fragments": []},
            page_number=1,
            flight_interface_closure=interface_closure,
        )

        self.assertEqual(result["status"], "accepted_unique_pair")
        self.assertEqual(
            result["summary"]["interface_closure_transition"],
            "2 open flight proposals -> 1 valid port assignments -> 2 closed profiles -> 1 admissible flight pairs",
        )

    def test_topology_certified_dual_role_edge_is_not_rejected_as_annotation(self):
        local_profiles, local_dimensions, bands, frames, topology = _fixture()
        profiles = local_profiles["scope_results"][0]["assembly"]["profiles"]
        local_profiles["scope_results"][0]["assembly"]["profiles"] = []
        dual_role_edge = profiles[0]["source_edge_refs"][0]
        open_boundaries = {
            "scope_results": [
                {
                    "scope_ref": "section.scope",
                    "dual_role_structural_override_edge_refs": [dual_role_edge],
                }
            ]
        }
        interface_closure = {
            "scope_results": [
                {
                    "scope_ref": "section.scope",
                    "open_flight_proposal_count": 2,
                    "valid_port_assignment_count": 1,
                    "closed_profiles": [profiles[0], profiles[2]],
                }
            ]
        }

        result = certify_flight_profile_pair(
            local_profiles,
            local_dimensions,
            bands,
            frames,
            topology,
            [{"baseline": {"primitive_ref": dual_role_edge.split(".segment[", 1)[0]}}],
            {"fragments": []},
            page_number=1,
            open_structural_boundary_assembly=open_boundaries,
            flight_interface_closure=interface_closure,
        )

        self.assertEqual(result["status"], "accepted_unique_pair")


if __name__ == "__main__":
    unittest.main()
