import copy
import unittest

from src.drawing_engine.disciplines.concrete.flight_interface_closure import certify_flight_interface_closure


def _port(ref, role, side, point, vertex):
    return {
        "id": ref,
        "source_boundary_role": role,
        "side": side,
        "point_display": list(point),
        "vertex_ref": vertex,
        "state": "open_interface",
    }


def _chain(ref, role, points, vertices, drawing):
    return {
        "id": ref,
        "role": role,
        "boundary_eligible": True,
        "ordered_points_display": [list(point) for point in points],
        "ordered_vertex_refs": list(vertices),
        "split_edge_refs": [
            f"drawing[{drawing}].item[{index}].segment[0]"
            for index in range(len(points) - 1)
        ],
        "source_edge_refs": [
            f"drawing[{drawing}].item[{index}].segment[0]"
            for index in range(len(points) - 1)
        ],
        "endpoint_ports": [
            _port(f"{ref}.start", role, "start", points[0], vertices[0]),
            _port(f"{ref}.end", role, "end", points[-1], vertices[-1]),
        ],
    }


def _fixture():
    chains = []
    proposals = []
    geometries = [
        ([(0, 0), (10, 5), (20, 10), (30, 15)], [(0, 5), (30, 20)]),
        ([(30, 15), (40, 20), (50, 25), (60, 30)], [(30, 20), (60, 35)]),
    ]
    for flight_index, (step_points, waist_points) in enumerate(geometries):
        step_vertices = [f"s{flight_index}.{index}" for index in range(len(step_points))]
        waist_vertices = [f"w{flight_index}.{index}" for index in range(len(waist_points))]
        # Landing vertices are shared with optional native interface paths.
        step_vertices[0 if flight_index else -1] = f"landing.step.{flight_index}"
        waist_vertices[0 if flight_index else -1] = f"landing.waist.{flight_index}"
        step = _chain(f"step.{flight_index}", "stepped_upper_surface", step_points, step_vertices, 10 + flight_index * 2)
        waist = _chain(f"waist.{flight_index}", "waist_underside", waist_points, waist_vertices, 11 + flight_index * 2)
        chains.extend([step, waist])
        proposals.append(
            {
                "id": f"proposal.{flight_index}",
                "stepped_surface_chain_ref": step["id"],
                "waist_underside_chain_ref": waist["id"],
                "dimension_certificate": {
                    "scale_points_per_mm": 0.1,
                    "dimension_refs": ["dimension.scale"],
                },
            }
        )
    open_boundaries = {
        "scope_results": [
            {
                "scope_ref": "section.scope",
                "metric_scale_points_per_mm": 0.1,
                "branch_free_chains": chains,
                "flight_boundary_proposals": proposals,
            }
        ]
    }
    attachments = []
    for index, x in enumerate((0.0, 30.0, 60.0)):
        attachments.append(
            {
                "dimension_ref": f"dimension.station.{index}",
                "status": "accepted",
                "value_mm": 500.0,
                "measured_endpoints": [
                    {"point_display": [x, -10.0]},
                    {"point_display": [x, 40.0]},
                ],
            }
        )
    local = {
        "scope_results": [
            {
                "scope_ref": "section.scope",
                "local_ownership": {"attachments": attachments},
            }
        ]
    }
    plan = {
        "plan_band_certificate": {
            "id": "plan.bands",
            "state": "resolved_relative_unsigned",
            "bands_are_disjoint": True,
            "bands": [
                {"id": "band.1", "sweep_width_mm": 500.0},
                {"id": "band.2", "sweep_width_mm": 500.0},
            ],
            "evidence_refs": ["plan.dimension.1", "plan.dimension.2"],
        }
    }
    return open_boundaries, local, plan


class FlightInterfaceClosureTest(unittest.TestCase):
    def test_unique_station_assignment_closes_two_profiles_with_physical_end_caps(self):
        open_boundaries, local, plan = _fixture()

        result = certify_flight_interface_closure(
            open_boundaries,
            local,
            plan,
            page_number=1,
        )

        self.assertEqual(result["status"], "accepted_unique_assignment")
        self.assertEqual(result["summary"]["open_flight_proposal_count"], 2)
        self.assertEqual(result["summary"]["valid_port_assignment_count"], 1)
        self.assertEqual(result["summary"]["closed_profile_count"], 2)
        self.assertEqual(
            result["summary"]["temporary_closed_boundary_hypothesis_count"], 2
        )
        self.assertTrue(
            all(profile["closure"]["derived_physical_interface_count"] == 2 for profile in result["closed_profiles"])
        )
        interfaces = [
            item
            for profile in result["closed_profiles"]
            for item in profile["derived_physical_interfaces"]
        ]
        self.assertTrue(interfaces)
        self.assertTrue(all(item["native_drawing_edge"] is False for item in interfaces))
        self.assertFalse(result["contract"]["landing_geometry_constructed"])

    def test_native_landing_chain_is_preferred_over_available_end_cap(self):
        open_boundaries, local, plan = _fixture()
        chains = open_boundaries["scope_results"][0]["branch_free_chains"]
        chains.extend(
            [
                _chain(
                    "landing.native.0",
                    "support_or_landing_boundary",
                    [(30, 15), (30, 20)],
                    ["landing.step.0", "landing.waist.0"],
                    30,
                ),
                _chain(
                    "landing.native.1",
                    "support_or_landing_boundary",
                    [(30, 15), (30, 20)],
                    ["landing.step.1", "landing.waist.1"],
                    30,
                ),
            ]
        )

        result = certify_flight_interface_closure(open_boundaries, local, plan, page_number=1)

        self.assertEqual(result["status"], "accepted_unique_assignment")
        accepted = result["certificates"][0]
        landing = [
            option
            for assignment in accepted["assignments"]
            for classification, option in zip(
                assignment["classified_interfaces"], assignment["interface_options"]
            )
            if classification["classification"] == "landing_interface"
        ]
        self.assertEqual({item["kind"] for item in landing}, {"native_support_or_landing_chain"})

    def test_native_path_cannot_bypass_common_interface_station_gate(self):
        open_boundaries, local, plan = _fixture()
        chains = open_boundaries["scope_results"][0]["branch_free_chains"]
        chains.append(
            _chain(
                "landing.detour",
                "support_or_landing_boundary",
                [(30, 15), (35, 18), (40, 20)],
                ["landing.step.0", "landing.detour.mid", "landing.waist.1"],
                35,
            )
        )

        result = certify_flight_interface_closure(open_boundaries, local, plan, page_number=1)
        options = [
            item
            for rows in result["scope_results"][0]["interface_options_by_proposal"].values()
            for item in rows
            if item["left_port_ref"].endswith("step.0.end")
            and item["right_port_ref"].endswith("waist.1.start")
        ]

        self.assertFalse(
            any(item["kind"] == "native_support_or_landing_chain" for item in options)
        )

    def test_missing_plan_station_certificate_abstains(self):
        open_boundaries, local, plan = _fixture()
        plan["plan_band_certificate"]["bands"] = [{"id": "band.1"}]

        result = certify_flight_interface_closure(open_boundaries, local, plan, page_number=1)

        self.assertEqual(result["status"], "insufficient_constraints")
        self.assertEqual(result["closed_profiles"], [])

    def test_multiple_complete_assignments_remain_ambiguous(self):
        open_boundaries, local, plan = _fixture()
        scope = open_boundaries["scope_results"][0]
        scope["branch_free_chains"] = []
        scope["flight_boundary_proposals"] = []
        for index, offset in enumerate((0.0, 10.0)):
            step = _chain(
                f"ambiguous.step.{index}",
                "stepped_upper_surface",
                [(offset, offset), (offset + 10, offset + 10)],
                [f"as{index}.0", f"as{index}.1"],
                50 + index * 2,
            )
            waist = _chain(
                f"ambiguous.waist.{index}",
                "waist_underside",
                [(offset, offset + 10), (offset + 10, offset)],
                [f"aw{index}.0", f"aw{index}.1"],
                51 + index * 2,
            )
            scope["branch_free_chains"].extend([step, waist])
            scope["flight_boundary_proposals"].append(
                {
                    "id": f"ambiguous.proposal.{index}",
                    "stepped_surface_chain_ref": step["id"],
                    "waist_underside_chain_ref": waist["id"],
                    "dimension_certificate": {"scale_points_per_mm": 0.1},
                }
            )
        attachments = []
        for axis, values in (("x", (0.0, 10.0, 20.0)), ("y", (0.0, 10.0, 20.0))):
            for index, coordinate in enumerate(values):
                endpoints = (
                    [[coordinate, -10.0], [coordinate, 30.0]]
                    if axis == "x"
                    else [[-10.0, coordinate], [30.0, coordinate]]
                )
                attachments.append(
                    {
                        "dimension_ref": f"dimension.{axis}.station.{index}",
                        "status": "accepted",
                        "value_mm": 400.0,
                        "measured_endpoints": [
                            {"point_display": endpoints[0]},
                            {"point_display": endpoints[1]},
                        ],
                    }
                )
        local["scope_results"][0]["local_ownership"]["attachments"] = attachments

        result = certify_flight_interface_closure(open_boundaries, local, plan, page_number=1)

        self.assertNotEqual(result["status"], "accepted_unique_assignment")
        self.assertEqual(result["closed_profiles"], [])
        self.assertTrue(result["temporary_closed_boundary_hypotheses"])
        self.assertTrue(
            all(
                item["record_type"] == "temporary_closed_boundary_hypothesis"
                and item["search_input_only"]
                and not item["quantity_eligible"]
                for item in result["temporary_closed_boundary_hypotheses"]
            )
        )

    def test_section_cap_bound_is_invariant_to_plan_sweep_width(self):
        open_boundaries, local, plan = _fixture()

        baseline = certify_flight_interface_closure(
            open_boundaries, local, plan, page_number=1
        )
        widened_plan = copy.deepcopy(plan)
        for band in widened_plan["plan_band_certificate"]["bands"]:
            band["sweep_width_mm"] = 50_000.0
        widened = certify_flight_interface_closure(
            open_boundaries, local, widened_plan, page_number=1
        )

        def cap_bounds(result):
            return sorted(
                (
                    cap["cap_bound_basis"],
                    cap["section_local_cap_bound_points"],
                    cap["profile_endpoint_separation_envelope_points"],
                )
                for scope in result["scope_results"]
                for options in scope["interface_options_by_proposal"].values()
                for option in options
                if (cap := option.get("derived_physical_interface"))
            )

        self.assertEqual(cap_bounds(baseline), cap_bounds(widened))
        self.assertTrue(cap_bounds(baseline))
        self.assertEqual(
            {row[0] for row in cap_bounds(baseline)},
            {"section_profile_endpoint_and_station_envelopes"},
        )


if __name__ == "__main__":
    unittest.main()
