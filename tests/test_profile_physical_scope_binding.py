import copy
import json
import unittest

from src.drawing_engine.disciplines.concrete.profile_physical_scope_binding import (
    bind_profiles_to_physical_scopes,
    validate_profile_physical_scope_binding,
)


def profile(profile_ref="profile.1", scope_ref="scope.1"):
    return {
        "id": profile_ref,
        "record_type": "assembled_profile_candidate",
        "record_version": "0.1.0",
        "state": "resolved",
        "scope_ref": scope_ref,
        "quantity_eligible": False,
        "closure": {
            "closed": True,
            "branch_free": True,
            "unique_completion": True,
            "scale_bounded": True,
            "dimensionally_redundant": True,
        },
        "evidence_refs": [f"edge.{profile_ref}", "dimension.1", "dimension.2"],
    }


def assembly(*profiles, scopes=None):
    return {
        "profiles": list(profiles),
        "scopes": scopes
        or [
            {
                "scope_ref": item["scope_ref"],
                "view_ref": f"view.{item['scope_ref']}",
                "profiles": [item],
                "abstentions": [],
            }
            for item in profiles
        ],
    }


def physical_scope(scope_ref="physical.1", view_ref="scope.1"):
    return {
        "id": scope_ref,
        "state": "resolved_relative",
        "physical_object_identity_state": "resolved_relative",
        "view_ids": [view_ref],
        "parent_view_ids": [view_ref],
        "section_view_ids": [],
        "supporting_projection_view_ids": [],
        "relation_refs": ["cut.1"],
        "shared_coordinate_scope_id": "coordinate.1",
        "reprojection_validation_refs": ["coordinate.1.reprojection.1"],
        "quantity_aggregation_eligible": False,
        "solid_aggregation_eligible": False,
        "evidence_refs": ["cut.1", "coordinate.1"],
    }


def frame_graph(*scopes):
    return {
        "relations": [
            {
                "id": "cut.1",
                "state": "accepted",
                "integration_certificate": {"status": "passed"},
            }
        ],
        "object_scopes": list(scopes),
        "shared_coordinate_system": {
            "scopes": [
                {
                    "id": "coordinate.1",
                    "state": "resolved_relative",
                    "reprojection_validations": [
                        {"status": "pass", "evidence_refs": ["dimension.1"]}
                    ],
                }
            ]
        },
    }


class ProfilePhysicalScopeBindingTest(unittest.TestCase):
    def test_binds_exact_title_scope_without_authorizing_solid_or_quantity(self):
        item = profile()
        result = bind_profiles_to_physical_scopes(
            assembly(item), frame_graph(physical_scope()), page_number=7
        )

        self.assertEqual(validate_profile_physical_scope_binding(result), [])
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["summary"]["binding_count"], 1)
        binding = result["bindings"][0]
        self.assertEqual(binding["profile_ref"], "profile.1")
        self.assertEqual(binding["physical_scope_ref"], "physical.1")
        self.assertEqual(binding["membership_match_basis"], "exact_title_scope")
        self.assertEqual(binding["membership_role"], "parent_projection")
        self.assertTrue(binding["step5_reconstruction_input_eligible"])
        self.assertFalse(binding["step4_kernel_invocation_eligible"])
        self.assertFalse(binding["additive_component_identity_established"])
        self.assertIsNone(binding["physical_component_ref"])
        self.assertFalse(binding["quantity_eligible"])
        self.assertFalse(result["contract"]["physical_origin_established"])
        self.assertFalse(result["contract"]["component_geometry_established"])
        self.assertFalse(item["quantity_eligible"])

    def test_unique_unsplit_source_view_alias_can_bind(self):
        item = profile()
        scoped = assembly(
            item,
            scopes=[
                {
                    "scope_ref": "scope.1",
                    "view_ref": "view.source",
                    "profiles": [item],
                    "abstentions": [],
                }
            ],
        )
        result = bind_profiles_to_physical_scopes(
            scoped,
            frame_graph(physical_scope(view_ref="view.source")),
            page_number=1,
        )

        self.assertEqual(result["bindings"][0]["membership_match_basis"], "unique_unsplit_source_view")

    def test_split_source_view_alias_cannot_collapse_title_scopes(self):
        first = profile("profile.1", "scope.1")
        second = profile("profile.2", "scope.2")
        scoped = assembly(
            first,
            second,
            scopes=[
                {
                    "scope_ref": "scope.1",
                    "view_ref": "view.source",
                    "profiles": [first],
                    "abstentions": [],
                },
                {
                    "scope_ref": "scope.2",
                    "view_ref": "view.source",
                    "profiles": [second],
                    "abstentions": [],
                },
            ],
        )
        result = bind_profiles_to_physical_scopes(
            scoped,
            frame_graph(physical_scope(view_ref="view.source")),
            page_number=1,
        )

        self.assertEqual(result["bindings"], [])
        self.assertEqual(
            {item["reason_code"] for item in result["abstentions"]},
            {"physical_scope_membership_unresolved"},
        )

    def test_ambiguous_physical_scope_membership_abstains(self):
        second_scope = physical_scope("physical.2")
        result = bind_profiles_to_physical_scopes(
            assembly(profile()),
            frame_graph(physical_scope(), second_scope),
            page_number=1,
        )

        self.assertEqual(result["bindings"], [])
        self.assertEqual(
            result["abstentions"][0]["reason_code"],
            "ambiguous_physical_scope_membership",
        )
        self.assertEqual(
            result["abstentions"][0]["candidate_physical_scope_refs"],
            ["physical.1", "physical.2"],
        )

    def test_top_level_profile_cannot_claim_another_scopes_membership(self):
        item = profile()
        scoped = assembly(
            item,
            scopes=[
                {
                    "scope_ref": "scope.1",
                    "view_ref": "view.scope.1",
                    "profiles": [],
                    "abstentions": [],
                }
            ],
        )
        result = bind_profiles_to_physical_scopes(
            scoped, frame_graph(physical_scope()), page_number=1
        )

        self.assertEqual(result["bindings"], [])
        self.assertEqual(
            result["abstentions"][0]["reason_code"],
            "profile_scope_membership_mismatch",
        )

    def test_unclosed_step2_certificate_cannot_bind(self):
        graph = frame_graph(physical_scope())
        graph["shared_coordinate_system"]["scopes"][0]["reprojection_validations"][0][
            "status"
        ] = "unknown"
        result = bind_profiles_to_physical_scopes(
            assembly(profile()), graph, page_number=1
        )

        self.assertEqual(result["bindings"], [])
        self.assertEqual(
            result["abstentions"][0]["reason_code"],
            "physical_scope_membership_unresolved",
        )
        self.assertIn("physical.1", result["diagnostics"]["rejected_physical_scopes"])

    def test_reordering_profiles_and_scopes_is_byte_identical(self):
        first = profile("profile.1", "scope.1")
        second = profile("profile.2", "scope.2")
        first_scope = physical_scope("physical.1", "scope.1")
        second_scope = physical_scope("physical.2", "scope.2")
        original = bind_profiles_to_physical_scopes(
            assembly(first, second),
            frame_graph(first_scope, second_scope),
            page_number=3,
        )
        reordered = bind_profiles_to_physical_scopes(
            assembly(second, first),
            frame_graph(copy.deepcopy(second_scope), copy.deepcopy(first_scope)),
            page_number=3,
        )

        encode = lambda value: json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self.assertEqual(encode(original), encode(reordered))

    def test_validator_rejects_direct_kernel_authorization(self):
        result = bind_profiles_to_physical_scopes(
            assembly(profile()), frame_graph(physical_scope()), page_number=1
        )
        result["bindings"][0]["step4_kernel_invocation_eligible"] = True

        self.assertTrue(
            any(
                "must not directly authorize Step 4" in item
                for item in validate_profile_physical_scope_binding(result)
            )
        )


if __name__ == "__main__":
    unittest.main()
