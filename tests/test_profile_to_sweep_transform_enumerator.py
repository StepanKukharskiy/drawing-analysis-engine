import unittest

from src.drawing_engine.disciplines.concrete.profile_to_sweep_transform_enumerator import enumerate_profile_to_sweep_transforms


class ProfileToSweepTransformEnumeratorTest(unittest.TestCase):
    def test_enumerates_longitudinal_sign_and_shared_reflection(self):
        boundary = {
            "id": "temporary.boundary",
            "record_type": "temporary_closed_boundary_hypothesis",
            "ordered_boundary_display": [[0.0, 0.0], [10.0, 0.0], [10.0, 2.0], [0.0, 2.0]],
            "assignment_specific_caps": [
                {
                    "interface_option_ref": "cap.interface",
                    "classification": "landing_interface",
                    "cap_endpoints_display": [[10.0, 0.0], [10.0, 2.0]],
                    "evidence_refs": ["interface.evidence"],
                },
                {
                    "interface_option_ref": "cap.support",
                    "classification": "external_support_end",
                    "cap_endpoints_display": [[0.0, 2.0], [0.0, 0.0]],
                    "evidence_refs": ["support.evidence"],
                },
            ],
            "evidence_refs": ["boundary.evidence"],
        }
        sweep = {
            "id": "bounded.sweep",
            "interval_mm": [200.0, 1175.0],
            "sweep_width_mm": 975.0,
            "evidence_refs": ["sweep.evidence"],
        }
        gauge = {
            "section_scale_points_per_mm": 1.0,
            "flight_contact_coordinate_display": 0.0,
            "landing_top_coordinate_display": 0.0,
            "evidence_refs": ["gauge.evidence"],
        }

        result = enumerate_profile_to_sweep_transforms(
            boundary, sweep, gauge, clear_sweep_origin_mm=200.0
        )

        self.assertEqual(len(result), 4)
        self.assertEqual({item["longitudinal_sign"] for item in result}, {-1, 1})
        self.assertEqual({item["shared_coordinate_reflection_sign"] for item in result}, {-1, 1})
        self.assertEqual({item["mapped_interface_plane_x_mm"] for item in result}, {0.0})
        self.assertEqual(
            {item["sweep_start_maps_to_band_endpoint_mm"] for item in result},
            {200.0, 1175.0},
        )
        self.assertTrue(
            all(
                item["mapping_certificate"]["origin_derived_from_cap_to_interface_correspondence"]
                for item in result
            )
        )
        self.assertTrue(all(not item["quantity_eligible"] for item in result))

    def test_preserves_independently_certified_landing_bearing_anchor(self):
        boundary = {
            "id": "temporary.boundary.with_landing",
            "record_type": "temporary_closed_boundary_hypothesis",
            "ordered_boundary_display": [
                [-10.0, 0.0],
                [4.0, 0.0],
                [4.0, 2.0],
                [-10.0, 2.0],
            ],
            "assignment_specific_caps": [
                {
                    "interface_option_ref": "cap.outer_landing_end",
                    "classification": "landing_interface",
                    "cap_endpoints_display": [[4.0, 0.0], [4.0, 2.0]],
                    "evidence_refs": ["cap.evidence"],
                },
                {
                    "interface_option_ref": "cap.support",
                    "classification": "external_support_end",
                    "cap_endpoints_display": [[-10.0, 2.0], [-10.0, 0.0]],
                    "evidence_refs": ["support.evidence"],
                },
            ],
            "evidence_refs": ["boundary.evidence"],
        }
        sweep = {
            "id": "bounded.sweep",
            "interval_mm": [200.0, 1175.0],
            "sweep_width_mm": 975.0,
            "evidence_refs": ["sweep.evidence"],
        }
        gauge = {
            "section_scale_points_per_mm": 1.0,
            "flight_contact_coordinate_display": 0.0,
            "landing_top_coordinate_display": 0.0,
            "evidence_refs": ["gauge.evidence"],
        }
        authorization = {
            "record_type": "overlap_geometry_authorization_certificate",
            "state": "accepted",
            "authorized_seam_kind": "overlap_union",
            "independent_of_candidate_mesh_intersection": True,
            "evidence_refs": ["native.landing.overlap"],
        }

        result = enumerate_profile_to_sweep_transforms(
            boundary,
            sweep,
            gauge,
            clear_sweep_origin_mm=200.0,
            additional_interface_anchors=(
                {
                    "kind": "certified_section_contact_station",
                    "station_local_mm": 0.0,
                    "source_ref": "station.flight_landing",
                    "evidence_refs": ["station.evidence"],
                    "landing_overlap_authorization": authorization,
                },
            ),
        )

        self.assertEqual(len(result), 8)
        certified = [
            item
            for item in result
            if item["interface_anchor_kind"]
            == "certified_section_contact_station"
        ]
        self.assertEqual(len(certified), 4)
        self.assertEqual(
            {item["interface_station_local_mm"] for item in certified}, {0.0}
        )
        self.assertTrue(
            all(item["landing_overlap_authorization"] == authorization for item in certified)
        )
        self.assertTrue(
            all(
                item["mapping_certificate"][
                    "interface_anchor_is_independently_certified_station"
                ]
                for item in certified
            )
        )


if __name__ == "__main__":
    unittest.main()
