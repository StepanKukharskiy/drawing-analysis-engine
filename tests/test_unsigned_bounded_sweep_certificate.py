import unittest

from src.drawing_engine.disciplines.concrete.unsigned_bounded_sweep_certificate import (
    certify_unsigned_bounded_sweep,
    validate_unsigned_bounded_sweep_certificate,
)


class UnsignedBoundedSweepCertificateTest(unittest.TestCase):
    def test_two_mirrored_rectangular_sweeps_publish_invariant_certificate(self):
        certificate = certify_unsigned_bounded_sweep(
            page_number=1,
            hypothesis_ref="hypothesis.1",
            physical_scope_ref="physical.1",
            orientation_certificate={
                "id": "signed_orientation.1",
                "status": "insufficient_constraints",
            },
            signed_transform={
                "state": "unresolved",
                "candidates": [
                    {"sign": 1, "offset_mm": 50.0},
                    {"sign": -1, "offset_mm": 950.0},
                ],
            },
            extents_by_axis_mm={"X": 300.0, "Y": 200.0, "Z": 2150.0},
            plan_bounded_axes=["X", "Z"],
            dimension_refs_by_axis={
                "X": ["dimension.300"],
                "Y": ["dimension.200"],
                "Z": ["dimension.2150"],
            },
            external_interface_count=0,
            evidence_refs=["plan.profile", "section.profile"],
        )

        self.assertIsNotNone(certificate)
        self.assertEqual(validate_unsigned_bounded_sweep_certificate(certificate), [])
        self.assertEqual(certificate["signed_orientation_state"], "unresolved")
        self.assertFalse(certificate["signed_physical_placement_resolved"])
        self.assertTrue(certificate["canonical_preview_authorized"])
        self.assertAlmostEqual(certificate["invariant_volume_mm3"], 129_000_000.0)

    def test_interfaces_or_orientation_contradiction_abstain(self):
        common = {
            "page_number": 1,
            "hypothesis_ref": "hypothesis.1",
            "physical_scope_ref": "physical.1",
            "signed_transform": {
                "state": "unresolved",
                "candidates": [
                    {"sign": 1, "offset_mm": 50.0},
                    {"sign": -1, "offset_mm": 950.0},
                ],
            },
            "extents_by_axis_mm": {"X": 300.0, "Y": 200.0, "Z": 2150.0},
            "plan_bounded_axes": ["X", "Z"],
            "dimension_refs_by_axis": {
                "X": ["dimension.300"],
                "Y": ["dimension.200"],
                "Z": ["dimension.2150"],
            },
            "evidence_refs": [],
        }
        self.assertIsNone(
            certify_unsigned_bounded_sweep(
                **common,
                orientation_certificate={"status": "contradiction"},
                external_interface_count=0,
            )
        )
        self.assertIsNone(
            certify_unsigned_bounded_sweep(
                **common,
                orientation_certificate={"status": "insufficient_constraints"},
                external_interface_count=1,
            )
        )


if __name__ == "__main__":
    unittest.main()
