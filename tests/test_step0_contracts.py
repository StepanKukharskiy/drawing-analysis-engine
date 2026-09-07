import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.drawing_engine.project.analysis_job_manifest import (
    build_analysis_job_result,
    load_analysis_job_result,
    validate_analysis_job_result,
)
from src.drawing_engine.core.exchange_records import build_exchange_records, validate_exchange_records


ROOT = Path(__file__).resolve().parents[1]


class Step0ContractsTest(unittest.TestCase):
    def test_exchange_records_have_all_four_stable_types(self):
        payload = build_exchange_records(
            {
                "attachments": [
                    {
                        "id": "dimension.1",
                        "proposal_status": "accepted",
                        "value_mm": 400,
                        "text_bbox_display": [1, 2, 3, 4],
                        "primitive_refs": ["segment.1"],
                    }
                ]
            },
            {
                "segments": [
                    {
                        "id": "segment.1",
                        "view_id": "view.1",
                        "state": "resolved",
                        "title_anchor_ref": "text.1",
                        "bbox_display": [0, 0, 10, 10],
                        "primitive_refs": ["drawing[1]"],
                        "evidence_refs": ["view.1", "text.1"],
                    }
                ]
            },
            [
                {
                    "id": "contour.1",
                    "bbox_display": [0, 0, 5, 5],
                    "primitive_refs": ["segment.2"],
                    "topology": {"closed": True},
                }
            ],
            {
                "scopes": [
                    {
                        "id": "scope.1",
                        "state": "resolved_relative",
                        "view_mappings": {"view.1": {"u": "X", "v": "Z", "normal": "Y"}},
                        "evidence_refs": ["relation.1"],
                    }
                ]
            },
        )
        self.assertEqual(validate_exchange_records(payload), [])
        self.assertEqual(payload["dimension_chain_candidates"][0]["state"], "detector_qualified")
        self.assertIsNone(payload["physical_component_transforms"][0]["origin_xyz_mm"])
        self.assertEqual(payload["physical_component_transforms"][0]["axis_signs"], "unresolved")
        self.assertEqual(payload["raw_contour_candidates"][0]["record_type"], "raw_contour_candidate")
        self.assertEqual(payload["assembled_profile_candidates"], [])
        self.assertEqual(payload["profile_physical_scope_bindings"], [])
        self.assertEqual(payload["physical_component_hypotheses"], [])
        self.assertEqual(
            payload["physical_component_identity_placement_certificates"], []
        )

    def test_exchange_records_consume_validated_profiles_without_recasting_raw_contours(self):
        profile = {
            "record_type": "assembled_profile_candidate",
            "record_version": "0.1.0",
            "id": "assembled_profile_candidate.page_0001.scope_a.evidence_b",
            "state": "resolved",
            "scope_ref": "scope.1",
            "quantity_eligible": False,
            "source_edge_refs": ["segment.1"],
            "derived_bridge_refs": [],
            "closure": {
                "closed": True,
                "branch_free": True,
                "unique_completion": True,
                "scale_bounded": True,
                "dimensionally_redundant": True,
            },
            "evidence_refs": ["segment.1", "dimension.1", "dimension.2"],
        }
        payload = build_exchange_records(
            {},
            {},
            [
                {
                    "id": "contour.1",
                    "primitive_refs": ["drawing[1]"],
                    "topology": {"closed": True},
                }
            ],
            {},
            profile_assembly={"profiles": [profile]},
        )

        self.assertEqual(validate_exchange_records(payload), [])
        self.assertEqual(payload["assembled_profile_candidates"], [profile])
        self.assertEqual(payload["raw_contour_candidates"][0]["source_contour_ref"], "contour.1")
        self.assertNotEqual(
            payload["raw_contour_candidates"][0]["record_type"],
            payload["assembled_profile_candidates"][0]["record_type"],
        )

    def test_exchange_records_publish_profile_physical_scope_bindings(self):
        binding = {
            "record_type": "profile_physical_scope_binding",
            "record_version": "0.1.0",
            "id": "profile_physical_scope_binding.page_0001.evidence_a",
            "state": "accepted",
            "profile_ref": "profile.1",
            "physical_scope_ref": "physical.1",
            "step5_reconstruction_input_eligible": True,
            "step4_kernel_invocation_eligible": False,
            "additive_component_identity_established": False,
            "quantity_eligible": False,
            "evidence_refs": ["profile.1", "physical.1"],
        }
        payload = build_exchange_records(
            {}, {}, [], {}, profile_scope_binding={"bindings": [binding]}
        )

        self.assertEqual(validate_exchange_records(payload), [])
        self.assertEqual(payload["profile_physical_scope_bindings"], [binding])

    def test_exchange_records_publish_quantity_free_component_hypotheses(self):
        hypothesis = {
            "record_type": "physical_component_hypothesis",
            "record_version": "0.1.0",
            "id": "physical_component_hypothesis.page_0001.evidence_a",
            "state": "candidate",
            "profile_refs": ["profile.1"],
            "binding_refs": ["binding.1"],
            "additive_component_identity_established": False,
            "physical_component_ref": None,
            "physical_transform_refs": [],
            "step4_kernel_invocation_eligible": False,
            "quantity_eligible": False,
            "evidence_refs": ["profile.1", "binding.1"],
        }
        payload = build_exchange_records(
            {},
            {},
            [],
            {},
            physical_component_hypothesis={"hypotheses": [hypothesis]},
        )

        self.assertEqual(validate_exchange_records(payload), [])
        self.assertEqual(payload["physical_component_hypotheses"], [hypothesis])

    def test_exchange_records_publish_slice2_certificates_without_step4_authority(self):
        certificate = {
            "record_type": "physical_component_identity_placement_certificate",
            "record_version": "0.1.0",
            "id": "physical_component_identity_placement_certificate.page_0001.evidence_a",
            "state": "abstained",
            "status": "insufficient_constraints",
            "hypothesis_ref": "hypothesis.1",
            "slice3_input_eligible": False,
            "physical_component_ref": None,
            "physical_transform_ref": None,
            "step4_kernel_invocation_eligible": False,
            "quantity_eligible": False,
            "evidence_refs": ["hypothesis.1"],
        }
        payload = build_exchange_records(
            {},
            {},
            [],
            {},
            physical_component_reclosure={"certificates": [certificate]},
        )

        self.assertEqual(validate_exchange_records(payload), [])
        self.assertEqual(
            payload["physical_component_identity_placement_certificates"],
            [certificate],
        )

    def test_exchange_record_validation_rejects_wrong_record_type(self):
        payload = build_exchange_records({}, {}, [], {})
        payload["dimension_chain_candidates"] = [
            {"id": "bad", "record_type": "view_scope_candidate", "record_version": "0.1.0", "evidence_refs": []}
        ]
        self.assertTrue(validate_exchange_records(payload))

    def test_resolved_view_scopes_require_disjoint_native_membership(self):
        payload = build_exchange_records(
            {},
            {
                "segments": [
                    {
                        "id": "scope.1",
                        "view_id": "view.parent",
                        "state": "resolved",
                        "title_anchor_ref": "title.1",
                        "primitive_refs": ["drawing[1]"],
                    },
                    {
                        "id": "scope.2",
                        "view_id": "view.parent",
                        "state": "resolved",
                        "title_anchor_ref": "title.2",
                        "primitive_refs": ["drawing[1]"],
                    },
                ]
            },
            [],
            {},
        )
        self.assertTrue(
            any("overlap another resolved scope" in item for item in validate_exchange_records(payload))
        )

    def test_job_result_is_content_addressed_and_path_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "drawing.pdf"
            artifact = root / "drawing.engineering-graph.json"
            source.write_bytes(b"%PDF fixture")
            artifact.write_text("{}\n", encoding="utf-8")
            payload = build_analysis_job_result(
                source,
                {"version": "test", "sha256": "a" * 64},
                {"engineering_graph": artifact},
                [],
                {
                    "source_pdf_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "canonical_engineering_graph_sha256": "b" * 64,
                    "pipeline": {"version": "test", "sha256": "a" * 64},
                    "ruleset": {"version": "test", "sha256": "c" * 64},
                    "generated_at": "2026-01-01T00:00:00+00:00",
                    "model_versions": [],
                },
            )
            path = root / "result.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(validate_analysis_job_result(payload), [])
            self.assertEqual(load_analysis_job_result(path), payload)
            self.assertEqual(payload["source"]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(payload["artifacts"]["engineering_graph"]["filename"], artifact.name)

    def test_job_result_rejects_unlisted_or_path_bearing_artifacts(self):
        payload = {
            "schema_version": "0.1.0",
            "state": "complete",
            "document_key": "pdf-sha256:" + "0" * 64,
            "artifacts": {
                "engineering_graph": {"filename": "../graph.json", "bytes": 1, "sha256": "0" * 64},
                "secret": {"filename": "secret", "bytes": 1, "sha256": "0" * 64},
            },
        }
        errors = validate_analysis_job_result(payload)
        self.assertTrue(any("unsupported artifact" in item for item in errors))
        self.assertTrue(any("must not contain a path" in item for item in errors))

    def test_frozen_fixture_manifest_matches_every_recorded_byte(self):
        manifest = json.loads((ROOT / "config" / "step0_fixture_manifest.json").read_text(encoding="utf-8"))
        for item in [*manifest["fixtures"], *manifest["baselines"]]:
            path = ROOT / item["path"]
            if item["path"].endswith('.py'):
                from tools.source_revision import frozen_source_path
                path = frozen_source_path(ROOT, item["path"], item["sha256"])
            self.assertTrue(path.is_file(), item["path"])
            self.assertEqual(path.stat().st_size, item["bytes"], item["path"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"], item["path"])
        self.assertEqual(manifest["semantic_no_regression"]["K1"]["net_concrete_m3"], 1.8102)
        self.assertEqual(manifest["semantic_no_regression"]["K7"]["net_concrete_m3"], 1.6632)


if __name__ == "__main__":
    unittest.main()
