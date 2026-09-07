import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from src.drawing_engine.pipelines.generate_object_agnostic_bundle import generate
from src.drawing_engine.project.analysis_job_manifest import load_analysis_job_result
from src.drawing_engine.core.object_agnostic_understanding import SECTION_SEARCH_RE, resolve_text_roles, understand_page
from src.drawing_engine.core.dimension_attachment import attach_dimensions
from src.drawing_engine.core.exchange_records import validate_exchange_records


ROOT = Path(__file__).resolve().parents[1]


class ObjectAgnosticUnderstandingTest(unittest.TestCase):
    def test_progress_events_follow_completed_analysis_stages(self):
        document = fitz.open()
        page = document.new_page(width=300, height=200)
        page.insert_text((40, 60), "PLAN")
        events = []

        understand_page(
            page,
            progress_callback=lambda stage, fraction, snapshot: events.append(
                (stage, fraction, snapshot["page"])
            ),
        )
        document.close()

        self.assertEqual(
            [stage for stage, _, _ in events],
            [
                "native_geometry_scan",
                "native_geometry_scan",
                "native_geometry",
                "dimensions",
                "view_relations",
                "quantity_closure",
                "reinforcement",
            ],
        )
        self.assertEqual([fraction for _, fraction, _ in events], sorted(fraction for _, fraction, _ in events))
        self.assertTrue(all(page_number == 1 for _, _, page_number in events))

    def test_section_labels_can_be_embedded_in_longer_titles(self):
        self.assertIsNotNone(SECTION_SEARCH_RE.search("5-5 (Reinforcement inner layer)"))

    def test_every_native_token_has_one_primary_role(self):
        document = fitz.open(ROOT / "1.pdf")
        page = document[0]
        roles = resolve_text_roles(page, tuple(attach_dimensions(page)))
        self.assertTrue(roles)
        self.assertTrue(all(isinstance(item["resolved_role"], str) for item in roles))
        self.assertTrue(all("epistemic_state" in item for item in roles))
        self.assertTrue(any(item["resolved_role"] == "section_label" for item in roles))
        self.assertTrue(
            all(
                item["epistemic_state"] == "inferred"
                for item in roles
                if item["resolved_role"] == "dimension_candidate"
            )
        )

    def test_compound_rebar_callout_is_an_identifier_outside_a_table(self):
        document = fitz.open()
        page = document.new_page(width=300, height=200)
        page.insert_text((40, 60), "12/594")

        with patch("src.drawing_engine.core.object_agnostic_understanding._outlined_text_roles", return_value=[]):
            roles = resolve_text_roles(page, ())
        document.close()

        callout = next(item for item in roles if item["text"] == "12/594")
        self.assertEqual(callout["resolved_role"], "identifier_candidate")
        self.assertEqual(callout["basis"], "compound_rebar_callout_syntax_outside_table_grid")

    def test_second_compound_callout_line_is_count_and_spacing(self):
        document = fitz.open()
        page = document.new_page(width=300, height=200)
        page.insert_text((40, 60), "12/594")
        page.insert_text((40, 75), "14/100")

        with patch("src.drawing_engine.core.object_agnostic_understanding._outlined_text_roles", return_value=[]):
            roles = resolve_text_roles(page, ())
        document.close()

        upper = next(item for item in roles if item["text"] == "12/594")
        lower = next(item for item in roles if item["text"] == "14/100")
        self.assertEqual(upper["resolved_role"], "identifier_candidate")
        self.assertEqual(upper["paired_count_spacing"]["count"], 14)
        self.assertEqual(lower["resolved_role"], "rebar_count_spacing")
        self.assertEqual(lower["semantic_value"]["spacing_mm"], 100)

    def test_textless_sheet_keeps_resolved_profile_after_dimension_adjudication(self):
        document = fitz.open(ROOT / "test.pdf")
        record = understand_page(document[0])
        self.assertGreater(len(record["observation_graph"]["nodes"]), 1000)
        self.assertGreater(len(record["engineering_graph"]["view_hypotheses"]), 0)
        self.assertGreater(len(record["engineering_graph"]["contour_hypotheses"]), 0)
        self.assertEqual(record["engineering_graph"]["specialised_solver"]["status"], "resolved")
        self.assertEqual(
            record["engineering_graph"]["specialised_solver"]["selected"],
            "generic_dimensioned_profile_extrusion",
        )
        self.assertGreater(
            record["engineering_graph"]["quantities"][0]["net_concrete_m3"],
            0.0,
        )
        self.assertTrue(
            record["engineering_graph"]["dimension_adjudication"]["contract"][
                "all_proposals_reach_ownership"
            ]
        )

    def test_split_bundle_preserves_compact_graph_and_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = generate(ROOT / "2.pdf", Path(directory))
            bundle = json.loads(outputs["bundle"].read_text())
            engineering = json.loads(outputs["engineering"].read_text())
            evidence = json.loads(outputs["evidence"].read_text())
            observation = json.loads(outputs["observation"].read_text())
            job_result = load_analysis_job_result(outputs["analysis_job_result"])
            self.assertEqual(engineering["pipeline"]["version"], "0.9.0")
            self.assertEqual(
                engineering["result_binding"]["canonical_engineering_graph_sha256"],
                bundle["result_binding"]["canonical_engineering_graph_sha256"],
            )
            self.assertEqual(engineering["pipeline"]["sha256"], bundle["pipeline"]["sha256"])
            self.assertTrue(bundle["analysis_cache_key"].startswith("analysis-sha256:"))
            self.assertEqual(
                bundle["analysis_inputs"]["source_pdf_sha256"],
                bundle["document_key"].removeprefix("pdf-sha256:"),
            )
            self.assertEqual(
                bundle["analysis_inputs"]["pipeline_sha256"],
                bundle["pipeline"]["sha256"],
            )
            self.assertEqual(
                bundle["analysis_inputs"]["estimation_profile_sha256"],
                bundle["estimation_profile"]["sha256"],
            )
            self.assertEqual(bundle["processing_options"], {"page_rotation_removed": True})
            self.assertTrue(engineering["canonical_knowledge_graph"]["validation"]["relation_endpoints_exist"])
            self.assertTrue(bundle["contract"]["one_primary_text_role_per_token"])
            self.assertEqual(job_result["state"], "complete")
            self.assertEqual(
                bundle["files"]["analysis_job_result"],
                str(outputs["analysis_job_result"].resolve()),
            )
            replay = json.loads(outputs["solver_replay"].read_text())
            self.assertTrue(replay["contract"]["quantity_change_requires_solver_reclosure"])
            self.assertFalse(replay["quantity_replay"]["changed"])
            self.assertEqual(
                replay["quantity_replay"]["before"]["sha256"],
                replay["quantity_replay"]["after"]["sha256"],
            )
            self.assertGreater(len(observation["pages"][0]["nodes"]), len(engineering["pages"][0]["claims"]))
            self.assertTrue(evidence["pages"][0]["claims"])
            self.assertEqual(
                validate_exchange_records(engineering["pages"][0]["exchange_records"]),
                [],
            )
            self.assertEqual(engineering["pages"][0]["specialised_solver"]["status"], "resolved")
            self.assertTrue(engineering["pages"][0]["calculation_contours"])
            self.assertTrue(
                engineering["pages"][0]["dimension_adjudication"]["contract"][
                    "unique_geometry_ownership_required"
                ]
            )
            self.assertTrue(engineering["pages"][0]["section_rebar_observations"]["sections"])
            self.assertTrue(engineering["pages"][0]["view_frame_graph"]["frames"])
            self.assertTrue(engineering["canonical_knowledge_graph"]["entities"])
            self.assertTrue(engineering["canonical_knowledge_graph"]["validation"]["supported_relation_types_only"])
            self.assertIn("dimension_of", engineering["canonical_knowledge_graph"]["summary"]["relations_by_type"])
            self.assertTrue(evidence["pages"][0]["calculation_contours"])
            self.assertTrue(evidence["pages"][0]["section_rebar_observations"])
            groups = engineering["pages"][0]["reinforcement_groups"]
            self.assertEqual(len(groups), 2)
            diameter_claims = [claim for claim in engineering["pages"][0]["claims"] if claim["kind"] == "reinforcement_diameter"]
            self.assertEqual(len(diameter_claims), 2)
            takeoff = engineering["pages"][0]["reinforcement_quantities"]
            self.assertEqual(takeoff["status"], "partial")
            self.assertIn(
                "pre_family_solver_candidate_quantities",
                engineering["pages"][0]["rebar_program"],
            )


if __name__ == "__main__":
    unittest.main()
