import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.drawing_engine.disciplines.mep.mep_attribute_binding import validate_mep_attribute_bindings
from src.drawing_engine.disciplines.mep.mep_cross_sheet_runs import (
    build_mep_cross_sheet_runs,
    validate_mep_cross_sheet_runs,
)
from src.drawing_engine.disciplines.mep.mep_route_observations import validate_mep_route_graph
from src.drawing_engine.disciplines.mep.mep_outlined_route_composites import validate_mep_outlined_route_composites
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import validate_mep_terminology_proposals


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "fixtures" / "mep" / "m_and_p_coordination"
CHECKPOINT_DIR = FIXTURE_DIR / "real_m2_m5_checkpoint"
TRUTH_PATH = FIXTURE_DIR / "m_and_p_coordination_m2_m5_checkpoint_truth.json"
REGISTRY_PATH = FIXTURE_DIR / "m_and_p_coordination.sheet-registry.json"
SOURCE_PDF = ROOT / "M&P mark-up against shop systems piping.pdf"
GENERATOR = ROOT / 'tools/generate_mep_m2_m5_real_checkpoint.py'


def _load(name):
    return json.loads((CHECKPOINT_DIR / name).read_text(encoding="utf-8"))


class MepRealM2M5CheckpointTest(unittest.TestCase):
    def test_frozen_real_checkpoint_replays_without_source_pdf(self):
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        m2 = _load("pages_1a_1b.terminology-proposals.json")
        m3 = _load("pages_1a_1b.route-observations.json")
        m3_5 = _load("pages_1a_1b.outlined-route-composites.json")
        m4 = _load("pages_1a_1b.attribute-bindings.json")
        frozen_m5 = _load("pages_1a_1b.cross-sheet-runs.json")
        review = _load("pages_1a_1b.review-freeze.json")

        self.assertEqual(validate_mep_terminology_proposals(m2), [])
        self.assertEqual(validate_mep_route_graph(m3), [])
        self.assertEqual(validate_mep_outlined_route_composites(m3_5), [])
        self.assertEqual(validate_mep_attribute_bindings(m4), [])
        self.assertEqual(validate_mep_cross_sheet_runs(frozen_m5), [])
        replay = build_mep_cross_sheet_runs(
            sheet_registry=registry,
            route_graph=m3,
            attribute_bindings=m4,
        )
        self.assertEqual(replay, frozen_m5)
        self.assertEqual(len(review["accepted_page_local_route_scopes"]), 4)
        self.assertEqual(len(review["accepted_outlined_route_composites"]), 4)
        self.assertEqual(len(review["ambiguous_attribute_targets"]), 0)
        self.assertEqual(len(review["accepted_attribute_bindings"]), 12)
        self.assertEqual(review["accepted_continuation_endpoints"], [])
        self.assertTrue(review["freeze_contract"]["no_real_m5_closure_claimed"])
        self.assertFalse(review["quantity_eligible"])

    def test_real_checkpoint_certifies_composites_and_overlap_without_continuation(self):
        truth = json.loads(TRUTH_PATH.read_text(encoding="utf-8"))
        m4 = _load("pages_1a_1b.attribute-bindings.json")
        m5 = _load("pages_1a_1b.cross-sheet-runs.json")

        composites = _load("pages_1a_1b.outlined-route-composites.json")
        self.assertEqual(composites["summary"]["accepted_composite_count"], 4)
        self.assertTrue(
            all(row["native_strokes_preserved"] for row in composites["accepted_composites"])
        )
        self.assertEqual(m4["summary"]["accepted_relation_count"], 12)
        self.assertEqual(m4["summary"]["abstained_relation_count"], 0)
        self.assertEqual(m5["summary"]["overlap_duplicate_candidate_count"], 4)
        self.assertEqual(m5["summary"]["accepted_overlap_duplicate_count"], 4)
        self.assertEqual(m5["summary"]["canonical_projected_segment_count"], 2)
        self.assertEqual(m5["summary"]["accepted_continuation_count"], 0)
        self.assertEqual(m5["summary"]["physical_run_hypothesis_count"], 0)
        self.assertEqual(m5["summary"]["partial_2_5d_centreline_segment_count"], 4)
        self.assertEqual(m5["summary"]["resolved_3d_centreline_segment_count"], 0)
        self.assertEqual(truth["expected"]["m5_physical_run_hypotheses"], 0)
        self.assertTrue(all(row["state"] == "accepted" for row in m5["overlap_duplicate_candidates"]))
        self.assertTrue(
            all(
                row["certificates"]["matching_transformed_path_signatures"]
                and row["certificates"]["mutual_unique_semantic_target_pairing"]
                for row in m5["overlap_duplicate_candidates"]
            )
        )
        self.assertEqual(m5["continuation_candidates"], [])

    def test_live_regeneration_matches_frozen_real_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "checkpoint"
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(GENERATOR),
                    "--source",
                    str(SOURCE_PDF),
                    "--registry",
                    str(REGISTRY_PATH),
                    "--truth",
                    str(TRUTH_PATH),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            generated = sorted(output_dir.glob("*.json"))
            self.assertEqual(len(generated), 7)
            for path in generated:
                self.assertEqual(
                    json.loads(path.read_text(encoding="utf-8")),
                    json.loads((CHECKPOINT_DIR / path.name).read_text(encoding="utf-8")),
                    path.name,
                )


if __name__ == "__main__":
    unittest.main()
