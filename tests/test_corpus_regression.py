import json
import tempfile
import unittest
from pathlib import Path

from tools.generate_corpus_regression import build_report


class CorpusRegressionTest(unittest.TestCase):
    def _write_graph(self, directory: Path, stem: str, fingerprint: str) -> None:
        payload = {
            "pipeline": {"version": "0.2.0", "sha256": fingerprint},
            "timing": {"drawing_understanding_seconds": 1.0, "page_understanding_seconds": [1.0]},
            "canonical_knowledge_graph": {"summary": {"entity_count": 2}, "validation": {"relation_endpoints_exist": True}},
            "pages": [
                {
                    "page": 1,
                    "view_hypotheses": [],
                    "quantities": [],
                    "rebar_program": {"family_constrained_3d": {"path_count": 0}},
                    "solid_preview": {
                        "rebar_paths": [
                            {"placement_status": "drawing_constrained"},
                            {"placement_status": "drawing_constrained"},
                        ]
                    },
                }
            ],
        }
        (directory / f"{stem}.engineering-graph.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_uniform_fingerprint_is_required(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            sources = [directory / "a.pdf", directory / "b.pdf"]
            self._write_graph(directory, "a", "same")
            self._write_graph(directory, "b", "same")
            report = build_report(sources, directory, directory)
            self.assertTrue(report["uniform_pipeline"])
            self.assertEqual(report["pipeline_fingerprints"], ["same"])
            self.assertEqual(report["results"][0]["pages"][0]["rebar_3d_path_count"], 2)
            self.assertEqual(
                report["results"][0]["pages"][0]["rebar_3d_path_basis"],
                "solid_preview.drawing_constrained_rebar_paths",
            )

            self._write_graph(directory, "b", "different")
            report = build_report(sources, directory, directory)
            self.assertFalse(report["uniform_pipeline"])
            self.assertEqual(report["status"], "incomplete_or_mixed")


if __name__ == "__main__":
    unittest.main()
