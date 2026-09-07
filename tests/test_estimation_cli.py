import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from src.drawing_engine.cli import export_comparison, inspect_result, write_json


ROOT = Path(__file__).resolve().parents[1]


class EstimationCliTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "client.pdf"
        self.source.write_bytes(b"fixture PDF bytes; schedule extraction is mocked")
        self.source_hash = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.graphs = self.root / "frozen"
        self.graphs.mkdir()
        self.bundle = self.graphs / "client.bundle.json"
        self.binding = {"source_pdf_sha256": self.source_hash,
                        "canonical_engineering_graph_sha256": "c" * 64}
        self.pipeline = {"version": "fixture", "sha256": "d" * 64}
        self.files = {name: str(self.graphs / f"{name}.json") for name in
                      ("engineering_graph", "observation_graph", "evidence_store")}
        for name, filename in self.files.items():
            payload = {"result_binding": self.binding, "pipeline": self.pipeline,
                       "pages": [{"page": 1}]}
            if name == "engineering_graph":
                payload.update({
                    "canonical_knowledge_graph": {"unresolved": [{"id": "candidate", "reason": "ambiguous"}]},
                    "pages": [{"page": 1, "quantities": [{"net_concrete_m3": 2.0}],
                               "reinforcement_quantities": {"mass_kg": None,
                                                            "totals": {"convention_dependent_mass_kg": 17}},
                               "projected": {"length_m": 5}, "physical": None,
                               "unresolved": [{"reason": "no physical identity", "evidence_refs": ["drawing[4]"]}]}],
                })
            write_json(Path(filename), payload)
        write_json(self.bundle, {"document_key": f"pdf-sha256:{self.source_hash}",
                                 "pipeline": self.pipeline, "files": self.files})
        self.declarations = {"pages": [{"page": 1, "native_text_available": True,
                                      "concrete": [{"value": 1.8, "bbox_display": [1, 2, 3, 4]}],
                                      "reinforcement": [{"value": 18}]}]}
        # Fast orchestration tests do not render PDFs. The live parity case
        # below exercises the actual audit renderer and SQLite import.
        def audit(project, output):
            output.write_bytes(b"mock audit; real renderer covered by live test")
            return {"source": str(project.source())}
        mock = patch("src.drawing_engine.exports.estimation_project_export.render_project_audit", side_effect=audit)
        mock.start()
        self.addCleanup(mock.stop)

    def export(self, name="export"):
        with patch("src.drawing_engine.disciplines.concrete.schedule_comparison.extract_declared_schedules", return_value=self.declarations):
            return export_comparison(self.source, self.root / name, frozen_bundle=self.bundle)

    def test_export_preserves_native_bytes_nulls_evidence_and_separate_channels(self):
        result = self.export()
        manifest = inspect_result(result)
        self.assertEqual(manifest["contract"]["takeoff_completeness"], "not_established")
        graph = inspect_result(result, "engineering_graph")
        self.assertEqual(graph["pages"][0]["projected"], {"length_m": 5})
        self.assertIsNone(graph["pages"][0]["physical"])
        for name, original in self.files.items():
            from src.drawing_engine.exports.estimation_project_export import FrozenProject
            self.assertEqual(FrozenProject(result.parent, manifest).raw(name), Path(original).read_bytes())
        page = inspect_result(result, "comparison", "/pages/0")
        self.assertEqual(page["comparison"]["concrete"]["delta"], 0.2)
        self.assertIsNone(page["comparison"]["reinforcement_mass"]["calculated"])
        self.assertEqual(page["calculated_from_drawing"]["reinforcement_convention_dependent_mass_kg"], 17)
        self.assertIsNone(page["approved_for_quote"])
        self.assertEqual(page["declared_by_designer"]["concrete"][0]["bbox_display"], [1, 2, 3, 4])
        self.assertEqual(inspect_result(result, "unresolved", "/pages/0/unresolved"), graph["pages"][0]["unresolved"])
        from src.drawing_engine.project.project_packed_store import PackedProjectStore
        project = manifest["project"]
        with PackedProjectStore(result.parent / "project.sqlite", readonly=True) as store:
            restored = b''.join(store.iter_artifact_bytes(project_id=project["project_id"], document_id=project["document_id"],
                                      snapshot_id=project["snapshot_id"], name="engineering_graph"))
            self.assertEqual(json.loads(restored), graph)
            self.assertEqual(store.connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            self.assertEqual(store.connection.execute("SELECT count(*) FROM hot_nodes").fetchone()[0], 0)
        self.assertTrue((result.parent / "audit.pdf").exists())
        self.assertEqual([p.name for p in result.parent.rglob('*.json')], ['result.json'])
        self.assertFalse(list(result.parent.rglob('*.html')))

    def test_schedule_changes_never_change_frozen_calculation(self):
        first = inspect_result(self.export(), "comparison")
        self.declarations["pages"][0]["concrete"][0]["value"] = 900
        second = inspect_result(self.export("second"), "comparison")
        self.assertEqual(first["frozen_calculation"]["sha256"], second["frozen_calculation"]["sha256"])
        self.assertEqual(first["pages"][0]["calculated_from_drawing"], second["pages"][0]["calculated_from_drawing"])
        self.assertNotEqual(first["pages"][0]["comparison"], second["pages"][0]["comparison"])

    def test_rejects_wrong_source_before_extracting_schedule(self):
        self.source.write_bytes(b"different source")
        with patch("src.drawing_engine.disciplines.concrete.schedule_comparison.extract_declared_schedules") as extract:
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                export_comparison(self.source, self.root / "export", frozen_bundle=self.bundle)
            extract.assert_not_called()
        self.assertFalse((self.root / "export/result.json").exists())

    def test_rejects_mixed_evidence_binding(self):
        path = Path(self.files["evidence_store"])
        payload = json.loads(path.read_text())
        payload["result_binding"]["canonical_engineering_graph_sha256"] = "f" * 64
        write_json(path, payload)
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            self.export()
        self.assertFalse((self.root / "export/result.json").exists())

    def test_missing_evidence_and_existing_outputs_fail_closed(self):
        result = self.export()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.export()
        self.assertTrue(result.exists())
        Path(self.files["evidence_store"]).unlink()
        with self.assertRaises(FileNotFoundError):
            self.export("missing")
        self.assertFalse((self.root / "missing/result.json").exists())

    def test_missing_evidence_page_cannot_publish_a_dangling_page_link(self):
        path = Path(self.files["evidence_store"])
        payload = json.loads(path.read_text())
        payload["pages"] = []
        write_json(path, payload)
        with self.assertRaisesRegex(ValueError, "page coverage mismatch"):
            self.export()
        self.assertFalse((self.root / "export/result.json").exists())

    def test_portable_inspection_and_tampered_artifact_rejection(self):
        result = self.export()
        moved = self.root / "moved"
        shutil.move(result.parent, moved)
        result = moved / result.name
        self.assertEqual(inspect_result(result, "comparison", "/pages/0/page"), 1)
        database = moved / "project.sqlite"
        original = database.read_bytes()
        database.write_bytes(original + b"tampered")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            inspect_result(result, "comparison")
        manifest = inspect_result(result)
        manifest["artifacts"]["project_database"]["path"] = "../client.pdf"
        write_json(result, manifest)
        with self.assertRaisesRegex(ValueError, "escapes"):
            inspect_result(result, "comparison")

    def test_json_export_is_explicit_exact_and_does_not_change_database(self):
        from src.drawing_engine.exports.estimation_project_export import FrozenProject, export_optional
        result = self.export()
        project = FrozenProject(result.parent, inspect_result(result))
        before = project.database.read_bytes()
        exported = export_optional(project, self.root / 'json', artifact='engineering_graph')
        index = json.loads(exported.read_text())
        path = exported.parent / index['artifacts']['engineering_graph']['path']
        self.assertEqual(path.read_bytes(), Path(self.files['engineering_graph']).read_bytes())
        self.assertEqual(before, project.database.read_bytes())

    def test_wrong_snapshot_and_index_abstain(self):
        result = self.export()
        manifest = inspect_result(result)
        manifest['records']['comparison']['sha256'] = '0' * 64
        write_json(result, manifest)
        with self.assertRaisesRegex(ValueError, 'index mismatch'):
            inspect_result(result, 'comparison')
        manifest['project']['snapshot_id'] = 'f' * 64
        write_json(result, manifest)
        with self.assertRaisesRegex(KeyError, 'snapshot not found'):
            inspect_result(result, 'comparison')

    def test_fresh_mode_uses_existing_generator_before_schedule_extraction(self):
        events = []
        def generate(source, output, profile, progress_callback):
            events.append("drawing")
            shutil.copytree(self.graphs, output, dirs_exist_ok=True)
            return {"bundle": output / self.bundle.name}
        def extract(source):
            events.append("declarations")
            return self.declarations
        with patch("src.drawing_engine.pipelines.generate_object_agnostic_bundle.generate", side_effect=generate), \
             patch("src.drawing_engine.disciplines.concrete.schedule_comparison.extract_declared_schedules", side_effect=extract):
            result = export_comparison(self.source, self.root / "fresh")
        self.assertEqual(events, ["drawing", "declarations"])
        self.assertEqual(inspect_result(result)["mode"], "fresh_analysis")


class EstimationCliLiveTest(unittest.TestCase):
    def test_client_frozen_replay_preserves_existing_comparison_and_all_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            result = export_comparison(ROOT / "1.pdf", Path(directory) / "export",
                                       frozen_bundle=ROOT / "output/object_agnostic/1.object-agnostic-bundle.json")
            actual = inspect_result(result, "comparison")
            expected = json.loads((ROOT / "output/estimates/1.estimate-comparison.json").read_text())
            # The current engine adds this explicit null channel to the older
            # saved comparison. All previously emitted fields must stay exact.
            expected["pages"][0]["calculated_from_drawing"]["reinforcement_convention_dependent_mass_kg"] = None
            self.assertEqual(actual["pages"], expected["pages"])
            from src.drawing_engine.disciplines.concrete.schedule_comparison import build_estimate_comparison
            direct = build_estimate_comparison(ROOT / "1.pdf", ROOT / "output/object_agnostic/1.engineering-graph.json")
            self.assertEqual(actual["pages"], direct["pages"])
            self.assertEqual(actual["frozen_calculation"]["sha256"], expected["frozen_calculation"]["sha256"])
            self.assertEqual(actual["result_currency"], "historical_superseded")
            self.assertIn("engineering_graph_source_hash_missing", actual["binding_issues"])
            import fitz
            from src.drawing_engine.exports.estimation_project_export import FrozenProject, render_project_audit
            with fitz.open(result.parent / 'audit.pdf') as audit:
                self.assertEqual(len(audit), 3)
                self.assertTrue({'DETECTED_VIEWS', 'DIMENSIONS', 'CROSS_VIEW_MATCHES', 'DETAIL_LINKS'} <=
                                {g['name'] for g in audit.get_ocgs().values()})
            project = FrozenProject(result.parent, inspect_result(result))
            before = hashlib.sha256(project.database.read_bytes()).hexdigest()
            with patch('src.drawing_engine.audit.render_object_agnostic_audit.understand_page', side_effect=AssertionError('audit reran extraction')), \
                 patch('src.drawing_engine.audit.render_object_agnostic_audit.extract_declared_schedules', side_effect=AssertionError('audit reread schedules')):
                render_project_audit(project, Path(directory) / 'replayed.pdf')
            self.assertEqual(hashlib.sha256(project.database.read_bytes()).hexdigest(), before)
