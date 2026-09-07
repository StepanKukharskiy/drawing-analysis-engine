import copy
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch

from src.drawing_engine.project.project_knowledge_store import ProjectKnowledgeStore


class ProjectKnowledgeStoreTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "project.sqlite"
        self.scope = {"project_id": "project", "document_id": "document"}
        self.artifact = {"schema_version": "0.1.0", "layer": "example", "document": {
            "document_key": "source-A", "source_pdf_sha256": "a" * 64}, "items": [
            {"id": "ahu", "record_type": "equipment", "page_ref": "page.a", "state": "observed",
             "evidence_refs": ["text", "missing"], "value_channels": {
                 "calculated": None, "declared": 9, "reviewed": None, "approved_for_quote": None}},
            {"id": "text", "record_type": "text", "page_ref": "page.a", "text": "AHU-1"}]}
        self.inputs = {**self.scope, "source_sha256": "a" * 64, "context": {"extractor": "v1", "parameter": 1},
                       "artifacts": {"example": self.artifact}}

    def test_restart_idempotence_graph_links_and_channels(self):
        with ProjectKnowledgeStore(self.path) as store:
            snapshot = store.import_snapshot(**self.inputs)
            self.assertEqual(snapshot, store.import_snapshot(**self.inputs))
            node = store.query(**self.scope, record_type="equipment")[0]
            self.assertEqual(node["payload"], self.artifact["items"][0])
            neighbors = store.neighbors(**self.scope, node_id=node["id"])
            self.assertEqual({r["target_ref"]: r["resolution"] for r in neighbors},
                             {"missing": "unresolved", "text": "unique_record", "page.a": "unresolved"})
            store.add_review(**self.scope, snapshot_id=snapshot, node_id=node["id"],
                             review={"reviewer": "engineer", "decision": "needs_body_evidence"})
        with ProjectKnowledgeStore(self.path) as store:
            self.assertEqual(store.artifact(**self.scope, name="example"), self.artifact)
            self.assertEqual(store.query(**self.scope, source_id="ahu")[0]["id"], node["id"])
            self.assertEqual(len(store.reviews(**self.scope)), 1)

    def test_evidence_export_is_complete_for_one_snapshot_and_rejects_corruption(self):
        import hashlib
        import json
        import zipfile
        import zlib
        from tools.render_mep_interpretation_acceptance import export_snapshot
        output = Path(self.directory.name) / 'evidence.zip'
        with ProjectKnowledgeStore(self.path) as store:
            original = store.import_snapshot(**self.inputs)
            node = store.query(**self.scope, source_id='ahu')[0]
            store.add_review(**self.scope, snapshot_id=original, node_id=node['id'],
                             review={'reviewer': 'test', 'decision': 'needs_review'})
            changed = copy.deepcopy(self.inputs)
            changed['artifacts']['example']['items'][0]['state'] = 'abstained'
            current = store.import_snapshot(**changed)
            scope = {**self.scope, 'snapshot_id': original}
            baseline = Path(self.directory.name) / 'before.json'
            baseline.write_bytes(b'{"before":true}')
            comparison = [('comparison-baselines/before.json', baseline,
                           hashlib.sha256(baseline.read_bytes()).hexdigest())]
            result = export_snapshot(store, scope, output, comparison_files=comparison)
            self.assertEqual(result['review_overlay_count'], 1)
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(json.loads(archive.read('artifacts/example.json')), self.artifact)
                manifest = json.loads(archive.read('manifest.json'))
                self.assertEqual(manifest['snapshot']['id'], original)
                self.assertFalse(manifest['drawing_discovery_complete'])
                self.assertFalse(manifest['engineer_approved'])
                self.assertEqual(len(manifest['artifacts']), 1)
                self.assertEqual(archive.read('comparison-baselines/before.json'), baseline.read_bytes())
            self.assertEqual(store.snapshot(**self.scope)['id'], current)
            preserved = output.read_bytes()
            source = Path(self.directory.name) / 'another-source.pdf'
            source.write_bytes(b'%PDF-another-source')
            with self.assertRaisesRegex(ValueError, 'source PDF differs'):
                export_snapshot(store, scope, output, source=source)
            self.assertEqual(output.read_bytes(), preserved)
            baseline.write_bytes(b'{"before":false}')
            with self.assertRaisesRegex(ValueError, 'comparison baseline changed'):
                export_snapshot(store, scope, output, comparison_files=comparison)
            self.assertEqual(output.read_bytes(), preserved)
            store.connection.execute('UPDATE artifacts SET payload_zlib=? WHERE sha256 IN (SELECT artifact_sha256 FROM snapshot_artifacts WHERE snapshot_id=?)',
                                     (zlib.compress(b'{}'), original))
            with self.assertRaisesRegex(ValueError, 'content hash changed'):
                export_snapshot(store, scope, output)
            self.assertEqual(output.read_bytes(), preserved)

    def test_context_change_invalidates_active_graph_without_losing_history_or_reviews(self):
        with ProjectKnowledgeStore(self.path) as store:
            old = store.import_snapshot(**self.inputs)
            node = store.query(**self.scope, source_id="ahu")[0]
            store.add_review(**self.scope, snapshot_id=old, node_id=node["id"],
                review={"reviewer": "engineer", "decision": "unknown"})
            changed = copy.deepcopy(self.inputs)
            changed["context"]["parameter"] = 2
            changed["artifacts"]["example"]["items"][0]["state"] = "abstained"
            current = store.import_snapshot(**changed)
            self.assertNotEqual(old, current)
            self.assertEqual(store.query(**self.scope, source_id="ahu")[0]["state"], "abstained")
            self.assertEqual(store.query(**self.scope, snapshot_id=old, source_id="ahu")[0]["state"], "observed")
            self.assertEqual(store.reviews(**self.scope), [])
            self.assertEqual(len(store.reviews(**self.scope, snapshot_id=old)), 1)
            self.assertEqual(store.import_snapshot(**self.inputs), old)
            self.assertEqual(store.snapshot(**self.scope)["id"], current)
            with self.assertRaises(KeyError):
                store.add_review(**self.scope, snapshot_id=current, node_id=node["id"],
                    review={"reviewer": "engineer", "decision": "accepted"})

    def test_failed_import_rolls_back_everything_and_keeps_current_pointer(self):
        with ProjectKnowledgeStore(self.path) as store:
            old = store.import_snapshot(**self.inputs)
            changed = copy.deepcopy(self.inputs)
            changed["artifacts"]["example"]["items"][0]["state"] = "changed"
            with patch("src.drawing_engine.project.project_knowledge_store._references", side_effect=RuntimeError("interrupted")):
                with self.assertRaises(RuntimeError):
                    store.import_snapshot(**changed)
            self.assertEqual(store.snapshot(**self.scope)["id"], old)
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0], 1)
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0], 1)

    def test_document_project_isolation_and_source_mismatch(self):
        with ProjectKnowledgeStore(self.path) as store:
            old = store.import_snapshot(**self.inputs)
            other = copy.deepcopy(self.inputs)
            other["project_id"] = "other"
            other["artifacts"]["example"]["items"][1]["id"] = "other-text"
            store.import_snapshot(**other)
            with self.assertRaises(KeyError):
                store.query(project_id="other", document_id="document", snapshot_id=old)
            self.assertEqual(store.query(**self.scope, source_id="other-text"), [])
            other["source_sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "source revision"):
                store.import_snapshot(**other)
            self.assertEqual(store.query(**self.scope, source_id="ahu' OR 1=1 --"), [])

    def test_ambiguous_references_do_not_pick_a_record_or_cross_raw_page_ids(self):
        self.artifact["items"][0]["evidence_refs"] = ["drawing[1]", "text"]
        self.artifact["items"].extend([
            {"id": "drawing[1]", "record_type": "primitive", "page_ref": page} for page in ("page.a", "page.b")])
        other = copy.deepcopy(self.artifact)
        other["layer"] = "other-layer"
        self.inputs["artifacts"]["other"] = other
        with ProjectKnowledgeStore(self.path) as store:
            store.import_snapshot(**self.inputs)
            node = store.query(**self.scope, source_id="ahu")[0]
            neighbors = store.neighbors(**self.scope, node_id=node["id"])
            self.assertTrue(all(r["resolution"] == "multiple_records" for r in neighbors
                                if r["field"] == "/evidence_refs"))
            raw = next(r for r in neighbors if r["target_ref"] == "drawing[1]")
            self.assertEqual({r["page_ref"] for r in raw["targets"]}, {"page.a"})

    def test_unsupported_database_schema_fails_without_rewriting_it(self):
        with sqlite3.connect(self.path) as connection:
            connection.execute("PRAGMA user_version=99")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            ProjectKnowledgeStore(self.path)

    def test_large_containers_remain_exact_in_artifact_and_stale_dependencies_fail(self):
        self.artifact["items"][0]["large_evidence"] = "x" * 70000
        with ProjectKnowledgeStore(self.path) as store:
            store.import_snapshot(**self.inputs)
            node = store.query(**self.scope, source_id="ahu")[0]
            self.assertIsNone(node["payload"])
            self.assertTrue(node["payload_in_artifact_only"])
            self.assertEqual(store.artifact(**self.scope, name="example"), self.artifact)
            self.inputs["artifacts"]["derived"] = {"schema_version": "0.1.0", "layer": "derived",
                "upstream_contract_ref": {"layer": "example", "payload_sha256": "0" * 64}}
            with self.assertRaisesRegex(ValueError, "stale artifact"):
                store.import_snapshot(**self.inputs)

    def test_reader_keeps_previous_snapshot_during_a_writer_transaction(self):
        with ProjectKnowledgeStore(self.path) as reader:
            old = reader.import_snapshot(**self.inputs)
            started, release = threading.Event(), threading.Event()
            errors = []

            def write():
                try:
                    with ProjectKnowledgeStore(self.path) as writer:
                        writer.connection.execute("BEGIN IMMEDIATE")
                        writer.connection.execute("UPDATE active_documents SET document_id='in-flight'")
                        started.set()
                        if not release.wait(5):
                            raise TimeoutError("reader did not complete")
                        writer.connection.rollback()
                except Exception as error:
                    errors.append(error)
                    started.set()

            thread = threading.Thread(target=write)
            thread.start()
            try:
                self.assertTrue(started.wait(5))
                self.assertEqual(reader.snapshot(**self.scope)["id"], old)
                self.assertEqual(reader.query(**self.scope, source_id="ahu")[0]["state"], "observed")
                with ProjectKnowledgeStore(self.path) as newly_opened_reader:
                    self.assertEqual(newly_opened_reader.snapshot(**self.scope)["id"], old)
            finally:
                release.set()
                thread.join(5)
            self.assertEqual(errors, [])

    def test_search_competitor_lists_stay_exact_without_becoming_millions_of_edges(self):
        self.artifact["items"][0]["primitive_candidate_refs"] = ["candidate." + str(i) for i in range(1000)]
        with ProjectKnowledgeStore(self.path) as store:
            store.import_snapshot(**self.inputs)
            node = store.query(**self.scope, source_id="ahu")[0]
            self.assertEqual(node["payload"], self.artifact["items"][0])
            self.assertIn("primitive_candidate_refs", node["artifact_only_reference_fields"])
            edges = store.neighbors(**self.scope, node_id=node["id"])
            self.assertFalse(any(r["field"] == "/primitive_candidate_refs" for r in edges))
            self.assertTrue(any(r["target_ref"] == "text" for r in edges))
            self.assertEqual(store.artifact(**self.scope, name="example"), self.artifact)

    def test_native_query_rows_keep_all_nodes_links_and_exact_pointer_bodies(self):
        native = {'id': 'drawing[1]', 'points': [[1, 2], [3, 4]]}
        source = {'id': 'native-row', 'page_ref': 'page.a', 'source_primitive_ref': 'drawing[1]',
                  'source_native_segment': native, 'evidence_refs': ['text']}
        self.artifact['items'][0]['source_rows'] = [source]
        with ProjectKnowledgeStore(self.path) as store:
            snapshot = store.import_snapshot(**self.inputs)
            for ref, expected in [('native-row', source), ('drawing[1]', native)]:
                node = store.query(**self.scope, source_id=ref)[0]
                self.assertEqual(node['page_ref'], 'page.a')
                self.assertTrue(node['payload_in_artifact_only'])
                self.assertEqual(store.node_payload(**self.scope, node_id=node['id']), expected)
            node = store.query(**self.scope, source_id='native-row')[0]
            neighbors = store.neighbors(**self.scope, node_id=node['id'])
            self.assertEqual(next(r for r in neighbors if r['target_ref'] == 'drawing[1]')['resolution'], 'unique_record')
            self.assertEqual(next(r for r in neighbors if r['target_ref'] == 'text')['resolution'], 'unique_record')
            self.assertEqual(store.artifact(**self.scope, name='example'), self.artifact)
            with self.assertRaises(KeyError):
                store.node_payload(project_id='other', document_id='document',
                                   snapshot_id=snapshot, node_id=node['id'])
