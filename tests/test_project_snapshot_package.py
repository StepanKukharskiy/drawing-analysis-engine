import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from src.drawing_engine.project.project_knowledge_store import ProjectKnowledgeStore
from src.drawing_engine.project.project_snapshot_package import export_compact_snapshot, validate_compact_snapshot


class CompactSnapshotPackageTest(unittest.TestCase):
    def test_one_snapshot_roundtrips_artifacts_and_reviews_without_indexes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pdf"
            audit = root / "audit.pdf"
            source.write_bytes(b"source revision")
            audit.write_bytes(b"marked audit")
            import hashlib
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            database = root / "project.sqlite"
            artifact = {"schema_version": "0.1.0", "layer": "test_layer",
                "document": {"source_pdf_sha256": digest},
                "records": [{"id": "item-1", "record_type": "test_item",
                             "state": "observed", "value": 7}]}
            with ProjectKnowledgeStore(database) as store:
                snapshot = store.import_snapshot(project_id="p", document_id="d",
                    source_sha256=digest, context={"scope": "test"},
                    artifacts={"evidence": artifact})
                node = store.query(project_id="p", document_id="d",
                                   snapshot_id=snapshot, record_type="test_item")[0]
                store.add_review(project_id="p", document_id="d", snapshot_id=snapshot,
                    node_id=node["id"], review={"reviewer": "fixture-test",
                        "decision": "retain_observed"})
            compact = root / "fixture.sqlite"
            report = export_compact_snapshot(source_database=database,
                output_database=compact, project_id="p", document_id="d",
                snapshot_id=snapshot, source_pdf=source, audit_pdf=audit,
                ruleset={"method_version": "test"})
            self.assertEqual(report["artifact_count"], 1)
            self.assertEqual(report["review_decision_count"], 1)
            self.assertEqual(validate_compact_snapshot(compact)["snapshot_id"], snapshot)
            with self.assertRaisesRegex(ValueError,"unsupported project database schema"):
                ProjectKnowledgeStore(compact)
            with sqlite3.connect(compact) as connection:
                self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0],1001)
                tables = {row[0] for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")}
                self.assertEqual(tables, {"fixture", "artifacts", "reviews"})
                self.assertEqual(json.loads(connection.execute(
                    "SELECT ruleset_json FROM fixture").fetchone()[0]),
                    {"method_version": "test"})
                connection.execute("UPDATE reviews SET payload_json=?",
                    (json.dumps({"reviewer": "fixture-test", "decision": "changed"}),))
                connection.commit()
            with self.assertRaisesRegex(ValueError,"review target"):
                validate_compact_snapshot(compact)


if __name__ == "__main__":
    unittest.main()
