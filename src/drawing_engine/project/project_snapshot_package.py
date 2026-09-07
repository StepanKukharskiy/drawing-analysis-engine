"""Compact, exact snapshot packages for distribution outside the project store.

The project database remains the mutable development history.  A fixture
package keeps one frozen snapshot's canonical artifact bodies, source/audit
hashes and review overlays without copying the rebuildable node, edge or FTS
indexes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import zlib

from src.drawing_engine.project.project_knowledge_store import ProjectKnowledgeStore, _hash, _json


VERSION = "0.1.0"
SQLITE_USER_VERSION = 1001
_EXPECTED_COLUMNS = {
    "fixture": ["id", "schema_version", "snapshot_id", "project_id", "document_id",
        "source_reference", "source_sha256", "context_json", "snapshot_manifest_json",
        "ruleset_json", "audit_reference", "audit_sha256"],
    "artifacts": ["name", "sha256", "byte_count", "payload_zlib"],
    "reviews": ["id", "node_id", "artifact_sha256", "pointer", "source_id",
        "payload_json", "created_at"],
}


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compressed_blocks(blocks):
    compressor = zlib.compressobj(level=9)
    output = []
    digest = hashlib.sha256()
    byte_count = 0
    for block in blocks:
        digest.update(block)
        byte_count += len(block)
        output.append(compressor.compress(block))
    output.append(compressor.flush())
    return b"".join(output), digest.hexdigest(), byte_count


def export_compact_snapshot(*, source_database, output_database, project_id,
                            document_id, source_pdf, audit_pdf, ruleset,
                            snapshot_id=None):
    """Export one exact snapshot without project-history projection indexes."""
    source_database = Path(source_database)
    output_database = Path(output_database)
    source_pdf = Path(source_pdf)
    audit_pdf = Path(audit_pdf)
    if not isinstance(ruleset, dict) or not ruleset:
        raise ValueError("a non-empty ruleset is required")
    if output_database.exists():
        raise ValueError("compact snapshot destination already exists")
    output_database.parent.mkdir(parents=True, exist_ok=True)

    with ProjectKnowledgeStore(source_database) as store:
        snapshot = store.snapshot(project_id=project_id, document_id=document_id,
                                  snapshot_id=snapshot_id)
        manifest = json.loads(snapshot["manifest_json"])
        if _hash(manifest) != snapshot["id"]:
            raise ValueError("project snapshot manifest hash changed")
        source_sha256 = _file_sha256(source_pdf)
        if source_sha256 != snapshot["source_sha256"]:
            raise ValueError("source PDF differs from the frozen snapshot")
        artifact_links = list(store.connection.execute(
            "SELECT name,artifact_sha256 FROM snapshot_artifacts "
            "WHERE snapshot_id=? ORDER BY name", (snapshot["id"],)))
        reviews = list(store.connection.execute(
            "SELECT r.id,r.node_id,n.artifact_sha256,n.pointer,n.source_id,"
            "r.payload_json,r.created_at FROM reviews r JOIN nodes n ON n.id=r.node_id "
            "WHERE r.snapshot_id=? ORDER BY r.id", (snapshot["id"],)))

        connection = sqlite3.connect(output_database)
        try:
            connection.executescript("""
                PRAGMA journal_mode=DELETE;
                PRAGMA synchronous=FULL;
                CREATE TABLE fixture (
                    id INTEGER PRIMARY KEY CHECK(id=1), schema_version TEXT NOT NULL,
                    snapshot_id TEXT NOT NULL, project_id TEXT NOT NULL,
                    document_id TEXT NOT NULL, source_reference TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL, context_json TEXT NOT NULL,
                    snapshot_manifest_json TEXT NOT NULL, ruleset_json TEXT NOT NULL,
                    audit_reference TEXT NOT NULL, audit_sha256 TEXT NOT NULL);
                CREATE TABLE artifacts (
                    name TEXT PRIMARY KEY, sha256 TEXT NOT NULL UNIQUE,
                    byte_count INTEGER NOT NULL, payload_zlib BLOB NOT NULL)
                    WITHOUT ROWID;
                CREATE TABLE reviews (
                    id TEXT PRIMARY KEY, node_id TEXT NOT NULL,
                    artifact_sha256 TEXT NOT NULL, pointer TEXT NOT NULL,
                    source_id TEXT,
                    payload_json TEXT NOT NULL, created_at TEXT NOT NULL)
                    WITHOUT ROWID;
            """)
            connection.execute("PRAGMA user_version=%d" % SQLITE_USER_VERSION)
            audit_sha256 = _file_sha256(audit_pdf)
            connection.execute(
                "INSERT INTO fixture VALUES(1,?,?,?,?,?,?,?,?,?,?,?)", (
                    VERSION, snapshot["id"], project_id, document_id,
                    source_pdf.name, source_sha256, snapshot["context_json"],
                    snapshot["manifest_json"], _json(ruleset), audit_pdf.name,
                    audit_sha256))
            artifacts = []
            for row in artifact_links:
                payload, digest, byte_count = _compressed_blocks(
                    store.iter_artifact_bytes(project_id=project_id,
                        document_id=document_id, snapshot_id=snapshot["id"],
                        name=row["name"]))
                if digest != row["artifact_sha256"]:
                    raise ValueError("snapshot artifact content hash changed: " + row["name"])
                connection.execute("INSERT INTO artifacts VALUES(?,?,?,?)",
                                   (row["name"], digest, byte_count, payload))
                artifacts.append({"name": row["name"], "sha256": digest,
                                  "byte_count": byte_count})
            connection.executemany("INSERT INTO reviews VALUES(?,?,?,?,?,?,?)",
                [(row["id"], row["node_id"], row["artifact_sha256"], row["pointer"],
                  row["source_id"], row["payload_json"], row["created_at"])
                 for row in reviews])
            connection.commit()
            connection.execute("VACUUM")
        finally:
            connection.close()

    report = validate_compact_snapshot(output_database)
    report.update(
        source={"reference": source_pdf.name, "sha256": source_sha256},
        audit={"reference": audit_pdf.name, "sha256": audit_sha256},
        ruleset=ruleset, artifacts=artifacts,
        review_decisions={"count": len(reviews),
            "ids_sha256": hashlib.sha256(_json([row["id"] for row in reviews]).encode()).hexdigest()})
    return report


def validate_compact_snapshot(path):
    """Verify the package schema and every exact canonical artifact body."""
    path = Path(path)
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        if connection.execute("PRAGMA user_version").fetchone()[0] != SQLITE_USER_VERSION:
            raise ValueError("unsupported compact snapshot schema")
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("compact snapshot SQLite integrity check failed")
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if tables != set(_EXPECTED_COLUMNS):
            raise ValueError("compact snapshot table inventory changed")
        for table, columns in _EXPECTED_COLUMNS.items():
            if [row[1] for row in connection.execute("PRAGMA table_info(" + table + ")")] != columns:
                raise ValueError("compact snapshot table schema changed: " + table)
        fixture = connection.execute("SELECT * FROM fixture WHERE id=1").fetchone()
        if fixture is None:
            raise ValueError("compact snapshot fixture record is missing")
        manifest = json.loads(fixture["snapshot_manifest_json"])
        if (_hash(manifest) != fixture["snapshot_id"]
                or manifest.get("source_sha256") != fixture["source_sha256"]
                or manifest.get("project_id") != fixture["project_id"]
                or manifest.get("document_id") != fixture["document_id"]):
            raise ValueError("compact snapshot identity differs from its manifest")
        expected = manifest.get("artifacts", {})
        rows = connection.execute(
            "SELECT name,sha256,byte_count,payload_zlib FROM artifacts ORDER BY name").fetchall()
        if {row["name"]: row["sha256"] for row in rows} != expected:
            raise ValueError("compact snapshot artifact inventory differs")
        for row in rows:
            raw = zlib.decompress(row["payload_zlib"])
            if (len(raw) != row["byte_count"]
                    or hashlib.sha256(raw).hexdigest() != row["sha256"]):
                raise ValueError("compact snapshot artifact body changed: " + row["name"])
        artifact_hashes = {row["sha256"] for row in rows}
        review_count = 0
        for review in connection.execute(
                "SELECT id,node_id,artifact_sha256,pointer,payload_json FROM reviews ORDER BY id"):
            payload = json.loads(review["payload_json"])
            if (review["artifact_sha256"] not in artifact_hashes
                    or _hash([review["artifact_sha256"], review["pointer"]]) != review["node_id"]
                    or not isinstance(payload, dict) or not payload.get("reviewer")
                    or not payload.get("decision")
                    or _hash([fixture["snapshot_id"], review["node_id"], payload]) != review["id"]):
                raise ValueError("compact review target is not resolvable")
            review_count += 1
        return {"schema_version": VERSION, "layer": "compact_project_snapshot_package",
            "snapshot_id": fixture["snapshot_id"], "project_id": fixture["project_id"],
            "document_id": fixture["document_id"], "artifact_count": len(rows),
            "review_decision_count": review_count,
            "database_bytes": path.stat().st_size, "database_sha256": _file_sha256(path)}
    finally:
        connection.close()


def validate_snapshot_package(directory):
    """Verify the compact database together with its manifest and marked audit."""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files = manifest.get("files", {})
    database_record = files.get("database", {})
    audit_record = files.get("marked_audit_pdf", {})
    for record in (database_record, audit_record):
        name = record.get("path")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("compact package contains an unsafe file reference")
    database = directory / database_record["path"]
    audit = directory / audit_record["path"]
    report = validate_compact_snapshot(database)
    if (report["database_sha256"] != database_record.get("sha256")
            or report["database_bytes"] != database_record.get("bytes")
            or _file_sha256(audit) != audit_record.get("sha256")
            or audit.stat().st_size != audit_record.get("bytes")):
        raise ValueError("compact package file hash or size changed")
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        fixture = connection.execute("SELECT * FROM fixture WHERE id=1").fetchone()
        artifacts = [dict(row) for row in connection.execute(
            "SELECT name,sha256,byte_count FROM artifacts ORDER BY name")]
        reviews = [row[0] for row in connection.execute("SELECT id FROM reviews ORDER BY id")]
        if (manifest.get("snapshot_id") != fixture["snapshot_id"]
                or manifest.get("source") != {"reference": fixture["source_reference"],
                                              "sha256": fixture["source_sha256"]}
                or manifest.get("audit") != {"reference": fixture["audit_reference"],
                                             "sha256": fixture["audit_sha256"]}
                or manifest.get("ruleset") != json.loads(fixture["ruleset_json"])
                or manifest.get("artifacts") != artifacts
                or manifest.get("review_decisions") != {"count": len(reviews),
                    "ids_sha256": hashlib.sha256(_json(reviews).encode()).hexdigest()}):
            raise ValueError("compact package manifest differs from SQLite")
        certified = manifest.get("certified_result", {})
        artifact_hashes = {row["name"]: row["sha256"] for row in artifacts}
        if certified.get("sha256") != artifact_hashes.get(certified.get("artifact")):
            raise ValueError("compact package certified result hash changed")
    finally:
        connection.close()
    return report
