#!/usr/bin/env python3
"""Build and verify a chunked shadow project database before any cutover.

The legacy database is read only.  ``active`` migrates active snapshots first;
``history`` adds every historical snapshot and review.  ``verify`` writes a
content-addressed parity and storage-efficiency report into the shadow database.
``cutover`` refuses to run unless all-history parity, a 70% whole-store
reduction and the 2 GiB projection-index ceiling all pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sqlite3
import subprocess
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.drawing_engine.project.project_knowledge_store import (CHUNK_BYTES, ProjectKnowledgeStore,
    _artifact_chunks, _insert_fts, _json)


CHUNK_ONLY_NAMES = {
    "automatic-discovery", "route-observations", "fitting-route-observations",
    "outlined-route-composites", "fitting-outlined-route-composites",
    "outlined-route-connections", "trace-source-queries", "stroke-ownership-queries",
}
AUDIT_NAMES = {"sheet-registry", "network-hierarchy", "hvac-inventory", "item-catalog",
               "cross-sheet-runs", "attribute-bindings", "projected-identity-bindings",
               "review-outcomes"}
MIN_WHOLE_STORE_REDUCTION = 0.70
MAX_WHOLE_STORE_FRACTION = 0.30
MAX_PROJECTION_INDEX_BYTES = 2 * 1024 ** 3
MINIMUM_TRANSACTION_FREE_BYTES = 6 * 1024 ** 3


def _projection_group(name):
    """Classify SQLite objects that duplicate the canonical artifact projection."""
    if name == "nodes" or name.startswith("nodes_") or name.startswith("sqlite_autoindex_nodes_"):
        return "nodes"
    if name == "edges" or name.startswith("edges_") or name.startswith("sqlite_autoindex_edges_"):
        return "edges"
    if (name == "fts_nodes" or name.startswith("sqlite_autoindex_fts_nodes_")
            or name == "node_fts" or name.startswith("node_fts_")):
        return "fts5"
    return None


def _database_bytes(path):
    files = [path, Path(str(path) + "-wal"), Path(str(path) + "-shm")]
    return {"logical": sum(item.stat().st_size for item in files if item.exists()),
            "allocated": sum(item.stat().st_blocks * 512 for item in files if item.exists())}


def _dbstat_projection_bytes(path):
    """Measure projection B-trees with SQLite page accounting, fail closed if unavailable."""
    path = Path(path)
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        object_names = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_schema WHERE type IN ('table','index') ORDER BY name")
            if _projection_group(row[0])]
        try:
            rows = connection.execute(
                "SELECT name,sum(pgsize) FROM dbstat WHERE name IN (" +
                ",".join("?" for _ in object_names) + ") GROUP BY name", object_names).fetchall()
        except sqlite3.OperationalError as exc:
            if "dbstat" not in str(exc).lower():
                raise
            rows = None
    method = "python_sqlite_dbstat"
    if rows is None:
        executable = shutil.which("sqlite3")
        if executable is None:
            return {"status": "unavailable", "reason": "SQLite dbstat support is required",
                    "cutover_gate_passed": False}
        quoted = ",".join("'" + name.replace("'", "''") + "'" for name in object_names)
        sql = ("SELECT name,coalesce(sum(pgsize),0) FROM dbstat WHERE name IN (" + quoted +
               ") GROUP BY name ORDER BY name;")
        completed = subprocess.run([executable, "-readonly", "-separator", "\t", str(path), sql],
                                   check=False, capture_output=True, text=True)
        if completed.returncode:
            return {"status": "unavailable", "reason": completed.stderr.strip() or "dbstat failed",
                    "cutover_gate_passed": False}
        rows = []
        for line in completed.stdout.splitlines():
            name, size = line.rsplit("\t", 1)
            rows.append((name, int(size)))
        method = "sqlite3_cli_dbstat"
    objects = {name: size for name, size in rows}
    groups = {"nodes": 0, "edges": 0, "fts5": 0}
    for name, size in objects.items():
        groups[_projection_group(name)] += size
    total = sum(groups.values())
    return {"status": "measured", "method": method, "objects": objects, "groups": groups,
            "total_bytes": total, "maximum_bytes": MAX_PROJECTION_INDEX_BYTES,
            "cutover_gate_passed": total <= MAX_PROJECTION_INDEX_BYTES}


def _storage_efficiency(source_path, shadow_path, *, projection=None):
    source = _database_bytes(source_path)
    shadow = _database_bytes(shadow_path)
    reductions = {kind: (1.0 - shadow[kind] / source[kind]) if source[kind] else None
                  for kind in ("logical", "allocated")}
    whole_store_passed = all(value is not None and value >= MIN_WHOLE_STORE_REDUCTION
                             for value in reductions.values())
    projection = projection or _dbstat_projection_bytes(shadow_path)
    return {"source_bytes": source, "shadow_bytes": shadow,
            "whole_store": {"minimum_reduction": MIN_WHOLE_STORE_REDUCTION,
                "maximum_shadow_fraction": MAX_WHOLE_STORE_FRACTION,
                "reduction": reductions, "cutover_gate_passed": whole_store_passed},
            "projection_indexes": projection,
            "cutover_gate_passed": whole_store_passed and projection["cutover_gate_passed"]}


def _blocks(source, digest):
    row = source.execute("SELECT rowid,length(payload_zlib) FROM artifacts WHERE sha256=?", (digest,)).fetchone()
    if row is None:
        raise ValueError("legacy artifact is missing: " + digest)
    inflater = zlib.decompressobj()
    with source.blobopen("artifacts", "payload_zlib", row[0], readonly=True) as blob:
        while blob.tell() < len(blob):
            pending = blob.read(min(1024 * 1024, len(blob) - blob.tell()))
            while pending:
                raw = inflater.decompress(pending, CHUNK_BYTES)
                if raw:
                    yield raw
                pending = inflater.unconsumed_tail
    raw = inflater.flush()
    if raw:
        yield raw
    if not inflater.eof or inflater.unused_data:
        raise ValueError("legacy artifact compression stream is incomplete")


def _rows(cursor, size=10000):
    while True:
        batch = cursor.fetchmany(size)
        if not batch:
            return
        yield batch


class ShadowMigration:
    def __init__(self, source_path, shadow_path, progress=print):
        self.source_path, self.shadow_path = Path(source_path), Path(shadow_path)
        if self.source_path.resolve() == self.shadow_path.resolve():
            raise ValueError("shadow database must differ from the active database")
        if not self.source_path.is_file():
            raise ValueError("source database does not exist")
        self.source = sqlite3.connect(self.source_path.resolve().as_uri() + "?mode=ro", uri=True)
        self.source.row_factory = sqlite3.Row
        if self.source.execute("PRAGMA user_version").fetchone()[0] != 1:
            raise ValueError("migration source must be schema 1")
        self.store = ProjectKnowledgeStore(self.shadow_path)
        self.shadow = self.store.connection
        if self.shadow.execute("PRAGMA user_version").fetchone()[0] != 2:
            raise ValueError("shadow database must be schema 2")
        self.progress = progress
        identity = {"source_path": str(self.source_path.resolve()),
                    "source_schema": 1,
                    "source_snapshots_sha256": self._fingerprint(self.source,
                        "SELECT id,project_id,document_id,source_sha256,context_json,manifest_json,created_at FROM snapshots ORDER BY id"),
                    "source_active_documents_sha256": self._fingerprint(self.source,
                        "SELECT project_id,document_id,snapshot_id FROM active_documents ORDER BY project_id,document_id"),
                    "source_snapshot_artifacts_sha256": self._fingerprint(self.source,
                        "SELECT snapshot_id,name,artifact_sha256 FROM snapshot_artifacts ORDER BY snapshot_id,name"),
                    "source_artifacts_sha256": self._fingerprint(self.source,
                        "SELECT sha256 FROM artifacts ORDER BY sha256"),
                    "source_reviews_sha256": self._fingerprint(self.source,
                        "SELECT id,snapshot_id,node_id,payload_json,created_at FROM reviews ORDER BY id")}
        prior = self._manifest("source")
        if prior is not None and prior != identity:
            # Early 0.2 shadows recorded only the snapshot fingerprint. Upgrade
            # that manifest once, but never excuse a disagreement in a field it
            # already froze.
            if any(identity.get(key) != value for key, value in prior.items()):
                raise ValueError("shadow manifest belongs to another source database state")
        self._put_manifest("source", identity)

    def close(self):
        self.store.__exit__(None, None, None)
        self.source.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _manifest(self, key):
        row = self.shadow.execute("SELECT value_json FROM migration_manifest WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def _put_manifest(self, key, value):
        with self.shadow:
            self.shadow.execute("INSERT INTO migration_manifest VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json",
                                (key, _json(value)))

    @staticmethod
    def _fingerprint(connection, sql, parameters=()):
        digest, count = hashlib.sha256(), 0
        for batch in _rows(connection.execute(sql, parameters)):
            for row in batch:
                digest.update(_json(list(row)).encode()); digest.update(b"\n")
                count += 1
        return {"rows": count, "sha256": digest.hexdigest()}

    def _snapshot_ids(self, phase):
        if phase == "active":
            return [row[0] for row in self.source.execute("SELECT DISTINCT snapshot_id FROM active_documents ORDER BY snapshot_id")]
        return [row[0] for row in self.source.execute("SELECT id FROM snapshots ORDER BY created_at,id")]

    def migrate(self, phase):
        if phase not in {"active", "history"}:
            raise ValueError("migration phase must be active or history")
        started = time.perf_counter()
        before_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        snapshot_ids = self._snapshot_ids(phase)
        for snapshot_id in snapshot_ids:
            self._migrate_snapshot(snapshot_id, copy_active=(phase == "active"))
        if phase == "history":
            self._copy_active_documents()
        report = {"phase": phase, "status": "complete", "snapshot_count": len(snapshot_ids),
                  "elapsed_seconds": time.perf_counter() - started,
                  "max_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "max_rss_before": before_rss,
                  "source_disk_bytes": _database_bytes(self.source_path),
                  "shadow_disk_bytes": _database_bytes(self.shadow_path)}
        self._put_manifest("phase:" + phase, report)
        self.progress(json.dumps(report, sort_keys=True))
        return report

    def estimate(self):
        """Estimate the completed shadow before another long write transaction."""
        projection = _dbstat_projection_bytes(self.shadow_path)
        storage = _storage_efficiency(self.source_path, self.shadow_path, projection=projection)
        source_counts = {table: self.source.execute("SELECT count(*) FROM " + table).fetchone()[0]
                         for table in ("artifacts", "nodes", "edges")}
        shadow_counts = {table: self.shadow.execute("SELECT count(*) FROM " + table).fetchone()[0]
                         for table in ("artifacts", "nodes", "edges")}
        missing = {table: source_counts[table] - shadow_counts[table] for table in source_counts}
        migrated_digests = [row[0] for row in self.shadow.execute("SELECT sha256 FROM artifacts")]
        migrated_set = set(migrated_digests)
        missing_digests = [row[0] for row in self.source.execute("SELECT sha256 FROM artifacts ORDER BY sha256")
                           if row[0] not in migrated_set]
        missing_compressed = sum(self.source.execute(
            "SELECT length(payload_zlib) FROM artifacts WHERE sha256=?", (digest,)).fetchone()[0]
            for digest in missing_digests)
        migrated_source_compressed = sum(self.source.execute(
            "SELECT length(payload_zlib) FROM artifacts WHERE sha256=?", (digest,)).fetchone()[0]
            for digest in migrated_digests)
        chunk_bytes = self.shadow.execute(
            "SELECT coalesce(sum(length(payload_zlib)),0) FROM artifact_chunks").fetchone()[0]
        chunk_ratio = chunk_bytes / migrated_source_compressed if migrated_source_compressed else 1.0
        projected_rows = shadow_counts["nodes"] + shadow_counts["edges"]
        projection_per_row = (projection.get("total_bytes", 0) / projected_rows
                              if projection.get("status") == "measured" and projected_rows else 0)
        added = int(missing_compressed * chunk_ratio +
                    (missing["nodes"] + missing["edges"]) * projection_per_row)
        current = storage["shadow_bytes"]
        central = {kind: current[kind] + added for kind in ("logical", "allocated")}
        target = {kind: int(storage["source_bytes"][kind] * MAX_WHOLE_STORE_FRACTION)
                  for kind in ("logical", "allocated")}
        free_bytes = shutil.disk_usage(self.shadow_path.parent).free
        report = {"status": "estimated",
                  "method": "observed chunk-compression and projection bytes per indexed row",
                  "source_counts": source_counts, "shadow_counts": shadow_counts,
                  "remaining_counts": missing, "remaining_artifact_sha256": missing_digests,
                  "current_shadow_bytes": current,
                  "estimated_final_bytes": central,
                  "estimated_final_range_bytes": {
                      kind: {"lower": max(current[kind], int(central[kind] * .90)),
                             "upper": int(central[kind] * 1.25)}
                      for kind in ("logical", "allocated")},
                  "maximum_whole_store_target_bytes": target,
                  "whole_store_target_already_impossible": any(current[kind] > target[kind]
                                                                 for kind in target),
                  "estimated_target_passed": all(central[kind] <= target[kind] for kind in target),
                  "projection_indexes": projection,
                  "capacity_now": {"free_bytes": free_bytes,
                                   "minimum_transaction_free_bytes": MINIMUM_TRANSACTION_FREE_BYTES,
                                   "minimum_floor_passed": free_bytes >= MINIMUM_TRANSACTION_FREE_BYTES},
                  "storage_efficiency_now": storage}
        self._put_manifest("estimate:final", report)
        return report

    def _migrate_snapshot(self, snapshot_id, *, copy_active):
        if self.shadow.execute("SELECT 1 FROM snapshots WHERE id=?", (snapshot_id,)).fetchone():
            return
        snapshot = self.source.execute("SELECT * FROM snapshots WHERE id=?", (snapshot_id,)).fetchone()
        if snapshot is None:
            raise ValueError("legacy snapshot disappeared during migration")
        artifacts = list(self.source.execute("SELECT name,artifact_sha256 FROM snapshot_artifacts WHERE snapshot_id=? ORDER BY name",
                                             (snapshot_id,)))
        for row in artifacts:
            names = [item[0] for item in self.source.execute("SELECT DISTINCT name FROM snapshot_artifacts WHERE artifact_sha256=? ORDER BY name",
                                                             (row["artifact_sha256"],))]
            self._migrate_artifact(row["artifact_sha256"], names)
        with self.shadow:
            self.shadow.execute("INSERT INTO snapshots VALUES(?,?,?,?,?,?,?)", tuple(snapshot))
            self.shadow.executemany("INSERT INTO snapshot_artifacts VALUES(?,?,?)",
                [(snapshot_id, row["name"], row["artifact_sha256"]) for row in artifacts])
            reviews = self.source.execute("SELECT * FROM reviews WHERE snapshot_id=? ORDER BY id", (snapshot_id,))
            self.shadow.executemany("INSERT INTO reviews VALUES(?,?,?,?,?)", [tuple(row) for row in reviews])
            if copy_active:
                active = self.source.execute("SELECT project_id,document_id,snapshot_id FROM active_documents WHERE snapshot_id=?",
                                             (snapshot_id,)).fetchall()
                self.shadow.executemany("INSERT INTO active_documents VALUES(?,?,?) ON CONFLICT(project_id,document_id) DO UPDATE SET snapshot_id=excluded.snapshot_id",
                                        [tuple(row) for row in active])
        self.progress(json.dumps({"phase": "snapshot_migrated", "snapshot_id": snapshot_id,
                                  "artifact_count": len(artifacts)}, sort_keys=True))

    def _migrate_artifact(self, digest, names):
        if self.shadow.execute("SELECT 1 FROM artifacts WHERE sha256=?", (digest,)).fetchone():
            return
        self._preflight_capacity(digest, names)
        started = time.perf_counter()
        keep_inline = not set(names).intersection(CHUNK_ONLY_NAMES)
        self.shadow.execute("CREATE TEMP TABLE IF NOT EXISTS object_spans(pointer TEXT PRIMARY KEY,start INTEGER NOT NULL,length INTEGER NOT NULL) WITHOUT ROWID")
        self.shadow.execute("DELETE FROM object_spans")
        pending = []

        def span(pointer, start, length):
            pending.append((pointer, start, length))
            if len(pending) >= 10000:
                self.shadow.executemany("INSERT INTO object_spans VALUES(?,?,?)", pending)
                pending.clear()

        with self.shadow:
            self.shadow.execute("INSERT INTO artifacts(sha256,payload_zlib,byte_count,chunk_count) VALUES(?,NULL,NULL,NULL)", (digest,))
            byte_count, chunk_count = _artifact_chunks(self.shadow, digest, _blocks(self.source, digest), span_callback=span)
            if pending:
                self.shadow.executemany("INSERT INTO object_spans VALUES(?,?,?)", pending)
                pending.clear()
            self.shadow.execute("UPDATE artifacts SET byte_count=?,chunk_count=? WHERE sha256=?", (byte_count, chunk_count, digest))
            source_nodes = self.source.execute("SELECT id,artifact_sha256,pointer,source_id,record_type,page_ref,state,payload_json FROM nodes WHERE artifact_sha256=? ORDER BY pointer",
                                               (digest,))
            node_count = 0
            for batch in _rows(source_nodes):
                pointers = [row["pointer"] for row in batch]
                spans = {row[0]:(row[1],row[2]) for row in self.shadow.execute(
                    "SELECT pointer,start,length FROM object_spans WHERE pointer IN (" + ",".join("?" for _ in pointers) + ")", pointers)}
                if len(spans) != len(batch):
                    raise ValueError("canonical JSON spans do not cover every indexed node")
                values, fts = [], []
                for row in batch:
                    start, length = spans[row["pointer"]]
                    inline = row["payload_json"] if keep_inline else None
                    values.append((*tuple(row)[:7], inline, start, length))
                    if row["source_id"] and row["payload_json"] is not None and len(row["payload_json"]) <= 65536:
                        fts.append((row["id"], row["source_id"], row["record_type"], row["page_ref"], row["state"], row["payload_json"]))
                self.shadow.executemany("INSERT INTO nodes VALUES(?,?,?,?,?,?,?,?,?,?)", values)
                _insert_fts(self.shadow, fts)
                node_count += len(batch)
            edges = self.source.execute("SELECT e.node_id,e.field,e.target_ref FROM edges e JOIN nodes n ON n.id=e.node_id WHERE n.artifact_sha256=? ORDER BY e.node_id,e.field,e.target_ref",
                                        (digest,))
            edge_count = 0
            for batch in _rows(edges):
                self.shadow.executemany("INSERT INTO edges VALUES(?,?,?)", [tuple(row) for row in batch])
                edge_count += len(batch)
        report = {"sha256": digest, "names": names, "bytes": byte_count, "chunks": chunk_count,
                  "nodes": node_count, "edges": edge_count, "inline_payloads_retained": keep_inline,
                  "elapsed_seconds": time.perf_counter() - started}
        self._put_manifest("artifact:" + digest, report)
        self.progress(json.dumps({"phase": "artifact_migrated", **report}, sort_keys=True))

    def _preflight_capacity(self, digest, names):
        """Reserve room for B-tree page rewrites as well as new row content."""
        artifact = self.source.execute("SELECT length(payload_zlib) FROM artifacts WHERE sha256=?", (digest,)).fetchone()[0]
        node = self.source.execute("SELECT count(*),coalesce(sum(length(id)+length(pointer)+length(coalesce(source_id,''))+length(record_type)+length(coalesce(page_ref,''))+length(coalesce(state,''))+length(coalesce(payload_json,''))),0) FROM nodes WHERE artifact_sha256=?",
                                   (digest,)).fetchone()
        edge = self.source.execute("SELECT count(*),coalesce(sum(length(e.node_id)+length(e.field)+length(e.target_ref)),0) FROM edges e JOIN nodes n ON n.id=e.node_id WHERE n.artifact_sha256=?",
                                   (digest,)).fetchone()
        indexed_content = artifact + node[1] + edge[1] + node[0] * 96 + edge[0] * 64
        estimated_growth = int(indexed_content * 1.5)
        database_size = self.shadow_path.stat().st_size if self.shadow_path.exists() else 0
        btree_rewrite_reserve = int(database_size * .65)
        minimum_remaining = MINIMUM_TRANSACTION_FREE_BYTES
        required_free = minimum_remaining + estimated_growth + btree_rewrite_reserve
        free = shutil.disk_usage(self.shadow_path.parent).free
        report = {"artifact_sha256": digest, "names": names, "free_bytes": free,
                  "required_free_bytes": required_free, "minimum_remaining_bytes": minimum_remaining,
                  "estimated_growth_bytes": estimated_growth,
                  "btree_rewrite_reserve_bytes": btree_rewrite_reserve}
        if free < required_free:
            report["status"] = "blocked_before_transaction"
            self._put_manifest("capacity:block", report)
            raise RuntimeError("insufficient free space for fail-closed artifact transaction: " + _json(report))

    def _copy_active_documents(self):
        rows = [tuple(row) for row in self.source.execute("SELECT * FROM active_documents ORDER BY project_id,document_id")]
        with self.shadow:
            self.shadow.execute("DELETE FROM active_documents")
            self.shadow.executemany("INSERT INTO active_documents VALUES(?,?,?)", rows)

    def verify(self, scope):
        if scope not in {"active", "all"}:
            raise ValueError("verification scope must be active or all")
        started = time.perf_counter()
        ids = self._snapshot_ids("active" if scope == "active" else "history")
        checks = []
        for snapshot_id in ids:
            parameters = (snapshot_id,)
            query_sql = "SELECT n.id,sa.name,n.artifact_sha256,n.pointer,n.source_id,n.record_type,n.page_ref,n.state FROM nodes n JOIN snapshot_artifacts sa ON sa.artifact_sha256=n.artifact_sha256 WHERE sa.snapshot_id=? ORDER BY n.id,sa.name"
            edge_sql = "SELECT e.node_id,e.field,e.target_ref FROM edges e JOIN nodes n ON n.id=e.node_id JOIN snapshot_artifacts sa ON sa.artifact_sha256=n.artifact_sha256 WHERE sa.snapshot_id=? ORDER BY e.node_id,e.field,e.target_ref"
            review_sql = "SELECT id,snapshot_id,node_id,payload_json,created_at FROM reviews WHERE snapshot_id=? ORDER BY id"
            source_query, shadow_query = self._fingerprint(self.source, query_sql, parameters), self._fingerprint(self.shadow, query_sql, parameters)
            source_edges, shadow_edges = self._fingerprint(self.source, edge_sql, parameters), self._fingerprint(self.shadow, edge_sql, parameters)
            source_reviews, shadow_reviews = self._fingerprint(self.source, review_sql, parameters), self._fingerprint(self.shadow, review_sql, parameters)
            artifacts = list(self.source.execute("SELECT name,artifact_sha256 FROM snapshot_artifacts WHERE snapshot_id=? ORDER BY name", parameters))
            artifact_digests = {row["name"]: row["artifact_sha256"] for row in artifacts}
            export_rows = []
            snapshot = self.source.execute("SELECT project_id,document_id FROM snapshots WHERE id=?", parameters).fetchone()
            for artifact in artifacts:
                digest, size = hashlib.sha256(), 0
                for block in self.store.iter_artifact_bytes(project_id=snapshot[0], document_id=snapshot[1],
                        snapshot_id=snapshot_id, name=artifact["name"]):
                    digest.update(block); size += len(block)
                export_rows.append((artifact["name"], digest.hexdigest(), size))
            export_ok = all(digest == artifact_digests[name] for name, digest, size in export_rows)
            audit_artifacts = sorted(set(row["name"] for row in artifacts).intersection(AUDIT_NAMES))
            audit = self._audit_parity(snapshot_id, snapshot[0], snapshot[1], set(artifact_digests))
            check = {"snapshot_id": snapshot_id,
                     "query": {"source": source_query, "shadow": shadow_query, "parity": source_query == shadow_query},
                     "neighbors": {"source": source_edges, "shadow": shadow_edges, "parity": source_edges == shadow_edges},
                     "reviews": {"source": source_reviews, "shadow": shadow_reviews, "parity": source_reviews == shadow_reviews},
                     "canonical_export": {"artifacts": len(export_rows), "parity": export_ok},
                     "audit": {"artifact_names": audit_artifacts, **audit}}
            check["parity"] = all(check[key]["parity"] for key in ("query", "neighbors", "reviews", "canonical_export", "audit"))
            checks.append(check)
            self.progress(json.dumps({"phase": "snapshot_verified", **check}, sort_keys=True))
        complete = self._completeness(scope)
        projection = (_dbstat_projection_bytes(self.shadow_path) if scope == "all" else
                      {"status": "not_evaluated_for_active_scope",
                       "maximum_bytes": MAX_PROJECTION_INDEX_BYTES, "cutover_gate_passed": False})
        storage = _storage_efficiency(self.source_path, self.shadow_path, projection=projection)
        report = {"schema_version": "0.2.0", "layer": "project_store_shadow_migration_manifest",
                  "scope": scope, "snapshots": checks, "completeness": complete,
                  "all_parity": bool(checks) and all(row["parity"] for row in checks),
                  "vector_retrieval": {"status": "not_implemented", "role": "proposal_only_sidecar_after_fts5_evaluation"},
                  "fts5": {"enabled": True, "engineering_authority_inferred": False},
                  "elapsed_seconds": time.perf_counter() - started,
                  "source_disk_bytes": _database_bytes(self.source_path),
                  "shadow_disk_bytes": _database_bytes(self.shadow_path),
                  "storage_efficiency": storage}
        report["correctness_gate_passed"] = scope == "all" and report["all_parity"] and complete["parity"]
        report["cutover_ready"] = report["correctness_gate_passed"] and storage["cutover_gate_passed"]
        report["manifest_sha256"] = hashlib.sha256(_json(report).encode()).hexdigest()
        self._put_manifest("verification:" + scope, report)
        return report

    def _audit_parity(self, snapshot_id, project_id, document_id, artifact_names):
        required = {"sheet-registry", "network-hierarchy", "hvac-inventory", "item-catalog"}
        if not required.issubset(artifact_names):
            return {"status": "not_applicable", "parity": True,
                    "reason": "snapshot lacks the complete inspection projection inputs"}
        from src.drawing_engine.project.mep_project_review import load_review_page

        def projection(path, selected=None):
            value = load_review_page(path, project=project_id, document=document_id,
                                     snapshot=snapshot_id, selected=selected, limit=10)
            return {key:value.get(key) for key in ("rows", "selected", "pagination", "processing_scope",
                "network_summary", "hvac_summary", "outcome_count", "source_sha256")}

        legacy = projection(self.source_path)
        shadow = projection(self.shadow_path)
        selected = legacy["rows"][0]["id"] if legacy["rows"] else None
        legacy_selected = projection(self.source_path, selected) if selected else None
        shadow_selected = projection(self.shadow_path, selected) if selected else None
        legacy_hash = hashlib.sha256(_json([legacy, legacy_selected]).encode()).hexdigest()
        shadow_hash = hashlib.sha256(_json([shadow, shadow_selected]).encode()).hexdigest()
        return {"status": "compared", "list_and_selected_sha256": {"source": legacy_hash, "shadow": shadow_hash},
                "parity": legacy_hash == shadow_hash, "selected_source_id": selected}

    def _completeness(self, scope):
        tables = ("snapshots", "snapshot_artifacts", "artifacts", "nodes", "edges", "reviews")
        result = {}
        for table in tables:
            source = self.source.execute("SELECT count(*) FROM " + table).fetchone()[0]
            shadow = self.shadow.execute("SELECT count(*) FROM " + table).fetchone()[0]
            result[table] = {"source": source, "shadow": shadow,
                             "parity": source == shadow if scope == "all" else shadow <= source}
        result["parity"] = all(row["parity"] for row in result.values())
        return result

    def cutover(self, backup):
        report = self._manifest("verification:all")
        if not report or not report.get("cutover_ready"):
            raise ValueError("complete parity and whole-store efficiency manifest is required before cutover")
        self.shadow.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        current_storage = _storage_efficiency(self.source_path, self.shadow_path)
        if not current_storage["cutover_gate_passed"]:
            raise ValueError("current whole-store or projection-index efficiency no longer permits cutover")
        backup = Path(backup)
        if backup.exists():
            raise ValueError("cutover backup path already exists")
        self.source.close()
        self.store.__exit__(None, None, None)
        with sqlite3.connect(self.source_path) as connection:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        with sqlite3.connect(self.shadow_path) as connection:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        os.replace(self.source_path, backup)
        os.replace(self.shadow_path, self.source_path)
        return {"cutover": "complete", "active_database": str(self.source_path),
                "recoverable_schema_1_backup": str(backup), "manifest_sha256": report["manifest_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("active", "history", "estimate", "verify-active", "verify-all", "cutover"))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--shadow", required=True, type=Path)
    parser.add_argument("--backup", type=Path)
    args = parser.parse_args()
    with ShadowMigration(args.source, args.shadow) as migration:
        if args.command in {"active", "history"}:
            result = migration.migrate(args.command)
        elif args.command == "estimate":
            result = migration.estimate()
        elif args.command.startswith("verify"):
            result = migration.verify("active" if args.command == "verify-active" else "all")
        else:
            if args.backup is None:
                parser.error("cutover requires --backup")
            result = migration.cutover(args.backup)
        print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
