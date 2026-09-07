"""Bounded pilot for packing dense evidence records without losing provenance.

The production project store deliberately remains unchanged.  This module
tests one narrow storage policy: records below ``source_rows`` are kept as
compressed descriptor packs instead of one SQLite node and several edge rows
per record.  Descriptors retain the production node ID, JSON pointer, query
fields, and outgoing references.  Payloads are always resolved from the exact,
content-addressed artifact body.

This is an artifact-level pilot, not a snapshot store or migration tool.
"""

from contextlib import AbstractContextManager
import hashlib
import heapq
import itertools
import json
from pathlib import Path
import sqlite3
import zlib

from src.drawing_engine.project.project_knowledge_store import _hash, _json, _records, _references


_DENSE_COLLECTION_FIELDS = frozenset({"source_rows"})
_BLOOM_BITS = 16384
_BLOOM_HASHES = 4


def _page_ref(record, inherited):
    source = record.get("source")
    return record.get("page_ref") or (source.get("page_ref") if isinstance(source, dict) else None) or inherited


def _descriptor(digest, pointer, page_ref, record):
    return {
        "id": _hash([digest, pointer]),
        "artifact_sha256": digest,
        "pointer": pointer,
        "source_id": record.get("id"),
        "record_type": record.get("record_type") or record.get("entity_type") or pointer.split("/")[-2],
        "page_ref": page_ref,
        "state": record.get("state") or record.get("epistemic_state"),
        "references": _references(record),
    }


def _walk_regular(value, digest, pointer="", page_ref=None):
    """Yield ordinary descriptors and dense collection roots separately."""
    if isinstance(value, dict):
        page_ref = _page_ref(value, page_ref)
        if pointer and (isinstance(value.get("id"), str) or value.get("record_type") or
                        (value.get("page_ref") and "pages/" in pointer)):
            yield "record", _descriptor(digest, pointer, page_ref, value)
        for key, child in value.items():
            child_pointer = pointer + "/" + str(key).replace("~", "~0").replace("/", "~1")
            if key in _DENSE_COLLECTION_FIELDS and isinstance(child, list):
                yield "dense", (child, child_pointer, page_ref)
            else:
                yield from _walk_regular(child, digest, child_pointer, page_ref)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_regular(child, digest, pointer + "/" + str(index), page_ref)


def _bloom(values):
    bits = bytearray(_BLOOM_BITS // 8)
    for value in values:
        if not value:
            continue
        digest = hashlib.sha256(value.encode()).digest()
        for index in range(_BLOOM_HASHES):
            position = int.from_bytes(digest[index * 4:index * 4 + 4], "big") % _BLOOM_BITS
            bits[position // 8] |= 1 << (position % 8)
    return bytes(bits)


def _maybe_contains(bits, value):
    if not value:
        return True
    digest = hashlib.sha256(value.encode()).digest()
    return all(bits[position // 8] & (1 << (position % 8)) for position in (
        int.from_bytes(digest[index * 4:index * 4 + 4], "big") % _BLOOM_BITS
        for index in range(_BLOOM_HASHES)))


class PackedEvidenceIndexPilot(AbstractContextManager):
    """Small SQLite pilot proving packed evidence remains exactly queryable."""

    def __init__(self, path, *, pack_records=2048):
        if pack_records < 1:
            raise ValueError("pack_records must be positive")
        self.pack_records = pack_records
        self.connection = sqlite3.connect(Path(path))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS pilot_artifacts (
                sha256 TEXT PRIMARY KEY, payload_zlib BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS pilot_artifact_names (
                name TEXT PRIMARY KEY, artifact_sha256 TEXT NOT NULL
                    REFERENCES pilot_artifacts(sha256));
            CREATE TABLE IF NOT EXISTS pilot_nodes (
                id TEXT PRIMARY KEY, artifact_sha256 TEXT NOT NULL,
                pointer TEXT NOT NULL, source_id TEXT, record_type TEXT NOT NULL,
                page_ref TEXT, state TEXT, references_json TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS pilot_nodes_source
                ON pilot_nodes(source_id, page_ref);
            CREATE TABLE IF NOT EXISTS pilot_dense_packs (
                id TEXT PRIMARY KEY, artifact_sha256 TEXT NOT NULL,
                collection_pointer TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
                payload_zlib BLOB NOT NULL, record_count INTEGER NOT NULL,
                source_bloom BLOB NOT NULL, node_bloom BLOB NOT NULL,
                page_refs_json TEXT NOT NULL, record_types_json TEXT NOT NULL,
                states_json TEXT NOT NULL, metadata_sha256 TEXT NOT NULL);
        """)

    def __exit__(self, *args):
        self.connection.close()

    def import_artifact(self, name, artifact):
        """Add one immutable artifact; a name cannot silently change content."""
        if not name or not isinstance(artifact, dict):
            raise ValueError("name and artifact dictionary are required")
        digest = _hash(artifact)
        existing = self.connection.execute(
            "SELECT artifact_sha256 FROM pilot_artifact_names WHERE name=?", (name,)).fetchone()
        if existing:
            if existing[0] != digest:
                raise ValueError("artifact name already identifies different content")
            return digest
        encoded = _json(artifact).encode()
        with self.connection:
            artifact_exists = self.connection.execute(
                "SELECT 1 FROM pilot_artifacts WHERE sha256=?", (digest,)).fetchone()
            self.connection.execute("INSERT OR IGNORE INTO pilot_artifacts VALUES(?,?)",
                                    (digest, zlib.compress(encoded, level=1)))
            if not artifact_exists:
                for kind, value in _walk_regular(artifact, digest):
                    if kind == "record":
                        self._insert_node(value)
                        continue
                    rows, pointer, inherited_page = value
                    chunk = []
                    for row_pointer, page_ref, record in _records(rows, pointer, inherited_page):
                        chunk.append(_descriptor(digest, row_pointer, page_ref, record))
                        if len(chunk) == self.pack_records:
                            self._insert_pack(digest, pointer, chunk)
                            chunk = []
                    if chunk:
                        self._insert_pack(digest, pointer, chunk)
            self.connection.execute("INSERT INTO pilot_artifact_names VALUES(?,?)", (name, digest))
        return digest

    def _insert_node(self, row):
        self.connection.execute("INSERT INTO pilot_nodes VALUES(?,?,?,?,?,?,?,?)", (
            row["id"], row["artifact_sha256"], row["pointer"], row["source_id"],
            row["record_type"], row["page_ref"], row["state"], _json(row["references"])))

    def _insert_pack(self, digest, pointer, rows):
        body = _json(rows).encode()
        body_hash = hashlib.sha256(body).hexdigest()
        pack_id = _hash([digest, pointer, rows[0]["pointer"], rows[-1]["pointer"], body_hash])
        source_bloom = _bloom(row["source_id"] for row in rows)
        node_bloom = _bloom(row["id"] for row in rows)
        pages = _json(sorted({row["page_ref"] for row in rows if row["page_ref"]}))
        types = _json(sorted({row["record_type"] for row in rows}))
        states = _json(sorted({row["state"] for row in rows if row["state"]}))
        metadata_hash = _hash([digest, pointer, body_hash, len(rows), source_bloom.hex(), node_bloom.hex(),
                               pages, types, states])
        self.connection.execute("INSERT INTO pilot_dense_packs VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (
            pack_id, digest, pointer, body_hash, zlib.compress(body, level=1), len(rows),
            source_bloom, node_bloom, pages, types, states, metadata_hash))

    @staticmethod
    def _verify_pack_metadata(pack):
        actual = _hash([pack["artifact_sha256"], pack["collection_pointer"],
                        pack["payload_sha256"], pack["record_count"],
                        pack["source_bloom"].hex(), pack["node_bloom"].hex(),
                        pack["page_refs_json"], pack["record_types_json"], pack["states_json"]])
        if actual != pack["metadata_sha256"]:
            raise ValueError("dense evidence pack metadata hash changed")

    @staticmethod
    def _decode_pack(row):
        body = zlib.decompress(row["payload_zlib"])
        if hashlib.sha256(body).hexdigest() != row["payload_sha256"]:
            raise ValueError("dense evidence pack content hash changed")
        return json.loads(body)

    @staticmethod
    def _matches(row, filters):
        return all(value is None or row[field] == value for field, value in filters.items())

    def _packed_candidates(self, filters, *, node_id=None):
        for pack in self.connection.execute("SELECT * FROM pilot_dense_packs ORDER BY id"):
            self._verify_pack_metadata(pack)
            if filters.get("source_id") and not _maybe_contains(pack["source_bloom"], filters["source_id"]):
                continue
            if node_id and not _maybe_contains(pack["node_bloom"], node_id):
                continue
            if filters.get("page_ref") and filters["page_ref"] not in json.loads(pack["page_refs_json"]):
                continue
            if filters.get("record_type") and filters["record_type"] not in json.loads(pack["record_types_json"]):
                continue
            if filters.get("state") and filters["state"] not in json.loads(pack["states_json"]):
                continue
            for row in self._decode_pack(pack):
                if (node_id is None or row["id"] == node_id) and self._matches(row, filters):
                    yield row

    def query(self, *, source_id=None, record_type=None, page_ref=None, state=None, limit=100, offset=0):
        """Query regular and packed records with deterministic bounded paging."""
        if not 1 <= limit <= 1000 or offset < 0:
            raise ValueError("query limit must be 1..1000 and offset nonnegative")
        filters = {"source_id": source_id, "record_type": record_type,
                   "page_ref": page_ref, "state": state}
        clauses, parameters = [], []
        for field, value in filters.items():
            if value is not None:
                clauses.append(field + "=?")
                parameters.append(value)
        sql = "SELECT * FROM pilot_nodes"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        regular = (self._row_descriptor(row) for row in self.connection.execute(sql, parameters))
        rows = heapq.nsmallest(offset + limit,
                               itertools.chain(regular, self._packed_candidates(filters)),
                               key=lambda row: row["id"])
        return rows[offset:offset + limit]

    @staticmethod
    def _row_descriptor(row):
        result = dict(row)
        result["references"] = json.loads(result.pop("references_json"))
        return result

    def _node(self, node_id):
        row = self.connection.execute("SELECT * FROM pilot_nodes WHERE id=?", (node_id,)).fetchone()
        if row:
            return self._row_descriptor(row)
        rows = list(self._packed_candidates({"source_id": None, "record_type": None,
                                             "page_ref": None, "state": None}, node_id=node_id))
        if len(rows) != 1:
            raise KeyError(node_id)
        return rows[0]

    def artifact(self, name):
        row = self.connection.execute(
            "SELECT a.sha256,a.payload_zlib FROM pilot_artifacts a JOIN pilot_artifact_names n "
            "ON n.artifact_sha256=a.sha256 WHERE n.name=?", (name,)).fetchone()
        if row is None:
            raise KeyError(name)
        artifact = json.loads(zlib.decompress(row["payload_zlib"]))
        if _hash(artifact) != row["sha256"]:
            raise ValueError("artifact content hash changed")
        return artifact

    def node_payload(self, node_id):
        node = self._node(node_id)
        row = self.connection.execute("SELECT payload_zlib FROM pilot_artifacts WHERE sha256=?",
                                      (node["artifact_sha256"],)).fetchone()
        artifact = json.loads(zlib.decompress(row[0]))
        if _hash(artifact) != node["artifact_sha256"]:
            raise ValueError("artifact content hash changed")
        value = artifact
        for token in node["pointer"].split("/")[1:]:
            key = token.replace("~1", "/").replace("~0", "~")
            value = value[int(key)] if isinstance(value, list) else value[key]
        return value

    def neighbors(self, node_id):
        node = self._node(node_id)
        output = []
        for field, target_ref in node["references"]:
            raw = target_ref.startswith(("drawing[", "page["))
            matches = self.query(source_id=target_ref, page_ref=node["page_ref"] if raw else None,
                                 limit=1000)
            output.append({"field": field, "target_ref": target_ref,
                           "resolution": "unresolved" if not matches else
                           "unique_record" if len(matches) == 1 else "multiple_records",
                           "targets": matches, "engineering_authority_inferred": False})
        return output

    def stats(self):
        regular = self.connection.execute("SELECT COUNT(*) FROM pilot_nodes").fetchone()[0]
        packs, dense = self.connection.execute(
            "SELECT COUNT(*),COALESCE(SUM(record_count),0) FROM pilot_dense_packs").fetchone()
        return {"regular_records": regular, "dense_logical_records": dense,
                "pack_rows": packs, "relational_rows": regular + packs,
                "logical_records": regular + dense, "edge_rows": 0}
