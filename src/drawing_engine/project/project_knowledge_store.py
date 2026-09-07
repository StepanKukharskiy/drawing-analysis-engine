"""Durable, content-addressed project snapshots and evidence-reference queries.

SQLite stores immutable artifact bodies as well as an index. A node is an
artifact record, not a newly inferred physical entity. Multiple representations
of one source ID remain separate; reference traversal exposes every alternative.
The active snapshot pointer moves only after a complete transactional import.
"""

from contextlib import AbstractContextManager
import hashlib
import json
from pathlib import Path
import sqlite3
import zlib


VERSION = "0.2.0"
SCHEMA_VERSION = 2
CHUNK_BYTES = 4 * 1024 * 1024
# Exhaustive search membership is evidence, not a relation to an accepted
# target. Preserve these potentially millions of refs in the immutable body;
# the search-scope node/pointer remains indexed without duplicating every list.
_ARTIFACT_ONLY_REFERENCE_FIELDS = {
    "primitive_candidate_refs", "cross_boundary_primitive_refs", "unsupported_native_item_refs",
}
_FTS_TEXT_FIELDS = {"text", "raw_text", "label", "description", "tag", "name", "title",
                    "mark", "system", "item_class", "item_type"}


def _json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _fts_text(payload_json):
    if not payload_json:
        return ""
    value = json.loads(payload_json)
    output = []

    def walk(item, key=None):
        if isinstance(item, dict):
            for child_key, child in item.items():
                if child_key in _FTS_TEXT_FIELDS:
                    walk(child, child_key)
                elif isinstance(child, (dict, list)):
                    walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child, key)
        elif key in _FTS_TEXT_FIELDS and isinstance(item, (str, int, float)):
            output.append(str(item))
    walk(value)
    return " ".join(output)


def _insert_fts(connection, rows):
    rows = [row for row in rows if row[1]]
    if not rows:
        return
    connection.executemany("INSERT OR IGNORE INTO fts_nodes(node_id) VALUES(?)", [(row[0],) for row in rows])
    identifiers = [row[0] for row in rows]
    mapping = {row[1]: row[0] for row in connection.execute(
        "SELECT rowid,node_id FROM fts_nodes WHERE node_id IN (" + ",".join("?" for _ in identifiers) + ")",
        identifiers)}
    connection.executemany("INSERT INTO node_fts(rowid,source_id,record_type,page_ref,state,body) VALUES(?,?,?,?,?,?)",
        [(mapping[row[0]], row[1], row[2], row[3], row[4], _fts_text(row[5])) for row in rows])


def _escape(key):
    return str(key).replace("~", "~0").replace("/", "~1")


class _ObjectSpanScanner:
    """Incrementally locate every JSON object without materializing its parent.

    Project artifacts use the canonical JSON emitted by :func:`_json`.  Spans
    are byte offsets into that immutable representation, so a nested record can
    later be recovered from only the chunks intersecting its own body.
    """
    def __init__(self, callback):
        self.callback = callback
        self.stack = []
        self.offset = 0
        self.mode = "normal"
        self.token = bytearray()
        self.string_is_key = False
        self.escaped = False

    def _value_path(self):
        if not self.stack:
            return ""
        parent = self.stack[-1]
        if parent["kind"] == "object":
            if parent["state"] != "value":
                raise ValueError("invalid canonical JSON object value")
            return parent["path"] + "/" + _escape(parent["key"])
        if parent["state"] not in ("value", "value_or_end"):
            raise ValueError("invalid canonical JSON array value")
        return parent["path"] + "/" + str(parent["index"])

    def _complete_value(self):
        if not self.stack:
            return
        parent = self.stack[-1]
        if parent["kind"] == "object":
            if parent["state"] != "value":
                raise ValueError("invalid canonical JSON object state")
            parent["state"] = "comma_or_end"
        else:
            if parent["state"] not in ("value", "value_or_end"):
                raise ValueError("invalid canonical JSON array state")
            parent["state"] = "comma_or_end"

    def feed(self, data):
        index = 0
        while index < len(data):
            byte = data[index]
            absolute = self.offset + index
            if self.mode == "string":
                self.token.append(byte)
                if self.escaped:
                    self.escaped = False
                elif byte == 92:
                    self.escaped = True
                elif byte == 34:
                    value = json.loads(self.token.decode("ascii"))
                    self.mode = "normal"
                    if self.string_is_key:
                        frame = self.stack[-1]
                        frame["key"] = value
                        frame["state"] = "colon"
                    else:
                        self._complete_value()
                    self.token.clear()
                index += 1
                continue
            if self.mode == "scalar":
                if byte not in b",]}":
                    self.token.append(byte)
                    index += 1
                    continue
                self.mode = "normal"
                self.token.clear()
                self._complete_value()
                continue  # Reprocess the delimiter.
            if byte in b" \t\r\n":
                index += 1
                continue
            if byte == 34:
                self.mode = "string"
                self.token[:] = b'"'
                self.string_is_key = bool(self.stack and self.stack[-1]["kind"] == "object"
                                          and self.stack[-1]["state"] == "key_or_end")
                index += 1
                continue
            if byte in (123, 91):  # { or [
                path = self._value_path()
                self.stack.append({"kind": "object" if byte == 123 else "array", "path": path,
                    "start": absolute, "state": "key_or_end" if byte == 123 else "value_or_end",
                    "key": None, "index": 0})
                index += 1
                continue
            if byte in (125, 93):  # } or ]
                if not self.stack:
                    raise ValueError("unbalanced canonical JSON")
                frame = self.stack.pop()
                if (byte == 125) != (frame["kind"] == "object"):
                    raise ValueError("mismatched canonical JSON container")
                if frame["kind"] == "object":
                    self.callback(frame["path"], frame["start"], absolute + 1 - frame["start"])
                self._complete_value()
                index += 1
                continue
            frame = self.stack[-1] if self.stack else None
            if byte == 58:  # :
                if not frame or frame["kind"] != "object" or frame["state"] != "colon":
                    raise ValueError("invalid canonical JSON colon")
                frame["state"] = "value"
                index += 1
                continue
            if byte == 44:  # ,
                if not frame or frame["state"] != "comma_or_end":
                    raise ValueError("invalid canonical JSON comma")
                if frame["kind"] == "object":
                    frame["state"] = "key_or_end"
                    frame["key"] = None
                else:
                    frame["index"] += 1
                    frame["state"] = "value"
                index += 1
                continue
            self.mode = "scalar"
            self.token[:] = bytes((byte,))
            index += 1
        self.offset += len(data)

    def finish(self):
        if self.mode == "scalar":
            self._complete_value()
            self.mode = "normal"
        if self.mode != "normal" or self.stack:
            raise ValueError("truncated canonical JSON")


def _artifact_chunks(connection, digest, blocks, *, span_callback=None, chunk_bytes=CHUNK_BYTES):
    """Store exact canonical bytes in bounded, independently verified chunks."""
    pending = bytearray()
    offset = 0
    ordinal = 0
    body_hash = hashlib.sha256()
    scanner = _ObjectSpanScanner(span_callback or (lambda *_: None))

    def flush(size):
        nonlocal offset, ordinal
        raw = bytes(pending[:size])
        del pending[:size]
        connection.execute("INSERT INTO artifact_chunks VALUES(?,?,?,?,?,?)",
            (digest, ordinal, offset, len(raw), hashlib.sha256(raw).hexdigest(),
             zlib.compress(raw, level=1)))
        offset += len(raw)
        ordinal += 1

    for block in blocks:
        if not block:
            continue
        body_hash.update(block)
        scanner.feed(block)
        pending.extend(block)
        while len(pending) >= chunk_bytes:
            flush(chunk_bytes)
    scanner.finish()
    if pending or ordinal == 0:
        flush(len(pending))
    if body_hash.hexdigest() != digest:
        raise ValueError("artifact content hash changed while chunking")
    return offset, ordinal


def _chunk_bytes(connection, digest, start, length):
    end = start + length
    output = bytearray()
    rows = connection.execute("SELECT uncompressed_offset,uncompressed_size,payload_sha256,payload_zlib "
        "FROM artifact_chunks WHERE artifact_sha256=? AND uncompressed_offset<? "
        "AND uncompressed_offset+uncompressed_size>? ORDER BY ordinal", (digest, end, start))
    for row in rows:
        raw = zlib.decompress(row["payload_zlib"])
        if len(raw) != row["uncompressed_size"] or hashlib.sha256(raw).hexdigest() != row["payload_sha256"]:
            raise ValueError("artifact chunk content hash changed")
        left = max(start, row["uncompressed_offset"]) - row["uncompressed_offset"]
        right = min(end, row["uncompressed_offset"] + len(raw)) - row["uncompressed_offset"]
        output.extend(raw[left:right])
    if len(output) != length:
        raise ValueError("artifact chunk coverage is incomplete")
    return bytes(output)


def _records(value, pointer="", page_ref=None):
    if isinstance(value, dict):
        source = value.get("source")
        page_ref = value.get("page_ref") or (source.get("page_ref") if isinstance(source, dict) else None) or page_ref
        if pointer and (isinstance(value.get("id"), str) or value.get("record_type") or
                        (value.get("page_ref") and "pages/" in pointer)):
            yield pointer, page_ref, value
        for key, child in value.items():
            yield from _records(child, pointer + "/" + _escape(key), page_ref)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _records(child, pointer + "/" + str(index), page_ref)


def _references(row):
    def walk(value, path=""):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in _ARTIFACT_ONLY_REFERENCE_FIELDS:
                    continue
                field = path + "/" + _escape(key)
                if key.endswith(("_ref", "_refs")) or key in {"from", "to"}:
                    refs = child if isinstance(child, list) else [child]
                    for ref in refs:
                        if isinstance(ref, str) and ref:
                            yield field, ref
                # Child records own their own reference edges.
                if not (isinstance(child, dict) and child.get("id")):
                    yield from walk(child, field)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                if not (isinstance(child, dict) and child.get("id")):
                    yield from walk(child, path + "/" + str(i))
    return sorted(set(walk(row)))


class ProjectKnowledgeStore(AbstractContextManager):
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1, 2):
            self.connection.close()
            raise ValueError("unsupported project database schema")
        # A long import must not block readers of the previous frozen snapshot.
        self.connection.execute("PRAGMA journal_mode=WAL")
        if version in (1, 2):
            return  # Opening a reader must not run schema-write pragmas.
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id TEXT PRIMARY KEY, project_id TEXT NOT NULL, document_id TEXT NOT NULL,
                source_sha256 TEXT NOT NULL, context_json TEXT NOT NULL,
                manifest_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS active_documents (
                project_id TEXT NOT NULL, document_id TEXT NOT NULL,
                snapshot_id TEXT NOT NULL REFERENCES snapshots(id), PRIMARY KEY(project_id, document_id));
            CREATE TABLE IF NOT EXISTS artifacts (
                sha256 TEXT PRIMARY KEY, payload_zlib BLOB,
                byte_count INTEGER, chunk_count INTEGER);
            CREATE TABLE IF NOT EXISTS artifact_chunks (
                artifact_sha256 TEXT NOT NULL REFERENCES artifacts(sha256),
                ordinal INTEGER NOT NULL, uncompressed_offset INTEGER NOT NULL,
                uncompressed_size INTEGER NOT NULL, payload_sha256 TEXT NOT NULL,
                payload_zlib BLOB NOT NULL,
                PRIMARY KEY(artifact_sha256, ordinal)) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS snapshot_artifacts (
                snapshot_id TEXT NOT NULL REFERENCES snapshots(id), name TEXT NOT NULL,
                artifact_sha256 TEXT NOT NULL REFERENCES artifacts(sha256),
                PRIMARY KEY(snapshot_id, name));
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY, artifact_sha256 TEXT NOT NULL REFERENCES artifacts(sha256),
                pointer TEXT NOT NULL, source_id TEXT, record_type TEXT NOT NULL,
                page_ref TEXT, state TEXT, payload_json TEXT,
                payload_start INTEGER, payload_length INTEGER,
                UNIQUE(artifact_sha256, pointer)) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS nodes_source ON nodes(source_id, page_ref);
            CREATE INDEX IF NOT EXISTS nodes_kind ON nodes(record_type, state, page_ref);
            CREATE TABLE IF NOT EXISTS edges (
                node_id TEXT NOT NULL REFERENCES nodes(id), field TEXT NOT NULL,
                target_ref TEXT NOT NULL, PRIMARY KEY(node_id, field, target_ref)) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS edges_target ON edges(target_ref);
            CREATE TABLE IF NOT EXISTS reviews (
                id TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL REFERENCES snapshots(id),
                node_id TEXT NOT NULL REFERENCES nodes(id), payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS fts_nodes (
                rowid INTEGER PRIMARY KEY, node_id TEXT NOT NULL UNIQUE REFERENCES nodes(id));
            CREATE VIRTUAL TABLE IF NOT EXISTS node_fts USING fts5(
                source_id, record_type, page_ref, state, body, content='',
                tokenize='unicode61');
            CREATE TABLE IF NOT EXISTS migration_manifest (
                key TEXT PRIMARY KEY, value_json TEXT NOT NULL) WITHOUT ROWID;
            PRAGMA user_version=2;
        """)

    def __exit__(self, *args):
        self.connection.close()

    def import_snapshot(self, *, project_id, document_id, source_sha256, context, artifacts, progress=None):
        """Import a complete dependency set. New context/content supersedes the old set.

        Source bytes are verified by the caller; their digest is checked against
        every artifact that declares it. This is indexing, not engineering validation.
        """
        if not project_id or not document_id or not artifacts or not isinstance(context, dict) or not context:
            raise ValueError("project, document, context and artifacts are required")
        if len(source_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_sha256):
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
        document_keys = set()
        encoded = {}
        for name, payload in sorted(artifacts.items()):
            if not isinstance(payload, dict) or not payload.get("schema_version") or not payload.get("layer"):
                raise ValueError("each artifact must have a layer and schema_version")
            document = payload.get("document", {})
            if document.get("source_pdf_sha256") not in (None, source_sha256):
                raise ValueError("artifact source revision mismatch")
            if document.get("document_key"):
                document_keys.add(document["document_key"])
            encoded[name] = _hash(payload)
        if len(document_keys) > 1:
            raise ValueError("artifacts from different documents cannot share a snapshot")
        by_layer = {}
        for name, payload in artifacts.items():
            by_layer.setdefault(payload["layer"], set()).add(encoded[name])
        for payload in artifacts.values():
            for key, contract in payload.items():
                if key.endswith("_contract_ref") and isinstance(contract, dict):
                    layer, digest = contract.get("layer"), contract.get("payload_sha256")
                    if digest and layer in by_layer and digest not in by_layer[layer]:
                        raise ValueError("stale artifact dependency: " + key)
        manifest = {"importer_version": VERSION, "project_id": project_id, "document_id": document_id,
            "source_sha256": source_sha256, "context": context,
            "artifacts": encoded}
        snapshot_id = _hash(manifest)
        if self.connection.execute("PRAGMA user_version").fetchone()[0] == 2:
            return self._import_snapshot_v2(snapshot_id=snapshot_id, manifest=manifest,
                project_id=project_id, document_id=document_id, source_sha256=source_sha256,
                context=context, artifacts=artifacts, encoded=encoded, progress=progress)
        with self.connection:
            # Serialize competing writers before inspecting the active revision.
            self.connection.execute("BEGIN IMMEDIATE")
            if self.connection.execute("SELECT 1 FROM snapshots WHERE id=?", (snapshot_id,)).fetchone():
                return snapshot_id  # Reimporting history never rolls the active revision backwards.
            self.connection.execute("INSERT INTO snapshots(id,project_id,document_id,source_sha256,context_json,manifest_json) VALUES(?,?,?,?,?,?)",
                (snapshot_id, project_id, document_id, source_sha256, _json(context), _json(manifest)))
            for name, digest in encoded.items():
                node_count = 0
                exists = self.connection.execute("SELECT 1 FROM artifacts WHERE sha256=?", (digest,)).fetchone()
                if not exists:
                    self.connection.execute("INSERT INTO artifacts VALUES(?,?)",
                        (digest, zlib.compress(_json(artifacts[name]).encode(), level=1)))
                    for pointer, page_ref, record in _records(artifacts[name]):
                        node_id = _hash([digest, pointer])
                        # Complete search rows repeat the same native geometry
                        # in many queries. Keep every node, page, pointer and
                        # reference edge, but store its body once in the exact
                        # compressed artifact, just like large containers.
                        record_body = None if 'source_rows' in pointer.split('/') else _json(record)
                        # Containers keep an exact artifact pointer; do not repeat
                        # whole pages/region trees in every indexed parent node.
                        if record_body is not None and len(record_body) > 65536:
                            record_body = None
                        self.connection.execute("INSERT INTO nodes VALUES(?,?,?,?,?,?,?,?)",
                            (node_id, digest, pointer, record.get("id"),
                             record.get("record_type") or record.get("entity_type") or pointer.split("/")[-2],
                             page_ref, record.get("state") or record.get("epistemic_state"), record_body))
                        self.connection.executemany("INSERT INTO edges VALUES(?,?,?)",
                            [(node_id, field, ref) for field, ref in _references(record)])
                        node_count += 1
                        if progress and node_count % 250000 == 0:
                            progress({'phase': 'artifact_index_progress', 'artifact': name, 'indexed_nodes': node_count})
                self.connection.execute("INSERT INTO snapshot_artifacts VALUES(?,?,?)", (snapshot_id, name, digest))
                if progress:
                    progress({'phase': 'artifact_indexed', 'artifact': name, 'new_indexed_nodes': node_count})
            self.connection.execute("INSERT INTO active_documents VALUES(?,?,?) ON CONFLICT(project_id,document_id) DO UPDATE SET snapshot_id=excluded.snapshot_id",
                (project_id, document_id, snapshot_id))
        return snapshot_id

    def _import_snapshot_v2(self, *, snapshot_id, manifest, project_id, document_id,
                            source_sha256, context, artifacts, encoded, progress):
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            if self.connection.execute("SELECT 1 FROM snapshots WHERE id=?", (snapshot_id,)).fetchone():
                return snapshot_id
            self.connection.execute("INSERT INTO snapshots(id,project_id,document_id,source_sha256,context_json,manifest_json) VALUES(?,?,?,?,?,?)",
                (snapshot_id, project_id, document_id, source_sha256, _json(context), _json(manifest)))
            for name, digest in encoded.items():
                node_count = 0
                exists = self.connection.execute("SELECT 1 FROM artifacts WHERE sha256=?", (digest,)).fetchone()
                if not exists:
                    body = _json(artifacts[name]).encode()
                    spans = {}
                    self.connection.execute("INSERT INTO artifacts(sha256,payload_zlib,byte_count,chunk_count) VALUES(?,NULL,NULL,NULL)",
                                            (digest,))
                    byte_count, chunk_count = _artifact_chunks(self.connection, digest, (body,),
                        span_callback=lambda pointer, start, length: spans.__setitem__(pointer, (start, length)))
                    self.connection.execute("UPDATE artifacts SET byte_count=?,chunk_count=? WHERE sha256=?",
                                            (byte_count, chunk_count, digest))
                    for pointer, page_ref, record in _records(artifacts[name]):
                        node_id = _hash([digest, pointer])
                        record_body = None if 'source_rows' in pointer.split('/') else _json(record)
                        if record_body is not None and len(record_body) > 65536:
                            record_body = None
                        start, length = spans[pointer]
                        self.connection.execute("INSERT INTO nodes VALUES(?,?,?,?,?,?,?,?,?,?)",
                            (node_id, digest, pointer, record.get("id"),
                             record.get("record_type") or record.get("entity_type") or pointer.split("/")[-2],
                             page_ref, record.get("state") or record.get("epistemic_state"),
                             record_body, start, length))
                        self.connection.executemany("INSERT INTO edges VALUES(?,?,?)",
                            [(node_id, field, ref) for field, ref in _references(record)])
                        _insert_fts(self.connection, [(node_id, record.get("id"),
                            record.get("record_type") or record.get("entity_type") or pointer.split("/")[-2],
                            page_ref, record.get("state") or record.get("epistemic_state"), record_body)])
                        node_count += 1
                    del body, spans
                self.connection.execute("INSERT INTO snapshot_artifacts VALUES(?,?,?)", (snapshot_id, name, digest))
                if progress:
                    progress({'phase': 'artifact_indexed', 'artifact': name, 'new_indexed_nodes': node_count})
            self.connection.execute("INSERT INTO active_documents VALUES(?,?,?) ON CONFLICT(project_id,document_id) DO UPDATE SET snapshot_id=excluded.snapshot_id",
                (project_id, document_id, snapshot_id))
        return snapshot_id

    def snapshot(self, *, project_id, document_id, snapshot_id=None):
        if snapshot_id is None:
            row = self.connection.execute("SELECT snapshot_id FROM active_documents WHERE project_id=? AND document_id=?",
                (project_id, document_id)).fetchone()
            if row is None:
                raise KeyError("document is not imported")
            snapshot_id = row[0]
        row = self.connection.execute("SELECT * FROM snapshots WHERE id=? AND project_id=? AND document_id=?",
            (snapshot_id, project_id, document_id)).fetchone()
        if row is None:
            raise KeyError("snapshot is outside this project/document")
        return dict(row)

    def query(self, *, project_id, document_id, snapshot_id=None, record_type=None,
              state=None, page_ref=None, source_id=None, limit=100, offset=0):
        if not 1 <= limit <= 1000 or offset < 0:
            raise ValueError("query limit must be 1..1000 and offset nonnegative")
        snapshot = self.snapshot(project_id=project_id, document_id=document_id, snapshot_id=snapshot_id)
        clauses = ["sa.snapshot_id=?"]
        parameters = [snapshot["id"]]
        for field, value in (("record_type", record_type), ("state", state), ("page_ref", page_ref), ("source_id", source_id)):
            if value is not None:
                clauses.append("n." + field + "=?")
                parameters.append(value)
        rows = self.connection.execute("SELECT DISTINCT n.* FROM nodes n JOIN snapshot_artifacts sa ON sa.artifact_sha256=n.artifact_sha256 WHERE "
            + " AND ".join(clauses) + " ORDER BY n.id LIMIT ? OFFSET ?", (*parameters, limit, offset)).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload_json"]) if row["payload_json"] else None,
                 "payload_in_artifact_only": row["payload_json"] is None,
                 "artifact_only_reference_fields": sorted(_ARTIFACT_ONLY_REFERENCE_FIELDS)} for row in rows]

    def _node(self, snapshot_id, node_id):
        row = self.connection.execute("SELECT n.* FROM nodes n JOIN snapshot_artifacts sa ON sa.artifact_sha256=n.artifact_sha256 WHERE sa.snapshot_id=? AND n.id=?",
            (snapshot_id, node_id)).fetchone()
        if row is None:
            raise KeyError("node does not belong to frozen snapshot")
        return row

    def neighbors(self, *, project_id, document_id, node_id, snapshot_id=None):
        """One evidence-reference hop, preserving absent and ambiguous targets.

        Raw primitive IDs are page scoped; stable typed IDs may explicitly cross
        sheets. Reference resolution is not a physical-connectivity certificate.
        """
        snapshot = self.snapshot(project_id=project_id, document_id=document_id, snapshot_id=snapshot_id)
        node = self._node(snapshot["id"], node_id)
        output = []
        for edge in self.connection.execute("SELECT field,target_ref FROM edges WHERE node_id=? ORDER BY field,target_ref", (node_id,)):
            raw = edge["target_ref"].startswith(("drawing[", "page["))
            matches = self.connection.execute("SELECT DISTINCT n.id,n.source_id,n.page_ref,n.record_type,n.state FROM nodes n JOIN snapshot_artifacts sa ON sa.artifact_sha256=n.artifact_sha256 WHERE sa.snapshot_id=? AND n.source_id=?"
                + (" AND n.page_ref=?" if raw else "") + " ORDER BY n.id",
                (snapshot["id"], edge["target_ref"], *([node["page_ref"]] if raw else []))).fetchall()
            output.append({"field": edge["field"], "target_ref": edge["target_ref"],
                "resolution": "unresolved" if not matches else "unique_record" if len(matches) == 1 else "multiple_records",
                "targets": [dict(row) for row in matches], "engineering_authority_inferred": False})
        return output

    def node_payload(self, *, project_id, document_id, node_id, snapshot_id=None):
        """Resolve an indexed record exactly, including artifact-only bodies."""
        snapshot = self.snapshot(project_id=project_id, document_id=document_id, snapshot_id=snapshot_id)
        node = self._node(snapshot['id'], node_id)
        if node['payload_json'] is not None:
            return json.loads(node['payload_json'])
        if self.connection.execute("PRAGMA user_version").fetchone()[0] == 2:
            if node['payload_start'] is None or node['payload_length'] is None:
                raise ValueError("indexed record lacks chunk-local payload coordinates")
            return json.loads(_chunk_bytes(self.connection, node['artifact_sha256'],
                                           node['payload_start'], node['payload_length']))
        row = self.connection.execute('SELECT payload_zlib FROM artifacts WHERE sha256=?',
                                      (node['artifact_sha256'],)).fetchone()
        value = json.loads(zlib.decompress(row[0]))
        for token in node['pointer'].split('/')[1:]:
            key = token.replace('~1', '/').replace('~0', '~')
            value = value[int(key)] if isinstance(value, list) else value[key]
        return value

    def artifact(self, *, project_id, document_id, name, snapshot_id=None):
        snapshot = self.snapshot(project_id=project_id, document_id=document_id, snapshot_id=snapshot_id)
        row = self.connection.execute("SELECT a.* FROM artifacts a JOIN snapshot_artifacts sa ON sa.artifact_sha256=a.sha256 WHERE sa.snapshot_id=? AND sa.name=?",
            (snapshot["id"], name)).fetchone()
        if row is None:
            raise KeyError(name)
        if self.connection.execute("PRAGMA user_version").fetchone()[0] == 2:
            return json.loads(b"".join(self.iter_artifact_bytes(project_id=project_id,
                document_id=document_id, name=name, snapshot_id=snapshot["id"])))
        return json.loads(zlib.decompress(row["payload_zlib"]))

    def iter_artifact_bytes(self, *, project_id, document_id, name, snapshot_id=None):
        """Yield exact canonical artifact bytes without assembling the artifact."""
        snapshot = self.snapshot(project_id=project_id, document_id=document_id, snapshot_id=snapshot_id)
        row = self.connection.execute("SELECT a.* FROM artifacts a JOIN snapshot_artifacts sa ON sa.artifact_sha256=a.sha256 WHERE sa.snapshot_id=? AND sa.name=?",
            (snapshot["id"], name)).fetchone()
        if row is None:
            raise KeyError(name)
        digest = hashlib.sha256()
        byte_count = 0
        if self.connection.execute("PRAGMA user_version").fetchone()[0] == 1:
            inflater = zlib.decompressobj()
            payload = row["payload_zlib"]
            for offset in range(0, len(payload), 1024 * 1024):
                pending = payload[offset:offset + 1024 * 1024]
                while pending:
                    raw = inflater.decompress(pending, CHUNK_BYTES)
                    if raw:
                        digest.update(raw); byte_count += len(raw)
                        yield raw
                    pending = inflater.unconsumed_tail
            raw = inflater.flush()
            if raw:
                digest.update(raw); byte_count += len(raw)
                yield raw
            valid = inflater.eof and not inflater.unused_data
        else:
            if row["payload_zlib"] is not None:
                raise ValueError("artifact content hash changed: unexpected monolithic payload")
            valid = True
            for chunk in self.connection.execute("SELECT * FROM artifact_chunks WHERE artifact_sha256=? ORDER BY ordinal",
                                                 (row["sha256"],)):
                raw = zlib.decompress(chunk["payload_zlib"])
                if (len(raw) != chunk["uncompressed_size"]
                        or hashlib.sha256(raw).hexdigest() != chunk["payload_sha256"]
                        or chunk["uncompressed_offset"] != byte_count):
                    raise ValueError("artifact chunk content hash changed")
                digest.update(raw); byte_count += len(raw)
                yield raw
            valid = byte_count == row["byte_count"] and row["chunk_count"] > 0
        if not valid or digest.hexdigest() != row["sha256"]:
            raise ValueError("artifact content hash changed")

    def search(self, *, project_id, document_id, text, snapshot_id=None, limit=100, offset=0):
        """Lexical retrieval only; results never establish an engineering relation."""
        if self.connection.execute("PRAGMA user_version").fetchone()[0] != 2:
            raise ValueError("FTS5 search requires project database schema 2")
        if not isinstance(text, str) or not text.strip() or len(text) > 500:
            raise ValueError("search text must contain 1..500 characters")
        if not 1 <= limit <= 1000 or offset < 0:
            raise ValueError("search limit must be 1..1000 and offset nonnegative")
        snapshot = self.snapshot(project_id=project_id, document_id=document_id, snapshot_id=snapshot_id)
        rows = self.connection.execute("SELECT n.*,bm25(node_fts) AS rank FROM node_fts "
            "JOIN fts_nodes f ON f.rowid=node_fts.rowid JOIN nodes n ON n.id=f.node_id "
            "JOIN snapshot_artifacts sa ON sa.artifact_sha256=n.artifact_sha256 "
            "WHERE node_fts MATCH ? AND sa.snapshot_id=? ORDER BY rank,n.id LIMIT ? OFFSET ?",
            (text, snapshot["id"], limit, offset)).fetchall()
        return [{**dict(row), "retrieval_authority": "proposal_only",
                 "engineering_authority_inferred": False} for row in rows]

    def add_review(self, *, project_id, document_id, snapshot_id, node_id, review):
        """Append a graph-bound overlay without editing facts or granting approval."""
        self.snapshot(project_id=project_id, document_id=document_id, snapshot_id=snapshot_id)
        self._node(snapshot_id, node_id)
        if not isinstance(review, dict) or not review.get("reviewer") or not review.get("decision"):
            raise ValueError("reviewer and decision are required")
        review_id = _hash([snapshot_id, node_id, review])
        with self.connection:
            self.connection.execute("INSERT OR IGNORE INTO reviews(id,snapshot_id,node_id,payload_json) VALUES(?,?,?,?)",
                (review_id, snapshot_id, node_id, _json(review)))
        return review_id

    def reviews(self, *, project_id, document_id, snapshot_id=None):
        snapshot = self.snapshot(project_id=project_id, document_id=document_id, snapshot_id=snapshot_id)
        return [{**dict(row), "review": json.loads(row["payload_json"])} for row in self.connection.execute(
            "SELECT * FROM reviews WHERE snapshot_id=? ORDER BY id", (snapshot["id"],))]
