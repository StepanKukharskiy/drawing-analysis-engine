#!/usr/bin/env python3
"""Independent NIST reference snapshot importer and read-only audit/app view."""
from __future__ import annotations
import argparse
from contextlib import closing, ExitStack
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.drawing_engine.project.mep_project_review import _artifact, _indexed
from src.drawing_engine.disciplines.mep.mep_document_links import validate_document_reference_links
from src.drawing_engine.project.project_knowledge_store import ProjectKnowledgeStore
from src.drawing_engine.project.project_packed_store import PackedProjectStore

PROJECT = "nist-medium-office-document-links"
DOCUMENT = "nist-reference-plumbing-medium-office"
SOURCE = ROOT / "fixtures/mep/nist_document_links/nist-reference-plumbing-full.pdf"


def import_reference_snapshot(database, payload, source):
    validate_document_reference_links(payload, source=source)
    source_hash = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    context = {"scope": payload["scope"], "source_validation": "complete_native_replay",
               "implementation_sha256": hashlib.sha256((ROOT / "src/drawing_engine/disciplines/mep/mep_document_links.py").read_bytes()).hexdigest(),
               "independent_from_mep_coordination": True, "quantity_eligible": False}
    with ProjectKnowledgeStore(database) as store:
        return store.import_snapshot(project_id=PROJECT, document_id=DOCUMENT, source_sha256=source_hash,
                                     context=context, artifacts={"document-links": payload})


def load_reference_review(database, *, snapshot=None, selected=None):
    with ExitStack() as stack:
        connection = stack.enter_context(closing(sqlite3.connect(
            Path(database).resolve().as_uri() + "?mode=ro", uri=True)))
        connection.row_factory = sqlite3.Row
        version = connection.execute('PRAGMA user_version').fetchone()[0]
        packed = stack.enter_context(PackedProjectStore(database)) if version == 3 else None
        versions = ([{key:row[key] for key in ('id','created_at','active')}
                     for row in packed.snapshots(project_id=PROJECT,document_id=DOCUMENT)]
                    if packed else [dict(r) for r in connection.execute(
            "SELECT s.id,s.created_at,(a.snapshot_id=s.id) AS active FROM snapshots s LEFT JOIN active_documents a ON a.project_id=s.project_id AND a.document_id=s.document_id WHERE s.project_id=? AND s.document_id=? ORDER BY s.created_at DESC,s.id", (PROJECT, DOCUMENT))])
        snapshot = snapshot or next((r["id"] for r in versions if r["active"]), None)
        if not snapshot or snapshot not in {r["id"] for r in versions}:
            raise ValueError("reference snapshot is outside independent project")
        payload = (json.loads(b''.join(packed.iter_artifact_bytes(
            project_id=PROJECT,document_id=DOCUMENT,snapshot_id=snapshot,
            name='document-links'))) if packed else _artifact(connection,snapshot,'document-links'))
        validate_document_reference_links(payload)
        rows = []
        if packed:
            nodes = packed.collection(project_id=PROJECT,document_id=DOCUMENT,
                snapshot_id=snapshot,name='document-links',collection='reference_links',
                include_payload=False)
            by_id = {node['source_id']:node for node in nodes}
        for link in payload['reference_links']:
            node = by_id[link['id']] if packed else _indexed(
                connection,snapshot,'document-links',link['id'])
            rows.append({**link,'pointer':node['pointer'],
                         'artifact_sha256':node['artifact_sha256']})
        if selected is None:
            selected = next((r["id"] for r in rows if r["state"] == "accepted"), rows[0]["id"] if rows else None)
        row = next((r for r in rows if r["id"] == selected), None)
        if selected and row is None:
            raise ValueError("selected reference is outside snapshot")
        markers = {m["id"]: m for m in payload["markers"]}
        views = {v["id"]: v for v in payload["views"]}
        return {"snapshot_id": snapshot, "versions": versions, "project": PROJECT, "document": payload["document"],
                "scope": payload["scope"], "summary": payload["summary"], "authority": payload["authority"],
                "rows": rows, "selected": row,
                "source_marker": markers.get(row["source_marker_ref"]) if row else None,
                "destination_marker": markers.get(row["destination_marker_ref"]) if row else None,
                "destination_view": views.get(row["destination_view_ref"]) if row else None}


def reference_crop(review, source, *, side="source", detail=False):
    import fitz
    if hashlib.sha256(Path(source).read_bytes()).hexdigest() != review["document"]["source_pdf_sha256"]:
        raise ValueError("reference crop source hash mismatch")
    marker = review[f"{side}_marker"]
    if marker is None:
        raise ValueError("no accepted destination to crop")
    with fitz.open(source) as pdf:
        page = pdf[marker["source_page_number"] - 1]
        box = fitz.Rect(marker["bbox_display"])
        pad = max(box.width, box.height) * (.7 if detail else 5)
        clip = (box + (-pad, -pad, pad, pad)) & page.rect
        if side == "destination" and not detail:
            view = review["destination_view"]
            baseline = view["caption_baseline_display"]
            clip = fitz.Rect(baseline[0] - 35, max(0, box.y0 - box.height),
                             baseline[2] + 35, baseline[1] + 40) & page.rect
        paths = marker["source_paths"]
        color = (.05, .3, .85) if review["selected"]["state"] == "accepted" else (.7, .36, .02)
        for path in paths:
            points = [fitz.Point(p) for p in path["points_display"]]
            page.draw_polyline(points, color=color, width=.65, overlay=True)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(min(4, 1500/clip.width), min(4, 1500/clip.width)), clip=clip, alpha=False)
        return pixmap.tobytes("png"), {"source_page_number": marker["source_page_number"],
            "clip_display": list(clip), "marker_ref": marker["id"], "source_paths": paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=ROOT / "data/projects/mep/project.sqlite")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--import-artifact", type=Path)
    parser.add_argument("--snapshot")
    parser.add_argument("--selected")
    parser.add_argument("--crop", choices=("source", "destination"))
    parser.add_argument("--detail", action="store_true")
    args = parser.parse_args()
    if args.import_artifact:
        print(import_reference_snapshot(args.database, json.loads(args.import_artifact.read_text()), args.source))
    else:
        review = load_reference_review(args.database, snapshot=args.snapshot, selected=args.selected)
        if args.crop:
            sys.stdout.buffer.write(reference_crop(review, args.source, side=args.crop, detail=args.detail)[0])
        else:
            print(json.dumps(review))


if __name__ == "__main__":
    main()
