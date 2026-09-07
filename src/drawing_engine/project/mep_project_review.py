#!/usr/bin/env python3
"""Read-only snapshot projection for the MEP inspection app and marked audit.

Inspection rows are not declared schedules or physical items. Every connection
shown here is a frozen source reference, never a label/geometry similarity join.
"""
from __future__ import annotations

import argparse
from contextlib import closing, ExitStack
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import zlib

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.drawing_engine.project.project_packed_store import PackedProjectStore


def _display_attributes(attributes, overlay):
    """Presentation only: never propagate values between connected segments."""
    result = {}
    for field in ('system', 'size', 'elevation'):
        rows = [r for r in attributes if r.get('relation_type') == 'route_' + field and r['state'] == 'accepted']
        values = {json.dumps({k: v for k, v in r['candidate'].items() if k not in {'raw_text', 'terminology_entry_ref'}}, sort_keys=True) for r in rows}
        state = overlay.get(field, {}).get('state') or ('conflicted' if len(values) > 1 else 'accepted' if values else 'unknown')
        text = 'Conflicting' if state == 'conflicted' else 'Unknown'
        candidate = rows[0]['candidate'] if state == 'accepted' and len(values) == 1 else None
        if candidate:
            kind = candidate.get('kind', '')
            unit = candidate.get('unit') or '(unit unknown)'
            if field == 'system':
                text = {'heating_hot_water_supply': 'HHWS', 'heating_hot_water_return': 'HHWR',
                        'chilled_water_supply': 'CHWS', 'chilled_water_return': 'CHWR'}.get(kind, kind.replace('_', ' '))
            elif field == 'size':
                if kind == 'rectangular_duct_size':
                    text = f"{candidate['width']:g} x {candidate['height']:g} {unit}"
                else:
                    text = f"{candidate['value']:g} {unit}" + (' (nominal)' if kind in {'nominal_size', 'nominal_diameter'} else '')
            else:
                basis = {'bottom': 'Bottom', 'centreline': 'CL', 'top': 'Top'}.get(candidate.get('basis'), 'Reference unknown')
                text = f"{basis} {candidate['value']:.4f}".rstrip('0').rstrip('.') + f' {unit}'
                if unit == 'ft':
                    inches = Fraction(abs(candidate['value']) * 12).limit_denominator(64)
                    if abs(float(inches) / 12 - abs(candidate['value'])) < 1e-6:
                        feet = int(inches // 12)
                        remainder = inches - feet * 12
                        whole = int(remainder)
                        fraction = remainder - whole
                        inch_text = str(whole) + (f' {fraction}' if fraction else '')
                        sign = '-' if candidate['value'] < 0 else ''
                        text = f'{basis} {sign}{feet}\' {inch_text}"'
        result[field] = {'state': state, 'text': text, 'candidate': candidate}
    return result


def _display_length(occurrence):
    value = occurrence.get('projected_2d_length_m')
    valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    return {'state': 'derived' if valid else 'unknown', 'text': f'{value:.2f} m' if valid else 'Unknown',
            'value': value if valid else None, 'unit': 'm', 'basis': 'projected_2d',
            'scope': 'this_page_occurrence', 'source_occurrence_ref': occurrence['id']}


def _artifact_body(connection, snapshot_id, name):
    row = connection.execute("SELECT a.* FROM artifacts a JOIN snapshot_artifacts sa ON sa.artifact_sha256=a.sha256 WHERE sa.snapshot_id=? AND sa.name=?", (snapshot_id, name)).fetchone()
    if row is None:
        raise ValueError(f"snapshot lacks {name}")
    if connection.execute("PRAGMA user_version").fetchone()[0] == 1:
        body = zlib.decompress(row['payload_zlib'])
    else:
        if row['payload_zlib'] is not None:
            raise ValueError("frozen artifact digest mismatch")
        parts = []
        offset = 0
        for chunk in connection.execute('SELECT * FROM artifact_chunks WHERE artifact_sha256=? ORDER BY ordinal',(row['sha256'],)):
            raw = zlib.decompress(chunk['payload_zlib'])
            if (chunk['uncompressed_offset'] != offset or len(raw) != chunk['uncompressed_size']
                    or hashlib.sha256(raw).hexdigest() != chunk['payload_sha256']):
                raise ValueError("frozen artifact digest mismatch")
            parts.append(raw); offset += len(raw)
        body = b''.join(parts)
        if offset != row['byte_count']:
            raise ValueError("frozen artifact digest mismatch")
    if hashlib.sha256(body).hexdigest() != row['sha256']:
        raise ValueError("frozen artifact digest mismatch")
    return body


def _artifact(connection, snapshot_id, name):
    return json.loads(_artifact_body(connection, snapshot_id, name))


def _chunk_node_body(connection, row):
    start, end = row['payload_start'], row['payload_start'] + row['payload_length']
    parts = []
    for chunk in connection.execute('SELECT * FROM artifact_chunks WHERE artifact_sha256=? AND uncompressed_offset<? AND uncompressed_offset+uncompressed_size>? ORDER BY ordinal',
                                    (row['artifact_sha256'],end,start)):
        raw = zlib.decompress(chunk['payload_zlib'])
        if len(raw) != chunk['uncompressed_size'] or hashlib.sha256(raw).hexdigest() != chunk['payload_sha256']:
            raise ValueError('frozen node chunk digest mismatch')
        left=max(start,chunk['uncompressed_offset'])-chunk['uncompressed_offset']
        right=min(end,chunk['uncompressed_offset']+len(raw))-chunk['uncompressed_offset']
        parts.append(raw[left:right])
    body=b''.join(parts)
    if len(body)!=row['payload_length']:
        raise ValueError('frozen node chunk coverage is incomplete')
    return body


def _indexed(connection, snapshot_id, artifact, source_id, *, require_payload=True,
             packed=None):
    if packed is not None:
        return packed.indexed(artifact, source_id, require_payload=require_payload)
    rows = connection.execute("SELECT n.* FROM nodes n INDEXED BY nodes_source JOIN snapshot_artifacts sa ON n.artifact_sha256=sa.artifact_sha256 WHERE sa.snapshot_id=? AND sa.name=? AND n.source_id=? ORDER BY n.id", (snapshot_id, artifact, source_id)).fetchall()
    if len(rows) != 1:
        raise ValueError(f"non-unique or unindexed {artifact} record: {source_id}")
    row = dict(rows[0])
    body = row.pop("payload_json")
    if require_payload and body is None:
        if connection.execute('PRAGMA user_version').fetchone()[0] != 2:
            raise ValueError(f"non-unique or unindexed {artifact} record: {source_id}")
        body = _chunk_node_body(connection,row)
    row["payload"] = json.loads(body) if body is not None else None
    return row


class _PackedReviewReader:
    """Expose the frozen audit reader contract over verified schema-v3 packs."""

    def __init__(self, store, *, project, document, snapshot):
        self.store = store
        self.frozen = store.snapshot(project_id=project, document_id=document,
                                     snapshot_id=snapshot)
        self.snapshot_key = self.frozen["snapshot_key"]
        self.artifact_keys = {row["name"]: row["artifact_key"] for row in
            store.connection.execute(
                "SELECT name,artifact_key FROM snapshot_artifacts WHERE snapshot_key=?",
                (self.snapshot_key,))}
        self.artifact_names = {value: key for key, value in self.artifact_keys.items()}

    def artifact(self, _connection, snapshot, name):
        if snapshot != self.frozen["id"] or name not in self.artifact_keys:
            raise ValueError(f"snapshot lacks {name}")
        return json.loads(b"".join(self.store.iter_artifact_bytes(
            project_id=self.frozen["project_id"], document_id=self.frozen["document_id"],
            snapshot_id=snapshot, name=name)))

    def has_artifact(self, name):
        return name in self.artifact_keys

    def indexed(self, artifact, source_id, *, require_payload=True):
        artifact_key = self.artifact_keys.get(artifact)
        rows = [row for row in self.store._source_matches(self.snapshot_key, source_id)
                if row["artifact_key"] == artifact_key]
        rows.sort(key=lambda row: row["id"])
        if len(rows) != 1:
            raise ValueError(f"non-unique or unindexed {artifact} record: {source_id}")
        row = rows[0]
        payload = None
        if require_payload:
            payload = json.loads(self.store._chunk_bytes(
                row["artifact_key"], row["payload_start"], row["payload_length"]))
            if not isinstance(payload, dict) or payload.get("id") != source_id:
                raise ValueError(f"non-unique or unindexed {artifact} record: {source_id}")
        return {**self.store._public_row(row), "payload": payload}

    def node_edges(self, node_id):
        row = self.store._node(self.snapshot_key, node_id)
        return [{"field": field, "target_ref": target_ref}
                for field, target_ref in sorted(row["references"])]

    def source_records(self, source_id):
        rows = []
        for row in sorted(self.store._source_matches(self.snapshot_key, source_id),
                          key=lambda item: item["id"]):
            rows.append({
                "id": row["id"], "pointer": row["pointer"],
                "page_ref": row["page_ref"], "artifact_sha256": row["artifact_sha256"],
                "artifact": self.artifact_names[row["artifact_key"]],
                "source_id": row["source_id"], "record_type": row["record_type"],
                "state": row["state"], "reference_edges": [
                    {"field": field, "target_ref": target_ref}
                    for field, target_ref in sorted(row["references"])]})
        return rows

    def reviews(self):
        return [{"node_id": row["node_id"], "review": row["review"],
                 "created_at": row["created_at"]} for row in self.store.reviews(
                    project_id=self.frozen["project_id"],
                    document_id=self.frozen["document_id"],
                    snapshot_id=self.frozen["id"])]


def load_review(database, *, project, document, snapshot=None, selected=None, compare_to=None, _projection=None):
    """Open SQLite in read-only mode and pin all reads to one exact snapshot."""
    with ExitStack() as stack:
        connection = stack.enter_context(closing(sqlite3.connect(
            Path(database).resolve().as_uri() + "?mode=ro", uri=True)))
        connection.row_factory = sqlite3.Row
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version not in (1,2,3):
            raise ValueError("unsupported project database schema")
        if version == 3:
            versions = [{**dict(row), "id": row["id"].hex(),
                         "source_sha256": row["source_sha256"].hex()}
                        for row in connection.execute(
                "SELECT s.id,s.created_at,s.source_sha256,(a.snapshot_key=s.snapshot_key) AS active "
                "FROM snapshots s LEFT JOIN active_documents a ON a.project_id=s.project_id "
                "AND a.document_id=s.document_id WHERE s.project_id=? AND s.document_id=? "
                "ORDER BY s.created_at DESC,s.id", (project, document))]
        else:
            versions = [dict(row) for row in connection.execute("SELECT s.id,s.created_at,s.source_sha256,(a.snapshot_id=s.id) AS active FROM snapshots s LEFT JOIN active_documents a ON a.project_id=s.project_id AND a.document_id=s.document_id WHERE s.project_id=? AND s.document_id=? ORDER BY s.created_at DESC,s.id", (project, document))]
        if snapshot is None:
            snapshot = next((row["id"] for row in versions if row["active"]), None)
        if snapshot is None or snapshot not in {row["id"] for row in versions}:
            raise ValueError("snapshot is outside this project/document")
        packed = None
        if version == 3:
            store = stack.enter_context(PackedProjectStore(database, readonly=True))
            packed = _PackedReviewReader(store, project=project, document=document,
                                         snapshot=snapshot)
            frozen = {**packed.frozen,
                      "context_json": json.dumps(packed.frozen["context"])}
            read_artifact = packed.artifact if _projection is None else _projection.artifact
        else:
            frozen = dict(connection.execute("SELECT * FROM snapshots WHERE id=?", (snapshot,)).fetchone())
            read_artifact = _artifact if _projection is None else _projection.artifact
        registry = read_artifact(connection, snapshot, "sheet-registry")
        network = read_artifact(connection, snapshot, "network-hierarchy")
        hvac = read_artifact(connection, snapshot, "hvac-inventory")
        catalog = read_artifact(connection, snapshot, "item-catalog")
        pages = {row["page_ref"]: row for row in registry["pages"]}
        observations = {row["id"]: row for row in hvac["source_observations"]}
        lines = {row["item_occurrence_ref"]: row for row in catalog["takeoff_lines"]}
        rows = []
        for artifact, records, kind in (("network-hierarchy", network["segments"], "segment"), ("hvac-inventory", hvac["items"], "hvac")):
            for record in records:
                node = _indexed(connection, snapshot, artifact, record["id"], packed=packed)
                marks = []
                for ref in record.get("source_occurrence_refs", []):
                    occurrence = _indexed(connection, snapshot, "cross-sheet-runs", ref, packed=packed)["payload"]
                    marks.append({"page_ref": occurrence["page_ref"], "points_display": occurrence["points_display"], "source_ref": ref,
                                  "source_primitive_refs": occurrence.get("source_primitive_refs") or [occurrence["source_primitive_ref"]],
                                  "role": "projected_segment", 'display_values': {'projected_length': _display_length(occurrence)}})
                for ref in record.get("source_observation_refs", []):
                    observation = observations.get(ref)
                    if observation is not None and observation.get("bbox_display"):
                        marks.append({"page_ref": observation["page_ref"], "bbox_display": observation["bbox_display"], "source_ref": ref, "role": "observed_tag", "text": observation.get("text"), "method": observation.get("method"), "confidence": observation.get("confidence")})
                projected_identities = [_indexed(connection, snapshot, 'projected-identity-bindings', ref, packed=packed)['payload']
                                        for ref in record.get('projected_identity_relation_refs', [])]
                identity_assessments = [_indexed(connection, snapshot, 'projected-identity-bindings', ref, packed=packed)['payload']
                                       for ref in record.get('projected_identity_assessment_refs', [])]
                for identity in projected_identities:
                    paths = [{'points_display': path['points_display'], 'source_primitive_refs': path.get('source_primitive_refs', []),
                              'role': 'identified_projected_body'} for path in identity['body_paths']]
                    paths.extend({'bbox_display': mark['bbox_display'], 'role': 'identified_equipment_tag',
                                  'source_observation_refs': [mark['source_ref']]} for mark in marks)
                    points = [point for path in paths for point in (path.get('points_display') or
                              [path['bbox_display'][:2], path['bbox_display'][2:]])]
                    marks = [{'page_ref': identity['page_ref'], 'source_ref': identity['id'],
                              'role': 'identified_projected_body_and_tag_ports_unresolved', 'evidence_paths': paths,
                              'bbox_display': [min(p[0] for p in points), min(p[1] for p in points),
                                               max(p[0] for p in points), max(p[1] for p in points)],
                              'source_primitive_refs': identity['source_primitive_refs']}]
                if record.get('geometry_only') and not record.get('source_occurrence_refs'):
                    marks.append({'page_ref': record['page_refs'][0], 'points_display': record['points_display'],
                                  'source_ref': record['route_target_refs'][0], 'source_primitive_refs': record['source_primitive_refs'],
                                  'role': 'projected_geometry_system_size_elevation_unknown'})
                for mark in marks:
                    page = pages[mark["page_ref"]]
                    mark["page_number"] = page["page_number"]
                    mark["page_size_display"] = page["page_size_display"]
                    mark["clip_display"] = mark_clip(mark)
                targets = set(record.get("route_target_refs", record.get("target_refs", [])))
                catalog_rows = [{"occurrence": item, "takeoff_line": lines.get(item["id"])} for item in catalog["item_occurrences"] if targets.intersection(item["target_refs"])]
                attributes = [_indexed(connection, snapshot, "attribute-bindings", ref, packed=packed)["payload"] for ref in record.get("attribute_relation_refs", record.get("source_relation_refs", []))]
                attribute_labels = [{"relation_ref": attribute["id"], "state": attribute["state"], "kind": attribute.get("relation_type"), "text": attribute.get("candidate", {}).get("raw_text") or attribute.get("relation_type", "attribute")}
                                    for attribute in attributes]
                system_labels = sorted({attribute["text"] for attribute in attribute_labels
                                        if attribute["state"] == "accepted" and attribute["kind"] == "route_system"})
                rows.append({"id": record["id"], "node_id": node["id"], "artifact": artifact, "pointer": node["pointer"], "artifact_sha256": node["artifact_sha256"], "kind": kind,
                    "label": record.get("candidate", {}).get("tag") or record.get("item_class") or "; ".join(system_labels) or "Projected segment",
                    "state": record["state"], "payload": record, "marks": marks, "attributes": attributes, "attribute_labels": attribute_labels, "catalog_rows": catalog_rows,
                    **({'projected_identities': projected_identities} if projected_identities else {}),
                    **({'projected_identity_assessments': identity_assessments} if identity_assessments else {}),
                    "run_refs": [run["id"] for run in network["runs"] if record["id"] in run.get("segment_refs", [])],
                    'network_refs': [item['id'] for item in network['networks'] if record['id'] in item.get('segment_refs', [])],
                    'semantic_overlay': record.get('semantic_overlay', {}),
                    'display_values': _display_attributes(attributes, record.get('semantic_overlay', {})),
                    'scoped_trace_summary': [{key: trace[key] for key in ('id', 'state', 'scope_ref', 'projected_scope_complete',
                        'reasons', 'coverage', 'endpoint_classifications') if key in trace}
                        for trace in network.get('scoped_traces', []) if trace['id'] in record.get('scoped_trace_refs', [])],
                    "declared_schedule_link_state": "no_explicit_declared_row_reference", "quantity_eligible": False})
        outcome_artifact = None
        if (packed.has_artifact("review-outcomes") if packed is not None else
                connection.execute("SELECT 1 FROM snapshot_artifacts WHERE snapshot_id=? AND name='review-outcomes'",(snapshot,)).fetchone()):
            outcome_artifact = read_artifact(connection,snapshot,"review-outcomes")
            if (outcome_artifact.get("layer") != "mep_bounded_review_outcomes"
                    or outcome_artifact.get("document",{}).get("source_pdf_sha256") != frozen["source_sha256"]):
                raise ValueError("bounded outcome artifact differs from frozen source")
        outcomes = outcome_artifact.get("outcomes",[]) if outcome_artifact else []
        outcome_ids = set()
        for index, outcome in enumerate(outcomes):
            if (not outcome.get("id") or outcome["id"] in outcome_ids or outcome.get("page_ref") not in pages
                    or outcome.get("state") not in {"accepted","abstained","rejected"}
                    or outcome.get("quantity_eligible") is not False
                    or outcome.get("physical_continuation_established",False) is not False
                    or outcome.get("physical_item_identity_established",False) is not False
                    or not isinstance(outcome.get("subject_refs"),list)
                    or not all(isinstance(ref,str) for ref in outcome["subject_refs"])
                    or not isinstance(outcome.get("reason_codes"),list)):
                raise ValueError("invalid bounded outcome identity, scope or authority")
            outcome_ids.add(outcome["id"])
            for row in rows:
                if (row["kind"] != "outcome" and outcome["page_ref"] in {mark["page_ref"] for mark in row["marks"]}
                        and set(outcome["subject_refs"]).intersection(_row_refs(row))):
                    row.setdefault("bounded_outcomes",[]).append(outcome)
            node = _indexed(connection,snapshot,"review-outcomes",outcome["id"],require_payload=False,packed=packed)
            marks = []
            outcome_pointer = node['pointer']
            paths = [(path,f"{outcome_pointer}/source_paths/{path_index}") for path_index,path in enumerate(outcome.get("source_paths",[]))]
            diagnostic = outcome.get("trace_diagnostics",{})
            trace = diagnostic.get("trace_points_display",[])
            if len(trace)>1:
                paths.append(({"points_display":trace,"role":"frozen_partial_native_trace","incident_source_primitive_refs":diagnostic.get("incident_source_primitive_refs",[])},f"{outcome_pointer}/trace_diagnostics/trace_points_display"))
            marker_points = [(point,f"{outcome_pointer}/trace_diagnostics/competing_next_points_display/{i}","competing_trace_point_display_marker") for i,point in enumerate(diagnostic.get("competing_next_points_display",[]))]
            if diagnostic.get("failure_point_display"):
                marker_points.append((diagnostic["failure_point_display"],f"{outcome_pointer}/trace_diagnostics/failure_point_display","trace_failure_display_marker"))
            for point,pointer,role in marker_points:
                paths.append(({"bbox_display":[point[0]-1,point[1]-1,point[0]+1,point[1]+1],"role":role,
                               "source_primitive_refs":diagnostic.get("incident_source_primitive_refs",[]),"display_marker_only":True},pointer))
            for path,source_pointer in paths:
                page = pages.get(path.get("page_ref",outcome["page_ref"]))
                if page is None:
                    raise ValueError("outcome path is outside frozen page registry")
                mark = {key:value for key,value in path.items() if key in {"bbox_display","points_display","source_primitive_refs","source_observation_refs","incident_source_primitive_refs","role","display_marker_only"}}
                if not mark.get("points_display") and not mark.get("bbox_display"):
                    continue
                points, box = mark.get("points_display"), mark.get("bbox_display")
                valid_numbers = lambda values: all(isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) for value in values)
                if ((points is not None and (not isinstance(points,list) or len(points)<2 or any(not isinstance(point,list) or len(point)!=2 or not valid_numbers(point) for point in points)))
                        or (box is not None and (not isinstance(box,list) or len(box)!=4 or not valid_numbers(box) or box[2]<=box[0] or box[3]<=box[1]))):
                    raise ValueError("invalid bounded outcome source geometry")
                mark.update(page_ref=page["page_ref"],page_number=page["page_number"],page_size_display=page["page_size_display"],
                            source_ref=outcome["id"],source_path_pointer=source_pointer)
                mark.setdefault("role","bounded_outcome_evidence")
                mark["clip_display"] = mark_clip(mark)
                marks.append(mark)
            grouped = {}
            for mark in marks:
                page_ref = mark["page_ref"]
                if page_ref not in grouped:
                    grouped[page_ref] = {key:mark[key] for key in ("page_ref","page_number","page_size_display","source_ref")}
                    grouped[page_ref].update(role="bounded_outcome_evidence",evidence_paths=[],source_primitive_refs=[])
                grouped[page_ref]["evidence_paths"].append(mark)
                grouped[page_ref]["source_primitive_refs"].extend(mark.get("source_primitive_refs",[]))
            marks = list(grouped.values())
            for mark in marks:
                points = []
                for path in mark["evidence_paths"]:
                    points.extend(path.get("points_display",[]))
                    if path.get("bbox_display"):
                        box = path["bbox_display"]
                        points.extend([box[:2],box[2:]])
                mark["bbox_display"] = [min(point[0] for point in points),min(point[1] for point in points),max(point[0] for point in points),max(point[1] for point in points)]
                mark["clip_display"] = mark_clip(mark)
            closeups = []
            for mark in marks:
                focus = [path for path in mark["evidence_paths"] if path["role"] in {"native_leader","annotation_text"}]
                if not any(path["role"]=="native_leader" for path in focus):
                    continue
                focus_points = [point for path in focus for point in (path.get("points_display") or [path["bbox_display"][:2],path["bbox_display"][2:]])]
                box = [min(point[0] for point in focus_points),min(point[1] for point in focus_points),max(point[0] for point in focus_points),max(point[1] for point in focus_points)]
                closeup = {**mark,"bbox_display":box,"role":"native_leader_contact_closeup",
                           "presentation_focus_source_path_pointers":[path["source_path_pointer"] for path in focus]}
                closeup["clip_display"] = mark_clip(closeup)
                if closeup["clip_display"] != mark["clip_display"]:
                    closeups.append(closeup)
            marks.extend(closeups)
            rows.append({"id":outcome["id"],"node_id":node["id"],"artifact":"review-outcomes","pointer":node["pointer"],"artifact_sha256":node["artifact_sha256"],
                "kind":"outcome","label":outcome.get("label") or outcome.get("description") or outcome.get("subject_kind","Bounded outcome").replace("_"," "),"state":outcome["state"],"payload":outcome,
                "marks":marks,"attributes":[],"attribute_labels":[],"catalog_rows":[],"run_refs":[],"bounded_outcomes":[outcome],
                "declared_schedule_link_state":"no_explicit_declared_row_reference","quantity_eligible":False})
        for row in rows:
            if row['kind'] == 'outcome':
                row['current_projected_identity_context'] = [identity for item in rows
                    for identity in item.get('projected_identities', [])
                    if identity['proposal_ref'] in row['payload'].get('subject_refs', [])]
                if row['current_projected_identity_context']:
                    tag = row['current_projected_identity_context'][0]['equipment_tag']
                    row['label'] = f'{tag}: prior strict port assessment; current body/tag identified in 2D'
            for junction in network['junctions']:
                if junction.get('relation_type') != 'projected_fitting_body_branch' or row['id'] not in junction['segment_refs']:
                    continue
                paths = [{**path, 'role': 'inferred_fitting_native_body'} for path in junction['body_paths']]
                points = [point for path in paths for point in path['points_display']] + junction['port_points_display']
                box = [min(p[0] for p in points), min(p[1] for p in points), max(p[0] for p in points), max(p[1] for p in points)]
                radius = max(box[2]-box[0], box[3]-box[1]) * .035
                paths.extend({'bbox_display': [p[0]-radius,p[1]-radius,p[0]+radius,p[1]+radius],
                              'role': 'inferred_fitting_port_display_marker', 'display_marker_only': True,
                              'source_port_ref': ref} for ref,p in zip(junction['endpoint_refs'],junction['port_points_display']))
                page = pages[junction['page_ref']]
                row['marks'].append({'page_ref': junction['page_ref'], 'page_number': page['page_number'],
                    'page_size_display': page['page_size_display'], 'bbox_display': box, 'evidence_paths': paths,
                    'source_ref': junction['source_projected_identity_binding_ref'],
                    'role': 'inferred_fitting_body_and_port_closeup',
                    'presentation_padding_display': 3 * max(box[2]-box[0],box[3]-box[1])})
        _add_junction_context(rows, network["junctions"])
        boundary_evidence = {}
        for row in rows:
            for mark in row["marks"]:
                for entry in mark["junction_context"]:
                    ref = entry["junction"].get("source_boundary_connection_ref") or entry['junction'].get('source_projected_identity_binding_ref')
                    if ref not in boundary_evidence:
                        if packed is not None:
                            matches = [{key: record[key] for key in
                                ("id", "pointer", "page_ref", "artifact_sha256", "artifact", "reference_edges")}
                                for record in packed.source_records(ref)]
                        else:
                            matches = [dict(record) for record in connection.execute("SELECT n.id,n.pointer,n.page_ref,n.artifact_sha256,sa.name AS artifact FROM nodes n INDEXED BY nodes_source JOIN snapshot_artifacts sa ON sa.artifact_sha256=n.artifact_sha256 WHERE sa.snapshot_id=? AND n.source_id=? ORDER BY n.id",(snapshot,ref))]
                            for match in matches:
                                match["reference_edges"] = [dict(edge) for edge in connection.execute("SELECT field,target_ref FROM edges WHERE node_id=? ORDER BY field,target_ref",(match["id"],))]
                        boundary_evidence[ref] = {"source_ref": ref, "resolution": "unresolved" if not matches else "unique_record" if len(matches)==1 else "multiple_records",
                                                  "records": matches, "engineering_authority_inferred": False}
                    entry["native_boundary_evidence"] = boundary_evidence[ref]
        rows.sort(key=lambda row: (min((mark["page_number"] for mark in row["marks"]), default=999999), row["kind"], row["id"]))
        for index, row in enumerate(rows, 1):
            row["schedule_label"] = f"R{index:03d}"
        selected_row = next((row for row in rows if selected in (row["id"], row["node_id"])), None)
        if selected and selected_row is None:
            raise ValueError("inspection row does not belong to frozen snapshot")
        evidence = []
        if selected_row:
            edges = (packed.node_edges(selected_row["node_id"]) if packed is not None else
                     connection.execute("SELECT field,target_ref FROM edges WHERE node_id=? ORDER BY field,target_ref", (selected_row["node_id"],)))
            for edge in edges:
                raw = edge["target_ref"].startswith(("drawing[", "page["))
                page_refs = {mark["page_ref"] for mark in selected_row["marks"]}
                if packed is not None:
                    matches = [{key: row[key] for key in ("id", "source_id", "record_type", "page_ref", "state", "pointer")}
                               for row in packed.source_records(edge["target_ref"])
                               if not raw or row["page_ref"] in page_refs]
                else:
                    matches = [dict(row) for row in connection.execute("SELECT DISTINCT n.id,n.source_id,n.record_type,n.page_ref,n.state,n.pointer FROM nodes n INDEXED BY nodes_source JOIN snapshot_artifacts sa ON sa.artifact_sha256=n.artifact_sha256 WHERE sa.snapshot_id=? AND n.source_id=? ORDER BY n.id", (snapshot, edge["target_ref"])) if not raw or row["page_ref"] in page_refs]
                evidence.append({**dict(edge), "resolution": "unresolved" if not matches else "unique_record" if len(matches) == 1 else "multiple_records", "targets": matches, "engineering_authority_inferred": False})
        reviews = (packed.reviews() if packed is not None else
                   [{"node_id": row["node_id"], "review": json.loads(row["payload_json"]), "created_at": row["created_at"]} for row in connection.execute("SELECT * FROM reviews WHERE snapshot_id=? ORDER BY id", (snapshot,))])
        context = json.loads(frozen["context_json"])
        execution_pages = context.get("upstream_run_context", {}).get("parameters", {}).get("page_numbers")
        registered_pages = {page["page_number"] for page in registry["pages"]}
        if execution_pages is not None and (not isinstance(execution_pages,list) or not set(execution_pages).issubset(registered_pages)):
            raise ValueError("execution page scope differs from snapshot registry")
        processing_scope = {"execution_page_numbers": execution_pages,
            "registered_page_count": len(registered_pages),
            "outside_execution_scope_page_numbers": sorted(registered_pages-set(execution_pages)) if execution_pages is not None else None,
            "inventory_complete": False,
            "accepted_branch_record_count": sum(junction.get("state") == "accepted" and junction.get("relation_type") in {"projected_branch","projected_native_branch","projected_fitting_body_branch"} for junction in network["junctions"]),
            "identified_projected_equipment_count": sum(item.get('authority',{}).get('projected_body_tag_identity_established') is True for item in hvac['items']),
            "accepted_equipment_port_binding_count": sum(item.get("authority",{}).get("equipment_port_binding_established") is True for item in hvac["items"])}
        result = {"schema_version": "0.1.0", "layer": "mep_snapshot_inspection", "project_id": project, "document_id": document,
            "snapshot_id": snapshot, "versions": versions, "source_sha256": frozen["source_sha256"], "context": context, "processing_scope": processing_scope,
            "document": registry["document"], "pages": [{key: page[key] for key in ("page_ref", "page_number", "page_size_display", "role")} for page in registry["pages"]],
            "rows": rows, "selected_id": selected_row["id"] if selected_row else None, "evidence": evidence, "reviews": reviews,
            "network_summary": network["summary"], "network_coverage": network["coverage"], "hvac_summary": hvac["summary"], "hvac_coverage": hvac["coverage"],
            "runs": network["runs"], "junctions": network["junctions"], "networks": network["networks"],
            "bounded_outcomes":outcomes,"outcome_artifact_authority":outcome_artifact.get("authority",{}) if outcome_artifact else None,
            "outcome_coverage":outcome_artifact.get("coverage",{}) if outcome_artifact else None,
            "outcome_summary":outcome_artifact.get("summary",{}) if outcome_artifact else None,
            "quantity_eligible": False, "schedule_role": "drawing_inspection_register_not_declared_schedule"}
    if compare_to:
        previous = load_review(database,project=project,document=document,snapshot=compare_to)
        result["comparison"] = compare_reviews(previous,result)
    else:
        result["comparison"] = None
    return result


def _row_refs(row):
    refs = {row["id"]}
    for key in ("route_target_refs","target_refs","source_fragment_refs","source_occurrence_refs","source_observation_refs","attribute_relation_refs","source_relation_refs"):
        refs.update(row["payload"].get(key,[]))
    for attribute in row["attributes"]:
        refs.add(attribute["id"])
        if attribute.get("proposal_ref"):
            refs.add(attribute["proposal_ref"])
        for key in ("proposal_evidence_refs","binding_evidence_refs"):
            refs.update(attribute.get(key,[]))
    for mark in row["marks"]:
        refs.add(mark["source_ref"])
        refs.update(mark.get("source_primitive_refs",[]))
    return refs


def _changes(before, after, pointer=""):
    """Exact JSON field deltas; array changes retain complete ordered evidence."""
    if before == after:
        return []
    if isinstance(before,dict) and isinstance(after,dict):
        changes = []
        for key in sorted(before.keys() | after.keys()):
            path = pointer + "/" + key.replace("~","~0").replace("/","~1")
            if key not in before or key not in after:
                changes.append({"pointer":path,"before_present":key in before,"after_present":key in after,
                                "before":before.get(key),"after":after.get(key)})
            else:
                changes.extend(_changes(before[key],after[key],path))
        return changes
    return [{"pointer":pointer or "/","before_present":True,"after_present":True,"before":before,"after":after}]


def _stable_json(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True)


def _item_identity(row):
    """A complete exact native/target signature, never a tag or coordinate match."""
    pages = sorted({mark["page_ref"] for mark in row["marks"]})
    targets = sorted(set(row["payload"].get("route_target_refs",row["payload"].get("target_refs",[]))))
    if targets and pages:
        return _stable_json([row["kind"],pages,"targets",targets])
    native = sorted({(mark["page_ref"],ref) for mark in row["marks"] for ref in mark.get("source_primitive_refs",[])})
    observations = sorted(set(row["payload"].get("source_observation_refs",[])))
    if native:
        return _stable_json([row["kind"],"native",native])
    if observations and pages:
        return _stable_json([row["kind"],pages,"observations",observations])
    return None


def _attribute_semantics(record):
    return {key:record.get(key) for key in ("state","relation_type","page_ref","target_kind","target_refs")} | {
        "candidate":{key:value for key,value in record.get("candidate",{}).items() if key!="raw_text"}}


def _attribute_identity(record):
    targets = sorted(set(record.get("target_refs",[]) or record.get("target_fragment_refs",[])))
    if not targets or not record.get("page_ref"):
        return None
    return _stable_json([record["page_ref"],record.get("target_kind"),targets,record.get("relation_type"),
                         _attribute_semantics(record)["candidate"]])


def _compare_records(before, after, *, same_source, identity=None, semantics=None):
    old = {record["id"]:record for record in before}
    new = {record["id"]:record for record in after}
    if len(old)!=len(before) or len(new)!=len(after):
        raise ValueError("comparison requires unique source record IDs")
    pairs = [(ref,ref,"exact_source_id") for ref in sorted(old.keys() & new.keys())] if same_source else []
    used_old,used_new = {pair[0] for pair in pairs},{pair[1] for pair in pairs}
    if same_source and identity:
        old_keys,new_keys = {},{}
        for records,keys in ((before,old_keys),(after,new_keys)):
            for record in records:
                key = identity(record)
                if key is not None:
                    keys.setdefault(key,[]).append(record["id"])
        for key in sorted(old_keys.keys() & new_keys.keys()):
            left,right = old_keys[key],new_keys[key]
            if len(left)==len(right)==1 and left[0] not in used_old and right[0] not in used_new:
                pairs.append((left[0],right[0],"unique_exact_target_or_native_identity"))
                used_old.add(left[0])
                used_new.add(right[0])
    pairs += [(ref,None,"no_exact_identity_match") for ref in sorted(old.keys()-used_old)]
    pairs += [(None,ref,"no_exact_identity_match") for ref in sorted(new.keys()-used_new)]
    records = []
    for old_id,new_id,match in pairs:
        left,right = old.get(old_id),new.get(new_id)
        changes = _changes(left["content"],right["content"]) if left and right else []
        state = "added" if left is None else "removed" if right is None else "changed" if changes else "unchanged"
        record = {"before_id":old_id,"after_id":new_id,"change":state,"match_basis":match,
                  "before":left,"after":right,"field_changes":changes}
        if semantics and left and right:
            record["outcome_changed"] = semantics(left["content"]) != semantics(right["content"])
        records.append(record)
    return {"counts":{state:sum(record["change"]==state for record in records) for state in ("added","removed","changed","unchanged")},"records":records}


def compare_reviews(before, after):
    """Compare frozen evidence records; this is not physical identity matching."""
    same_source = before["source_sha256"] == after["source_sha256"]
    def items(review):
        result = []
        for row in review["rows"]:
            if row["kind"] == "outcome":
                continue
            marks = [{key:value for key,value in mark.items() if key not in {"clip_display","junction_context"}} for mark in row["marks"]]
            result.append({"id":row["id"],"identity":_item_identity(row),"label":row["label"],"schedule_label":row["schedule_label"],
                           "artifact":row["artifact"],"pointer":row["pointer"],"artifact_sha256":row["artifact_sha256"],
                           "content":{"record":row["payload"],"source_geometry":marks,"catalog_rows":row["catalog_rows"]}})
        return result
    def attributes(review):
        indexed = {}
        for row in review["rows"]:
            for attribute in row["attributes"]:
                entry = indexed.setdefault(attribute["id"],{"id":attribute["id"],"row_refs":[],"content":attribute})
                if entry["content"] != attribute:
                    raise ValueError("attribute ID resolves to conflicting records")
                entry["row_refs"].append(row["id"])
        return list(indexed.values())
    def raw(review,key):
        return [{"id":record["id"],"content":record} for record in review.get(key,[])]
    return {"baseline_snapshot_id":before["snapshot_id"],"current_snapshot_id":after["snapshot_id"],
            "same_source_pdf":same_source,"physical_identity_inferred":False,"quantity_comparison_established":False,
            "scope_changes":_changes(before["processing_scope"],after["processing_scope"]),
            "items":_compare_records(items(before),items(after),same_source=same_source,identity=lambda record:record["identity"]),
            "attributes":_compare_records(attributes(before),attributes(after),same_source=same_source,
                                           identity=lambda record:_attribute_identity(record["content"]),semantics=_attribute_semantics),
            "connections":_compare_records(raw(before,"junctions"),raw(after,"junctions"),same_source=same_source),
            "outcomes":_compare_records(raw(before,"bounded_outcomes"),raw(after,"bounded_outcomes"),same_source=same_source)}


def _add_junction_context(rows, junctions):
    """Display explicit frozen junction membership, including abstentions.

    Context is not a connection certificate. Missing member geometry stays
    explicit; no neighbour is discovered through proximity or equal labels.
    """
    indexed = {row["id"]: row for row in rows}
    for row in rows:
        for mark in row["marks"]:
            context = []
            for junction in junctions:
                if (row["id"] not in junction.get("segment_refs", [])
                        or mark.get('role') == 'inferred_fitting_body_and_port_closeup'
                        or junction.get("page_ref") != mark["page_ref"]
                        or junction.get("relation_type") not in {"projected_collinear_boundary_join", "projected_native_bend","projected_native_branch","projected_fitting_body_branch"}):
                    continue
                members = []
                missing = []
                for ref in junction["segment_refs"]:
                    candidates = [member for member in indexed.get(ref, {}).get("marks", [])
                                  if member["page_ref"] == mark["page_ref"] and member.get("points_display")]
                    if len(candidates) != 1:
                        missing.append(ref)
                        continue
                    member = candidates[0]
                    members.append({"row_id": ref, "source_ref": member["source_ref"],
                                    "points_display": member["points_display"],
                                    "source_primitive_refs": member["source_primitive_refs"]})
                context.append({"junction": junction, "member_segments": members,
                                "missing_or_ambiguous_member_geometry_refs": missing,
                                "display_context_complete": not missing,
                                "presentation_establishes_physical_continuity": False})
            mark["junction_context"] = context
            mark["clip_display"] = mark_clip(mark)


def mark_clip(mark):
    points = mark.get("points_display")
    box = mark.get("bbox_display") if not points else [min(p[0] for p in points), min(p[1] for p in points), max(p[0] for p in points), max(p[1] for p in points)]
    box = list(box)
    for entry in mark.get("junction_context", []):
        context_points = [point for member in entry["member_segments"] for point in member["points_display"]]
        context_points.extend(entry["junction"].get("centreline_points_display", []))
        context_points.extend(entry["junction"].get("port_points_display", []))
        if entry["junction"].get("point_display"):
            context_points.append(entry["junction"]["point_display"])
        for x, y in context_points:
            box = [min(box[0], x), min(box[1], y), max(box[2], x), max(box[3], y)]
    pad = mark.get('presentation_padding_display', max(70, min(220, max(box[2]-box[0], box[3]-box[1]) * .15)))
    width, height = mark["page_size_display"]
    return [max(0, box[0]-pad), max(0, box[1]-pad), min(width, box[2]+pad), min(height, box[3]+pad)]


def source_crop(source, review, row, *, mark_index=0):
    """Render an exact frozen source region; no discovery or coordinate guessing."""
    import fitz
    source = Path(source)
    with source.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != review["source_sha256"]:
            raise ValueError("source PDF differs from selected snapshot")
    mark = row["marks"][mark_index]
    with fitz.open(source) as pdf:
        page = pdf[mark["page_number"] - 1]
        if any(abs(a-b) > .05 for a,b in zip((page.rect.width, page.rect.height), mark["page_size_display"])):
            raise ValueError("source display frame differs from frozen geometry")
        clip = fitz.Rect(mark["clip_display"])
        scale = min(12 if mark.get('role') == 'inferred_fitting_body_and_port_closeup' else 2,
                    1300 / max(clip.width, clip.height))
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
        return {"png": pix.tobytes("png"), "clip_display": list(clip), "mark": mark}


class _ReviewProjection:
    """Request-local, read-only projections; never hydrate unrelated row bodies.

    Large immutable artifacts are hash checked once, then SQLite's JSON reader
    selects only requested fields/pointers. No persistent cache or new index is
    needed, including for historical payload-null nodes.
    """
    def __init__(self, connection, snapshot):
        self.connection, self.snapshot = connection, snapshot
        self.bodies, self.values = {}, {}

    def field(self, artifact, path):
        if artifact not in self.bodies:
            try:
                body = _artifact_body(self.connection,self.snapshot,artifact)
            except ValueError as error:
                if str(error).startswith('snapshot lacks'):
                    return None
                raise
            self.bodies[artifact] = body.decode('utf-8')
        value = self.connection.execute('SELECT json_extract(?,?)', (self.bodies[artifact], path)).fetchone()[0]
        return json.loads(value) if isinstance(value, str) and value[:1] in ('{', '[') else value

    def nodes(self, artifact, collection, *, refs=None, incoming=None):
        clauses = ['sa.snapshot_id=?', 'sa.name=?', 'n.pointer LIKE ?', "instr(substr(n.pointer,?),'/')=0"]
        args = [self.snapshot, artifact, f'/{collection}/%', len(collection)+3]
        if refs is not None:
            if not refs:
                return []
            clauses.append('n.source_id IN ('+','.join('?' for _ in refs)+')')
            args.extend(sorted(refs))
        if incoming is not None:
            if not incoming:
                return []
            clauses.append('EXISTS (SELECT 1 FROM edges e WHERE e.node_id=n.id AND e.target_ref IN ('+','.join('?' for _ in incoming)+'))')
            args.extend(sorted(incoming))
        return [dict(row) for row in self.connection.execute('SELECT n.*,sa.name AS artifact FROM nodes n JOIN snapshot_artifacts sa ON sa.artifact_sha256=n.artifact_sha256 WHERE '+' AND '.join(clauses)+f' ORDER BY CAST(substr(n.pointer,{len(collection)+3}) AS INTEGER)',args)]

    def payload(self, node):
        if node['payload_json'] is not None:
            return json.loads(node['payload_json'])
        if self.connection.execute('PRAGMA user_version').fetchone()[0] == 2:
            value = json.loads(_chunk_node_body(self.connection,node))
            if not isinstance(value,dict) or value.get('id') != node['source_id']:
                raise ValueError('artifact pointer differs from indexed record')
            return value
        path = '$' + ''.join(f'[{part}]' if part.isdigit() else '.'+json.dumps(part.replace('~1','/').replace('~0','~')) for part in node['pointer'].split('/')[1:])
        value = self.field(node['artifact'], path)
        if not isinstance(value,dict) or value.get('id') != node['source_id']:
            raise ValueError('artifact pointer differs from indexed record')
        return value

    def records(self, artifact, collection, **query):
        return [self.payload(node) for node in self.nodes(artifact,collection,**query)]

    def artifact(self, connection, snapshot, name):
        if snapshot != self.snapshot:
            raise ValueError('projection snapshot mismatch')
        return self.values[name]

    def prepare(self, selected_node=None):
        # Project page metadata in SQLite: nested observations stay in the blob.
        registry_body = self.field('sheet-registry','$.document')
        pages = [json.loads(row[0]) for row in self.connection.execute("SELECT json_object('page_ref',json_extract(value,'$.page_ref'),'page_number',json_extract(value,'$.page_number'),'page_size_display',json_extract(value,'$.page_size_display'),'role',json_extract(value,'$.role')) FROM json_each(?, '$.pages')", (self.bodies['sheet-registry'],))]
        self.values['sheet-registry'] = {'document':registry_body,'pages':pages}
        for name in ('network-hierarchy','hvac-inventory','review-outcomes'):
            self.values[name] = {key:self.field(name,'$.'+key) for key in ('summary','document','authority','layer')}
            self.values[name]['coverage'] = {'projection':'summary_and_selected_record_only','full_artifact_retained':True}
        network,hvac,outcomes = (self.values[name] for name in ('network-hierarchy','hvac-inventory','review-outcomes'))
        network.update(segments=[],runs=[],networks=[],junctions=[],scoped_traces=[])
        hvac.update(items=[],source_observations=[])
        outcomes['outcomes'] = []
        self.values['item-catalog'] = {'item_occurrences':[],'takeoff_lines':[]}
        if selected_node is None:
            return
        record = self.payload(selected_node)
        refs = {record['id']}
        for edge in self.connection.execute('SELECT target_ref FROM edges WHERE node_id=?',(selected_node['id'],)):
            refs.add(edge[0])
        network['junctions'] = self.records('network-hierarchy','junctions',incoming={record['id']})
        member_refs = {ref for joint in network['junctions'] for ref in joint.get('segment_refs',[])}
        if selected_node['artifact']=='network-hierarchy':
            member_refs.add(record['id'])
        network['segments'] = self.records('network-hierarchy','segments',refs=member_refs)
        for key in ('runs','networks'):
            network[key] = self.records('network-hierarchy',key,incoming={record['id']})
        trace_refs = {ref for item in network['segments'] for ref in item.get('scoped_trace_refs',[])}
        network['scoped_traces'] = self.records('network-hierarchy','scoped_traces',refs=trace_refs)
        hvac['items'] = self.records('hvac-inventory','items',refs={record['id']})
        if selected_node['artifact']=='review-outcomes':
            hvac['items'] = self.records('hvac-inventory','items',incoming=set(record.get('subject_refs',[])))
        observation_refs = {ref for item in hvac['items'] for ref in item.get('source_observation_refs',[])}
        hvac['source_observations'] = self.records('hvac-inventory','source_observations',refs=observation_refs)
        outcomes['outcomes'] = ([record] if selected_node['artifact']=='review-outcomes' else
                                self.records('review-outcomes','outcomes',incoming=refs))
        targets = {ref for item in network['segments']+hvac['items'] for ref in item.get('route_target_refs',item.get('target_refs',[]))}
        catalog = self.values['item-catalog']
        catalog['item_occurrences'] = self.records('item-catalog','item_occurrences',incoming=targets)
        catalog['takeoff_lines'] = self.records('item-catalog','takeoff_lines',incoming={item['id'] for item in catalog['item_occurrences']})


class _PackedReviewProjection:
    """Exact selected-row projection built only through schema-v3 store APIs."""

    def __init__(self, reader):
        self.reader = reader
        self.store = reader.store
        self.snapshot = reader.frozen['id']
        self.scope = {'project_id':reader.frozen['project_id'],
                      'document_id':reader.frozen['document_id'],
                      'snapshot_id':self.snapshot}
        self.values = {}

    def records(self, artifact, collection, *, refs=None, incoming=None):
        try:
            rows = self.store.collection(**self.scope, name=artifact,
                collection=collection, source_ids=refs, incoming_refs=incoming)
        except KeyError:
            return []
        return [row['payload'] for row in rows]

    def artifact(self, _connection, snapshot, name):
        if snapshot != self.snapshot:
            raise ValueError('projection snapshot mismatch')
        return self.values[name]

    def _header(self, name):
        try:
            artifact = self.reader.artifact(None, self.snapshot, name)
        except ValueError as error:
            if str(error).startswith('snapshot lacks'):
                return None
            raise
        return {key:artifact.get(key) for key in
                ('summary','document','authority','layer','coverage') if key in artifact}

    def prepare(self, selected_node=None):
        registry = self.reader.artifact(None, self.snapshot, 'sheet-registry')
        self.values['sheet-registry'] = {
            'document':registry['document'],
            'pages':[{key:page[key] for key in
                      ('page_ref','page_number','page_size_display','role')}
                     for page in registry['pages']]}
        network = self._header('network-hierarchy') or {}
        hvac = self._header('hvac-inventory') or {}
        outcomes = self._header('review-outcomes') or {}
        for value in (network,hvac,outcomes):
            value['coverage'] = {'projection':'summary_and_selected_record_only',
                                 'full_artifact_retained':True}
        network.update(segments=[],runs=[],networks=[],junctions=[],scoped_traces=[])
        hvac.update(items=[],source_observations=[])
        outcomes['outcomes'] = []
        self.values.update({'network-hierarchy':network, 'hvac-inventory':hvac,
                            'review-outcomes':outcomes,
                            'item-catalog':{'item_occurrences':[],'takeoff_lines':[]}})
        if selected_node is None:
            return
        record = selected_node['payload']
        refs = {record['id']} | {edge['target_ref'] for edge in selected_node['references']}
        network['junctions'] = self.records('network-hierarchy','junctions',incoming={record['id']})
        member_refs = {ref for joint in network['junctions']
                       for ref in joint.get('segment_refs',[])}
        if selected_node['artifact'] == 'network-hierarchy':
            member_refs.add(record['id'])
        network['segments'] = self.records('network-hierarchy','segments',refs=member_refs)
        for key in ('runs','networks'):
            network[key] = self.records('network-hierarchy',key,incoming={record['id']})
        trace_refs = {ref for item in network['segments']
                      for ref in item.get('scoped_trace_refs',[])}
        network['scoped_traces'] = self.records(
            'network-hierarchy','scoped_traces',refs=trace_refs)
        hvac['items'] = self.records('hvac-inventory','items',refs={record['id']})
        if selected_node['artifact'] == 'review-outcomes':
            hvac['items'] = self.records('hvac-inventory','items',
                                         incoming=set(record.get('subject_refs',[])))
        observation_refs = {ref for item in hvac['items']
                            for ref in item.get('source_observation_refs',[])}
        hvac['source_observations'] = self.records(
            'hvac-inventory','source_observations',refs=observation_refs)
        outcomes['outcomes'] = ([record] if selected_node['artifact']=='review-outcomes'
            else self.records('review-outcomes','outcomes',incoming=refs))
        targets = {ref for item in network['segments']+hvac['items']
                   for ref in item.get('route_target_refs',item.get('target_refs',[]))}
        catalog = self.values['item-catalog']
        catalog['item_occurrences'] = self.records(
            'item-catalog','item_occurrences',incoming=targets)
        catalog['takeoff_lines'] = self.records(
            'item-catalog','takeoff_lines',
            incoming={item['id'] for item in catalog['item_occurrences']})


def _list_sql(projection):
    """SQL extracts compact fields before LIMIT; row payloads never reach Python."""
    pages = {page['page_ref']:page['page_number'] for page in projection.values['sheet-registry']['pages']}
    # Preserve the full reader's historical strict-port assessment label using
    # only exact frozen body/tag relations already attached to an HVAC item.
    labels = {}
    for row in projection.connection.execute("""SELECT o.source_id,json_extract(identity.payload_json,'$.equipment_tag') AS tag
      FROM snapshot_artifacts hsa JOIN nodes h ON h.artifact_sha256=hsa.artifact_sha256
      JOIN edges he ON he.node_id=h.id AND he.field='/projected_identity_relation_refs'
      JOIN nodes identity INDEXED BY nodes_source ON identity.source_id=he.target_ref
      JOIN snapshot_artifacts isa ON isa.artifact_sha256=identity.artifact_sha256 AND isa.snapshot_id=hsa.snapshot_id AND isa.name='projected-identity-bindings'
      JOIN edges oe ON oe.target_ref=json_extract(identity.payload_json,'$.proposal_ref') AND oe.field='/subject_refs'
      JOIN nodes o ON o.id=oe.node_id
      JOIN snapshot_artifacts osa ON osa.artifact_sha256=o.artifact_sha256 AND osa.snapshot_id=hsa.snapshot_id AND osa.name='review-outcomes'
      WHERE hsa.snapshot_id=? AND hsa.name='hvac-inventory' AND h.pointer LIKE '/items/%'
        AND instr(substr(h.pointer,8),'/')=0 AND o.pointer LIKE '/outcomes/%' AND instr(substr(o.pointer,11),'/')=0
      ORDER BY CAST(substr(h.pointer,8) AS INTEGER),he.target_ref""",(projection.snapshot,)):
        labels.setdefault(row['source_id'],f"{row['tag']}: prior strict port assessment; current body/tag identified in 2D")
    # Fallback observation/occurrence page references support older snapshots.
    return """WITH indexed AS (
      SELECT n.*,sa.name AS artifact,coalesce(n.payload_json,
        CASE WHEN sa.name='review-outcomes' THEN json_extract(:outcome_body,'$.outcomes['||substr(n.pointer,11)||']') END) AS body
      FROM nodes n JOIN snapshot_artifacts sa ON sa.artifact_sha256=n.artifact_sha256
      WHERE sa.snapshot_id=:snapshot AND sa.name IN ('network-hierarchy','hvac-inventory','review-outcomes') AND (
        (sa.name='network-hierarchy' AND n.pointer LIKE '/segments/%' AND instr(substr(n.pointer,11),'/')=0) OR
        (sa.name='hvac-inventory' AND n.pointer LIKE '/items/%' AND instr(substr(n.pointer,8),'/')=0) OR
        (sa.name='review-outcomes' AND n.pointer LIKE '/outcomes/%' AND instr(substr(n.pointer,11),'/')=0))
    ), candidates AS (
      SELECT n.id AS node_id,n.source_id AS id,n.pointer,n.artifact_sha256,n.state,n.artifact,
        CASE n.artifact WHEN 'network-hierarchy' THEN 'segment' WHEN 'hvac-inventory' THEN 'hvac' ELSE 'outcome' END AS kind,
        coalesce(json_extract(:labels,'$."'||n.source_id||'"'),json_extract(n.body,'$.candidate.tag'),json_extract(n.body,'$.item_class'),json_extract(n.body,'$.label'),
          CASE WHEN n.artifact='network-hierarchy' THEN
            (SELECT group_concat(raw_text,'; ') FROM (SELECT DISTINCT coalesce(json_extract(a.payload_json,'$.candidate.raw_text'),'route_system') AS raw_text
             FROM edges e JOIN nodes a INDEXED BY nodes_source ON a.source_id=e.target_ref
             JOIN snapshot_artifacts asa ON asa.artifact_sha256=a.artifact_sha256 AND asa.snapshot_id=:snapshot AND asa.name='attribute-bindings'
             WHERE e.node_id=n.id AND a.state='accepted' AND json_extract(a.payload_json,'$.relation_type')='route_system'
               AND e.field IN ('/attribute_relation_refs','/source_relation_refs') ORDER BY raw_text)) END,
          CASE n.artifact WHEN 'network-hierarchy' THEN 'Projected segment' ELSE coalesce(json_extract(n.body,'$.description'),replace(json_extract(n.body,'$.subject_kind'),'_',' '),'Bounded outcome') END) AS label,
        json_extract(n.body,'$.item_type') AS item_type,
        json_extract(n.body,'$.page_refs') AS page_refs_json,
        coalesce(n.page_ref,(SELECT value FROM json_each(n.body,'$.page_refs') ORDER BY json_extract(:pages,'$."'||value||'"') LIMIT 1),
          (SELECT refnode.page_ref FROM edges e JOIN nodes refnode INDEXED BY nodes_source ON refnode.source_id=e.target_ref
           JOIN snapshot_artifacts refsa ON refsa.artifact_sha256=refnode.artifact_sha256 AND refsa.snapshot_id=:snapshot
           WHERE e.node_id=n.id AND e.field IN ('/source_occurrence_refs','/source_observation_refs')
           ORDER BY json_extract(:pages,'$."'||refnode.page_ref||'"') LIMIT 1)) AS page_ref
      FROM indexed n
    ), numbered AS (
      SELECT *,coalesce(json_extract(:pages,'$."'||page_ref||'"'),999999) AS page_number,
        row_number() OVER (ORDER BY coalesce(json_extract(:pages,'$."'||page_ref||'"'),999999),kind,id) AS ordinal FROM candidates
    ), filtered AS (
      SELECT * FROM numbered WHERE (:kind='all' OR kind=:kind) AND
        (:search='' OR instr(lower(label||' '||id||' '||coalesce(state,'')||' '||printf('R%03d',ordinal)),lower(:search))>0)
    ) """, {'snapshot':projection.snapshot,'pages':json.dumps(pages),'labels':json.dumps(labels),'outcome_body':projection.bodies.get('review-outcomes','{}')}


def _packed_list_rows(reader, pages):
    """Build compact app rows from the three bounded schema-v3 collections."""
    specs = (('network-hierarchy','segments','segment'),
             ('hvac-inventory','items','hvac'),
             ('review-outcomes','outcomes','outcome'))
    rows = []
    identity_tags = {}
    source_page = {}
    for artifact,collection,kind in specs:
        try:
            records = reader.store.collection(
                project_id=reader.frozen['project_id'],
                document_id=reader.frozen['document_id'],
                snapshot_id=reader.frozen['id'], name=artifact,
                collection=collection)
        except KeyError:
            records = []
        for node in records:
            node['artifact'], node['kind'] = artifact, kind
            record = node['payload']
            if kind == 'hvac':
                for ref in record.get('projected_identity_relation_refs',[]):
                    identity = reader.indexed('projected-identity-bindings',ref)['payload']
                    if identity.get('proposal_ref') and identity.get('equipment_tag'):
                        identity_tags[identity['proposal_ref']] = identity['equipment_tag']
            rows.append(node)
    page_numbers = {ref:page['page_number'] for ref,page in pages.items()}
    for node in rows:
        record,kind = node['payload'],node['kind']
        refs = list(record.get('page_refs') or [])
        if node.get('page_ref'):
            refs.append(node['page_ref'])
        if not refs:
            for edge in node['references']:
                if edge['field'] not in {'/source_occurrence_refs','/source_observation_refs'}:
                    continue
                target = edge['target_ref']
                if target not in source_page:
                    matches = reader.source_records(target)
                    source_page[target] = min((row['page_ref'] for row in matches
                                               if row['page_ref'] in page_numbers),
                                              key=page_numbers.get,default=None)
                if source_page[target]:
                    refs.append(source_page[target])
        node['_page_refs'] = sorted(set(refs),key=lambda ref:page_numbers.get(ref,999999))
        node['_page_number'] = min((page_numbers.get(ref,999999) for ref in refs),default=999999)
        label = record.get('candidate',{}).get('tag') or record.get('item_class') or record.get('label')
        if kind == 'segment' and not label:
            labels = set()
            for ref in record.get('attribute_relation_refs',record.get('source_relation_refs',[])):
                attribute = reader.indexed('attribute-bindings',ref)['payload']
                if attribute.get('state') == 'accepted' and attribute.get('relation_type') == 'route_system':
                    labels.add(attribute.get('candidate',{}).get('raw_text') or 'route_system')
            label = '; '.join(sorted(labels)) or 'Projected segment'
        if kind == 'outcome':
            tags = [identity_tags[ref] for ref in record.get('subject_refs',[]) if ref in identity_tags]
            label = (f'{tags[0]}: prior strict port assessment; current body/tag identified in 2D'
                     if tags else label or record.get('description')
                     or record.get('subject_kind','Bounded outcome').replace('_',' '))
        node['_label'] = label
    rows.sort(key=lambda row:(row['_page_number'],row['kind'],row['source_id']))
    for ordinal,row in enumerate(rows,1):
        row['_ordinal'] = ordinal
    return rows


def _compact_packed_row(node, pages):
    record = node['payload']
    return {'node_id':node['id'], 'id':node['source_id'], 'pointer':node['pointer'],
            'artifact_sha256':node['artifact_sha256'], 'state':node['state'],
            'artifact':node['artifact'], 'kind':node['kind'], 'label':node['_label'],
            'page_ref':node['_page_refs'][0] if node['_page_refs'] else None,
            'page_number':node['_page_number'],
            'schedule_label':f"R{node['_ordinal']:03d}",
            'marks':[{'page_number':pages[ref]['page_number']}
                     for ref in node['_page_refs'] if ref in pages],
            'payload':{'item_type':record.get('item_type')}}


def _packed_page_comparison(reader, baseline, current, database, project, document):
    same_source = baseline['source_sha256'] == current['source_sha256']
    channels = {'items':(('network-hierarchy','segments'),('hvac-inventory','items')),
                'attributes':(('attribute-bindings','relations'),),
                'connections':(('network-hierarchy','junctions'),),
                'outcomes':(('review-outcomes','outcomes'),)}
    comparison = {'baseline_snapshot_id':baseline['id'],
        'current_snapshot_id':current['snapshot_id'],'same_source_pdf':same_source,
        'physical_identity_inferred':False,'quantity_comparison_established':False,
        'comparison_scope':'Exact indexed source IDs and record payloads; not semantic coverage. Detail compares only the selected exact ID and incident evidence.',
        'scope_changes':[]}
    scope = {'project_id':project,'document_id':document}
    for channel,collections in channels.items():
        versions = []
        for snapshot_id in (baseline['id'],current['snapshot_id']):
            records = []
            for artifact,collection in collections:
                try:
                    records.extend(reader.store.collection(**scope,snapshot_id=snapshot_id,
                        name=artifact,collection=collection))
                except KeyError:
                    pass
            versions.append({row['source_id']:row for row in records})
        old,new = versions
        counts = {key:0 for key in ('added','removed','changed','unchanged','not_compared')}
        if not same_source:
            counts['added'],counts['removed'] = len(new),len(old)
        else:
            for ref in old.keys()-new.keys(): counts['removed'] += 1
            for ref in new.keys()-old.keys(): counts['added'] += 1
            for ref in old.keys()&new.keys():
                if old[ref]['payload_length'] > 65536 or new[ref]['payload_length'] > 65536:
                    counts['not_compared'] += 1
                else:
                    counts['unchanged' if old[ref]['payload']==new[ref]['payload'] else 'changed'] += 1
        comparison[channel] = {'counts':counts,'records':[]}
    if current['selected']:
        ref,artifact = current['selected']['id'],current['selected']['artifact']
        try:
            exists = bool(reader.store.collection(**scope,snapshot_id=baseline['id'],
                name=artifact,collection={'network-hierarchy':'segments',
                    'hvac-inventory':'items','review-outcomes':'outcomes'}[artifact],
                source_ids={ref},include_payload=False))
        except KeyError:
            exists = False
        if same_source and exists:
            before = load_review_page(database,project=project,document=document,
                                      snapshot=baseline['id'],selected=ref,limit=1)
            detail = compare_reviews({**before,'rows':[before['selected']]},
                                     {**current,'rows':[current['selected']]})
            for channel in channels:
                comparison[channel]['records'] = detail[channel]['records']
            comparison['scope_changes'] = detail['scope_changes']
        else:
            comparison['selected_match_state'] = 'no_exact_source_id_match'
    return comparison


def _load_review_page_v3(database, *, project, document, snapshot, selected,
                         compare_to, limit, offset, kind, search):
    with PackedProjectStore(database) as store:
        versions = store.snapshots(project_id=project,document_id=document)
        snapshot = snapshot or next((row['id'] for row in versions if row['active']),None)
        if snapshot not in {row['id'] for row in versions}:
            raise ValueError('snapshot is outside this project/document')
        reader = _PackedReviewReader(store,project=project,document=document,snapshot=snapshot)
        projection = _PackedReviewProjection(reader)
        projection.prepare()
        pages = {page['page_ref']:page for page in projection.values['sheet-registry']['pages']}
        candidates = _packed_list_rows(reader,pages)
        facets = {name:sum(row['kind']==name for row in candidates)
                  for name in ('segment','hvac','outcome')}
        facets = {key:value for key,value in facets.items() if value}
        needle = search.lower()
        filtered = [row for row in candidates if (kind=='all' or row['kind']==kind) and
                    (not needle or needle in (' '.join((row['_label'] or '',row['source_id'],
                        row.get('state') or '',f"R{row['_ordinal']:03d}"))).lower())]
        rows = [_compact_packed_row(row,pages) for row in filtered[offset:offset+limit]]
        selected_node = None
        if selected:
            matches = [row for row in candidates if selected in (row['source_id'],row['id'])]
            if len(matches) != 1:
                raise ValueError('inspection row does not belong to frozen snapshot')
            selected_node = matches[0]
            projection.prepare(selected_node)
        result = load_review(database,project=project,document=document,snapshot=snapshot,
                             selected=selected,_projection=projection)
        selected_row = next((row for row in result['rows']
                             if row['id']==result['selected_id']),None)
        if selected_row:
            selected_row['schedule_label'] = f"R{selected_node['_ordinal']:03d}"
            selected_row['list_offset'] = ((selected_node['_ordinal']-1)//limit)*limit
        result.update(rows=rows,selected=selected_row,pagination={
            'total':len(candidates),'matching':len(filtered),'limit':limit,'offset':offset,
            'kind':kind,'search':search,'facets':facets},
            read_scope='paginated_list_and_exact_selected_row',
            outcome_count=facets.get('outcome',0))
        junctions = projection.records('network-hierarchy','junctions')
        hvac_items = [row['payload'] for row in candidates if row['kind']=='hvac']
        result['processing_scope'].update(
            accepted_branch_record_count=sum(row.get('state')=='accepted' and
                row.get('relation_type') in {'projected_branch','projected_native_branch',
                    'projected_fitting_body_branch'} for row in junctions),
            identified_projected_equipment_count=sum(row.get('authority',{}).get(
                'projected_body_tag_identity_established') is True for row in hvac_items),
            accepted_equipment_port_binding_count=sum(row.get('authority',{}).get(
                'equipment_port_binding_established') is True for row in hvac_items))
        result['network_summary'] = projection.values['network-hierarchy'].get('summary') or {}
        result['hvac_summary'] = projection.values['hvac-inventory'].get('summary') or {}
        if compare_to:
            baseline = next((row for row in versions if row['id']==compare_to),None)
            if baseline is None:
                raise ValueError('baseline snapshot is outside this project/document')
            result['comparison'] = _packed_page_comparison(
                reader,baseline,result,database,project,document)
        return result


def load_review_page(database, *, project, document, snapshot=None, selected=None,
                     compare_to=None, limit=50, offset=0, kind='all', search=''):
    """Opt-in app path: SQL-paginated summaries plus one exact row on demand."""
    if not 1 <= limit <= 100 or offset < 0 or kind not in {'all','segment','hvac','outcome'} or len(search)>200:
        raise ValueError('invalid inspection page parameters')
    with closing(sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro',uri=True)) as probe:
        if probe.execute('PRAGMA user_version').fetchone()[0] == 3:
            return _load_review_page_v3(database,project=project,document=document,
                snapshot=snapshot,selected=selected,compare_to=compare_to,limit=limit,
                offset=offset,kind=kind,search=search)
    with closing(sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro',uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        versions = [dict(row) for row in connection.execute('SELECT s.id,s.created_at,s.source_sha256,(a.snapshot_id=s.id) AS active FROM snapshots s LEFT JOIN active_documents a ON a.project_id=s.project_id AND a.document_id=s.document_id WHERE s.project_id=? AND s.document_id=? ORDER BY s.created_at DESC,s.id',(project,document))]
        snapshot = snapshot or next((row['id'] for row in versions if row['active']),None)
        if snapshot not in {row['id'] for row in versions}:
            raise ValueError('snapshot is outside this project/document')
        projection = _ReviewProjection(connection,snapshot)
        projection.prepare()
        sql,args = _list_sql(projection)
        args.update(kind=kind,search=search,limit=limit,offset=offset)
        kinds = {'network-hierarchy':'segment','hvac-inventory':'hvac','review-outcomes':'outcome'}
        facets = {kinds[row['artifact']]:row['count'] for row in connection.execute(sql+'SELECT artifact,count(*) AS count FROM indexed GROUP BY artifact',args)}
        total = sum(facets.values())
        matching = (connection.execute(sql+'SELECT count(*) FROM filtered',args).fetchone()[0] if search else
                    total if kind=='all' else facets.get(kind,0))
        totals = {'total':total,'matching':matching}
        def compact(row):
            item = dict(row)
            item['schedule_label'] = f"R{item.pop('ordinal'):03d}"
            page_numbers = {page['page_ref']:page['page_number'] for page in projection.values['sheet-registry']['pages']}
            refs = json.loads(item.pop('page_refs_json') or '[]') or [item['page_ref']]
            item['marks'] = [{'page_number':number} for number in sorted({page_numbers[ref] for ref in refs if ref in page_numbers})]
            item['payload'] = {'item_type':item.pop('item_type')}
            return item
        rows = [compact(row) for row in connection.execute(sql+'SELECT * FROM filtered ORDER BY ordinal LIMIT :limit OFFSET :offset',args)]
        selected_summary = None
        if selected:
            found = connection.execute(sql+'SELECT * FROM numbered WHERE id=:selected OR node_id=:selected',{**args,'selected':selected}).fetchall()
            if len(found)!=1:
                raise ValueError('inspection row does not belong to frozen snapshot')
            selected_summary = compact(found[0])
            node = dict(connection.execute('SELECT * FROM nodes WHERE id=?',(selected_summary['node_id'],)).fetchone())
            node['artifact'] = selected_summary['artifact']
            projection.prepare(node)
        result = load_review(database,project=project,document=document,snapshot=snapshot,selected=selected,_projection=projection)
        selected_row = next((row for row in result['rows'] if row['id']==result['selected_id']),None)
        if selected_row:
            selected_row['schedule_label'] = selected_summary['schedule_label']
            selected_row['list_offset'] = ((int(selected_summary['schedule_label'][1:])-1)//limit)*limit
        result.update(rows=rows,selected=selected_row,pagination={**totals,'limit':limit,'offset':offset,'kind':kind,'search':search,'facets':facets},
                      read_scope='paginated_list_and_exact_selected_row',outcome_count=facets.get('outcome',0))
        # Full-snapshot counters come from immutable summaries, never the detail subset.
        network = projection.values['network-hierarchy']['summary'] or {}
        hvac = projection.values['hvac-inventory']['summary'] or {}
        def count_records(artifact,collection,condition):
            return connection.execute("SELECT count(*) FROM nodes n JOIN snapshot_artifacts sa ON sa.artifact_sha256=n.artifact_sha256 WHERE sa.snapshot_id=? AND sa.name=? AND n.pointer LIKE ? AND instr(substr(n.pointer,?),'/')=0 AND "+condition,(snapshot,artifact,'/'+collection+'/%',len(collection)+3)).fetchone()[0]
        result['processing_scope'].update(
            accepted_branch_record_count=count_records('network-hierarchy','junctions',"n.state='accepted' AND json_extract(n.payload_json,'$.relation_type') IN ('projected_branch','projected_native_branch','projected_fitting_body_branch')"),
            identified_projected_equipment_count=count_records('hvac-inventory','items',"json_extract(n.payload_json,'$.authority.projected_body_tag_identity_established')=1"),
            accepted_equipment_port_binding_count=count_records('hvac-inventory','items',"json_extract(n.payload_json,'$.authority.equipment_port_binding_established')=1"))
        result['network_summary'],result['hvac_summary'] = network,hvac
        if compare_to:
            baseline = next((row for row in versions if row['id']==compare_to),None)
            if baseline is None:
                raise ValueError('baseline snapshot is outside this project/document')
            result['comparison'] = _page_comparison(connection,projection,baseline,result,database,project,document)
        return result


def _page_comparison(connection, projection, baseline, current, database, project, document):
    same_source = baseline['source_sha256']==current['source_sha256']
    channels = {'items':(('network-hierarchy','segments'),('hvac-inventory','items')),
                'attributes':(('attribute-bindings','relations'),),
                'connections':(('network-hierarchy','junctions'),),'outcomes':(('review-outcomes','outcomes'),)}
    comparison = {'baseline_snapshot_id':baseline['id'],'current_snapshot_id':current['snapshot_id'],
                  'same_source_pdf':same_source,'physical_identity_inferred':False,'quantity_comparison_established':False,
                  'comparison_scope':'Exact indexed source IDs and record payloads; not semantic coverage. Detail compares only the selected exact ID and incident evidence.',
                  'scope_changes':[]}
    for channel,collections in channels.items():
        conditions = ' OR '.join("(sa.name=? AND n.pointer LIKE ? AND instr(substr(n.pointer,?),'/')=0)" for _ in collections)
        params = [value for artifact,collection in collections for value in (artifact,'/'+collection+'/%',len(collection)+3)]
        artifact_names = ','.join("'"+artifact+"'" for artifact,_ in collections)
        query = 'SELECT n.source_id,n.payload_json,n.id AS node_id FROM nodes n JOIN snapshot_artifacts sa ON sa.artifact_sha256=n.artifact_sha256 WHERE sa.snapshot_id=? AND sa.name IN ('+artifact_names+') AND ('+conditions+')'
        counts = {key:0 for key in ('added','removed','changed','unchanged','not_compared')}
        # SQLite compares payloads in place; Python receives aggregate counts only.
        sql = 'WITH old AS ('+query+'), new AS ('+query+''') SELECT change,count(*) FROM (
          SELECT CASE WHEN old.source_id IS NULL THEN 'added'
            WHEN old.node_id=new.node_id THEN 'unchanged'
            WHEN old.payload_json IS NULL OR new.payload_json IS NULL THEN 'not_compared'
            WHEN old.payload_json=new.payload_json THEN 'unchanged' ELSE 'changed' END AS change
          FROM new LEFT JOIN old ON old.source_id=new.source_id AND ?
          UNION ALL SELECT 'removed' FROM old LEFT JOIN new ON old.source_id=new.source_id AND ? WHERE new.source_id IS NULL
        ) GROUP BY change'''
        for state,count in connection.execute(sql,[baseline['id'],*params,current['snapshot_id'],*params,same_source,same_source]):
            counts[state] = count
        comparison[channel] = {'counts':counts,'records':[]}
    if current['selected']:
        ref = current['selected']['id']
        exists = connection.execute('SELECT 1 FROM nodes n INDEXED BY nodes_source JOIN snapshot_artifacts sa ON sa.artifact_sha256=n.artifact_sha256 WHERE sa.snapshot_id=? AND sa.name=? AND n.source_id=?',(baseline['id'],current['selected']['artifact'],ref)).fetchone()
        if same_source and exists:
            before = load_review_page(database,project=project,document=document,snapshot=baseline['id'],selected=ref,limit=1)
            detail = compare_reviews({**before,'rows':[before['selected']]},{**current,'rows':[current['selected']]})
            for channel in channels:
                comparison[channel]['records'] = detail[channel]['records']
            comparison['scope_changes'] = detail['scope_changes']
        else:
            comparison['selected_match_state'] = 'no_exact_source_id_match'
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("database", "project", "document"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--snapshot")
    parser.add_argument("--compare-to", help="Exact baseline snapshot in the same project/document")
    parser.add_argument("--selected")
    parser.add_argument("--source")
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--crop", action="store_true")
    output_mode.add_argument('--gzip-json', action='store_true', help='Losslessly compress JSON for local app transport')
    parser.add_argument("--mark-index", type=int, default=0)
    parser.add_argument('--page-size', type=int, help='Opt-in SQL-paginated app projection (1..100)')
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--kind', default='all')
    parser.add_argument('--search', default='')
    args = parser.parse_args()
    options = dict(project=args.project,document=args.document,snapshot=args.snapshot,selected=args.selected,compare_to=args.compare_to)
    result = (load_review_page(args.database,**options,limit=args.page_size,offset=args.offset,kind=args.kind,search=args.search)
              if args.page_size is not None else load_review(args.database,**options))
    if args.crop:
        row = result.get('selected') or next(row for row in result["rows"] if row["id"] == result["selected_id"])
        sys.stdout.buffer.write(source_crop(args.source, result, row, mark_index=args.mark_index)["png"])
    elif args.gzip_json:
        body = (json.dumps(result, ensure_ascii=True) + '\n').encode('utf-8')
        sys.stdout.buffer.write(gzip.compress(body, compresslevel=1, mtime=0))
    else:
        print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
