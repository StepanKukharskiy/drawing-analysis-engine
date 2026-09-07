#!/usr/bin/env python3
"""Simple drawing-first MEP review from an exact frozen SQLite snapshot.

Selected source marks are highlighted where clear of source text. Short local
readouts are bounded for legibility; full per-record values and provenance stay
in the companion manifest/app. This renderer never discovers a route, assigns
attributes, sums installed lengths, or invents connections between ports.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

import fitz

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.drawing_engine.project.mep_project_review import load_review
from src.drawing_engine.audit.mep_audit_palette import SYSTEM_COLORS
from src.drawing_engine.audit.render_mep_partial_audit import (
    _Column, _source_text_boxes, _readable_highlight, _readable_anchor, _readable_line_parts, _clip_line,
)

BLUE = (.04, .39, .72)
LIGHT = (.54, .67, .73)
PURPLE = (.53, .22, .62)
AMBER = (.7, .42, .08)
INK = (.08, .15, .22)
MUTED = (.35, .42, .48)


def _route_color(row):
    """Color accepted system identity only; color never supplies identity."""
    field = row.get('display_values', {}).get('system', {})
    if field.get('state') != 'accepted':
        return LIGHT
    candidate = field.get('candidate') or {}
    kind = candidate.get('kind') or {
        'HHWS': 'heating_hot_water_supply', 'HHWR': 'heating_hot_water_return',
        'CHWS': 'chilled_water_supply', 'CHWR': 'chilled_water_return',
    }.get(field.get('text'))
    return SYSTEM_COLORS.get(kind, LIGHT)


def _junction_color(junction, entries):
    colors = {tuple(e['color']) for e in entries if e['row_ref'] in junction.get('segment_refs', [])}
    return next(iter(colors)) if len(colors) == 1 else LIGHT


def _box(points):
    return fitz.Rect(min(p[0] for p in points), min(p[1] for p in points),
                     max(p[0] for p in points), max(p[1] for p in points))


def _paths(mark):
    if mark.get("points_display"):
        return [mark["points_display"]]
    paths = []
    for path in mark.get("evidence_paths", []):
        if path.get("points_display"):
            paths.append(path["points_display"])
        elif path.get("bbox_display"):
            rect = fitz.Rect(path["bbox_display"])
            paths.append([list(p) for p in (rect.tl, rect.tr, rect.br, rect.bl, rect.tl)])
    if not paths and mark.get("bbox_display"):
        rect = fitz.Rect(mark["bbox_display"])
        paths.append([list(p) for p in (rect.tl, rect.tr, rect.br, rect.bl, rect.tl)])
    return paths


def _field(values, key):
    value = values.get(key)
    if isinstance(value, dict):
        if value.get("state") in {"unknown", "abstained", "unresolved"}:
            return "Unknown"
        if value.get("state") == "conflicted":
            return "Conflicting values"
        return str(value.get("text") or "Unknown")
    return "Unknown"


def _display(row, mark):
    values = row.get("display_values", {})
    mark_values = mark.get("display_values", {})
    return {"system": _field(values, "system"), "size": _field(values, "size"),
            "elevation": _field(values, "elevation"),
            "projected_length": _field(mark_values, "projected_length")}


def drawing_entries(review):
    """Select drawing marks, never promote outcomes or infer missing geometry."""
    connected = {ref for network in [*review.get("networks", []), *review.get('runs', [])] if len(network["segment_refs"]) > 1
                 for ref in network["segment_refs"]}
    network_refs = defaultdict(list)
    branch_junctions = defaultdict(list)
    for network in review.get("networks", []):
        for ref in network["segment_refs"]:
            network_refs[ref].append(network["id"])
    for junction in review.get('junctions', []):
        if junction.get('state') == 'accepted' and len(junction.get('port_points_display', [])) >= 3:
            for ref in junction.get('segment_refs', []):
                branch_junctions[(ref, junction['page_ref'])].append(junction)
    entries, omitted = [], []
    for row in review["rows"]:
        if row["kind"] not in {"segment", "hvac"}:
            omitted.append({"row_ref": row["id"], "reason": "technical_outcome_in_app_not_drawing_item"})
            continue
        identified = row.get("payload", {}).get("authority", {}).get("projected_body_tag_identity_established") is True
        for index, mark in enumerate(row["marks"]):
            if mark.get("role") == "inferred_fitting_body_and_port_closeup":
                omitted.append({"row_ref": row["id"], "mark_index": index,
                                "source_ref": mark.get("source_ref"),
                                "reason": "supplemental_fitting_context_not_another_segment"})
                continue
            paths = _paths(mark)
            if not paths:
                omitted.append({"row_ref": row["id"], "mark_index": index, "reason": "no_frozen_geometry_to_mark"})
                continue
            display = _display(row, mark)
            has_attributes = any(display[key] not in {"Unknown", "Conflicting values"} for key in ("system", "size", "elevation"))
            completed = row.get('payload', {}).get('projected_scope_complete') is True
            branches = sorted(branch_junctions[(row['id'], mark['page_ref'])], key=lambda j: j['id'])
            priority_reasons = [name for name, enabled in (
                ('identified_hvac', identified), ('completed_projected_scope', completed),
                ('accepted_multiport_branch_member', bool(branches)), ('recovered_attribute', has_attributes),
                ('connected_network_member', row['id'] in connected)) if enabled]
            priority = (8 if identified else 0) + (8 if completed else 0) + (6 if branches else 0) + (4 if has_attributes else 0) + (2 if row["id"] in connected else 0)
            kind = "identified_hvac" if identified else "route" if row["kind"] == "segment" else "unresolved_hvac_observation"
            points = [p for path in paths for p in path]
            bbox = _box(points)
            # Use an actual point on the displayed path, not its bounding-box centre.
            path = max(paths, key=lambda p: sum(math.dist(a,b) for a,b in zip(p,p[1:])))
            anchor = path[len(path)//2] if len(path) > 2 else [(a+b)/2 for a,b in zip(path[0],path[-1])]
            entries.append({"key": row["id"] + ":" + str(index), "row_ref": row["id"], "mark_index": index,
                "page_ref": mark["page_ref"], "page_number": mark["page_number"],
                "kind": kind, "label": row["label"], "priority": priority,
                "priority_reasons": priority_reasons,
                "priority_branch_junction_refs": [j['id'] for j in branches],
                "closeup_focus_display": deepcopy(branches[0]['port_points_display'][0]) if branches else anchor,
                "closeup_focus_basis": 'frozen_branch_port_context_only' if branches else 'segment_path_anchor',
                "network_refs": sorted(network_refs[row["id"]]), "display_values": display,
                "frozen_display_values": {"row": deepcopy(row.get("display_values", {})), "mark": deepcopy(mark.get("display_values", {}))},
                "paths": deepcopy(paths), "bbox_display": list(bbox), "anchor_display": anchor,
                "source_ref": mark.get("source_ref"), "source_primitive_refs": mark.get("source_primitive_refs", []),
                "artifact": row.get("artifact"), "pointer": row.get("pointer"), "artifact_sha256": row.get("artifact_sha256"),
                "color": PURPLE if identified else _route_color(row) if kind == "route" else AMBER,
                "source_mark": deepcopy(mark), "printed_readouts": [], "quantity_eligible": False})
    per_page = defaultdict(list)
    for entry in entries:
        per_page[entry["page_number"]].append(entry)
    for rows in per_page.values():
        counts = Counter()
        for row in sorted(rows, key=lambda e: (e["kind"], e["anchor_display"][1], e["anchor_display"][0], e["key"])):
            prefix = "H" if row["kind"] == "identified_hvac" else "T" if row["kind"] == "unresolved_hvac_observation" else "R"
            counts[prefix] += 1
            row["short_id"] = prefix + str(counts[prefix])
    return entries, omitted


def _draw_paths(page, paths, boxes, color, width, matrix=None):
    visible_parts=0
    for path in paths:
        points = [fitz.Point(p) * matrix if matrix is not None else fitz.Point(p) for p in path]
        extent = _box(points) + (-width-2, -width-2, width+2, width+2)
        local_boxes = [box for box in boxes if extent.intersects(box)]
        _readable_highlight(page, points, local_boxes, color, width)
        visible_parts += sum(len(_readable_line_parts(a,b,local_boxes,width/2+1)) for a,b in zip(points,points[1:]))
    return visible_parts


def _junction_paths(junction):
    if junction.get("state") != "accepted":
        return []
    # Three unordered port centres never become a line through three ports.
    paths=([junction["centreline_points_display"]] if junction.get("centreline_points_display") else [])
    return paths + [p["points_display"] for p in junction.get("body_paths",[]) if p.get("points_display")]


def _label(page, entry, rect, boxes, occupied, *, size, matrix=None, text=None):
    point = fitz.Point(entry["anchor_display"])
    if matrix is not None:
        point *= matrix
        if not rect.contains(point):
            # A long member's midpoint may lie outside a branch close-up. Use
            # only its actually visible path, never a new line between ports.
            pieces = [_clip_line(fitz.Point(a)*matrix, fitz.Point(b)*matrix, rect)
                      for path in entry['paths'] for a,b in zip(path,path[1:])]
            pieces = [piece for piece in pieces if piece]
            if pieces:
                a,b = max(pieces, key=lambda p: math.dist(*p))
                point = (fitz.Point(a)+fitz.Point(b))/2
    if not rect.contains(point):
        return None
    label = text or entry["short_id"]
    width = fitz.get_text_length(label, fontsize=size) + size*.7
    height = size*1.6
    candidates = [fitz.Rect(point.x+dx*size, point.y+dy*size,
                            point.x+dx*size+width, point.y+dy*size+height)
                  for dx in (-5,-3,1,3) for dy in (-5,-3,1,3)]
    candidates.sort(key=lambda r: math.dist(r.tl + (r.br-r.tl)/2, point))
    tag = next((r for r in candidates if rect.contains(r) and not any(r.intersects(b) for b in [*boxes,*occupied])), None)
    if tag is None:
        return None
    origin = fitz.Point(min(max(point.x, tag.x0),tag.x1), min(max(point.y,tag.y0),tag.y1))
    _readable_highlight(page,[origin,point],boxes,entry["color"],max(.7,size*.055))
    _readable_anchor(page,point,boxes,entry["color"],size*.12,max(.7,size*.055))
    page.draw_rect(tag,color=entry["color"],fill=(1,1,1),width=max(.7,size*.055))
    page.insert_text((tag.x0+size*.3,tag.y0+size*1.12),label,fontsize=size,color=INK)
    occupied.append(tag)
    return tag


def _readout(column, entry):
    values = entry["display_values"]
    title = entry["short_id"] + "  " + (entry["label"] if entry["kind"] == "identified_hvac" else values["system"])
    column.text(title, bold=True, color=entry["color"], gap=2)
    if entry["kind"] == "identified_hvac":
        column.text("HVAC body + tag (2D)",gap=1)
        column.text("Elevation: " + values["elevation"],gap=1)
        column.text("Ports unresolved",size=column.size*.85,color=MUTED,gap=10)
    else:
        column.text("Size: " + values["size"] + "   2D length: " + values["projected_length"],gap=1)
        column.text("Elevation: " + values["elevation"],gap=10)


def _windows(entries, pages, maximum):
    """Bounded presentation windows around priority anchors, not discovery scopes."""
    candidates = []
    for page in pages:
        pending = sorted((e for e in entries if e["page_number"] == page["page_number"] and e["priority"]),
                         key=lambda e:(-e["priority"],e["anchor_display"][1],e["anchor_display"][0],e["key"]))
        page_rect = fitz.Rect(0,0,*page["page_size_display"])
        width = min(page_rect.width, max(400, page_rect.width*.24))
        height = min(page_rect.height, width*.82)
        for _ in range(max(2, math.ceil(maximum / max(1, len(pages))))):
            if not pending:
                break
            anchor = pending[0]["closeup_focus_display"]
            x = min(max(0,anchor[0]-width/2),page_rect.width-width)
            y = min(max(0,anchor[1]-height/2),page_rect.height-height)
            clip = fitz.Rect(x,y,x+width,y+height)
            selected = [e for e in pending if clip.contains(fitz.Point(e["closeup_focus_display"]))]
            pending = [e for e in pending if e not in selected]
            candidates.append({"source_page_number":page["page_number"],"clip_display":list(clip),
                               "priority":sum(e["priority"] for e in selected),"entry_keys":[e["key"] for e in selected],
                               "priority_reasons":sorted({r for e in selected for r in e['priority_reasons']}),
                               "priority_branch_junction_refs":sorted({r for e in selected for r in e['priority_branch_junction_refs']})})
    chosen = sorted(candidates,key=lambda c:(-c["priority"],c["source_page_number"],c["clip_display"]))[:maximum]
    return sorted(chosen,key=lambda c:(c["source_page_number"],c["clip_display"]))


def render_network_drawing_audit(source, output, review, *, preview_pages=None, maximum_closeups=24,
                                 include_isolated_geometry=False, single_page=False, write_manifest=True,
                                 source_opacity=1.0):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("drawing audit must not overwrite source PDF")
    if hashlib.sha256(source.read_bytes()).hexdigest() != review["source_sha256"]:
        raise ValueError("drawing source differs from frozen snapshot")
    if maximum_closeups < 0 or maximum_closeups > 32:
        raise ValueError("closeup budget must be between zero and 32")
    if not 0 < source_opacity <= 1 or (source_opacity != 1 and not single_page):
        raise ValueError('source opacity must be in (0, 1]; fading currently requires a single-page audit')
    registered = {p["page_number"]:p for p in review["pages"]}
    if preview_pages is not None and not set(preview_pages).issubset(registered):
        raise ValueError("preview page outside snapshot")
    pages = [p for p in review["pages"] if p["role"] != "divider" and
             (preview_pages is None or p["page_number"] in preview_pages)]
    if single_page and (len(pages) != 1 or maximum_closeups != 0):
        raise ValueError("single-page audit requires exactly one drawing and zero closeups")
    entries, omitted = drawing_entries(review)
    selected_numbers = {p["page_number"] for p in pages}
    windows = _windows(entries,pages,maximum_closeups)
    page_entries = defaultdict(list)
    mapped_page_entries = defaultdict(list)
    for entry in entries:
        mapped_page_entries[entry['page_number']].append(entry)
        entry['presentation_included'] = include_isolated_geometry or entry['kind'] != 'route' or bool(entry['priority'])
        entry['presentation_omission_reason'] = None if entry['presentation_included'] else 'isolated_geometry_without_accepted_attributes_or_completed_scope'
        if entry['presentation_included']:
            page_entries[entry["page_number"]].append(entry)
    manifest = {"schema_version":"0.1.0","layer":"mep_simple_network_drawing_audit",
        "snapshot_id":review["snapshot_id"],"source_pdf_sha256":review["source_sha256"],
        "project_id":review["project_id"],"document_id":review["document_id"],
        "processing_scope":deepcopy(review["processing_scope"]),"entries":entries,"omitted_rows":omitted,
        "junction_paths":[{"junction_ref":j["id"],"page_ref":j.get("page_ref"),"paths":_junction_paths(j),
                           "evidence_refs":j.get("evidence_refs",[]),"physical_continuation_established":False}
                          for j in review.get("junctions",[]) if _junction_paths(j)],
        "pages":[],"closeups":windows,"length_basis":"per_occurrence_projected_2d_not_installed",
        "source_presentation": {"opacity": source_opacity, "original_source_preserved": True},
        "authority":{"physical_continuation_established":False,"installed_length_established":False,
                     "physical_count_established":False,"engineer_approved":False,"quantity_eligible":False}}
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        digest=hashlib.sha256(output.read_bytes()).hexdigest()[:12]
        archived=output.with_name(output.stem+"."+digest+output.suffix)
        if not archived.exists():
            archived.write_bytes(output.read_bytes())
        old_manifest=output.with_suffix(".manifest.json")
        if old_manifest.exists() and not archived.with_suffix(".manifest.json").exists():
            archived.with_suffix(".manifest.json").write_bytes(old_manifest.read_bytes())
    with fitz.open(source) as original, fitz.open(source) as pdf, fitz.open() as overlays:
        if len(original) != len(registered):
            raise ValueError("source page count differs from snapshot registry")
        if not pages:
            raise ValueError("no drawing pages in presentation scope")
        pdf.select([p["page_number"]-1 for p in pages])
        pdf.new_page(pno=0,width=1190,height=842)
        overview_map={p["page_number"]:i+1 for i,p in enumerate(pages)}
        boxes_by_page={p["page_number"]:_source_text_boxes(original[p["page_number"]-1]) for p in pages}
        for record in pages:
            number=record["page_number"]; original_target=pdf[overview_map[number]]
            if original_target.rotation or original_target.cropbox != original_target.mediabox:
                raise ValueError("drawing renderer currently requires uncropped, unrotated source pages")
            rect=fitz.Rect(original_target.rect); width=rect.width*.29; size=max(12,rect.height/82)
            original_target.set_mediabox(fitz.Rect(0,0,rect.width+width,rect.height))
            if source_opacity != 1:
                # A white transparency veil fades the original vector drawing
                # without rasterizing or altering its selectable text. Authored
                # annotations are faded in this presentation copy only.
                original_target.draw_rect(rect, color=None, fill=(1,1,1),
                                          fill_opacity=1-source_opacity, overlay=True)
                for annotation in original_target.annots() or []:
                    prior = annotation.opacity
                    annotation.set_opacity(source_opacity * (prior if prior >= 0 else 1))
            # Author on a transparent form and place it once. Repeatedly
            # committing shapes into the original CAD page makes PyMuPDF
            # rescan its entire large content stream for every short stroke.
            page=overlays.new_page(width=rect.width+width,height=rect.height)
            audit_number=overview_map[number]+1
            column=_Column(page,rect.width+width*.06,width*.88,y=size*1.5,size=size)
            footer_text=("Frozen SQLite review: mep_review. Native marks retain record IDs. This audit covers only this source page; complete detection, physical continuity and installed quantities remain unresolved."
                         if single_page else "Selected geometry is highlighted where clear of source text. All segments and values, including unmarked isolated geometry, remain in the app and manifest.")
            footer_measure=_Column(None,column.x,column.width,y=0,size=size*.8)
            footer_measure.text(footer_text,gap=4)
            footer_top=rect.height-35-footer_measure.y
            column.text(f"DRAWING {number}",size=size*1.5,bold=True,gap=size*.5)
            counts=Counter(e["kind"] for e in page_entries[number])
            column.text(f"{counts['route']} projected segment records" if single_page else f"{counts['route']} recovered route segments",gap=4)
            column.text(f"{counts['identified_hvac']} identified HVAC bodies",gap=4)
            column.text(f"{counts['unresolved_hvac_observation']} unresolved HVAC observations",gap=size*.8)
            omitted_count=len(mapped_page_entries[number])-len(page_entries[number])
            if omitted_count:
                column.text(f"{omitted_count} isolated geometry entries not highlighted; values remain in the app.",size=size*.8,color=MUTED,gap=size*.6)
            column.text("Red: hot water supply / HHWS",color=SYSTEM_COLORS['heating_hot_water_supply'],gap=3)
            column.text("Orange: hot water return / HHWR",color=SYSTEM_COLORS['heating_hot_water_return'],gap=3)
            column.text("Blue: chilled supply / CHWS; cyan: return / CHWR",color=SYSTEM_COLORS['chilled_water_supply'],gap=3)
            if include_isolated_geometry:
                column.text("Grey: system unknown / unclassified geometry",color=LIGHT,gap=3)
            column.text("Purple: identified HVAC body + tag",color=PURPLE,gap=3)
            column.text("Amber: HVAC tag; body / ports unknown",color=AMBER,gap=size*.8)
            if single_page:
                column.text("Amber dashed: recorded unresolved alternatives / stops. Click marks for evidence IDs.",color=AMBER,size=size*.85,gap=size*.5)
            column.text("2D length = projected length in this view, not installed length. Unknown = not recovered.",size=size*.85,color=MUTED,gap=size*.8)
            execution=review["processing_scope"].get("execution_page_numbers")
            if execution is None or number not in execution:
                column.text("This page is not in the recorded execution scope. No detection completeness is claimed.",color=AMBER,gap=size)
            rows=page_entries[number]; boxes=boxes_by_page[number]
            for entry in mapped_page_entries[number]:
                points=[p for path in entry["paths"] for p in path]
                if any(not (-1e-4<=p[0]<=rect.width+1e-4 and -1e-4<=p[1]<=rect.height+1e-4) for p in points):
                    raise ValueError("frozen drawing mark outside source page")
            for entry in rows:
                entry["overview_visible_highlight_part_count"]=_draw_paths(page,entry["paths"],boxes,entry["color"],size*(.14 if entry["priority"] else .065))
                if single_page:
                    box = (fitz.Rect(entry['bbox_display']) + (-1, -1, 1, 1)) & rect
                    if not box.is_empty:
                        annotation = original_target.add_rect_annot(box)
                        annotation.set_border(width=0)
                        annotation.set_opacity(0)
                        annotation.set_info(title=entry['short_id'] + ' / ' + entry['kind'],
                            content=entry['row_ref'] + '\n' + str(entry['artifact']) + str(entry['pointer']) + '\n' + '\n'.join(entry['source_primitive_refs']))
                        annotation.update()
            if single_page:
                for row in review['rows']:
                    if row['kind'] != 'outcome':
                        continue
                    for mark in row['marks']:
                        if mark['page_number'] != number:
                            continue
                        for path in _paths(mark):
                            page.draw_polyline(path, color=AMBER, width=size*.07, dashes='[3 2] 0')
                        annotation = original_target.add_rect_annot(fitz.Rect(mark['bbox_display']))
                        annotation.set_colors(stroke=AMBER)
                        annotation.set_border(width=.4,dashes=[3,2])
                        annotation.set_info(title='Recorded outcome / ' + row['label'], content=row['id'] + '\n' + str(row.get('payload',{}).get('reason','See frozen outcome payload')))
                        annotation.update()
            for junction in review.get("junctions",[]):
                if junction.get("page_ref") != record["page_ref"] or junction["state"] != "accepted":
                    continue
                _draw_paths(page,_junction_paths(junction),boxes,_junction_color(junction, rows),size*.12)
            occupied=[]; labelled=[]
            for entry in rows:
                if entry['kind'] != 'unresolved_hvac_observation':
                    continue
                caption=entry['label']+": body / ports unknown"
                tag=_label(page,entry,rect,boxes,occupied,size=size*.7,text=caption)
                if tag is None:
                    caption=entry['label']
                    tag=_label(page,entry,rect,boxes,occupied,size=size*.7,text=caption)
                if tag is not None:
                    entry['printed_readouts'].append({'audit_page_number':audit_number,
                        'kind':'unresolved_tag_callout','text':caption,'label_rect_display':list(tag)})
                    labelled.append(entry['key'])
            priority=sorted((e for e in rows if e["priority"]),key=lambda e:(-e["priority"],e["short_id"]))
            for entry in priority[:12]:
                measured=_Column(None,column.x,column.width,y=column.y,size=column.size)
                _readout(measured,entry)
                if measured.y > footer_top-size*.3:
                    break
                tag=_label(page,entry,rect,boxes,occupied,size=size*.85)
                if tag is None:
                    continue
                _readout(column,entry)
                entry["printed_readouts"].append({"audit_page_number":audit_number,"kind":"overview","label_rect_display":list(tag)})
                labelled.append(entry["key"])
            if column.y>footer_top:
                raise ValueError("drawing overview header leaves no readable footer space")
            column.y=footer_top
            column.text(footer_text,size=size*.8,color=MUTED,gap=4)
            manifest["pages"].append({"source_page_number":number,"audit_page_number":audit_number,
                "marked_entry_keys":[e["key"] for e in rows],"readout_entry_keys":labelled,
                "omitted_isolated_geometry_entry_keys":[e['key'] for e in mapped_page_entries[number] if not e['presentation_included']],
                "text_shielded_entry_keys":[e["key"] for e in rows if not e["overview_visible_highlight_part_count"]],
                "visible_marker_entry_keys":[e['key'] for e in rows if e['overview_visible_highlight_part_count'] or e['printed_readouts']],
                "source_annotation_count":sum(1 for _ in original[number-1].annots() or []),
                "source_content_retained":True,"detection_complete":False})

        # Freeze the source overlay document before grafting: extending a PDF
        # after creating a PyMuPDF graft map can leave the map's xref range stale.
        for number,index in overview_map.items():
            target_page=pdf[index]
            target_page.show_pdf_page(target_page.rect,overlays,index-1,overlay=True)

        for window in windows:
            number=window["source_page_number"]; clip=fitz.Rect(window["clip_display"])
            page=pdf.new_page(width=1190,height=842)
            column=_Column(page,38,1110,y=25,size=13)
            column.text(f"DRAWING {number} - NETWORK CLOSE-UP",size=23,bold=True,gap=5)
            column.text("Size / 2D length / elevation belong to each labelled segment. 2D length covers the full segment in this view; a crop can show only part.",size=11,color=MUTED,gap=0)
            target=fitz.Rect(30,102,848,778)
            scale=min(target.width/clip.width,target.height/clip.height)
            actual=fitz.Rect(target.x0,target.y0,target.x0+clip.width*scale,target.y0+clip.height*scale)
            pix=original[number-1].get_pixmap(matrix=fitz.Matrix(2,2),clip=clip,annots=True,alpha=False)
            page.insert_image(actual,stream=pix.tobytes("png"))
            matrix=fitz.Matrix(scale,0,0,scale,actual.x0-clip.x0*scale,actual.y0-clip.y0*scale)
            boxes=[b*matrix for b in boxes_by_page[number] if clip.intersects(b)]
            rows=[e for e in page_entries[number] if clip.intersects(fitz.Rect(e["bbox_display"])+(-.01,-.01,.01,.01))]
            # Clip display strokes at the picture boundary; never change evidence.
            for entry in rows:
                for path in entry["paths"]:
                    pieces=[_clip_line(fitz.Point(a),fitz.Point(b),clip) for a,b in zip(path,path[1:])]
                    _draw_paths(page,[piece for piece in pieces if piece],boxes,entry["color"],2 if entry["priority"] else .7,matrix)
            for junction in review.get("junctions",[]):
                if junction.get("page_ref") != registered[number]["page_ref"]:
                    continue
                for path in _junction_paths(junction):
                    pieces=[_clip_line(fitz.Point(a),fitz.Point(b),clip) for a,b in zip(path,path[1:])]
                    _draw_paths(page,[piece for piece in pieces if piece],boxes,_junction_color(junction, entries),2,matrix)
            sidebar=_Column(page,878,278,y=109,size=13)
            sidebar.text("RECOVERED VALUES",size=16,bold=True,gap=12)
            occupied=[]; labelled=[]
            selected=sorted((e for e in rows if e["key"] in window["entry_keys"]),key=lambda e:(-e["priority"],e["short_id"]))
            for entry in selected:
                measured=_Column(None,sidebar.x,sidebar.width,y=sidebar.y,size=sidebar.size)
                _readout(measured,entry)
                if measured.y>713:
                    break
                tag=_label(page,entry,actual,boxes,occupied,size=12,matrix=matrix)
                if tag is None:
                    continue
                _readout(sidebar,entry)
                entry["printed_readouts"].append({"audit_page_number":page.number+1,"kind":"closeup","label_rect_display":list(tag)})
                labelled.append(entry["key"])
            sidebar.y=max(sidebar.y,729)
            sidebar.text("Unknowns and unlabelled segments remain in the app. HVAC ports unresolved.",size=10,color=MUTED,gap=0)
            link=fitz.Rect(30,792,700,819)
            page.insert_text((35,810),f"Back to full drawing {number}",fontsize=13,color=BLUE)
            page.insert_link({"kind":fitz.LINK_GOTO,"from":link,"page":overview_map[number],"to":fitz.Point(0,0)})
            overview=pdf[overview_map[number]]
            overview.insert_link({"kind":fitz.LINK_GOTO,"from":clip,"page":page.number,"to":fitz.Point(0,0)})
            window.update(audit_page_number=page.number+1,readout_entry_keys=labelled,
                          source_to_audit_matrix=list(matrix),source_annotations_rendered=True)

        cover=pdf[0]; column=_Column(cover,46,1095,y=34,size=15)
        column.text("RECOVERED MEP NETWORKS",size=31,bold=True,gap=14)
        column.text("Drawing-first review: routes, HVAC bodies and recovered measurements.",size=18,gap=14)
        column.text("Red = hot supply; orange = hot return. Blue = chilled supply; cyan = chilled return. Grey = unknown system. Purple = identified HVAC body/tag. Amber = unresolved HVAC observation.",gap=12)
        column.text("Size, projected 2D length and elevation are local to each labelled record. No installed length, physical equipment count or engineer approval is claimed.",color=MUTED,gap=20)
        mapped=[e for e in entries if e['page_number'] in selected_numbers]
        displayed=[e for e in mapped if e['presentation_included']]
        presentation_omitted=[e for e in mapped if not e['presentation_included']]
        column.text(f"{len(pages)} drawing overviews | {len(windows)} close-ups | {len(displayed)} mapped drawing entries",size=19,bold=True,gap=15)
        if presentation_omitted:
            column.text(f"{len(presentation_omitted)} isolated geometry entries are not highlighted for readability. They remain in the app and manifest; no discovery records were removed.",size=12,color=MUTED,gap=12)
        column.text("PAGE COVERAGE",size=18,bold=True,gap=8)
        coverage_columns=[_Column(cover,x,515,y=column.y,size=13) for x in (46,620)]
        split=math.ceil(len(review["pages"])/2)
        for index,record in enumerate(review["pages"]):
            number=record["page_number"]
            count=len(page_entries[number]); labelled=sum(bool(e["printed_readouts"]) for e in page_entries[number])
            scope=review["processing_scope"].get("execution_page_numbers")
            hidden=len(mapped_page_entries[number])-count
            status="divider" if record["role"]=="divider" else "not in this presentation" if number not in selected_numbers else "not processed" if scope is None or number not in scope else f"{count} entries; {labelled} readouts" + (f"; {hidden} not highlighted" if hidden else "")
            coverage_columns[index//split].text(f"Drawing {number}: {status}",size=13,gap=2)
        column.y=max(c.y for c in coverage_columns)+20
        column.text("All source drawings remain unchanged. Overview highlights retain the native PDF and its annotations. Labels are deliberately limited; per-segment measurements and exact evidence remain available in the app/manifest. Detection is incomplete.",size=12,color=MUTED,gap=0)
        manifest["coverage"]={"registered_page_count":len(registered),"overview_source_page_numbers":sorted(selected_numbers),
            "marked_entry_count":len(displayed),"labelled_entry_count":sum(bool(e["printed_readouts"]) for e in displayed),
            "visible_overview_marker_count":sum(len(p['visible_marker_entry_keys']) for p in manifest['pages']),
            "mapped_entry_count":len(mapped),"presentation_omitted_entry_count":len(presentation_omitted),
            "presentation_omitted_entry_keys":[e['key'] for e in presentation_omitted],
            "presentation_filter":{"include_isolated_geometry":include_isolated_geometry,
                "preserved":"connected_or_attributed_routes_completed_scopes_and_equipment",
                "omission_reason":"isolated_geometry_without_accepted_attributes_or_completed_scope"},
            "unlabelled_entry_keys":[e["key"] for e in displayed if not e["printed_readouts"]],
            "unlabelled_reason":"bounded_readout_density_or_unknown_isolated_geometry_values_in_app",
            "outside_presentation_entry_keys":[e["key"] for e in entries if e["page_number"] not in selected_numbers],
            "maximum_closeups":maximum_closeups,"all_snapshot_rows_in_manifest":False,
            "all_drawing_eligible_marks_in_manifest":True,"detection_complete":False}
        if single_page:
            pdf.select([1])
            for record in manifest['pages']:
                record['audit_page_number'] = 1
            for entry in entries:
                for readout in entry['printed_readouts']:
                    readout['audit_page_number'] = 1
        manifest["page_count"]=len(pdf)
        pdf.set_metadata({"title":"Recovered MEP networks - drawing review","subject":"Frozen projected drawing evidence; no installed quantities","creator":"Rebar drawing audit"})
        pdf.save(output,deflate=True,garbage=3)
    manifest["pdf_sha256"]=hashlib.sha256(output.read_bytes()).hexdigest()
    path=output.with_suffix(".manifest.json")
    if write_manifest:
        path.write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database",type=Path,default=ROOT/"data/projects/mep/project.sqlite")
    parser.add_argument("--project",default="mep-coordination")
    parser.add_argument("--document",default="coordination-set")
    parser.add_argument("--snapshot",required=True)
    parser.add_argument("--source",type=Path,default=ROOT/"M&P mark-up against shop systems piping.pdf")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--preview-pages",type=int,nargs="*")
    parser.add_argument("--maximum-closeups",type=int,default=24)
    parser.add_argument("--include-isolated-geometry",action="store_true",help="Also highlight isolated geometry without recovered attributes")
    args=parser.parse_args()
    review=load_review(args.database,project=args.project,document=args.document,snapshot=args.snapshot)
    result=render_network_drawing_audit(args.source,args.output,review,preview_pages=args.preview_pages,maximum_closeups=args.maximum_closeups,
                                      include_isolated_geometry=args.include_isolated_geometry)
    print(json.dumps({"pdf":str(args.output),"page_count":result["page_count"],"coverage":result["coverage"]},indent=2))


if __name__ == "__main__":
    main()
