"""Independent ruled schedule/reference observations for cage review.

Document anchors, declared values and applicability have separate states. The
geometry extractor never receives these records. Filename order is not identity.
"""
import re

import fitz

from src.drawing_engine.disciplines.detail.native_cage_assembly import compact


def cell(native, box):
    if box is None:
        return {"text": "", "evidence_refs": []}
    rect = fitz.Rect(box)
    words = [(i, w) for i, w in enumerate(native.words) if rect.contains(fitz.Point((w[0]+w[2])/2, (w[1]+w[3])/2))]
    return {"text": " ".join(w[4] for _, w in words), "bbox_display": list(box),
            "evidence_refs": [f"word[{i}]" for i, _ in words]}


def tables(native):
    return [{"bbox_display": list(t.bbox), "rows": [[cell(native, box) for box in row.cells] for row in t.rows]}
            for t in native.page.find_tables().tables]


def document_anchor(native, observed_tables):
    candidates = []
    for table in observed_tables:
        rows = table["rows"]
        codes = [c for r in rows for c in r if re.fullmatch(r"РЧ\s+\d{4}-\d+(?:\.\d+)*(?:\s+СБ)?", c["text"])]
        if len(codes) != 1:
            continue
        sheets = []
        for i, row in enumerate(rows):
            for j, header in enumerate(row):
                if header["text"] != "Лист":
                    continue
                below = next((r[j] for r in rows[i+1:] if j < len(r) and r[j]["text"]), None)
                if below and below["text"].isdigit():
                    sheets.append({"value": int(below["text"]), "evidence_refs": header["evidence_refs"]+below["evidence_refs"]})
        revisions = [r[j:j+3] for r in rows for j in range(len(r)-2) if r[j]["text"].isdigit()
                     and r[j+1]["text"] == "Зам." and re.fullmatch(r"\d{4}-\d+\.\d+", r[j+2]["text"])]
        if len(sheets) == len(revisions) == 1:
            revision = revisions[0]
            candidates.append({"document_code": codes[0]["text"], "sheet": sheets[0]["value"],
                "revision": [c["text"] for c in revision], "bbox_display": table["bbox_display"],
                "evidence_refs": codes[0]["evidence_refs"] + sheets[0]["evidence_refs"]
                                 + [ref for c in revision for ref in c["evidence_refs"]]})
    unique = {(c["document_code"], c["sheet"], tuple(c["revision"])) for c in candidates}
    return min(candidates, key=lambda c: fitz.Rect(c["bbox_display"]).get_area()) if len(unique) == 1 else None


def reference_rows(native, observed_tables, mark):
    records = []
    for table in observed_tables:
        rows = table["rows"]
        headers = [(i, r) for i, r in enumerate(rows) if any(c["text"] == "Поз." for c in r)
                   and any(compact(c["text"]) == "№ЛИСТА" for c in r)]
        if len(headers) != 1:
            continue
        index, header = headers[0]
        columns = {role: [j for j, c in enumerate(header) if compact(c["text"]) == text]
                   for role, text in (("position", "ПОЗ."), ("sheet", "№ЛИСТА"), ("count", "КОЛ-ВО,ШТ"),
                                      ("length", "ДЛИНАЕД.,ММ"), ("diameter", "ДИАМЕТР"))}
        if any(len(v) != 1 for v in columns.values()):
            continue
        columns = {k: v[0] for k, v in columns.items()}
        group = None
        for row in rows[index+1:]:
            nonempty = [c for c in row if c["text"]]
            if len(nonempty) == 1:
                group = nonempty[0] if compact(nonempty[0]["text"]) == compact("Каркас "+mark) else None
                continue
            if group is None:
                continue
            values = {k: row[j] for k, j in columns.items()}
            if not values["position"]["text"].isdigit() or not values["sheet"]["text"].isdigit():
                continue
            records.append({"part_mark": values["position"]["text"], "assembly_mark": mark,
                "destination_sheet": int(values["sheet"]["text"]), "state": "observed",
                "declared": {k: float(values[k]["text"]) if re.fullmatch(r"\d+(?:\.\d+)?", values[k]["text"]) else None
                             for k in ("count", "length", "diameter")}, "cells": values,
                "evidence_refs": group["evidence_refs"]+[ref for c in values.values() for ref in c["evidence_refs"]],
                "header_evidence_refs": [ref for c in header for ref in c["evidence_refs"]]})
    return records


def link_references(source_anchor, destination_anchor, rows, assembly):
    links = []
    for row in rows:
        base = re.sub(r"\s+СБ$", "", source_anchor["document_code"]) if source_anchor else ""
        target = re.sub(r"\s+СБ$", "", destination_anchor["document_code"]) if destination_anchor else ""
        identity = bool(base and destination_anchor and target.startswith(base+".")
                        and row["destination_sheet"] == destination_anchor["sheet"]
                        and source_anchor["revision"] == destination_anchor["revision"])
        matches = [p for p in assembly["child_parts"] if p["mark"] == row["part_mark"]]
        unique = sum(r["part_mark"] == row["part_mark"] for r in rows) == 1
        accepted = identity and len(matches) == 1 and unique and assembly["state"] == "derived"
        links.append({"part_mark": row["part_mark"], "destination_sheet": row["destination_sheet"],
            "reference_identity_state": "derived" if identity else "unknown",
            "detail_applicability_state": "derived" if accepted else "unknown",
            "destination_part_ref": matches[0]["id"] if accepted else None,
            "basis": "native sheet, hierarchical document code, identical change record, exact cage/child mark and unique row",
            "source_evidence_refs": row["evidence_refs"]+row["header_evidence_refs"],
            "physical_identity_established": False, "quantity_eligible": False,
            "count_comparison": {"declared": row["declared"]["count"],
                "observed_projections": matches[0]["projected_count"] if accepted else None,
                "physical_calculated": None, "delta": None, "reason": "projection count has no physical quantity authority"}})
    return links
