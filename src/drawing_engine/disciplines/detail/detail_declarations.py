"""Independent native specification observations for named detail definitions.

Receives a PDF page only. Grid boundaries and semantic column headers bind the
declarations; no drawing calculation, profile or inferred quantity is an input.
"""
from __future__ import annotations

import re
import math
import fitz

from src.drawing_engine.disciplines.concrete.schedule_comparison import _grid_regions
from src.drawing_engine.core.vector_topology import extract_page_topology


def extract_detail_declarations(page, assembly_mark):
    words = page.get_text("words")
    segments = extract_page_topology(page)["segments"]
    records, unresolved = [], []
    for grid in _grid_regions(page):
        box = fitz.Rect(grid["bbox_display"])
        if box.get_area() > page.rect.get_area() / 2:
            continue
        xs = grid["xs"]
        header_bottom = grid["ys"][1]
        columns = {}
        for role, pattern in {"assembly": r"Tag|Марка", "position": r"Det\.|поз\.",
                              "description": r"Name(?:/Наименование)?|Наименование", "count": r"Q-ty|Кол-во",
                              "unit_mass": r"Unit"}.items():
            hits = {(j, i) for i, w in enumerate(words) if box.y0 < (w[1] + w[3]) / 2 < header_bottom
                    and re.fullmatch(pattern, w[4], re.I)
                    for j, (a, b) in enumerate(zip(xs, xs[1:])) if a < (w[0] + w[2]) / 2 < b}
            if len({j for j, _ in hits}) == 1:
                columns[role] = {"index": next(iter(hits))[0], "evidence_refs": [f"word[{i}]" for _, i in sorted(hits)]}
        if len(columns) != 5:
            continue

        def limits(role):
            j = columns[role]["index"]
            return xs[j:j + 2]

        ma, mb = limits("unit_mass")
        mass_header = [(i, w) for i, w in enumerate(words) if ma < (w[0] + w[2]) / 2 < mb
                       and box.y0 < (w[1] + w[3]) / 2 < header_bottom]
        mass_tokens = {w[4].rstrip(".,/").lower() for _, w in mass_header}
        mass_is_kg = bool(mass_tokens & {"kg", "кг"}) and bool(mass_tokens & {"weight", "масса"})
        columns["unit_mass"]["evidence_refs"] = [f"word[{i}]" for i, _ in mass_header]

        def rules(left, right):
            return [s for s in segments if s["kind"] == "line"
                    and abs(s["start_display"][1] - s["end_display"][1]) < .1
                    and min(s["start_display"][0], s["end_display"][0]) <= left + .2
                    and max(s["start_display"][0], s["end_display"][0]) >= right - .2
                    and box.y0 - .2 <= s["start_display"][1] <= box.y1 + .2]

        def bounds(y, lines):
            above = [s for s in lines if s["start_display"][1] < y]
            below = [s for s in lines if s["start_display"][1] > y]
            if not above or not below:
                return None
            return max(above, key=lambda s: s["start_display"][1]), min(below, key=lambda s: s["start_display"][1])

        a, b = limits("assembly")
        anchors = [(i, w) for i, w in enumerate(words) if w[4] == assembly_mark and a < (w[0] + w[2]) / 2 < b
                   and header_bottom < w[1] < box.y1]
        for anchor_i, anchor in anchors:
            group = bounds((anchor[1] + anchor[3]) / 2, rules(a, b))
            if group is None:
                unresolved.append({"reason": "assembly declaration lacks complete ruled body", "evidence_refs": [f"word[{anchor_i}]"]})
                continue
            top, bottom = [s["start_display"][1] for s in group]
            left, _ = limits("position")
            _, right = limits("unit_mass")
            body_rules = rules(left, right)
            pa, pb = limits("position")
            positions = [(i, w) for i, w in enumerate(words) if w[4].isdigit() and pa < (w[0] + w[2]) / 2 < pb
                         and top < (w[1] + w[3]) / 2 < bottom]
            for position_i, position in positions:
                row = bounds((position[1] + position[3]) / 2, body_rules)
                if row is None:
                    continue
                y0, y1 = [s["start_display"][1] for s in row]
                if y0 < top - .2 or y1 > bottom + .2:
                    continue
                if sum(y0 < (w[1] + w[3]) / 2 < y1 for _, w in positions) != 1:
                    unresolved.append({"reason": "multiple positions share one undivided specification body",
                                       "evidence_refs": [s["id"] for s in row]})
                    continue
                cells = {}
                for role in ("description", "count", "unit_mass"):
                    ca, cb = limits(role)
                    content = [(i, w) for i, w in enumerate(words) if ca < (w[0] + w[2]) / 2 < cb and y0 < (w[1] + w[3]) / 2 < y1]
                    content.sort(key=lambda item: (round(item[1][1], 1), item[1][0]))
                    cells[role] = {"text": " ".join(w[4] for _, w in content),
                                   "bbox_display": [ca, y0, cb, y1], "evidence_refs": [f"word[{i}]" for i, _ in content]}
                description = cells["description"]["text"]
                length = re.findall(r"L\s*=\s*(\d+(?:[.,]\d+)?)", description)
                plate = re.findall(r"Б-ПН-(\d+)[xх×](\d+)", description)
                bar = re.findall(r"\b(\d+)-[АA]\d+[СC]", description)

                def number(role):
                    value = cells[role]["text"].replace(",", ".")
                    return float(value) if re.fullmatch(r"\d+(?:\.\d+)?", value) else None

                length_mm = float(length[0].replace(",", ".")) if len(length) == 1 else None
                stock = [float(x) for x in plate[0]] if len(plate) == 1 and re.search(r"Sheet|Лист", description) else None
                records.append({"assembly_mark": assembly_mark, "part_mark": position[4], "state": "observed",
                                "declared": {"count": number("count"), "unit_mass_kg": number("unit_mass") if mass_is_kg else None,
                                             "length_mm": length_mm,
                                             "plate_section_mm": stock,
                                             "rectangular_stock_volume_m3": math.prod(stock) * length_mm / 1e9 if stock and length_mm else None,
                                             "bar_diameter_mm": float(bar[0]) if len(bar) == 1 else None},
                                "cells": cells, "column_bindings": columns,
                                "evidence_refs": [f"word[{anchor_i}]", f"word[{position_i}]"] + [s["id"] for s in group + row],
                                "bbox_display": [box.x0, y0, box.x1, y1]})
    if not records:
        unresolved.append({"reason": "no supported independently bound specification rows; declaration coverage remains unknown",
                           "evidence_refs": []})
    return {"page": page.number + 1, "assembly_mark": assembly_mark, "records": records, "unresolved": unresolved,
            "contract": {"source": "independent_native_specification", "calculation_input": False}}


def compare_detail(assembly, declarations):
    comparisons = []
    for part in assembly["child_parts"]:
        matches = [r for r in declarations["records"] if r["assembly_mark"] == assembly["mark"] and r["part_mark"] == part["mark"]]
        unique = len(matches) == 1 and sum(p["mark"] == part["mark"] for p in assembly["child_parts"]) == 1
        comparisons.append({"part_id": part["id"], "scope_match": "mutually_unique_assembly_and_position" if unique else "unresolved",
                            "calculated": part["calculated"], "declared": matches[0]["declared"] if unique else None,
                            "approved": None, "delta": None, "severity": "unresolved",
                            "reason": "no independently eligible same-unit physical quantity pair",
                            "drawing_evidence_refs": [ref for view in part.get("views", part.get("shaft_projections", [])) for ref in view["evidence_refs"]],
                            "declaration_evidence_refs": matches[0]["evidence_refs"] if unique else []})
        if unique and part["kind"] == "plate":
            declared_volume = matches[0]["declared"]["rectangular_stock_volume_m3"]
            calculated_volume = part["calculated"].get("volume_m3")
            if calculated_volume is not None and declared_volume is not None:
                delta = calculated_volume - declared_volume
                comparisons[-1].update({"channel": "plate_volume_m3", "delta": delta,
                                        "severity": "pass" if math.isclose(calculated_volume, declared_volume, rel_tol=1e-6) else "review",
                                        "reason": "closed rectangular plate versus independently declared rectangular stock for the same part"})
            elif part.get("conditional") and declared_volume is not None:
                comparisons[-1]["conditional_comparison"] = {
                    "calculated_under_mm_convention": part["conditional"]["volume_m3"],
                    "declared_rectangular_stock_volume_m3": declared_volume,
                    "delta_under_mm_convention": part["conditional"]["volume_m3"] - declared_volume,
                    "quantity_authority": False}
    return {"assembly_mark": assembly["mark"], "comparisons": comparisons, "approved": None}
