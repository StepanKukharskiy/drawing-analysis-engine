"""Bounded ladder-cage interpretation; schedules never enter geometry recovery.

One native paint must contain repeated closed rectangular transverse outlines
and two rails interrupted only by those outlines. Counts describe projections.
Named fan leaders and a unique caption establish detail context, not physical
placement or fabrication authority.
"""
from collections import defaultdict
import re

import fitz

from src.drawing_engine.disciplines.detail.native_detail_assembly import NativeDetail, center


def compact(text):
    return re.sub(r"\s+", "", text).upper()


def fan_leaders(native):
    """An authored star with one labelled shelf; crossings supply no branches."""
    result = []
    for i, drawing in enumerate(native.drawings):
        items = drawing["items"]
        if drawing.get("fill") is not None or not 2 <= len(items) <= 12 or any(x[0] != "l" for x in items):
            continue
        incidence = defaultdict(list)
        for edge, item in enumerate(items):
            for point in item[1:3]:
                incidence[tuple(point)].append(edge)
        hubs = [p for p, edges in incidence.items() if len(edges) == len(items)]
        if len(hubs) != 1:
            continue
        hub = hubs[0]
        ends = [tuple(x[2] if tuple(x[1]) == hub else x[1]) for x in items]
        shelves = [(j, p) for j, p in enumerate(ends) if abs(p[1] - hub[1]) < .01]
        if len(shelves) != 1:
            continue
        shelf_i, end = shelves[0]
        box = fitz.Rect(min(end[0], hub[0]), hub[1] - 30, max(end[0], hub[0]), hub[1] + 4)
        labels = [t for t in native.text if box.contains(fitz.Point(center(t["bbox_display"])))
                  and abs(t["bbox_display"][3] - hub[1]) < 4]
        if len(labels) == 1:
            result.append({"text": labels[0]["text"], "targets": [list(p) for j, p in enumerate(ends) if j != shelf_i],
                           "evidence_refs": labels[0]["evidence_refs"] + native.refs(i)})
    return result


def ladder(segments):
    """Replay exact endpoints within one paint, keeping every original ID."""
    if not segments or any(s["kind"] != "line" for s in segments):
        return None
    incidence = defaultdict(list)
    for i, s in enumerate(segments):
        for key in ("start", "end"):
            incidence[tuple(s[key])].append(i)
    remaining = set(range(len(segments)))
    rectangles, other = [], []
    while remaining:
        stack = [remaining.pop()]
        component = set(stack)
        while stack:
            for key in ("start", "end"):
                for i in incidence[tuple(segments[stack[-1]][key])]:
                    if i in remaining:
                        remaining.remove(i)
                        component.add(i)
                        stack.insert(0, i)
            stack.pop()
        rows = [segments[i] for i in sorted(component)]
        points = {tuple(s[k]) for s in rows for k in ("start", "end")}
        xs, ys = sorted({p[0] for p in points}), sorted({p[1] for p in points})
        closed = (len(rows) == len(points) == 4 and len(xs) == len(ys) == 2
                  and all(len(incidence[p]) == 2 for p in points)
                  and all(s["start"][0] == s["end"][0] or s["start"][1] == s["end"][1] for s in rows))
        if closed and ys[1] - ys[0] > 5 * (xs[1] - xs[0]) > 0:
            rectangles.append({"bbox_display": [xs[0], ys[0], xs[1], ys[1]], "evidence_refs": [s["id"] for s in rows]})
        else:
            other.extend(rows)
    if len(rectangles) < 3:
        return None
    rectangles.sort(key=lambda r: r["bbox_display"][0])
    boxes = [r["bbox_display"] for r in rectangles]
    if (max(b[1] for b in boxes) - min(b[1] for b in boxes) > .01
            or max(b[3] for b in boxes) - min(b[3] for b in boxes) > .01
            or max(b[2]-b[0] for b in boxes) > 1.02 * min(b[2]-b[0] for b in boxes)
            or any(a[2] >= b[0] for a, b in zip(boxes, boxes[1:]))):
        return None
    horizontal = defaultdict(list)
    for s in other:
        if abs(s["start"][1] - s["end"][1]) < .001:
            horizontal[round(s["start"][1], 3)].append(s)
    if len(horizontal) != 4:
        return None
    levels = sorted(horizontal)
    if not boxes[0][1] < levels[0] < levels[-1] < boxes[0][3]:
        return None
    interval_rows = []
    for level in levels:
        rows = sorted(horizontal[level], key=lambda s: min(s["start"][0], s["end"][0]))
        spans = [sorted((s["start"][0], s["end"][0])) for s in rows]
        if len(spans) != len(boxes) + 1:
            return None
        if any(abs(left[1]-b[0]) > .01 or abs(right[0]-b[2]) > .01
               for left, right, b in zip(spans, spans[1:], boxes)):
            return None
        interval_rows.append((rows, spans))
    rails = []
    accounted = {ref for r in rectangles for ref in r["evidence_refs"]}
    for j in (0, 2):
        lower, upper = interval_rows[j], interval_rows[j+1]
        if any(abs(a-b) > .01 for x, y in zip(lower[1], upper[1]) for a, b in zip(x, y)):
            return None
        left, right = lower[1][0][0], lower[1][-1][1]
        caps = [s for s in other if abs(s["start"][0]-s["end"][0]) < .001
                and min(abs(s["start"][0]-left), abs(s["start"][0]-right)) < .01
                and abs(min(s["start"][1], s["end"][1])-levels[j]) < .01
                and abs(max(s["start"][1], s["end"][1])-levels[j+1]) < .01]
        if len(caps) != 2 or abs(caps[0]["start"][0]-caps[1]["start"][0]) < .01:
            return None
        refs = [s["id"] for s in lower[0]+upper[0]+caps]
        accounted.update(refs)
        rails.append({"bbox_display": [left, levels[j], right, levels[j+1]], "evidence_refs": refs,
                      "interruption_refs": [r["evidence_refs"] for r in rectangles]})
    if accounted != {s["id"] for s in segments}:
        return None
    return {"transverse": rectangles, "rails": rails, "complete_paint_coverage": True}


def extract_cage(page, mark):
    native = NativeDetail(page)
    titles = [t for t in native.text if re.fullmatch(r"Схема каркаса\s+.+", t["text"], re.I)]
    selected = [t for t in titles if compact(re.sub(r"^Схема каркаса\s+", "", t["text"], flags=re.I)) == compact(mark)]
    result = {"schema_version": "native_cage_assembly.v1", "mark": mark, "page": page.number+1,
              "state": "unknown", "child_parts": [], "unresolved": [], "parent_occurrences": [],
              "calculated": {"physical_assembly_count": None, "installed_length_m": None, "mass_kg": None},
              "approved": None, "fabrication_release": False, "quantity_scope": "projected_detail_definition"}
    if len(selected) != 1:
        result["unresolved"].append("missing or ambiguous exact cage caption")
        return result, native
    title = selected[0]
    result["title"] = title
    fans = fan_leaders(native)
    candidates = []
    for i, segments in native.by_drawing.items():
        box = native.drawings[i]["rect"]
        # Caption centred above the paint; another caption in between competes.
        if (native.drawings[i].get("fill") is not None or box.y0 < title["bbox_display"][3]
                or not box.x0 < center(title["bbox_display"])[0] < box.x1
                or any(title["bbox_display"][3] < t["bbox_display"][1] < box.y0
                       and box.x0 < center(t["bbox_display"])[0] < box.x1 for t in titles)):
            continue
        shape = ladder(segments)
        if shape:
            candidates.append((i, shape))
    if len(candidates) != 1:
        result["unresolved"].append("missing or competing complete native ladder paint")
        return result, native
    i, shape = candidates[0]
    result["drawing_ref"] = f"drawing[{i}]"
    used_marks = set()
    for role in ("transverse", "rails"):
        rows = shape[role]
        bindings = []
        for fan in fans:
            if not re.fullmatch(r"\d+", fan["text"]):
                continue
            hits = [[k for k, row in enumerate(rows) if (fitz.Rect(row["bbox_display"])+(-.01,-.01,.01,.01)).contains(fitz.Point(p))]
                    for p in fan["targets"]]
            # Rails need a separate target on each retained rail; repeated
            # transverse outlines need one uniquely scoped example target.
            if all(len(h) == 1 for h in hits) and (role == "transverse" or {h[0] for h in hits} == set(range(len(rows)))):
                bindings.append(fan)
        if len(bindings) != 1 or bindings[0]["text"] in used_marks:
            result["unresolved"].append(f"{role}: missing or ambiguous native mark leader")
            continue
        fan = bindings[0]
        used_marks.add(fan["text"])
        result["child_parts"].append({"id": "part:"+fan["text"], "mark": fan["text"], "role": role,
            "state": "derived", "projections": rows, "projected_count": len(rows), "annotation": fan,
            "physical_count": None, "cutting_length_mm": None, "mass_kg": None, "approved": None,
            "basis": "complete same-paint ladder closure and unique native position leader"})
    parent = [fan for fan in fans if compact(fan["text"]) == compact(mark)]
    result["parent_occurrences"] = [{**fan, "state": "observed", "leader_target_count": len(fan["targets"]),
        "physical_count": None, "reason": "authored repetition targets; physical placement identity not established"} for fan in parent]
    result["state"] = "derived" if len(result["child_parts"]) == 2 else "unknown"
    result["complete_paint_coverage"] = shape["complete_paint_coverage"]
    result["unresolved"].append("Physical repetition, 3D placement, bar diameter, cutting lengths and approval remain unresolved")
    return result, native
