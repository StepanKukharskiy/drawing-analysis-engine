"""Bounded native, orthographic plate-and-bar detail interpretation.

No specification records enter this module. The supported subset is an explicitly
named detail, a dot-marked rectangular face, a same-mark native edge leader on
its named rectangular section, and circular ends / straight shafts. Labelled
concentric rings and uniquely shared arrow stems remain explicit annotation
evidence. Other details abstain. Coordinates
are display points; all source references reuse the page-global native topology.
"""
from __future__ import annotations

from collections import defaultdict
import math
import re

import fitz

from src.drawing_engine.disciplines.concrete.generic_prismatic_solver import _box_mesh
from src.drawing_engine.core.vector_topology import extract_page_topology, point_distance_to_segment


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def near(a, b, tolerance=.25):
    return math.dist(a, b) <= tolerance


def part_mark(leader):
    match = re.fullmatch(r"(\d+)(?:\s*\(TYP\))?", leader["text"].strip(), re.I)
    return match.group(1) if match else None


def lines_of_text(words):
    groups = defaultdict(list)
    for index, word in enumerate(words):
        groups[tuple(word[5:7])].append((index, word))
    result = []
    for values in groups.values():
        box = fitz.Rect(values[0][1][:4])
        for _, word in values:
            box |= fitz.Rect(word[:4])
        result.append({"text": " ".join(w[4] for _, w in values), "bbox_display": list(box),
                       "evidence_refs": [f"word[{i}]" for i, _ in values]})
    return result


class NativeDetail:
    def __init__(self, page):
        self.page = page
        self.words = page.get_text("words")
        self.text = lines_of_text(self.words)
        self.drawings = page.get_drawings()
        self.topology = extract_page_topology(page)
        self.by_drawing = defaultdict(list)
        for segment in self.topology["segments"]:
            index = int(segment["drawing_ref"][8:-1])
            self.by_drawing[index].append({**segment, "start": segment["start_display"],
                                           "end": segment["end_display"],
                                           "sample_points": segment["sample_points_display"]})
        self.lines = []
        self.rectangles = []
        self.circles = []
        for index, drawing in enumerate(self.drawings):
            segments = self.by_drawing[index]
            if drawing.get("fill") is None:
                self.lines.extend(s for s in segments if s["kind"] == "line")
            box = drawing["rect"]
            if drawing.get("fill") is not None or min(box.width, box.height) <= 1:
                continue
            corners = [(box.x0, box.y0), (box.x1, box.y0), (box.x1, box.y1), (box.x0, box.y1)]
            if len(segments) == 4 and all(s["kind"] == "line" for s in segments):
                edges = [(s["start"], s["end"]) for s in segments]
                if all(sum((near(a, c) and near(b, d)) or (near(b, c) and near(a, d))
                           for a, b in edges) == 1 for c, d in zip(corners, corners[1:] + corners[:1])):
                    self.rectangles.append(index)
            if (len(segments) == 4 and all(s["kind"] == "cubic" for s in segments)
                    and abs(box.width - box.height) < .02 * box.width
                    and all(near(s["end"], segments[(i + 1) % 4]["start"])
                            for i, s in enumerate(segments))):
                radius = box.width / 2
                if all(abs(math.dist(point, center(box)) - radius) < .02 * radius
                       for s in segments for point in s["sample_points"]):
                    self.circles.append(index)
        self._triangles = []
        self._filled_lines = []
        for index, drawing in enumerate(self.drawings):
            if drawing.get("fill") is None:
                continue
            points = {tuple(p) for s in drawing["items"] if s[0] == "l" for p in s[1:3]}
            if len(points) == 3:
                self._triangles.append((index, list(points)))
            if self.by_drawing[index] and all(s["kind"] == "line" for s in self.by_drawing[index]):
                self._filled_lines.append(index)
        self.leaders = self._leaders()

    def refs(self, index):
        return [s["id"] for s in self.by_drawing[index]]

    def record(self, index):
        return {"drawing_ref": f"drawing[{index}]", "bbox_display": list(self.drawings[index]["rect"]),
                "evidence_refs": self.refs(index)}

    def _leader_head(self, base, stem_end, *, arrow_only=False):
        heads = []
        for h, points in self._triangles:
            box = self.drawings[h]["rect"]
            if not (box.x0 - 1 <= base[0] <= box.x1 + 1 and box.y0 - 1 <= base[1] <= box.y1 + 1):
                continue
            if len(points) == 3:
                for tip in points:
                    other = [p for p in points if p != tip]
                    midpoint = tuple((a + b) / 2 for a, b in zip(*other))
                    stem = (base[0] - stem_end[0], base[1] - stem_end[1])
                    along_stem = abs(stem[0] * (tip[1] - base[1]) - stem[1] * (tip[0] - base[0])) / max(math.hypot(*stem), .001)
                    attached = near(base, midpoint) or (any(near(base, p) for p in other) and along_stem < .15)
                    if attached and math.dist(tip, base) > 1.5 * math.dist(*other):
                        heads.append(("arrow", tip, self.refs(h)))
        if len(heads) == 1:
            return heads[0]
        if heads:
            return None
        if arrow_only:
            return None
        # CAD dots may be fans of separately painted triangles.
        fan = [h for h in self._filled_lines
               if max(self.drawings[h]["rect"].width, self.drawings[h]["rect"].height) < 8
               and any(near(s["start"], base) or near(s["end"], base) for s in self.by_drawing[h])]
        radial = [p for h in fan for s in self.by_drawing[h] for p in (s["start"], s["end"])
                  if math.dist(p, base) > .5]
        angles = sorted({round(math.atan2(p[1] - base[1], p[0] - base[0]), 3) for p in radial})
        if len(angles) >= 8 and max(b - a for a, b in zip(angles, angles[1:] + [angles[0] + 2 * math.pi])) <= .6:
            radii = [math.dist(p, base) for p in radial]
            if max(radii) / min(radii) <= 1.1:
                return "dot", base, [ref for h in fan for ref in self.refs(h)]
        # An authored plain endpoint is an observation, not an inferred arrow.
        return "plain", base, []

    def _leaders(self):
        result, stems = [], []
        for index, drawing in enumerate(self.drawings):
            items = drawing["items"]
            if (drawing.get("fill") is not None or not 2 <= len(items) <= 4
                    or any(s[0] != "l" for s in items)
                    or not all(near(a[2], b[1]) for a, b in zip(items, items[1:]))
                    or near(items[0][1], items[-1][2])):
                continue
            first, shelf = items[0], items[-1]
            if abs(shelf[1].y - shelf[2].y) > .1:
                continue
            shelf_box = fitz.Rect(min(shelf[1].x, shelf[2].x), shelf[1].y - 25,
                                  max(shelf[1].x, shelf[2].x), shelf[1].y + 2)
            labels = [line for line in self.text if shelf_box.contains(fitz.Point(center(line["bbox_display"])))
                      and abs(line["bbox_display"][3] - shelf[1].y) < 4]
            if len(labels) != 1:
                continue
            head = self._leader_head(tuple(first[1]), tuple(first[2]))
            if head is None:
                continue
            kind, target, head_refs = head
            record = {"text": labels[0]["text"], "kind": kind, "target_display": list(target),
                      "evidence_refs": labels[0]["evidence_refs"] + self.refs(index) + head_refs}
            result.append(record)
            stems.append((index, items[:-1], record))
        # A separate arrow branch can end exactly on one already labelled stem.
        # Crossing strokes do not connect; only the branch's authored endpoint
        # supplies incidence, with a unique compatible non-colour stroke style.
        for index, drawing in enumerate(self.drawings):
            items = drawing["items"]
            if drawing.get("fill") is not None or len(items) != 1 or items[0][0] != "l":
                continue
            segment = items[0]
            head = self._leader_head(tuple(segment[1]), tuple(segment[2]), arrow_only=True)
            if head is None or head[0] != "arrow":
                continue
            owners = []
            for owner, paths, record in stems:
                if self.by_drawing[index][0]["style"] != self.by_drawing[owner][0]["style"]:
                    continue
                if any(point_distance_to_segment(tuple(segment[2]), tuple(p[1]), tuple(p[2])) < .1
                       and min(p[1].x, p[2].x) - .1 <= segment[2].x <= max(p[1].x, p[2].x) + .1
                       and min(p[1].y, p[2].y) - .1 <= segment[2].y <= max(p[1].y, p[2].y) + .1 for p in paths):
                    owners.append(record)
            if len(owners) == 1:
                record = owners[0]
                result.append({"text": record["text"], "kind": "arrow", "target_display": list(head[1]),
                               "shared_label_stem": True,
                               "evidence_refs": record["evidence_refs"] + self.refs(index) + head[2]})
        return result

    def dimensions(self, index, axis, *, family_box=None):
        """Replay complete two-witness/two-tick chains for one profile extent.

        Native numeric labels must be inside the bounded interval or attached to
        its midpoint by an authored two-line connector. Coincident authored
        baselines remain separate evidence, not competing engineering values.
        """
        box = family_box if family_box is not None else self.drawings[index]["rect"]
        witness_gap = max(3, .2 * min(box.width, box.height)) if family_box is not None else 3
        lo, hi = box[axis], box[axis + 2]
        other = 1 - axis
        candidates = []
        for baseline in self.lines:
            a, b = baseline["start"], baseline["end"]
            if abs(a[other] - b[other]) > .1 or min(a[axis], b[axis]) > lo + .15 or max(a[axis], b[axis]) < hi - .15:
                continue
            ordinate = a[other]
            if box[other] < ordinate < box[other + 2] or min(abs(ordinate - box[other]), abs(ordinate - box[other + 2])) > max(box.width, box.height):
                continue
            witnesses, ticks = [], []
            for position in (lo, hi):
                crossing = [0., 0.]
                crossing[axis], crossing[other] = position, ordinate
                ws = [s for s in self.lines if abs(s["start"][axis] - position) < .15
                      and abs(s["end"][axis] - position) < .15
                      and min(s["start"][other], s["end"][other]) <= ordinate <= max(s["start"][other], s["end"][other])]
                ts = []
                for h, drawing in enumerate(self.drawings):
                    r = drawing["rect"]
                    if drawing.get("fill") is None or not near(center(r), crossing, .18) or min(r.width, r.height) < 1:
                        continue
                    points = [p for s in self.by_drawing[h] if s["kind"] == "line" for p in (s["start"], s["end"])]
                    if (abs(r.width - r.height) < .12 * max(r.width, r.height) and points
                            and min(max(abs((p[0] - crossing[0]) + sign * (p[1] - crossing[1])) for p in points)
                                    for sign in (-1, 1)) < .2 * r.width):
                        ts.append(h)
                # The drafting gap is bounded by the measured local tick size,
                # not a page-specific adjustment to the native snap tolerance.
                anchor_limit = max([witness_gap] + [.5 * max(self.drawings[h]["rect"].width,
                                                            self.drawings[h]["rect"].height) for h in ts])
                # A witness may overrun the profile edge. Replay contact with
                # the complete native segment, including a bounded drafting gap.
                ws = [s for s in ws if min(point_distance_to_segment(
                    (position, edge) if axis == 0 else (edge, position), s["start"], s["end"])
                    for edge in (box[other], box[other + 2])) < anchor_limit]
                if ws and ts:
                    witnesses.extend(s["id"] for s in ws)
                    ticks.extend(ref for h in ts for ref in self.refs(h))
            if not witnesses or len(set(ticks)) < 2:
                continue
            # Both endpoints must close, not merely two strokes at one tick.
            if not all(any(abs(s["start"][axis] - pos) < .15 for s in self.lines if s["id"] in witnesses)
                       for pos in (lo, hi)):
                continue
            for wi, word in enumerate(self.words):
                if not re.fullmatch(r"\d+(?:[.,]\d+)?", word[4]):
                    continue
                value = float(word[4].replace(",", "."))
                if value <= 0:
                    continue
                c = center(word)
                direct = (lo <= c[axis] <= hi and abs(word[other + 2] - ordinate) < 2
                          and abs(c[other] - ordinate) < 20)
                connectors = []
                midpoint = [0., 0.]
                midpoint[axis], midpoint[other] = (lo + hi) / 2, ordinate
                if axis == 0:
                    for di, d in enumerate(self.drawings):
                        items = d["items"]
                        if (len(items) == 2 and all(s[0] == "l" for s in items)
                                and near(items[0][1], midpoint) and near(items[0][2], items[1][1])
                                and abs(items[1][1].y - items[1][2].y) < .1
                                and min(items[1][1].x, items[1][2].x) <= c[0] <= max(items[1][1].x, items[1][2].x)
                                and abs(word[3] - items[1][1].y) < 2):
                            connectors.extend(self.refs(di))
                    # Short intervals can put their label on a separately
                    # authored collinear extension ending at one terminal.
                    if c[axis] < lo or c[axis] > hi:
                        terminal = lo if c[axis] < lo else hi
                        label_height = word[3] - word[1]
                        for extension in self.lines:
                            a, b = extension["start"], extension["end"]
                            if (abs(a[other] - ordinate) > .1 or abs(b[other] - ordinate) > .1
                                    or min(abs(a[axis] - terminal), abs(b[axis] - terminal)) > .15
                                    or not min(a[axis], b[axis]) <= c[axis] <= max(a[axis], b[axis])
                                    or abs(c[axis] - terminal) > 4 * label_height
                                    or abs(word[other + 2] - ordinate) >= 2):
                                continue
                            # An intervening marked terminal belongs to another
                            # interval; its label cannot leap across that tick.
                            blocked = any(d.get("fill") is not None and min(d["rect"].width, d["rect"].height) > 1
                                          and abs(center(d["rect"])[other] - ordinate) < .18
                                          and min(c[axis], terminal) + .2 < center(d["rect"])[axis] < max(c[axis], terminal) - .2
                                          for d in self.drawings)
                            if not blocked:
                                connectors.append(extension["id"])
                if direct or connectors:
                    candidates.append({"axis": "xy"[axis], "value": value, "state": "direct",
                                       "source_unit": "unspecified_linear_drawing_unit",
                                       "profile_ref": f"drawing[{index}]", "measured_interval_display": [lo, hi],
                                       "points_per_unit": (hi - lo) / value,
                                       "evidence_refs": [f"word[{wi}]", baseline["id"]] + witnesses + ticks + connectors})
        # Distinct native labels remain visible; only unanimous values close.
        return candidates


def extract_assembly(page, mark):
    native = NativeDetail(page)
    titles = [line for line in native.text if re.fullmatch(r"Embedded part\s+\S+", line["text"], re.I)]
    selected = [line for line in titles if line["text"].split()[-1] == mark]
    result = {"schema_version": "native_detail_assembly.v1", "page": page.number + 1, "mark": mark,
              "state": "unknown", "child_parts": [], "unresolved": [],
              "approved": None, "assembly_count": None,
              "contract": {"scope": "one_detail_definition", "not_job_takeoff": True,
                           "physical_placement_transform": None, "fabrication_release": False}}

    def stop(reason, refs=()):
        result["unresolved"].append({"reason": reason, "state": "unknown", "evidence_refs": list(refs)})

    if len(selected) != 1:
        stop("unique native assembly title unavailable")
        return result, native
    title = selected[0]
    result["title"] = title
    unit_pattern = re.compile(r"^(?:all\s+(?:linear\s+)?dimensions\s+(?:are\s+)?in|все\s+(?:линейные\s+)?размеры\s+(?:даны\s+)?в)\s+(mm|мм|cm|см|inches|m|м)[.]?$", re.I)
    unit_records = [line for line in native.text if unit_pattern.fullmatch(line["text"].strip())]
    units = {unit_pattern.fullmatch(line["text"].strip()).group(1).lower() for line in unit_records}
    native_mm = bool(units) and units <= {"mm", "мм"}
    result["linear_unit_evidence"] = {"state": "direct" if native_mm else "unknown", "unit": "mm" if native_mm else None,
                                      "observations": unit_records,
                                      "searched_native_line_count": len(native.text),
                                      "supported_form": "explicit global linear-dimension unit sentence"}
    tx, ty = center(title["bbox_display"])
    row = [t for t in titles if abs(center(t["bbox_display"])[1] - ty) < 25]
    bottom = min([t["bbox_display"][1] for t in titles if t["bbox_display"][1] > ty + 25] + [page.rect.height])
    right = min([t["bbox_display"][0] for t in row if center(t["bbox_display"])[0] > tx] + [page.rect.width])
    plan_options = []
    for index in native.rectangles:
        box = native.drawings[index]["rect"]
        if not (box.x0 <= tx <= box.x1 and title["bbox_display"][3] < box.y0 < box.y1 < bottom):
            continue
        leaders = [l for l in native.leaders if l["kind"] == "dot" and part_mark(l)
                   and box.contains(fitz.Point(l["target_display"]))]
        if len(leaders) == 1:
            plan_options.append((index, leaders[0]))
    if len(plan_options) != 1:
        stop("unique dot-marked rectangular face unavailable", title["evidence_refs"])
        return result, native
    plan, plan_label = plan_options[0]
    pb = native.drawings[plan]["rect"]
    section_options = []
    for index in native.rectangles:
        box = native.drawings[index]["rect"]
        if not (pb.x1 < box.x0 < box.x1 < right and abs(box.y0 - pb.y0) < .2 and abs(box.y1 - pb.y1) < .2):
            continue
        for leader in native.leaders:
            if leader["kind"] in {"arrow", "plain"} and part_mark(leader) == part_mark(plan_label) and any(
                    point_distance_to_segment(leader["target_display"], s["start"], s["end"]) < .15
                    for s in native.by_drawing[index]):
                section_options.append((index, leader))
    if len(section_options) != 1:
        stop("same-mark orthographic section is missing or ambiguous", plan_label["evidence_refs"])
        return result, native
    section, section_label = section_options[0]
    sb = native.drawings[section]["rect"]
    section_titles = [line for line in native.text if re.fullmatch(r"(\w)-\1", line["text"])
                      and sb.x0 - sb.height < center(line["bbox_display"])[0] < sb.x1 + sb.height
                      and title["bbox_display"][3] < line["bbox_display"][1] < sb.y0]
    if len(section_titles) != 1:
        stop("unique named section unavailable")
        return result, native
    cut_mark = section_titles[0]["text"].split("-")[0]
    cuts = [(i, w) for i, w in enumerate(native.words) if w[4] == cut_mark
            and pb.x0 <= center(w)[0] <= pb.x1 + 5 and title["bbox_display"][3] < w[1] < bottom]
    if not (len(cuts) == 2 and min(w[3] for _, w in cuts) < pb.y0 and max(w[1] for _, w in cuts) > pb.y1):
        stop("section label lacks unique flanking face markers")
        return result, native
    result["supporting_views"] = {"face": native.record(plan), "section": native.record(section),
                                  "section_title": section_titles[0],
                                  "cut_marker_refs": [f"word[{i}]" for i, _ in cuts],
                                  "cutting_plane_metric_transform": None}
    dimensions = {"width": native.dimensions(plan, 0), "height": native.dimensions(plan, 1),
                  "thickness": native.dimensions(section, 0), "section_height": native.dimensions(section, 1)}
    plate = {"id": f"part:{part_mark(plan_label)}", "mark": part_mark(plan_label), "kind": "plate",
             "state": "derived", "views": [native.record(plan), native.record(section)],
             "annotations": [plan_label, section_label], "dimensions": dimensions,
             "outer_contour": {**native.record(plan), "kind": "closed_native_rectangle",
                               "points_display": [list(pb.tl), list(pb.tr), list(pb.br), list(pb.bl)]},
             "solid": None, "calculated": {"volume_m3": None}, "approved": None,
             "physical": {"job_count": None}, "conditional": {}}
    result["child_parts"].append(plate)
    ends = [i for i in native.circles if pb.contains(native.drawings[i]["rect"])]
    shafts = [i for i in native.rectangles if abs(native.drawings[i]["rect"].x0 - sb.x1) < .2
              and sb.y0 < native.drawings[i]["rect"].y0 < native.drawings[i]["rect"].y1 < sb.y1]
    ring_labels = {}
    for ring in ends:
        rb = native.drawings[ring]["rect"]
        inner = [i for i in ends if i != ring and rb.contains(native.drawings[i]["rect"])
                 and near(center(rb), center(native.drawings[i]["rect"]), .1)
                 and rb.width > 1.25 * native.drawings[i]["rect"].width]
        labels = [l for l in native.leaders if l["kind"] in {"plain", "arrow"} and part_mark(l)
                  and abs(math.dist(l["target_display"], center(rb)) - rb.width / 2) < .25]
        if len(inner) != 1 or len(labels) != 1:
            continue
        eb = native.drawings[inner[0]]["rect"]
        rows = [s for s in shafts if abs(center(native.drawings[s]["rect"])[1] - center(eb)[1]) < .2
                and abs(native.drawings[s]["rect"].height - eb.width) < .1]
        if len(rows) != 1:
            continue
        label = labels[0]
        label_box = native.words[int(label["evidence_refs"][0][5:-1])][:4]
        qualifiers = [line for line in native.text if re.fullmatch(r"\(TYP\)", line["text"], re.I)
                      and abs(line["bbox_display"][0] - label_box[0]) < .25
                      and 0 <= line["bbox_display"][1] - label_box[3] < label_box[3] - label_box[1]]
        ring_labels[ring] = {**label, "ring_target": native.record(ring), "inner_end_ref": f"drawing[{inner[0]}]",
                             "qualifier_observations": qualifiers,
                             "evidence_refs": label["evidence_refs"] + native.refs(ring)
                                              + [ref for line in qualifiers for ref in line["evidence_refs"]]}
    ends = [i for i in ends if i not in ring_labels]
    bar_pairs = []
    for end in ends:
        eb = native.drawings[end]["rect"]
        end_labels = [l for l in native.leaders if l["kind"] == "arrow" and part_mark(l)
                      and abs(math.dist(l["target_display"], center(eb)) - eb.width / 2) < .25]
        end_labels += [label for label in ring_labels.values() if label["inner_end_ref"] == f"drawing[{end}]"]
        for shaft in shafts:
            bb = native.drawings[shaft]["rect"]
            shaft_labels = [l for l in native.leaders if l["kind"] == "arrow" and part_mark(l)
                            and bb.x0 <= l["target_display"][0] <= bb.x1
                            and min(abs(l["target_display"][1] - bb.y0), abs(l["target_display"][1] - bb.y1)) < .2]
            bar_pairs.extend((a, b) for a in end_labels for b in shaft_labels if part_mark(a) == part_mark(b) != plate["mark"])
    bar_marks = {part_mark(a) for a, _ in bar_pairs}
    resolved_ends = False
    if len(bar_marks) == 1 and ends and shafts:
        # Every circle must fit exactly one observed shaft row. This classifies
        # the projected symbols; it does not multiply hidden physical instances.
        matches = [[s for s in shafts if abs(center(native.drawings[e]["rect"])[1] - center(native.drawings[s]["rect"])[1]) < .2
                    and abs(native.drawings[e]["rect"].width - native.drawings[s]["rect"].height) < .1]
                   for e in ends]
        resolved_ends = all(len(m) == 1 for m in matches) and set(s for m in matches for s in m) == set(shafts)
        # Repeated round profiles must form a complete, nonoverlapping column /
        # row lattice with equal style and diameter, rather than accepting an
        # arbitrary extra circle near a shaft row.
        columns = sorted({round(center(native.drawings[e]["rect"])[0], 1) for e in ends})
        rows = sorted({round(center(native.drawings[e]["rect"])[1], 1) for e in ends})
        diameters = [native.drawings[e]["rect"].width for e in ends]
        styles = {str(native.by_drawing[e][0]["style"]) for e in ends + shafts}
        resolved_ends &= (len(ends) == len(columns) * len(rows) and len(styles) == 1
                          and max(diameters) / min(diameters) < 1.025
                          and all(b - a > max(diameters) for a, b in zip(columns, columns[1:])))
        family_box = fitz.Rect(native.drawings[shafts[0]]["rect"])
        for shaft in shafts[1:]:
            family_box |= native.drawings[shaft]["rect"]
        lengths = native.dimensions(shafts[0], 0, family_box=family_box)
        welds = [l for l in native.leaders if not l["text"].isdigit()
                 and re.search(r"GOST|ГОСТ|weld|свар", l["text"], re.I)
                 and any(near(l["target_display"], native.drawings[s]["rect"].tl)
                         or near(l["target_display"], native.drawings[s]["rect"].bl) for s in shafts)]
        resolved_ends &= len(welds) == 1
        bar = {"id": f"part:{next(iter(bar_marks))}", "mark": next(iter(bar_marks)), "kind": "welded_bar_family",
               "state": "derived" if resolved_ends else "unknown",
               "annotations": [l for pair in bar_pairs for l in pair] + welds,
               "length_dimensions": lengths,
               "end_role_basis": "same-mark arrows, native weld interface and complete congruent projected lattice",
               "end_projections": [native.record(i) for i in ends],
               "shaft_projections": [native.record(i) for i in shafts],
               "projected": {"observed_end_count": len(ends), "observed_shaft_count": len(shafts)},
               "physical": {"count": None, "length_each_mm": None, "diameter_mm": None},
               "calculated": {"total_length_m": None, "mass_kg": None}, "approved": None}
        result["child_parts"].append(bar)
        stop("bar physical multiplicity, diameter and fabrication length are not closed; projected observations are not quantities",
             [r for i in ends + shafts for r in native.refs(i)])
    if not resolved_ends:
        stop("internal round symbols cannot all be distinguished from plate holes", [r for i in ends for r in native.refs(i)])
    plate["internal_round_symbols"] = {"state": "derived" if resolved_ends else "unknown",
                                        "role": "bar_end_projections" if resolved_ends else "unresolved",
                                        "source_refs": [f"drawing[{i}]" for i in ends], "hole_count": None}
    if ring_labels:
        plate["annotation_rings"] = {"state": "derived" if resolved_ends else "unknown",
                                      "records": list(ring_labels.values()),
                                      "basis": "unique labelled concentric enclosure and same-mark shaft/end correspondence"}
    values = {name: {d["value"] for d in records} for name, records in dimensions.items()}
    closed_dimensions = all(len(v) == 1 for v in values.values())
    if closed_dimensions:
        v = {name: next(iter(items)) for name, items in values.items()}
        scales = [(pb.width / v["width"]), (pb.height / v["height"]),
                  (sb.width / v["thickness"]), (sb.height / v["section_height"])]
        closed_dimensions = v["height"] == v["section_height"] and max(scales) / min(scales) < 1.025
        plate["reprojection"] = {"points_per_unit": scales, "relative_scale_spread": max(scales) / min(scales) - 1,
                                 "shared_height_equal": v["height"] == v["section_height"], "passed": closed_dimensions}
    if not closed_dimensions:
        stop("plate native dimension chains or cross-view metric consistency remain open")
    # Unexplained closed internal profiles prohibit net plate volume. Axis and
    # leader strokes remain observations; they never become holes by subtraction.
    explained_refs = {ref for leader in native.leaders if leader["kind"] != "plain" for ref in leader["evidence_refs"]}
    explained_refs.update(section_label["evidence_refs"])
    if resolved_ends:
        explained_refs.update(ref for label in ring_labels.values() for ref in label["evidence_refs"])
    extra = []
    for i, drawing in enumerate(native.drawings):
        if i == plan or i in ends or not pb.contains(drawing["rect"]):
            continue
        segments = native.by_drawing[i]
        if not segments or all(s["id"] in explained_refs for s in segments):
            continue
        endpoints = [p for s in segments for p in (s["start"], s["end"])]
        closed = len(segments) >= 2 and all(sum(near(p, q) for q in endpoints) >= 2 for p in endpoints)
        if closed or drawing.get("fill") is not None:
            extra.append(i)
    # A hole may be serialized as several open paint paths. Replay exact native
    # vertex incidence across those paths as well; path-local closure alone is
    # insufficient. Unconnected crossings do not create graph edges here.
    parent, native_edges = {}, set()
    degenerate_refs, duplicate_refs = [], []

    def root(vertex):
        parent.setdefault(vertex, vertex)
        while parent[vertex] != vertex:
            vertex = parent[vertex]
        return vertex

    for i, segments in native.by_drawing.items():
        if i == plan or i in ends:
            continue
        for segment in segments:
            if (segment["id"] in explained_refs or not pb.contains(fitz.Rect(segment["bbox_display"]))):
                continue
            if segment["kind"] == "line":
                if math.dist(segment["start"], segment["end"]) < 1e-7:
                    degenerate_refs.append(segment["id"])
                    continue
                edge = tuple(sorted((tuple(segment["start"]), tuple(segment["end"]))))
                if edge in native_edges:
                    duplicate_refs.append(segment["id"])
                    continue
                native_edges.add(edge)
            # Global vertices include a tolerance for observation grouping.
            # A snapped microstroke is not a closed hole: replay exact authored
            # endpoints within those IDs before claiming an enclosed profile.
            a = root((segment["start_vertex_id"], tuple(segment["start"])))
            b = root((segment["end_vertex_id"], tuple(segment["end"])))
            if a == b:
                extra.append(i)
            else:
                parent[a] = b
    if degenerate_refs or duplicate_refs:
        plate["native_stroke_observations"] = {"zero_length_refs": degenerate_refs,
                                                "duplicate_straight_edge_refs": duplicate_refs,
                                                "vertex_replay": "exact authored endpoints within page-global native IDs",
                                                "role": "retained observations; no enclosed area from a point or repeated edge"}
    extra = sorted(set(extra))
    if extra:
        stop("unexplained internal closed profiles block net plate volume", [r for i in extra for r in native.refs(i)])
    if closed_dimensions and resolved_ends and not extra:
        plate["solid"] = _box_mesh((v["width"], v["height"], v["thickness"]))
        plate["solid"]["coordinate_basis"] = "relative orthographic profile; numeric axes require linear-unit confirmation"
        plate["solid"]["vertices_xyz_drawing_units"] = plate["solid"].pop("vertices_xyz_mm")
        plate["conditional"] = {"linear_unit": "mm", "volume_m3": v["width"] * v["height"] * v["thickness"] / 1e9,
                                 "state": "convention_dependent", "strict_quantity_eligible": False}
        if native_mm:
            plate["calculated"] = {"volume_m3": plate["conditional"]["volume_m3"], "state": "derived",
                                   "basis": "watertight rectangular solid and native global millimetre units"}
            plate["conditional"] = {}
            plate["solid"]["vertices_xyz_mm"] = plate["solid"].pop("vertices_xyz_drawing_units")
            plate["solid"]["coordinate_basis"] = "relative orthographic millimetre frame"
        else:
            stop("no supported native global unit sentence; millimetre plate volume remains conditional pending drawing-unit evidence")
    result["state"] = "derived"
    return result, native
