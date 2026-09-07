#!/usr/bin/env python3
"""Render the corrected page-5 source-coverage and route-recovery audit."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import fitz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.generate_mep_observed_takeoff import _artifact_json
from src.drawing_engine.project.project_packed_store import PackedProjectStore
from src.drawing_engine.disciplines.mep.mep_route_callout_tracing import validate_route_callout_traces


WHITE = (1.0, 1.0, 1.0)
INK = (.055, .075, .11)
MUTED = (.34, .39, .45)
AMBER = (.94, .49, .04)
GREEN = (.00, .47, .34)
RED = (.86, .10, .13)
SYSTEM_COLOURS = {
    "heating_hot_water_supply": (.96, .03, .16),
    "heating_hot_water_return": (1.0, .37, .02),
    "chilled_water_supply": (.03, .27, .95),
    "chilled_water_return": (.42, .56, 1.0),
    "condenser_water_supply": (.00, .55, .22),
    "condenser_water_return": (.18, .68, .38),
    "condensate_drain": (.12, .16, .21),
}
SYSTEM_LABELS = {
    "heating_hot_water_supply": "HHWS",
    "heating_hot_water_return": "HHWR",
    "chilled_water_supply": "CHWS",
    "chilled_water_return": "CHWR",
    "condenser_water_supply": "CWS",
    "condenser_water_return": "CWR",
    "condensate_drain": "CD",
}


def _read(path: Path):
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _midpoint(points):
    if not points:
        return fitz.Point(0, 0)
    if len(points) == 1:
        return fitz.Point(*points[0])
    lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
    target, walked = sum(lengths) / 2.0, 0.0
    for index, length in enumerate(lengths):
        if walked + length >= target:
            ratio = 0 if not length else (target - walked) / length
            a, b = points[index], points[index + 1]
            return fitz.Point(a[0] + ratio * (b[0] - a[0]),
                              a[1] + ratio * (b[1] - a[1]))
        walked += length
    return fitz.Point(*points[-1])


def _draw_polyline(page, points, *, colour, width, dashes=None, halo=True,
                   opacity=.96):
    if len(points) < 2:
        return
    if halo:
        shape = page.new_shape()
        shape.draw_polyline([fitz.Point(*point) for point in points])
        shape.finish(color=WHITE, width=width + 4.5, stroke_opacity=.82,
                     lineCap=1, lineJoin=1)
        shape.commit(overlay=True)
    shape = page.new_shape()
    shape.draw_polyline([fitz.Point(*point) for point in points])
    shape.finish(color=colour, width=width, dashes=dashes,
                 stroke_opacity=opacity, lineCap=1, lineJoin=1)
    shape.commit(overlay=True)


def _point_at_fraction(points, fraction):
    if len(points) < 2:
        return _midpoint(points)
    lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
    target, walked = sum(lengths) * fraction, 0.0
    for index, length in enumerate(lengths):
        if walked + length >= target:
            ratio = 0 if not length else (target - walked) / length
            a, b = points[index], points[index + 1]
            return fitz.Point(a[0] + ratio * (b[0] - a[0]),
                              a[1] + ratio * (b[1] - a[1]))
        walked += length
    return fitz.Point(*points[-1])


def _label(page, points, text, colour, bounds, occupied, *, size=12, required=False):
    width = min(310.0, max(105.0, size * .52 * len(text) + 16))
    height = size + 10
    offsets = ((10, -height - 8), (10, 8), (-width - 10, -height - 8),
               (-width - 10, 8), (-width / 2, -height - 8),
               (-width / 2, 8))
    if required:
        offsets += tuple((dx, dy) for distance in (2, 3, 4)
                         for dx in (10, -width - 10, -width / 2)
                         for dy in (-distance * (height + 12), distance * (height + 12)))
    anchor = None
    box = None
    for fraction in (.5, .25, .75):
        point = _point_at_fraction(points, fraction)
        for dx, dy in offsets:
            x = min(max(bounds.x0 + 4, point.x + dx), bounds.x1 - width - 4)
            y = min(max(bounds.y0 + 4, point.y + dy), bounds.y1 - height - 4)
            candidate = fitz.Rect(x, y, x + width, y + height)
            padded = fitz.Rect(candidate.x0 - 5, candidate.y0 - 5,
                               candidate.x1 + 5, candidate.y1 + 5)
            if not any(padded.intersects(other) for other in occupied):
                anchor, box = point, candidate
                break
        if box is not None:
            break
    if box is None:
        if required:
            raise ValueError(f"required trace label has no clear placement: {text}")
        return False
    nearest_x = min(max(anchor.x, box.x0), box.x1)
    nearest_y = min(max(anchor.y, box.y0), box.y1)
    page.draw_line(anchor, (nearest_x, nearest_y), color=colour, width=1.1,
                   stroke_opacity=.9, overlay=True)
    page.draw_rect(box, color=colour, fill=WHITE, width=1.5,
                   fill_opacity=.93, stroke_opacity=.95, overlay=True)
    page.insert_text((box.x0 + 7, box.y0 + size + 1), text, fontsize=size,
                     fontname="hebo", color=colour, overlay=True)
    occupied.append(box)
    return True


def _candidate_system_label(row):
    systems = row.get("candidate_systems") or []
    if not systems:
        return "MEP?"
    return "/".join(SYSTEM_LABELS.get(value, "?") for value in systems) + "?"


def _candidate_size_label(row):
    sizes = row.get("candidate_nominal_sizes_inches") or []
    if len(sizes) != 1:
        return "SIZE?"
    value = float(sizes[0])
    common = {0.5: '1/2"', 0.75: '3/4"', 1.0: '1"', 1.25: '1 1/4"',
              1.5: '1 1/2"', 2.0: '2"', 2.5: '2 1/2"', 3.0: '3"',
              4.0: '4"', 6.0: '6"', 10.0: '10"'}
    return common.get(round(value, 4), f'{value:g}"')


def _candidate_colour(row, styles):
    systems = row.get("candidate_systems") or []
    if systems:
        colours = [SYSTEM_COLOURS[value] for value in systems
                   if value in SYSTEM_COLOURS]
        if colours:
            return tuple(sum(value[index] for value in colours) / len(colours)
                         for index in range(3))
    return tuple(styles[row["style_id"]].get("stroke") or GREEN)


def _traced_candidate_rows(supported, identified, tracing):
    """Presentation-only proposals: accepted M4 attributes are never changed."""
    anchors = {row["id"]: f"R{index:02d}" for index, row in enumerate(identified, 1)}
    traces = {row["component_ref"]: row for row in tracing.get("component_traces", [])}
    output = []
    for original in supported:
        row = dict(original)
        trace = traces.get(row["id"], {})
        reached = sorted({anchors[path["anchor"]["component_ref"]]
                          for path in trace.get("paths_to_anchors", [])
                          if path["anchor"]["kind"] == "accepted_M4_system_anchor"
                          and path["anchor"]["component_ref"] in anchors
                          and len(path["component_path"]) > 1})
        if reached:
            row["trace_label"] = "via " + "/".join(reached)
            row["candidate_systems"] = sorted(set(row.get("candidate_systems", [])) |
                                               set(trace.get("candidate_systems", [])))
        output.append(row)
    return output


def render(*, source: Path, database: Path, recovery_path: Path,
           output: Path, page_number: int = 5, dpi: int = 150):
    recovery = _read(recovery_path)
    if "route_callout_tracing" in recovery:
        errors = validate_route_callout_traces(recovery)
        if errors:
            raise ValueError(errors)
    if recovery.get("validation", {}).get("status") != "valid_page5_diagnostic_recovery":
        raise ValueError("page-wide recovery did not pass validation")
    if recovery.get("acceptance_gate", {}).get("status") != (
            "accepted_page5_source_coverage_for_diagnostic_render"):
        raise ValueError("page-wide source coverage is not closed")
    with PackedProjectStore(database) as store:
        snapshot = store.snapshot(project_id="mep-coordination",
                                  document_id="coordination-set")
        registry = _artifact_json(
            store, project_id="mep-coordination", document_id="coordination-set",
            name="sheet-registry")
    registered = next(row for row in registry["pages"]
                      if row["page_number"] == page_number)
    scale = float(registered["fields"]["scale"]["drawing_inches_per_paper_inch"])
    styles = _read(Path(recovery["inputs"]["denominator_path"]))[
        "native_descriptor_pack"]["styles"]

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(output.name + ".partial")
    if temporary_output.exists():
        raise ValueError(f"stale temporary PDF exists: {temporary_output}")
    with fitz.open(source) as source_pdf:
        source_page = source_pdf[page_number - 1]
        source_rect = source_page.rect
        pixmap = source_page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72),
                                        alpha=False, annots=True)
        base_jpeg = pixmap.tobytes("jpeg", jpg_quality=86)
    panel_width = 720.0
    with fitz.open() as pdf:
        page = pdf.new_page(width=source_rect.width + panel_width,
                            height=source_rect.height)
        drawing = fitz.Rect(0, 0, source_rect.width, source_rect.height)
        page.insert_image(drawing, stream=base_jpeg, keep_proportion=True,
                          overlay=False)
        page.draw_rect(drawing, color=None, fill=WHITE, fill_opacity=.53, overlay=True)

        outlined = recovery["outlined_corridor_components"]
        identified = [row for row in outlined
                      if row["state"] == "identified_mep_route"]
        supported_outlined = [row for row in outlined
                              if row["state"] == "supported_unidentified_mep_candidate"]
        supported_single = recovery["single_centreline_components"]
        supported = _traced_candidate_rows(
            [*supported_outlined, *supported_single], identified,
            recovery.get("route_callout_tracing", {}))
        candidate_groups = sorted({
            (row["style_id"], tuple(row.get("candidate_systems") or []))
            for row in supported if row.get("style_id") is not None
        })
        for style_id, candidate_systems in candidate_groups:
            rows = [row for row in supported
                    if row.get("style_id") == style_id
                    and tuple(row.get("candidate_systems") or []) == candidate_systems]
            halo = page.new_shape()
            trace_shape = page.new_shape()
            for row in rows:
                points = [fitz.Point(*point) for point in row["polyline_display"]]
                if len(points) >= 2:
                    halo.draw_polyline(points)
                    trace_shape.draw_polyline(points)
            halo.finish(color=WHITE, width=12.0, stroke_opacity=.82,
                        lineCap=1, lineJoin=1)
            halo.commit(overlay=True)
            stroke = _candidate_colour(rows[0], styles)
            trace_shape.finish(color=stroke, width=7.5, dashes="[18 11] 0",
                               stroke_opacity=.91, lineCap=1, lineJoin=1)
            trace_shape.commit(overlay=True)
        for row in identified:
            system = row["attributes"]["route_system"].get("value")
            colour = SYSTEM_COLOURS.get(system, GREEN)
            _draw_polyline(page, row["polyline_display"], colour=colour,
                           width=11.5, halo=True)

        occupied = []
        for connector in recovery.get("observed_projected_connector_candidates", []):
            point = fitz.Point(*connector["point_display"])
            letter = "L" if connector.get("generic_class") == "elbow_L" else "T"
            page.draw_circle(point, 11.0, color=INK, fill=WHITE, width=2.2,
                             fill_opacity=.92, stroke_opacity=.96, overlay=True)
            page.insert_text((point.x - 4.2, point.y + 5.1), letter, fontsize=14,
                             fontname="hebo", color=INK, overlay=True)
            occupied.append(fitz.Rect(point.x - 15, point.y - 15,
                                      point.x + 15, point.y + 15))

        # Closed anchored-colour glyphs remain in the evidence record but are
        # not marked as connectors on the primary sheet.  Repeated pipe-support
        # and ceiling-attachment symbols share the same CAD colours, so a ring
        # would overstate their identity.

        # Identified rows always receive direct labels.  For the recovered
        # unknown network, label only the longest traces; all are still drawn.
        for index, row in enumerate(identified, 1):
            attrs = row["attributes"]
            system = attrs["route_system"].get("value")
            system_label = SYSTEM_LABELS.get(system, "SYSTEM UNKNOWN")
            size = attrs["route_size"].get("value")
            size_label = str(size) if size is not None else "SIZE UNKNOWN"
            metres = float(row["projected_path_display_points"] or 0) / 72 * scale * .0254
            _label(page, row["polyline_display"],
                   f"R{index:02d} {system_label} {size_label} {metres:.2f}m",
                   SYSTEM_COLOURS.get(system, GREEN), drawing, occupied, size=12)
        ranked = sorted(supported, key=lambda item: (-item["projected_path_display_points"], item["id"]))
        for index, row in sorted(enumerate(ranked, 1),
                                 key=lambda item: (not bool(item[1].get("trace_label")), item[0])):
            if index > 80 and not row.get("trace_label"):
                continue
            metres = row["projected_path_display_points"] / 72 * scale * .0254
            colour = _candidate_colour(row, styles)
            _label(page, row["polyline_display"],
                   f"U{index:02d} {_candidate_system_label(row)} "
                   f"{_candidate_size_label(row)} {metres:.2f}m"
                   + (" " + row["trace_label"] if row.get("trace_label") else ""),
                   colour, drawing, occupied, size=11, required=bool(row.get("trace_label")))

        panel = fitz.Rect(source_rect.width, 0, source_rect.width + panel_width,
                          source_rect.height)
        page.draw_rect(panel, color=None, fill=(.965, .975, .985), overlay=True)
        page.draw_rect(fitz.Rect(panel.x0, 0, panel.x1, 190), color=None,
                       fill=INK, overlay=True)
        x, y = panel.x0 + 42, 70
        page.insert_text((x, y), "PAGE 5 MEP AUDIT", fontsize=29,
                         fontname="hebo", color=WHITE); y += 48
        page.insert_text((x, y), "PROJECTED 2D - ENGINEER REVIEW",
                         fontsize=16, fontname="hebo", color=(.78, .86, .94)); y = 255
        page.insert_text((x, y), "VISUAL KEY", fontsize=23, fontname="hebo", color=INK); y += 43
        for colour, title, subtitle, dashed in (
            (RED, "SOLID SYSTEM COLOUR", "Unique accepted M4 identity", False),
            ((.10, .45, .92), "DASHED SOURCE COLOUR", "Supported MEP; system UNKNOWN", True),
        ):
            page.draw_line((x, y - 5), (x + 92, y - 5), color=colour,
                           width=8 if not dashed else 6,
                           dashes="[15 9] 0" if dashed else None)
            page.insert_text((x + 112, y), title, fontsize=17, fontname="hebo", color=INK)
            page.insert_text((x + 112, y + 25), subtitle, fontsize=14, color=MUTED)
            y += 66
        page.draw_circle(fitz.Point(x + 18, y - 5), 11, color=INK, fill=WHITE,
                         width=2.0)
        page.insert_text((x + 13.8, y), "L", fontsize=14, fontname="hebo", color=INK)
        page.insert_text((x + 48, y), "L/T = PROJECTED CONNECTOR CANDIDATE",
                         fontsize=15, fontname="hebo", color=INK)
        y += 42
        if recovery.get("route_callout_tracing"):
            page.insert_text((x, y), "via Rxx = traced to labelled route; binding pending",
                             fontsize=15, fontname="hebo", color=MUTED)
            y += 30
        y += 35
        page.insert_text((x, y), "SYSTEM TYPE / LENGTH", fontsize=23,
                         fontname="hebo", color=INK); y += 45
        metres_per_point = scale * .0254 / 72.0
        system_lengths = Counter()
        for row in identified:
            system = row["attributes"]["route_system"].get("value")
            system_lengths[system] += float(
                row.get("projected_path_display_points") or 0.0) * metres_per_point
        for system, length in sorted(
                system_lengths.items(), key=lambda item: SYSTEM_LABELS.get(
                    item[0], str(item[0]))):
            label = SYSTEM_LABELS.get(system, str(system or "UNKNOWN"))
            colour = SYSTEM_COLOURS.get(system, INK)
            page.draw_line((x, y - 5), (x + 42, y - 5), color=colour, width=8)
            page.insert_text((x + 62, y), label, fontsize=18,
                             fontname="hebo", color=INK)
            page.insert_text((panel.x1 - 170, y), f"{length:.2f} m", fontsize=18,
                             fontname="hebo", color=INK)
            y += 42
        if not system_lengths:
            page.insert_text((x, y), "NONE IDENTIFIED", fontsize=18,
                             fontname="hebo", color=MUTED); y += 42

        y += 30
        page.insert_text((x, y), "CANDIDATE LENGTH", fontsize=23,
                         fontname="hebo", color=INK); y += 45
        candidate_lengths = Counter()
        for row in supported:
            candidate_lengths[_candidate_system_label(row)] += float(
                row.get("projected_path_display_points") or 0.0) * metres_per_point
        for label, candidate_length in sorted(candidate_lengths.items()):
            page.insert_text((x, y), label, fontsize=18,
                             fontname="hebo", color=INK)
            page.insert_text((panel.x1 - 170, y), f"{candidate_length:.2f} m", fontsize=18,
                             fontname="hebo", color=INK); y += 42

        y += 30
        page.insert_text((x, y), "CONNECTOR TYPE / AMOUNT", fontsize=23,
                         fontname="hebo", color=INK); y += 45
        connector_counts = Counter(
            row.get("generic_class")
            for row in recovery.get("observed_projected_connector_candidates", [])
            if row.get("state") == "observed_projected_connector_candidate"
            and row.get("generic_class"))
        if connector_counts:
            for connector, count in sorted(connector_counts.items()):
                label = {"elbow_L": "L / ELBOW (OBSERVED)",
                         "tee_T": "T / TEE (OBSERVED)"}.get(
                             connector, connector.upper().replace("_", " "))
                page.insert_text((x, y), label,
                                 fontsize=18, fontname="hebo", color=INK)
                page.insert_text((panel.x1 - 105, y), str(count), fontsize=18,
                                 fontname="hebo", color=INK); y += 42
        else:
            page.insert_text((x, y), "NONE IDENTIFIED", fontsize=18,
                             fontname="hebo", color=MUTED)
            page.insert_text((panel.x1 - 105, y), "0", fontsize=18,
                             fontname="hebo", color=MUTED)
        pdf.set_metadata({
            "title": "Page 5 exhaustive MEP route recovery diagnostic",
            "subject": "Corrected page-wide source coverage; engineer review pending",
            "keywords": "MEP, page 5, exhaustive denominator, route recovery, diagnostic",
        })
        pdf.save(temporary_output, garbage=4, deflate=True, clean=True)
    os.replace(temporary_output, output)

    with fitz.open(output) as check:
        if len(check) != 1:
            raise ValueError("page-5 diagnostic must contain exactly one page")
        images = check[0].get_images(full=True)
        xobjects = check[0].get_xobjects()
        drawings = len(check[0].get_drawings())
        if len(images) != 1 or xobjects:
            raise ValueError("audit base must be one raster image and vector overlay only")
    manifest = {
        "schema_version": "0.1.0",
        "artifact": "mep_page5_page_wide_route_recovery_diagnostic",
        "development_status": "engineer_review_pending_not_engineer_facing",
        "source_pdf": str(source.resolve()), "source_pdf_sha256": _sha(source),
        "source_page": page_number, "project_snapshot_v3": snapshot["id"],
        "recovery_path": str(recovery_path.resolve()),
        "recovery_sha256": _sha(recovery_path),
        "output_pdf": str(output.resolve()), "output_pdf_sha256": _sha(output),
        "pdf_page_count": 1,
        "source_base": {"representation": "raster_jpeg", "dpi": dpi,
                        "pixel_width": pixmap.width, "pixel_height": pixmap.height},
        "overlay": {"representation": "native_pdf_vector",
                    "vector_drawing_count": drawings},
        "route_callout_tracing": recovery.get("route_callout_tracing", {}).get("summary"),
        "installed_length": None, "purchase_length": None,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=ROOT / "M&P mark-up against shop systems piping.pdf")
    parser.add_argument("--database", type=Path,
                        default=ROOT / "data/projects/mep/project-v3.sqlite")
    parser.add_argument("--recovery", type=Path, default=ROOT /
                        "output/mep-page5-page-wide-recovery-v20-2026-09-03/recovery.json")
    parser.add_argument("--output", type=Path, default=ROOT /
                        "output/pdf/mep_page5_network_tracing_2026-09-03.pdf")
    parser.add_argument("--page", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(render(source=args.source, database=args.database,
                            recovery_path=args.recovery, output=args.output,
                            page_number=args.page), indent=2))


if __name__ == "__main__":
    main()
