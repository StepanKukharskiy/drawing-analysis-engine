#!/usr/bin/env python3
"""Render a coverage-limited marked MEP PDF from frozen certificates only.

This presents assisted or automatic frozen evidence, never solves quantities.
The JSON manifest preserves the exact catalog IDs and evidence used by the app.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

import fitz

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.drawing_engine.audit.audit_presentation import audit_font_file, item_palette_color
from src.drawing_engine.disciplines.mep.mep_item_catalog import validate_mep_item_catalog
from src.drawing_engine.disciplines.mep.mep_takeoff_adapter import build_mep_takeoff_adapter
from src.drawing_engine.project.takeoff_intelligence import canonical_sha256


FIXTURE = ROOT / "fixtures/mep/m_and_p_coordination"
CHECKPOINT = FIXTURE / "real_m2_m5_checkpoint"
DEFAULT_SOURCE = ROOT / "M&P mark-up against shop systems piping.pdf"
DEFAULT_OUTPUT = ROOT / "output/pdf/m_and_p_coordination_partial_marked_audit.pdf"
DEFAULT_INPUTS = {
    "catalog": FIXTURE / "m_and_p_coordination.mep-item-catalog.json",
    "coverage": FIXTURE / "m_and_p_coordination.occurrence-coverage-audit.json",
    "composites": CHECKPOINT / "pages_1a_1b.outlined-route-composites.json",
    "geometry": CHECKPOINT / "pages_1a_1b.bounded-local-3d.json",
    "annotations": FIXTURE / "m_and_p_coordination.annotation-observations.json",
}
AUDIT_PAGE_CACHE_VERSION = "1.0.0"


def _atomic_bytes(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(body)
    partial.replace(path)


def _source_page_sha256(document, page_number):
    """Hash one self-contained source page, independent of sibling pages."""
    with fitz.open() as single:
        single.insert_pdf(document, from_page=page_number - 1, to_page=page_number - 1,
                          links=False, annots=True)
        body = single.tobytes(garbage=0, deflate=False, no_new_id=True)
    return hashlib.sha256(body).hexdigest()


def _cached_source_raster(page, *, page_number, cache_dir, raster_dpi):
    """Return a hash-bound JPEG background and whether it was reused."""
    source_page_sha256 = _source_page_sha256(page.parent, page_number)
    key = hashlib.sha256(json.dumps({
        "version": AUDIT_PAGE_CACHE_VERSION,
        "source_page_sha256": source_page_sha256,
        "raster_dpi": raster_dpi,
        "jpeg_quality": 85,
        "annotations_rendered": True,
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    jpeg = cache_dir / f"{key}.jpg"
    receipt_path = cache_dir / f"{key}.json"
    reused = False
    receipt = None
    if jpeg.exists() and receipt_path.exists():
        try:
            receipt = json.loads(receipt_path.read_text())
            reused = (receipt.get("cache_key") == key
                      and receipt.get("jpeg_sha256") == hashlib.sha256(jpeg.read_bytes()).hexdigest())
        except (OSError, ValueError):
            reused = False
    if not reused:
        image = page.get_pixmap(matrix=fitz.Matrix(raster_dpi / 72, raster_dpi / 72),
                                annots=True)
        body = image.tobytes("jpeg", jpg_quality=85)
        receipt = {
            "schema_version": "mep_audit_source_page_cache.0.1.0",
            "cache_key": key,
            "source_page_sha256": source_page_sha256,
            "raster_dpi": raster_dpi,
            "jpeg_quality": 85,
            "width_pixels": image.width,
            "height_pixels": image.height,
            "jpeg_sha256": hashlib.sha256(body).hexdigest(),
        }
        _atomic_bytes(jpeg, body)
        _atomic_bytes(receipt_path,
                      (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())
    return jpeg.read_bytes(), receipt, reused


def _cached_detail_raster(page, *, source_page_sha256, clip, cache_dir):
    """Cache one exact close-up raster independently from its audit card."""
    clip_values = [round(float(value), 6) for value in clip]
    key = hashlib.sha256(json.dumps({
        "version": AUDIT_PAGE_CACHE_VERSION,
        "kind": "detail_crop",
        "source_page_sha256": source_page_sha256,
        "clip_display": clip_values,
        "scale": [2, 2], "jpeg_quality": 90,
        "annotations_rendered": True,
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    jpeg = cache_dir / f"{key}.detail.jpg"
    receipt_path = cache_dir / f"{key}.detail.json"
    reused = False
    receipt = None
    if jpeg.exists() and receipt_path.exists():
        try:
            receipt = json.loads(receipt_path.read_text())
            reused = (receipt.get("cache_key") == key
                      and receipt.get("jpeg_sha256") == hashlib.sha256(jpeg.read_bytes()).hexdigest())
        except (OSError, ValueError):
            reused = False
    if not reused:
        image = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip, annots=True)
        body = image.tobytes("jpeg", jpg_quality=90)
        receipt = {
            "schema_version": "mep_audit_detail_cache.0.1.0",
            "cache_key": key, "source_page_sha256": source_page_sha256,
            "clip_display": clip_values, "scale": [2, 2], "jpeg_quality": 90,
            "width_pixels": image.width, "height_pixels": image.height,
            "jpeg_sha256": hashlib.sha256(body).hexdigest(),
        }
        _atomic_bytes(jpeg, body)
        _atomic_bytes(receipt_path,
                      (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())
    return jpeg.read_bytes(), receipt, reused
INK = (0.08, 0.14, 0.22)
MUTED = (0.32, 0.39, 0.46)
BLUE = (0.02, 0.42, 0.68)
AMBER = (0.66, 0.33, 0.02)
PAPER = (0.95, 0.97, 0.98)
WHITE = (1, 1, 1)


def _unique(rows, key="id"):
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate {key} in frozen records")
    return result


def build_manifest(inputs: dict, *, compact_regions=False) -> dict:
    """Project existing evidence, fail closed on mismatched or missing targets."""
    return _build_manifest(inputs, automatic="discovery" in inputs, compact_regions=compact_regions)


def build_automatic_manifest(inputs: dict, *, compact_regions=False) -> dict:
    """Render only the catalog targets certified by the frozen discovery packet."""
    if "discovery" not in inputs:
        raise ValueError("automatic audit requires a frozen discovery packet")
    return _build_manifest(inputs, automatic=True, compact_regions=compact_regions)


def _review_discovery_record(discovery):
    if discovery is None:
        return None
    result = deepcopy({key: value for key, value in discovery.items() if key != "regions"})
    if "regions" in discovery:
        regions = discovery["regions"]
        result["region_search_summary"] = {
            "region_count": len(regions),
            "incomplete_region_count": sum(not region["relevant_competitor_search_complete"] for region in regions),
            "full_region_evidence_sha256": canonical_sha256(regions),
            "full_evidence_contract": "input_payload_sha256.discovery",
        }
    return result


def _automatic_discovery(inputs, pages):
    discovery = inputs["discovery"]
    if discovery.get("execution_mode") != "automatic_frozen_replay" or discovery.get("authority", {}).get("reviewed_selectors_used") is not False:
        raise ValueError("automatic discovery must explicitly exclude reviewed selectors")
    for name in ("catalog", "composites", "geometry", "annotations"):
        if discovery.get("input_payload_sha256", {}).get(name) != canonical_sha256(inputs[name]):
            raise ValueError(f"{name}: frozen automatic discovery hash mismatch")
    discovery_pages = _unique(discovery["pages"], "page_ref")
    if set(discovery_pages) != set(pages):
        raise ValueError("automatic discovery page registry mismatch")
    if any(not row.get("discovery_state") or row.get("item_inventory_complete") is not False
           for row in discovery_pages.values()):
        raise ValueError("automatic page discovery state or incomplete inventory missing")
    targets = _unique(discovery["accepted_targets"])
    expected = {}
    for occurrence in inputs["catalog"]["item_occurrences"]:
        if len(occurrence["target_refs"]) != 1:
            raise ValueError("automatic audit requires one certified page-local target")
        ref = occurrence["target_refs"][0]
        expected.setdefault(ref, []).append(occurrence["id"])
        if targets.get(ref, {}).get("page_ref") != occurrence["source"]["page_ref"]:
            raise ValueError("automatic target does not certify the occurrence source page")
    if set(targets) != set(expected) or any(sorted(row.get("item_occurrence_refs", [])) != sorted(expected[ref])
                                            for ref, row in targets.items()):
        raise ValueError("automatic accepted target and catalog membership mismatch")
    proposals = deepcopy(discovery.get("unresolved_proposals", []))
    _unique(proposals)
    for row in proposals:
        if row.get("page_ref") not in pages or row.get("quantity_eligible") is not False:
            raise ValueError("unresolved proposal has invalid page or quantity authority")
        if not all(row.get(key) for key in ("id", "description", "reason")):
            raise ValueError("unresolved proposal identity or explanation missing")
        if row["id"] in expected or row["id"] in {ref for refs in expected.values() for ref in refs}:
            raise ValueError("unresolved proposal aliases an accepted item or target")
        geometry = row.get("source_geometry", {})
        points = geometry.get("points_display", [])
        box = geometry.get("bbox_display")
        if points and (len(points) < 2 or any(len(point) != 2 or not all(math.isfinite(v) for v in point) for point in points)):
            raise ValueError("invalid unresolved proposal geometry")
        if box is not None and (len(box) != 4 or not all(math.isfinite(v) for v in box) or box[0] >= box[2] or box[1] >= box[3]):
            raise ValueError("invalid unresolved proposal bounding box")
    return discovery_pages, proposals


def _build_manifest(inputs: dict, *, automatic: bool, compact_regions=False) -> dict:
    catalog = inputs["catalog"]
    errors = validate_mep_item_catalog(catalog)
    if errors:
        raise ValueError("invalid frozen catalog: " + "; ".join(errors))
    document = catalog["document"]
    for name, payload in inputs.items():
        if any(payload["document"].get(key) != document[key] for key in document):
            raise ValueError(f"{name}: source document mismatch")
    contracts = [("m5a_contract_ref", "geometry")]
    if not automatic:
        contracts.append(("coverage_audit_contract_ref", "coverage"))
    for key, name in contracts:
        if (catalog.get(key) or {}).get("payload_sha256") != canonical_sha256(inputs[name]):
            raise ValueError(f"{name}: frozen catalog hash mismatch")
    if inputs["geometry"]["m3_5_contract_ref"]["payload_sha256"] != canonical_sha256(inputs["composites"]):
        raise ValueError("composites: frozen geometry hash mismatch")
    if ("bindings" in inputs) != ("terminology" in inputs):
        raise ValueError("source label context requires both bindings and terminology")
    if "bindings" in inputs:
        if catalog["m4_contract_ref"]["payload_sha256"] != canonical_sha256(inputs["bindings"]):
            raise ValueError("bindings: frozen catalog hash mismatch")
        if inputs["bindings"]["m2_contract_ref"]["payload_sha256"] != canonical_sha256(inputs["terminology"]):
            raise ValueError("terminology: frozen binding hash mismatch")
    if not automatic and inputs["coverage"]["m0_contract_ref"]["payload_sha256"] != canonical_sha256(inputs["annotations"]):
        raise ValueError("annotations: frozen coverage hash mismatch")
    pages = _unique(catalog["pages"], "page_ref")
    page_numbers = sorted(row["pdf_page_number"] for row in pages.values())
    if page_numbers != list(range(1, document["page_count"] + 1)):
        raise ValueError("frozen page registry is incomplete")
    discovery_pages, proposals = _automatic_discovery(inputs, pages) if automatic else ({}, [])
    coverage = {} if automatic else _unique(inputs["coverage"]["pages"], "page_ref")
    if not automatic and set(coverage) != set(pages):
        raise ValueError("coverage page registry mismatch")
    composites = _unique(inputs["composites"]["accepted_composites"])
    segments = _unique(inputs["geometry"]["bounded_local_3d_segments"])
    takeoff = _unique(catalog["takeoff_lines"], "item_occurrence_ref")
    shared = {row["native_record_refs"][0]: row
              for row in build_mep_takeoff_adapter(catalog)["occurrences"]}
    rows = []
    for occurrence in sorted(catalog["item_occurrences"], key=lambda row: row["id"]):
        if len(occurrence["target_refs"]) != 1:
            raise ValueError("partial renderer requires one certified page-local target")
        composite = composites.get(occurrence["target_refs"][0])
        page_ref = occurrence["source"]["page_ref"]
        if not composite or composite["state"] != "accepted" or composite["page_ref"] != page_ref:
            raise ValueError("occurrence has no accepted target on its source page")
        points = composite["derived_geometry"]["centreline_points_display"]
        if len(points) < 2 or any(len(point) != 2 or not all(math.isfinite(v) for v in point) for point in points):
            raise ValueError("invalid page-local target geometry")
        geometry_ref = occurrence.get("associated_bounded_local_3d_segment_ref")
        segment = segments.get(geometry_ref)
        if geometry_ref and (not segment or segment["state"] != "accepted"
                or composite["id"] not in segment["source_route_composite_refs"]
                or geometry_ref not in occurrence["evidence_refs"]):
            raise ValueError("bounded geometry is not backed by the occurrence evidence")
        rows.append({
            "id": occurrence["id"],
            "shared_takeoff_occurrence_id": shared[occurrence["id"]]["id"],
            "source": deepcopy(occurrence["source"]),
            "occurrence": deepcopy(occurrence),
            "takeoff_line": deepcopy(takeoff[occurrence["id"]]),
            "evidence_refs": deepcopy(occurrence["evidence_refs"]),
            "source_primitive_refs": deepcopy(composite["member_source_primitive_refs"]),
            "target_ref": composite["id"],
            "centreline_points_display": deepcopy(points),
            "bounded_local_3d_segment_ref": geometry_ref,
            "duplicate_projection_links": [],
            "detail_section_references": [],
            "detail_section_reference_state": "unresolved_discovery_not_run",
            "quantity_eligible": False,
        })
    for row in rows:
        if "bindings" in inputs:
            observations = {item["id"]: item for item in inputs["terminology"]["source_observations"]}
            context = {}
            for relation in inputs["bindings"]["relations"]:
                if (relation["state"] != "accepted" or relation["target_refs"] != [row["target_ref"]]
                        or relation["id"] not in row["evidence_refs"]):
                    continue
                bound_evidence = [entry for entry in inputs["bindings"]["binding_evidence"]
                                  if entry["id"] in relation["binding_evidence_refs"]]
                for ref in relation["proposal_evidence_refs"]:
                    observation = observations.get(ref)
                    if observation is None or observation["page_ref"] != row["source"]["page_ref"]:
                        continue
                    entry = context.setdefault(ref, {"observation_ref": ref, "text": observation["text"],
                        "bbox_display": observation["bbox_display"], "relation_refs": [], "contact_points_display": []})
                    entry["relation_refs"].append(relation["id"])
                    for evidence in bound_evidence:
                        point = evidence.get("automatic_search_certificate", {}).get("contact_point_display")
                        if evidence.get("target_refs") == [row["target_ref"]] and point is not None and point not in entry["contact_points_display"]:
                            entry["contact_points_display"].append(point)
            row["source_label_context"] = list(context.values())
        segment = segments.get(row["bounded_local_3d_segment_ref"])
        if segment and segment["certificates"].get("duplicate_occurrence_agreement") is True:
            row["duplicate_projection_links"] = [{
                "occurrence_ref": other["id"], "source": deepcopy(other["source"]),
                "evidence_ref": segment["id"], "additive": False,
                "relationship": "duplicate_projection_of_bounded_local_geometry",
            } for other in rows if other["id"] != row["id"]
                and other["bounded_local_3d_segment_ref"] == segment["id"]]
    page_records = []
    for page in sorted(pages.values(), key=lambda row: row["pdf_page_number"]):
        record = deepcopy(page)
        record["coverage_record"] = deepcopy(coverage.get(page["page_ref"]))
        record["execution_mode"] = "automatic_frozen_replay" if automatic else "assisted_frozen_replay"
        discovery = discovery_pages.get(page["page_ref"])
        record["discovery_record"] = _review_discovery_record(discovery) if compact_regions else deepcopy(discovery)
        record["stage_coverage"] = {
            "M0_M1": "frozen_source_observations_available",
            "M3_M5A": ("automatic_certified_subset_only" if automatic else "review_selected_subset_only") if page["item_occurrence_refs"] else "no_certified_item_geometry_supplied",
            "automatic_item_discovery": discovery_pages[page["page_ref"]]["discovery_state"] if automatic else "not_run",
            "detail_section_reference_discovery": "not_run",
            "M7_calculated_declared_reviewed_approved": "unresolved",
        }
        page_records.append(record)
    return {
        "schema_version": "0.1.0", "layer": "mep_partial_marked_audit",
        "execution_mode": "automatic_frozen_replay" if automatic else "assisted_frozen_replay", "coverage_complete": False,
        "geometry_selection_basis": "automatic bounded target discovery; no reviewed geometry selectors; frozen source registration retained" if automatic else "reviewed native primitive and annotation selections; not automatic package discovery",
        "document": deepcopy(document),
        "input_payload_sha256": {name: canonical_sha256(payload) for name, payload in inputs.items()},
        "pages": page_records, "item_rows": rows,
        "bounded_local_3d_segments": deepcopy(list(segments.values())),
        "reviewed_unresolved_findings": [] if automatic else deepcopy(inputs["coverage"]["reviewed_unresolved_findings"]),
        "unresolved_proposals": proposals,
        "original_annotation_observations": deepcopy(inputs["annotations"]["annotations"]),
        "reference_register": {"state": "unresolved_discovery_not_run", "references": [], "complete": False},
        "authority": {"renderer_establishes_facts": False, "quantity_eligible": False,
                      "approved_for_quote": False, "duplicate_projections_additive": False},
    }


class _Column:
    """A measured text column; layout overflow is an error, never hidden text."""

    def __init__(self, page, x, width, y=30, size=14):
        self.page, self.x, self.width, self.y, self.size = page, x, width, y, size

    def _layout(self, text, size, bold):
        size = size or self.size
        font = fitz.Font(fontfile=audit_font_file(bold))
        lines = []
        for paragraph in str(text).split("\n"):
            line = ""
            for word in paragraph.split(" "):
                candidate = (line + " " + word).strip()
                if font.text_length(candidate, fontsize=size) > self.width and line:
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            lines.append(line)
        if any(font.text_length(line, fontsize=size) > self.width for line in lines):
            raise ValueError(f"unbreakable audit text exceeds column: {text}")
        height = len(lines) * size * 1.4 + size * 0.6
        return lines, height

    def text(self, text, *, size=None, bold=False, color=INK, gap=8):
        size = size or self.size
        lines, height = self._layout(text, size, bold)
        rect = fitz.Rect(self.x, self.y, self.x + self.width, self.y + height)
        if self.page is None:
            self.y += height + gap
            return rect
        if rect.y1 > self.page.rect.height - 30:
            raise ValueError("audit sidebar overflow; use a continuation sheet")
        result = self.page.insert_textbox(rect, "\n".join(lines), fontsize=size,
            fontname="AuditBold" if bold else "AuditRegular", fontfile=audit_font_file(bold),
            color=color, lineheight=1.4)
        if result < 0:
            raise ValueError(f"audit text clipped: {text}")
        self.y += height + gap
        return rect

    def heading(self, text):
        return self.text(text, size=self.size * 1.25, bold=True, color=BLUE, gap=10)


def _value(value):
    return "unresolved" if value is None else str(value)


def _measurement(value):
    if value is None:
        return "unresolved"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _preview(column, segment, occurrence_id):
    """A schematic projection of supplied XYZ only, with no derived quantity."""
    column.text("LOCAL 3D | " + occurrence_id, bold=True, size=column.size * .8)
    column.text(segment["label"], color=AMBER)
    points = segment["centreline_points_xyz_m"]
    projected = [(x - .6 * y, .32 * x + .32 * y - z) for x, y, z in points]
    x0, x1 = min(p[0] for p in projected), max(p[0] for p in projected)
    y0, y1 = min(p[1] for p in projected), max(p[1] for p in projected)
    height = column.size * 5
    scale = min((column.width - 70) / max(x1 - x0, .001), (height - 20) / max(y1 - y0, .001))
    display = [fitz.Point(column.x + 35 + (x - x0) * scale, column.y + 8 + (y - y0) * scale) for x, y in projected]
    if column.page is not None:
        column.page.draw_polyline(display, color=BLUE, width=5)
        for point in (display[0], display[-1]):
            column.page.draw_circle(point, 6, color=AMBER, fill=WHITE, width=2)
    column.y += height
    envelope = segment["physical_envelope_dimension"]
    column.text("Schematic XYZ projection; open circles = unresolved analysis caps. "
                "Absolute origin unresolved. Width agreement tolerance "
                f"{envelope['agreement_tolerance_m']:.3f} m. Display values rounded; exact evidence in manifest.", size=column.size * .85, color=MUTED)


def _render_item_row(column, row, segments):
    """The same measured row is used beside the source and on continuations."""
    size = column.size
    occurrence = row["occurrence"]
    top = column.y
    navigation = []
    column.text(row["id"], bold=True, color=BLUE)
    column.text("App ID: " + row["shared_takeoff_occurrence_id"], size=size * .8, color=MUTED)
    column.text(occurrence["description"], bold=True)
    counts = occurrence["counts"]
    column.text(f"Observed: {_value(counts['observed_occurrence_count'])} projected occurrence | "
                f"Physical: {_value(counts['physical_instance_count'])}\n"
                f"Deduplicated projected count: {_value(counts['deduplicated_projected_count'])}; nonadditive")
    for channel, value in row["takeoff_line"]["value_channels"].items():
        label = channel.replace("_", " ").capitalize()
        column.text(f"{label}: {_value(value['value'])} | unit: {_value(value['unit'])} | "
                    f"basis: {value.get('reason', value.get('state', 'unresolved')).replace('_', ' ')}", gap=3)
    dimensions = [f"{value['dimension_type'].replace('_', ' ')}: {_measurement(value['value'])} {value['unit']}"
                  for value in occurrence["typed_dimensions"]]
    elevations = [f"{value['elevation_type']} elevation: {_measurement(value['value'])} {value['unit']}"
                  for value in occurrence["elevations"]]
    column.text("\n".join(dimensions + elevations) or "Dimensions / elevations: unresolved")
    column.text("Manufacturer: " + _value(occurrence["manufacturer"]) + " | Model: " + _value(occurrence["model_number"]) +
                "\nSpecification: " + _value(occurrence["specification_reference"]) + " | Mounting: " + _value(occurrence["mounting_type"]))
    column.text("Detail / section applicability: unresolved. No certified reference supplied.")
    for link in row["duplicate_projection_links"]:
        rect = column.text(f"Duplicate projection on p.{link['source']['pdf_page_number']}: {link['occurrence_ref']} (nonadditive)", color=BLUE)
        navigation.append((rect, link["occurrence_ref"]))
    column.text("State: " + occurrence["epistemic_state"] + "; physical identity, installed quantity and approval unresolved.", color=AMBER)
    column.text("Native evidence: " + "; ".join(row["source_primitive_refs"]), size=size * .8, color=MUTED)
    column.text("Target: " + row["target_ref"], size=size * .8, color=MUTED)
    if row["bounded_local_3d_segment_ref"]:
        segment = segments[row["bounded_local_3d_segment_ref"]]
        column.text(segment["id"], size=size * .8, color=MUTED)
        _preview(column, segment, row["id"])
    if column.page is not None:
        row["schedule_rect_display"] = [column.x, top, column.x + column.width, column.y]
        row["audit_schedule_page_number"] = column.page.number + 1
    column.y += 14
    if column.page is not None:
        column.page.draw_line((column.x, column.y), (column.x + column.width, column.y), color=(.7, .76, .8), width=1)
    column.y += 16
    return navigation


def _item_row_height(column, row, segments):
    measure = _Column(None, column.x, column.width, y=0, size=column.size)
    _render_item_row(measure, row, segments)
    return measure.y


def _render_sidebar(page, record, rows, findings, segments, source_width, proposals=()):
    sidebar = page.rect.width - source_width
    page.draw_rect(fitz.Rect(source_width, 0, page.rect.width, page.rect.height), color=None, fill=PAPER, width=0)
    size = 15 if page.rect.height > 1200 else 10
    column = _Column(page, source_width + 28, sidebar - 56, size=size)
    column.text("PARTIAL MARKED M&P AUDIT", size=size * 1.7, bold=True)
    automatic = record["execution_mode"] == "automatic_frozen_replay"
    column.text(("AUTOMATIC TARGET DISCOVERY | BOUNDED SUBSET ONLY" if automatic else "ASSISTED REPLAY | COVERAGE LIMITED") +
                " | NOT FOR QUOTATION", bold=True, color=AMBER)
    column.text(f"Source sheet {record['pdf_page_number']} | {_value(record['drawing_sheet_number'])} | "
                + record["sheet_role"].replace("_", " "), bold=True)
    column.text("Original source at native size. Blue = certified occurrence evidence; amber dashed = "
                + ("unresolved automatic proposal." if automatic else "unresolved authored hint.")
                + " Original engineer markups remain claims.")
    column.text("Geometry comes from frozen automatic target certificates, without reviewed geometry selectors. "
                "Source registration remains the supplied frozen registry; package completeness is not established."
                if automatic else "Geometry uses reviewed 1A/1B selections. Automatic package item discovery and "
                "plan/detail/section reference discovery have not run.", color=AMBER)
    column.heading("PAGE COVERAGE")
    column.text((record["stage_coverage"]["automatic_item_discovery"] if automatic else
                 record["coverage_audit_state"]).replace("_", " "))
    column.text("M0/M1: frozen observations. M3-M5A: " +
                (("automatic certified subset only." if automatic else "review-selected subset only.") if rows else "no certified geometry supplied.") +
                " M7: calculated / declared / reviewed / approved unresolved.")
    proposal_navigation = None
    if proposals:
        proposal_navigation = column.text(f"{len(proposals)} unresolved automatic proposals: see linked appendix. "
                                          "These are not accepted items or item counts.", bold=True, color=AMBER)
    column.heading("PARTIAL CERTIFIED OCCURRENCES" if automatic else "ITEM SCHEDULE")
    if automatic:
        column.text("Accepted relations only; missing system, size or identity remain unresolved. "
                    "An occurrence is not a fully identified physical item.", color=AMBER)
    if not rows:
        column.text("No accepted item rows in this frozen subset. This does not establish that "
                    "items are absent from the drawing.", bold=True)
        column.text("Observed amount: unresolved | Physical amount: unresolved\n"
                    "Calculated amount / unit / basis: unresolved\n"
                    "Declared amount / unit / basis: unresolved\n"
                    "Dimensions / specifications / detail links: unresolved")
    navigation = []
    deferred = []
    continuation_link = None
    for index, row in enumerate(rows):
        if automatic and column.y + _item_row_height(column, row, segments) > page.rect.height - 300:
            deferred = rows[index:]
            continuation_link = column.text("Further partial certified occurrences continue on linked schedule pages. "
                                            "Every exact occurrence ID remains marked on this source sheet.", bold=True, color=BLUE)
            break
        navigation.extend(_render_item_row(column, row, segments))
    if findings:
        column.heading("UNRESOLVED MARKUP HINTS - NOT ITEM COUNTS")
        for finding in findings:
            top = column.y
            column.text(finding["id"], bold=True, color=AMBER)
            column.text(finding["description"] + ": " + finding["reason"])
            column.text("Amount / unit: unresolved | M4 target: unresolved", color=AMBER)
            finding["schedule_rect_display"] = [column.x, top, column.x + column.width, column.y]
    column.heading("REFERENCE REGISTER")
    column.text("Unresolved: reference discovery not run. An empty register is not proof that "
                "this sheet has no detail or section callouts.")
    column.text("No summation of projection rows. Full evidence and stable IDs are embedded "
                "in audit-manifest.json and supplied in the companion manifest.", size=size * .85, color=MUTED)
    return navigation, proposal_navigation, deferred, continuation_link


def _render_schedule_continuations(pdf, pending, segments):
    navigation = []
    continuation_pages = []
    for record, rows, sidebar_link in pending:
        column = None
        first_target = None
        for row in rows:
            measure = column or _Column(None, 32, 1126, size=12)
            height = _item_row_height(measure, row, segments)
            if column is None or column.y + height > 1654:
                page = pdf.new_page(width=1190, height=1684)
                continuation_pages.append(page.number + 1)
                column = _Column(page, 32, 1126, size=12)
                column.heading("PARTIAL CERTIFIED OCCURRENCES - SCHEDULE CONTINUATION")
                column.text(f"Source p.{record['pdf_page_number']} | {_value(record['drawing_sheet_number'])} | "
                            "AUTOMATIC BOUNDED SUBSET ONLY | NOT FOR QUOTATION", color=AMBER)
                column.text("Accepted relations only; item identity and unobserved attributes remain unresolved. "
                            "Click each row for its exact source target.")
                if first_target is None:
                    first_target = (page.number, fitz.Point(column.x, column.y))
            if column.y + height > 1654:
                raise ValueError("certified occurrence row exceeds a continuation page")
            navigation.extend((column.page.number, rect, target) for rect, target in _render_item_row(column, row, segments))
        if first_target is not None:
            target_index, target_point = first_target
            pdf[record["pdf_page_number"] - 1].insert_link({"kind": fitz.LINK_GOTO, "from": sidebar_link,
                "page": target_index, "to": target_point, "zoom": 1.2})
    return navigation, continuation_pages


def _proposal_box(proposal):
    geometry = proposal.get("source_geometry", {})
    if geometry.get("bbox_display") is not None:
        return fitz.Rect(geometry["bbox_display"])
    points = geometry.get("points_display", [])
    if points:
        return fitz.Rect(min(p[0] for p in points) - 2, min(p[1] for p in points) - 2,
                         max(p[0] for p in points) + 2, max(p[1] for p in points) + 2)
    return None


def _render_proposals(pdf, manifest, sidebar_links):
    """Paginate candidates separately, preserving every frozen proposal and link."""
    pages = {row["page_ref"]: row for row in manifest["pages"]}
    proposals = sorted(manifest.get("unresolved_proposals", []),
                       key=lambda row: (pages[row["page_ref"]]["pdf_page_number"], row["id"]))
    column = None
    first_on_source = {}
    appendix_pages = []
    protected = {ref: [fitz.Rect(row["source_mark_rect_display"]) for row in manifest["item_rows"]
                      if row["source"]["page_ref"] == ref] for ref in pages}
    for proposal in proposals:
        record = pages[proposal["page_ref"]]
        box = _proposal_box(proposal)
        draw_source_frame = box is not None and not any(box.intersects(other) for other in protected[proposal["page_ref"]])
        blocks = [
            (proposal["id"], {"bold": True, "color": AMBER}),
            (f"Source p.{record['pdf_page_number']} | {_value(record['drawing_sheet_number'])} | " + proposal["description"], {}),
            ("Unresolved: " + proposal["reason"].replace("_", " ") + " | Amount / unit: unresolved; not an accepted item.", {}),
            ("Evidence: " + ("; ".join(proposal.get("evidence_refs", [])) or "unresolved"), {"size": 8, "color": MUTED}),
        ]
        if box is not None and not draw_source_frame:
            blocks.append(("Source frame omitted to preserve overlapping labels/frames; this row still links to the exact source region.",
                           {"size": 8, "color": AMBER}))
        # Measure before writing: never partially print or silently omit a row.
        measure = column or _Column(None, 32, 1126, size=10)
        height = sum(measure._layout(text, style.get("size", 10), style.get("bold", False))[1] + 8 for text, style in blocks) + 14
        if column is None or column.y + height > 810:
            page = pdf.new_page(width=1190, height=842)
            appendix_pages.append(page.number + 1)
            column = _Column(page, 32, 1126, size=10)
            column.heading("UNRESOLVED AUTOMATIC PROPOSALS - NOT THE ITEM SCHEDULE")
            column.text("Frozen discovery candidates only. Geometry, identity and quantities are not accepted; click a row for source evidence.", color=AMBER)
        if column.y + height > 810:
            raise ValueError("unresolved proposal exceeds an appendix page")
        top = column.y
        for text, style in blocks:
            column.text(text, **style)
        rect = fitz.Rect(column.x, top, column.x + column.width, column.y)
        proposal["schedule_rect_display"] = list(rect)
        proposal["audit_schedule_page_number"] = column.page.number + 1
        first_on_source.setdefault(proposal["page_ref"], (column.page.number, rect.tl))
        source_index = record["pdf_page_number"] - 1
        source_rect = fitz.Rect(record["source_rect_display"])
        if box is not None and not source_rect.contains(box):
            raise ValueError("unresolved proposal geometry outside source page")
        column.page.insert_link({"kind": fitz.LINK_GOTO, "from": rect, "page": source_index,
                                 "to": box.tl if box is not None else fitz.Point(0, 0), "zoom": 1.2})
        proposal["source_mark_state"] = "no_local_geometry_supplied" if box is None else "appendix_only_overlap_preserves_source_legibility"
        if draw_source_frame:
            source_page = pdf[source_index]
            source_page.draw_rect(box, color=AMBER, width=1.5, dashes="[7 5] 0", stroke_opacity=.65)
            # Link only the frame, never a large hitbox over unrelated item tags.
            for border in (fitz.Rect(box.x0, box.y0, box.x1, min(box.y0 + 3, box.y1)),
                           fitz.Rect(box.x0, max(box.y0, box.y1 - 3), box.x1, box.y1),
                           fitz.Rect(box.x0, box.y0, min(box.x0 + 3, box.x1), box.y1),
                           fitz.Rect(max(box.x0, box.x1 - 3), box.y0, box.x1, box.y1)):
                source_page.insert_link({"kind": fitz.LINK_GOTO, "from": border, "page": column.page.number,
                                         "to": rect.tl, "zoom": 1.2})
            protected[proposal["page_ref"]].append(box)
            proposal["source_mark_rect_display"] = list(box)
            proposal["source_mark_state"] = "drawn_nonoverlapping_candidate_frame"
        column.y += 14
    for source_index, page_ref, rect in sidebar_links:
        target_index, point = first_on_source[page_ref]
        pdf[source_index].insert_link({"kind": fitz.LINK_GOTO, "from": rect,
                                      "page": target_index, "to": point, "zoom": 1.2})
    manifest["proposal_appendix_page_numbers"] = appendix_pages


def render(source: Path, output: Path, manifest: dict) -> dict:
    """Keep original PDF content and annotations; append a schedule margin."""
    if source.resolve() == output.resolve():
        raise ValueError("audit output must not overwrite the source PDF")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash != manifest["document"]["source_pdf_sha256"]:
        raise ValueError("source PDF hash differs from frozen evidence")
    manifest = deepcopy(manifest)
    rows = _unique(manifest["item_rows"])
    segments = _unique(manifest["bounded_local_3d_segments"])
    annotations = _unique(manifest["original_annotation_observations"])
    output.parent.mkdir(parents=True, exist_ok=True)
    # Edit a separate in-memory copy. insert_pdf drops some grouped /IRT cloud
    # annotations; opening the complete document retains their original objects.
    with fitz.open(source) as original, fitz.open(source) as pdf:
        if len(original) != len(manifest["pages"]):
            raise ValueError("source page count differs from frozen evidence")
        navigation = []
        proposal_navigation = []
        pending_schedules = []
        for record in manifest["pages"]:
            index = record["pdf_page_number"] - 1
            page = pdf[index]
            # Existing display-space evidence must never silently move.
            if page.rotation or page.cropbox != page.mediabox or page.rect.tl != fitz.Point(0, 0):
                raise ValueError("partial renderer requires uncropped, unrotated source pages")
            source_rect = fitz.Rect(page.rect)
            source_annotation_count = sum(1 for _ in page.annots() or [])
            # Only original native text is authoritative here. Raster-only text
            # is not covered by this presentation collision check.
            native_text_obstacles = [fitz.Rect(span["bbox"]) + (-3, -3, 3, 3)
                for block in original[index].get_text("dict")["blocks"]
                for line in block.get("lines", []) for span in line["spans"]
                if span["text"].strip()]
            sidebar = 800 if source_rect.height > 1200 else 480
            page.set_mediabox(fitz.Rect(0, 0, source_rect.width + sidebar, source_rect.height))
            record["source_rect_display"] = list(source_rect)
            record["source_to_audit_matrix"] = [1, 0, 0, 1, 0, 0]
            record["original_annotation_count"] = source_annotation_count
            record["source_tag_obstacle_basis"] = "original_native_text_spans_only_raster_text_unchecked"
            page_rows = [rows[ref] for ref in record["item_occurrence_refs"]]
            target_obstacles = [fitz.Rect(min(p[0] for p in row["centreline_points_display"]),
                                         min(p[1] for p in row["centreline_points_display"]),
                                         max(p[0] for p in row["centreline_points_display"]),
                                         max(p[1] for p in row["centreline_points_display"])) + (-6, -6, 6, 6)
                                for row in page_rows]
            findings = [finding for finding in manifest["reviewed_unresolved_findings"] if finding["page_ref"] == record["page_ref"]]
            proposals = [proposal for proposal in manifest.get("unresolved_proposals", []) if proposal["page_ref"] == record["page_ref"]]
            item_links, proposal_link, deferred, continuation_link = _render_sidebar(page, record, page_rows, findings, segments, source_rect.width, proposals)
            navigation.extend((index, rect, target) for rect, target in
                              item_links)
            if deferred:
                pending_schedules.append((record, deferred, continuation_link))
            if proposal_link is not None:
                proposal_navigation.append((index, record["page_ref"], proposal_link))
            # Exact centerlines are highlighted without replacing source geometry.
            occupied = []
            for row in sorted(page_rows, key=lambda row: min(p[1] for p in row["centreline_points_display"])):
                points = [fitz.Point(point) for point in row["centreline_points_display"]]
                if any(point not in source_rect for point in points):
                    raise ValueError("target geometry outside source page")
                page.draw_polyline(points, color=BLUE, width=6, stroke_opacity=.55)
                left = min(p.x for p in points)
                top = min(p.y for p in points)
                tag_width = fitz.Font(fontfile=audit_font_file(True)).text_length(row["id"], fontsize=13) + 16
                tag_left = min(left, source_rect.x1 - tag_width - 2)
                tag = None
                for offset in [0, *[sign * step for step in range(33, int(source_rect.height), 33) for sign in (-1, 1)]]:
                    candidate = fitz.Rect(tag_left, max(2, top - 35) + offset,
                                          tag_left + tag_width, max(2, top - 35) + offset + 28)
                    if candidate.y1 > source_rect.y1 - 30:
                        continue
                    if source_rect.contains(candidate) and not any(candidate.intersects(other) for other in [*occupied, *native_text_obstacles, *target_obstacles]):
                        tag = candidate
                        break
                if tag is None:
                    raise ValueError("source occurrence label has no safe lane clear of native text and other labels")
                occupied.append(tag)
                # Alternate endpoint callout lanes for adjacent projections;
                # superposed leaders would make exact identities hard to read.
                anchor = (min if len(occupied) % 2 else max)(points, key=lambda point: point.x)
                leader_origin = fitz.Point(min(max(anchor.x, tag.x0), tag.x1), tag.y1)
                page.draw_line(leader_origin, anchor, color=BLUE, width=1.5)
                page.draw_circle(anchor, 5, color=BLUE, fill=WHITE, width=2)
                page.draw_rect(tag, color=BLUE, fill=WHITE, fill_opacity=.96)
                _Column(page, tag.x0 + 6, tag.width - 12, tag.y0 + 2, 13).text(row["id"], size=13, bold=True, color=BLUE, gap=0)
                row["source_mark_rect_display"] = list(tag)
                row["source_leader_anchor_display"] = [anchor.x, anchor.y]
                row["source_target_bbox_display"] = [min(p.x for p in points), min(p.y for p in points) - 5,
                                                         max(p.x for p in points), max(p.y for p in points) + 5]
            for finding in findings:
                boxes = [annotations[ref]["geometry"]["rect_display"] for ref in finding["annotation_refs"]
                         if ref in annotations and annotations[ref]["page_ref"] == record["page_ref"]]
                finding["source_mark_rects_display"] = boxes
                for box in boxes:
                    page.draw_rect(fitz.Rect(box), color=AMBER, width=2, dashes="[7 5] 0")
                if boxes:
                    box = fitz.Rect(boxes[0])
                    left = min(box.x0, source_rect.width - 640)
                    top = max(2, box.y0 - 34)
                    tag = fitz.Rect(left, top, left + 630, top + 28)
                    while any(tag.intersects(other) for other in [*occupied, *native_text_obstacles]):
                        tag += (0, -33, 0, -33)
                    if tag.y0 < 0:
                        raise ValueError("hint labels need a nonoverlapping layout")
                    occupied.append(tag)
                    page.draw_line(tag.bl, box.tl, color=AMBER, width=1)
                    page.draw_rect(tag, color=AMBER, fill=WHITE, fill_opacity=.96)
                    _Column(page, tag.x0 + 6, tag.width - 12, tag.y0 + 2, 13).text(
                        "HINT | " + finding["id"], size=13, bold=True, color=AMBER, gap=0)
                    finding["source_id_rect_display"] = list(tag)
                    page.insert_link({"kind": fitz.LINK_GOTO, "from": tag, "page": index,
                                      "to": fitz.Point(finding["schedule_rect_display"][:2]), "zoom": 1.2})
                    page.insert_link({"kind": fitz.LINK_GOTO, "from": fitz.Rect(finding["schedule_rect_display"]), "page": index,
                                      "to": box.tl, "zoom": 1.2})
        continuation_links, continuation_pages = _render_schedule_continuations(pdf, pending_schedules, segments)
        navigation.extend(continuation_links)
        manifest["accepted_schedule_continuation_page_numbers"] = continuation_pages
        for row in rows.values():
            source_index = row["source"]["pdf_page_number"] - 1
            schedule_index = row["audit_schedule_page_number"] - 1
            pdf[source_index].insert_link({"kind": fitz.LINK_GOTO, "from": fitz.Rect(row["source_mark_rect_display"]),
                "page": schedule_index, "to": fitz.Point(row["schedule_rect_display"][:2]), "zoom": 1.2})
            pdf[schedule_index].insert_link({"kind": fitz.LINK_GOTO, "from": fitz.Rect(row["schedule_rect_display"]),
                "page": source_index, "to": fitz.Point(row["source_leader_anchor_display"]), "zoom": 1.5})
        for index, rect, target in navigation:
            target_row = rows[target]
            pdf[index].insert_link({"kind": fitz.LINK_GOTO, "from": rect,
                "page": target_row["audit_schedule_page_number"] - 1,
                "to": fitz.Point(target_row["schedule_rect_display"][:2]), "zoom": 1.2})
        _render_proposals(pdf, manifest, proposal_navigation)
        pdf.set_toc([[1, f"p.{row['pdf_page_number']} | {_value(row['drawing_sheet_number'])} | PARTIAL AUDIT", row["pdf_page_number"]]
                     for row in manifest["pages"]] +
                    [[1, "Partial certified occurrence schedule continuation", page] for page in continuation_pages] +
                    [[1, "Unresolved automatic proposals", page] for page in manifest["proposal_appendix_page_numbers"]])
        mode = "automatic target discovery" if manifest["execution_mode"] == "automatic_frozen_replay" else "assisted replay"
        pdf.set_metadata({"title": f"Partial marked M&P audit - {mode}, coverage limited",
                          "subject": "Frozen drawing evidence; unresolved quantities; not for quotation",
                          "creator": "Rebar frozen MEP audit renderer"})
        encoded = (json.dumps(manifest, indent=2, ensure_ascii=True) + "\n").encode()
        pdf.embfile_add("audit-manifest.json", encoded, desc="Frozen IDs, evidence, page coverage and unresolved authority")
        pdf.save(output, garbage=0, deflate=True)
    output.with_suffix(".manifest.json").write_bytes(encoded)
    return manifest


def _human_measurement(value, unit):
    """Display drawing units, recovering fractions only within stored precision."""
    if value is None:
        return "not identified"
    if unit in {"ft", "in"}:
        inches = abs(value) * (12 if unit == "ft" else 1)
        fraction = Fraction(inches).limit_denominator(64)
        if abs(float(fraction) - inches) < 1e-5:
            whole, numerator = divmod(fraction.numerator, fraction.denominator)
            feet, whole = divmod(whole, 12) if unit == "ft" else (None, whole)
            part = f"{whole}" + (f" {numerator}/{fraction.denominator}" if numerator else "")
            if feet is None and whole == 0 and numerator:
                part = f"{numerator}/{fraction.denominator}"
            return ("-" if value < 0 else "") + (f"{feet}' - {part}\"" if feet is not None else f"{part} in")
    return f"{_measurement(value)} {unit}"


def _human_item(row):
    occurrence = row["occurrence"]
    system = (occurrence.get("system") or {}).get("kind")
    title = {
        "heating_hot_water_supply": "Heating-water supply pipe",
        "heating_hot_water_return": "Heating-water return pipe",
        "chilled_water_supply": "Chilled-water supply pipe",
        "chilled_water_return": "Chilled-water return pipe",
    }.get(system, "Route segment - system unknown" if not system else system.replace("_", " ").capitalize())
    dimensions = occurrence["typed_dimensions"]
    elevations = occurrence["elevations"]
    fields = [("Nominal size" if dimension["dimension_type"] == "nominal_size" else dimension["dimension_type"].replace("_", " ").capitalize(),
               _human_measurement(dimension["value"], dimension["unit"])) for dimension in dimensions]
    if not dimensions:
        fields.append(("Size", "not identified"))
    fields.extend((elevation["elevation_type"].capitalize() + " elevation", _human_measurement(elevation["value"], elevation["unit"]))
                  for elevation in elevations)
    if not elevations:
        fields.append(("Elevation", "not identified"))
    for key, name in (("manufacturer", "Manufacturer"), ("model_number", "Model"),
                      ("specification_reference", "Specification"), ("mounting_type", "Mounting")):
        if occurrence.get(key) is not None:
            fields.append((name, str(occurrence[key])))
    for channel, value in row["takeoff_line"]["value_channels"].items():
        if value["value"] is not None:
            fields.append((channel.replace("_", " ").capitalize(), _human_measurement(value["value"], value["unit"])))
    return title, fields


def _item_presentation(manifest):
    """Short labels alias frozen IDs; they never become designer part marks."""
    rows = sorted(manifest["item_rows"], key=lambda row: (row["source"]["pdf_page_number"],
        min(p[1] for p in row["centreline_points_display"]), min(p[0] for p in row["centreline_points_display"]), row["id"]))
    for index, row in enumerate(rows):
        title, fields = _human_item(row)
        row["presentation"] = {"label": f"E{index + 1:02d}", "color_rgb": list(item_palette_color(index)),
                               "title": title, "fields": [list(field) for field in fields], "designer_part_mark": False}
    return rows


def _human_row(column, row):
    display = row["presentation"]
    top = column.y
    column.text(display["label"] + "  " + display["title"], bold=True, color=INK, size=column.size * 1.1, gap=4)
    for name, value in display["fields"]:
        column.text(f"{name}: {value}", gap=2)
    bottom = column.y
    if column.page is not None:
        column.page.draw_line((column.x - 10, top), (column.x - 10, bottom), color=display["color_rgb"], width=4)
        row["schedule_rect_display"] = [column.x - 14, top, column.x + column.width, bottom]
        row["audit_schedule_page_number"] = column.page.number + 1
    column.y += column.size * .65


def _human_review_entries(manifest):
    """Presentation aliases only: never merge observations or promote targets."""
    pages = {page["page_ref"]: page for page in manifest["pages"]}
    entries = []
    reasons = {
        "no_complete_native_dot_leader": "Leader incomplete",
        "no_unique_leader_contact_target": "Target ambiguous",
        "unresolved_competing_envelope_at_contact": "Competing shapes at contact",
        "incomplete_relevant_competitor_search": "Search incomplete",
        "semantic_kind_or_equipment_port_not_certified": "Meaning or port unresolved",
        "unresolved_native_stroke_at_leader_contact": "Unresolved line at contact",
        "Unclassified tag": "Unclassified text",
        "equipment identity and explicit port connectivity are unresolved.": "Identity and port unresolved",
    }
    for proposal in manifest["unresolved_proposals"]:
        record = pages[proposal["page_ref"]]
        if record["discovery_record"]["discovery_state"] == "not_processed":
            raise ValueError("needs-review observation belongs to an unprocessed page")
        geometry = proposal.get("source_geometry", {})
        box, points = geometry.get("bbox_display"), geometry.get("points_display")
        if box is not None:
            kind, point = "text", box[:2]
        elif points:
            kind, point = "geometry", points[0]
        else:
            raise ValueError("needs-review observation requires exact source geometry")
        plain = list(dict.fromkeys(reasons.get(reason.strip(), "Identification remains unresolved")
                                  for reason in proposal["reason"].split(";")))
        entries.append({"proposal_ref": proposal["id"], "page_ref": proposal["page_ref"],
                        "pdf_page_number": record["pdf_page_number"], "kind": kind,
                        "source_point_display": list(point), "source_geometry": deepcopy(geometry),
                        "observed_text": proposal["description"] if kind == "text" else None,
                        "reason": "; ".join(plain), "quantity_eligible": False})
    entries.sort(key=lambda row: (row["pdf_page_number"], row["source_point_display"][1],
                                  row["source_point_display"][0], row["proposal_ref"]))
    counts = {"text": 0, "geometry": 0}
    for entry in entries:
        kind = entry["kind"]
        counts[kind] += 1
        entry["label"] = f"{'U' if kind == 'text' else 'G'}{counts[kind]:03d}"
    for record in manifest["pages"]:
        local = [entry for entry in entries if entry["page_ref"] == record["page_ref"]]
        record["needs_review"] = {"text_observation_count": sum(entry["kind"] == "text" for entry in local),
                                  "geometry_observation_count": sum(entry["kind"] == "geometry" for entry in local),
                                  "audit_page_numbers": [], "counts_are_parts": False}
    return entries


def _human_review_text(column, entry):
    return column.text(f"{entry['label']} | {entry['observed_text']}\nNeeds review: {entry['reason']}",
                       size=10, gap=5)


def _render_human_review_register(pdf, manifest, entries, page_map):
    """Readable text rows and compact shape indexes, separate from E-items."""
    page_numbers, toc = [], []
    for record in manifest["pages"]:
        local = [entry for entry in entries if entry["page_ref"] == record["page_ref"]]
        for entry in local:
            source_rect = fitz.Rect(record["source_rect_display"])
            geometry = entry["source_geometry"]
            if (geometry.get("bbox_display") is not None and not source_rect.contains(fitz.Rect(geometry["bbox_display"]))
                    or any(fitz.Point(point) not in source_rect for point in geometry.get("points_display", []))):
                raise ValueError("needs-review geometry outside source page")
        for kind in ("text", "geometry"):
            remaining = [entry for entry in local if entry["kind"] == kind]
            while remaining:
                page = pdf.new_page(width=1190, height=842)
                page_numbers.append(page.number + 1)
                record["needs_review"]["audit_page_numbers"].append(page.number + 1)
                title = "Observed text" if kind == "text" else "Unlabelled shape index"
                toc.append([1, f"Needs review | source {record['pdf_page_number']} | {title}", page.number + 1])
                header = _Column(page, 30, 1130, y=22, size=10)
                header.text("NEEDS IDENTIFICATION | NOT FOUND ELEMENTS", size=20, bold=True, gap=4)
                back = header.text(f"Source page {record['pdf_page_number']} | {record['drawing_sheet_number'] or 'No sheet number shown'} | {title} | click to return to drawing",
                                   size=11, bold=True, color=BLUE, gap=3)
                header.text("Observation records only, not parts or quantities. Click a U-row or G-label for its exact source location.", color=MUTED, gap=2)
                if kind == "geometry":
                    header.text("All G-labels are unlabelled shapes: identity, system, size and quantity are not established.", color=MUTED, gap=0)
                page.insert_link({"kind": fitz.LINK_GOTO, "from": back,
                                  "page": page_map[record["pdf_page_number"]], "to": fitz.Point(0, 0)})
                start, bottom = max(120, header.y + 8), 765
                displayed = []
                if kind == "text":
                    for x in (30, 610):
                        column = _Column(page, x, 550, y=start, size=10)
                        while remaining:
                            entry = remaining[0]
                            measure = _Column(None, x, 550, y=0, size=10)
                            _human_review_text(measure, entry)
                            if measure.y > bottom - start:
                                raise ValueError("needs-review text row exceeds a complete column")
                            if column.y + measure.y > bottom:
                                break
                            rect = _human_review_text(column, entry)
                            displayed.append((entry, rect))
                            remaining.pop(0)
                else:
                    cell_width = 1130 / 12
                    label_height = max(_Column(None, 0, cell_width - 16, size=10)._layout(entry["label"], 10, True)[1]
                                       for entry in remaining)
                    cell_height = label_height + 6
                    capacity = int((bottom - start) // cell_height) * 12
                    if capacity < 1:
                        raise ValueError("needs-review shape index has no usable rows")
                    for index, entry in enumerate(remaining[:capacity]):
                        x, y = 30 + index % 12 * cell_width, start + index // 12 * cell_height
                        rect = fitz.Rect(x, y, x + cell_width - 5, y + cell_height - 3)
                        page.draw_rect(rect, color=(.79, .83, .86), fill=PAPER, width=.5)
                        _Column(page, x + 8, rect.width - 16, y=y + 1, size=10).text(entry["label"], bold=True, gap=0)
                        displayed.append((entry, rect))
                    remaining = remaining[capacity:]
                for entry, rect in displayed:
                    entry["audit_page_number"] = page.number + 1
                    entry["audit_rect_display"] = list(rect)
                    page.insert_link({"kind": fitz.LINK_GOTO, "from": rect,
                                      "page": page_map[record["pdf_page_number"]],
                                      "to": fitz.Point(entry["source_point_display"]), "zoom": 2.0})
                _Column(page, 30, 1130, y=789, size=9).text(
                    "U/G labels are review aliases, not designer marks. Repeated observations are not deduplicated physical items. No quantity or approval is implied.",
                    color=MUTED, gap=0)
        if local:
            review = record["needs_review"]
            pdf[page_map[record["pdf_page_number"]]].insert_link({"kind": fitz.LINK_GOTO,
                "from": fitz.Rect(review["sidebar_link_rect_display"]), "page": review["audit_page_numbers"][0] - 1,
                "to": fitz.Point(30, 22)})
    manifest["human_review_register"] = {"entries": entries, "audit_page_numbers": page_numbers,
        "source_proposals_sha256": canonical_sha256(manifest["unresolved_proposals"]),
        "counts_are_parts": False, "quantity_eligible": False}
    return toc


def _human_plan_sidebar(page, record, rows, source_width):
    width = page.rect.width - source_width
    page.draw_rect(fitz.Rect(source_width, 0, page.rect.width, page.rect.height), color=None, fill=PAPER)
    size = 25 if page.rect.height > 1200 else 11
    column = _Column(page, source_width + 44, width - 88, y=44, size=size)
    column.text("ELEMENTS FOUND", size=size * 1.9, bold=True)
    column.text(f"Source PDF page {record['pdf_page_number']} | {record['drawing_sheet_number'] or 'No sheet number shown'}", bold=True)
    column.text("PROCESSED - identification is incomplete", color=AMBER)
    column.text("Match each colored label to the same-colored segment on the drawing. Click a row for a close-up.")
    if not rows:
        column.heading("No elements identified yet")
        column.text("The page was processed, but no element could be identified reliably enough to list. This does not mean the drawing is empty.")
    footer = [
        ("WHAT IS STILL UNKNOWN", {"bold": True, "size": size * .9}),
        ("These are marked drawing segments, not separate physical part counts. Lengths, complete quantities and connections have not been established.", {"size": size * .85}),
        ("Unidentified symbols and labels are not included as found parts. Their evidence is retained separately.", {"size": size * .85}),
        ("Review IDs are not designer part marks. Colors identify elements only; they do not mean approval.", {"size": size * .8, "color": MUTED}),
        ("Not approved for quotation.", {"bold": True, "color": AMBER, "size": size * .85}),
    ]
    review = record.get("needs_review", {})
    review_count = review.get("text_observation_count", 0) + review.get("geometry_observation_count", 0)
    if review_count:
        footer.insert(1, (f"Needs identification: {review['text_observation_count']} text observations + "
            f"{review['geometry_observation_count']} unlabelled shapes (not parts). Click for the separate register.",
            {"size": size * .85, "bold": True, "color": BLUE}))
    footer_measure = _Column(None, column.x, column.width, y=0, size=size)
    for text, options in footer:
        footer_measure.text(text, **options)
    footer_top = page.rect.height - 38 - footer_measure.y
    deferred, continuation = [], None
    for index, row in enumerate(rows):
        measure = _Column(None, column.x, column.width, y=0, size=size)
        _human_row(measure, row)
        if column.y + measure.y > footer_top - size * 4:
            deferred = rows[index:]
            continuation = column.text("More elements are listed in the close-up schedule. Click here to continue.", bold=True, color=BLUE)
            break
        _human_row(column, row)
    column.y = max(column.y, footer_top)
    for text, options in footer:
        rect = column.text(text, **options)
        if review_count and text.startswith("Needs identification:"):
            review["sidebar_link_rect_display"] = list(rect)
    return deferred, continuation


def _source_text_boxes(page):
    boxes = [fitz.Rect(span["bbox"]) for block in page.get_text("dict")["blocks"]
             for line in block.get("lines", []) for span in line["spans"] if span["text"].strip()]
    # FreeText appearances may not be returned by native text extraction.
    boxes.extend(fitz.Rect(annot.rect) for annot in page.annots() or []
                 if annot.type[0] in (fitz.PDF_ANNOT_FREE_TEXT, fitz.PDF_ANNOT_TEXT))
    return boxes


def _readable_line_parts(a, b, boxes, padding):
    """Display-only gaps protect source lettering; item geometry stays intact."""
    a, b = fitz.Point(a), fitz.Point(b)
    delta = b - a
    squared = delta.x ** 2 + delta.y ** 2
    if squared < 1e-12:
        return []
    hidden = []
    for box in boxes:
        clipped = _clip_line(a, b, box + (-padding, -padding, padding, padding))
        if clipped:
            hidden.append(tuple(((p.x - a.x) * delta.x + (p.y - a.y) * delta.y) / squared for p in clipped))
    parts, cursor = [], 0.0
    for start, end in sorted(hidden):
        if start > cursor:
            parts.append((a + delta * cursor, a + delta * start))
        cursor = max(cursor, end)
    if cursor < 1:
        parts.append((a + delta * cursor, b))
    return parts


def _commit_text_safe_shape(page, shape, color, width):
    """Multiply preserves dark lettering even when the PDF stores it as paths."""
    # A local resource dictionary must not shadow inherited fonts or images.
    ancestor = page.xref
    ancestors = {ancestor}
    kind, resources = page.parent.xref_get_key(ancestor, "Resources")
    while kind == "null":
        parent_kind, parent_ref = page.parent.xref_get_key(ancestor, "Parent")
        if parent_kind != "xref":
            break
        ancestor = int(parent_ref.split()[0])
        if ancestor in ancestors:
            raise ValueError("cyclic PDF resource ancestry")
        ancestors.add(ancestor)
        kind, resources = page.parent.xref_get_key(ancestor, "Resources")
    if ancestor != page.xref and kind != "null":
        page.parent.xref_set_key(page.xref, "Resources", resources)
    owner, key = page.xref, "Resources"
    for child in ("ExtGState", "MepTextSafe"):
        kind, value = page.parent.xref_get_key(owner, key)
        if kind == "xref":
            owner, key = int(value.split()[0]), child
        else:
            key += "/" + child
    definition = "<< /Type /ExtGState /BM /Multiply >>"
    suffix = ""
    while True:
        kind, value = page.parent.xref_get_key(owner, key + suffix)
        if kind == "null":
            page.parent.xref_set_key(owner, key + suffix, definition)
            break
        if kind == "dict" and "".join(value.split()) == "".join(definition.split()):
            break
        suffix = str(int(suffix or "0") + 1)
    state_name = key.rsplit("/", 1)[-1] + suffix
    shape.finish(color=color, width=width, closePath=False)
    shape.totalcont = f"q /{state_name} gs\n" + shape.totalcont + "Q\n"
    shape.commit()


def _readable_highlight(page, points, boxes, color, width, halo_width=0):
    shape = page.new_shape()
    for a, b in zip(points, points[1:]):
        for start, end in _readable_line_parts(a, b, boxes, max(width, halo_width) / 2 + 1):
            shape.draw_line(start, end)
    _commit_text_safe_shape(page, shape, color, width)


def _readable_anchor(page, anchor, boxes, color, radius, width):
    footprint = fitz.Rect(anchor.x - radius, anchor.y - radius, anchor.x + radius, anchor.y + radius)
    if not any(footprint.intersects(box + (-width, -width, width, width)) for box in boxes):
        shape = page.new_shape()
        shape.draw_circle(anchor, radius)
        _commit_text_safe_shape(page, shape, color, width)


def _human_mark(page, row, source_rect, obstacles, occupied, text_boxes):
    points = [fitz.Point(point) for point in row["centreline_points_display"]]
    if any(point not in source_rect for point in points):
        raise ValueError("target geometry outside source page")
    display = row["presentation"]
    color, label = display["color_rgb"], display["label"]
    anchor = min(points, key=lambda point: (point.y, point.x))
    size = 26 if source_rect.height > 1200 else 12
    width = fitz.Font(fontfile=audit_font_file(True)).text_length(label, fontsize=size) + size * .8
    height = size * 1.65
    candidates = [fitz.Rect(anchor.x + size * .6 + dx * size * 2,
                            anchor.y - height - size * .5 + dy * size * 2,
                            anchor.x + size * .6 + dx * size * 2 + width,
                            anchor.y - size * .5 + dy * size * 2)
                  for dx in range(-6, 7) for dy in range(-8, 9)]
    candidates.sort(key=lambda rect: math.dist((rect.x0 + rect.width / 2, rect.y0 + rect.height / 2), anchor))
    tag = next((rect for rect in candidates if source_rect.contains(rect)
                and not any(rect.intersects(other) for other in [*obstacles, *occupied])), None)
    if tag is None:
        raise ValueError("no legible local lane for the element review label")
    _readable_highlight(page, points, text_boxes, color, size * .3, size * .5)
    origin = fitz.Point(min(max(anchor.x, tag.x0), tag.x1), min(max(anchor.y, tag.y0), tag.y1))
    _readable_highlight(page, [origin, anchor], text_boxes, color, 2)
    _readable_anchor(page, anchor, text_boxes, color, size * .18, 2)
    page.draw_rect(tag, color=color, fill=WHITE, width=2)
    _Column(page, tag.x0 + size * .25, tag.width - size * .5, tag.y0 + size * .1, size).text(label, bold=True, color=INK, gap=0)
    occupied.append(tag)
    row["source_mark_rect_display"] = list(tag)
    row["source_leader_anchor_display"] = list(anchor)


def _clip_line(a, b, rect):
    """Clip a display-only highlight, never alter the source segment."""
    low, high = 0.0, 1.0
    delta = [b[i] - a[i] for i in (0, 1)]
    for coordinate, minimum, maximum in ((0, rect.x0, rect.x1), (1, rect.y0, rect.y1)):
        if abs(delta[coordinate]) < 1e-12:
            if not minimum <= a[coordinate] <= maximum:
                return None
        else:
            start, end = sorted(((minimum - a[coordinate]) / delta[coordinate], (maximum - a[coordinate]) / delta[coordinate]))
            low, high = max(low, start), min(high, end)
            if low > high:
                return None
    return [fitz.Point(*(a[i] + t * delta[i] for i in (0, 1))) for t in (low, high)]


def _detail_window(row, source_rect):
    """Frame accepted labels and their own contact; never choose nearby text."""
    contexts = row.get("source_label_context", [])
    alternatives = []
    for seed in contexts:
        if not seed["contact_points_display"]:
            continue
        anchor = fitz.Point(seed["contact_points_display"][0])
        nearby = [entry for entry in contexts if any(math.dist(anchor, point) <= 180 for point in entry["contact_points_display"])]
        box = fitz.Rect(anchor.x, anchor.y, anchor.x, anchor.y)
        refs = set()
        for entry in nearby:
            box |= fitz.Rect(entry["bbox_display"])
            for point in entry["contact_points_display"]:
                box.include_point(point)
            refs.update(entry["relation_refs"])
        alternatives.append((-len(refs), box.width * box.height, seed["observation_ref"], box, anchor, nearby))
    if alternatives:
        _, _, _, box, anchor, contexts = min(alternatives, key=lambda item: item[:3])
        box += (-30, -30, 30, 30)
        width, height = max(240, box.width), max(200, box.height)
        center = (box.tl + box.br) / 2
        clip = fitz.Rect(center.x - width / 2, center.y - height / 2, center.x + width / 2, center.y + height / 2) & source_rect
        return clip, anchor, [entry["observation_ref"] for entry in contexts]
    anchor = fitz.Point(row["source_leader_anchor_display"])
    return fitz.Rect(anchor.x - 145, anchor.y - 120, anchor.x + 175, anchor.y + 120) & source_rect, anchor, []


def _human_closeups(pdf, original, rows, page_map, *, raster=False,
                    page_cache_dir=None, source_page_hashes=None, cache_stats=None):
    """Four readable item cards per sheet; source plans retain full extents."""
    detail_pages = []
    protected = {number: _source_text_boxes(original[number - 1])
                 for number in {row["source"]["pdf_page_number"] for row in rows}}
    # show_pdf_page omits annotation objects. Bake this in-memory source copy
    # so vector excerpts preserve the same visible markups as raster excerpts.
    # The source file and the vector edition's original annotations are untouched.
    original.bake(annots=True, widgets=False)
    for offset in range(0, len(rows), 4):
        page = pdf.new_page(width=1190, height=842)
        detail_pages.append(page.number + 1)
        header = _Column(page, 30, 1130, y=22, size=12)
        header.text("FOUND ELEMENTS | CLOSE-UP REVIEW", size=21, bold=True)
        header.text("Only the colored segment is the listed element. E-labels match the full plans; nearby source text may describe other elements.", size=10, color=MUTED)
        for local, row in enumerate(rows[offset:offset + 4]):
            x, y = 30 + (local % 2) * 580, 103 + (local // 2) * 345
            card = fitz.Rect(x, y, x + 550, y + 325)
            display = row["presentation"]
            page.draw_rect(card, color=(.79, .83, .86), fill=WHITE)
            page.draw_rect(fitz.Rect(x, y, x + 550, y + 5), color=None, fill=display["color_rgb"])
            column = _Column(page, x + 14, 522, y=y + 14, size=12)
            column.text(display["label"] + "  " + display["title"], bold=True, color=INK, size=15, gap=3)
            column.text(f"Source page {row['source']['pdf_page_number']} | {row['source']['drawing_sheet_number']}", size=10, color=MUTED, gap=0)
            image_rect = fitz.Rect(x + 14, y + 72, x + 322, y + 279)
            number = row["source"]["pdf_page_number"]
            source_rect = original[number - 1].rect
            clip, anchor, context_refs = _detail_window(row, source_rect)
            scale = min(image_rect.width / clip.width, image_rect.height / clip.height)
            destination = fitz.Rect(image_rect.x0 + (image_rect.width - clip.width * scale) / 2,
                                    image_rect.y0 + (image_rect.height - clip.height * scale) / 2,
                                    image_rect.x0 + (image_rect.width + clip.width * scale) / 2,
                                    image_rect.y0 + (image_rect.height + clip.height * scale) / 2)
            if raster:
                body, _, reused = _cached_detail_raster(
                    original[number - 1], source_page_sha256=source_page_hashes[number],
                    clip=clip, cache_dir=page_cache_dir)
                page.insert_image(destination, stream=body)
                if cache_stats is not None:
                    cache_stats["reused_detail_count" if reused
                                else "rendered_detail_count"] += 1
            else:
                page.show_pdf_page(destination, original, number - 1, clip=clip)
            transform = clip.torect(destination)
            text_boxes = [box * transform for box in protected[number] if box.intersects(clip)]
            for a, b in zip(row["centreline_points_display"], row["centreline_points_display"][1:]):
                clipped = _clip_line(a, b, clip)
                if clipped:
                    projected = [point * transform for point in clipped]
                    _readable_highlight(page, projected, text_boxes, display["color_rgb"], 3.5, 6)
            _readable_anchor(page, anchor * transform, text_boxes, display["color_rgb"], 4, 1.5)
            page.draw_rect(image_rect, color=(.79, .83, .86), width=.6)
            fields = _Column(page, x + 338, 194, y=y + 73, size=12)
            for name, value in display["fields"]:
                fields.text(name.upper(), size=8, color=MUTED, gap=0)
                fields.text(value, bold=True, size=12, gap=3)
            fields.text("Length: not established", size=10, color=AMBER)
            if fields.y > y + 287:
                raise ValueError("element close-up attributes exceed card")
            label = _Column(page, x + 14, 522, y=y + 294, size=10)
            label.text("Drawing excerpt | click to see the full highlighted segment", color=BLUE, gap=0)
            row["detail_card_rect_display"] = list(card)
            row["detail_crop_source_rect_display"] = list(clip)
            row["detail_context_observation_refs"] = context_refs
            row["detail_contact_point_display"] = list(anchor)
            row["audit_detail_page_number"] = page.number + 1
            page.insert_link({"kind": fitz.LINK_GOTO, "from": card, "page": page_map[number], "to": anchor, "zoom": 1.4})
        _Column(page, 30, 1130, y=789, size=9).text("Drawing observations only. Separate physical counts, installed lengths and quote approval are not established.", color=MUTED, gap=0)
    return detail_pages


def _overlay_colour(row, source_colours):
    """Copy unanimous native paint for display only; never infer a system."""
    from src.drawing_engine.audit.mep_audit_palette import SYSTEM_COLORS
    refs = row["source_primitive_refs"]
    colours = [source_colours.get((row["source"]["page_ref"], ref), set()) for ref in refs]
    if colours and all(len(values) == 1 for values in colours):
        unique = set.union(*colours)
        if len(unique) == 1:
            return list(next(iter(unique))), "native_source_strokes"
    system = (row["occurrence"].get("system") or {}).get("kind")
    if system in SYSTEM_COLORS and system != "unknown_or_conflicted":
        return list(SYSTEM_COLORS[system]), "accepted_system_palette"
    return list(SYSTEM_COLORS["unknown_or_conflicted"]), "unresolved_source_colour"


def render_overlay(source: Path, output: Path, manifest: dict, *, page_cache_dir,
                   source_opacity=.47, raster_dpi=120, route_graph=None) -> dict:
    """One static drawing per processed page, with opaque frozen route geometry."""
    if source.resolve() == output.resolve() or output.exists():
        raise ValueError("overlay output must be a new PDF")
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["document"]["source_pdf_sha256"]:
        raise ValueError("source PDF hash differs from frozen evidence")
    if manifest["execution_mode"] != "automatic_frozen_replay":
        raise ValueError("overlay requires explicit automatic page coverage")
    if not 0 < source_opacity <= 1 or not 96 <= raster_dpi <= 300:
        raise ValueError("invalid source opacity or raster DPI")
    manifest = deepcopy(manifest)
    rows = _item_presentation(manifest)
    source_colours = {}
    if route_graph is not None:
        if route_graph["document"] != manifest["document"]:
            raise ValueError("route colours belong to a different source document")
        manifest["input_payload_sha256"]["route_observations"] = canonical_sha256(route_graph)
        for scope in route_graph["pages"]:
            for fragment in scope["fragments"]:
                colour = fragment.get("style", {}).get("stroke")
                if (fragment.get("source_kind") == "native_pdf_vector"
                        and fragment["page_ref"] == scope["page_ref"]
                        and isinstance(colour, (list, tuple)) and len(colour) == 3
                        and all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in colour)):
                    key = (fragment["page_ref"], fragment["source_primitive_ref"])
                    source_colours.setdefault(key, set()).add(tuple(colour))
    for row in rows:
        colour, basis = _overlay_colour(row, source_colours)
        row["presentation"].update(color_rgb=colour, colour_basis=basis,
            colour_source_primitive_refs=row["source_primitive_refs"] if basis == "native_source_strokes" else [],
            colour_establishes_identity=False)
    records = [r for r in manifest["pages"] if r["discovery_record"]["discovery_state"] != "not_processed"]
    if not records:
        raise ValueError("no processed source pages")
    output.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(source) as original, fitz.open() as pdf:
        if len(original) != manifest["document"]["page_count"]:
            raise ValueError("source page count differs from frozen evidence")
        for record in records:
            number = record["pdf_page_number"]
            source_page = original[number - 1]
            page = pdf.new_page(width=source_page.rect.width, height=source_page.rect.height)
            body, _, _ = _cached_source_raster(source_page, page_number=number,
                cache_dir=Path(page_cache_dir), raster_dpi=raster_dpi)
            page.insert_image(page.rect, stream=body)
            page.draw_rect(page.rect, color=None, fill=WHITE, fill_opacity=1-source_opacity)
            record["audit_source_page_number"] = page.number + 1
            record["source_to_audit_matrix"] = [1, 0, 0, 1, 0, 0]
            for row in rows:
                if row["source"]["pdf_page_number"] != number:
                    continue
                points = [fitz.Point(p) for p in row["centreline_points_display"]]
                if len(points) < 2 or any(p not in page.rect for p in points):
                    raise ValueError("target geometry outside source page")
                identified = bool((row["occurrence"].get("system") or {}).get("kind"))
                page.draw_polyline(points, color=row["presentation"]["color_rgb"],
                    width=max(2, min(8, page.rect.height * .003)), stroke_opacity=1,
                    dashes=None if identified else "[8 5] 0", closePath=False)
        manifest["presentation"] = {"edition": "static_overlay", "source_rendering": "raster",
            "source_opacity": source_opacity, "overlay_opacity": 1,
            "processed_source_page_numbers": [r["pdf_page_number"] for r in records],
            "identified_style": "solid", "unknown_system_style": "dashed",
            "colours": "native_source_strokes_with_system_or_neutral_fallback",
            "interactive_links": False, "source_pdf_preserved_unmodified": True}
        pdf.set_metadata({"title": "Drawing audit overlay", "creator": "Rebar audit renderer"})
        pdf.embfile_add("audit-manifest.json", json.dumps(manifest, ensure_ascii=True).encode(),
                        desc="Frozen source IDs and interpretation evidence")
        pdf.save(output, garbage=4, deflate=True)
    return manifest


def render_items(source: Path, output: Path, manifest: dict, *, source_rendering="vector",
                 raster_dpi=120, page_cache_dir=None, write_manifest=True) -> dict:
    """Readable review edition; frozen evidence stays embedded, not in 200 pages of jargon."""
    if source.resolve() == output.resolve():
        raise ValueError("audit output must not overwrite the source PDF")
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["document"]["source_pdf_sha256"]:
        raise ValueError("source PDF hash differs from frozen evidence")
    if manifest["execution_mode"] != "automatic_frozen_replay":
        raise ValueError("item review edition requires explicit automatic page coverage")
    if source_rendering not in {"vector", "raster"} or not 96 <= raster_dpi <= 300:
        raise ValueError("invalid source rendering mode or raster DPI")
    raster = source_rendering == "raster"
    page_cache_dir = (Path(page_cache_dir) if page_cache_dir is not None
                      else output.with_suffix(".page-cache"))
    # The full search packet is already frozen in the extraction run. Do not
    # duplicate millions of regional primitive IDs inside a reading edition.
    # Preserve its hash, coverage totals and every item/proposal evidence ID.
    manifest = deepcopy({**manifest, "pages": [
        {**record, "discovery_record": _review_discovery_record(record.get("discovery_record"))}
        for record in manifest["pages"]]})
    rows = _item_presentation(manifest)
    review_entries = _human_review_entries(manifest)
    records = [record for record in manifest["pages"] if record["discovery_record"]["discovery_state"] != "not_processed"]
    if not records:
        raise ValueError("no processed source pages")
    output.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(source) as original, (fitz.open() if raster else fitz.open(source)) as pdf:
        if len(original) != len(manifest["pages"]):
            raise ValueError("source page count differs from frozen evidence")
        if raster:
            source_page_hashes = {}
            for record in records:
                original_page = original[record["pdf_page_number"] - 1]
                page = pdf.new_page(width=original_page.rect.width, height=original_page.rect.height)
                body, receipt, reused = _cached_source_raster(
                    original_page, page_number=record["pdf_page_number"],
                    cache_dir=page_cache_dir, raster_dpi=raster_dpi)
                page.insert_image(page.rect, stream=body)
                record["source_raster"] = {"dpi": raster_dpi,
                                            "width_pixels": receipt["width_pixels"],
                                            "height_pixels": receipt["height_pixels"],
                                            "jpeg_quality": 85, "original_annotations_rendered": True,
                                            "editable_original_annotations_retained": False,
                                            "page_cache_key": receipt["cache_key"],
                                            "source_page_sha256": receipt["source_page_sha256"],
                                            "page_cache_reused": reused}
                source_page_hashes[record["pdf_page_number"]] = receipt["source_page_sha256"]
        else:
            pdf.select([record["pdf_page_number"] - 1 for record in records])
        pdf.new_page(pno=0, width=1190, height=842)
        page_map = {record["pdf_page_number"]: index + 1 for index, record in enumerate(records)}
        for record in manifest["pages"]:
            record["audit_source_page_number"] = page_map.get(record["pdf_page_number"], -1) + 1 or None
        pending = []
        for record in records:
            number = record["pdf_page_number"]
            page = pdf[page_map[number]]
            if page.rotation or page.cropbox != page.mediabox or page.rect.tl != fitz.Point(0, 0):
                raise ValueError("item renderer requires uncropped, unrotated source pages")
            source_rect = fitz.Rect(page.rect)
            record["source_rect_display"] = list(source_rect)
            record["source_to_audit_matrix"] = [1, 0, 0, 1, 0, 0]
            record["original_annotation_count"] = sum(1 for _ in original[number - 1].annots() or [])
            page.set_mediabox(fitz.Rect(0, 0, source_rect.width + (1150 if source_rect.height > 1200 else 520), source_rect.height))
            page_rows = [row for row in rows if row["source"]["pdf_page_number"] == number]
            deferred, continuation = _human_plan_sidebar(page, record, page_rows, source_rect.width)
            if deferred:
                pending.append((page.number, deferred, continuation))
            text_boxes = _source_text_boxes(original[number - 1])
            obstacles = [box + (-3, -3, 3, 3) for box in text_boxes]
            obstacles.extend(fitz.Rect(min(p[0] for p in row["centreline_points_display"]), min(p[1] for p in row["centreline_points_display"]),
                                        max(p[0] for p in row["centreline_points_display"]), max(p[1] for p in row["centreline_points_display"])) + (-7, -7, 7, 7)
                             for row in page_rows)
            occupied = []
            for row in page_rows:
                _human_mark(page, row, source_rect, obstacles, occupied, text_boxes)
        detail_cache_stats = {"reused_detail_count": 0, "rendered_detail_count": 0}
        detail_pages = _human_closeups(
            pdf, original, rows, page_map, raster=raster,
            page_cache_dir=page_cache_dir if raster else None,
            source_page_hashes=source_page_hashes if raster else None,
            cache_stats=detail_cache_stats)
        for source_index, deferred, rect in pending:
            for row in deferred:
                row["schedule_rect_display"] = row["detail_card_rect_display"]
                row["audit_schedule_page_number"] = row["audit_detail_page_number"]
            pdf[source_index].insert_link({"kind": fitz.LINK_GOTO, "from": rect,
                "page": deferred[0]["audit_detail_page_number"] - 1, "to": fitz.Point(deferred[0]["detail_card_rect_display"][:2])})
        for row in rows:
            page = pdf[page_map[row["source"]["pdf_page_number"]]]
            rects = [row["source_mark_rect_display"]]
            if row["audit_schedule_page_number"] == page.number + 1:
                rects.append(row["schedule_rect_display"])
            for rect in rects:
                page.insert_link({"kind": fitz.LINK_GOTO, "from": fitz.Rect(rect), "page": row["audit_detail_page_number"] - 1,
                                  "to": fitz.Point(row["detail_card_rect_display"][:2]), "zoom": 1.0})
        review_toc = _render_human_review_register(pdf, manifest, review_entries, page_map)
        cover = pdf[0]
        column = _Column(cover, 44, 1102, y=35, size=13)
        column.text("DRAWING ELEMENT REVIEW", size=30, bold=True)
        column.text(f"{len(records)} of {len(manifest['pages'])} source pages processed | {len(rows)} marked drawing segments", size=17, bold=True, color=BLUE)
        column.text("Each element has one color and a short E-label, repeated on the drawing, in its schedule and in the close-ups. "
                    "Only the listed attributes are established. Review IDs are not designer part marks.")
        column.text("PAGE COVERAGE | click a row to open the marked drawing", size=13, bold=True)
        for record in manifest["pages"]:
            found = [row for row in rows if row["source"]["pdf_page_number"] == record["pdf_page_number"]]
            processed = record in records
            text = f"PDF {record['pdf_page_number']:02d}   |   {record['drawing_sheet_number'] or 'No sheet number shown'}   |   "
            text += (f"Processed - {len(found)} marked segments" if found else "Processed - no elements identified yet") if processed else "Not processed"
            rect = column.text(text, size=12, gap=1, color=INK if processed else MUTED)
            if processed:
                cover.insert_link({"kind": fitz.LINK_GOTO, "from": rect, "page": page_map[record["pdf_page_number"]], "to": fitz.Point(0, 0)})
        column.text("HOW TO READ THE RESULTS", size=13, bold=True, gap=2)
        column.text("System, nominal size and bottom elevation are shown where found. A segment with an unknown system stays labelled as such. "
                    "Unidentified shapes and unbound labels are listed separately in Needs identification, not counted as parts. Page processing does not mean complete identification.", size=11, gap=3)
        column.text("Lengths and physical part counts are not established. Calculated quantities, declared quantities, engineer review and quote approval "
                    "remain separate; none is supplied by this presentation. Item data and evidence IDs are embedded; full search evidence remains in the extraction run.", size=11, color=MUTED, gap=2)
        column.text((f"Source backgrounds: {raster_dpi} dpi raster images, including visible original markups. " if raster else "Source backgrounds: original PDF vectors. ") +
                    "Colored E-labels and links are added for this review. Original markups are not new verified findings. The original PDF is unchanged.", size=10, color=MUTED, gap=0)
        manifest["presentation"] = {"edition": "human_item_review", "processed_source_page_numbers": sorted(page_map),
                                    "detail_page_numbers": detail_pages, "palette": "shared_rebar_detail_palette",
                                    "source_rendering": source_rendering, "source_pdf_preserved_unmodified": True,
                                    "highlights_avoid_source_text_and_note_boxes": True,
                                    "unresolved_proposals_display": "separate_human_review_register_and_complete_embedded_evidence",
                                    "needs_review_page_numbers": manifest["human_review_register"]["audit_page_numbers"],
                                    "labels_are_designer_marks": False}
        if raster:
            manifest["presentation"]["source_page_cache"] = {
                "version": AUDIT_PAGE_CACHE_VERSION,
                "cache_directory": str(page_cache_dir),
                "reused_page_count": sum(bool(record.get("source_raster", {}).get(
                    "page_cache_reused")) for record in records),
                "rendered_page_count": sum(not bool(record.get("source_raster", {}).get(
                    "page_cache_reused")) for record in records),
                **detail_cache_stats,
            }
        pdf.set_toc([[1, "Page coverage and how to read", 1]] +
                    [[1, f"Source {record['pdf_page_number']} | {record['drawing_sheet_number'] or 'No sheet number shown'} | elements found", page_map[record["pdf_page_number"]] + 1] for record in records] +
                    [[1, f"Element close-ups {index + 1}", number] for index, number in enumerate(detail_pages)] + review_toc)
        pdf.set_metadata({"title": "Drawing element review - marked parts and known attributes", "subject": "Processed-page coverage; matching item colors; no quotation authority", "creator": "Rebar item audit renderer"})
        encoded = (json.dumps(manifest, indent=2, ensure_ascii=True) + "\n").encode()
        pdf.embfile_add("audit-manifest.json", encoded, desc="Exact source IDs, attributes, page coverage and unresolved proposals")
        pdf.subset_fonts()
        pdf.save(output, garbage=4, deflate=True)
    if write_manifest:
        output.with_suffix(".manifest.json").write_bytes(encoded)
    return manifest


def render_compact_items(source, output, manifest, *, raster_dpi=120,
                         page_cache_dir=None):
    """Compare like-for-like editions; keep the smaller final file, not both."""
    if source.resolve() == output.resolve():
        raise ValueError("audit output must not overwrite the source PDF")
    temporary_root = ROOT / "tmp/pdfs"
    temporary_root.mkdir(parents=True, exist_ok=True)
    page_cache_dir = (Path(page_cache_dir) if page_cache_dir is not None
                      else output.with_suffix(".page-cache"))
    page_cache_dir.mkdir(parents=True, exist_ok=True)
    policy_path = page_cache_dir / "source-rendering-policy.json"
    policy_key = hashlib.sha256(json.dumps({
        "version": AUDIT_PAGE_CACHE_VERSION,
        "source_pdf_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "raster_dpi": raster_dpi,
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    policy = None
    if policy_path.exists():
        try:
            candidate = json.loads(policy_path.read_text())
            if candidate.get("policy_key") == policy_key and candidate.get("selected") in {"vector", "raster"}:
                policy = candidate
        except (OSError, ValueError):
            pass
    if policy is None:
        previous_comparison = output.with_suffix(".size-comparison.json")
        previous_manifest = output.with_suffix(".manifest.json")
        if output.exists() and previous_comparison.exists() and previous_manifest.exists():
            try:
                comparison = json.loads(previous_comparison.read_text())
                rendered = json.loads(previous_manifest.read_text())
                selected = comparison.get("selected")
                if (comparison.get("raster_dpi") == raster_dpi
                        and selected in {"vector", "raster"}
                        and rendered.get("document", {}).get("source_pdf_sha256")
                        == hashlib.sha256(source.read_bytes()).hexdigest()
                        and rendered.get("presentation", {}).get("source_rendering") == selected):
                    policy = {"policy_key": policy_key, "selected": selected,
                              "bytes": comparison["bytes"],
                              "migrated_from_existing_output": True}
            except (KeyError, OSError, ValueError):
                pass
    selected_mode = policy.get("selected") if policy else None
    render_key = (hashlib.sha256(json.dumps({
        "version": AUDIT_PAGE_CACHE_VERSION,
        "source_pdf_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "input_manifest_sha256": canonical_sha256(manifest),
        "selected_rendering": selected_mode,
        "raster_dpi": raster_dpi,
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                  if selected_mode else None)
    render_receipt_path = output.with_suffix(".render-cache.json")
    if render_key and output.exists() and output.with_suffix(".manifest.json").exists() \
            and render_receipt_path.exists():
        try:
            receipt = json.loads(render_receipt_path.read_text())
            if (receipt.get("render_key") == render_key
                    and receipt.get("pdf_sha256") == hashlib.sha256(output.read_bytes()).hexdigest()
                    and receipt.get("manifest_sha256") == hashlib.sha256(
                        output.with_suffix(".manifest.json").read_bytes()).hexdigest()):
                comparison = dict(json.loads(output.with_suffix(
                    ".size-comparison.json").read_text()))
                comparison.update({"selection_reused": True,
                                   "complete_output_reused": True,
                                   "source_page_cache_directory": str(page_cache_dir)})
                print(json.dumps(comparison), flush=True)
                return comparison
        except (OSError, ValueError):
            pass
    with tempfile.TemporaryDirectory(prefix="mep-size-comparison-", dir=temporary_root) as directory:
        paths = {mode: Path(directory) / (mode + ".pdf") for mode in ("vector", "raster")}
        modes = [policy["selected"]] if policy else ["vector", "raster"]
        for mode in modes:
            render_items(source, paths[mode], manifest, source_rendering=mode,
                         raster_dpi=raster_dpi, page_cache_dir=page_cache_dir)
        if policy:
            sizes = dict(policy["bytes"])
            sizes[policy["selected"]] = paths[policy["selected"]].stat().st_size
            chosen = policy["selected"]
        else:
            sizes = {mode: path.stat().st_size for mode, path in paths.items()}
            chosen = min(sizes, key=sizes.get)
        comparison = {"bytes": sizes, "selected": chosen, "raster_dpi": raster_dpi,
                      "selection_reused": policy is not None,
                      "source_page_cache_directory": str(page_cache_dir)}
        # Keep the exact rendered winning PDF and matching embedded manifest.
        output.parent.mkdir(parents=True, exist_ok=True)
        paths[chosen].replace(output)
        paths[chosen].with_suffix(".manifest.json").replace(output.with_suffix(".manifest.json"))
        output.with_suffix(".size-comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        if not policy_path.exists():
            _atomic_bytes(policy_path, (json.dumps({
                "schema_version": "mep_audit_rendering_policy.0.1.0",
                "policy_key": policy_key, "selected": chosen, "bytes": sizes,
                "raster_dpi": raster_dpi,
                "migrated_from_existing_output": bool(policy and policy.get(
                    "migrated_from_existing_output")),
            }, indent=2, sort_keys=True) + "\n").encode())
        render_key = hashlib.sha256(json.dumps({
            "version": AUDIT_PAGE_CACHE_VERSION,
            "source_pdf_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "input_manifest_sha256": canonical_sha256(manifest),
            "selected_rendering": chosen,
            "raster_dpi": raster_dpi,
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        _atomic_bytes(render_receipt_path, (json.dumps({
            "schema_version": "mep_audit_render_cache.0.1.0",
            "render_key": render_key,
            "pdf_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(
                output.with_suffix(".manifest.json").read_bytes()).hexdigest(),
        }, indent=2, sort_keys=True) + "\n").encode())
        print(json.dumps(comparison), flush=True)
    return comparison


def _snapshot_comparison_pages(pdf, review, selected):
    """Paginate exact selected-record deltas; full channel deltas stay in manifest."""
    comparison = review.get("comparison")
    if not comparison:
        return []
    pages = []
    column = None
    def write(text, *, heading=False, color=INK):
        nonlocal column
        size = 11 if heading else 8
        # Long JSON pointers and native IDs may not contain spaces.
        text = " ".join(" ".join(word[start:start+105] for start in range(0,len(word),105)) for word in str(text).split(" "))
        probe = _Column(None,32,778,size=size)
        lines,_ = probe._layout(text,size,heading)
        for line in lines:
            if column is None or column.y+size*2+8 > 545:
                page = pdf.new_page(width=842,height=595)
                pages.append(page.number+1)
                column = _Column(page,32,778,y=28,size=9)
                column.heading("Frozen snapshot evidence comparison")
                column.text(f"Baseline {comparison['baseline_snapshot_id']}\nCurrent {comparison['current_snapshot_id']}",size=8)
                page.insert_text((32,570),"Evidence differences do not establish physical identity, continuity, quantities or approval.",fontsize=8,color=MUTED)
            column.text(line,size=size,bold=heading,color=color,gap=3)
    write("Record changes, not an improvement score",heading=True)
    write("Items, attributes, connections and bounded outcomes are counted separately. Same source PDF: " + str(comparison["same_source_pdf"]).lower() + ".")
    for channel in ("items","attributes","connections","outcomes"):
        counts = comparison[channel]["counts"]
        write(channel+": "+" | ".join(f"{state} {counts[state]}" for state in ("added","removed","changed","unchanged")))
    changed_connections = [entry for entry in comparison["connections"]["records"] if entry["change"]=="changed"]
    if changed_connections:
        retained_abstentions = sum(entry["before"]["content"].get("state")==entry["after"]["content"].get("state")=="abstained" for entry in changed_connections)
        write(f"Of {len(changed_connections)} changed connection records, {retained_abstentions} remain abstained in both snapshots. Changed diagnostic reasons are not accepted connections.",color=AMBER)
    summary = review.get("outcome_summary") or {}
    if "expected_bindings" in summary:
        write(f"Reviewed benchmark: {summary.get('correct_bindings')}/{summary['expected_bindings']} correct bindings; {summary.get('wrong_bindings_in_reviewed_scope')} wrong. Fully identified and markable items: {summary.get('correctly_identified_and_markable_items')}/{summary.get('reviewed_item_denominator')}. Evaluation-only scope; not whole-package recall.",color=AMBER)
    write("The manifest retains every record and exact before/after field value. Below: selected inspection entries and their attribute/connection evidence.")
    baseline_selected = {entry["before_id"] for entry in comparison["items"]["records"] if entry["after_id"] in selected}
    for channel in ("items","attributes","connections"):
        for index,entry in enumerate(comparison[channel]["records"]):
            before,after = entry["before"],entry["after"]
            relevant = (entry["after_id"] in selected if channel=="items" else
                        bool(set((after or {}).get("row_refs",[])).intersection(selected) or set((before or {}).get("row_refs",[])).intersection(baseline_selected)) if channel=="attributes" else
                        bool(set((after or {}).get("content",{}).get("segment_refs",[])).intersection(selected) or set((before or {}).get("content",{}).get("segment_refs",[])).intersection(baseline_selected)))
            if not relevant:
                continue
            write(f"{channel}: {entry['change']} | {entry['after_id'] or entry['before_id']}")
            write(f"Exact manifest evidence: /comparison/{channel}/records/{index}")
            if entry["before_id"] != entry["after_id"]:
                write(f"Before ID: {entry['before_id']} | After ID: {entry['after_id']}")
            sides = (("Both snapshots",after),) if entry["change"]=="unchanged" else (("Before",before),("After",after))
            for side,record in sides:
                if record and channel=="attributes":
                    body = record["content"]
                    candidate = body.get("candidate",{})
                    value = candidate.get("raw_text") or " ".join(str(candidate.get(key,"")) for key in ("kind","value","unit")).strip()
                    write(f"{side}: {body.get('relation_type','attribute').replace('_',' ')} = {value}; {body.get('state','unknown')}. Exact target and native evidence remain in the linked manifest record.")
                if record and channel=="connections":
                    body = record["content"]
                    write(f"{side}: {body.get('relation_type','connection').replace('_',' ')}; {body.get('state','unknown')}; physical continuation: {str(body.get('physical_continuation_established',False)).lower()}.")
            for change in entry["field_changes"]:
                write("Changed field: "+change["pointer"])
    return pages


def render_snapshot_review(source, output, review, *, selected_ids=None):
    """A frozen inspection schedule plus exact marked crops, without discovery.

    An explicit crop selection is review presentation scope, not an extraction
    selector. All snapshot rows remain in the schedule and manifest.
    """
    from src.drawing_engine.project.mep_project_review import source_crop
    if Path(source).resolve() == Path(output).resolve():
        raise ValueError("audit output cannot overwrite source PDF")
    with Path(source).open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != review["source_sha256"]:
            raise ValueError("source PDF differs from selected snapshot")
    selected = set(selected_ids) if selected_ids is not None else {row["id"] for row in review["rows"]}
    if not selected.issubset({row["id"] for row in review["rows"]}):
        raise ValueError("selected audit row is outside the frozen snapshot")
    pdf = fitz.open()
    schedule_links, detail_links, destinations = [], [], {}
    scope = review["processing_scope"]
    scope_text = "Execution PDF pages: " + (", ".join(map(str,scope["execution_page_numbers"])) if scope["execution_page_numbers"] is not None else "not recorded") + f" / {scope['registered_page_count']} registered. "
    scope_text += f"Projected branch records: {scope['accepted_branch_record_count']}; 2D equipment identities: {scope.get('identified_projected_equipment_count',0)}; equipment port bindings: {scope['accepted_equipment_port_binding_count']}. Package inventory incomplete."
    rows_per_page = 20
    for offset in range(0, max(1, len(review["rows"])), rows_per_page):
        page = pdf.new_page(width=842, height=595)
        column = _Column(page, 32, 778, y=26, size=10)
        column.heading("MEP drawing inspection schedule")
        column.text(f"Snapshot {review['snapshot_id'][:16]} | {review['project_id']} / {review['document_id']}", size=9)
        column.text("Projected records only. No physical count, declared value or quote approval is inferred. "
                    "Blue = projected segment; amber = observed HVAC / bounded outcome. Row links open marked source crops.", size=9, gap=12)
        column.text(scope_text,size=9,color=AMBER,gap=10)
        y = column.y
        for row in review["rows"][offset:offset+rows_per_page]:
            page.draw_line((32, y+17), (810, y+17), color=(.85,.89,.85), width=.5)
            text = f"{row['schedule_label']}   {row['label']}   |   {row['state']}   |   PDF p. " + ", ".join(map(str, sorted({mark["page_number"] for mark in row["marks"]})))
            if row["id"] not in selected:
                text += "   |   crop not selected in this edition"
            while fitz.get_text_length(text,fontname="helv",fontsize=9)>766:
                text = text[:-5]+"..."
            page.insert_text((36, y+12), text, fontsize=9, color=BLUE if row["kind"] == "segment" else AMBER)
            destinations[row["id"]] = (page.number, y)
            schedule_links.append((page.number, fitz.Rect(32,y,810,y+17), row["id"]))
            y += 18
        page.insert_text((32,570), f"All {len(review['rows'])} frozen rows retained | {len(selected)} rows selected for marked crops | Engineer review remains a separate overlay", fontsize=8, color=MUTED)
    comparison_pages = _snapshot_comparison_pages(pdf,review,selected)
    audit_rows = []
    for row in review["rows"]:
        if row["id"] not in selected:
            continue
        marks = row["marks"] or [None]
        first_detail = len(pdf)
        for mark_index, mark in enumerate(marks):
            page = pdf.new_page(width=842, height=595)
            column = _Column(page, 32, 778, y=26, size=10)
            column.heading(f"{row['schedule_label']} | {row['label']} | {row['state']}")
            column.text(f"Frozen {review['snapshot_id'][:16]} | {row['artifact']}{row['pointer']}", size=8, gap=7)
            column.text(scope_text,size=8,color=AMBER,gap=5)
            if mark is not None:
                crop = source_crop(source, review, row, mark_index=mark_index)
                clip = fitz.Rect(crop["clip_display"])
                bounds = fitz.Rect(32,max(122,column.y+8),560,466)
                ratio = min(bounds.width/clip.width, bounds.height/clip.height)
                target = fitz.Rect(bounds.x0+(bounds.width-clip.width*ratio)/2, bounds.y0+(bounds.height-clip.height*ratio)/2, 0, 0)
                target.x1, target.y1 = target.x0+clip.width*ratio, target.y0+clip.height*ratio
                page.insert_image(target, stream=crop["png"])
                def mapped(point):
                    return fitz.Point(target.x0+(point[0]-clip.x0)*ratio, target.y0+(point[1]-clip.y0)*ratio)
                for path in mark.get("evidence_paths",[]):
                    path_color = MUTED if path.get("role") in {"complete_native_search_scope","unresolved_body_outline_candidate"} else AMBER
                    points = path.get("points_display",[])
                    if path.get("bbox_display"):
                        box = path["bbox_display"]
                        points = [box[:2],[box[2],box[1]],box[2:],[box[0],box[3]],box[:2]]
                    for start,end in zip(points,points[1:]):
                        clipped = _clip_line(start,end,clip)
                        if clipped:
                            page.draw_line(mapped(clipped[0]),mapped(clipped[1]),color=path_color,width=1.2,stroke_opacity=.65)
                for entry in mark.get("junction_context", []):
                    junction = entry["junction"]
                    color = BLUE if junction["state"] == "accepted" else AMBER
                    for member in entry["member_segments"]:
                        page.draw_polyline([mapped(point) for point in member["points_display"]], color=color, width=1.5, stroke_opacity=.6)
                    for path in junction.get('body_paths', []):
                        page.draw_polyline([mapped(point) for point in path['points_display']], color=color, width=1.2)
                    if len(junction.get("centreline_points_display", [])) >= 2:
                        page.draw_polyline([mapped(point) for point in junction["centreline_points_display"]], color=color, width=2.5,
                                           dashes=None if junction["state"] == "accepted" else "[5 4] 0")
                    if junction.get("point_display"):
                        page.draw_circle(mapped(junction["point_display"]),4,color=color,fill=WHITE,fill_opacity=.7,width=1.5)
                    for point in junction.get("port_points_display",[]):
                        page.draw_circle(mapped(point),3,color=color,fill=WHITE,fill_opacity=.6,width=1)
                if mark.get("points_display"):
                    page.draw_polyline([mapped(point) for point in mark["points_display"]], color=BLUE, width=2, stroke_opacity=.7)
                elif not mark.get("evidence_paths"):
                    box = mark["bbox_display"]
                    page.draw_rect(fitz.Rect(mapped(box[:2]),mapped(box[2:])), color=AMBER, width=1.5)
                page.draw_rect(target, color=(.75,.82,.75), width=.5)
                caption = _Column(page, 32, 528, y=480, size=9)
                caption.text(f"Source PDF page {mark['page_number']} | {mark['role'].replace('_',' ')} | click crop to return to schedule", gap=3)
                caption.text(mark["source_ref"], size=8, gap=3)
                detail_links.append((page.number, target, row["id"]))
            facts = _Column(page, 586, 224, y=max(122,column.y+8), size=10)
            facts.heading("Evidence boundary")
            facts.text("Source geometry is frozen. A projected connection is not physical continuity. Display is not engineer approval.", size=10)
            if row.get('projected_identities'):
                facts.text('Accepted 2D body/tag identity, inferred from native cabinet geometry and a repeated tag convention. Equipment ports, physical instance and count remain unresolved.',size=9,color=BLUE)
            if row['payload'].get('geometry_only'):
                facts.text('Geometry only: system, size and elevation are unknown. No annotation propagates through this projected branch.',size=9,color=AMBER)
            if row["kind"]=="outcome":
                if row.get('current_projected_identity_context'):
                    facts.text('Retained earlier strict-port outcome. A separate current certificate now identifies this body/tag in 2D; its ports remain unresolved.',size=9,color=BLUE)
                facts.text("Bounded diagnostic / evaluation outcome. The displayed state does not accept equipment identity or a physical port.",size=9,color=AMBER)
                for reason in row["payload"].get("reason_codes",[]):
                    facts.text(reason.replace("_"," "),size=8,color=AMBER,gap=4)
                if row["payload"].get("trace_diagnostics"):
                    facts.text("Small amber boxes mark frozen failure/competing trace points only. No line bridges competing points or gaps.",size=8,color=AMBER)
                gaps = sorted({entry["first_collinear_gap_display_points"] for entry in row["payload"].get("port_attachment_observations",[]) if entry.get("first_collinear_gap_display_points") is not None})
                if gaps:
                    facts.text("Frozen forward-native gap: "+", ".join(f"{gap:.6f}" for gap in gaps)+" display points. Gap remains unrepaired.",size=8,color=AMBER)
                roles = sorted({path["role"] for path in (mark or {}).get("evidence_paths",[])})
                facts.text("Source roles: "+", ".join(role.replace("_"," ") for role in roles)+". Gray = search scope / body candidates; amber = diagnostic paths. No body identity is inferred. Exact native IDs and paths remain in the manifest.",size=8)
            elif row.get("bounded_outcomes"):
                caption = _Column(page,32,528,y=520,size=8)
                caption.text(f"{len(row['bounded_outcomes'])} exact-subject bounded outcomes in the inspection register and manifest. Diagnostic records do not enlarge accepted coverage.",size=8)
            for attribute in row.get("attribute_labels", []):
                facts.text(f"{attribute['kind']}: {attribute['text']} ({attribute['state']})", size=9, color=BLUE)
            contexts = (mark or {}).get("junction_context", [])
            for context_index,entry in enumerate(contexts):
                junction = entry["junction"]
                if facts.y > 355:
                    facts.text(f"{len(contexts)-context_index} further exact junction records are drawn and retained in the manifest; sidebar space is bounded.",size=8,color=AMBER)
                    break
                facts.text(f"{junction['relation_type'].replace('_',' ')}: {junction['state']}. "
                           f"{len(entry['member_segments'])} referenced member segments shown. No physical continuity.", size=9,
                           color=BLUE if junction["state"] == "accepted" else AMBER)
                if junction.get('epistemic_state') == 'inferred':
                    facts.text('Inferred from native body, ring and paired interfaces. Physical fitting class remains ambiguous.',size=8,color=AMBER)
                if junction.get("source_boundary_connection_ref"):
                    facts.text(junction["source_boundary_connection_ref"],size=7,gap=4)
                if not entry["display_context_complete"]:
                    facts.text("Member geometry missing/ambiguous; display context incomplete.",size=9,color=AMBER)
                if junction["relation_type"] in {"projected_native_branch", "projected_fitting_body_branch"}:
                    facts.text("Port points are display markers only; they are not joined by an inferred centreline.",size=8,color=AMBER)
            for run in review["runs"]:
                if run["id"] in row["run_refs"]:
                    facts.text(f"Run: {run['trace_state'].replace('_',' ')}; physical continuation: {str(run['physical_continuation_established']).lower()}.", size=9)
            reasons = row["payload"].get("unresolved_reasons", [])
            if reasons:
                facts.text("Unresolved: " + "; ".join(reason.replace('_',' ') for reason in reasons), size=9, color=AMBER)
            if row["kind"]!="outcome":
                facts.text(f"Exact target-linked catalog rows: {len(row['catalog_rows'])}. Declared schedule link: unresolved; no explicit reference.", size=9)
            overlays = [entry for entry in review["reviews"] if entry["node_id"] == row["node_id"]]
            facts.text(f"Engineer overlays on this row: {len(overlays)}. Full provenance, independent value channels and decisions are preserved in the manifest.", size=9)
            page.insert_text((32,567), f"{row['id']} | source SHA {review['source_sha256'][:16]}", fontsize=8,color=MUTED)
        audit_rows.append({"id":row["id"], "schedule_label":row["schedule_label"], "schedule_page_number":destinations[row["id"]][0]+1,
                           "marked_page_numbers":list(range(first_detail+1,len(pdf)+1))})
    starts = {row["id"]:row["marked_page_numbers"][0]-1 for row in audit_rows}
    for page_number, rect, ref in schedule_links:
        if ref in starts:
            pdf[page_number].insert_link({"kind":fitz.LINK_GOTO,"from":rect,"page":starts[ref],"to":fitz.Point(0,0)})
    for page_number, rect, ref in detail_links:
        dest,y = destinations[ref]
        pdf[page_number].insert_link({"kind":fitz.LINK_GOTO,"from":rect,"page":dest,"to":fitz.Point(0,y)})
    manifest = deepcopy(review)
    manifest["presentation"] = {"edition":"sqlite_snapshot_inspection", "audit_rows":audit_rows,
        "marked_row_count":len(selected), "snapshot_row_count":len(review["rows"]),
        "selected_crop_scope_is_not_extraction_scope":True, "presentation_establishes_review":False,
        "comparison_page_numbers":comparison_pages,
        "page_count":len(pdf), "reciprocal_schedule_links":True}
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        previous_manifest = output.with_suffix(".manifest.json")
        previous_body = previous_manifest.read_bytes() if previous_manifest.exists() else b""
        previous_snapshot = json.loads(previous_body).get("snapshot_id","unindexed") if previous_body else "unindexed"
        revision = hashlib.sha256(output.read_bytes()+previous_body).hexdigest()[:12]
        archive = output.with_name(f"{output.stem}.{previous_snapshot[:16]}.{revision}{output.suffix}")
        for original, archived in ((output,archive),(previous_manifest,archive.with_suffix(".manifest.json"))):
            if original.exists():
                if archived.exists() and archived.read_bytes()!=original.read_bytes():
                    raise ValueError("existing audit archive differs; refusing to overwrite history")
                if not archived.exists():
                    shutil.copy2(original,archived)
        manifest["presentation"]["preserved_previous_audit"] = str(archive.resolve())
    pdf.set_metadata({"title":"MEP frozen snapshot inspection", "subject":review["snapshot_id"]})
    pdf.save(output,deflate=True,garbage=4)
    pdf.close()
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest,ensure_ascii=True,indent=2)+"\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    for name, path in DEFAULT_INPUTS.items():
        parser.add_argument("--" + name, type=Path, default=path)
    parser.add_argument("--discovery", type=Path, help="Frozen automatic discovery packet; replaces reviewed coverage input")
    parser.add_argument("--presentation", choices=("items", "evidence"), help="Readable items by default for automatic discovery; evidence retains the technical audit")
    parser.add_argument("--source-rendering", choices=("auto", "vector", "raster"), default="auto", help="For items: compare raster/vector file sizes by default")
    parser.add_argument("--raster-dpi", type=int, default=120)
    parser.add_argument("--audit-page-cache", type=Path,
                        help="Content-addressed source-page render cache")
    parser.add_argument("--bindings", type=Path, help="Frozen M4 bindings for evidence-centered close-ups")
    parser.add_argument("--terminology", type=Path, help="Hash-linked source text observations for close-ups")
    parser.add_argument("--database", type=Path, help="Read all presentation inputs from one frozen SQLite snapshot")
    parser.add_argument("--project")
    parser.add_argument("--document")
    parser.add_argument("--snapshot")
    parser.add_argument("--compare-to", help="Exact baseline snapshot for evidence-level comparison")
    parser.add_argument("--snapshot-record", action="append", help="Exact row ID to include as marked crop; all rows remain in schedule")
    args = parser.parse_args()
    if args.database:
        if not args.project or not args.document:
            parser.error("--database requires --project and --document")
        from src.drawing_engine.project.mep_project_review import load_review
        review = load_review(args.database, project=args.project, document=args.document, snapshot=args.snapshot, compare_to=args.compare_to)
        render_snapshot_review(args.source, args.output, review, selected_ids=args.snapshot_record)
        print(args.output)
        print(args.output.with_suffix(".manifest.json"))
        return
    names = [name for name in DEFAULT_INPUTS if name != "coverage" or args.discovery is None]
    inputs = {name: json.loads(getattr(args, name).read_text()) for name in names}
    if args.discovery is not None:
        inputs["discovery"] = json.loads(args.discovery.read_text())
    for name in ("bindings", "terminology"):
        if getattr(args, name) is not None:
            inputs[name] = json.loads(getattr(args, name).read_text())
    human = args.presentation == "items" or (args.presentation is None and args.discovery is not None)
    manifest = build_manifest(inputs, compact_regions=human)
    del inputs
    if human:
        if args.source_rendering == "auto":
            render_compact_items(args.source, args.output, manifest, raster_dpi=args.raster_dpi,
                                 page_cache_dir=args.audit_page_cache)
        else:
            render_items(args.source, args.output, manifest, source_rendering=args.source_rendering,
                         raster_dpi=args.raster_dpi, page_cache_dir=args.audit_page_cache)
    else:
        render(args.source, args.output, manifest)
    print(args.output)
    print(args.output.with_suffix(".manifest.json"))


if __name__ == "__main__":
    main()
