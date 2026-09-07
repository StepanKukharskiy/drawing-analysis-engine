"""Rectangular outer-face DXF from frozen native contours and face dimensions.

No declarations or solids are inputs to outline eligibility. Internal geometry,
installed counts, net quantities and fabrication approval remain separate.
"""
from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path


def _filename_mark(value, fallback):
    text = unicodedata.normalize("NFC", str(value or ""))
    text = re.sub(r"[^\w-]+", "-", text).strip("-_")
    return text.encode("utf-8")[:64].decode("utf-8", errors="ignore").rstrip("-_") or fallback


def export_plate_dxfs(assembly, output, *, units="native"):
    if units not in {"native", "mm"}:
        raise ValueError("DXF units must be native or mm")
    unit_evidence = assembly.get("linear_unit_evidence", {})
    native_mm = unit_evidence.get("state") == "direct" and unit_evidence.get("unit") == "mm"
    report = {"schema_version": "detail_dxf.v2", "scope": "one_outer_face_outline_per_file",
              "requested_units": units, "fabrication_release": False, "approved": None, "parts": []}
    used_names = set()
    for index, part in enumerate(assembly.get("child_parts", [])):
        record = {"part_ref": part["id"], "part_mark": part.get("mark"), "state": "abstained",
                  "path": None, "reasons": [], "physical_count": None}
        report["parts"].append(record)
        if part["kind"] != "plate":
            record["reasons"].append("unsupported_part_kind_for_outer_face_outline")
            continue
        contour = part.get("outer_contour", {})
        points = contour.get("points_display", [])
        if (contour.get("kind") != "closed_native_rectangle" or len(points) != 4
                or not contour.get("drawing_ref") or len(set(contour.get("evidence_refs", []))) != 4
                or any(len(p) != 2 or any(not math.isfinite(v) for v in p) for p in points)):
            record["reasons"].append("closed_native_outer_contour_required")
            continue
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        lo, hi = (min(xs), min(ys)), (max(xs), max(ys))
        if (set(map(tuple, points)) != {(x, y) for x in (lo[0], hi[0]) for y in (lo[1], hi[1])}
                or lo[0] >= hi[0] or lo[1] >= hi[1]
                or any((a[0] == b[0]) == (a[1] == b[1]) for a, b in zip(points, points[1:] + points[:1]))):
            record["reasons"].append("closed_native_rectangular_cycle_required")
            continue
        dimensions = part.get("dimensions", {})
        values = {key: {d["value"] for d in dimensions.get(key, [])}
                  for key in ("width", "height")}
        if any(len(v) != 1 for v in values.values()):
            record["reasons"].append("unique_native_face_dimensions_required")
            continue
        size = {key: next(iter(v)) for key, v in values.items()}
        if any(not math.isfinite(v) or v <= 0 for v in size.values()):
            record["reasons"].append("positive_finite_face_dimensions_required")
            continue
        if any(d.get("profile_ref") != contour["drawing_ref"] or not d.get("evidence_refs")
               or len(d.get("measured_interval_display", [])) != 2
               or any(abs(a-b) > .15 for a,b in zip(d["measured_interval_display"], (lo[axis],hi[axis])))
               for axis,key in enumerate(("width", "height")) for d in dimensions[key]):
            record["reasons"].append("face_dimension_scope_mismatch")
            continue
        scales = [(hi[0]-lo[0])/size["width"], (hi[1]-lo[1])/size["height"]]
        if max(scales)/min(scales) >= 1.025:
            record["reasons"].append("face_dimension_scale_mismatch")
            continue
        if not native_mm and unit_evidence.get("observations"):
            record["reasons"].append("unsupported_or_conflicting_native_unit_evidence")
            continue
        basis = "native_mm" if native_mm else "assumed_mm" if units == "mm" else "unspecified_drawing_units"
        suffix = {"native_mm": "mm", "assumed_mm": "mm-assumed", "unspecified_drawing_units": "unitless"}[basis]
        detail_name = _filename_mark(assembly.get("mark"), "detail")
        part_name = _filename_mark(part.get("mark"), f"unmarked-{index + 1}")
        stem = f"{detail_name}-part-{part_name}-outline"
        filename = f"{stem}.{suffix}.dxf"
        occurrence = 1
        while filename.casefold() in used_names:
            occurrence += 1
            filename = f"{stem}-{occurrence}.{suffix}.dxf"
        used_names.add(filename.casefold())
        relative = f"dxf/{filename}"
        path = Path(output) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        width, height = size["width"], size["height"]
        outline = [(0, 0), (width, 0), (width, height), (0, height)]
        # R2000 entities need valid layout/block ownership and document tables.
        # A bare ENTITIES section can parse as tags yet fail in Rhino/OpenDesign.
        import ezdxf

        document = ezdxf.new("R2000", units=4 if basis != "unspecified_drawing_units" else 0)
        document.modelspace().add_lwpolyline(outline, close=True, dxfattribs={"layer": "0"})
        document.saveas(path)
        record.update({"state": "derived" if native_mm else "convention_dependent" if units == "mm" else "observed",
                       "path": relative, "unit_basis": basis, "units": "mm" if basis != "unspecified_drawing_units" else None,
                       "outline_xy": outline, "source_contour_ref": contour["drawing_ref"],
                       "closed_contour_count": 1, "geometry": "uncompensated_rectangular_outer_outline",
                       "internal_geometry_exported": False, "complete_part_profile_established": False,
                       "entity_type": "LWPOLYLINE", "layer": "0", "dxf_version": "AC1015",
                       "strict_quantity_eligible": False, "fabrication_release": False,
                       "evidence_refs": sorted({ref for key in ("width", "height") for d in dimensions[key]
                                                for ref in d.get("evidence_refs", [])} |
                                               set(contour["evidence_refs"]))})
    return report
