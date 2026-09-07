"""Marked native drawing and compact review pages for the detail CLI."""
from __future__ import annotations

import fitz

from src.drawing_engine.audit.audit_presentation import audit_font_file


INK = (0.08, 0.16, 0.20)
GREEN = (0.07, 0.47, 0.38)
ORANGE = (0.70, 0.30, 0.08)
PURPLE = (0.52, 0.19, 0.64)
BLUE = (0.04, 0.44, 0.65)


def _text(page, rect, text, *, size=12, color=INK, bold=False):
    remaining = page.insert_textbox(fitz.Rect(rect), text, fontsize=size,
                                   fontname="DetailBold" if bold else "DetailSans",
                                   fontfile=audit_font_file(bold), color=color, lineheight=1.25)
    if remaining < 0:
        raise ValueError("detail audit text does not fit its review panel")


def _native_marks(page, references, lookup, color, title, *, dashed=False):
    """Replay stored primitives; never trace a replacement line from a bbox."""
    for ref in dict.fromkeys(references):
        if ref not in lookup:
            raise ValueError(f"detail audit missing native evidence: {ref}")
        row = lookup[ref]
        if "start_display" in row:
            if row.get("control_points_display"):
                points = [row["start_display"], *row["control_points_display"], row["end_display"]]
                if len(points) != 4:
                    raise ValueError("unsupported native curve evidence")
                page.draw_bezier(*points, color=color, width=.8, dashes="[2 2] 0" if dashed else None)
            else:
                page.draw_line(row["start_display"], row["end_display"], color=color, width=.8,
                               dashes="[2 2] 0" if dashed else None)
        else:
            page.draw_rect(fitz.Rect(row["bbox_display"]), color=color, width=.7)
        # Native IDs remain inspectable without printing a wall of identifiers.
        rect = fitz.Rect(row["bbox_display"]) + (-.7, -.7, .7, .7)
        annot = page.add_rect_annot(rect)
        annot.set_border(width=0)
        annot.set_opacity(0)
        annot.set_info(title=title, content=ref)
        annot.update()


def write_detail_audit(source, output, assembly, comparison, crop, dxf, evidence):
    """Render only the selected detail page; source.pdf preserves the full input."""
    with fitz.open(source) as raw, fitz.open() as marked, fitz.open() as pdf:
        source_index = assembly["page"] - 1
        marked.insert_pdf(raw, from_page=source_index, to_page=source_index)
        page = marked[0]
        # Evidence coordinates use the engine's display frame. Rotate the source
        # into that frame exactly as the other native audit renderers do.
        page.remove_rotation()
        words = {r["id"]: r for r in evidence["words"]}
        lookup = {r["id"]: r for r in evidence["words"] + evidence["topology"]["segments"]}
        support = assembly.get("supporting_views", {})
        support_refs = support.get("cut_marker_refs", []) + support.get("section_title", {}).get("evidence_refs", [])
        _native_marks(page, support_refs, lookup, BLUE, "Named section correspondence; metric cutting transform unresolved")
        if support.get("face") and support.get("section"):
            for label in ("face", "section"):
                box = fitz.Rect(support[label]["bbox_display"]) + (-3, -3, 3, 3)
                page.draw_rect(box, color=BLUE, width=.6, dashes="[3 2] 0")
                page.insert_text((box.x0, box.y1 + 8), label.upper() + " / same part mark", fontsize=5.5, color=BLUE)
        for part in assembly["child_parts"]:
            color = GREEN if part["kind"] == "plate" else ORANGE
            for view in part.get("views", []) + part.get("end_projections", []) + part.get("shaft_projections", []):
                rect = fitz.Rect(view["bbox_display"])
                page.draw_rect(rect, color=color, width=1.0)
                annot = page.add_rect_annot(rect)
                annot.set_colors(stroke=color)
                annot.set_info(title=f"Part {part['mark']} - {part['kind']}",
                               content="\n".join([part["id"], view["drawing_ref"], *view.get("evidence_refs", [])]))
                annot.update()
                _native_marks(page, view.get("evidence_refs", []), lookup, color,
                              f"Part {part['mark']} / {part['kind']} / native profile")
            for annotation in part.get("annotations", []):
                _native_marks(page, annotation["evidence_refs"], lookup, color,
                              f"Part {part['mark']} / {annotation['kind']} leader / {annotation['text']}")
            dimensions = [d for group in part.get("dimensions", {}).values() for d in group]
            dimensions += part.get("length_dimensions", [])
            for dimension in dimensions:
                _native_marks(page, dimension["evidence_refs"], lookup, PURPLE,
                              f"Part {part['mark']} / complete native dimension chain / {dimension['value']} drawing units")
                for ref in dimension.get("evidence_refs", []):
                    if ref in words:
                        annot = page.add_rect_annot(fitz.Rect(words[ref]["bbox_display"]))
                        annot.set_colors(stroke=color)
                        annot.set_info(title=f"Part {part['mark']} - native dimension",
                                       content=f"Value: {dimension['value']} drawing units\n" + "\n".join(dimension["evidence_refs"]))
                        annot.update()
        for issue in assembly.get("unresolved", []):
            # Keep unresolved alternatives attached to their own source locations.
            for ref in issue.get("evidence_refs", []):
                if ref in lookup:
                    annot = page.add_rect_annot(fitz.Rect(lookup[ref]["bbox_display"]) + (-1, -1, 1, 1))
                    annot.set_colors(stroke=ORANGE)
                    annot.set_border(width=.3, dashes=[2, 2])
                    annot.set_info(title="UNRESOLVED", content=issue["reason"] + "\n" + ref)
                    annot.update()
        source_width, source_height = page.rect.width, page.rect.height
        page.set_mediabox(fitz.Rect(0, 0, source_width + 480, source_height))
        _text(page, (source_width + 25, 40, source_width + 455, 110),
              assembly['mark'] + " / INTERPRETATION", size=24, bold=True)
        _text(page, (source_width + 25, 130, source_width + 455, 470),
              "Green: plate contours and mark leaders.\nOrange: accepted bar projections, when present. Round symbols retain their recorded role.\nPurple: dimension chains, terminals and native labels.\nBlue: face / named section evidence.\nDashed amber: unresolved source evidence; click for the recorded reason.", size=18)
        _text(page, (source_width + 25, 510, source_width + 455, 860),
              "Named section correspondence is preserved. A metric cutting-plane transform is unresolved.\n\nSupported quantities remain qualified. Native dimensions do not supply a missing unit convention, physical bar count or diameter.\n\nThe rest of the sheet is outside this detail scope.", size=17)
        _text(page, (source_width + 25, 940, source_width + 455, 1260),
              "SQLite records: assembly, evidence_store, comparison, unresolved.\n\nUse CLI inspect with each native reference retained in the drawing annotations. Declarations remain independent; no approval is granted.", size=17)
        cover = pdf.new_page(width=1190, height=842)
        cover.draw_rect(cover.rect, fill=(0.97, 0.98, 0.97), color=None)
        _text(cover, (40, 28, 1150, 68), f"{assembly['mark']} / Detail observations", size=25, bold=True)
        _text(cover, (40, 78, 1150, 115),
              "Green: plate. Orange: bars. Purple: dimension chains. Blue: section evidence. Native records: project.sqlite.", size=12)
        if not page.get_contents():
            _text(cover, (40, 140, 740, 480), "No native drawing content available.")
        else:
            cover.show_pdf_page(fitz.Rect(40, 135, 770, 535), marked, 0, clip=crop)
        _text(cover, (40, 550, 770, 605),
              f"Source page {assembly['page']} - selected detail scope. The next page retains the complete marked source sheet.", size=11)
        _text(cover, (810, 140, 1150, 175), "RESULT BOUNDARIES", size=15, bold=True)
        _text(cover, (810, 190, 1150, 355),
              "Scope: one detail definition.\nAssembly count: unresolved.\nComplete takeoff: not established.\nFabrication release: not granted.\nApproval: separate engineer decision.", size=13)
        unit = assembly.get("linear_unit_evidence", {})
        _text(cover, (810, 370, 1150, 490),
              "Drawing units: " + ("native millimetres" if unit.get("unit") == "mm" else "unresolved") +
              ".\nDXF units: " + ("explicit millimetre export assumption" if dxf["requested_units"] == "mm" and unit.get("unit") != "mm"
                                  else "native units, or unitless if unresolved") + ".", size=13)
        _text(cover, (40, 680, 1150, 755),
              "Calculated, conditional and declared values remain separate. A closed DXF contour does not establish material, "
              "physical count, fabrication approval or cutting-machine settings.", size=13, bold=True)
        _text(cover, (40, 785, 1150, 815), "Portable project / source.pdf + project.sqlite + audit.pdf + applicable DXF / result.json binds file hashes", size=10)
        pdf.insert_pdf(marked)
        # One review page per part keeps arbitrary evidence/part counts from
        # overflowing a fixed summary table.
        for part in assembly["child_parts"]:
            review = pdf.new_page(width=1190, height=842)
            _text(review, (40, 30, 1150, 75), f"{assembly['mark']} / Part {part['mark']} / {part['kind']}", size=23, bold=True)
            values = {key: ", ".join(f"{v:g}" for v in sorted({d["value"] for d in records})) or "unresolved"
                      for key, records in part.get("dimensions", {}).items()}
            if values:
                detail = "\n".join(f"{key.replace('_', ' ').capitalize()}: {value} drawing units" for key, value in values.items())
            else:
                projected = part.get("projected", {})
                detail = "\n".join(f"{key.replace('_', ' ').capitalize()}: {value}" for key, value in projected.items())
                lengths = sorted({d["value"] for d in part.get("length_dimensions", [])})
                detail += "\nNative length annotation: " + (", ".join(f"{v:g}" for v in lengths) or "unresolved")
            _text(review, (40, 110, 560, 320), "DRAWING OBSERVATIONS\n\n" + detail, size=14)
            calculated = part.get("calculated", {})
            conditional = part.get("conditional", {})
            quantities = "\n".join(f"{key.replace('_', ' ')}: {value if value is not None else 'unresolved'}"
                                   for key, value in calculated.items() if key not in {"state", "basis"})
            if conditional:
                quantities += f"\nConditional volume: {conditional['volume_m3']:.9g} m3, assuming millimetres."
            _text(review, (620, 110, 1150, 330), "CALCULATION\n\n" + quantities + "\nApproval: unresolved.", size=14)
            record = next((r for r in dxf["parts"] if r["part_ref"] == part["id"]), {})
            _text(review, (40, 365, 1150, 465), "DXF\n" + (record.get("path") or "No supported outline available") +
                  "\n" + (record.get("unit_basis") or "; ".join(record.get("reasons", []))) +
                  ("\n2D outer contour only; internal geometry is not exported."
                   if record.get("internal_geometry_exported") is False else ""), size=13)
            matched = next((r for r in comparison["comparisons"] if r.get("part_id") == part["id"]), {})
            declared = matched.get("declared", {})
            declared_lines = "\n".join(f"{key.replace('_', ' ')}: {value}"
                                       for key, value in declared.items() if value is not None)
            _text(review, (40, 510, 1150, 650),
                  "INDEPENDENT DECLARATIONS\n" + (declared_lines or "No uniquely matched declaration.") +
                  "\nStrict comparison: " + matched.get("severity", "unresolved") +
                  "; delta: " + str(matched.get("delta") if matched.get("delta") is not None else "unresolved"), size=12)
            _text(review, (40, 710, 1150, 795),
                  "NATIVE EVIDENCE\nPart reference: " + part["id"] +
                  ". Click annotations on the full drawing page for native path references.\nFrozen assembly SHA-256: " +
                  comparison["frozen_calculation"]["sha256"], size=11)
        # Preserve all reasons without fitting an unbounded list into one box.
        for issue in assembly.get("unresolved", []):
            review = pdf.new_page(width=1190, height=842)
            _text(review, (40, 35, 1150, 80), f"{assembly['mark']} / Unresolved observation", size=23, bold=True)
            _text(review, (40, 130, 1150, 600), issue["reason"], size=18)
            _text(review, (40, 700, 1150, 795),
                  "State: " + issue.get("state", "unknown") + ". Evidence references are retained in SQLite's unresolved record and frozen assembly. "
                  "An unresolved observation is not a zero quantity.", size=12)
        pdf.set_metadata({"title": f"{assembly['mark']} detail observations", "author": "Drawing understanding engine",
                          "subject": "Native evidence, separate quantity channels and conditional DXF export"})
        pdf.save(output, garbage=4, deflate=True)


def write_cage_audit(project, output):
    """Source marks replay exclusively from the existing frozen SQLite store."""
    from src.drawing_engine.exports.estimation_project_export import checked_path

    assembly = project.load("assembly")
    links = project.load("document_links")
    declarations = project.load("declarations")
    with fitz.open() as pdf:
        for key, evidence_key in (("source_pdf", "evidence_store"), ("reference_pdf", "reference_evidence")):
            evidence = project.load(evidence_key)
            with fitz.open(checked_path(project.root, project.manifest["artifacts"][key])) as raw:
                pdf.insert_pdf(raw, from_page=evidence["page"]-1, to_page=evidence["page"]-1)
            page = pdf[-1]
            page.remove_rotation()
            lookup = {r["id"]: r for r in evidence["words"]+evidence["topology"]["segments"]}
            if key == "source_pdf":
                if assembly.get("title"):
                    _native_marks(page, assembly["title"]["evidence_refs"], lookup, BLUE, "Exact cage caption")
                for part in assembly["child_parts"]:
                    refs = part["annotation"]["evidence_refs"]+[ref for p in part["projections"] for ref in p["evidence_refs"]]
                    _native_marks(page, refs, lookup, GREEN if part["role"] == "transverse" else PURPLE, "Position "+part["mark"]+" / projected geometry")
                for occurrence in assembly["parent_occurrences"]:
                    _native_marks(page, occurrence["evidence_refs"], lookup, BLUE, "Authored repetition targets; physical count unresolved")
                for note in declarations["repetition_note_observations"]:
                    _native_marks(page, note["evidence_refs"], lookup, ORANGE, "Independent repetition declaration; no quantity authority")
                anchor = links["destination"]
            else:
                anchor = links["source"]
                for link in links["links"]:
                    _native_marks(page, link["source_evidence_refs"], lookup,
                                  GREEN if link["detail_applicability_state"] == "derived" else ORANGE,
                                  "Cross-file position "+link["part_mark"]+" / "+link["detail_applicability_state"])
            if anchor:
                _native_marks(page, anchor["evidence_refs"], lookup, BLUE, "Native document / actual sheet / change record")
            width, height = page.rect.width, page.rect.height
            page.set_mediabox(fitz.Rect(0, 0, width+360, height))
            _text(page, (width+20, 35, width+340, 105), assembly["mark"]+" / CAGE REVIEW", size=20, bold=True)
            _text(page, (width+20, 125, width+340, 450),
                  "Green: transverse child outlines or matched reference rows.\nPurple: two projected longitudinal rails.\nBlue: named scope, authored repetition and native document anchors.\nAmber: declarations or unresolved references.\n\nCounts describe drawing projections. Physical count, cutting length, mass and approval remain unresolved.", size=14)
        review = pdf.new_page(width=1190, height=842)
        _text(review, (40, 30, 1150, 90), assembly["mark"]+" / CHILDREN, REPETITION AND REFERENCES", size=22, bold=True)
        lines = ["Position "+p["mark"]+f": {p['projected_count']} projected {p['role']} outlines; physical count unresolved." for p in assembly["child_parts"]]
        lines += [f"Parent annotation: {p['leader_target_count']} authored targets; physical repetition unresolved." for p in assembly["parent_occurrences"]]
        _text(review, (40, 115, 1150, 320), "\n\n".join(lines) or "No supported cage interpretation.", size=16)
        refs = [f"Position {l['part_mark']} -> actual drawing sheet {l['destination_sheet']}: {l['detail_applicability_state']}." for l in links["links"]]
        _text(review, (40, 350, 1150, 500), "\n".join(refs) or "No uniquely bound cross-file reference.", size=16)
        _text(review, (40, 530, 1150, 680),
              "Declared counts and lengths were read only after geometry was frozen. They never fill calculated quantities.\nDXF unsupported: projected cage has no qualified cutting paths or physical bar geometry.\n"+"\n".join(assembly["unresolved"]), size=13)
        _text(review, (40, 720, 1150, 805), "Frozen assembly SHA-256: "+project.manifest["frozen_assembly_sha256"]+
              "\nNative words, strokes, source hashes and both document anchors remain inspectable in project.sqlite.", size=11)
        pdf.save(output, garbage=4, deflate=True)
