"""Portable detail artifacts over the native companion, used by estimation_cli."""
from __future__ import annotations

from html import escape
import json
from pathlib import Path
import shutil

import fitz

from src.drawing_engine.project.analysis_job_manifest import sha256_file
from src.drawing_engine.disciplines.detail.detail_declarations import compare_detail, extract_detail_declarations
from src.drawing_engine.disciplines.detail.native_detail_assembly import extract_assembly
from src.drawing_engine.audit.detail_audit_pdf import write_detail_audit
from src.drawing_engine.exports.detail_dxf_export import export_plate_dxfs
from src.drawing_engine.exports.estimation_project_export import standard_delivery, artifact_record


@standard_delivery
def export_detail(source, output, *, assembly_mark, page_number=1, dxf_units="native"):
    from src.drawing_engine.cli import write_json

    source, output = Path(source).resolve(strict=True), Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    local_source = output / "source.pdf"
    shutil.copyfile(source, local_source)
    implementations = [Path(__file__), (Path(__file__).resolve().parents[3] / 'src/drawing_engine/disciplines/detail/native_detail_assembly.py'),
                       (Path(__file__).resolve().parents[3] / 'src/drawing_engine/disciplines/detail/detail_declarations.py'),
                       (Path(__file__).resolve().parents[3] / 'src/drawing_engine/core/vector_topology.py'),
                       (Path(__file__).resolve().parents[3] / 'src/drawing_engine/core/pdf_native_metadata.py'),
                       (Path(__file__).resolve().parents[3] / 'src/drawing_engine/cli.py'),
                       (Path(__file__).resolve().parents[3] / 'src/drawing_engine/disciplines/concrete/generic_prismatic_solver.py'),
                       (Path(__file__).resolve().parents[3] / 'src/drawing_engine/disciplines/concrete/schedule_comparison.py'),
                       (Path(__file__).resolve().parents[3] / 'src/drawing_engine/audit/detail_audit_pdf.py'),
                       (Path(__file__).resolve().parents[3] / 'src/drawing_engine/exports/detail_dxf_export.py'),
                       (Path(__file__).resolve().parents[3] / 'src/drawing_engine/exports/estimation_project_export.py'),
                       (Path(__file__).resolve().parents[3] / 'src/drawing_engine/project/project_packed_store.py')]
    implementation = {p.name: sha256_file(p) for p in implementations}
    with fitz.open(local_source) as doc:
        if page_number < 1 or page_number > len(doc):
            raise ValueError("page outside source PDF")
        page = doc[page_number - 1]
        assembly, native = extract_assembly(page, assembly_mark)
        assembly_path = output / "assembly.json"
        write_json(assembly_path, assembly)
        frozen_hash = sha256_file(assembly_path)
        # No declared values exist until the drawing-derived artifact is frozen.
        declarations = extract_detail_declarations(page, assembly_mark)
        comparison = compare_detail(assembly, declarations)
        comparison["frozen_calculation"] = {"artifact": "assembly.json", "sha256": frozen_hash}
        if sha256_file(assembly_path) != frozen_hash:
            raise ValueError("detail assembly changed during independent comparison")
        dxf = export_plate_dxfs(assembly, output, units=dxf_units)
        dxf["frozen_assembly_sha256"] = frozen_hash
        dxf["source_sha256"] = sha256_file(local_source)
        write_json(output / "dxf.json", dxf)
        notes = [line for line in native.text if any(term in line["text"].lower()
                 for term in ("allowance", "припуск", "general data", "общие данные"))]
        # Preserve whole native text blocks for notes split over multiple lines.
        note_blocks = [{"bbox_display": list(b[:4]), "text": b[4]} for b in page.get_text("blocks")
                       if any(term in b[4].lower() for term in ("allowance", "припуск", "general data", "общие данные"))]
        evidence = {"page": page_number, "source_sha256": sha256_file(local_source),
                    "coordinates": "PDF display points", "page_bbox_display": list(page.rect),
                    "words": [{"id": f"word[{i}]", "text": w[4], "bbox_display": list(w[:4])}
                              for i, w in enumerate(native.words)],
                    "topology": native.topology, "native_note_lines": notes, "native_note_blocks": note_blocks}
        write_json(output / "evidence.json", evidence)
        write_json(output / "declarations.json", declarations)
        write_json(output / "comparison.json", comparison)
        write_json(output / "unresolved.json", {"assembly": assembly["unresolved"], "declarations": declarations["unresolved"],
                                                "comparisons": comparison["comparisons"],
                                                "dxf": [r for r in dxf["parts"] if r["state"] == "abstained"],
                                                "unrun": ["mesh/cage repetition", "cross-file assembly", "quote approval"]})
        boxes = [fitz.Rect(v["bbox_display"]) for c in assembly["child_parts"]
                 for v in c.get("views", c.get("shaft_projections", []))]
        if boxes:
            crop = fitz.Rect(boxes[0])
            for box in boxes[1:]:
                crop |= box
            crop |= fitz.Rect(assembly["title"]["bbox_display"])
            crop = (crop + (-60, -10, 100, 90)) & page.rect
        else:
            crop = page.rect
        section_title = assembly.get("supporting_views", {}).get("section_title", {}).get("bbox_display")
        if section_title:
            crop |= fitz.Rect(section_title) + (-10, -10, 10, 10)
        write_json(output / "audit-scope.json", {"crop": list(crop)})
    if implementation != {p.name: sha256_file(p) for p in implementations}:
        raise ValueError("detail implementation changed during export")
    files = {"source_pdf": "source.pdf", "assembly": "assembly.json", "evidence_store": "evidence.json",
             "declarations": "declarations.json", "comparison": "comparison.json", "unresolved": "unresolved.json",
             "audit_scope": "audit-scope.json", "dxf_report": "dxf.json"}
    files.update({f"dxf_plate_{i + 1}": r["path"] for i, r in enumerate(dxf["parts"]) if r["path"]})
    manifest = {"schema_version": "estimation_cli.v1", "execution_status": "succeeded", "mode": "native_detail_assembly",
                "source_sha256": sha256_file(local_source), "implementation_sha256": implementation,
                "pymupdf_version": fitz.VersionBind,
                "frozen_assembly_sha256": frozen_hash, "page": page_number, "assembly_mark": assembly_mark,
                "contract": {"takeoff_completeness": "not_established", "assembly_state": assembly["state"],
                             "approval_granted_by_execution": False, "quantity_scope": "one_detail_definition",
                             "client_comparison_parity": "separate_validation_track",
                             "project_delivery": "source_sqlite_audit_and_applicable_exports.v1",
                             "dxf_scope": "2d_outer_face_outlines_without_internal_geometry_or_fabrication_approval"},
                "artifacts": {key: artifact_record(output, output / value) for key, value in files.items()}}
    write_json(output / "result.json", manifest)
    return output / "result.json"


def write_report(output, assembly, declarations, comparison, crop, evidence):
    rows, outlines = [], []
    colors = {"plate": "#137b63", "welded_bar_family": "#b25219"}
    for part in assembly["child_parts"]:
        color = colors[part["kind"]]
        views = part.get("views", []) + part.get("end_projections", []) + part.get("shaft_projections", [])
        for view in views:
            x0, y0, x1, y1 = view["bbox_display"]
            outlines.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="none" stroke="{color}" stroke-width="1"><title>{escape(part["id"] + " " + view["drawing_ref"])}</title></rect>')
        dimensions = {key: sorted({d["value"] for d in values}) for key, values in part.get("dimensions", {}).items()}
        body = {"native_dimensions": dimensions, "projected": part.get("projected"),
                "native_length_dimension": sorted({d["value"] for d in part.get("length_dimensions", [])}),
                "calculated": part["calculated"], "conditional": part.get("conditional"), "physical": part["physical"], "approved": None}
        rows.append(f'<article><h2 style="color:{color}">{escape(part["mark"])} · {escape(part["kind"])}</h2><pre>{escape(json.dumps(body, indent=2))}</pre></article>')
    references = set()

    def collect(value):
        if isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, str) and (value.startswith("word[") or value.startswith("drawing[")):
            references.add(value)

    collect(assembly)
    collect(declarations)
    lookup = {r["id"]: r for r in evidence["words"] + evidence["topology"]["segments"]}
    ref_rows = "".join(f'<tr><td>{escape(ref)}</td><td>{escape(str(lookup[ref].get("text", lookup[ref].get("kind"))))}</td><td>{escape(str(lookup[ref]["bbox_display"]))}</td></tr>' for ref in sorted(references) if ref in lookup)
    unresolved = "".join(f'<li>{escape(r["reason"])}</li>' for r in assembly["unresolved"])
    notes = "\n\n".join(r["text"] for r in evidence["native_note_blocks"])
    declared_summary = json.dumps([{"part": r["part_mark"], "declared": r["declared"]} for r in declarations["records"]], indent=2)
    comparison_summary = json.dumps(comparison["comparisons"], indent=2)
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>{escape(assembly['mark'])} assembly audit</title>
<style>body{{font:16px system-ui;margin:32px auto;max-width:1200px;color:#182b34;background:#f7f8f6}}h1{{margin-bottom:8px}}a{{color:#125a89}}svg{{width:100%;background:white}}.parts{{display:flex;gap:24px;flex-wrap:wrap}}article{{flex:1;min-width:300px;background:white;padding:20px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}}table{{border-collapse:collapse;width:100%;font-size:12px}}td{{border-bottom:1px solid #ddd;padding:6px;overflow-wrap:anywhere}}li{{margin:10px 0}}</style>
<h1>{escape(assembly['mark'])} · detail assembly</h1><p>Execution succeeded. Complete takeoff and fabrication release are not established. Green: plate; orange: bar projections. Hover outlines for native drawing references.</p>
<p><a href="source.pdf#page={assembly['page']}">Source PDF · page {assembly['page']}</a> · <a href="assembly.json">Frozen assembly</a> · <a href="evidence.json">Native evidence</a> · <a href="comparison.json">Comparisons</a> · <a href="result.json">Hashed artifact index</a></p>
<svg viewBox="{crop.x0} {crop.y0} {crop.width} {crop.height}" role="img" aria-label="Native drawing with separate plate and bar outlines"><image href="drawing.png" x="{crop.x0}" y="{crop.y0}" width="{crop.width}" height="{crop.height}"/>{''.join(outlines)}</svg>
<div class="parts">{''.join(rows)}</div><h2>Unresolved quantity authority</h2><ul>{unresolved}</ul>
<h2>Independent declarations</h2><pre>{escape(declared_summary)}</pre>
<p>Only independently eligible quantities receive a discrepancy. Conditional comparisons retain their assumptions; declared mass and count never fill calculated values.</p><details><summary>Discrepancies and unresolved comparisons</summary><pre>{escape(comparison_summary)}</pre></details>
<h2>Drawing notes</h2><pre>{escape(notes)}</pre><details><summary>Native evidence references and page coordinates</summary><table><tr><th>Reference</th><th>Observation</th><th>Display bbox</th></tr>{ref_rows}</table></details></html>'''
    (output / "audit.html").write_text(html, encoding="utf-8")
