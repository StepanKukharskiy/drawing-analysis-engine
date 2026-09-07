"""One cage and one explicit reference PDF through the shared delivery contract."""
from pathlib import Path
import shutil

import fitz

from src.drawing_engine.disciplines.detail.native_cage_assembly import extract_cage
from src.drawing_engine.disciplines.detail.native_detail_assembly import NativeDetail
from src.drawing_engine.disciplines.detail.cage_document_links import tables, document_anchor, reference_rows, link_references
from src.drawing_engine.project.analysis_job_manifest import sha256_file
from src.drawing_engine.exports.estimation_project_export import standard_delivery, artifact_record


def evidence(native):
    return {"page": native.page.number+1, "coordinates": "PDF display points",
            "words": [{"id": f"word[{i}]", "text": w[4], "bbox_display": list(w[:4])} for i, w in enumerate(native.words)],
            "topology": native.topology}


@standard_delivery
def export_cage(source, output, *, assembly_mark, reference, page_number=1, reference_page=1):
    from src.drawing_engine.cli import write_json

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, output/"source.pdf")
    shutil.copyfile(reference, output/"reference.pdf")
    root = Path(__file__).resolve().parents[3]
    owners = ["disciplines/detail/native_cage_assembly.py", "disciplines/detail/cage_document_links.py",
              "disciplines/detail/native_detail_assembly.py", "core/vector_topology.py", "core/pdf_native_metadata.py",
              "exports/cage_assembly_export.py", "exports/estimation_project_export.py", "audit/detail_audit_pdf.py",
              "project/project_packed_store.py", "cli.py"]
    implementation = {name: sha256_file(root/"src/drawing_engine"/name) for name in owners}
    with fitz.open(output/"source.pdf") as drawing, fitz.open(output/"reference.pdf") as reference_pdf:
        if not 1 <= page_number <= len(drawing) or not 1 <= reference_page <= len(reference_pdf):
            raise ValueError("page outside explicit source/reference PDF")
        assembly, native = extract_cage(drawing[page_number-1], assembly_mark)
        write_json(output/"assembly.json", assembly)
        frozen = sha256_file(output/"assembly.json")
        # Only now observe declarations, document anchors and reference rows.
        reference_native = NativeDetail(reference_pdf[reference_page-1])
        ref_tables = tables(reference_native)
        source_anchor = document_anchor(reference_native, ref_tables)
        destination_anchor = document_anchor(native, tables(native))
        rows = reference_rows(reference_native, ref_tables, assembly_mark)
        links = link_references(source_anchor, destination_anchor, rows, assembly)
        notes = [t for t in native.text if re_note(t["text"], assembly_mark)]
        declarations = {"records": rows, "repetition_note_observations": notes,
                        "calculation_input": False, "frozen_assembly_sha256": frozen}
        references = {"source": source_anchor, "destination": destination_anchor, "links": links,
                      "reference_source_sha256": sha256_file(output/"reference.pdf"),
                      "scope": "one explicitly supplied reference page and one destination page",
                      "all_project_references_complete": False}
        if sha256_file(output/"assembly.json") != frozen:
            raise ValueError("declarations changed frozen cage interpretation")
        for name, value in (("declarations", declarations), ("document_links", references),
                            ("evidence_store", evidence(native)), ("reference_evidence", evidence(reference_native)),
                            ("dxf_report", {"state": "unsupported", "reason": "projected cage has no qualified cutting paths or physical bar geometry", "fabrication_release": False}),
                            ("unresolved", {"geometry": assembly["unresolved"], "cross_file": [l for l in links if l["detail_applicability_state"] != "derived"],
                                            "missing_reference_rows": not bool(rows), "quote_approval": None})):
            write_json(output/(name+".json"), value)
    if implementation != {name: sha256_file(root/"src/drawing_engine"/name) for name in owners}:
        raise ValueError("cage implementation changed during export")
    artifacts = {"source_pdf": "source.pdf", "reference_pdf": "reference.pdf"}
    artifacts.update({name: name+".json" for name in ("assembly", "declarations", "document_links", "evidence_store", "reference_evidence", "dxf_report", "unresolved")})
    manifest = {"schema_version": "estimation_cli.v1", "execution_status": "succeeded", "mode": "native_cage_assembly",
        "source_sha256": sha256_file(output/"source.pdf"), "page": page_number, "assembly_mark": assembly_mark,
        "implementation_sha256": implementation, "pymupdf_version": fitz.VersionBind,
        "frozen_assembly_sha256": frozen,
        "contract": {"project_delivery": "source_sqlite_audit_and_applicable_exports.v1", "takeoff_completeness": "not_established",
                     "quantity_scope": "projected_detail_definition", "approval_granted_by_execution": False},
        "artifacts": {name: artifact_record(output, output/path) for name, path in artifacts.items()}}
    write_json(output/"result.json", manifest)
    return output/"result.json"


def re_note(text, mark):
    # Preserve the full authored sentence fragment; never turn prose into count.
    return "Примечание" in text and mark.replace(" ", "") in text.replace(" ", "")
