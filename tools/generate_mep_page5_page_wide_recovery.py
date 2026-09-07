#!/usr/bin/env python3
"""Generate page-wide path roles and dual-channel projected route recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from fractions import Fraction
from typing import Any, Mapping

import fitz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.generate_mep_observed_takeoff import _artifact_json
from src.drawing_engine.disciplines.mep.mep_native_path_pack import NativePathPack
from src.drawing_engine.disciplines.mep.mep_page_wide_path_classification import (
    PageWidePathRolePack, validate_page_wide_path_roles,
    write_page_wide_path_roles,
)
from src.drawing_engine.disciplines.mep.mep_page_wide_route_recovery import (
    build_page_wide_route_recovery, discover_page_wide_outline_corridors,
    nominate_annotation_outline_styles,
    validate_page_wide_route_recovery,
)
from src.drawing_engine.project.project_packed_store import PackedProjectStore
from src.drawing_engine.disciplines.mep.mep_route_callout_tracing import (
    build_route_callout_traces, validate_route_callout_traces,
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp",
                                     delete=False) as stream:
        stream.write(body)
        temporary = Path(stream.name)
    os.replace(temporary, path)


_SYSTEMS = {
    "CHWS": "chilled_water_supply",
    "CHWR": "chilled_water_return",
    "HHWS": "heating_hot_water_supply",
    "HHWR": "heating_hot_water_return",
    "CWS": "condenser_water_supply",
    "CWR": "condenser_water_return",
    "CD": "condensate_drain",
}
_ROUTE_LINE = re.compile(
    r'(?P<size>(?:\d+\s+)?\d+/\d+|\d+(?:\.\d+)?)"\s*[Øø]?\s*'
    r'(?P<system>CHWS|CHWR|HHWS|HHWR|CWS|CWR|CD)\b', re.IGNORECASE)


def _size_inches(raw: str) -> float:
    if " " in raw:
        whole, fraction = raw.split(maxsplit=1)
        return float(whole) + float(Fraction(fraction))
    if "/" in raw:
        return float(Fraction(raw))
    return float(raw)


def _text_and_problem_regions(source: Path, page_number: int):
    with fitz.open(source) as pdf:
        page = pdf[page_number - 1]
        matrix = page.rotation_matrix
        words = page.get_text("words", sort=False)
        boxes = [list(fitz.Rect(word[:4]) * matrix) for word in words if str(word[4]).strip()]
        problem = []
        lines = {}
        for word in words:
            lines.setdefault((word[5], word[6]), []).append(word)
        route_annotations = []
        for line_words in lines.values():
            line_words.sort(key=lambda word: word[7])
            text = " ".join(str(word[4]) for word in line_words)
            match = _ROUTE_LINE.search(text)
            if match is None:
                continue
            token = match.group("system").upper()
            box = [min(float(word[0]) for word in line_words),
                   min(float(word[1]) for word in line_words),
                   max(float(word[2]) for word in line_words),
                   max(float(word[3]) for word in line_words)]
            marker = [page_number, text, [round(value, 6) for value in box]]
            route_annotations.append({
                "id": "mep_native_route_annotation." + hashlib.sha256(
                    json.dumps(marker, sort_keys=True).encode()).hexdigest()[:20],
                "state": "observed_native_text",
                "raw_text": text,
                "bbox_display": [round(value, 6) for value in box],
                "system_token": token,
                "system_candidate": _SYSTEMS[token],
                "nominal_size_inches": _size_inches(match.group("size")),
                "route_identity_established": False,
                "quantity_eligible": False,
            })
        for word in words:
            text = str(word[4]).strip().upper()
            if not re.match(r"^(?:HUH|FCU|AHU|AC|VAV)-?\d", text):
                continue
            box = list(fitz.Rect(word[:4]) * matrix)
            height = max(1.0, box[3] - box[1])
            region = [max(page.rect.x0, box[0] - 25 * height),
                      max(page.rect.y0, box[1] - 18 * height),
                      min(page.rect.x1, box[2] + 25 * height),
                      min(page.rect.y1, box[3] + 18 * height)]
            problem.append({
                "id": "mep_equipment_tag_review_region." + hashlib.sha256(
                    json.dumps([text, box]).encode()).hexdigest()[:20],
                "label": text, "bbox_display": [round(value, 6) for value in region],
                "selection_method": "native_equipment_tag_plus_text_height_scaled_context",
                "reviewed_coordinates_used_to_select_region": False,
            })
        return boxes, problem, list(page.rect), route_annotations


def _anchored_style_ids(styles, certificates):
    colours = {
        tuple(round(float(value), 6) for value in row["native_stroke_rgb"])
        for row in certificates
        if row.get("state") == "accepted_one_to_one_review_correlation"
        and row.get("colour_alone_establishes_route_identity") is False
    }
    return {
        index for index, style in enumerate(styles)
        if style.get("stroke") is not None
        and tuple(round(float(value), 6) for value in style["stroke"]) in colours
    }


def _interface_points(*artifacts: Mapping[str, Any], page_ref: str):
    output = {}

    def visit(value: Any, owner: str, accepted: bool = False,
              interface_class: str | None = None):
        if isinstance(value, Mapping):
            current_owner = str(value.get("id") or owner)
            current_accepted = bool(
                accepted or value.get("state") == "accepted"
                or value.get("accepted_projected_connection") is True
                or value.get("port_identity_established") is True)
            raw_class = str(value.get("interface_class")
                            or value.get("generic_class")
                            or value.get("relation_type") or "").lower()
            current_class = interface_class
            if "tee" in raw_class:
                current_class = "tee"
            elif "branch" in raw_class:
                current_class = "branch_fitting"
            elif "port" in raw_class:
                current_class = "typed_port"
            for key in ("point_display", "endpoint_point_display",
                        "centreline_junction_display", "port_point_display",
                        "contact_point_display"):
                point = value.get(key)
                if (isinstance(point, list) and len(point) == 2
                        and all(isinstance(v, (int, float)) for v in point)):
                    marker = (round(float(point[0]), 6), round(float(point[1]), 6),
                              current_owner)
                    output[marker] = {
                        "id": "mep_page_wide_interface_candidate." + hashlib.sha256(
                            json.dumps(marker).encode()).hexdigest()[:20],
                        "point_display": list(marker[:2]),
                        "source_record_ref": current_owner,
                        "state": "accepted" if current_accepted else "candidate",
                        "interface_class": current_class,
                    }
            for child in value.values():
                visit(child, current_owner, current_accepted, current_class)
        elif isinstance(value, list):
            for child in value:
                visit(child, owner, accepted, interface_class)

    for artifact in artifacts:
        for key in ("fitting_hypotheses", "projected_fitting_bindings", "identities"):
            for row in artifact.get(key, []):
                if row.get("page_ref") == page_ref:
                    visit(row, str(row.get("id") or key))
    return sorted(output.values(), key=lambda row: (row["point_display"], row["id"]))


def generate(*, source: Path, database: Path, denominator_path: Path,
             trace_path: Path, output_dir: Path, page_number: int = 5,
             reuse_path_role_manifest: Path | None = None) -> dict[str, Any]:
    denominator = _read(denominator_path)
    trace = _read(trace_path)
    if denominator["page_number"] != page_number:
        raise ValueError("denominator belongs to another source page")
    if denominator["source"]["pdf_sha256"] != _sha(source):
        raise ValueError("source PDF differs from denominator")
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite immutable recovery: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=output_dir.name + ".", suffix=".partial", dir=output_dir.parent))
    try:
        page_ref = denominator["page_ref"]
        descriptor_info = denominator["native_descriptor_pack"]
        path_info = denominator["native_authored_path_pack"]
        paths = NativePathPack(Path(path_info["path"]), path_info)
        styles = descriptor_info["styles"]
        text_boxes, problem_regions, page_rect, route_annotations = _text_and_problem_regions(
            source, page_number)
        with PackedProjectStore(database) as store:
            snapshot = store.snapshot(project_id="mep-coordination",
                                      document_id="coordination-set")
            composites = _artifact_json(
                store, project_id="mep-coordination", document_id="coordination-set",
                name="outlined-route-composites")
            bindings = _artifact_json(
                store, project_id="mep-coordination", document_id="coordination-set",
                name="attribute-bindings")
            route_graph = _artifact_json(
                store, project_id="mep-coordination", document_id="coordination-set",
                name="route-observations")
            registry = _artifact_json(
                store, project_id="mep-coordination", document_id="coordination-set",
                name="sheet-registry")
            fitting = _artifact_json(
                store, project_id="mep-coordination", document_id="coordination-set",
                name="fitting-hypotheses")
            equipment = _artifact_json(
                store, project_id="mep-coordination", document_id="coordination-set",
                name="equipment-2d-identities")
            try:
                fitting_bindings = _artifact_json(
                    store, project_id="mep-coordination", document_id="coordination-set",
                    name="projected-fitting-bindings")
            except KeyError:
                fitting_bindings = {}
        outlined = [row for row in composites.get("accepted_composites", [])
                    if row.get("page_ref") == page_ref]
        outlined_drawings = {
            int(ref.removeprefix("drawing[").split("]", 1)[0])
            for row in outlined for ref in row.get("member_source_primitive_refs", [])
        }
        anchored_styles = _anchored_style_ids(
            styles, trace.get("colour_mapping_certificates", []))
        registered = next(row for row in registry["pages"]
                          if row["page_number"] == page_number)
        drawing_scale = float(registered["fields"]["scale"][
            "drawing_inches_per_paper_inch"])
        role_path = staging / f"page-{page_number:03d}.authored-path-roles.pack"
        if reuse_path_role_manifest is not None:
            role_manifest = _read(reuse_path_role_manifest)
            if (role_manifest.get("denominator_sha256") != _sha(denominator_path)
                    or role_manifest.get("trace_sha256") != _sha(trace_path)):
                raise ValueError("path-role checkpoint inputs differ")
            source_role_path = Path(role_manifest["path"])
            if not source_role_path.is_file():
                source_role_path = reuse_path_role_manifest.parent / role_path.name
            shutil.copyfile(source_role_path, role_path)
            role_manifest = {key: value for key, value in role_manifest.items()
                             if key not in {"path", "denominator_sha256", "trace_sha256"}}
        else:
            role_manifest = write_page_wide_path_roles(
                path_pack=paths, styles=styles, output_path=role_path,
                text_boxes_display=text_boxes, anchored_style_ids=anchored_styles,
                outlined_member_drawing_ordinals=outlined_drawings,
                page_rect_display=page_rect)
        _write(staging / "path-role-manifest.json", {
            **role_manifest, "path": str((output_dir / role_path.name).resolve()),
            "denominator_sha256": _sha(denominator_path),
            "trace_sha256": _sha(trace_path),
        })
        roles = PageWidePathRolePack(role_path, role_manifest)
        errors = validate_page_wide_path_roles(path_pack=paths, role_pack=roles)
        fragment_sources = {
            row["id"]: row["source_primitive_ref"]
            for page in route_graph.get("pages", []) if page.get("page_ref") == page_ref
            for row in page.get("fragments", []) if row.get("source_primitive_ref")
        }
        anchored_systems = {
            row.get("system") for row in trace.get("colour_mapping_certificates", [])
            if row.get("state") == "accepted_one_to_one_review_correlation"
        }
        neutral_annotations = [
            row for row in route_annotations
            if row.get("system_candidate") not in anchored_systems
        ]
        annotation_style = nominate_annotation_outline_styles(
            path_pack=paths, role_pack=roles, styles=styles,
            route_annotations=neutral_annotations,
            drawing_inches_per_paper_inch=drawing_scale,
            anchored_style_ids=anchored_styles)
        annotated_widths = {}
        annotated_width_evidence = {}
        for style_id in anchored_styles:
            stroke = tuple(round(float(value), 6)
                           for value in styles[style_id]["stroke"])
            systems = {
                row.get("system") for row in trace.get("colour_mapping_certificates", [])
                if row.get("state") == "accepted_one_to_one_review_correlation"
                and tuple(round(float(value), 6)
                          for value in row["native_stroke_rgb"]) == stroke}
            annotated_widths[style_id] = sorted({
                float(row["nominal_size_inches"]) / drawing_scale * 72.0
                for row in route_annotations if row["system_candidate"] in systems})
            annotated_width_evidence[style_id] = sorted(
                row["id"] for row in route_annotations
                if row["system_candidate"] in systems)
        for correlation in annotation_style["supported_style_correlations"]:
            annotated_widths[int(correlation["style_id"])] = sorted({
                float(value) / drawing_scale * 72.0
                for value in correlation["nominal_size_candidates_inches"]})
            annotated_width_evidence[int(correlation["style_id"])] = [correlation["id"]]
        outline_discovery = discover_page_wide_outline_corridors(
            page_ref=page_ref, path_pack=paths, role_pack=roles, styles=styles,
            legacy_composites=outlined,
            additional_style_ids=annotation_style["nominated_style_ids"],
            annotated_widths_by_style=annotated_widths)
        current_outlined = outline_discovery["accepted_corridors"]
        payload = build_page_wide_route_recovery(
            page_ref=page_ref, path_pack=paths, role_pack=roles, styles=styles,
            text_boxes_display=text_boxes, outlined_composites=current_outlined,
            m4_relations=bindings.get("relations", []),
            fragment_source_refs=fragment_sources,
            interface_points=_interface_points(
                fitting, fitting_bindings, equipment, page_ref=page_ref),
            problem_regions=problem_regions,
            route_annotations=route_annotations,
            drawing_inches_per_paper_inch=drawing_scale,
            annotation_supported_style_ids=annotation_style[
                "nominated_style_ids"],
            annotation_style_correlations=annotation_style[
                "supported_style_correlations"])
        payload["outline_corridor_discovery"] = {
            "summary": outline_discovery["summary"],
            "authority": outline_discovery["authority"],
            "annotation_supported_width_search": [
                {"style_id": style_id, "width_candidates_display_points": widths,
                 "evidence_refs": annotated_width_evidence[style_id],
                 "system_identity_established": False}
                for style_id, widths in sorted(annotated_widths.items())],
        }
        payload["annotation_style_nomination"] = annotation_style
        payload["route_annotations"] = route_annotations
        payload["route_callout_tracing"] = build_route_callout_traces(payload)
        errors.extend(validate_route_callout_traces(payload))
        errors.extend(validate_page_wide_route_recovery(payload))
        if denominator["acceptance_gate"].get("legacy_M3_primary_route_count") != 0:
            errors.append("v2 denominator grants legacy M3 primary route authority")
        payload["development_status"] = (
            "page5_source_coverage_closed_engineer_review_pending"
            if not errors else "development_rejected")
        payload["inputs"] = {
            "source_pdf": str(source.resolve()), "source_pdf_sha256": _sha(source),
            "source_page": page_number, "project_snapshot_v3": snapshot["id"],
            "denominator_path": str(denominator_path.resolve()),
            "denominator_sha256": _sha(denominator_path),
            "trace_path": str(trace_path.resolve()), "trace_sha256": _sha(trace_path),
            "native_text_box_count": len(text_boxes),
            "equipment_tag_problem_region_count": len(problem_regions),
            "accepted_outlined_composite_count": len(outlined),
            "current_page_wide_corridor_count": len(current_outlined),
            "anchored_route_style_ids": sorted(anchored_styles),
            "annotation_nominated_style_ids": annotation_style[
                "nominated_style_ids"],
        }
        payload["path_role_pack"] = {
            **role_manifest,
            "path": str((output_dir / role_path.name).resolve()),
        }
        payload["validation"] = {
            "status": "valid_page5_diagnostic_recovery" if not errors else "development_rejected",
            "errors": errors,
        }
        _write(staging / "recovery.json", payload)
        os.replace(staging, output_dir)
        return payload
    except BaseException:
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=ROOT / "M&P mark-up against shop systems piping.pdf")
    parser.add_argument("--database", type=Path,
                        default=ROOT / "data/projects/mep/project-v3.sqlite")
    parser.add_argument("--denominator", type=Path, default=ROOT /
                        "output/mep-source-denominator-page5-v2-2026-09-03/manifest.json")
    parser.add_argument("--trace", type=Path, default=ROOT /
                        "output/mep-audit-trace-network-2026-09-03/trace-network.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT /
                        "output/mep-page5-page-wide-recovery-v20-2026-09-03")
    parser.add_argument("--trace-frozen-recovery", type=Path,
                        help="Replay tracing over a frozen current recovery without re-extraction")
    parser.add_argument("--reuse-path-role-manifest", type=Path, default=ROOT /
                        "output/mep-page5-page-wide-recovery-2026-09-03/path-role-manifest.json")
    parser.add_argument("--page", type=int, default=5)
    args = parser.parse_args()
    if args.trace_frozen_recovery:
        payload = _read(args.trace_frozen_recovery)
        errors = validate_page_wide_route_recovery(payload)
        if errors or payload.get("validation", {}).get("errors"):
            raise ValueError("frozen recovery failed validation")
        if _sha(Path(payload["inputs"]["source_pdf"])) != payload["inputs"]["source_pdf_sha256"]:
            raise ValueError("frozen recovery source PDF changed")
        payload["route_callout_tracing"] = build_route_callout_traces(payload)
        errors = validate_route_callout_traces(payload)
        if errors:
            raise ValueError(errors)
        payload["inputs"]["tracing_parent_recovery"] = {
            "path": str(args.trace_frozen_recovery.resolve()),
            "sha256": _sha(args.trace_frozen_recovery),
        }
        args.output_dir.mkdir(parents=True, exist_ok=False)
        _write(args.output_dir / "recovery.json", payload)
        print(json.dumps(payload["route_callout_tracing"]["summary"], indent=2))
        return
    payload = generate(source=args.source, database=args.database,
                       denominator_path=args.denominator, trace_path=args.trace,
                       output_dir=args.output_dir, page_number=args.page,
                       reuse_path_role_manifest=args.reuse_path_role_manifest)
    print(json.dumps({
        "development_status": payload["development_status"],
        "coverage": payload["coverage"],
        "path_roles": payload["path_role_pack"]["role_path_counts"],
        "validation": payload["validation"],
    }, indent=2))


if __name__ == "__main__":
    main()
