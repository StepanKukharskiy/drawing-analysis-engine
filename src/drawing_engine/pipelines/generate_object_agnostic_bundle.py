#!/usr/bin/env python3
"""Generate split observation, engineering, and evidence graphs for PDFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import fitz

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.drawing_engine.core.object_agnostic_understanding import page_summary, understand_page
from src.drawing_engine.disciplines.rebar.estimation_profile import DEFAULT_PROFILE_PATH, load_estimation_profile
from src.drawing_engine.core.canonical_knowledge_graph import build_canonical_knowledge_graph
from src.drawing_engine.core.automatic_adjudication import RULESET_VERSION, adjudicate_canonical_graph, ruleset_sha256
from src.drawing_engine.project.analysis_job_manifest import build_analysis_job_result
from src.drawing_engine.project.review_feedback import canonical_graph_sha256, write_review_record
from src.drawing_engine.disciplines.concrete.solver_replay import build_solver_replay
from src.drawing_engine.project.placement_pair_review import (
    adjudicate_placement_pair_catalog,
    build_placement_pair_catalog,
)
from src.drawing_engine.core.pdf_page_provenance import validate_extracted_page_provenance


PIPELINE_VERSION = "0.11.0"
PROGRESS_PREFIX = "REBAR_PROGRESS "


def pipeline_identity() -> dict[str, str]:
    """Return a reproducible fingerprint of the active graph-producing code."""

    inventory = ROOT / "src/drawing_engine/resources/structural_runtime_files.txt"
    names = [line.strip() for line in inventory.read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if not names or len(names) != len(set(names)):
        raise ValueError("empty or duplicate structural fingerprint inventory")
    for name in names:
        path = Path(name)
        if (path.is_absolute() or ".." in path.parts or path.as_posix() != name
                or not (ROOT / path).resolve().is_relative_to(ROOT)):
            raise ValueError("unsafe structural fingerprint input")
    files = [inventory, *(ROOT / name for name in sorted(names))]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {"version": PIPELINE_VERSION, "sha256": digest.hexdigest()}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _page_summaries(records: list[dict], canonical_graph: dict) -> list[dict]:
    summaries = []
    for record in records:
        page_key = f"page:{record['page']}"
        summary = page_summary(record)
        summary["canonical_entities"] = sum(
            item.get("provenance", {}).get("page_key") == page_key
            for item in canonical_graph.get("entities", [])
        )
        summary["canonical_relations"] = sum(
            item.get("provenance", {}).get("page_key") == page_key
            for item in canonical_graph.get("relations", [])
        )
        unresolved = [
            item
            for item in canonical_graph.get("unresolved", [])
            if item.get("page_key") == page_key
        ]
        summary["canonical_unresolved"] = len(unresolved)
        summary["canonical_cut_candidates"] = sum(
            item.get("kind") == "cut_at_relation_candidate"
            for item in unresolved
        )
        summary["cutting_plane_candidates"] = record["engineering_graph"].get(
            "view_frame_graph", {}
        ).get("summary", {}).get("cutting_plane_candidate_count", 0)
        summaries.append(summary)
    return summaries


def generate(
    source: Path,
    output_dir: Path,
    estimation_profile_path: Path = DEFAULT_PROFILE_PATH,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    *,
    page_numbers: list[int] | None = None,
) -> dict[str, Path]:
    started = perf_counter()
    generated_at = datetime.now(timezone.utc).isoformat()
    pipeline = pipeline_identity()
    source_pdf_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    provenance_path = source.with_suffix(".page-provenance.json")
    source_page_provenance = None
    if provenance_path.exists():
        source_page_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        validate_extracted_page_provenance(source_page_provenance, source)
    document_key = f"pdf-sha256:{source_pdf_sha256}"
    raw = fitz.open(source)
    selected_pages = list(range(1, len(raw) + 1)) if page_numbers is None else list(page_numbers)
    if (not selected_pages or any(type(n) is not int or not 1 <= n <= len(raw) for n in selected_pages)
            or selected_pages != sorted(set(selected_pages))):
        raw.close()
        raise ValueError("selected pages must be unique, ordered, one-based source page numbers")
    display = fitz.open()
    display.insert_pdf(raw)
    for number in selected_pages:
        display[number - 1].remove_rotation()
    records = []
    page_seconds = []
    estimation_profile = load_estimation_profile(estimation_profile_path)
    profile_sha256 = hashlib.sha256(Path(estimation_profile["source_path"]).read_bytes()).hexdigest()
    processing_options = {"page_rotation_removed": True}
    if page_numbers is not None:
        processing_options["source_page_numbers"] = selected_pages
    analysis_inputs = {
        "source_pdf_sha256": source_pdf_sha256,
        "pipeline_sha256": pipeline["sha256"],
        "estimation_profile_sha256": profile_sha256,
        "processing_options": processing_options,
    }
    analysis_cache_key = hashlib.sha256(
        json.dumps(analysis_inputs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    page_count = len(selected_pages)
    for page_index, number in enumerate(selected_pages):
        page = display[number - 1]
        page_started = perf_counter()
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "page_started",
                    "progress": round(8 + 66 * page_index / page_count),
                    "page": page.number + 1,
                    "page_count": page_count,
                }
            )

        def report_page_progress(stage: str, fraction: float, snapshot: dict[str, Any]) -> None:
            if progress_callback is None:
                return
            progress_callback(
                {
                    "stage": stage,
                    "progress": round(8 + 66 * (page_index + fraction) / page_count),
                    "page": page.number + 1,
                    "page_count": page_count,
                    "snapshot": snapshot,
                }
            )

        records.append(
            understand_page(
                page,
                estimation_profile=estimation_profile,
                progress_callback=report_page_progress,
            )
        )
        page_seconds.append(round(perf_counter() - page_started, 6))
    understanding_seconds = round(perf_counter() - started, 6)
    timing = {
        "drawing_understanding_seconds": understanding_seconds,
        "page_understanding_seconds": page_seconds,
    }
    stem = source.stem
    observation_path = output_dir / f"{stem}.drawing-scene.json"
    engineering_path = output_dir / f"{stem}.engineering-graph.json"
    evidence_path = output_dir / f"{stem}.evidence-store.json"
    bundle_path = output_dir / f"{stem}.object-agnostic-bundle.json"
    job_result_path = output_dir / f"{stem}.analysis-job-result.json"
    observation_pages = [{"page": item["page"], **item["observation_graph"]} for item in records]
    engineering_pages = []
    for item in records:
        page_payload = {"page": item["page"], **item["engineering_graph"]}
        # The canonical graph is document-wide below.  Avoid serializing the
        # same entities and relations once per page as well as at the root.
        page_payload.pop("canonical_knowledge_graph", None)
        engineering_pages.append(page_payload)
    canonical_graph = build_canonical_knowledge_graph(
        engineering_pages,
        document_key=document_key,
    )
    result_binding = {
        "source_pdf_sha256": source_pdf_sha256,
        "canonical_engineering_graph_sha256": canonical_graph_sha256(canonical_graph),
        "pipeline": pipeline,
        "ruleset": {"version": RULESET_VERSION, "sha256": ruleset_sha256()},
        "generated_at": generated_at,
        "model_versions": [
            item["engineering_graph"].get("perception_routing", {}).get("model_provenance", {})
            for item in records
        ],
    }
    if progress_callback is not None:
        progress_callback({"stage": "finalizing_graph", "progress": 75, "page_count": page_count})
    placement_pair_catalog = build_placement_pair_catalog(canonical_graph)
    placement_pair_adjudication = adjudicate_placement_pair_catalog(placement_pair_catalog)
    adjudication_started = perf_counter()
    adjudication = adjudicate_canonical_graph(canonical_graph)
    timing["automatic_adjudication_seconds"] = round(perf_counter() - adjudication_started, 6)
    adjudication["report"]["timing"] = {
        "drawing_understanding_seconds": timing["drawing_understanding_seconds"],
        "automatic_adjudication_seconds": timing["automatic_adjudication_seconds"],
    }
    automatic_delta_path = output_dir / f"{stem}.automatic-review-delta.json"
    automatic_report_path = output_dir / f"{stem}.automatic-review-report.json"
    effective_overlay_path = output_dir / f"{stem}.effective-canonical-overlay.json"
    solver_replay_path = output_dir / f"{stem}.solver-replay.json"
    solver_replay = build_solver_replay(canonical_graph, adjudication["overlay"], engineering_pages)
    write_json(observation_path, {"schema_version": "0.1.0", "pipeline": pipeline, "result_binding": result_binding, "source_pdf": str(source.resolve()), "source_page_provenance": source_page_provenance, "timing": timing, "pages": observation_pages})
    profile_identity = {
        "id": estimation_profile["id"],
        "source_path": estimation_profile["source_path"],
        "sha256": profile_sha256,
    }
    write_json(engineering_path, {"schema_version": "0.1.0", "pipeline": pipeline, "result_binding": result_binding, "document_key": document_key, "source_pdf": str(source.resolve()), "source_page_provenance": source_page_provenance, "processing_options": processing_options, "timing": timing, "estimation_profile": profile_identity, "canonical_knowledge_graph": canonical_graph, "placement_pair_catalog": placement_pair_catalog, "automatic_placement_pair_adjudication": placement_pair_adjudication, "pages": engineering_pages})
    write_json(evidence_path, {"schema_version": "0.1.0", "pipeline": pipeline, "result_binding": result_binding, "source_pdf": str(source.resolve()), "source_page_provenance": source_page_provenance, "timing": timing, "pages": [{"page": item["page"], **item["evidence_store"]} for item in records]})
    write_review_record(automatic_delta_path, canonical_graph, adjudication["delta"], allow_empty=True)
    write_json(automatic_report_path, adjudication["report"])
    write_json(
        effective_overlay_path,
        adjudication["overlay"]
        or {
            "schema_version": "0.1.0",
            "layer": "reviewed_canonical_overlay",
            "document_key": document_key,
            "base_canonical_graph_sha256": solver_replay["base_canonical_graph_sha256"],
            "effective_entities": canonical_graph.get("entities", []),
            "effective_relations": canonical_graph.get("relations", []),
            "review_annotations": [],
            "validation": {"status": "pass", "base_graph_unchanged": True, "schedule_values_used": False},
            "summary": {"decision_count": 0},
            "contract": {"base_observations_remain_authoritative": True, "reviewed_does_not_mean_direct": True, "schedule_values_used": False},
        },
    )
    write_json(solver_replay_path, solver_replay)
    write_json(
        bundle_path,
        {
            "schema_version": "0.1.0",
            "pipeline_mode": "object_agnostic_fail_closed",
            "pipeline": pipeline,
            "generated_at": generated_at,
            "result_binding": result_binding,
            "document_key": document_key,
            "analysis_cache_key": f"analysis-sha256:{analysis_cache_key}",
            "analysis_inputs": analysis_inputs,
            "source_pdf": str(source.resolve()),
            "source_page_provenance": source_page_provenance,
            "timing": timing,
            "estimation_profile": profile_identity,
            "processing_options": processing_options,
            "files": {
                "observation_graph": str(observation_path.resolve()),
                "engineering_graph": str(engineering_path.resolve()),
                "evidence_store": str(evidence_path.resolve()),
                "automatic_review_delta": str(automatic_delta_path.resolve()),
                "automatic_review_report": str(automatic_report_path.resolve()),
                "effective_canonical_overlay": str(effective_overlay_path.resolve()),
                "solver_replay": str(solver_replay_path.resolve()),
                "analysis_job_result": str(job_result_path.resolve()),
            },
            "page_summaries": _page_summaries(records, canonical_graph),
            "contract": {
                "object_template_required": False,
                "schedule_values_used_as_predictions": False,
                "one_primary_text_role_per_token": True,
                "claim_epistemic_state_required": True,
                "solid_output_requires_closed_constraints": True,
                "automatic_candidate_adjudication": True,
                "automatic_adjudication_can_abstain": True,
                "certified_relations_replayed_as_solver_inputs": True,
                "quantity_change_requires_solver_reclosure": True,
            },
        },
    )
    job_result = build_analysis_job_result(
        source,
        pipeline,
        {
            "bundle": bundle_path,
            "observation_graph": observation_path,
            "engineering_graph": engineering_path,
            "evidence_store": evidence_path,
            "automatic_review_delta": automatic_delta_path,
            "automatic_review_report": automatic_report_path,
            "effective_canonical_overlay": effective_overlay_path,
            "solver_replay": solver_replay_path,
        },
        _page_summaries(records, canonical_graph),
        result_binding,
    )
    write_json(job_result_path, job_result)
    display.close()
    raw.close()
    return {
        "bundle": bundle_path,
        "observation": observation_path,
        "engineering": engineering_path,
        "evidence": evidence_path,
        "automatic_delta": automatic_delta_path,
        "automatic_report": automatic_report_path,
        "effective_overlay": effective_overlay_path,
        "solver_replay": solver_replay_path,
        "analysis_job_result": job_result_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "object_agnostic")
    parser.add_argument("--estimation-profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument(
        "--progress-json",
        action="store_true",
        help="emit machine-readable progress events on stdout",
    )
    args = parser.parse_args()
    progress_callback = (
        lambda event: print(PROGRESS_PREFIX + json.dumps(event, ensure_ascii=True), flush=True)
        if args.progress_json
        else None
    )
    for source in args.inputs:
        outputs = generate(
            source,
            args.output_dir,
            estimation_profile_path=args.estimation_profile,
            progress_callback=progress_callback,
        )
        print(outputs["bundle"])


if __name__ == "__main__":
    main()
