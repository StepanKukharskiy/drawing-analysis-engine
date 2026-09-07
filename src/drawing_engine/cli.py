"""Thin, app-independent concrete/rebar comparison and evidence export.

Run ``estimation audit`` in a folder of PDFs after adding bin/ to PATH.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from src.drawing_engine.project.analysis_job_manifest import ALLOWED_ARTIFACTS, sha256_file
from src.drawing_engine.disciplines.rebar.estimation_profile import DEFAULT_PROFILE_PATH
from src.drawing_engine.exports.estimation_project_export import FrozenProject, artifact_record, standard_delivery, render_project_audit


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def progress(record):
    # The app callback includes full drawing snapshots; keep CLI progress bounded.
    fields = {key: value for key, value in record.items()
              if key in {"stage", "progress", "page", "page_count"}}
    print(json.dumps({"event": "progress", **fields}), file=sys.stderr, flush=True)


@standard_delivery
def export_comparison(source: Path, output: Path, *, frozen_bundle: Path | None = None,
                      estimation_profile: Path = DEFAULT_PROFILE_PATH,
                      page_number: int | None = None, task: str = "structural") -> Path:
    """Execute existing engines, preserving native artifacts and quantity channels.

    The CLI runs this producer under run_artifact_job; direct Python callers must
    likewise account for their output and temporary paths. A new output directory
    is required. Failure retains partial files without publishing result.json.
    """
    from src.drawing_engine.pipelines.generate_object_agnostic_bundle import generate
    from src.drawing_engine.disciplines.concrete.schedule_comparison import write_estimate_comparison

    source, output = Path(source).resolve(strict=True), Path(output).absolute()
    if source.suffix.lower() != ".pdf":
        raise ValueError("source must be a PDF")
    if page_number is not None and frozen_bundle is not None:
        raise ValueError("page selection requires fresh analysis; replay uses the frozen scope")
    output.mkdir(parents=True, exist_ok=False)
    source_dir = output / "source"
    source_dir.mkdir()
    local_source = source_dir / source.name
    shutil.copyfile(source, local_source)
    artifacts = {"source_pdf": local_source}
    provenance = source.with_suffix(".page-provenance.json")
    if provenance.exists():
        local_provenance = local_source.with_suffix(".page-provenance.json")
        shutil.copyfile(provenance, local_provenance)
        artifacts["source_page_provenance"] = local_provenance
    graph_dir = output / "graphs"
    graph_dir.mkdir()
    if frozen_bundle is None:
        progress({"stage": "drawing_analysis"})
        options = {"page_numbers": [page_number]} if page_number is not None else {}
        paths = generate(local_source, graph_dir, estimation_profile, progress_callback=progress, **options)
        bundle_path = paths["bundle"]
    else:
        progress({"stage": "frozen_replay"})
        frozen_bundle = Path(frozen_bundle).resolve(strict=True)
        bundle = read_json(frozen_bundle)
        source_hash = sha256_file(local_source)
        if bundle.get("document_key") != f"pdf-sha256:{source_hash}":
            raise ValueError("frozen bundle source PDF hash mismatch")
        required = {"observation_graph", "engineering_graph", "evidence_store"}
        if not required <= bundle.get("files", {}).keys():
            raise ValueError("frozen bundle must contain observation, engineering and evidence artifacts")
        # Resolve siblings, never follow a manifest's arbitrary absolute paths.
        for name, value in bundle["files"].items():
            if name not in ALLOWED_ARTIFACTS | {"analysis_job_result"}:
                raise ValueError(f"unsupported bundle artifact: {name}")
            origin = frozen_bundle.parent / Path(value).name
            target = graph_dir / origin.name
            if target.exists():
                raise ValueError("duplicate bundle artifact filename")
            shutil.copyfile(origin, target)
        bundle_path = graph_dir / frozen_bundle.name
        if bundle_path.exists():
            raise ValueError("bundle filename collides with an artifact")
        shutil.copyfile(frozen_bundle, bundle_path)
    bundle = read_json(bundle_path)
    artifacts["bundle"] = bundle_path
    for name, value in bundle["files"].items():
        artifacts[name] = graph_dir / Path(value).name
    graph_path = artifacts["engineering_graph"]
    graph = read_json(graph_path)
    source_hash = sha256_file(local_source)
    page_indexes = {}
    for name in ("engineering_graph", "observation_graph", "evidence_store"):
        payload = graph if name == "engineering_graph" else read_json(artifacts[name])
        binding = payload.get("result_binding") or {}
        if binding.get("source_pdf_sha256", source_hash) != source_hash:
            raise ValueError(f"{name} source PDF hash mismatch")
        if payload.get("document_key", f"pdf-sha256:{source_hash}") != f"pdf-sha256:{source_hash}":
            raise ValueError(f"{name} document identity mismatch")
        if payload.get("pipeline") != bundle.get("pipeline"):
            raise ValueError(f"{name} pipeline mismatch")
        if payload.get("result_binding") != graph.get("result_binding"):
            raise ValueError(f"{name} result binding mismatch")
        pages = payload["pages"]
        page_indexes[name] = {page["page"]: index for index, page in enumerate(pages)}
        if len(page_indexes[name]) != len(pages) or page_indexes[name].keys() != page_indexes["engineering_graph"].keys():
            raise ValueError(f"{name} page coverage mismatch")
    frozen_hash = sha256_file(graph_path)
    progress({"stage": "independent_schedule_comparison"})
    comparison_path = output / "comparison.json"
    write_estimate_comparison(local_source, graph_path, comparison_path)
    comparison = read_json(comparison_path)
    if sha256_file(graph_path) != frozen_hash or comparison["frozen_calculation"]["sha256"] != frozen_hash:
        raise ValueError("engineering graph changed during comparison")
    artifacts["comparison"] = comparison_path
    unresolved_path = output / "unresolved.json"
    write_json(unresolved_path, {
        "canonical": graph.get("canonical_knowledge_graph", {}).get("unresolved", []),
        "pages": [{"page": page["page"], "unresolved": page.get("unresolved", [])}
                  for page in graph.get("pages", [])],
        "unavailable_comparisons": [
            {"page": page["page"], "channel": channel, "record": record,
             "comparison_pointer": f"/pages/{index}/comparison/{channel}"}
            for index, page in enumerate(comparison["pages"])
            for channel, record in page["comparison"].items()
            if record["calculated"] is None or record["declared"] is None],
        "contract": {"not_an_exhaustive_gap_inventory": True,
                     "nested_candidates_and_stops_retained_in_engineering_graph": True},
    })
    artifacts["unresolved"] = unresolved_path
    profile_path = output / "estimation-profile.json"
    shutil.copyfile(estimation_profile, profile_path)
    artifacts["estimation_profile"] = profile_path
    manifest = {
        "schema_version": "estimation_cli.v1",
        "execution_status": "succeeded",
        "mode": "frozen_replay" if frozen_bundle else "fresh_analysis",
        "source_sha256": source_hash,
        "result_currency": comparison["result_currency"],
        "binding_issues": comparison["binding_issues"],
        "pipeline": bundle["pipeline"],
        "contract": {
            "requested_task": task,
            "executed_tasks": ["concrete", "rebar"],
            "source_page_numbers": [page["page"] for page in graph["pages"]],
            "page_scope": "selected_pages" if page_number is not None else "frozen_scope" if frozen_bundle else "all_pages",
            "takeoff_completeness": "not_established",
            "comparison_scope": "existing_page_totals",
            "line_level_scope_matching": "not_implemented_by_this_command",
            "mep_hvac": "not_run",
            "detail_assembly_completion": "not_established",
            "quantity_channels_preserved_in_native_artifacts": True,
            "approval_granted_by_execution": False,
            "frozen_native_path_fields_are_historical_use_artifact_index": True,
            "project_delivery": "source_sqlite_audit_and_applicable_exports.v1",
            "dxf": "not_applicable_to_page_total_comparison",
        },
        "artifacts": {name: artifact_record(output, path)
                      for name, path in sorted(artifacts.items())},
        "page_links": [
            {"page": page["page"], "comparison": f"/pages/{index}",
             **{name: f"/pages/{indexes[page['page']]}" for name, indexes in page_indexes.items()}}
            for index, page in enumerate(comparison["pages"])],
    }
    progress({"stage": "project_database"})
    manifest_path = output / "result.json"
    write_json(manifest_path, manifest)
    return manifest_path


def discover_pdfs(source: Path) -> list[Path]:
    source = Path(source).resolve(strict=True)
    if source.is_dir():
        sources = sorted((p for p in source.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
                         key=lambda p: (p.name.casefold(), p.name))
    else:
        sources = [source] if source.is_file() and source.suffix.lower() == ".pdf" else []
    if not sources:
        raise ValueError("no PDF files found; supply a PDF or a folder containing PDFs")
    return sources


def resolve_audit_task(sources, page_number=None, *, interactive=False):
    """Choose a runner from native M1 title evidence; ask when inconclusive."""
    import fitz
    from src.drawing_engine.disciplines.mep.mep_sheet_registry import _native_tokens, _parse_page_fields, _stable_id
    detected = set()
    for source in sources:
        document_key = f"pdf-sha256:{sha256_file(source)}"
        with fitz.open(source) as pdf:
            pages = list(range(1, len(pdf) + 1)) if page_number is None else [page_number]
            if not pages or any(type(n) is not int or not 1 <= n <= len(pdf) for n in pages):
                raise ValueError("selected page is outside the source PDF")
            for number in pages:
                page = pdf[number - 1]
                ref = _stable_id("mep_page_observation", document_key, number, page.xref)
                _, fields = _parse_page_fields(_native_tokens(page, ref))
                detected.add("mep" if fields.get("discipline", {}).get("value") == "mechanical" else None)
    if detected == {"mep"}:
        selected, basis = "mep", "native_sheet_titles"
    else:
        if not interactive:
            raise ValueError("cannot auto-detect one pipeline for the selected pages; specify --task structural, concrete, rebar, mep or hvac")
        print("I could not identify one pipeline for the selected pages.", file=sys.stderr)
        while True:
            selected = input("Task [structural/concrete/rebar/mep/hvac]: ").strip().lower()
            if selected in {"structural", "concrete", "rebar", "mep", "hvac"}:
                break
            print("Enter one of the listed tasks.", file=sys.stderr)
        basis = "user_selection"
    print(json.dumps({"event": "task_selected", "task": selected, "basis": basis}), file=sys.stderr, flush=True)
    return selected


def export_audits(source: Path, output: Path, *, page_number=None, task="auto",
                  estimation_profile=DEFAULT_PROFILE_PATH) -> Path:
    """Fresh per-document projects; batch execution never aggregates quantities."""
    sources = discover_pdfs(source)
    if task == "auto":
        task = resolve_audit_task(sources, page_number)
    def produce(pdf, destination):
        progress({"stage": f"source:{pdf.name}"})
        if task in {"mep", "hvac"}:
            from src.drawing_engine.exports.estimation_project_export import export_fresh_mep
            return export_fresh_mep(pdf, destination, page_number=page_number, task=task)
        return export_comparison(pdf, destination, page_number=page_number, task=task,
                                 estimation_profile=estimation_profile)
    if len(sources) == 1:
        return produce(sources[0], output)
    output.mkdir(parents=True, exist_ok=False)
    documents = []
    for index, pdf in enumerate(sources, 1):
        destination = output / f"{index:03d}-{pdf.stem}"
        try:
            result = produce(pdf, destination)
            documents.append({"source": pdf.name, "execution_status": "succeeded",
                              "result": artifact_record(output, result)})
        except Exception as exc:
            documents.append({"source": pdf.name, "execution_status": "failed",
                              "error_type": type(exc).__name__, "reason": str(exc)})
    failed = sum(row["execution_status"] == "failed" for row in documents)
    result = output / "result.json"
    write_json(result, {"schema_version": "estimation_batch.v1",
        "execution_status": "partially_failed" if failed else "succeeded",
        "task": task, "requested_page": page_number, "documents": documents,
        "failed_document_count": failed, "quantities_aggregated": False})
    return result


def inspect_result(manifest_path: Path, artifact: str | None = None, pointer: str = ""):
    """Read native JSON through the portable index, checking its content hash."""
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") == "estimation_batch.v1":
        if artifact is not None:
            raise ValueError("choose a document result.json from the batch documents index to inspect artifacts")
        return select_pointer(manifest, pointer)
    if manifest.get("schema_version") != "estimation_cli.v1" or manifest.get("execution_status") != "succeeded":
        raise ValueError("not a completed estimation CLI export")
    payload = manifest
    if artifact is not None:
        if manifest.get("project", {}).get("schema_version") == 3:
            payload = FrozenProject(manifest_path.parent, manifest).load(artifact)
            return select_pointer(payload, pointer)
        record = manifest["artifacts"][artifact]
        path = (manifest_path.parent / record["path"]).resolve(strict=True)
        if not path.is_relative_to(manifest_path.parent):
            raise ValueError("artifact path escapes export")
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError("artifact content hash mismatch")
        if path.suffix != ".json":
            raise ValueError("inspect reads JSON artifacts; open binary deliverables with their PDF, DXF or SQLite reader")
        payload = read_json(path)
    return select_pointer(payload, pointer)


def select_pointer(payload, pointer):
    if pointer:
        if not pointer.startswith("/"):
            raise ValueError("JSON pointer must start with /")
        for token in pointer[1:].split("/"):
            token = token.replace("~1", "/").replace("~0", "~")
            payload = payload[int(token)] if isinstance(payload, list) else payload[token]
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("compare", help="analyze a PDF and export bounded comparison/evidence")
    compare.add_argument("source", type=Path)
    compare.add_argument("--output", type=Path, required=True, help="new export directory; never overwrites")
    compare.add_argument("--frozen-bundle", type=Path, help="explicit historical replay instead of fresh analysis")
    compare.add_argument("--estimation-profile", type=Path, default=DEFAULT_PROFILE_PATH)
    compare.add_argument("--reserve-gib", type=float, default=1, help="expected additional peak output/temp reserve")
    compare.add_argument("--max-growth-gib", type=float, default=1, help="monitored allocated growth budget")
    compare.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    cage = commands.add_parser("cage", help="interpret one native ladder cage and an explicit cross-file reference")
    cage.add_argument("source", type=Path)
    cage.add_argument("--assembly", required=True)
    cage.add_argument("--reference", type=Path, required=True)
    cage.add_argument("--page", type=int, default=1)
    cage.add_argument("--reference-page", type=int, default=1)
    cage.add_argument("--output", type=Path, required=True)
    cage.add_argument("--reserve-gib", type=float, default=1)
    cage.add_argument("--max-growth-gib", type=float, default=1)
    cage.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    detail = commands.add_parser("detail", help="extract a bounded native plate/bar assembly with independent declarations")
    detail.add_argument("source", type=Path)
    detail.add_argument("--assembly", required=True, help="exact native assembly title mark")
    detail.add_argument("--page", type=int, default=1, help="one-based source page")
    detail.add_argument("--output", type=Path, required=True, help="new export directory; never overwrites")
    detail.add_argument("--dxf-units", choices=("native", "mm"), default="native",
                        help="native preserves unresolved units; mm explicitly assumes millimetres for the DXF only")
    detail.add_argument("--reserve-gib", type=float, default=1)
    detail.add_argument("--max-growth-gib", type=float, default=1)
    detail.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    mep = commands.add_parser("mep", help="deliver a one-page MEP audit from an existing frozen SQLite project")
    mep.add_argument("source", type=Path)
    mep.add_argument("--database", type=Path, required=True)
    mep.add_argument("--project", default="mep-coordination")
    mep.add_argument("--document", default="coordination-set")
    mep.add_argument("--snapshot")
    mep.add_argument("--page", type=int, required=True)
    audit = commands.add_parser("audit", help="fresh audit of every PDF in the current folder, or one PDF")
    audit.add_argument("source", nargs="?", default=".", help="PDF, folder, or 'page' in 'audit page 2'")
    audit.add_argument("page_selection", nargs="?", type=int, help="page number after 'page'")
    audit.add_argument("--page", type=int, help="one-based source page; default: all pages")
    audit.add_argument("--task", choices=("auto", "structural", "concrete", "rebar", "mep", "hvac"), default="auto",
                       help="default: detect native MEP/HVAC titles, otherwise ask for a task")
    audit.add_argument("--estimation-profile", type=Path, default=DEFAULT_PROFILE_PATH)
    replay = commands.add_parser("replay-audit", help="regenerate an audit from frozen SQLite; no extraction")
    replay.add_argument("source", type=Path, help="project result.json")
    export = commands.add_parser("export", help="explicit JSON or detail HTML export from frozen SQLite")
    export.add_argument("source", type=Path, help="project result.json")
    export.add_argument("--format", choices=("json", "html"), default="json")
    export.add_argument("--artifact")
    for command in (mep, audit, replay, export):
        command.add_argument("--output", type=Path, required=command is not audit,
                             help="new PDF file for replay-audit; new directory otherwise")
        command.add_argument("--reserve-gib", type=float, default=1)
        command.add_argument("--max-growth-gib", type=float, default=1)
        command.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    inspect = commands.add_parser("inspect", help="inspect exported native records and evidence")
    inspect.add_argument("result", type=Path)
    inspect.add_argument("--artifact", help="artifact key from result.json, e.g. comparison or engineering_graph")
    inspect.add_argument("--pointer", default="", help="JSON pointer within the selected artifact")
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            if args.source == "page" and args.page_selection is not None:
                if args.page is not None:
                    raise ValueError("use either 'page N' or --page, not both")
                args.page, args.source = args.page_selection, "."
            elif args.page_selection is not None:
                raise ValueError("use 'audit page N' or 'audit drawing.pdf --page N'")
            args.source = Path(args.source)
            # Preserve explicit historical invocations while bare audit is fresh.
            if args.source.is_file() and args.source.suffix.lower() == ".json":
                if args.page is not None or args.output is None:
                    raise ValueError("replay-audit requires --output and uses the frozen page scope")
                args.command = "replay-audit"
            else:
                if args.page is not None and args.page < 1:
                    raise ValueError("page must be a positive one-based source page number")
                sources = discover_pdfs(args.source)  # Fail before creating staging for empty input folders.
                if args.task == "auto":
                    args.task = resolve_audit_task(sources, args.page,
                        interactive=not args.worker and sys.stdin.isatty())
                if args.output is None:
                    args.output = Path.cwd() / "estimation-output" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        if args.command == "inspect":
            print(json.dumps(inspect_result(args.result, args.artifact, args.pointer), indent=2))
        elif args.worker:
            with redirect_stdout(sys.stderr):
                if args.command == "audit":
                    result = export_audits(args.source, args.output, page_number=args.page,
                                           task=args.task, estimation_profile=args.estimation_profile)
                elif args.command in {"replay-audit", "export"}:
                    project = FrozenProject(args.source.resolve().parent, read_json(args.source))
                    if args.command == "replay-audit":
                        if args.output.exists():
                            raise ValueError("audit output already exists")
                        render_project_audit(project, args.output)
                        result = args.output
                    else:
                        from src.drawing_engine.exports.estimation_project_export import export_optional
                        result = export_optional(project, args.output, format=args.format, artifact=args.artifact)
                elif args.command == "mep":
                    from src.drawing_engine.exports.estimation_project_export import export_mep_page
                    result = export_mep_page(args.source, args.output, database=args.database,
                        project_id=args.project, document_id=args.document, snapshot_id=args.snapshot, page_number=args.page)
                elif args.command == "cage":
                    from src.drawing_engine.exports.cage_assembly_export import export_cage
                    result = export_cage(args.source, args.output, assembly_mark=args.assembly,
                                         reference=args.reference, page_number=args.page, reference_page=args.reference_page)
                elif args.command == "detail":
                    from src.drawing_engine.exports.detail_assembly_export import export_detail
                    result = export_detail(args.source, args.output, assembly_mark=args.assembly,
                                           page_number=args.page, dxf_units=args.dxf_units)
                else:
                    result = export_comparison(args.source, args.output, frozen_bundle=args.frozen_bundle,
                                               estimation_profile=args.estimation_profile)
            print(json.dumps({"event": "result", "manifest": str(result),
                              "takeoff_completeness": "not_established"}))
            if args.command == "audit" and read_json(result)["execution_status"] != "succeeded":
                return 1
        else:
            from src.drawing_engine.operations.run_artifact_job import run
            from src.drawing_engine.operations.artifact_disk_usage import GIB
            if args.output.exists() or args.output.is_symlink():
                raise ValueError("output already exists; choose a new export directory")
            if args.reserve_gib < 0 or args.max_growth_gib < 0:
                raise ValueError("disk budgets must be non-negative")
            temporary = args.output.absolute().with_name(args.output.name + ".temporary")
            if temporary.exists():
                raise ValueError("temporary directory already exists; choose a new export directory")
            # Account even for staging-directory creation; all subprocess tempfiles
            # (including OCR libraries) use this explicit job-local scope.
            from src.drawing_engine.operations.artifact_disk_usage import artifact_disk_usage
            with artifact_disk_usage("estimation-cli-staging", [temporary]):
                temporary.mkdir(parents=True)
            command = [sys.executable, "-B", "-m", "src.drawing_engine.cli", args.command,
                       str(args.source.resolve()), "--output", str(args.output.absolute()),
                       "--worker"]
            if args.command == "cage":
                command += ["--assembly", args.assembly, "--page", str(args.page),
                            "--reference", str(args.reference.resolve()), "--reference-page", str(args.reference_page)]
            elif args.command == "detail":
                command += ["--assembly", args.assembly, "--page", str(args.page), "--dxf-units", args.dxf_units]
            elif args.command == "compare":
                command += ["--estimation-profile", str(args.estimation_profile.resolve())]
                if args.frozen_bundle:
                    command += ["--frozen-bundle", str(args.frozen_bundle.resolve())]
            elif args.command == "audit":
                command += ["--task", args.task, "--estimation-profile", str(args.estimation_profile.resolve())]
                if args.page is not None:
                    command += ["--page", str(args.page)]
            elif args.command == "mep":
                command += ["--database", str(args.database.resolve()), "--project", args.project,
                            "--document", args.document, "--page", str(args.page)]
                if args.snapshot:
                    command += ["--snapshot", args.snapshot]
            elif args.command == "export":
                command += ["--format", args.format]
                if args.artifact:
                    command += ["--artifact", args.artifact]
            run(command, outputs=[args.output], temporary=[temporary],
                reserve_bytes=int(args.reserve_gib * GIB), max_growth_bytes=int(args.max_growth_gib * GIB),
                required_outputs=[args.output if args.command == "replay-audit" else args.output / "result.json"],
                cwd=Path(__file__).resolve().parents[2], env={**os.environ, "TMPDIR": str(temporary)})
    except (KeyboardInterrupt, EOFError, OSError, ValueError, KeyError, IndexError, RuntimeError, subprocess.CalledProcessError) as exc:
        if isinstance(exc, KeyboardInterrupt) or (isinstance(exc, subprocess.CalledProcessError) and exc.returncode in (130, -2)):
            print(json.dumps({"event": "cancelled", "message": "Audit cancelled."}), file=sys.stderr)
            return 130
        print(json.dumps({"event": "error", "error_type": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
