"""Standard deliveries over the existing schema-v3 compressed artifact store.

Native bytes are stored once, without envelopes or duplicated node payloads.
These snapshots offer artifact/JSON-pointer inspection, not the MEP store's
separately built record/edge search indexes.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import wraps
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import zlib

from src.drawing_engine.project.analysis_job_manifest import sha256_file
from src.drawing_engine.project.project_packed_store import PackedProjectStore, CHUNK_BYTES


def artifact_record(output, path):
    path = Path(path)
    return {"path": path.relative_to(output).as_posix(),
            "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def checked_path(root, record):
    path = (root / record["path"]).resolve(strict=True)
    if not path.is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes export")
    if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
        raise ValueError("artifact content hash mismatch")
    return path


def add_project_database(output, manifest, *, staging=None):
    """Import exact frozen bytes using v3's verified compressed chunk format."""
    output = Path(output)
    staging = Path(staging or output)
    records = {}
    for name, record in manifest["artifacts"].items():
        path = checked_path(staging, record)
        if path.suffix == ".json":
            records[name] = record
    package = {**manifest, "artifacts": dict(manifest["artifacts"])}
    index_raw = (json.dumps(package, indent=2, ensure_ascii=True) + "\n").encode()
    digest = manifest["source_sha256"]
    context = {"mode": manifest["mode"], "contract": manifest["contract"],
               "index_scope": "artifact_and_json_pointer", "assembly_mark": manifest.get("assembly_mark"),
               "page": manifest.get("page")}
    hashes = {name: record["sha256"] for name, record in records.items()}
    hashes["package_index"] = hashlib.sha256(index_raw).hexdigest()
    snapshot_id = hashlib.sha256(json.dumps([digest, context, hashes], sort_keys=True).encode()).hexdigest()
    database = output / "project.sqlite"
    with PackedProjectStore(database, create=True) as store:
        connection = store.connection
        with connection:
            cursor = connection.execute("INSERT INTO snapshots(id,project_id,document_id,source_sha256,context_json,manifest_json,created_at) VALUES(?,?,?,?,?,?,?)",
                (bytes.fromhex(snapshot_id), "estimation-cli", f"pdf-sha256:{digest}", bytes.fromhex(digest),
                 json.dumps(context), json.dumps(hashes), datetime.now(timezone.utc).isoformat()))
            snapshot_key = cursor.lastrowid
            for name, sha in hashes.items():
                existing = connection.execute("SELECT artifact_key FROM artifacts WHERE sha256=?", (bytes.fromhex(sha),)).fetchone()
                if existing:
                    key = existing[0]
                else:
                    length = len(index_raw) if name == "package_index" else records[name]["bytes"]
                    key = connection.execute("INSERT INTO artifacts(sha256,byte_count,chunk_count) VALUES(?,?,?)",
                        (bytes.fromhex(sha), length, (length + CHUNK_BYTES - 1) // CHUNK_BYTES)).lastrowid
                    with (BytesIO(index_raw) if name == "package_index" else (staging / records[name]["path"]).open("rb")) as source:
                        hasher, offset, ordinal = hashlib.sha256(), 0, 0
                        for raw in iter(lambda: source.read(CHUNK_BYTES), b""):
                            hasher.update(raw)
                            connection.execute("INSERT INTO artifact_chunks VALUES(?,?,?,?,?,?)",
                                (key, ordinal, offset, len(raw), hashlib.sha256(raw).digest(), zlib.compress(raw, 1)))
                            offset += len(raw)
                            ordinal += 1
                        if hasher.hexdigest() != sha or offset != length:
                            raise ValueError("native artifact changed during project import")
                connection.execute("INSERT INTO snapshot_artifacts VALUES(?,?,?)", (snapshot_key, name, key))
            connection.execute("INSERT INTO active_documents VALUES(?,?,?)", ("estimation-cli", f"pdf-sha256:{digest}", snapshot_key))
        if not store.quick_check()["passed"]:
            raise ValueError("project database integrity check failed")
    manifest["project"] = {"schema_version": 3, "project_id": "estimation-cli", "document_id": f"pdf-sha256:{digest}",
                           "snapshot_id": snapshot_id, "index_scope": "artifact_and_json_pointer"}
    manifest["records"] = {name: {"sha256": sha} for name, sha in hashes.items()}
    manifest["artifacts"]["project_database"] = artifact_record(output, database)


class FrozenProject:
    """Portable read-only resolver pinned to the published database and snapshot."""
    def __init__(self, root, manifest):
        self.root, self.manifest = Path(root), manifest
        self.database = checked_path(self.root, manifest["artifacts"]["project_database"])
        self.scope = {key: manifest["project"][key] for key in ("project_id", "document_id", "snapshot_id")}
        with PackedProjectStore(self.database, readonly=True) as store:
            snapshot = store.snapshot(**self.scope)
            if snapshot["source_sha256"] != manifest["source_sha256"]:
                raise ValueError("project source binding mismatch")
            if snapshot["manifest"] != {key: value["sha256"] for key, value in manifest["records"].items()}:
                raise ValueError("project artifact index mismatch")
            if any(snapshot['context'].get(key) != manifest.get(key) for key in ('mode', 'contract', 'page', 'assembly_mark')):
                raise ValueError('project scope differs from frozen snapshot')

    def raw(self, name):
        with PackedProjectStore(self.database, readonly=True) as store:
            return b"".join(store.iter_artifact_bytes(**self.scope, name=name))

    def load(self, name):
        return json.loads(self.raw(name))

    def artifact(self, name):
        raw = self.raw(name)
        return json.loads(raw), raw

    def source(self):
        source = checked_path(self.root, self.manifest["artifacts"]["source_pdf"])
        if sha256_file(source) != self.manifest["source_sha256"]:
            raise ValueError("source PDF binding mismatch")
        return source


def render_project_audit(project, output):
    """All interpretation and declarations come from frozen SQLite records."""
    source = project.source()
    mode = project.load("package_index")["mode"]
    if mode == "native_cage_assembly":
        from src.drawing_engine.audit.detail_audit_pdf import write_cage_audit
        write_cage_audit(project, output)
        return {"mode": mode, "source_page_numbers": [project.manifest["page"]], "reference_scope": "explicit supplied reference page"}
    if mode == "fresh_mep_analysis":
        from src.drawing_engine.audit.render_mep_partial_audit import build_automatic_manifest, render_overlay
        inputs = {key: project.load(name) for key, name in {
            "catalog": "item-catalog", "geometry": "bounded-local-3d",
            "composites": "outlined-route-composites", "annotations": "annotations",
            "discovery": "automatic-discovery", "bindings": "attribute-bindings",
            "terminology": "terminology-proposals"}.items()}
        import fitz
        # Image backgrounds keep dense native plans usable in desktop previews.
        with TemporaryDirectory(prefix="mep-audit-") as cache:
            report = render_overlay(source, output, build_automatic_manifest(inputs, compact_regions=True),
                                    page_cache_dir=Path(cache), route_graph=project.load("route-observations"))
        with fitz.open(output) as pdf:
            page_count = len(pdf)
        return {"mode": mode, "source_page_numbers": project.manifest["contract"]["source_page_numbers"],
                "page_count": page_count, "source_rendering": report["presentation"]["source_rendering"],
                "source_opacity": report["presentation"]["source_opacity"], "interactive_links": False,
                "interpretation": "bounded_native_discovery"}
    if mode == "native_detail_assembly":
        import fitz
        from src.drawing_engine.audit.detail_audit_pdf import write_detail_audit
        write_detail_audit(source, output, project.load("assembly"), project.load("comparison"),
                           fitz.Rect(project.load("audit_scope")["crop"]),
                           project.load("dxf_report"), project.load("evidence_store"))
        return {"mode": mode, "source_page_numbers": [project.manifest["page"]]}
    if mode == "mep_snapshot_page":
        from src.drawing_engine.audit.render_mep_network_drawing_audit import render_network_drawing_audit
        report = render_network_drawing_audit(source, output, project.load("mep_review"),
            preview_pages=[project.manifest["page"]], maximum_closeups=0,
            include_isolated_geometry=True, single_page=True, write_manifest=False, source_opacity=.25)
        return {"mode": mode, "source_page_numbers": [project.manifest["page"]], "page_count": report['page_count'],
                "source_opacity": .25, "system_palette": "hot_supply_red_return_orange_chilled_supply_blue_return_cyan",
                "coverage": {key: value for key, value in report["coverage"].items() if not isinstance(value, (list, dict))}}
    from src.drawing_engine.audit.render_object_agnostic_audit import build_audit
    build_audit(source, output, language="en", frozen_project=project, write_manifest=False)
    return {"mode": mode, "page_structure": "source_interpretation_solid_status",
            "source_page_numbers": [p["page"] for p in project.load("comparison")["pages"]]}


def standard_delivery(producer):
    """Keep legacy runner serialization job-local; publish only standard files."""
    @wraps(producer)
    def deliver(source, output, **kwargs):
        output = Path(output).absolute()
        if output.exists() or output.is_symlink():
            raise ValueError("output already exists; choose a new export directory")
        output = output.resolve()
        with TemporaryDirectory(prefix="delivery-") as temporary:
            staging = Path(temporary) / "native"
            result = producer(source, staging, **kwargs)
            manifest = json.loads(result.read_text())
            output.mkdir(parents=True, exist_ok=False)
            add_project_database(output, manifest, staging=staging)
            retained = {"project_database": manifest["artifacts"]["project_database"]}
            for name, record in manifest["artifacts"].items():
                if name == "project_database" or Path(record["path"]).suffix.lower() not in {".pdf", ".dxf"}:
                    continue
                original = checked_path(staging, record)
                target = output / record["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, target)
                retained[name] = artifact_record(output, target)
            manifest["artifacts"] = retained
            project = FrozenProject(output, manifest)
            manifest["audit"] = render_project_audit(project, output / "audit.pdf")
            retained["audit_pdf"] = artifact_record(output, output / "audit.pdf")
            if artifact_record(output, project.database) != retained["project_database"]:
                raise ValueError("audit mutated frozen database")
            from src.drawing_engine.cli import write_json
            write_json(output / "result.json", manifest)
        return output / "result.json"
    return deliver


@standard_delivery
def export_fresh_mep(source, output, *, page_number=None, task="mep"):
    """Fresh M0/M1/native text and the existing bounded MEP/HVAC engines."""
    from src.drawing_engine.cli import write_json, progress
    from src.drawing_engine.disciplines.mep.mep_sheet_registry import extract_mep_sheet_registry
    from src.drawing_engine.disciplines.mep.mep_text_observations import extract_mep_text_observations
    from src.drawing_engine.disciplines.mep.mep_annotation_observations import extract_pdf_annotation_observations
    from src.drawing_engine.disciplines.mep.mep_hvac_inventory import build_mep_hvac_inventory
    from src.drawing_engine.pipelines.generate_mep_automatic_items import run
    import fitz

    source, output = Path(source).resolve(strict=True), Path(output)
    with fitz.open(source) as pdf:
        pages = list(range(1, len(pdf) + 1)) if page_number is None else [page_number]
        if not pages or any(type(n) is not int or not 1 <= n <= len(pdf) for n in pages):
            raise ValueError("selected page is outside the source PDF")
    output.mkdir(parents=True, exist_ok=False)
    (output / "source").mkdir()
    local_source = output / "source" / source.name
    shutil.copyfile(source, local_source)
    progress({"stage": "mep_source_registration"})
    registry = extract_mep_sheet_registry(local_source, page_numbers=pages)
    native = extract_mep_text_observations(pdf_path=local_source, sheet_registry=registry)
    annotations = extract_pdf_annotation_observations(local_source, page_numbers=pages)
    for name, payload in (("sheet-registry", registry), ("native-text", native), ("annotations", annotations)):
        write_json(output / (name + ".json"), payload)
    progress({"stage": "mep_bounded_native_discovery"})
    run(source=local_source, registry=registry, native_text=native, annotations=annotations,
        output_dir=output / "discovery", page_numbers=pages, export_json=True,
        envelope_engine="page_topology", native_scratch_dir=output / "native-index")
    from src.drawing_engine.cli import read_json
    terminology = read_json(output / "discovery/terminology-proposals.json")
    bindings = read_json(output / "discovery/attribute-bindings.json")
    write_json(output / "hvac-inventory.json", build_mep_hvac_inventory(
        terminology=terminology, attribute_bindings=bindings))
    paths = {p.stem: p for p in sorted(output.rglob("*.json"))}
    paths["source_pdf"] = local_source
    manifest = {"schema_version": "estimation_cli.v1", "execution_status": "succeeded",
        "mode": "fresh_mep_analysis", "source_sha256": sha256_file(local_source),
        "contract": {"requested_task": task, "executed_tasks": ["mep", "hvac"],
            "source_page_numbers": pages, "page_scope": "selected_pages" if page_number else "all_pages",
            "takeoff_completeness": "not_established", "approval_granted_by_execution": False,
            "source_registration": "fresh", "historical_inputs_used": False,
            "interpretation": "existing_bounded_native_discovery",
            "exhaustive_source_denominator_route_certification": "not_established_by_this_runner",
            "physical_network_completion": "not_established",
            "installed_length": "not_established", "fitting_count": "not_established",
            "declared_comparison": "not_run", "dxf": "no_eligible_fabrication_outline_in_this_route_delivery",
            "full_search_evidence": "stored_native_json_artifacts"},
        "artifacts": {key: artifact_record(output, value) for key, value in paths.items()}}
    write_json(output / "result.json", manifest)
    return output / "result.json"


@standard_delivery
def export_mep_page(source, output, *, database, project_id, document_id, page_number, snapshot_id=None):
    """Package existing MEP records; no discovery, binding or quantity engine runs."""
    from src.drawing_engine.project.mep_project_review import load_review
    from src.drawing_engine.cli import write_json
    source, output = Path(source).resolve(strict=True), Path(output)
    with PackedProjectStore(database, readonly=True) as store:
        snapshot = store.snapshot(project_id=project_id, document_id=document_id, snapshot_id=snapshot_id)
        if snapshot['source_sha256'] != sha256_file(source):
            raise ValueError("MEP source PDF differs from frozen project")
        scope = dict(project_id=project_id, document_id=document_id, snapshot_id=snapshot['id'])
        review = load_review(database, project=project_id, document=document_id, snapshot=snapshot['id'])
        if page_number not in {p['page_number'] for p in review['pages'] if p['role'] != 'divider'}:
            raise ValueError("page outside frozen MEP drawing registry")
        output.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(source, output / source.name)
        write_json(output / 'mep-review.json', review)
        paths = {'source_pdf': output / source.name, 'mep_review': output / 'mep-review.json'}
        # Exact native records behind the review surface. Very large discovery
        # query packs stay in the upstream project and are explicitly out of scope.
        names = ('sheet-registry', 'network-hierarchy', 'hvac-inventory', 'item-catalog',
                 'cross-sheet-runs', 'attribute-bindings', 'projected-identity-bindings',
                 'review-outcomes', 'route-observations', 'bounded-local-3d')
        available = set(store.artifact_names(**scope))
        for name in names:
            if name not in available:
                continue
            path = output / (name + '.json')
            with path.open('wb') as stream:
                for chunk in store.iter_artifact_bytes(**scope, name=name):
                    stream.write(chunk)
            paths[name] = path
        write_json(output / 'upstream-snapshot.json', {
            'snapshot_id': snapshot['id'], 'source_sha256': snapshot['source_sha256'],
            'project_id': project_id, 'document_id': document_id,
            'artifact_manifest': snapshot['manifest'], 'context': snapshot['context'],
            'retained_native_artifacts': sorted(set(names) & available),
            'upstream_only_artifacts': sorted(available - set(names)),
            'deep_discovery_queries_require_upstream_project': True})
        paths['upstream_snapshot'] = output / 'upstream-snapshot.json'
    manifest = {'schema_version': 'estimation_cli.v1', 'execution_status': 'succeeded',
        'mode': 'mep_snapshot_page', 'page': page_number, 'source_sha256': sha256_file(source),
        'contract': {'takeoff_completeness': 'not_established', 'approval_granted_by_execution': False,
                     'audit_scope': 'one_source_page', 'interpretation': 'existing_frozen_snapshot',
                     'dxf': 'no_eligible_part_outline_in_this_network_review',
                     'deep_discovery_queries': 'upstream_project_only'},
        'artifacts': {key: artifact_record(output, value) for key, value in paths.items()}}
    write_json(output / 'result.json', manifest)
    return output / 'result.json'


def export_optional(project, output, *, format='json', artifact=None):
    """Explicit compatibility export; SQLite remains the primary record store."""
    output = Path(output)
    if format == 'html' and project.load('package_index')['mode'] != 'native_detail_assembly':
        raise ValueError('HTML export currently supports the detail review; use the interpretation PDF for this mode')
    names = [artifact] if artifact else list(project.manifest['records'])
    if format == 'html' and artifact:
        raise ValueError('HTML exports the complete detail review; omit --artifact')
    # Resolve keys before creating output; never use a native key as a path.
    if not set(names) <= project.manifest['records'].keys():
        raise ValueError('unknown frozen artifact')
    output.mkdir(parents=True, exist_ok=False)
    paths = {}
    for index, name in enumerate(names):
        path = output / f'record-{index + 1:03d}.json'
        path.write_bytes(project.raw(name))
        paths[name] = artifact_record(output, path)
    if format == 'html':
        import fitz
        from src.drawing_engine.exports.detail_assembly_export import write_report
        assembly, declarations = project.load('assembly'), project.load('declarations')
        comparison, evidence = project.load('comparison'), project.load('evidence_store')
        crop = fitz.Rect(project.load('audit_scope')['crop'])
        shutil.copyfile(project.source(), output / 'source.pdf')
        with fitz.open(project.source()) as doc:
            page = doc[assembly['page'] - 1]
            page.remove_rotation()
            page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=crop, alpha=False).save(output / 'drawing.png')
        # The existing HTML renderer links these conventional names.
        for name, filename in [('assembly','assembly.json'),('evidence_store','evidence.json'),('comparison','comparison.json')]:
            (output / filename).write_bytes(project.raw(name))
        write_report(output, assembly, declarations, comparison, crop, evidence)
    from src.drawing_engine.cli import write_json
    write_json(output / 'result.json', {'source_snapshot': project.scope, 'format': format, 'artifacts': paths})
    return output / 'result.json'
