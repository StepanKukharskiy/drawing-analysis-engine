#!/usr/bin/env python3
"""Build independent MEP page-cache shards and publish one ordered v3 store."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.drawing_engine.disciplines.mep.mep_v3_page_cache import MepV3PageCache
from src.drawing_engine.disciplines.mep.mep_native_descriptor_pack import RECORD
from src.drawing_engine.project.project_packed_store import PackedProjectStore


MIB = 1024 * 1024
DEFAULT_HEAVY_INDEX_BYTES = 256 * MIB


class GateFailure(RuntimeError):
    pass


def estimate_page_costs(registry, pages, *, cell_size=64,
                        heavy_index_bytes=DEFAULT_HEAVY_INDEX_BYTES):
    """Estimate cold I/O before extraction from frozen registry evidence."""
    selected = set(pages)
    estimates = []
    for page in registry['pages']:
        number = page['page_number']
        if number not in selected:
            continue
        metrics = page.get('quality_route', {}).get(
            'base_structural_quality_route', {}).get('metrics', {})
        primitives = int(metrics.get('native_path_paint_operation_count') or 0)
        width, height = page['page_size_display']
        spatial_cells = math.ceil(width / cell_size) * math.ceil(height / cell_size)
        descriptor_bytes = primitives * RECORD.size
        # Five 64-bit values are the packed cell staging upper bound. The
        # survivor-only index is normally smaller, so this deliberately errs
        # on the side of serial scheduling.
        spatial_index_bytes = primitives * 40
        expected_index_bytes = descriptor_bytes + spatial_index_bytes
        estimates.append({
            'page_number': number,
            'primitive_count': primitives,
            'spatial_cell_count': spatial_cells,
            'expected_descriptor_pack_bytes': descriptor_bytes,
            'expected_spatial_index_bytes': spatial_index_bytes,
            'expected_index_bytes': expected_index_bytes,
            'scheduling_class': ('disk_heavy' if expected_index_bytes >= heavy_index_bytes
                                 else 'lightweight'),
        })
    if len(estimates) != len(selected):
        raise GateFailure('registry omitted a requested page cost estimate')
    return sorted(estimates, key=lambda row: row['page_number'])


def scheduled_page_order(estimates):
    """Run heavy pages alone first; preserve page order within both lanes."""
    heavy = [row['page_number'] for row in estimates
             if row['scheduling_class'] == 'disk_heavy']
    light = [row['page_number'] for row in estimates
             if row['scheduling_class'] == 'lightweight']
    return heavy + light


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _rss_bytes(pid):
    if sys.platform == "darwin":
        class TaskInfo(ctypes.Structure):
            _fields_ = [
                ("virtual_size", ctypes.c_uint64),
                ("resident_size", ctypes.c_uint64),
                ("total_user", ctypes.c_uint64),
                ("total_system", ctypes.c_uint64),
                ("threads_user", ctypes.c_uint64),
                ("threads_system", ctypes.c_uint64),
                ("policy", ctypes.c_int32),
                ("faults", ctypes.c_int32),
                ("pageins", ctypes.c_int32),
                ("cow_faults", ctypes.c_int32),
                ("messages_sent", ctypes.c_int32),
                ("messages_received", ctypes.c_int32),
                ("syscalls_mach", ctypes.c_int32),
                ("syscalls_unix", ctypes.c_int32),
                ("context_switches", ctypes.c_int32),
                ("thread_count", ctypes.c_int32),
                ("running_threads", ctypes.c_int32),
                ("priority", ctypes.c_int32),
            ]
        info = TaskInfo()
        library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        size = library.proc_pidinfo(
            pid, 4, 0, ctypes.byref(info), ctypes.sizeof(info))
        return int(info.resident_size) if size == ctypes.sizeof(info) else 0
    statm = Path(f"/proc/{pid}/statm")
    try:
        resident_pages = int(statm.read_text().split()[1])
    except (FileNotFoundError, IndexError, ValueError):
        return 0
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _events(path):
    output = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            output.append(row)
    return output


def _stop(states):
    alive = [state for state in states.values() if state["process"].poll() is None]
    for state in alive:
        state["process"].send_signal(signal.SIGINT)
    deadline = time.monotonic() + 10
    while alive and time.monotonic() < deadline:
        alive = [state for state in alive if state["process"].poll() is None]
        time.sleep(.1)
    for state in alive:
        state["process"].terminate()
    deadline = time.monotonic() + 5
    while alive and time.monotonic() < deadline:
        alive = [state for state in alive if state["process"].poll() is None]
        time.sleep(.1)
    for state in alive:
        state["process"].kill()


def _run_pages(*, base_command, pages, workers, work_dir, shard_root,
               per_worker_limit, combined_limit, cost_by_page):
    pending = list(pages)
    running = {}
    completed = []
    combined_peak = 0

    def launch(number):
        prefix = work_dir / f"page-{number:03d}"
        stdout_path = prefix.with_suffix(".stdout.log")
        stderr_path = prefix.with_suffix(".stderr.log")
        shard = shard_root / f"page-{number:03d}.sqlite"
        output = shard_root / f"page-{number:03d}"
        stdout = stdout_path.open("wb")
        stderr = stderr_path.open("wb")
        command = [*base_command, "--output-dir", str(output), "--pages", str(number),
                   "--v3-page-cache-database", str(shard), "--resume"]
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        running[number] = {
            "page_number": number, "process": process, "command": command,
            "stdout_stream": stdout, "stderr_stream": stderr,
            "stdout": stdout_path, "stderr": stderr_path, "database": shard,
            "output": output, "started": time.monotonic(), "peak_rss_bytes": 0,
        }

    try:
        while pending or running:
            while pending and len(running) < workers:
                next_page = pending[0]
                next_is_heavy = cost_by_page[next_page].get(
                    'effective_scheduling_class', cost_by_page[next_page]['scheduling_class']) == 'disk_heavy'
                running_is_heavy = any(
                    cost_by_page[number].get(
                        'effective_scheduling_class', cost_by_page[number]['scheduling_class']) == 'disk_heavy'
                    for number in running)
                if running and (next_is_heavy or running_is_heavy):
                    break
                launch(pending.pop(0))
            rss = 0
            for state in running.values():
                current = _rss_bytes(state["process"].pid)
                state["peak_rss_bytes"] = max(state["peak_rss_bytes"], current)
                rss += current
                if current > per_worker_limit:
                    raise GateFailure(
                        f"page {state['page_number']} exceeded per-worker RSS limit: {current}")
            combined_peak = max(combined_peak, rss)
            if rss > combined_limit:
                raise GateFailure(f"workers exceeded combined RSS limit: {rss}")
            for number, state in list(running.items()):
                returncode = state["process"].poll()
                if returncode is None:
                    continue
                state["stdout_stream"].close()
                state["stderr_stream"].close()
                state["elapsed_seconds"] = round(time.monotonic() - state["started"], 6)
                state["returncode"] = returncode
                if returncode:
                    raise GateFailure(f"page {number} worker exited with {returncode}")
                events = _events(state["stdout"])
                compact = next((row for row in events
                                if row.get("phase") == "compact_native_index_complete"), None)
                record = next((row for row in reversed(events)
                               if row.get("page_number") == number
                               and "elapsed_seconds" in row), None)
                reused = next((row for row in events
                               if row.get("phase") == "verified_v3_page_cache_reused"), None)
                if compact is None and reused is None:
                    raise GateFailure(f"page {number} worker omitted phase evidence")
                performance_path = state["output"] / f"page-{number:03d}.performance.json"
                performance = json.loads(performance_path.read_text())
                state["reused"] = reused is not None
                state["phase_timings_seconds"] = {
                    "native_extraction": performance["native_descriptor_scan_seconds"],
                    "compact_native_index": (compact or {}).get("seconds"),
                    "page_computation": (record or {}).get("elapsed_seconds"),
                    "v3_page_transaction": performance["phases"]["v3_page_transaction"]["seconds"],
                    "worker_downstream": next(row["seconds"] for row in events
                        if row.get("phase") == "downstream_recomputation_complete"),
                }
                state["performance_profile"] = performance
                state["page_cache_key"] = performance["page_cache_key"]
                state["artifact_sha256"] = performance["page_cache_artifact_sha256"]
                completed.append(state)
                del running[number]
            time.sleep(.1)
    except BaseException:
        _stop(running)
        raise
    finally:
        for state in running.values():
            state["stdout_stream"].close()
            state["stderr_stream"].close()
    return sorted(completed, key=lambda row: row["page_number"]), combined_peak


def merge_page_caches(page_results, target):
    """Copy completed page artifacts in page order into one detached store."""
    target_store = PackedProjectStore(target, create=True)
    target_cache = MepV3PageCache(target_store)
    merged = []

    def canonical_records(source_cache, cache_key):
        for kind, payload in source_cache.iter_records(cache_key):
            if kind == "performance_profile":
                continue
            if kind == "page_record" and "elapsed_seconds" in payload:
                payload = {key: value for key, value in payload.items()
                           if key != "elapsed_seconds"}
            yield kind, payload

    try:
        for result in sorted(page_results, key=lambda row: row["page_number"]):
            source_store = PackedProjectStore(result["database"])
            source_cache = MepV3PageCache(source_store)
            try:
                manifest = source_cache.manifest(result["page_cache_key"])
                if manifest is None or manifest["artifact_sha256"] != result["artifact_sha256"]:
                    raise GateFailure(
                        f"page {result['page_number']} shard manifest changed before merge")
                written = target_cache.write(
                    cache_key=manifest["cache_key"], cache_inputs=manifest["cache_inputs"],
                    page_ref=manifest["page_ref"],
                    records=canonical_records(source_cache, manifest["cache_key"]))
                merged.append({
                    "page_number": result["page_number"],
                    "page_cache_key": manifest["cache_key"],
                    "source_artifact_sha256": manifest["artifact_sha256"],
                    "artifact_sha256": written["artifact_sha256"],
                })
            finally:
                source_store.connection.close()
        quick_check = target_store.connection.execute("PRAGMA quick_check").fetchone()[0]
        if quick_check != "ok":
            raise GateFailure("merged page cache failed SQLite quick_check")
    finally:
        target_store.connection.close()
    return merged


def _run_downstream(command, stdout_path, stderr_path, memory_limit, expected_pages):
    started = time.monotonic()
    peak = 0
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        while process.poll() is None:
            peak = max(peak, _rss_bytes(process.pid))
            if peak > memory_limit:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise GateFailure(f"downstream process exceeded RSS limit: {peak}")
            time.sleep(.1)
    if process.returncode:
        raise GateFailure(f"downstream process exited with {process.returncode}")
    events = _events(stdout_path)
    if not events or not all(any(row.get("phase") == "verified_v3_page_cache_reused"
                                    and row.get("page") == page for row in events)
                             for page in expected_pages):
        raise GateFailure("downstream replay omitted cache-reuse evidence")
    phase = next((row for row in events
                  if row.get("phase") == "downstream_recomputation_complete"), None)
    if phase is None:
        raise GateFailure("downstream replay omitted phase timing")
    return {
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "interpretation_seconds": phase["seconds"],
        "peak_rss_bytes": peak,
        "stdout": str(stdout_path), "stderr": str(stderr_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "registry", "native-text", "annotations", "output-dir",
                "database", "report", "work-dir"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--ocr", type=Path)
    parser.add_argument("--pages", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--region-size", type=float, default=64)
    parser.add_argument("--region-budget", type=int, default=4000)
    parser.add_argument("--per-worker-memory-limit-mib", type=int, default=1024)
    parser.add_argument("--combined-memory-limit-mib", type=int, default=2560)
    parser.add_argument("--force-concurrent-heavy", action="store_true",
                        help="Diagnostic only: bypass disk-heavy serialization")
    args = parser.parse_args()
    if args.workers != 2:
        parser.error("the production candidate is deliberately fixed at two workers")
    registry = json.loads(args.registry.read_text())
    pages = args.pages or [row["page_number"] for row in registry["pages"]]
    if (not pages or len(set(pages)) != len(pages)
            or any(number < 1 or number > registry["document"]["page_count"] for number in pages)):
        parser.error("invalid or duplicate page execution scope")
    estimates = estimate_page_costs(registry, pages, cell_size=args.region_size)
    if args.force_concurrent_heavy:
        for row in estimates:
            row['effective_scheduling_class'] = 'lightweight'
    cost_by_page = {row['page_number']: row for row in estimates}
    execution_order = (sorted(pages) if args.force_concurrent_heavy
                       else scheduled_page_order(estimates))
    partial_database = Path(str(args.database) + ".partial")
    partial_output = args.output_dir.with_name(args.output_dir.name + ".partial")
    for path in (args.database, args.output_dir):
        if path.exists():
            parser.error(f"publication target already exists: {path}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    shard_root = args.work_dir / "detached-shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "0.1.0", "layer": "mep_parallel_page_gate",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "pages": sorted(pages), "worker_count": args.workers,
        "page_cost_estimates": estimates,
        "execution_order": execution_order,
        "scheduling_mode": ("forced_concurrent_diagnostic" if args.force_concurrent_heavy
                            else "resource_aware"),
        "limits": {
            "per_worker_rss_bytes": args.per_worker_memory_limit_mib * MIB,
            "combined_rss_bytes": args.combined_memory_limit_mib * MIB,
        },
        "published": False,
    }
    base_command = [
        sys.executable, "-B", str(ROOT / "src/drawing_engine/pipelines/generate_mep_automatic_items.py"),
        "--source", str(args.source), "--registry", str(args.registry),
        "--native-text", str(args.native_text), "--annotations", str(args.annotations),
        "--envelope-engine", "page_topology",
    ]
    # Omit defaults so the child receives the generator's canonical parameter
    # types. In particular, argparse otherwise turns the default integer 64
    # into 64.0 and creates a different content-addressed cache identity.
    if args.region_size != 64:
        base_command.extend(("--region-size", str(args.region_size)))
    if args.region_budget != 4000:
        base_command.extend(("--region-budget", str(args.region_budget)))
    if args.ocr is not None:
        base_command.extend(("--ocr", str(args.ocr)))
    started = time.monotonic()
    try:
        results, combined_peak = _run_pages(
            base_command=base_command, pages=execution_order, workers=args.workers,
            work_dir=args.work_dir, shard_root=shard_root,
            per_worker_limit=args.per_worker_memory_limit_mib * MIB,
            combined_limit=args.combined_memory_limit_mib * MIB,
            cost_by_page=cost_by_page)
        report["workers"] = [{key: value for key, value in result.items() if key in {
                "page_number", "command", "stdout", "stderr", "elapsed_seconds",
                "returncode", "peak_rss_bytes", "phase_timings_seconds",
                "page_cache_key", "artifact_sha256", "performance_profile", "reused"}}
                for result in results]
        for row in report["workers"]:
            for key in ("stdout", "stderr"):
                row[key] = str(row[key])
        report["combined_worker_peak_rss_bytes"] = combined_peak
        merge_started = time.monotonic()
        reuse_partial = False
        if partial_database.exists():
            with sqlite3.connect(partial_database) as connection:
                rows = connection.execute(
                    "SELECT key,value_json FROM pilot_manifest WHERE key LIKE 'mep-page-cache:%'").fetchall()
            staged = {key.split(":", 1)[1]: json.loads(value)["artifact_sha256"]
                      for key, value in rows}
            expected = {row["page_cache_key"]: row["artifact_sha256"] for row in results}
            reuse_partial = staged == expected
            if not reuse_partial:
                partial_database.unlink()
        if reuse_partial:
            report["ordered_publication"] = [{
                "page_number": row["page_number"],
                "page_cache_key": row["page_cache_key"],
                "artifact_sha256": row["artifact_sha256"]} for row in results]
        else:
            report["ordered_publication"] = merge_page_caches(results, partial_database)
        report["partial_publication_reused"] = reuse_partial
        report["v3_publication_seconds"] = round(time.monotonic() - merge_started, 6)
        downstream_command = [*base_command, "--output-dir", str(partial_output),
                              "--pages", *(str(number) for number in sorted(pages)),
                              "--v3-page-cache-database", str(partial_database), "--resume"]
        report["downstream"] = _run_downstream(
            downstream_command, args.work_dir / "downstream.stdout.log",
            args.work_dir / "downstream.stderr.log",
            args.combined_memory_limit_mib * MIB, sorted(pages))
        with sqlite3.connect(partial_database) as connection:
            report["quick_check"] = connection.execute("PRAGMA quick_check").fetchone()[0]
            report["page_manifest_count"] = connection.execute(
                "SELECT count(*) FROM pilot_manifest WHERE key LIKE 'mep-page-cache:%'").fetchone()[0]
        if report["quick_check"] != "ok" or report["page_manifest_count"] != len(pages):
            raise GateFailure("staged publication failed final integrity checks")
        os.replace(partial_database, args.database)
        os.replace(partial_output, args.output_dir)
        report["published"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["completed_shards_retained_for_resume"] = True
        report["partial_publication_retained_for_resume"] = partial_database.exists()
        report["elapsed_seconds"] = round(time.monotonic() - started, 6)
        _atomic_json(args.report, report)
        raise
    report["elapsed_seconds"] = round(time.monotonic() - started, 6)
    report["database_bytes"] = args.database.stat().st_size
    report["passed"] = True
    _atomic_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
