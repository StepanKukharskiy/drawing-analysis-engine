"""Operational disk accounting, deliberately outside canonical evidence.

Allocation is st_blocks * 512, not exclusive APFS/compressed/clone ownership.
Free-space changes belong to the filesystem, not necessarily this process.
Never delete outputs or infer engineering/publication authority here.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import time
import uuid


GIB = 1024 ** 3
DEFAULT_MIN_FREE_BYTES = 3 * GIB
DEFAULT_LEDGER = Path(__file__).resolve().parents[3] / "data/operations/artifact-disk-usage.jsonl"


class DiskBudgetError(RuntimeError):
    pass


def _existing_parent(path):
    path = Path(path).absolute()
    while not path.exists():
        if path.parent == path:
            raise FileNotFoundError(path)
        path = path.parent
    return path if path.is_dir() else path.parent


def _roots(paths):
    """Remove overlapping roots; do not follow directory symlinks."""
    result = []
    for path in sorted({Path(p).absolute() for p in paths}, key=lambda p: (len(p.parts), str(p))):
        if not any(path == parent or parent in path.parents for parent in result):
            result.append(path)
    return result


def measure_paths(paths, *, exclude=()):
    excluded = {Path(p).absolute() for p in exclude}
    seen = set()
    total = {"logical_bytes": 0, "allocated_bytes": 0, "file_count": 0,
             "symlinks_skipped": 0, "errors": []}

    def visit(path):
        if path in excluded:
            return
        try:
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                total["symlinks_skipped"] += 1
                return
            if stat.S_ISDIR(info.st_mode):
                with os.scandir(path) as entries:
                    for entry in entries:
                        visit(Path(entry.path))
            elif stat.S_ISREG(info.st_mode):
                key = (info.st_dev, info.st_ino)
                if key not in seen:
                    seen.add(key)
                    total["logical_bytes"] += info.st_size
                    total["allocated_bytes"] += info.st_blocks * 512
                    total["file_count"] += 1
        except FileNotFoundError:
            pass  # Not created yet, or a producer's temporary file was removed.
        except OSError as exc:
            total["errors"].append({"path": str(path), "error": str(exc)})

    for path in _roots(paths):
        visit(path)
    total["complete"] = not total["errors"]
    return total


def free_space(paths):
    filesystems = {}
    for path in paths:
        probe = _existing_parent(path)
        device = str(probe.stat().st_dev)
        if device not in filesystems:
            usage = shutil.disk_usage(probe)
            filesystems[device] = {"probe_path": str(probe), "free_bytes": usage.free,
                                   "total_bytes": usage.total}
    return filesystems


def check_capacity(paths, *, min_free_bytes=DEFAULT_MIN_FREE_BYTES, reserve_bytes=0):
    if min_free_bytes < 0 or reserve_bytes < 0:
        raise ValueError("disk limits must be non-negative")
    snapshot = free_space(paths)
    required = min_free_bytes + reserve_bytes
    for row in snapshot.values():
        if row["free_bytes"] < required:
            raise DiskBudgetError(
                f"disk preflight: {row['probe_path']} has {row['free_bytes'] / GIB:.2f} GiB free; "
                f"requires {required / GIB:.2f} GiB including reserve; no automatic cleanup")
    return snapshot


def append_event(ledger, record):
    ledger = Path(ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()


@contextmanager
def artifact_disk_usage(label, paths, *, temporary_paths=(), ledger=None,
                        min_free_bytes=DEFAULT_MIN_FREE_BYTES, reserve_bytes=0):
    """Record before/after sizes even on failure. Scope directories explicitly.

    This is a preflight and accounting boundary, not a disk reservation/quota.
    Long-running callers must also call check_capacity at bounded write points.
    Records from nested operations overlap and must not be summed.
    """
    paths, temporary_paths = _roots(paths), _roots(temporary_paths)
    if not paths:
        raise ValueError("at least one output path is required")
    if any(path.is_symlink() for path in [*paths, *temporary_paths]):
        raise DiskBudgetError("output/staging roots must not be symlinks; name the actual artifact path")
    if any(a == b or a in b.parents or b in a.parents
           for a in paths for b in temporary_paths):
        raise ValueError("output and temporary scopes must not overlap")
    ledger = Path(ledger or os.environ.get("REBAR_DISK_USAGE_LOG", DEFAULT_LEDGER)).absolute()
    watched = [*paths, *temporary_paths, ledger]
    operation = uuid.uuid4().hex
    started = time.monotonic()
    before = measure_paths(paths, exclude=[ledger])
    temp_before = measure_paths(temporary_paths, exclude=[ledger])
    disk_before = free_space(watched)
    event = {"schema_version": "artifact_disk_usage.v1", "operation_id": operation,
             "label": label, "time_utc": datetime.now(timezone.utc).isoformat(),
             "output_paths": list(map(str, paths)), "temporary_paths": list(map(str, temporary_paths)),
             "min_free_bytes": min_free_bytes, "reserve_bytes": reserve_bytes,
             "before": before, "temporary_before": temp_before, "filesystems_before": disk_before}
    # Fail before producing data if the operational ledger cannot be opened.
    append_event(ledger, {**event, "phase": "started"})
    error = None
    try:
        if not before["complete"] or not temp_before["complete"]:
            raise DiskBudgetError("cannot measure output/temporary scope before writing")
        check_capacity(watched, min_free_bytes=min_free_bytes, reserve_bytes=reserve_bytes)
        yield
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        after = measure_paths(paths, exclude=[ledger])
        temp_after = measure_paths(temporary_paths, exclude=[ledger])
        try:
            disk_after = free_space(watched)
            delta = {key: after[key] - before[key]
                     for key in ("logical_bytes", "allocated_bytes", "file_count")}
            record = {**event, "phase": "finished", "status": "failed" if error else "completed",
                      "error_type": error, "after": after, "delta": delta,
                      "temporary_after": temp_after, "filesystems_after": disk_after,
                      "temporary_scope": "explicit_paths_only" if temporary_paths else "not_measured",
                      "elapsed_seconds": time.monotonic() - started,
                      "filesystem_free_change_bytes": {
                          key: row["free_bytes"] - disk_before[key]["free_bytes"]
                          for key, row in disk_after.items() if key in disk_before},
                      "below_free_floor": any(r["free_bytes"] < min_free_bytes for r in disk_after.values()),
                      "complete_measurement": after["complete"] and temp_after["complete"],
                      "automatic_cleanup": False, "engineering_authority": False}
            append_event(ledger, record)
            print(f"[disk] {label}: outputs {after['logical_bytes'] / GIB:.3f} GiB logical, "
                  f"{after['allocated_bytes'] / GIB:.3f} GiB allocated "
                  f"(change {delta['allocated_bytes'] / GIB:+.3f} GiB); "
                  f"temporary retained {str(round(temp_after['allocated_bytes'] / GIB, 3)) + ' GiB' if temporary_paths else 'not measured'}; "
                  f"free >= {min(r['free_bytes'] for r in disk_after.values()) / GIB:.2f} GiB; "
                  f"{record['status']}", file=sys.stderr)
        except OSError as exc:
            # Never mask the original producer failure or undo published data.
            print(f"[disk] ACCOUNTING FAILED for {label}: {exc}", file=sys.stderr)
