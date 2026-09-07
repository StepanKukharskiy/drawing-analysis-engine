"""Verify an engine repository and run its self-contained development gates."""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy.source.stage_source import read_manifest
from src.drawing_engine.operations.run_artifact_job import run


def check_index(root=ROOT):
    root = Path(root)
    manifest, names = read_manifest(root / "deploy/source/engine-manifest.json")
    aliases = manifest.get("aliases", {})
    expected = set(names) | set(aliases)
    def git(*args, **kwargs):
        return subprocess.check_output(["git", "-C", str(root), *args], **kwargs)
    entries = {}
    for row in git("ls-files", "--stage", "-z").split(b"\0"):
        if not row:
            continue
        header, path = row.split(b"\t", 1)
        mode, oid, stage = header.decode().split()
        name = path.decode()
        if stage != "0" or name in entries:
            raise ValueError("unmerged or duplicate index entry: " + name)
        entries[name] = (mode, oid)
    if set(entries) != expected:
        raise ValueError("index differs from engine manifest: " + json.dumps({
            "missing": sorted(expected - set(entries)), "extra": sorted(set(entries) - expected)}))
    for name in sorted(expected):
        path = root / name
        if name in aliases:
            if not path.is_symlink() or path.readlink().as_posix() != aliases[name]:
                raise ValueError("launcher alias differs: " + name)
            data, mode = aliases[name].encode(), "120000"
        else:
            if path.is_symlink() or not path.is_file():
                raise ValueError("missing or linked source: " + name)
            data = path.read_bytes()
            mode = "100755" if path.stat().st_mode & 0o111 else "100644"
        oid = git("hash-object", "--stdin", input=data).decode().strip()
        if entries[name] != (mode, oid):
            raise ValueError("index bytes/mode differ from checked source: " + name)
    untracked = git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
    if any(untracked):
        raise ValueError("unreviewed untracked files: " + repr([p.decode() for p in untracked if p]))
    return {"status": "passed", "index_files": len(expected)}


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def engine_tests(root=ROOT):
    manifest, _ = read_manifest(root / "deploy/source/engine-manifest.json")
    gates = json.loads((root / "config/test_gates.json").read_text())["gates"]
    excluded = [p for gate, spec in gates.items() if gate not in {"fast", "full-corpus"}
                for p in spec.get("patterns", [])]
    excluded += manifest.get("private_test_selectors", [])
    included = manifest.get("synthetic_test_selectors", [])
    def matches(name, patterns):
        return any(fnmatch.fnmatchcase(name, p) for p in patterns)
    loader = unittest.TestLoader()
    sys.path.insert(0, str(root / "tests"))
    tests = []
    for name in manifest["groups"]["tests"]:
        if not Path(name).name.startswith("test_"):
            continue
        for test in flatten(loader.loadTestsFromName(Path(name).stem)):
            if matches(test.id(), included) or not matches(test.id(), excluded):
                tests.append(test)
    if loader.errors:
        raise ValueError("engine test discovery failed:\n" + "\n".join(loader.errors))
    if not tests:
        raise ValueError("empty engine test gate")
    return tests


def smoke(scratch):
    import fitz
    source = scratch / "drawing.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page(width=1000, height=700)
        page.insert_text((50, 50), "SYNTHETIC ENGINE SMOKE")
        page.draw_rect(fitz.Rect(100, 100, 200, 180))
        pdf.save(source)
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    project = scratch / "delivery"
    commands = [
        ["audit", str(source), "--task", "structural", "--output", str(project)],
        ["inspect", str(project / "result.json")],
        ["replay-audit", str(project / "result.json"), "--output", str(scratch / "replay.pdf")],
    ]
    for args in commands:
        subprocess.run([sys.executable, "-B", str(ROOT / "bin/estimation"), *args],
                       cwd=scratch, check=True)
    for name in ("project.sqlite", "audit.pdf", "result.json"):
        if not (project / name).is_file():
            raise ValueError("missing delivery artifact: " + name)
    if hashlib.sha256(source.read_bytes()).hexdigest() != original:
        raise ValueError("smoke changed input PDF")
    return {"status": "passed", "commands": commands,
            "scope": "synthetic structural delivery, inspection and audit replay; no quantity claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check-index", "test", "smoke"))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--scratch", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.command == "check-index":
        print(json.dumps(check_index()))
        return
    if args.list:
        if args.command != "test":
            parser.error("--list requires test")
        print("\n".join(test.id() for test in engine_tests()))
        return
    report = (args.report or ROOT / "data/operations" / ("engine-" + args.command + ".json")).absolute()
    if not args.worker:
        # Own and measure this job's temporary tree, including unittest scratch.
        scratch = Path(tempfile.mkdtemp(prefix="rebar-engine-" )).resolve()
        env = {**os.environ, "TMPDIR": str(scratch), "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            run([sys.executable, "-B", str(Path(__file__).resolve()), args.command,
                 "--worker", "--scratch", str(scratch), "--report", str(report)],
                outputs=[report], temporary=[scratch], reserve_bytes=256 * 1024**2,
                max_growth_bytes=256 * 1024**2, env=env)
        finally:
            shutil.rmtree(scratch)
        return
    if args.command == "test":
        result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(engine_tests()))
        payload = {"status": "passed" if result.wasSuccessful() else "failed",
                   "tests_run": result.testsRun, "failures": len(result.failures),
                   "errors": len(result.errors), "skipped": [(t.id(), why) for t, why in result.skipped],
                   "private_fixtures_provisioned": False}
    else:
        payload = smoke(args.scratch)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2) + "\n")
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
