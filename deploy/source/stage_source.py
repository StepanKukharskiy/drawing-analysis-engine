"""Stage an explicit source selection, never a recursive workspace copy."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy.structural.stage_engine import validate_inputs


def read_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported source manifest")
    names = [name for values in manifest["groups"].values() for name in values]
    aliases = manifest.get("aliases", {})
    all_names = names + list(aliases)
    if not names or len(all_names) != len(set(all_names)):
        raise ValueError("empty or duplicate source selection")
    for name in all_names:
        p = PurePosixPath(name)
        if (not p.parts or p.is_absolute() or str(p) != name or ".." in p.parts or "\\" in name
                or (p.parts[0] == "src" and p.parts[1:2] != ("drawing_engine",))
                or (len(p.parts) > 1 and p.parts[0] not in {"src", "config", "models", "deploy", "bin", "tests", "review-app", "apps", "tools", "research", "docs", ".github"})
                or any(part in {"output", "outputs", "tmp", "data", "node_modules", ".cache", ".git", "__pycache__"} for part in p.parts)
                or p.suffix.lower() in {".pdf", ".sqlite", ".mp4", ".pyc"}
                or "secret" in p.name.lower() or p.name.startswith(".env")):
            raise ValueError("unsafe/private source selection: " + name)
    for name, target in aliases.items():
        if PurePosixPath(target).name != target or str(PurePosixPath(name).parent / target) not in names:
            raise ValueError("alias must name a selected sibling: " + name)
    return manifest, sorted(names)


def stage(root, destination, manifest_path, *, profile="source"):
    root = Path(root).resolve()
    destination = Path(destination).absolute()
    manifest, names = read_manifest(manifest_path)
    if profile == "runtime":
        names = sorted(manifest["groups"]["runtime"])
    elif profile == "engine":
        if manifest.get("repository_profile") != "engine":
            raise ValueError("engine profile requires the engine repository manifest")
        if any(name.startswith(("apps/", "review-app/", "research/")) for name in names):
            raise ValueError("engine repository cannot contain app or research source")
    elif profile != "source":
        raise ValueError("unknown stage profile")
    if destination.exists() or destination.is_symlink():
        raise ValueError("destination already exists; choose a new stage")
    if any(p.is_symlink() for p in [destination, *destination.parents]):
        raise ValueError("symlink destination")
    if destination == root or root.is_relative_to(destination):
        raise ValueError("destination cannot replace source")
    validate_inputs(root, names)
    for name, target in manifest.get("aliases", {}).items():
        if not (root / name).is_symlink() or (root / name).readlink().as_posix() != target:
            raise ValueError("source alias differs from manifest: " + name)
    records = [{"path": name, "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
                "executable": bool((root / name).stat().st_mode & 0o111)} for name in names]
    local_imports, external_imports = {}, set()
    selected = set(names)
    for name in names:
        if not name.endswith(".py"):
            continue
        imported = set()
        for node in ast.walk(ast.parse((root / name).read_text())):
            modules = ([n.name for n in node.names] if isinstance(node, ast.Import) else
                       [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            if isinstance(node, ast.ImportFrom) and node.module in {"src", "experiments", "deploy", "tests", "tools", "research"}:
                modules = [node.module + "." + n.name for n in node.names]
            for module in modules:
                first = module.split(".")[0]
                if first in {"src", "experiments", "deploy", "tests", "tools", "research"}:
                    path = module.replace(".", "/") + ".py"
                    if path not in selected:
                        raise ValueError(f"unlisted local import: {name} -> {module}")
                    imported.add(path)
                elif first and first not in sys.stdlib_module_names:
                    external_imports.add(first)
        local_imports[name] = sorted(imported)
    destination.mkdir(parents=True)
    for record in records:
        source, target = root / record["path"], destination / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(0o755 if record["executable"] else 0o644)
        if hashlib.sha256(target.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("source changed during staging; partial stage retained")
    for name, target in manifest.get("aliases", {}).items():
        (destination / name).symlink_to(target)
    if any(hashlib.sha256((root / r["path"]).read_bytes()).hexdigest() != r["sha256"] for r in records):
        raise ValueError("source changed before stage sealed; partial stage retained")
    result = {"schema_version": 1, "source_manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
              "profile": profile,
              "files": records, "aliases": manifest.get("aliases", {}), "local_imports": local_imports,
              "external_import_roots": sorted(external_imports), "private_fixtures_copied": False,
              "test_selectors": manifest.get("test_selectors", []),
              "limits": "Static imports include conditional branches; qualification must exercise actual resources and workers."}
    (destination / "source-stage.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def provision_private_fixtures(root, destination, manifest_path, profile):
    """Explicit hash-verified local overlay; never part of the source selection."""
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination == root or destination.is_relative_to(root):
        raise ValueError("private qualification overlay must be outside the workspace")
    manifest = json.loads(Path(manifest_path).read_text())
    records = manifest["profiles"][profile]
    names = [record["path"] for record in records]
    if not names or len(names) != len(set(names)) or (destination / "private-fixtures.json").exists():
        raise ValueError("duplicate private fixture")
    for record in records:
        name = record["path"]
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or str(path) != name or "\\" in name:
            raise ValueError("unsafe private fixture path")
        source, target = root / name, destination / name
        if (not source.is_file() or source.is_symlink() or not source.resolve().is_relative_to(root)
                or any(p.is_symlink() for p in source.parents)
                or target.exists() or target.is_symlink()
                or any(p.is_symlink() for p in target.parents)):
            raise ValueError("missing, linked or existing private fixture: " + name)
        if (source.stat().st_size != record["bytes"]
                or hashlib.sha256(source.read_bytes()).hexdigest() != record["sha256"]):
            raise ValueError("private fixture differs from frozen provisioning manifest: " + name)
    for record in records:
        target = destination / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / record["path"], target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("private fixture changed during copy; partial overlay retained")
    report = {"profile": profile, "source_release_inclusion": False, "files": records,
              "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()}
    (destination / "private-fixtures.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--profile", choices=("source", "runtime", "engine"), default="source")
    parser.add_argument("--private-profile", help="explicit local qualification overlay; source stage only")
    parser.add_argument("--private-manifest", type=Path, default=ROOT / "deploy/source/private-fixtures.json")
    args = parser.parse_args()
    if args.manifest is None:
        default = "engine-manifest.json" if args.profile == "engine" else "manifest.json"
        args.manifest = ROOT / "deploy/source" / default
    if args.private_profile and args.profile != "source":
        parser.error("private fixtures are not part of the runtime profile")
    result = stage(args.source, args.destination, args.manifest, profile=args.profile)
    if args.private_profile:
        provision_private_fixtures(args.source, args.destination, args.private_manifest, args.private_profile)
    print(json.dumps({"files": len(result["files"]), "aliases": result["aliases"], "destination": str(args.destination)}))
