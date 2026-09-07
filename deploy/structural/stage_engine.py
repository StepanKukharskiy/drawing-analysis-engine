"""Build a fresh allow-listed engine directory without modifying source files."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil


def read_inventory(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if not names or len(names) != len(set(names)):
        raise ValueError("empty or duplicate engine inventory")
    for name in names:
        parts = PurePosixPath(name)
        if parts.is_absolute() or ".." in parts.parts or "\\" in name or str(parts) != name:
            raise ValueError(f"unsafe engine path: {name}")
        if parts.parts[0] not in {"src", "config", "models", "requirements.txt"}:
            raise ValueError(f"unexpected engine path: {name}")
        if parts.parts[0] == "src" and parts.parts[1:2] != ("drawing_engine",):
            raise ValueError(f"retired src compatibility path: {name}")
    return sorted(names)


def validate_inputs(root: Path, names: list[str]) -> None:
    root = root.resolve()
    allowed = set(names)
    for name in names:
        if PurePosixPath(name).parts[0] == "experiments":
            raise ValueError(f"retired experiments source: {name}")
        if name.startswith("src/") and not name.startswith("src/drawing_engine/"):
            raise ValueError(f"retired src compatibility source: {name}")
        path = root / name
        if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"missing or external engine file: {name}")
        if any(part.is_symlink() for part in [path, *path.parents] if part != root.parent):
            raise ValueError(f"symlink engine input: {name}")
        if path.suffix != ".py":
            continue
        tree = ast.parse(path.read_text(), filename=name)
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    raise ValueError(f"relative import needs inventory review: {name}:{node.lineno}")
                modules = [node.module or ""]
                if node.module in {"src", "experiments", "tools", "research"}:
                    modules = [f"{node.module}.{alias.name}" for alias in node.names]
            elif isinstance(node, ast.Call) and ast.unparse(node.func) in {
                "__import__", "importlib.import_module", "import_module"
            }:
                if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
                    raise ValueError(f"dynamic import needs inventory review: {name}:{node.lineno}")
                modules = [node.args[0].value]
            for module in modules:
                if module.split(".")[0] == "experiments":
                    raise ValueError(f"retired experiments import: {name} -> {module}")
                if module.startswith("src.") and not module.startswith("src.drawing_engine."):
                    raise ValueError(f"retired src compatibility import: {name} -> {module}")
                if module.split(".")[0] in {"src", "experiments", "tools", "research"}:
                    dependency = module.replace(".", "/") + ".py"
                    if dependency not in allowed:
                        raise ValueError(f"unlisted dependency: {name} -> {dependency}")


def _stage_files(root: Path, destination: Path, names: list[str], kind: str, manifest_name: str) -> dict:
    root = root.resolve()
    destination = destination.resolve()
    if destination == root or root.is_relative_to(destination):
        raise ValueError("destination must not replace the source or its ancestors")
    validate_inputs(root, names)
    if destination.exists():
        raise ValueError("destination already exists; use a fresh staging directory")
    records = [{"path": name, "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
               for name in names]
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    destination.mkdir(parents=True)
    for name in names:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
        target.chmod(0o644)
    manifest = {"schema_version": "1.0.0", "kind": kind,
                "sha256": digest, "files": records}
    (destination / manifest_name).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def stage_engine(root: Path, destination: Path, inventory: Path) -> dict:
    return _stage_files(root, destination, read_inventory(inventory),
                        "structural_runtime", "structural-runtime.json")


def stage_build_context(root: Path, destination: Path) -> dict:
    # Railway uploads do not use .dockerignore. Stage only its explicit file
    # exceptions, never walk the repository or rely on the upload client's ignores.
    names = {".dockerignore"}
    for line in (root / ".dockerignore").read_text().splitlines():
        if not line.startswith("!") or line.endswith("/"):
            continue
        name = line[1:].replace("[[]", "[")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or str(path) != name or any(c in name for c in "*?\\"):
            raise ValueError(f"build context requires literal safe file paths: {line}")
        names.add(name)
    inventory = read_inventory(root / "deploy/structural/engine-files.txt")
    if not set(inventory) <= names:
        raise ValueError("build context omits engine inventory files")
    return _stage_files(root, destination, sorted(names),
                        "structural_build_context", "structural-build-context.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, default=Path(__file__).with_name("engine-files.txt"))
    parser.add_argument("--build-context", action="store_true", help="stage the source-only Railway upload instead of the engine")
    args = parser.parse_args()
    result = (stage_build_context(args.source, args.destination) if args.build_context
              else stage_engine(args.source, args.destination, args.inventory))
    print(f"Staged {len(result['files'])} {result['kind']} files: {result['sha256']}")
