"""Fail-closed sidecar locks for frozen review artifacts."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


def bytes_sha256(value: bytes) -> str:
    return sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    return bytes_sha256(path.read_bytes())


def sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".locked.json")


def lock_record(path: Path, *, lock_manifest_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": "0.1.0",
        "layer": "immutable_review_artifact_lock",
        "artifact_name": path.name,
        "artifact_sha256": file_sha256(path),
        "lock_manifest_sha256": str(lock_manifest_sha256),
        "contract": {"hash_mismatch_is_fatal": True, "artifact_mutation_allowed": False},
    }


def assert_write_allowed(path: Path, proposed_bytes: bytes) -> None:
    sidecar = sidecar_path(path)
    if not sidecar.is_file():
        return
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    if not path.is_file() or file_sha256(path) != record.get("artifact_sha256"):
        raise RuntimeError("locked review artifact no longer matches its immutable hash")
    if bytes_sha256(proposed_bytes) != record.get("artifact_sha256"):
        raise RuntimeError("locked review artifact cannot be overwritten")


def validate_locked_file(path: Path, record: Mapping[str, Any]) -> None:
    if record.get("contract", {}).get("hash_mismatch_is_fatal") is not True:
        raise ValueError("review lock must make hash mismatch fatal")
    if not path.is_file() or file_sha256(path) != record.get("artifact_sha256"):
        raise ValueError("locked review artifact hash mismatch")
