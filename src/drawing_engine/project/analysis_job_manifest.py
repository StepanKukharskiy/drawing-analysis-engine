"""Content-addressed analysis-job result manifest and read-side validation."""

from __future__ import annotations

import hashlib
import json
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "0.2.0"
ALLOWED_ARTIFACTS = {
    "bundle",
    "observation_graph",
    "engineering_graph",
    "evidence_store",
    "automatic_review_delta",
    "automatic_review_report",
    "effective_canonical_overlay",
    "solver_replay",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_analysis_job_result(
    source_pdf: Path,
    pipeline: Mapping[str, str],
    artifacts: Mapping[str, Path],
    page_summaries: list[dict[str, Any]],
    result_binding: Mapping[str, Any],
) -> dict[str, Any]:
    unknown = set(artifacts) - ALLOWED_ARTIFACTS
    if unknown:
        raise ValueError(f"unsupported artifact names: {sorted(unknown)}")
    source_hash = sha256_file(source_pdf)
    job_id = f"analysis-{source_hash[:16]}-{str(pipeline['sha256'])[:12]}"
    artifact_records = {}
    for name, path in sorted(artifacts.items()):
        artifact_records[name] = {
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "media_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "analysis_job_result_manifest",
        "job_id": job_id,
        "state": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "document_key": f"pdf-sha256:{source_hash}",
        "source": {
            "filename": source_pdf.name,
            "bytes": source_pdf.stat().st_size,
            "sha256": source_hash,
            "media_type": "application/pdf",
        },
        "pipeline": dict(pipeline),
        "result_binding": dict(result_binding),
        "artifacts": artifact_records,
        "page_summaries": page_summaries,
        "contract": {
            "artifact_names_allow_listed": True,
            "artifact_paths_not_exposed": True,
            "complete_requires_content_hashes": True,
            "partial_graph_is_never_complete": True,
        },
    }


def validate_analysis_job_result(payload: Mapping[str, Any]) -> list[str]:
    errors = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    if payload.get("state") != "complete":
        errors.append("result state must be complete")
    if not str(payload.get("document_key", "")).startswith("pdf-sha256:"):
        errors.append("document_key must be content-addressed")
    binding = payload.get("result_binding")
    if not isinstance(binding, dict):
        errors.append("result_binding is required")
    else:
        for key in ("source_pdf_sha256", "canonical_engineering_graph_sha256"):
            if len(str(binding.get(key, ""))) != 64:
                errors.append(f"result_binding.{key} is required")
        if binding.get("source_pdf_sha256") != str(payload.get("document_key", "")).removeprefix("pdf-sha256:"):
            errors.append("result_binding source does not match document_key")
        if not isinstance(binding.get("pipeline"), dict) or len(str(binding["pipeline"].get("sha256", ""))) != 64:
            errors.append("result_binding.pipeline is required")
        if not isinstance(binding.get("ruleset"), dict) or len(str(binding["ruleset"].get("sha256", ""))) != 64:
            errors.append("result_binding.ruleset is required")
        if not isinstance(binding.get("generated_at"), str):
            errors.append("result_binding.generated_at is required")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        errors.append("artifacts must be a non-empty object")
    else:
        unknown = set(artifacts) - ALLOWED_ARTIFACTS
        if unknown:
            errors.append(f"unsupported artifact names: {sorted(unknown)}")
        for name, item in artifacts.items():
            if Path(str(item.get("filename", ""))).name != item.get("filename"):
                errors.append(f"artifacts.{name}.filename must not contain a path")
            if len(str(item.get("sha256", ""))) != 64:
                errors.append(f"artifacts.{name}.sha256 is required")
            if not isinstance(item.get("bytes"), int) or item["bytes"] < 0:
                errors.append(f"artifacts.{name}.bytes must be non-negative")
    return errors


def load_analysis_job_result(path: Path) -> dict[str, Any]:
    """Read-side API used by an application without trusting filesystem paths."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_analysis_job_result(payload)
    if errors:
        raise ValueError("invalid analysis job result: " + "; ".join(errors))
    return payload
